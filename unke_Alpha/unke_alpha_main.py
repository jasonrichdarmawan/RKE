import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import nethook
import torch.optim as optim

import argparse

import numpy as np
import os
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
)
from .unke_alpha_hparams import unkeAlphaHyperParams

from torch import Tensor
from jaxtyping import Float
from beartype import beartype as typechecker

from peft import PeftModel, TaskType, get_peft_model
from peft.tuners.nullspacelora import NullSpaceLoraConfig
from typing import Any
from util.tok_dataset import flatten_masked_batch


def compute_ks(
    peft_model: PeftModel,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeAlphaHyperParams,
    layer: int,
):
    input_ids = tok(batch_data, padding=True, return_tensors="pt").to("cuda")
    idxs = [i.sum() - 1 for i in input_ids["attention_mask"]]
    with torch.no_grad():
        with nethook.Trace(
            module=peft_model.model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = peft_model(**input_ids)
            # layer_in_ks = tr.input #(bs:seq:h_dim)
            zs_out = tr.output  # (bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list = []
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i, idxs[i]])
    zs_out = torch.stack(zs_out_list, dim=0)

    return zs_out, idxs


def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],  # and 'mlp' in n
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


# @jaxtyped(typechecker=typechecker)
def apply_unke_alpha_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: unkeAlphaHyperParams,
    batch_data: list,
    # ex_data: list,
    second_moment_map: dict[str, dict[str, Any]],
    S_KpKp_map: dict[str, torch.Tensor],
    S_KpVp_map: dict[str, torch.Tensor],
    S_VpVp_map: dict[str, torch.Tensor],
    S_count_map: dict[str, int],
):
    task_type = TaskType.CAUSAL_LM
    target_modules = [
        tmp.format(layer)
        for layer in hparams.layers
        for tmp in hparams.rewrite_module_tmp
    ]
    peft_config = NullSpaceLoraConfig(
        task_type=task_type,
        r=hparams.r,
        target_modules=target_modules,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
    )
    peft_model = get_peft_model(model=model, peft_config=peft_config)
    peft_model.set_lora_second_moment_map(
        second_moment_map=second_moment_map, adapter_name="default"
    )
    peft_model.set_lora_S_KpKp_map(
        S_KpKp_map=S_KpKp_map, adapter_name="default"
    )
    peft_model.set_lora_S_KpVp_map(
        S_KpVp_map=S_KpVp_map, adapter_name="default"
    )
    peft_model.set_lora_S_VpVp_map(
        S_VpVp_map=S_VpVp_map, adapter_name="default"
    )
    peft_model.set_lora_S_count_map(
        S_count_map=S_count_map, adapter_name="default"
    )

    preserve_params = []
    for name, params in peft_model.model.named_parameters():
        # print(name)
        splitted_name = name.split(".")
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    weights = {
        param: nethook.get_parameter(peft_model.model, param)
        for param in preserve_params
    }

    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    z_layer = hparams.layers[-1]
    z_list = []
    for data in batch_data:

        cur_z = compute_z(
            peft_model,
            tok,
            data,
            z_layer,
            hparams,
        )

        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=0)  # (bs,h_dim)
    # print(zs.shape)
    batch_question = [i["question"] for i in batch_data]
    # Insert
    for i, layer in enumerate(hparams.layers):
        # print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
            next(peft_model.model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=peft_model.model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = peft_model(
                    **contexts_tok,
                    # lora_K_p_attention_mask=contexts_tok["attention_mask"]
                )
                layer_in_ks = tr.input  # (bs:seq:h_dim)
                layer_out_ks = tr.output  # (bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks

        cur_zs, idxs = compute_ks(peft_model, tok, batch_question, hparams, z_layer)

        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        # ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
        #     next(peft_model.model.parameters()).device
        # )
        #
        # with torch.no_grad():
        #     with nethook.Trace(
        #         module=peft_model.model,
        #         layer=hparams.layer_module_tmp.format(layer),
        #         retain_input=True,
        #         retain_output=True,
        #         detach=True,
        #         clone=True,
        #     ) as tr:
        #         _ = peft_model.model(**ex_tok)
        #         stat_in = tr.input
        #         stat_out = tr.output
        # stat_out = stat_out[0] if type(stat_out) is tuple else stat_out

        resid = targets / (
            len(hparams.layers) - i
        )  # Distribute residual across layers(1,4096)

        criterion = nn.MSELoss(reduction="none")
        # criterion = nn.MSELoss()

        _layer = nethook.get_module(
            peft_model.model, hparams.layer_module_tmp.format(layer)
        )

        for n, m in _layer.named_parameters():

            # m.requires_grad = True
            if any(k in n for k in ("lora_A", "lora_B")):
                m.requires_grad = True

        params = get_optimizer_params(_layer, hparams.lr)

        optimizer = optim.AdamW(params, lr=hparams.lr, eps=1e-8, betas=(0.9, 0.999))

        for i in range(len(idxs)):

            layer_out_ks[i, idxs[i]] += resid[i]

        # get_qwen2_causal_mask
        # llama2
        if hparams.model_name == "Llama3-8B-Instruct":
            input_causal_mask, input_position_ids, input_cache_position = (
                get_causal_mask(layer_in_ks, contexts_tok["attention_mask"])
            )
            # ex_causal_mask, ex_position_ids, ex_cache_position = get_causal_mask(
            #     stat_in, ex_tok["attention_mask"]
            # )
        elif hparams.model_name == "Qwen2.5-7B-Instruct":
            input_causal_mask, input_position_ids = get_qwen2_causal_mask(
                layer_in_ks, contexts_tok["attention_mask"]
            )
            # ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(
            #     stat_in, ex_tok["attention_mask"]
            # )

        best_loss = float("inf")
        best_weights = None
        patience = hparams.early_stop_patience
        no_improve_steps = 0

        update_loss_break_at = float("inf")
        update_abs_floor = hparams.update_abs_floor # absolute min MSE per element
        update_abs_cap = hparams.update_abs_cap # absolute max MSE per element
        update_rel_frac = hparams.update_rel_frac # fraction of initial update loss 0.5%

        reg_history = []
        reg_warmup_steps = hparams.reg_warmup_steps
        reg_loss_break_at = float("inf")
        reg_abs_floor = hparams.reg_abs_floor
        reg_abs_cap = hparams.reg_abs_cap
        reg_rel_frac = hparams.reg_rel_frac

        # warmup / previous-loss measurement
        prev_history = []
        prev_warmup_steps = hparams.prev_warmup_steps
        previous_loss_break_at = float("inf")
        prev_abs_floor = hparams.prev_abs_floor  # minimum absolute floor
        prev_abs_cap = hparams.prev_abs_cap  # maximum absolute cap
        prev_rel_frac = hparams.prev_rel_frac  # stop when loss drops to 0.5% of initial

        step = 0
        while True:
            # scheduler.step()
            optimizer.zero_grad()

            token_mask = contexts_tok["attention_mask"].unsqueeze(-1)  # (bs:seq:1)
            hidden_dim = layer_out_ks.shape[-1]
            mask_expanded = token_mask.expand(-1, -1, hidden_dim)  # (bs:seq:h_dim)
        
            if hparams.model_name == "Qwen2.5-7B-Instruct":
                # preservation_loss = criterion(
                #     _layer(
                #         stat_in,
                #         attention_mask=ex_causal_mask,
                #         position_ids=ex_position_ids,
                #     )[0],
                #     stat_out,
                # )

                update_loss = criterion(
                    _layer(
                        layer_in_ks,
                        attention_mask=input_causal_mask,
                        position_ids=input_position_ids,
                    )[0],
                    layer_out_ks,
                )

                regularization_loss = sum(
                    [
                        (delta_weight.norm() ** 2)
                        for delta_weight in peft_model.get_delta_weights(
                            adapter="default"
                        ).values()
                    ]
                )

                loss = update_loss + hparams.L2 * regularization_loss
                # loss =  criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids)[0], layer_out_ks)
            elif hparams.model_name == "Llama3-8B-Instruct":
                # preservation_loss = criterion(
                #     _layer(
                #         stat_in,
                #         attention_mask=ex_causal_mask,
                #         position_ids=ex_position_ids,
                #         cache_position=ex_cache_position,
                #     )[0],
                #     stat_out,
                # )

                pred = _layer(
                    layer_in_ks,
                    attention_mask=input_causal_mask,
                    position_ids=input_position_ids,
                    cache_position=input_cache_position,
                )
                pred = pred[0]  # (bs:seq:h_dim)

                update_loss_per_elem = criterion(pred, layer_out_ks)  # (bs:seq:h_dim)

                masked_loss = update_loss_per_elem * mask_expanded  # (bs:seq:h_dim)
                denom = mask_expanded.sum().clamp_min(1.0)
                update_loss = masked_loss.sum() / denom  # batch size, padding invariant

                # Alternative way: batch size invariant but not padding invariant
                # update_loss = criterion(
                #     _layer(
                #         layer_in_ks,
                #         attention_mask=input_causal_mask,
                #         position_ids=input_position_ids,
                #         cache_position=input_cache_position,
                #     )[0],
                #     layer_out_ks,
                # )

                if hparams.L2 == 0:
                    regularization_loss = torch.tensor(0.0).to(update_loss.device)
                else:
                    reg_dict = peft_model.get_regularization_losses_with_trace(adapter="default")
                    all_sq = 0.0
                    total_elems = 0
                    for name, (trace, out_features) in reg_dict.items():
                        all_sq += trace
                        total_elems += out_features
                    regularization_loss = hparams.L2 * (all_sq / total_elems) / len(batch_data)

                    # Alternative way: not parameter count invariant
                    # regularization_loss = sum(
                    #     [
                    #         (delta_weight.norm() ** 2)
                    #         for delta_weight in peft_model.get_delta_weights(
                    #             adapter="default"
                    #         ).values()
                    #     ]
                    # )

                # delta_weights_K_p = sum(
                #     [
                #         torch.trace(delta_weight_K_p)
                #         for delta_weight_K_p in peft_model.get_delta_weights_K_p(
                #             adapter="default"
                #         ).values()
                #     ]
                # )

                if hparams.previous_scale == 0:
                    previous_loss = torch.tensor(0.0).to(update_loss.device)
                else:
                    prev_dict = peft_model.get_previous_losses_with_trace(adapter="default")
                    all_traces = 0.0
                    total_elems = 0
                    for name, (trace, elems) in prev_dict.items():
                        all_traces += trace
                        total_elems += elems
                    previous_loss = hparams.previous_scale * (all_traces / total_elems) / len(batch_data)

                loss = (
                    update_loss
                    + regularization_loss
                    + previous_loss
                    # + hparams.L2 * regularization_loss
                    # + delta_weights_K_p
                )
                # loss = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
                # loss = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0][:,-1], layer_out_ks[:,-1])
                # loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out)+criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0][:,-1], layer_out_ks[:,-1])

            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            update_loss_item = update_loss.item()
            previous_loss_item = previous_loss.item()
            reg_loss_item = regularization_loss.item()

            if step == 0:
                rel_target = update_rel_frac * update_loss_item
                update_loss_break_at = max(
                    update_abs_floor, min(rel_target, update_abs_cap)
                )
                print(f"Update loss break at: {update_loss_break_at}")

            if step < reg_warmup_steps:
                reg_history.append(regularization_loss.item())
                if step == reg_warmup_steps - 1:
                    rel_target = reg_rel_frac * max(reg_history)
                    reg_loss_break_at = max(
                        reg_abs_floor, min(rel_target, reg_abs_cap)
                    )
                    print(f"Regularization loss break at: {reg_loss_break_at}")

            # different from update loss break
            # because prev loss increase as update loss decrease
            if step < prev_warmup_steps:
                prev_history.append(previous_loss_item)
                if step == prev_warmup_steps - 1:
                    rel_target = prev_rel_frac * max(prev_history)
                    previous_loss_break_at = max(
                        prev_abs_floor, min(rel_target, prev_abs_cap)
                    )
                    print(f"Previous loss break at: {previous_loss_break_at}")

            if (
                step % 10 == 0
                or step < reg_warmup_steps
                or step < prev_warmup_steps
                or (
                    update_loss_break_at < float("inf")
                    and reg_loss_break_at < float("inf")
                    and previous_loss_break_at < float("inf")
                    and update_loss_item < update_loss_break_at
                    and reg_loss_item < reg_loss_break_at
                    and previous_loss_item < previous_loss_break_at
                )
            ):
                print(
                    "Step [{}], Loss: {:.10f}, Update: {:.10f}, Regularization: {:.10f}, Previous: {:.10f}, Norm: {:.10f}, Layer: {}".format(
                        step + 1,
                        loss.item(),
                        update_loss,
                        reg_loss_item,
                        previous_loss_item,
                        norm,
                        layer,
                    )
                )

            edit_loss = update_loss_item + reg_loss_item + previous_loss_item
            if edit_loss < best_loss:
                best_loss = edit_loss
                best_weights = {
                    n: p.detach().cpu().clone()
                    for n, p in _layer.named_parameters()
                    if any(k in n for k in ("lora_B", "lora_A"))
                }
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if no_improve_steps >= patience:
                    print(
                        f"No improvement for {patience} steps, stopping early at step {step}."
                    )
                    break

            step += 1

            if (
                update_loss_break_at < float("inf")
                and reg_loss_break_at < float("inf")
                and previous_loss_break_at < float("inf")
                and update_loss_item < update_loss_break_at
                and reg_loss_item < reg_loss_break_at
                and previous_loss_item < previous_loss_break_at
            ):
                break

        if best_weights is not None:
            for n, p in _layer.named_parameters():
                if n in best_weights:
                    p.data.copy_(best_weights[n].to(p.device))

        for x in [
            layer_in_ks,
            layer_out_ks,
            cur_zs,
            targets,
            # stat_in, stat_out
        ]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    peft_model.merge_and_unload()

    with torch.no_grad():
        with nethook.TraceDict(
            module=peft_model.model,
            layers=target_modules,
            retain_input=True,
            retain_output=True,
            clone=True,
            detach=True,
            stop=True,
        ) as tr:
            peft_model(**contexts_tok)

        # for layer_name in target_modules:
        #     a = flatten_masked_batch(
        #         tr[layer_name].output, contexts_tok["attention_mask"]
        #     )
        #     second_moment_map[layer_name]["mom2"] += a.t().mm(a)
        #     second_moment_map[layer_name]["count"] += a.shape[0]

        for layer_name in target_modules:
            a = flatten_masked_batch(
                tr[layer_name].input, contexts_tok["attention_mask"]
            )
            value = a.t().mm(a).to("cpu")
            if S_KpKp_map[layer_name] is None:
                S_KpKp_map[layer_name] = value
            else:
                S_KpKp_map[layer_name] += value

            b = flatten_masked_batch(
                tr[layer_name].output, contexts_tok["attention_mask"]
            )
            value = a.t().mm(b).to("cpu")
            if S_KpVp_map[layer_name] is None:
                S_KpVp_map[layer_name] = value
            else:
                S_KpVp_map[layer_name] += value

            value = b.t().mm(b).to("cpu")
            if S_VpVp_map[layer_name] is None:
                S_VpVp_map[layer_name] = value
            else:
                S_VpVp_map[layer_name] += value

            if S_count_map[layer_name] is None:
                S_count_map[layer_name] = a.shape[0]
            else:
                S_count_map[layer_name] += a.shape[0]
                # S_count_map[layer_name] = 1

    return weights_copy


def get_qwen2_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (input_tensor.shape[0], input_tensor.shape[1]),
        input_tensor,
        0,
    )

    return attention_mask, position_ids


def get_causal_mask(input_tensor, attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(
        -1, 1
    )
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
            :, None, None, :
        ].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
            padding_mask, min_dtype
        )
    elif attention_mask.dim() == 4:  # shape [batch, 1, query_length, key_length]
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0],
            : mask_shape[1],
            offset : mask_shape[2] + offset,
            : mask_shape[3],
        ] = mask_slice

    # causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask, position_ids, cache_position
