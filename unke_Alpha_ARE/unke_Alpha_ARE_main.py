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
from .unke_Alpha_ARE_hparams import unkeAlphaAREHyperParams

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
    hparams: unkeAlphaAREHyperParams,
    layer: int,
    idxs_dict: dict,
):
    input_ids = tok(batch_data, padding=True, return_tensors="pt").to("cuda")
    zs_out_dict = {}

    with torch.no_grad():
        with nethook.Trace(
            module=peft_model.model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            peft_model(**input_ids)
            # layer_in_ks = tr.input #(bs:seq:h_dim)
            zs_out = tr.output  # (bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    for k, idxs in idxs_dict.items():
        zs_out_list = []
        for idx in idxs:
            zs_out_list.append(zs_out[k, idx])
        zs_out_dict[k] = zs_out_list
    return zs_out_dict


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
def apply_unke_Alpha_ARE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: unkeAlphaAREHyperParams,
    batch_data: list,
    # ex_data: list[str],
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
        S_KpKp_map=S_KpKp_map, S_count_map=S_count_map, adapter_name="default"
    )
    peft_model.set_lora_S_KpVp_map(
        S_KpVp_map=S_KpVp_map, S_count_map=S_count_map, adapter_name="default"
    )
    peft_model.set_lora_S_VpVp_map(
        S_VpVp_map=S_VpVp_map, S_count_map=S_count_map, adapter_name="default"
    )

    preserve_params = []
    for name, params in peft_model.named_parameters():
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
    zs_dict = {}
    idxs_dict = {}
    for k, data in enumerate(batch_data):

        idxs_list, target_list = compute_z(
            peft_model,
            tok,
            data,
            z_layer,
            hparams,
        )
        idxs_dict[k] = idxs_list
        zs_dict[k] = target_list
    batch_question_ans = [i["question"] + i["answer"] for i in batch_data]

    # Insert
    for i, layer in enumerate(hparams.layers):
        # print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
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
                peft_model(**contexts_tok)
                layer_in_ks = tr.input  # (bs:seq:h_dim)
                layer_out_ks = tr.output  # (bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks

        layer_out_ks = layer_out_ks.clone().contiguous()

        cur_zs_dict = compute_ks(
            peft_model, tok, batch_question_ans, hparams, z_layer, idxs_dict
        )
        targets_dict = {}
        for k, cur_zs_list in cur_zs_dict.items():
            zs_list = zs_dict[k]
            targets_list = [
                (a - b) / (len(hparams.layers) - i)
                for a, b in zip(zs_list, cur_zs_list)
            ]
            targets_dict[k] = targets_list

        # ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
        #     next(peft_model.model.parameters()).device
        # )

        # with torch.no_grad():
        #     with nethook.Trace(
        #         module=peft_model.model,
        #         layer=hparams.layer_module_tmp.format(layer),
        #         retain_input=True,
        #         retain_output=True,
        #         detach=True,
        #         clone=True,
        #     ) as tr:
        #         peft_model(**ex_tok)
        #         stat_in = tr.input
        #         stat_out = tr.output
        # stat_out = stat_out[0] if type(stat_out) is tuple else stat_out

        # resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers(1,4096)

        criterion = nn.MSELoss()

        _layer = nethook.get_module(
            peft_model.model, hparams.layer_module_tmp.format(layer)
        )

        for n, m in _layer.named_parameters():

            # m.requires_grad = True
            if any(k in n for k in ("lora_B", "lora_A")):
                m.requires_grad = True

        params = get_optimizer_params(_layer, hparams.lr)

        optimizer = optim.AdamW(params, lr=hparams.lr, eps=1e-8, betas=(0.9, 0.999))

        for k, idxs_list in idxs_dict.items():
            for j, idx in enumerate(idxs_list):
                resid = targets_dict[k][j]
                layer_out_ks[k, idx] = layer_out_ks[k, idx] + resid

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

        for step in range(hparams.optim_num_step):
            # scheduler.step()
            optimizer.zero_grad()
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

                # regularization_loss = sum(
                #     [
                #         (delta_weight.norm() ** 2)
                #         for delta_weight in peft_model.get_delta_weights(
                #             adapter="default"
                #         ).values()
                #     ]
                # )

                loss = (
                    # preservation_loss +
                    update_loss
                    # + hparams.L2 * regularization_loss
                )
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

                update_loss = criterion(
                    _layer(
                        layer_in_ks,
                        attention_mask=input_causal_mask,
                        position_ids=input_position_ids,
                        cache_position=input_cache_position,
                    )[0],
                    layer_out_ks,
                )

                regularization_loss = hparams.L2 * sum(
                    [
                        (delta_weight.norm() ** 2)
                        for delta_weight in peft_model.get_delta_weights(
                            adapter="default"
                        ).values()
                    ]
                )

                # previous_loss = hparams.L2 * sum(
                #     [
                #         torch.trace(
                #             delta_weight_K_p
                #         )
                #         for delta_weight_K_p in peft_model.get_delta_KpKp(
                #             adapter="default"
                #         ).values()
                #     ]
                # )

                previous_loss = 5e-6 * sum(
                    [
                        trace_KpKp_VpVp
                        for trace_KpKp_VpVp in peft_model.get_trace_KpKp_VpVp(
                            adapter="default"
                        ).values()
                    ]
                )

                loss = (
                    # preservation_loss +
                    update_loss
                    + regularization_loss
                    + previous_loss
                )
                # loss = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
                # loss = criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0][:,-1], layer_out_ks[:,-1])
                # loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out)+criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0][:,-1], layer_out_ks[:,-1])

            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            break_at = 5e-5
            if (
                step % 10 == 0
                or step == hparams.optim_num_step - 1
                or loss_item < break_at
            ):
                print(
                    "Step [{}/{}], Loss: {:.5f}, Update: {:.5f}, Regularization: {:.5f}, Previous: {:.5f}, Layer: {}".format(
                        step + 1,
                        hparams.optim_num_step,
                        loss_item,
                        update_loss.item(),
                        regularization_loss.item(),
                        previous_loss.item(),
                        layer,
                    )
                )
            if loss_item < break_at:
                break

        for x in [
            layer_in_ks,
            layer_out_ks,
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
                # S_count_map[layer_name] += a.shape[0]
                S_count_map[layer_name] = 1

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
    elif attention_mask.dim() == 4:
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
