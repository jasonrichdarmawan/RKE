import os

print(f"HF_HOME: {os.getenv('HF_HOME')=}")
print(f"PYTHONPATH: {os.getenv('PYTHONPATH')=}")

import json
import shutil
from pathlib import Path
from itertools import islice
from time import time
from typing import Tuple, Union, Callable, Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

from jaxtyping import Float
from torch import Tensor

from dsets import UnKEDataset, CounterFactDataset, MQUAKEDataset, EditeveryDataset
from original import originalHyperParams
from memit import MEMITHyperParams, apply_memit_to_model
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from AlphaEdit import AlphaEditHyperParams, apply_AlphaEdit_to_model, get_cov
from AlphaEdit_ARE import AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model
from unke import unkeHyperParams, apply_unke_to_model
from unke_Alpha import unkeAlphaHyperParams, apply_unke_alpha_to_model
from unke_Alpha_ARE import unkeAlphaAREHyperParams, apply_unke_Alpha_ARE_to_model
from unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from util import nethook, set_seed
from util.hparams import HyperParams
from util.globals import *
from util.tee import StreamToLogger

from tabulate import tabulate

import logging
import sys

from glue_eval.glue_eval import GLUEEval

ALG_DICT: dict[str, tuple[type[HyperParams], Callable[..., dict[str, Any]]]] = {
    "original": (originalHyperParams, None),
    "unke_Alpha_ARE": (unkeAlphaAREHyperParams, apply_unke_Alpha_ARE_to_model),
    "unke_Alpha": (unkeAlphaHyperParams, apply_unke_alpha_to_model),
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}

DS_DICT = {
    "unke": UnKEDataset,
    "cf": CounterFactDataset,
    "mquake": MQUAKEDataset,
    "editevery": EditeveryDataset,
}


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int,
    use_cache: bool,
    restore_weights_each_edit: bool,
    sequential_eval: bool,
    downstream_eval_steps: int,
):
    set_seed()
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    alg_dir = RESULTS_DIR / dir_name
    if alg_dir.exists():
        id_list = [
            int(str(x).split("_")[-1])
            for x in alg_dir.iterdir()
            if str(x).split("_")[-1].isnumeric()
        ]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0
    run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        filename=run_dir / "output.log",
        filemode="w",
    )
    log = logging.getLogger("evaluate_uns")
    sys.stdout = StreamToLogger(logger=log, level=logging.INFO, echo=sys.stdout)
    sys.stderr = StreamToLogger(logger=log, level=logging.ERROR, echo=sys.stderr)

    print(f"Results will be stored at {run_dir}")

    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Load model
    """
    float16 has precision issues
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    # apply_algo expects padding_side="right", while evaluate should use left padding
    # I was also confused why there is two tokenizers here
    # I will not refactor now to avoid breaking existing code
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if hparams.model_name == "Llama3-8B-Instruct":
        tokenizer.pad_token_id = tok.eos_token_id

    # Load data
    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)

    if alg_name in ["unke", "unke_ARE", "unke_Alpha", "unke_Alpha_ARE"]:
        with open(
            Path(DATA_DIR) / "alpaca_data.json", "r", encoding="utf-8"
        ) as json_file:
            ex_datas = json.load(json_file)
        if hparams.model_name == "Llama3-8B-Instruct":
            ex_datas = [
                get_llama_without_answer(i["instruction"] + i["input"]) + i["output"]
                for i in ex_datas
            ]
        elif hparams.model_name == "Qwen2.5-7B-Instruct":
            ex_datas = [
                get_qwen_without_answer(i["instruction"] + i["input"]) + i["output"]
                for i in ex_datas
            ]
    # Load null space projection matrices
    if alg_name in ["AlphaEdit", "AlphaEdit_ARE"]:
        name = model.config._name_or_path.rsplit("/")[-1]
        stats_dir = Path(STATS_DIR)
        file_extension = f"{name}/{hparams.mom2_dataset}_stats/null_space_project.pt"
        filename = stats_dir / file_extension
        if not os.path.exists(filename):
            W_out = nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
            )
            P = torch.zeros(
                (len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu"
            )
            del W_out
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model, tokenizer, layer, hparams)
            torch.save(P, filename)
        else:
            P = torch.load(filename)
    # Load second moment statistics for unke_Alpha and unke_Alpha_ARE
    if alg_name in ["unke_Alpha", "unke_Alpha_ARE"]:
        second_moment_map = {}
        S_KpKp_map = {}
        S_KpVp_map = {}
        S_VpVp_map = {}
        S_count_map = {}
        for i, layer in enumerate(hparams.layers):
            force_recompute = False
            for layer_name_template in hparams.rewrite_module_tmp:
                layer_name = layer_name_template.format(layer)
                second_moment = get_cov(
                    model=model,
                    tok=tok,
                    layer_name=layer_name,
                    mom2_dataset=hparams.mom2_dataset,
                    mom2_n_samples=(
                        hparams.mom2_n_samples
                        if not force_recompute
                        else hparams.mom2_n_samples // 10
                    ),
                    mom2_dtype=hparams.mom2_dtype,
                    force_recompute=force_recompute,
                    return_count=True,
                )
                second_moment_map[layer_name] = second_moment
                S_KpKp_map[layer_name] = None
                S_KpVp_map[layer_name] = None
                S_VpVp_map[layer_name] = None
                S_count_map[layer_name] = None

    # batch_size = num_edits
    num_edit_batches = len(ds) // num_edits + (1 if len(ds) % num_edits else 0)
    edited_data = []

    def print_output(data: dict):
        rows = []
        rows.append(["question", data["question"]])
        rows.append(["original_prediction", data["original_prediction"]])
        if ds_name in ["unke", "cf"]:
            rows.append(["para_question", data["para_question"]])
            rows.append(["para_prediction", data["para_prediction"]])
        rows.append(["answer", data["answer"]])
        # if ds_name in ["unke", "cf", "mquake"]:
        #     for idx in range(len(data["sub_question"])):
        #         rows.append([f"sub_question_{idx}", data["sub_question"][idx]])
        #         rows.append([f"sub_pred_{idx}", data["sub_pred"][idx]])
        #         rows.append([f"sub_answer_{idx}", data["sub_answer"][idx]])
        print(tabulate(tabular_data=rows, tablefmt="plain"))

    def evaluate(batch: list[dict], eval_batch_size: int = 1):
        eval_batches = len(batch) // eval_batch_size + (1 if len(batch) % eval_batch_size else 0)
        for batch_index in tqdm(range(eval_batches), desc="Generating outputs"):
            start_index = batch_index * eval_batch_size
            end_index = start_index + eval_batch_size
            sub_batch = batch[start_index:end_index]

            texts = []
            for data in sub_batch:
                texts.append(data["question"])
                if ds_name in ["unke", "cf"]:
                    texts.append(data["para_question"])
                if ds_name in ["unke", "cf", "mquake"]:
                    texts.extend(data["sub_question"])

            question = tokenizer(texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=question["input_ids"].to("cuda"),
                    attention_mask=question["attention_mask"].to("cuda"),
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=512,
                )
            # slice off prompts for each item and assign back
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(question["input_ids"], generated_ids)
            ]
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            idx = 0
            for data in sub_batch:
                # remove special tokens from answers
                if hparams.model_name == "Llama3-8B-Instruct":
                    data["answer"] = data["answer"][: -len("<|eot_id|>")]
                elif hparams.model_name == "Qwen2.5-7B-Instruct":
                    data["answer"] = data["answer"][: -len("<|im_end|>")]

                data["original_prediction"] = outputs[idx]
                idx += 1
                if ds_name in ["unke", "cf"]:
                    data["para_prediction"] = outputs[idx]
                    idx += 1
                if ds_name in ["unke", "cf", "mquake"]:
                    sub_q_len = len(data["sub_question"])
                    data["sub_pred"] = outputs[idx : idx + sub_q_len]
                    idx += sub_q_len

    glue_save_location = str(run_dir) + "/glue_eval/"
    os.makedirs(glue_save_location, exist_ok=True)

    for edit_batch_index in tqdm(range(num_edit_batches), desc="Edit batches"):
        start_index = edit_batch_index * num_edits
        end_index = start_index + num_edits
        batch = ds[start_index:end_index]
        # case_result_template = str(run_dir / "{}_edits-case_{}.json")

        kwargs = {}
        if alg_name in ["unke", "unke_ARE", "unke_Alpha", "unke_Alpha_ARE"]:
            kwargs["ex_data"] = random.sample(ex_datas, 20)
        if alg_name in ["AlphaEdit", "AlphaEdit_ARE"]:
            kwargs["P"] = P
        if alg_name in ["unke_Alpha", "unke_Alpha_ARE"]:
            kwargs["second_moment_map"] = second_moment_map

            # if restore_weights_each_edit, there is no point to preserve
            # prior edits
            if restore_weights_each_edit:
                kwargs["S_KpKp_map"] = S_KpKp_map.copy()
                kwargs["S_KpVp_map"] = S_KpVp_map.copy()
                kwargs["S_VpVp_map"] = S_VpVp_map.copy()
                kwargs["S_count_map"] = S_count_map.copy()
            else:
                kwargs["S_KpKp_map"] = S_KpKp_map
                kwargs["S_KpVp_map"] = S_KpVp_map
                kwargs["S_VpVp_map"] = S_VpVp_map
                kwargs["S_count_map"] = S_count_map

        start = time()
        if alg_name in [
            "unke",
            "unke_ARE",
            "MEMIT",
            "MEMIT_ARE",
            "AlphaEdit",
            "AlphaEdit_ARE",
            "unke_Alpha",
            "unke_Alpha_ARE",
        ]:
            weights_copy = apply_algo(model, tok, hparams, batch, **kwargs)
        exec_time = time() - start
        print(f"Execution took {exec_time:.1f}s")

        start = time()
        if not sequential_eval:
            evaluate(batch=batch)
            if edit_batch_index < 10:
                for data in batch:
                    print_output(data=data)

            edited_data.extend(batch)

        if downstream_eval_steps > 0 and (edit_batch_index + 1) % downstream_eval_steps == 0:
            glue_results = {"edit_num": edit_batch_index * num_edits, "batch_index": edit_batch_index}

            record_path = glue_save_location + f"batch_index={edit_batch_index}.json"
            
            glue_eval = GLUEEval(
                model=model,
                tokenizer=tok,
                number_of_tests=100,
            )
            glue_results = glue_eval.evaluate(
                glue_results=glue_results,
                record_path=record_path,
                nli_flag=True,
                sst_flag=True,
                cola_flag=True,
                rte_flag=True,
                mmlu_flag=True,
                mrpc_flag=True,
            )

            output_filename = record_path.replace(".json", "_glue.json")
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(glue_results, f, indent=4)

        # Restore weights
        if restore_weights_each_edit:
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")

    if sequential_eval:
        for edit_batch_index in tqdm(range(num_edit_batches), desc="Evaluate batches"):
            start_index = edit_batch_index * num_edits
            end_index = start_index + num_edits
            batch = ds[start_index:end_index]

            evaluate(batch=batch)
            if edit_batch_index < 10:
                for data in batch:
                    print_output(data=data)

            edited_data.extend(batch)

    path = str(
        run_dir
        / f"alg_name={alg_name}__sequential={int(sequential_eval)}__ds_name={ds_name}__dataset_size_limit={dataset_size_limit}__num_edits={num_edits}.json"
    )
    print(f"Saving to {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(edited_data, json_file, ensure_ascii=False, indent=4)

    print(f"Evaluation took {time() - start:.1f}s")


def get_project(model, tok, layer, hparams, use_svd=True) -> Union[
    dict[str, Float[Tensor, "in_features in_features"]],
    Float[Tensor, "in_features in_features"],
]:
    """
    Backward compatibility:
    memit, memit_ARE, AlphaEdit, AlphaEdit_ARE return a single projection matrix
    unke_Alpha, unke_Alpha_ARE return a dictionary of projection matrices for each module template
    """
    force_recompute = False

    layer_names = hparams.rewrite_module_tmp
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    projections = {}
    for layer_name in layer_names:
        ln = layer_name.format(layer)
        cov = get_cov(
            model=model,
            tok=tok,
            layer_name=ln,
            mom2_dataset=hparams.mom2_dataset,
            mom2_n_samples=(
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10
            ),
            mom2_dtype=hparams.mom2_dtype,
            force_recompute=force_recompute,
        ).to("cuda")
        if use_svd:
            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            threshold = hparams.nullspace_threshold
            small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
            projection = U[:, small_singular_indices] @ U[:, small_singular_indices].T
            print(
                f"Layer name: {ln}, singular values less than the threshold size: {len(small_singular_indices)}"
            )
        else:
            start_time = time()
            vals, vecs = torch.linalg.eigh(cov)
            threshold = hparams.nullspace_threshold
            small_eigenvalues_indices = (vals < threshold).nonzero(as_tuple=True)[0]
            projection = (
                vecs[:, small_eigenvalues_indices]
                @ vecs[:, small_eigenvalues_indices].T
            )
            end_time = time()
            print(
                f"Computed projection for layer {ln} in {end_time - start_time:.2f} seconds."
            )
            print(
                f"Layer name: {ln}, eigenvalues less than the threshold size: {len(small_eigenvalues_indices)}"
            )
        projections[ln] = projection

    return projections if len(projections) > 1 else projections[layer_name[0]]


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=[
            "original",
            "AlphaEdit",
            "AlphaEdit_ARE",
            "MEMIT",
            "MEMIT_ARE",
            "ROME",
            "FT",
            "MEND",
            "unke",
            "unke_ARE",
            "unke_Alpha",
            "unke_Alpha_ARE",
        ],
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
    )
    parser.add_argument(
        "--model_name",
        default="Llama3-8B-Instruct",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama3-8B-Instruct.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "editevery", "unke", "mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--restore_weights_each_edit",
        action="store_true",
        help="Restore weights after each edit",
    )
    parser.add_argument(
        "--sequential_eval",
        action="store_true",
        help="Perform sequential evaluation",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="Interval for downstream eval steps",
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        restore_weights_each_edit=args.restore_weights_each_edit,
        sequential_eval=args.sequential_eval,
        downstream_eval_steps=args.downstream_eval_steps,
    )
