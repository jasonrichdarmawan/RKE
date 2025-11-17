#!/usr/bin/env bash

# export variables from .env
set -a
source .env
set +a

dataset_size_limit=10
num_edits=1
downstream_eval_steps=5
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
hparams_fname="Llama3-8B-Instruct.json"
ds_name="unke"

run_exp() {
    alg_name="$1"
    python -m experiments.evaluate_uns \
        --alg_name "$alg_name" \
        --model_name "$model_name" \
        --hparams_fname "$hparams_fname" \
        --ds_name "$ds_name" \
        --dataset_size_limit "$dataset_size_limit" \
        --num_edits "$num_edits" \
        --downstream_eval_steps "$downstream_eval_steps"
}

# run_exp MEMIT
# run_exp MEMIT_ARE
# run_exp AlphaEdit
# run_exp AlphaEdit_ARE
# run_exp unke
# run_exp unke_ARE
run_exp unke_Alpha
run_exp unke_Alpha_ARE

# flush disks and give logs a moment to finish
sync
sleep 5

echo "All experiments completed."

runpodctl stop pod q27z8l6mtslzl3