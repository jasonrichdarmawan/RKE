## Installation

Please install mamba (a package manager) and tmux, then follow the following instructions:

1. Git clone, including the submodule

Note: This repository modified the `PEFT` library. The source code is available in the `peft` folder.

```
git clone --recurse-submodules -j8 [URL]
cd RKE
```

3. Install mamba (a package manager) and tmux
```
WORKSPACE=/workspace/rke

apt update
apt install -y tmux nvtop htop

yes | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

cd $WORKSPACE
tmux new -s rke
```

3. Install the python packages
```
tmux attach-session -t rke

WORKSPACE=$PWD

micromamba create --prefix $WORKSPACE/rke-env python=3.12 -y
micromamba activate $WORKSPACE/rke-env

pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
pip install "transformers[torch]==4.40.1"
pip install "datasets==3.6.0"
pip install "hf-transfer==0.1.9"
pip install "peft==0.10.0"

pip install "jaxtyping==0.3.3"
pip install "beartype==0.22.2"

pip install "python-dotenv==1.1.1"

pip install "tabulate==0.9.0"
pip install "ipykernel==6.30.1"

pip install "einops==0.8.1"
pip install "higher==0.2.1"
pip install "hydra-core==1.3.2"
pip install "matplotlib==3.10.7"
pip install "spacy==3.8.7"
pip install "scipy==1.16.2"
pip install "scikit-learn==1.7.2"
pip install "nltk==3.9.2"
pip install "accelerate==1.10.1"

pip install "rouge==1.0.1"
pip install "sentence-transformers==3.1.1"
```

4. Activate the mamba environment
```
tmux attach-session -t rke

WORKSPACE=/workspace/rke

micromamba activate $WORKSPACE/rke-env

cd $WORKSPACE/RKE
```

5. Prepare the `.env` and `.env_stats` file

Note: For convenience, `.env` file is meant for Ours and `.env_stats` file is meant for baselines. The only difference is the `GLOBALS_YAML` variable value.

`.env` file
```
HF_HOME=/workspace/rke/.cache/huggingface
HF_TOKEN=

CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT=/workspace/rke/RKE

PYTHONPATH=/workspace/rke/RKE:/workspace/rke/RKE/peft/src:$PYTHONPATH

GLOBALS_YAML=globals.yml
```

`.env_stats`
```
HF_HOME=/workspace/rke/.cache/huggingface
HF_TOKEN=

CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT=/workspace/rke/RKE

PYTHONPATH=/workspace/rke/RKE:/workspace/rke/RKE/peft/src:$PYTHONPATH

GLOBALS_YAML=globals_stats.yml
```

## Usage

To generate the output for evaluation, run the following:

1. Original outputs
```
set -a
source .env
set +a

dataset_size_limit=1000
num_edits=20
downstream_eval_steps=5
hparams_fname=Llama3-8B-Instruct.json
ds_name=unke

alg_name=original

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval
```

2. Ours outputs
```
set -a
source .env
set +a

dataset_size_limit=1000
num_edits=20
downstream_eval_steps=5
hparams_fname=Llama3-8B-Instruct.json
ds_name=unke

alg_name=unke

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval

alg_name=unke_ARE

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval

alg_name=unke_Alpha

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval

alg_name=unke_Alpha_ARE

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval
```

3. Baselines outputs
```
set -a
source .env_stats
set +a

dataset_size_limit=1000
num_edits=20
downstream_eval_steps=5
hparams_fname=Llama3-8B-Instruct.json
ds_name=unke

alg_name=MEMIT

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval

alg_name=MEMIT_ARE

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval

alg_name=AlphaEdit

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval

alg_name=AlphaEdit_ARE

python3 -m experiments.evaluate_uns     --alg_name=$alg_name     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=$dataset_size_limit     --num_edits=$num_edits     --downstream_eval_steps=$downstream_eval_steps     --sequential_eval
```
