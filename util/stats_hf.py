# %%

from dotenv import load_dotenv
load_dotenv()

# %%

from util import is_notebook

import os

if is_notebook():
    os.chdir(os.environ["PROJECT_ROOT"])

# %%

import sys
import argparse

from huggingface_hub import HfApi, snapshot_download

# %%

if is_notebook():
    sys.argv = [
        "main.py",
        "--repo_id", "jasonrichdarmawan/knowledge-edit-stats",
        "--folder_path", "/workspace/jason/AnyEdit/data/stats",
    ]

# %%

parser = argparse.ArgumentParser(
    description="Upload statistics to Hugging Face"
)

parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="The ID of the repository to upload to (e.g., username/repo_name).",
)

parser.add_argument(
    "--folder_path",
    type=str,
    required=True,
    help="The path to the local folder containing files to upload.",
)

args = parser.parse_args()
print(args)

# %%

api = HfApi()
api.upload_large_folder(
    repo_id=args.repo_id,
    repo_type="dataset",
    folder_path=args.folder_path,
)

raise SystemExit

# %%

snapshot_download(
    repo_id=args.repo_id,
    repo_type="dataset",
    local_dir=args.folder_path,
)

# %%
