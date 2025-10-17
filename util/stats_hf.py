# %%

from util import is_notebook

# %%

import sys
import argparse

from huggingface_hub import HfApi

# %%

if is_notebook():
    FOLDER_PATH = "/workspace/jason/AnyEdit/data/stats"
    PATH_IN_REPO = "data/stats"

    # FOLDER_PATH = "/workspace/jason/AnyEdit/output"
    # PATH_IN_REPO = "output"

    sys.argv = [
        "main.py",
        "--repo_id", "jasonrichdarmawan/knowledge-edit-stats",
        "--folder_path", FOLDER_PATH,
        "--path_in_repo", PATH_IN_REPO,
    ]
else:
    raise SystemExit("This script is intended to be run in a notebook environment.")

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

parser.add_argument(
    "--path_in_repo",
    type=str,
    required=True,
    help="The path within the repository where files will be uploaded.",
)

args = parser.parse_args()
print(args)

# %%

api = HfApi()
api.upload_folder(
    repo_id=args.repo_id,
    repo_type="dataset",
    folder_path=args.folder_path,
    path_in_repo=args.path_in_repo,
)

# %%