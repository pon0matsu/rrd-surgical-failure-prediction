"""Download the TabPFN v1 checkpoint required for reviewer reanalysis.

Run this once before steps that refit TabPFN models.  The script writes the
checkpoint to REVIEW_TABPFN_MODEL_BASE_PATH/models_diff/ when that environment
variable is set.  Otherwise it writes inside this Review method-reference
folder so reviewer reanalysis does not need to modify the conda package
directory.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import requests


# %%
# Checkpoint source and destination
CHECKPOINT_FILENAME = "prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
CHECKPOINT_URL = (
    "https://github.com/PriorLabs/TabPFN/raw/refs/tags/v1.0.0/"
    "tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
)
MODEL_BASE_PATH_ENV = "REVIEW_TABPFN_MODEL_BASE_PATH"


# %%
# Helper functions
def installed_tabpfn_base_path() -> Path:
    spec = importlib.util.find_spec("tabpfn")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("tabpfn is not installed in the active Python environment.")
    return Path(spec.origin).resolve().parent


def checkpoint_base_path() -> Path:
    env_value = os.environ.get(MODEL_BASE_PATH_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parent / "tabpfn_model_base"


def checkpoint_path() -> Path:
    return checkpoint_base_path() / "models_diff" / CHECKPOINT_FILENAME


def download_checkpoint(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(CHECKPOINT_URL, timeout=300)
    response.raise_for_status()
    destination.write_bytes(response.content)


# %%
# Run
def run_analysis() -> None:
    destination = checkpoint_path()
    if destination.is_file():
        print(f"TabPFN checkpoint already exists: {destination}")
        return
    print(f"Downloading TabPFN checkpoint to: {destination}")
    download_checkpoint(destination)
    print(f"Downloaded TabPFN checkpoint: {destination}")


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as exc:
        print(f"Failed to prepare TabPFN checkpoint: {exc}", file=sys.stderr)
        raise
