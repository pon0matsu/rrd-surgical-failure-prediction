#!/usr/bin/env python3
"""Check the compact Review conda environment on macOS CPU.

This check imports the packages needed by Review/ and confirms that TabPFN is
configured for CPU execution.  It does not run model refits and does not
download private data or checkpoints.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


REVIEW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW_ROOT))


PACKAGE_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("joblib", "joblib"),
    ("imblearn", "imbalanced-learn"),
    ("miceforest", "miceforest"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("torch", "pytorch"),
    ("tabpfn", "tabpfn"),
    ("requests", "requests"),
]


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "imported"))


def main() -> None:
    print("Review environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"REVIEW_TABPFN_DEVICE: {os.environ.get('REVIEW_TABPFN_DEVICE', 'cpu')}")
    print("")

    failures = []
    for module_name, package_name in PACKAGE_IMPORTS:
        try:
            version = version_of(module_name)
            print(f"OK {package_name}: {version}")
        except Exception as exc:  # noqa: BLE001 - report all import failures clearly.
            failures.append((package_name, repr(exc)))
            print(f"FAIL {package_name}: {exc}")

    try:
        import torch

        print("")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        print(f"torch.backends.mps.is_available(): {mps_available}")
        print("Review default is CPU; CUDA/MPS availability is not required.")
    except Exception as exc:  # noqa: BLE001
        failures.append(("torch runtime", repr(exc)))

    try:
        from review_modeling import TABPFN_DEVICE, tabpfn_checkpoint_status

        status = tabpfn_checkpoint_status()
        print("")
        print(f"Review TabPFN device used by scripts: {TABPFN_DEVICE}")
        print(f"TabPFN package found: {status['tabpfn_package_found']}")
        print(f"TabPFN checkpoint exists: {status['checkpoint_exists']}")
        print(f"Expected checkpoint: {status['expected_checkpoint_file']}")
        if not status["checkpoint_exists"]:
            print(
                "Checkpoint is needed only for steps that refit TabPFN models "
                "(for example Steps 42, 47, 48, and 49)."
            )
    except Exception as exc:  # noqa: BLE001
        failures.append(("Review TabPFN status", repr(exc)))

    if failures:
        print("")
        print("Environment check failed:")
        for package_name, error in failures:
            print(f"- {package_name}: {error}")
        raise SystemExit(1)

    print("")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
