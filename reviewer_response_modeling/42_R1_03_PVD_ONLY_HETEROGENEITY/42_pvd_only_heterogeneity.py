from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_common import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    TARGET_COLUMN,
    prepare_output_dir,
    write_csv,
)
from review_modeling import (  # noqa: E402
    feature_policy_table,
    load_submitted_train_holdout_before_imputation,
    prepare_private_intermediate_dir,
    read_rfecv36_features,
    require_tabpfn_checkpoint_available,
)


# %%
# Paths, cohorts, model settings, and analysis flags
STEP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STEP_DIR / "local_outputs"

CASE_FLOW_CSV = OUTPUT_DIR / "pvd_only_case_flow.csv"
POLICY_CSV = OUTPUT_DIR / "pvd_only_policy.csv"
TUNING_RESULTS_CSV = OUTPUT_DIR / "pvd_only_hyperparameter_tuning.csv"
MODEL_METRICS_CSV = OUTPUT_DIR / "pvd_only_metrics.csv"
MODEL_RUN_STATUS_CSV = OUTPUT_DIR / "pvd_only_model_run_status.csv"

PVD_RELATED_COLUMN = "術前所見__主病名1_PVDに伴う弁状裂孔による裂孔原性網膜剥離"
PRIMARY_FEATURE_SET = "submitted_rfecv36"
PRIMARY_MODEL = "tabpfn"
RESAMPLING_CONDITIONS = {
    "no_undersampling": False,
    "random_undersampling": True,
}
POSITIVE_CLASS_LABEL = 1
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = BOOTSTRAP_SEED

MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT = True
FEATURE_SELECTION_RERUN_IN_THIS_SCRIPT = False
IMPUTATION_RERUN_IN_THIS_SCRIPT = True
UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT = True
CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = True
TABPFN_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = False
CALIBRATION_RERUN_IN_THIS_SCRIPT = False


# %%
# Cohort construction
def pvd_case_flow(train_before: pd.DataFrame, holdout_before: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("ppv_train", train_before), ("ppv_holdout", holdout_before)]:
        pvd = keep_pvd_related_rows(frame)
        rows.append(
            {
                "split": split_name,
                "all_ppv_n": len(frame),
                "all_ppv_events": int(pd.to_numeric(frame[TARGET_COLUMN], errors="raise").sum()),
                "pvd_related_n": len(pvd),
                "pvd_related_events": int(pd.to_numeric(pvd[TARGET_COLUMN], errors="raise").sum()),
                "pvd_related_event_rate": float(pd.to_numeric(pvd[TARGET_COLUMN], errors="raise").mean()),
                "excluded_non_pvd_n": len(frame) - len(pvd),
                "excluded_non_pvd_percent": float((len(frame) - len(pvd)) / len(frame)),
            }
        )
    total = pd.concat([train_before, holdout_before], ignore_index=True)
    total_pvd = keep_pvd_related_rows(total)
    rows.append(
        {
            "split": "ppv_total",
            "all_ppv_n": len(total),
            "all_ppv_events": int(pd.to_numeric(total[TARGET_COLUMN], errors="raise").sum()),
            "pvd_related_n": len(total_pvd),
            "pvd_related_events": int(pd.to_numeric(total_pvd[TARGET_COLUMN], errors="raise").sum()),
            "pvd_related_event_rate": float(pd.to_numeric(total_pvd[TARGET_COLUMN], errors="raise").mean()),
            "excluded_non_pvd_n": len(total) - len(total_pvd),
            "excluded_non_pvd_percent": float((len(total) - len(total_pvd)) / len(total)),
        }
    )
    return pd.DataFrame(rows)


def keep_pvd_related_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if PVD_RELATED_COLUMN not in frame.columns:
        raise KeyError(f"Missing PVD-related RRD column: {PVD_RELATED_COLUMN}")
    return frame[pd.to_numeric(frame[PVD_RELATED_COLUMN], errors="coerce").eq(1)].copy()


def pvd_policy_table(feature_policy: pd.DataFrame) -> pd.DataFrame:
    feature_policy = feature_policy.copy()
    feature_policy.insert(0, "policy_section", "submitted_rfecv36_feature_policy")

    policy_rows = pd.DataFrame(
        [
            {
                "policy_section": "cohort_definition",
                "item": "pvd_related_column",
                "value": PVD_RELATED_COLUMN,
                "note": "Rows are kept when this submitted preoperative diagnosis dummy equals 1.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "model_refit_performed",
                "value": MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT,
                "note": "Models are refit because the cohort is restricted to PVD-related RRD.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "feature_selection_rerun",
                "value": FEATURE_SELECTION_RERUN_IN_THIS_SCRIPT,
                "note": "The submitted RFECV36 feature set is retained.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "imputation_rerun",
                "value": IMPUTATION_RERUN_IN_THIS_SCRIPT,
                "note": "MICE is refit after restricting the submitted train/hold-out split to PVD-related rows.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "under_sampling_rerun",
                "value": UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT,
                "note": "Random under-sampling is rerun on the PVD-only training data.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "classical_model_hyperparameter_tuning_rerun",
                "value": CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT,
                "note": "LR/RF/XGB/LGB grid search is run on the PVD-only training data.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "tabpfn_hyperparameter_tuning_rerun",
                "value": TABPFN_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT,
                "note": "TabPFN is fit with fixed submitted-style settings, without grid search.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "calibration_rerun",
                "value": CALIBRATION_RERUN_IN_THIS_SCRIPT,
                "note": "No probability recalibration is performed.",
            },
        ]
    )
    return pd.concat([feature_policy, policy_rows], ignore_index=True, sort=False)


def run_stage_script(script_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load staged script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_analysis()


# %%
# Run the complete analysis
def run_analysis() -> None:
    print(f"[{STEP_DIR.name}] Start Step 42 staged workflow.", flush=True)
    run_stage_script(STEP_DIR / "42_01_prepare_pvd_only_training_data.py")
    run_stage_script(STEP_DIR / "42_02_rebuild_pvd_only_models.py")
    print(f"[{STEP_DIR.name}] Done.", flush=True)


if __name__ == "__main__":
    run_analysis()
