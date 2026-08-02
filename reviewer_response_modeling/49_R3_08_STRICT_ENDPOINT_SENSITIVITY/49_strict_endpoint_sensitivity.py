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
    ORIGINAL_FAILURE_LEVEL_COLUMN,
    add_original_failure_level,
    feature_policy_table,
    load_submitted_train_holdout_before_imputation,
    prepare_private_intermediate_dir,
    read_rfecv36_features,
    require_tabpfn_checkpoint_available,
)


# %%
# Paths, endpoint definition, model settings, and analysis flags
STEP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STEP_DIR / "local_outputs"

CASE_FLOW_CSV = OUTPUT_DIR / "strict_endpoint_case_flow.csv"
POLICY_CSV = OUTPUT_DIR / "strict_endpoint_policy.csv"
TUNING_RESULTS_CSV = OUTPUT_DIR / "strict_endpoint_hyperparameter_tuning.csv"
MODEL_METRICS_CSV = OUTPUT_DIR / "strict_endpoint_metrics.csv"
MODEL_RUN_STATUS_CSV = OUTPUT_DIR / "strict_endpoint_model_run_status.csv"

STRICT_ENDPOINT_LABEL = "strict_anatomical_failure_excluding_retained_silicone_oil"
EXCLUDED_ORIGINAL_FAILURE_LEVEL = 2
STRICT_SUCCESS_LEVEL = 0
STRICT_FAILURE_LEVELS = [1, 3]
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
# Strict endpoint construction
def define_strict_endpoint(frame: pd.DataFrame) -> pd.DataFrame:
    strict = frame[~pd.to_numeric(frame[ORIGINAL_FAILURE_LEVEL_COLUMN], errors="coerce").eq(EXCLUDED_ORIGINAL_FAILURE_LEVEL)].copy()
    original_level = pd.to_numeric(strict[ORIGINAL_FAILURE_LEVEL_COLUMN], errors="raise").astype(int)
    strict[TARGET_COLUMN] = original_level.map({STRICT_SUCCESS_LEVEL: 0, 1: 1, 3: 1}).astype(int)
    return strict


def strict_endpoint_case_flow(train_before: pd.DataFrame, holdout_before: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("ppv_train", train_before), ("ppv_holdout", holdout_before)]:
        strict = define_strict_endpoint(frame)
        original_level = pd.to_numeric(frame[ORIGINAL_FAILURE_LEVEL_COLUMN], errors="raise").astype(int)
        rows.append(
            {
                "split": split_name,
                "original_n": len(frame),
                "original_events_level_1_to_3": int(pd.to_numeric(frame[TARGET_COLUMN], errors="raise").sum()),
                "excluded_level_2_retained_silicone_oil": int(original_level.eq(EXCLUDED_ORIGINAL_FAILURE_LEVEL).sum()),
                "strict_n": len(strict),
                "strict_events_level_1_or_3": int(pd.to_numeric(strict[TARGET_COLUMN], errors="raise").sum()),
                "strict_event_rate": float(pd.to_numeric(strict[TARGET_COLUMN], errors="raise").mean()),
            }
        )
    total = pd.concat([train_before, holdout_before], ignore_index=True)
    strict_total = define_strict_endpoint(total)
    total_level = pd.to_numeric(total[ORIGINAL_FAILURE_LEVEL_COLUMN], errors="raise").astype(int)
    rows.append(
        {
            "split": "ppv_total",
            "original_n": len(total),
            "original_events_level_1_to_3": int(pd.to_numeric(total[TARGET_COLUMN], errors="raise").sum()),
            "excluded_level_2_retained_silicone_oil": int(total_level.eq(EXCLUDED_ORIGINAL_FAILURE_LEVEL).sum()),
            "strict_n": len(strict_total),
            "strict_events_level_1_or_3": int(pd.to_numeric(strict_total[TARGET_COLUMN], errors="raise").sum()),
            "strict_event_rate": float(pd.to_numeric(strict_total[TARGET_COLUMN], errors="raise").mean()),
        }
    )
    return pd.DataFrame(rows)


def strict_endpoint_policy_table(feature_policy: pd.DataFrame) -> pd.DataFrame:
    feature_policy = feature_policy.copy()
    feature_policy.insert(0, "policy_section", "submitted_rfecv36_feature_policy")

    policy_rows = pd.DataFrame(
        [
            {
                "policy_section": "endpoint_definition",
                "item": "strict_endpoint",
                "value": STRICT_ENDPOINT_LABEL,
                "note": "Exclude original Failure level 2; recode level 0 as success and levels 1/3 as failure.",
            },
            {
                "policy_section": "endpoint_definition",
                "item": "excluded_original_failure_level",
                "value": EXCLUDED_ORIGINAL_FAILURE_LEVEL,
                "note": "Level 2 corresponds to retained silicone oil in the submitted failure-level definition.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "model_refit_performed",
                "value": MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT,
                "note": "Models are refit because the endpoint definition and imputed training set change.",
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
                "note": "MICE is refit after excluding level-2 rows.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "under_sampling_rerun",
                "value": UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT,
                "note": "Random under-sampling is rerun on the strict-endpoint training data.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "classical_model_hyperparameter_tuning_rerun",
                "value": CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT,
                "note": "LR/RF/XGB/LGB grid search is run on the strict-endpoint training data.",
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
            {
                "policy_section": "external_mapping_limit",
                "item": "wills_exact_strict_endpoint_mapping",
                "value": False,
                "note": (
                    "The submitted Wills file contains anatomical success/failure, but does not contain "
                    "Japanese-style Failure level 0/1/2/3 or retained silicone oil status at 6 months. "
                    "Exact level-2 exclusion therefore cannot be reproduced from the submitted Wills variables alone."
                ),
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
    print(f"[{STEP_DIR.name}] Start Step 49 staged workflow.", flush=True)
    run_stage_script(STEP_DIR / "49_01_prepare_strict_endpoint_training_data.py")
    run_stage_script(STEP_DIR / "49_02_rebuild_strict_endpoint_models.py")
    print(f"[{STEP_DIR.name}] Done.", flush=True)


if __name__ == "__main__":
    run_analysis()
