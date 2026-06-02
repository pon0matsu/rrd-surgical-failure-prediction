"""Step 49-01: prepare strict-endpoint train/hold-out data.

This staged script mirrors the data-preparation part of the submitted
imputation workflow after redefining the endpoint:

1. keep the submitted PPV train/hold-out split;
2. exclude original Failure level 2 rows;
3. recode level 0 as success and levels 1/3 as strict anatomical failure;
4. refit MICE on the strict-endpoint training set;
5. impute the strict-endpoint hold-out set from that training imputer;
6. save private imputed train/hold-out and fold files for auditability;
7. save Step 06 random-under-sampling train/excluded files as private
   intermediates.

No model fitting is performed in this staged script.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_common import SUBMITTED_CODE_ROOT  # noqa: E402
from review_submitted_rebuild_pipeline import (  # noqa: E402
    build_fold_imputation_files_like_submitted_step03,
    feature_columns_for_imputation,
    impute_new_data_from_training,
    write_fold_undersampling_files,
    write_train_holdout_imputation_files,
    write_undersampling_audit,
)


# %%
# Paths and settings
STEP_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = STEP_DIR / "49_strict_endpoint_sensitivity.py"
PRIVATE_DATA_DIR = STEP_DIR / "local_intermediate_rebuild_data_private"
PRIVATE_FOLD_DIR = STEP_DIR / "local_intermediate_fold_files_private"
PRIVATE_ROW_EXPORT_DIR = STEP_DIR / "local_row_exports_private"

SUBMITTED_AFTER_IMPUTATION_REFERENCE_CSV = SUBMITTED_CODE_ROOT / "train_data_after_imputation.csv"
TRAIN_AFTER_CSV = PRIVATE_DATA_DIR / "strict_endpoint_train_data_after_imputation.csv"
HOLDOUT_AFTER_CSV = PRIVATE_DATA_DIR / "strict_endpoint_holdout_data_after_imputation.csv"
MERGED_AFTER_CSV = PRIVATE_ROW_EXPORT_DIR / "strict_endpoint_train_test_merged_after_imputation.csv"
MERGED_SOURCE_INDEX_CSV = PRIVATE_ROW_EXPORT_DIR / "strict_endpoint_train_test_merged_after_imputation_source_index.csv"
MERGED_SCHEMA_CHECK_CSV = PRIVATE_ROW_EXPORT_DIR / "strict_endpoint_train_test_merged_after_imputation_column_schema_check.csv"


# Local core loader
def load_core_module():
    spec = importlib.util.spec_from_file_location("step49_strict_endpoint_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load core script: {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_private_data_dir() -> None:
    if PRIVATE_DATA_DIR.exists():
        if STEP_DIR.resolve() not in PRIVATE_DATA_DIR.resolve().parents:
            raise RuntimeError(f"Refusing to clear unexpected directory: {PRIVATE_DATA_DIR}")
        shutil.rmtree(PRIVATE_DATA_DIR)
    PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def reset_private_row_export_dir() -> None:
    if PRIVATE_ROW_EXPORT_DIR.exists():
        if STEP_DIR.resolve() not in PRIVATE_ROW_EXPORT_DIR.resolve().parents:
            raise RuntimeError(f"Refusing to clear unexpected directory: {PRIVATE_ROW_EXPORT_DIR}")
        shutil.rmtree(PRIVATE_ROW_EXPORT_DIR)
    PRIVATE_ROW_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def submitted_after_imputation_columns() -> list[str]:
    return list(pd.read_csv(SUBMITTED_AFTER_IMPUTATION_REFERENCE_CSV, nrows=0, encoding="utf-8-sig").columns)


def align_to_submitted_after_imputation_schema(
    train_after: pd.DataFrame,
    holdout_after: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_columns = submitted_after_imputation_columns()
    observed_columns = list(train_after.columns)
    holdout_columns = list(holdout_after.columns)

    missing_from_train = [column for column in reference_columns if column not in observed_columns]
    extra_in_train = [column for column in observed_columns if column not in reference_columns]
    missing_from_holdout = [column for column in reference_columns if column not in holdout_columns]
    extra_in_holdout = [column for column in holdout_columns if column not in reference_columns]

    schema_check = pd.DataFrame(
        [
            {
                "export_file": MERGED_AFTER_CSV.name,
                "reference_file": str(SUBMITTED_AFTER_IMPUTATION_REFERENCE_CSV.relative_to(SUBMITTED_CODE_ROOT)),
                "reference_column_count": len(reference_columns),
                "train_column_count": len(observed_columns),
                "holdout_column_count": len(holdout_columns),
                "train_columns_exact_match_before_reorder": observed_columns == reference_columns,
                "holdout_columns_exact_match_before_reorder": holdout_columns == reference_columns,
                "missing_from_train": "|".join(missing_from_train),
                "extra_in_train": "|".join(extra_in_train),
                "missing_from_holdout": "|".join(missing_from_holdout),
                "extra_in_holdout": "|".join(extra_in_holdout),
                "export_columns_reordered_to_submitted_reference": True,
                "private_patient_level_output": True,
            }
        ]
    )

    if missing_from_train or extra_in_train or missing_from_holdout or extra_in_holdout:
        raise ValueError("Strict-endpoint imputed tables do not match the submitted after-imputation column schema.")

    return train_after.loc[:, reference_columns].copy(), holdout_after.loc[:, reference_columns].copy(), schema_check


def write_train_test_merged_after_imputation(train_after: pd.DataFrame, holdout_after: pd.DataFrame) -> None:
    # REVIEWER REVISION ADDITION:
    # Save a private row-level export in the same shape as the submitted
    # train_test_merged_after_imputation.csv: train rows followed by hold-out
    # rows, without adding split metadata columns to the analysis matrix.
    train_after, holdout_after, schema_check = align_to_submitted_after_imputation_schema(train_after, holdout_after)

    merged = pd.concat([train_after, holdout_after], ignore_index=True)
    MERGED_AFTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_AFTER_CSV, index=False, encoding="utf-8-sig")

    source_index = pd.DataFrame(
        [
            {
                "export_file": MERGED_AFTER_CSV.name,
                "source_split": "train",
                "source_file": TRAIN_AFTER_CSV.name,
                "row_start_1_based": 1,
                "row_end_1_based": len(train_after),
                "rows": len(train_after),
                "private_patient_level_output": True,
            },
            {
                "export_file": MERGED_AFTER_CSV.name,
                "source_split": "holdout",
                "source_file": HOLDOUT_AFTER_CSV.name,
                "row_start_1_based": len(train_after) + 1,
                "row_end_1_based": len(merged),
                "rows": len(holdout_after),
                "private_patient_level_output": True,
            },
        ]
    )
    source_index.to_csv(MERGED_SOURCE_INDEX_CSV, index=False, encoding="utf-8-sig")
    schema_check.to_csv(MERGED_SCHEMA_CHECK_CSV, index=False, encoding="utf-8-sig")


# %%
# Run this stage
def run_analysis() -> None:
    core = load_core_module()
    core.prepare_output_dir(STEP_DIR)
    reset_private_data_dir()
    reset_private_row_export_dir()

    rfecv_features = core.read_rfecv36_features()
    train_before, holdout_before = core.load_submitted_train_holdout_before_imputation()
    train_before = core.add_original_failure_level(train_before)
    holdout_before = core.add_original_failure_level(holdout_before)
    case_flow = core.strict_endpoint_case_flow(train_before, holdout_before)
    strict_train_before = core.define_strict_endpoint(train_before)
    strict_holdout_before = core.define_strict_endpoint(holdout_before)
    imputation_features = feature_columns_for_imputation(
        strict_train_before,
        excluded_columns={core.ORIGINAL_FAILURE_LEVEL_COLUMN},
    )
    train_after, holdout_after = impute_new_data_from_training(
        strict_train_before,
        strict_holdout_before,
        imputation_features,
    )
    write_train_test_merged_after_imputation(train_after, holdout_after)
    write_train_holdout_imputation_files(
        strict_train_before,
        strict_holdout_before,
        train_after,
        holdout_after,
        PRIVATE_DATA_DIR,
        prefix="strict_endpoint",
    )

    fold_dir = core.prepare_private_intermediate_dir(STEP_DIR)
    folds, _fold_index = build_fold_imputation_files_like_submitted_step03(
        strict_train_before,
        imputation_features,
        fold_dir,
        prefix="strict_endpoint_rfecv36",
    )
    write_undersampling_audit(
        train_after,
        imputation_features,
        PRIVATE_DATA_DIR,
        prefix="strict_endpoint",
    )
    write_fold_undersampling_files(
        folds,
        imputation_features,
        fold_dir,
        prefix="strict_endpoint",
    )
    feature_policy = core.feature_policy_table(train_after, rfecv_features, core.PRIMARY_FEATURE_SET)
    policy = core.strict_endpoint_policy_table(feature_policy)

    core.write_csv(case_flow, core.CASE_FLOW_CSV)
    core.write_csv(policy, core.POLICY_CSV)


if __name__ == "__main__":
    run_analysis()
