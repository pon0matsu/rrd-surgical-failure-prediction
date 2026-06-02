from __future__ import annotations

from pathlib import Path

import miceforest as mf
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

from review_common import TARGET_COLUMN
from review_modeling import (
    ID_COLUMN,
    OLD_UNDERSAMPLING_RANDOM_STATE,
    binary_target,
    validate_features_exist,
)


# %%
# Submitted notebook settings
CV_FOLDS = 5
IMPUTATION_DATASETS = 5
IMPUTATION_ITERATIONS = 5
IMPUTATION_RANDOM_STATE = 0
SPLIT_RANDOM_STATE = 42


# %%
# Imputation corresponding to submitted Step 03
def feature_columns_for_imputation(frame: pd.DataFrame, excluded_columns: set[str] | None = None) -> list[str]:
    excluded = {ID_COLUMN, TARGET_COLUMN}
    if excluded_columns:
        excluded.update(excluded_columns)
    return [column for column in frame.columns if column not in excluded]


def impute_new_data_from_training(
    train_before_imputation: pd.DataFrame,
    new_before_imputation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_features_exist(train_before_imputation, feature_columns, "imputation training set")
    validate_features_exist(new_before_imputation, feature_columns, "imputation new set")
    x_train = train_before_imputation[feature_columns].apply(pd.to_numeric, errors="coerce")
    x_new = new_before_imputation[feature_columns].apply(pd.to_numeric, errors="coerce")
    kernel = mf.ImputationKernel(
        x_train,
        datasets=IMPUTATION_DATASETS,
        save_all_iterations=True,
        random_state=IMPUTATION_RANDOM_STATE,
    )
    kernel.mice(IMPUTATION_ITERATIONS, num_threads=1)
    train_features = kernel.complete_data(0)
    new_features = kernel.impute_new_data(x_new).complete_data(0)
    train_after = pd.concat(
        [
            train_before_imputation[[ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True),
            train_features.reset_index(drop=True),
        ],
        axis=1,
    )
    new_after = pd.concat(
        [
            new_before_imputation[[ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True),
            new_features.reset_index(drop=True),
        ],
        axis=1,
    )
    return train_after, new_after


def write_train_holdout_imputation_files(
    train_before: pd.DataFrame,
    holdout_before: pd.DataFrame,
    train_after: pd.DataFrame,
    holdout_after: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_before_imputation": output_dir / f"{prefix}_train_data_before_imputation.csv",
        "train_after_imputation": output_dir / f"{prefix}_train_data_after_imputation.csv",
        "holdout_before_imputation": output_dir / f"{prefix}_holdout_data_before_imputation.csv",
        "holdout_after_imputation": output_dir / f"{prefix}_holdout_data_after_imputation.csv",
    }
    train_before.to_csv(paths["train_before_imputation"], index=False, encoding="utf-8-sig")
    train_after.to_csv(paths["train_after_imputation"], index=False, encoding="utf-8-sig")
    holdout_before.to_csv(paths["holdout_before_imputation"], index=False, encoding="utf-8-sig")
    holdout_after.to_csv(paths["holdout_after_imputation"], index=False, encoding="utf-8-sig")
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "file": path.name,
                "rows": len(pd.read_csv(path, low_memory=False, encoding="utf-8-sig")),
            }
            for name, path in paths.items()
        ]
    )


def build_fold_imputation_files_like_submitted_step03(
    train_before_imputation: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    prefix: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = binary_target(train_before_imputation)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_RANDOM_STATE)
    folds = []
    rows = []
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(train_before_imputation[feature_columns], y)
    ):
        fold_train_before = train_before_imputation.iloc[train_index].copy()
        fold_validation_before = train_before_imputation.iloc[validation_index].copy()
        fold_train_after, fold_validation_after = impute_new_data_from_training(
            fold_train_before,
            fold_validation_before,
            feature_columns,
        )
        paths = {
            "train_before": output_dir / f"{prefix}_fold_{fold_index}_train_data_before_imputation.csv",
            "train_after": output_dir / f"{prefix}_fold_{fold_index}_train_data_after_imputation.csv",
            "validation_before": output_dir / f"{prefix}_fold_{fold_index}_val_data_before_imputation.csv",
            "validation_after": output_dir / f"{prefix}_fold_{fold_index}_val_data_after_imputation.csv",
        }
        fold_train_before.to_csv(paths["train_before"], index=False, encoding="utf-8-sig")
        fold_train_after.to_csv(paths["train_after"], index=False, encoding="utf-8-sig")
        fold_validation_before.to_csv(paths["validation_before"], index=False, encoding="utf-8-sig")
        fold_validation_after.to_csv(paths["validation_after"], index=False, encoding="utf-8-sig")
        folds.append(
            {
                "fold": fold_index,
                "train": fold_train_after,
                "validation": fold_validation_after,
                "train_after_file": paths["train_after"].name,
                "validation_after_file": paths["validation_after"].name,
            }
        )
        rows.append(
            {
                "fold": fold_index,
                "train_before_file": paths["train_before"].name,
                "train_after_file": paths["train_after"].name,
                "validation_before_file": paths["validation_before"].name,
                "validation_after_file": paths["validation_after"].name,
                "train_n": len(fold_train_after),
                "train_events": int(binary_target(fold_train_after).sum()),
                "validation_n": len(fold_validation_after),
                "validation_events": int(binary_target(fold_validation_after).sum()),
            }
        )
    return folds, pd.DataFrame(rows)


# %%
# Submitted Step 06 random under-sampling files
def write_undersampling_audit(
    frame_after_imputation: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = frame_after_imputation[[ID_COLUMN, *feature_columns]].copy()
    y_train = binary_target(frame_after_imputation)
    sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
    x_rus, y_rus = sampler.fit_resample(x_train, y_train)
    excluded_index = [index for index in range(len(x_train)) if index not in sampler.sample_indices_]
    x_excluded = x_train.iloc[excluded_index].copy()
    y_excluded = y_train[excluded_index]
    undersampled = x_rus.copy()
    undersampled[TARGET_COLUMN] = y_rus
    excluded = x_excluded.copy()
    excluded[TARGET_COLUMN] = y_excluded
    undersampled_path = output_dir / f"{prefix}_train_data_after_imputation_undersampled.csv"
    excluded_path = output_dir / f"{prefix}_train_data_after_imputation_excluded.csv"
    undersampled.to_csv(undersampled_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(
        [
            {
                "submitted_step": "06_Undersampling.ipynb",
                "fold": "final_train",
                "prefix": prefix,
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_after_resampling": len(y_rus),
                "train_events_after_resampling": int(np.sum(y_rus)),
                "excluded_n": len(y_excluded),
                "excluded_events": int(np.sum(y_excluded)),
                "undersampled_file": undersampled_path.name,
                "excluded_file": excluded_path.name,
            }
        ]
    )


def write_fold_undersampling_files(
    folds_after_imputation: list[dict[str, object]],
    feature_columns: list[str],
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds_after_imputation:
        fold_index = int(fold["fold"])
        train = fold["train"]
        x_train = train[[ID_COLUMN, *feature_columns]].copy()
        y_train = binary_target(train)
        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_rus, y_rus = sampler.fit_resample(x_train, y_train)
        excluded_index = [index for index in range(len(x_train)) if index not in sampler.sample_indices_]
        x_excluded = x_train.iloc[excluded_index].copy()
        y_excluded = y_train[excluded_index]

        undersampled = x_rus.copy()
        undersampled[TARGET_COLUMN] = y_rus
        excluded = x_excluded.copy()
        excluded[TARGET_COLUMN] = y_excluded

        undersampled_path = output_dir / f"{prefix}_fold_{fold_index}_undersampled_train_data.csv"
        excluded_path = output_dir / f"{prefix}_fold_{fold_index}_excluded_train_data.csv"
        undersampled.to_csv(undersampled_path, index=False, encoding="utf-8-sig")
        excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "submitted_step": "06_Undersampling.ipynb",
                "fold": fold_index,
                "prefix": prefix,
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_after_resampling": len(y_rus),
                "train_events_after_resampling": int(np.sum(y_rus)),
                "excluded_n": len(y_excluded),
                "excluded_events": int(np.sum(y_excluded)),
                "undersampled_file": undersampled_path.name,
                "excluded_file": excluded_path.name,
            }
        )
    return pd.DataFrame(rows)
