"""Review-local modeling data helpers.

These functions are copied in spirit from the submitted 20250725 notebooks and
the reviewer-revision audit scripts, but they live inside Review so the
reviewer package does not import code from the larger publication workspace.

Step-specific model rebuilding is kept in the relevant Step scripts.  This
module provides shared readers, submitted constants, TabPFN checkpoint handling,
and small data utilities.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import miceforest as mf
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from review_common import (
    REVIEW_ROOT,
    SUBMITTED_CODE_ROOT,
    TARGET_COLUMN,
    clean_binary_outcome,
)


# %%
# Submitted-data paths and model settings
ID_COLUMN = "ID"
ORIGINAL_FAILURE_LEVEL_COLUMN = "Failure level (6M)"

SUBMITTED_TRAIN_BEFORE_IMPUTATION_CSV = SUBMITTED_CODE_ROOT / "train_data_before_imputation.csv"
SUBMITTED_HOLDOUT_BEFORE_IMPUTATION_CSV = SUBMITTED_CODE_ROOT / "test_data_before_imputation.csv"
SUBMITTED_TRAIN_AFTER_IMPUTATION_CSV = SUBMITTED_CODE_ROOT / "train_data_after_imputation.csv"
SUBMITTED_HOLDOUT_AFTER_IMPUTATION_CSV = SUBMITTED_CODE_ROOT / "test_data_after_imputation.csv"
SUBMITTED_CLEANED_CSV = SUBMITTED_CODE_ROOT / "df_cleaned_final.csv"
SUBMITTED_RFECV_FEATURES_CSV = SUBMITTED_CODE_ROOT / "rfecv_features.csv"
SUBMITTED_PRIMARY_MODEL_PKL = SUBMITTED_CODE_ROOT / "FINAL_UNDERSAMPLING/tabpfn/tabpfn_model_UNDERSAMPLING_FINAL.pkl"
SUBMITTED_TABPFN_FINAL_PREDICTIONS_CSV = (
    SUBMITTED_CODE_ROOT / "FINAL_UNDERSAMPLING/tabpfn/Results_UnderSampling_with_pred_proba.csv"
)

OLD_MODEL_ORDER = ["tabpfn", "xgb", "LR", "rf", "lgb"]
OLD_MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "xgb": "XGBoost",
    "LR": "Logistic regression",
    "rf": "Random forest",
    "lgb": "LightGBM",
}
SUBMITTED_GRID_SEARCH_RANDOM_STATE = 42
OLD_MODEL_RANDOM_STATE = 123
OLD_CV_UNDERSAMPLING_MODEL_RANDOM_STATE = 0
OLD_UNDERSAMPLING_RANDOM_STATE = 42
TABPFN_DEVICE = os.environ.get("REVIEW_TABPFN_DEVICE", "cpu")
TABPFN_ENSEMBLES = 32
TABPFN_RANDOM_STATE = 0
TABPFN_CHECKPOINT_FILENAME = "prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
DEFAULT_TABPFN_MODEL_BASE_PATH = REVIEW_ROOT / "00_METHOD_REFERENCE" / "tabpfn_model_base"
TABPFN_MODEL_BASE_PATH_ENV = os.environ.get("REVIEW_TABPFN_MODEL_BASE_PATH", "")
TABPFN_MODEL_BASE_PATH = TABPFN_MODEL_BASE_PATH_ENV or str(DEFAULT_TABPFN_MODEL_BASE_PATH)
GRID_SEARCH_N_JOBS = 1
RIDGE_LOGISTIC_N_JOBS = 1
RIDGE_LOGISTIC_C_GRID = np.logspace(-4, 4, 9)

MODEL_REFIT_PERFORMED = True
FEATURE_SELECTION_RERUN = False
IMPUTATION_RERUN = True
UNDER_SAMPLING_RERUN = True
CALIBRATION_RERUN = False


OLD_FINAL_ORIGINAL_HYPERPARAMETERS = {
    "tabpfn": {
        "device_source": "cuda",
        "device_used_here": TABPFN_DEVICE,
        "N_ensemble_configurations": TABPFN_ENSEMBLES,
        "seed_source": "not explicitly set in submitted notebook",
        "seed_used_here": TABPFN_RANDOM_STATE,
    },
    "xgb": {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 200, "random_state": OLD_MODEL_RANDOM_STATE},
    "LR": {"C": 0.001, "penalty": "none", "solver": "saga", "random_state": OLD_MODEL_RANDOM_STATE},
    "rf": {"max_depth": 5, "n_estimators": 200, "random_state": OLD_MODEL_RANDOM_STATE},
    "lgb": {"learning_rate": 0.01, "max_depth": 5, "n_estimators": 50, "random_state": OLD_MODEL_RANDOM_STATE},
}

OLD_FINAL_UNDERSAMPLING_HYPERPARAMETERS = {
    "tabpfn": {
        "device_source": "cuda",
        "device_used_here": TABPFN_DEVICE,
        "N_ensemble_configurations": TABPFN_ENSEMBLES,
        "seed_source": "not explicitly set in submitted notebook",
        "seed_used_here": TABPFN_RANDOM_STATE,
    },
    "xgb": {"learning_rate": 0.01, "max_depth": 7, "n_estimators": 50, "random_state": OLD_MODEL_RANDOM_STATE},
    "LR": {"C": 0.001, "penalty": "none", "solver": "saga", "random_state": OLD_MODEL_RANDOM_STATE},
    "rf": {"max_depth": 10, "n_estimators": 100, "random_state": OLD_MODEL_RANDOM_STATE},
    "lgb": {"learning_rate": 0.01, "max_depth": 5, "n_estimators": 100, "random_state": OLD_MODEL_RANDOM_STATE},
}

OLD_CV_UNDERSAMPLING_HYPERPARAMETERS = {
    "tabpfn": OLD_FINAL_UNDERSAMPLING_HYPERPARAMETERS["tabpfn"],
    "xgb": {"learning_rate": 0.01, "max_depth": 7, "n_estimators": 50, "random_state": OLD_CV_UNDERSAMPLING_MODEL_RANDOM_STATE},
    "LR": {"C": 0.001, "penalty": "none", "solver": "saga", "random_state": OLD_CV_UNDERSAMPLING_MODEL_RANDOM_STATE},
    "rf": {"max_depth": 10, "n_estimators": 100, "random_state": OLD_CV_UNDERSAMPLING_MODEL_RANDOM_STATE},
    "lgb": {"learning_rate": 0.01, "max_depth": 5, "n_estimators": 100, "random_state": OLD_CV_UNDERSAMPLING_MODEL_RANDOM_STATE},
}

OLD_GRID_SEARCH_BEST_PARAMETERS = {
    "original": {
        "LR": {"C": 0.001, "penalty": "none", "solver": "saga"},
        "rf": {"max_depth": 5, "n_estimators": 200},
        "xgb": {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 200},
        "lgb": {"learning_rate": 0.01, "max_depth": 5, "n_estimators": 50},
    },
    "undersampling": {
        "LR": {"C": 0.001, "penalty": "none", "solver": "saga"},
        "rf": {"max_depth": 10, "n_estimators": 100},
        "xgb": {"learning_rate": 0.01, "max_depth": 7, "n_estimators": 50},
        "lgb": {"learning_rate": 0.01, "max_depth": 5, "n_estimators": 100},
    },
}

SUBMITTED_GRID_SEARCH_MODEL_LABELS = {
    "LR": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
}

SUBMITTED_GRID_SEARCH_GRIDS = {
    "LR": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["saga"],
        "penalty": ["none"],
    },
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
    },
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "lgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
}

COMPACT_FEATURE_COLUMNS = [
    "初回手術時年齢",
    "患者情報__性別",
    "術前所見_矯正視力_矯正視力",
    "術前所見_眼圧_眼圧",
    "術前所見__黄斑剥離",
    "術前所見_黄斑剥離期間_日",
    "術前所見__脈絡膜剥離",
    "V22a眼軸長(26mm以上)",
    "術前所見_裂孔数_個",
    "術前所見_最大裂孔大きさ_度_30-60",
    "術前所見_最大裂孔大きさ_度_60-90",
    "術前所見_最大裂孔大きさ_度_90-",
    "術前所見__最大裂孔位置_下耳側",
    "術前所見__最大裂孔位置_下鼻側",
    "術前所見_裂孔形態_種別_円孔",
    "術前所見_裂孔形態_種別_裂孔",
    "術前所見_裂孔形態_種別_黄斑円孔",
    "術前所見_網膜剥離範囲_現象_2",
    "術前所見_網膜剥離範囲_現象_3",
    "術前所見_網膜剥離範囲_現象_4",
    "術前所見_PVR_N/B/C_B",
    "術前所見_PVR_N/B/C_C",
    "術前所見__水晶体_IOL",
    "術前所見__水晶体_無水晶体眼",
    "眼手術歴_網膜剥離を除く網膜硝子体手術",
    "術前所見__主病名1_PVDに伴う弁状裂孔による裂孔原性網膜剥離",
]


# %%
# Feature and submitted input readers
def read_single_column_text_csv(path: Path) -> list[str]:
    frame = pd.read_csv(path, header=None, encoding="utf-8-sig")
    return frame.iloc[:, 0].dropna().astype(str).tolist()


def read_rfecv36_features() -> list[str]:
    features = read_single_column_text_csv(SUBMITTED_RFECV_FEATURES_CSV)
    if len(features) != 36:
        raise ValueError(f"Expected 36 submitted RFECV features, got {len(features)}")
    return features


def all46_features_from_submitted_after_imputation() -> list[str]:
    frame = pd.read_csv(SUBMITTED_TRAIN_AFTER_IMPUTATION_CSV, low_memory=False, encoding="utf-8-sig")
    feature_columns = [column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}]
    return [
        column
        for column in feature_columns
        if pd.to_numeric(frame[column], errors="coerce").nunique(dropna=True) > 1
    ]


def compact26_features() -> list[str]:
    return list(COMPACT_FEATURE_COLUMNS)


def validate_features_exist(frame: pd.DataFrame, feature_columns: list[str], dataset_name: str) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{dataset_name} missing required feature columns: {missing}")


def prepare_private_intermediate_dir(step_dir: Path) -> Path:
    output_dir = step_dir / "local_intermediate_fold_files_private"
    if output_dir.exists():
        if step_dir.resolve() not in output_dir.resolve().parents:
            raise RuntimeError(f"Refusing to clear unexpected intermediate directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_legacy_pickle_compatibility() -> None:
    import pathlib

    pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]
    try:
        import torch
    except Exception:
        return
    original_torch_load = torch.load

    def torch_load_cpu(*args: object, **kwargs: object) -> object:
        kwargs["map_location"] = torch.device("cpu")
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_cpu  # type: ignore[assignment]


def load_submitted_primary_tabpfn_model() -> object:
    configure_legacy_pickle_compatibility()
    model = joblib.load(SUBMITTED_PRIMARY_MODEL_PKL)
    try:
        import torch

        if hasattr(model, "device"):
            model.device = torch.device("cpu")
        inner_model = getattr(model, "model", None)
        if inner_model is not None and hasattr(inner_model, "to"):
            inner_model.to("cpu")
    except Exception:
        pass
    return model


def tabpfn_checkpoint_status() -> dict[str, object]:
    spec = importlib.util.find_spec("tabpfn")
    package_dir = Path(spec.origin).resolve().parent if spec and spec.origin else None
    package_checkpoint = (
        package_dir / "models_diff" / TABPFN_CHECKPOINT_FILENAME
        if package_dir is not None
        else Path("")
    )
    base_path = Path(TABPFN_MODEL_BASE_PATH).expanduser() if TABPFN_MODEL_BASE_PATH else None
    base_checkpoint = (
        base_path / "models_diff" / TABPFN_CHECKPOINT_FILENAME
        if base_path is not None
        else None
    )
    if base_checkpoint is not None and base_checkpoint.is_file():
        candidate = base_checkpoint
        runtime_base_path = str(base_path)
    elif package_checkpoint.is_file():
        candidate = package_checkpoint
        runtime_base_path = ""
    else:
        candidate = base_checkpoint if base_checkpoint is not None else package_checkpoint
        runtime_base_path = str(base_path) if base_path is not None else ""
    return {
        "tabpfn_package_found": package_dir is not None,
        "tabpfn_package_dir": str(package_dir) if package_dir is not None else "",
        "tabpfn_model_base_path_env": TABPFN_MODEL_BASE_PATH_ENV,
        "default_review_model_base_path": str(DEFAULT_TABPFN_MODEL_BASE_PATH),
        "runtime_model_base_path": runtime_base_path,
        "expected_checkpoint_file": str(candidate),
        "checkpoint_exists": bool(candidate.is_file()),
    }


def require_tabpfn_checkpoint_available() -> dict[str, object]:
    status = tabpfn_checkpoint_status()
    if not status["tabpfn_package_found"]:
        raise ModuleNotFoundError("tabpfn is not installed in the active Python environment.")
    if not status["checkpoint_exists"]:
        raise FileNotFoundError(
            "TabPFN checkpoint is required before running TabPFN model rebuilding. "
            f"Expected checkpoint file: {status['expected_checkpoint_file']}. "
            "Download the checkpoint once in this conda environment, or set "
            "REVIEW_TABPFN_MODEL_BASE_PATH to a directory containing "
            f"models_diff/{TABPFN_CHECKPOINT_FILENAME}."
        )
    return status


def write_stratified_imputed_fold_files(
    frame_after_imputation: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    *,
    prefix: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Save private fold-level train/validation CSVs for auditability."""

    columns = [ID_COLUMN, TARGET_COLUMN, *feature_columns]
    validate_features_exist(frame_after_imputation, feature_columns, prefix)
    y = binary_target(frame_after_imputation)
    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold_index, (train_index, validation_index) in enumerate(split.split(frame_after_imputation[feature_columns], y)):
        fold_train = frame_after_imputation.iloc[train_index][columns].copy()
        fold_validation = frame_after_imputation.iloc[validation_index][columns].copy()
        train_path = output_dir / f"{prefix}_fold_{fold_index}_train_data_after_imputation.csv"
        validation_path = output_dir / f"{prefix}_fold_{fold_index}_validation_data_after_imputation.csv"
        fold_train.to_csv(train_path, index=False, encoding="utf-8-sig")
        fold_validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "prefix": prefix,
                "fold": fold_index,
                "train_file": train_path.name,
                "validation_file": validation_path.name,
                "train_n": len(fold_train),
                "train_events": int(fold_train[TARGET_COLUMN].sum()),
                "validation_n": len(fold_validation),
                "validation_events": int(fold_validation[TARGET_COLUMN].sum()),
            }
        )
    return pd.DataFrame(rows)


def load_submitted_train_holdout_after_imputation() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SUBMITTED_TRAIN_AFTER_IMPUTATION_CSV, low_memory=False, encoding="utf-8-sig")
    holdout = pd.read_csv(SUBMITTED_HOLDOUT_AFTER_IMPUTATION_CSV, low_memory=False, encoding="utf-8-sig")
    return train, holdout


def load_submitted_after_imputation_fold_files(n_splits: int = 5) -> list[dict[str, object]]:
    folds = []
    for fold_index in range(n_splits):
        train_path = SUBMITTED_CODE_ROOT / f"fold_{fold_index}_train_data_after_imputation.csv"
        validation_path = SUBMITTED_CODE_ROOT / f"fold_{fold_index}_val_data_after_imputation.csv"
        if not train_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(f"Missing submitted fold files for fold {fold_index}")
        folds.append(
            {
                "fold": fold_index,
                "train": pd.read_csv(train_path, low_memory=False, encoding="utf-8-sig"),
                "validation": pd.read_csv(validation_path, low_memory=False, encoding="utf-8-sig"),
                "train_source_file": train_path.name,
                "validation_source_file": validation_path.name,
            }
        )
    return folds


def load_submitted_train_holdout_before_imputation() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SUBMITTED_TRAIN_BEFORE_IMPUTATION_CSV, low_memory=False, encoding="utf-8-sig")
    holdout = pd.read_csv(SUBMITTED_HOLDOUT_BEFORE_IMPUTATION_CSV, low_memory=False, encoding="utf-8-sig")
    return train, holdout


def add_original_failure_level(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.read_csv(
        SUBMITTED_CLEANED_CSV,
        usecols=[ID_COLUMN, ORIGINAL_FAILURE_LEVEL_COLUMN],
        low_memory=False,
        encoding="utf-8-sig",
    )
    cleaned = cleaned.drop_duplicates(subset=[ID_COLUMN])
    merged = frame.merge(cleaned, on=ID_COLUMN, how="left")
    if merged[ORIGINAL_FAILURE_LEVEL_COLUMN].isna().any():
        missing = int(merged[ORIGINAL_FAILURE_LEVEL_COLUMN].isna().sum())
        raise ValueError(f"Original failure level missing after ID merge for {missing} rows")
    return merged


# %%
# Imputation
def coerce_feature_matrix_for_miceforest(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    validate_features_exist(frame, feature_columns, "imputation input")
    out = frame[feature_columns].copy()
    for column in feature_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def fit_miceforest_on_training_rows(
    train_before_imputation: pd.DataFrame,
    holdout_before_imputation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = coerce_feature_matrix_for_miceforest(train_before_imputation, feature_columns)
    x_holdout = coerce_feature_matrix_for_miceforest(holdout_before_imputation, feature_columns)

    kernel = mf.ImputationKernel(
        x_train,
        datasets=5,
        save_all_iterations=True,
        random_state=0,
    )
    kernel.mice(5, num_threads=1)
    train_imputed_features = kernel.complete_data(0)
    holdout_imputed_features = kernel.impute_new_data(x_holdout).complete_data(0)

    train_imputed = pd.concat(
        [
            train_before_imputation[[ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True),
            train_imputed_features.reset_index(drop=True),
        ],
        axis=1,
    )
    holdout_imputed = pd.concat(
        [
            holdout_before_imputation[[ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True),
            holdout_imputed_features.reset_index(drop=True),
        ],
        axis=1,
    )
    return train_imputed, holdout_imputed


# %%
# Model fitting
def make_tabpfn_model() -> object:
    from tabpfn import TabPFNClassifier

    kwargs = {
        "device": TABPFN_DEVICE,
        "N_ensemble_configurations": TABPFN_ENSEMBLES,
        "seed": TABPFN_RANDOM_STATE,
    }
    runtime_base_path = tabpfn_checkpoint_status().get("runtime_model_base_path", "")
    if runtime_base_path:
        kwargs["base_path"] = runtime_base_path
    return TabPFNClassifier(**kwargs)


def old_final_hyperparameters(condition: str) -> dict[str, dict[str, object]]:
    if condition == "original":
        return OLD_FINAL_ORIGINAL_HYPERPARAMETERS
    if condition == "undersampling":
        return OLD_FINAL_UNDERSAMPLING_HYPERPARAMETERS
    raise ValueError(f"Unknown condition: {condition}")


def old_cv_hyperparameters(condition: str) -> dict[str, dict[str, object]]:
    if condition == "original":
        return OLD_FINAL_ORIGINAL_HYPERPARAMETERS
    if condition == "undersampling":
        return OLD_CV_UNDERSAMPLING_HYPERPARAMETERS
    raise ValueError(f"Unknown condition: {condition}")


def fit_model(model: object, x_train: np.ndarray, y_train: np.ndarray) -> object:
    if model.__class__.__name__ == "TabPFNClassifier":
        try:
            return model.fit(x_train, y_train, overwrite_warning=True)
        except TypeError:
            return model.fit(x_train, y_train)
    return model.fit(x_train, y_train)


def positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError(f"Model classes do not include positive class 1: {classes}")
    index = int(np.where(classes == 1)[0][0])
    return np.clip(probabilities[:, index], 1e-6, 1 - 1e-6)


def numeric_features(frame: pd.DataFrame, feature_columns: list[str], dataset_name: str) -> np.ndarray:
    validate_features_exist(frame, feature_columns, dataset_name)
    features = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        missing = features.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"{dataset_name} has missing values after imputation: {missing}")
    return features.to_numpy(dtype=np.float32)


def binary_target(frame: pd.DataFrame) -> np.ndarray:
    return clean_binary_outcome(frame[TARGET_COLUMN])


def old_hyperparameter_table() -> pd.DataFrame:
    rows = []
    for condition in ["original", "undersampling"]:
        final_params = old_final_hyperparameters(condition)
        cv_params = old_cv_hyperparameters(condition)
        grid_params = OLD_GRID_SEARCH_BEST_PARAMETERS.get(condition, {})
        for model_key in OLD_MODEL_ORDER:
            keys = sorted(
                set(final_params.get(model_key, {}))
                | set(cv_params.get(model_key, {}))
                | set(grid_params.get(model_key, {}))
            )
            for key in keys:
                rows.append(
                    {
                        "condition": condition,
                        "model": model_key,
                        "parameter": key,
                        "submitted_cv_notebook_value": cv_params.get(model_key, {}).get(key, ""),
                        "submitted_final_notebook_value": final_params.get(model_key, {}).get(key, ""),
                        "submitted_grid_search_best_value": grid_params.get(model_key, {}).get(key, ""),
                        "used_in_review_cv_reanalysis": cv_params.get(model_key, {}).get(key, ""),
                        "used_in_review_final_reanalysis": final_params.get(model_key, {}).get(key, ""),
                        "used_in_review_reanalysis": final_params.get(model_key, {}).get(key, ""),
                        "submitted_record_kept_unchanged": True,
                        "review_usage_policy": (
                            "Grid-search best values are retained as submitted provenance. "
                            "Review model rebuilding follows the values actually used in the "
                            "submitted CV and final model-fitting notebooks."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def feature_policy_table(train_frame: pd.DataFrame, feature_columns: list[str], feature_set_name: str) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        unique_n = int(pd.to_numeric(train_frame[feature], errors="coerce").nunique(dropna=True))
        rows.append(
            {
                "feature_set": feature_set_name,
                "feature": feature,
                "unique_values_in_training_rows": unique_n,
                "used_in_model": True,
            }
        )
    return pd.DataFrame(rows)
