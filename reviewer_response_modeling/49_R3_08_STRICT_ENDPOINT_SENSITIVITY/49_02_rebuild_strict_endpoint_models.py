from __future__ import annotations

from pathlib import Path
import ast
import contextlib
import importlib.util
import io
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_common import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, SUBMITTED_CODE_ROOT, wide_metric_summary  # noqa: E402
from review_modeling import (  # noqa: E402
    OLD_MODEL_RANDOM_STATE,
    OLD_UNDERSAMPLING_RANDOM_STATE,
    SUBMITTED_GRID_SEARCH_RANDOM_STATE,
    TABPFN_DEVICE,
    TABPFN_ENSEMBLES,
    binary_target,
    fit_model,
    make_tabpfn_model,
    numeric_features,
    positive_probability,
)

# %%
# Paths, submitted-model settings, and output files
STEP_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = STEP_DIR / "49_strict_endpoint_sensitivity.py"
PRIVATE_DATA_DIR = STEP_DIR / "local_intermediate_rebuild_data_private"
PRIVATE_FOLD_DIR = STEP_DIR / "local_intermediate_fold_files_private"
PRIVATE_PREDICTION_EXPORT_DIR = STEP_DIR / "local_prediction_exports_private"
PREDICTION_EXPORT_INDEX_CSV = PRIVATE_PREDICTION_EXPORT_DIR / "strict_endpoint_prediction_export_index.csv"

SUBMITTED_PREDICTION_REFERENCE_CSV = SUBMITTED_CODE_ROOT / "FINAL_UNDERSAMPLING/tabpfn/Results_UnderSampling_with_pred_proba.csv"
TRAIN_AFTER_CSV = PRIVATE_DATA_DIR / "strict_endpoint_train_data_after_imputation.csv"
HOLDOUT_AFTER_CSV = PRIVATE_DATA_DIR / "strict_endpoint_holdout_data_after_imputation.csv"

PRIMARY_FEATURE_SET = "submitted_rfecv36"
HOLDOUT_DATASET_NAME = "strict_endpoint_ppv_holdout"
CV_FOLDS = 5
POSITIVE_CLASS_LABEL = 1
DEFAULT_THRESHOLD = 0.5

MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT = True
FEATURE_SELECTION_RERUN_IN_THIS_SCRIPT = False
IMPUTATION_RERUN_IN_THIS_SCRIPT = False
UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT = True
CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = True
TABPFN_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = False
CALIBRATION_RERUN_IN_THIS_SCRIPT = False

CLASSICAL_MODEL_ORDER = ["LR", "rf", "xgb", "lgb"]
MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "LR": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
}

GRID_SEARCH_MODEL_LABELS = {
    "LR": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
}

STEP05_PARAM_GRIDS = {
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


# %%
# Local core loader and prepared data readers
def load_core_module():
    spec = importlib.util.spec_from_file_location("step49_strict_endpoint_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load core script: {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prepared_train_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_AFTER_CSV.is_file() or not HOLDOUT_AFTER_CSV.is_file():
        raise FileNotFoundError("Run 49_01_prepare_strict_endpoint_training_data.py before this script.")
    train_after = pd.read_csv(TRAIN_AFTER_CSV, low_memory=False, encoding="utf-8-sig")
    holdout_after = pd.read_csv(HOLDOUT_AFTER_CSV, low_memory=False, encoding="utf-8-sig")
    return train_after, holdout_after


def load_fold_train_validation(fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PRIVATE_FOLD_DIR / f"strict_endpoint_rfecv36_fold_{fold_index}_train_data_after_imputation.csv"
    validation_path = PRIVATE_FOLD_DIR / f"strict_endpoint_rfecv36_fold_{fold_index}_val_data_after_imputation.csv"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Run 49_01_prepare_strict_endpoint_training_data.py before this script.")
    train = pd.read_csv(train_path, low_memory=False, encoding="utf-8-sig")
    validation = pd.read_csv(validation_path, low_memory=False, encoding="utf-8-sig")
    return train, validation


def load_fold_undersampled_train(fold_index: int) -> pd.DataFrame:
    path = PRIVATE_FOLD_DIR / f"strict_endpoint_fold_{fold_index}_undersampled_train_data.csv"
    if not path.is_file():
        raise FileNotFoundError("Run 49_01_prepare_strict_endpoint_training_data.py before this script.")
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def load_final_undersampled_train() -> pd.DataFrame:
    path = PRIVATE_DATA_DIR / "strict_endpoint_train_data_after_imputation_undersampled.csv"
    if not path.is_file():
        raise FileNotFoundError("Run 49_01_prepare_strict_endpoint_training_data.py before this script.")
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def reset_private_prediction_export_dir() -> None:
    if PRIVATE_PREDICTION_EXPORT_DIR.exists():
        if STEP_DIR.resolve() not in PRIVATE_PREDICTION_EXPORT_DIR.resolve().parents:
            raise RuntimeError(f"Refusing to clear unexpected directory: {PRIVATE_PREDICTION_EXPORT_DIR}")
        shutil.rmtree(PRIVATE_PREDICTION_EXPORT_DIR)
    PRIVATE_PREDICTION_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def submitted_style_prediction_export_path(submitted_step: str, model_key: str) -> Path:
    if submitted_step == "09_Final_ORIGINAL.ipynb":
        return PRIVATE_PREDICTION_EXPORT_DIR / "FINAL_ORIGINAL" / model_key / "Results_ORIGINAL_with_pred_proba.csv"
    if submitted_step == "10_Final_UnderSampling.ipynb":
        return PRIVATE_PREDICTION_EXPORT_DIR / "FINAL_UNDERSAMPLING" / model_key / "Results_UnderSampling_with_pred_proba.csv"
    raise ValueError(f"Unexpected final submitted step: {submitted_step}")


def submitted_prediction_columns() -> list[str]:
    return list(pd.read_csv(SUBMITTED_PREDICTION_REFERENCE_CSV, nrows=0, encoding="utf-8-sig").columns)


def align_to_submitted_prediction_schema(export: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    reference_columns = submitted_prediction_columns()
    observed_columns = list(export.columns)
    missing = [column for column in reference_columns if column not in observed_columns]
    extra = [column for column in observed_columns if column not in reference_columns]
    if missing or extra:
        raise ValueError("Strict-endpoint prediction export does not match the submitted pred_proba column schema.")
    return export.loc[:, reference_columns].copy(), {
        "reference_prediction_csv": str(SUBMITTED_PREDICTION_REFERENCE_CSV.relative_to(SUBMITTED_CODE_ROOT)),
        "column_schema_matches_submitted_reference": observed_columns == reference_columns,
        "export_columns_reordered_to_submitted_reference": observed_columns != reference_columns,
    }


def write_holdout_prediction_export(
    holdout_after: pd.DataFrame,
    pred_proba: np.ndarray,
    *,
    analysis_name: str,
    submitted_step: str,
    resampling_condition: str,
    model_key: str,
    model_label: str,
    train_source: str,
) -> dict[str, object]:
    # REVIEWER REVISION ADDITION:
    # Keep the prediction CSV in the submitted Step 09/10 style: the complete
    # hold-out matrix plus a final pred_proba column, with no extra metadata
    # columns inside the row-level file.
    export = holdout_after.copy()
    export["pred_proba"] = pred_proba
    export, schema_info = align_to_submitted_prediction_schema(export)
    export_path = submitted_style_prediction_export_path(submitted_step, model_key)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(export_path, index=False, encoding="utf-8-sig")

    return {
        "analysis": analysis_name,
        "submitted_step": submitted_step,
        "feature_set": PRIMARY_FEATURE_SET,
        "resampling_condition": resampling_condition,
        "model": model_key,
        "model_label": model_label,
        "prediction_csv": str(export_path.relative_to(STEP_DIR)),
        "train_source": train_source,
        "holdout_source": HOLDOUT_AFTER_CSV.name,
        "rows": len(export),
        "columns": export.shape[1],
        "pred_proba_column": "pred_proba",
        **schema_info,
        "private_patient_level_output": True,
    }


# %%
# Submitted Step 05: hyperparameter tuning
def make_step05_base_models() -> dict[str, object]:
    return {
        "LR": LogisticRegression(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE),
        "rf": RandomForestClassifier(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE),
        "xgb": XGBClassifier(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE, verbosity=0),
        "lgb": LGBMClassifier(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE, verbosity=-1),
    }


def run_step05_grid_search(
    train_after: pd.DataFrame,
    feature_columns: list[str],
    *,
    analysis_name: str,
    use_random_undersampling: bool,
) -> pd.DataFrame:
    x_all = numeric_features(train_after, feature_columns, "Step05 tuning train")
    y_all = binary_target(train_after)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.30,
        random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE,
        stratify=y_all,
    )

    if use_random_undersampling:
        rus = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_fit, y_fit = rus.fit_resample(x_train, y_train)
        resampling_condition = "random_undersampling"
        resampling_note = f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE})"
    else:
        x_fit, y_fit = x_train, y_train
        resampling_condition = "no_undersampling"
        resampling_note = "none"

    rows = []
    models = make_step05_base_models()
    for model_key in CLASSICAL_MODEL_ORDER:
        grid_search = GridSearchCV(
            models[model_key],
            STEP05_PARAM_GRIDS[model_key],
            cv=5,
            scoring="roc_auc",
            n_jobs=1,
            verbose=0,
        )
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            grid_search.fit(x_fit, y_fit)

        rows.append(
            {
                "analysis": analysis_name,
                "feature_set": PRIMARY_FEATURE_SET,
                "submitted_step": "05_Hyperparameter_Tuning.ipynb",
                "resampling_condition": resampling_condition,
                "resampling_inside_internal_training_split": resampling_note,
                "model": model_key,
                "model_label": GRID_SEARCH_MODEL_LABELS[model_key],
                "grid_source": "submitted_05_Hyperparameter_Tuning.ipynb",
                "base_model_random_state": SUBMITTED_GRID_SEARCH_RANDOM_STATE,
                "grid_search_cv": "GridSearchCV(cv=5, scoring='roc_auc')",
                "internal_train_n_before_resampling": len(y_train),
                "internal_train_events_before_resampling": int(np.sum(y_train)),
                "internal_train_n_used_for_grid_search": len(y_fit),
                "internal_train_events_used_for_grid_search": int(np.sum(y_fit)),
                "internal_test_n": len(y_test),
                "internal_test_events": int(np.sum(y_test)),
                "best_parameters": str(dict(grid_search.best_params_)),
                "best_cv_auroc": float(grid_search.best_score_),
                "internal_test_auroc": float(grid_search.score(x_test, y_test)),
            }
        )
    return pd.DataFrame(rows)


def best_parameters_by_resampling(step05_tuning: pd.DataFrame) -> dict[str, dict[str, dict[str, object]]]:
    best: dict[str, dict[str, dict[str, object]]] = {"no_undersampling": {}, "random_undersampling": {}}
    for row in step05_tuning.itertuples(index=False):
        best[str(row.resampling_condition)][str(row.model)] = ast.literal_eval(str(row.best_parameters))
    return best


# %%
# Submitted-style model definitions
def make_tabpfn_like_submitted_notebook() -> object:
    # Submitted notebooks used TabPFNClassifier(device='cuda', N_ensemble_configurations=32).
    # The Review environment uses the same ensemble count and a local CPU device by default.
    return make_tabpfn_model()


def make_classical_models_from_step05_best(
    best_parameters: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "LR": LogisticRegression(**{**best_parameters["LR"], "random_state": OLD_MODEL_RANDOM_STATE}),
        "rf": RandomForestClassifier(**{**best_parameters["rf"], "random_state": OLD_MODEL_RANDOM_STATE}),
        "xgb": XGBClassifier(**{**best_parameters["xgb"], "random_state": OLD_MODEL_RANDOM_STATE, "verbosity": 0}),
        "lgb": LGBMClassifier(**{**best_parameters["lgb"], "random_state": OLD_MODEL_RANDOM_STATE, "verbosity": -1}),
    }


def fit_model_and_get_positive_probability(model: object, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, float, str]:
    start = time.perf_counter()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        fit_model(model, x_train, y_train)
    fit_seconds = time.perf_counter() - start
    return positive_probability(model, x_eval), fit_seconds, model.__class__.__name__


def metric_row(
    y_true: np.ndarray,
    pred_proba: np.ndarray,
    *,
    dataset: str,
    model_key: str,
) -> dict[str, object]:
    return wide_metric_summary(
        y_true,
        pred_proba,
        dataset=dataset,
        model=model_key,
        model_label=MODEL_LABELS[model_key],
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )


# %%
# Submitted Step 07 and Step 08: five-fold CV
def run_step07_cv_original(
    feature_columns: list[str],
    best_parameters: dict[str, dict[str, object]],
    *,
    analysis_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    status_rows = []
    for fold_index in range(CV_FOLDS):
        train, validation = load_fold_train_validation(fold_index)
        x_train = numeric_features(train, feature_columns, f"Step07 fold {fold_index} train")
        y_train = binary_target(train)
        x_val = numeric_features(validation, feature_columns, f"Step07 fold {fold_index} validation")
        y_val = binary_target(validation)

        models = {"tabpfn": make_tabpfn_like_submitted_notebook()}
        models.update(make_classical_models_from_step05_best(best_parameters))

        for model_key, model in models.items():
            status = {
                "analysis": analysis_name,
                "submitted_step": "07_CV_ORIGINAL.ipynb",
                "feature_set": PRIMARY_FEATURE_SET,
                "resampling_condition": "no_undersampling",
                "fold": fold_index,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "parameter_policy": "tabpfn_fixed_or_step05_selected_parameters",
                "best_parameters": "" if model_key == "tabpfn" else str(best_parameters[model_key]),
                "resampling": "none",
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(np.sum(y_train)),
                "train_n_used": len(y_train),
                "train_events_used": int(np.sum(y_train)),
                "validation_n": len(y_val),
                "validation_events": int(np.sum(y_val)),
                "feature_count": len(feature_columns),
                "fit_seconds": np.nan,
                "status": "started",
                "error": "",
            }
            try:
                pred_proba, fit_seconds, model_class = fit_model_and_get_positive_probability(model, x_train, y_train, x_val)
                status["fit_seconds"] = fit_seconds
                status["model_class"] = model_class
                metric_rows.append(
                    {
                        "analysis": analysis_name,
                        "submitted_step": "07_CV_ORIGINAL.ipynb",
                        "feature_set": PRIMARY_FEATURE_SET,
                        "resampling_condition": "no_undersampling",
                        "fold": fold_index,
                        "parameter_policy": status["parameter_policy"],
                        "best_parameters": status["best_parameters"],
                        "resampling": "none",
                        "feature_count": len(feature_columns),
                        **metric_row(
                            y_val,
                            pred_proba,
                            dataset=f"no_undersampling_validation_fold_{fold_index}",
                            model_key=model_key,
                        ),
                    }
                )
                status["status"] = "completed"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = repr(exc)
            status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows)


def run_step08_cv_undersampling(
    feature_columns: list[str],
    best_parameters: dict[str, dict[str, object]],
    *,
    analysis_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    status_rows = []
    for fold_index in range(CV_FOLDS):
        _, validation = load_fold_train_validation(fold_index)
        train_rus = load_fold_undersampled_train(fold_index)
        x_train = numeric_features(train_rus, feature_columns, f"Step08 fold {fold_index} undersampled train")
        y_train = binary_target(train_rus)
        x_val = numeric_features(validation, feature_columns, f"Step08 fold {fold_index} validation")
        y_val = binary_target(validation)

        models = {"tabpfn": make_tabpfn_like_submitted_notebook()}
        models.update(make_classical_models_from_step05_best(best_parameters))

        for model_key, model in models.items():
            status = {
                "analysis": analysis_name,
                "submitted_step": "08_CV_UnderSampling.ipynb",
                "feature_set": PRIMARY_FEATURE_SET,
                "resampling_condition": "random_undersampling",
                "fold": fold_index,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "parameter_policy": "tabpfn_fixed_or_step05_selected_parameters",
                "best_parameters": "" if model_key == "tabpfn" else str(best_parameters[model_key]),
                "resampling": f"fold_{fold_index}_undersampled_train_data.csv from Step06",
                "train_n_before_resampling": "",
                "train_events_before_resampling": "",
                "train_n_used": len(y_train),
                "train_events_used": int(np.sum(y_train)),
                "validation_n": len(y_val),
                "validation_events": int(np.sum(y_val)),
                "feature_count": len(feature_columns),
                "fit_seconds": np.nan,
                "status": "started",
                "error": "",
            }
            try:
                pred_proba, fit_seconds, model_class = fit_model_and_get_positive_probability(model, x_train, y_train, x_val)
                status["fit_seconds"] = fit_seconds
                status["model_class"] = model_class
                metric_rows.append(
                    {
                        "analysis": analysis_name,
                        "submitted_step": "08_CV_UnderSampling.ipynb",
                        "feature_set": PRIMARY_FEATURE_SET,
                        "resampling_condition": "random_undersampling",
                        "fold": fold_index,
                        "parameter_policy": status["parameter_policy"],
                        "best_parameters": status["best_parameters"],
                        "resampling": status["resampling"],
                        "feature_count": len(feature_columns),
                        **metric_row(
                            y_val,
                            pred_proba,
                            dataset=f"random_undersampling_validation_fold_{fold_index}",
                            model_key=model_key,
                        ),
                    }
                )
                status["status"] = "completed"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = repr(exc)
            status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows)


# %%
# Submitted Step 09 and Step 10: final hold-out evaluation
def run_final_holdout_step(
    train_after: pd.DataFrame,
    holdout_after: pd.DataFrame,
    feature_columns: list[str],
    best_parameters: dict[str, dict[str, object]],
    *,
    analysis_name: str,
    submitted_step: str,
    resampling_condition: str,
    train_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if resampling_condition == "random_undersampling":
        train_for_fit = load_final_undersampled_train()
    else:
        train_for_fit = train_after

    x_train = numeric_features(train_for_fit, feature_columns, f"{submitted_step} train")
    y_train = binary_target(train_for_fit)
    x_holdout = numeric_features(holdout_after, feature_columns, f"{submitted_step} holdout")
    y_holdout = binary_target(holdout_after)

    models = {"tabpfn": make_tabpfn_like_submitted_notebook()}
    models.update(make_classical_models_from_step05_best(best_parameters))

    metric_rows = []
    status_rows = []
    prediction_export_rows = []
    for model_key, model in models.items():
        status = {
            "analysis": analysis_name,
            "submitted_step": submitted_step,
            "feature_set": PRIMARY_FEATURE_SET,
            "resampling_condition": resampling_condition,
            "fold": "",
            "model": model_key,
            "model_label": MODEL_LABELS[model_key],
            "parameter_policy": "tabpfn_fixed_or_step05_selected_parameters",
            "best_parameters": "" if model_key == "tabpfn" else str(best_parameters[model_key]),
            "resampling": train_source,
            "train_n_before_resampling": "",
            "train_events_before_resampling": "",
            "train_n_used": len(y_train),
            "train_events_used": int(np.sum(y_train)),
            "holdout_n": len(y_holdout),
            "holdout_events": int(np.sum(y_holdout)),
            "feature_count": len(feature_columns),
            "fit_seconds": np.nan,
            "status": "started",
            "error": "",
        }
        try:
            pred_proba, fit_seconds, model_class = fit_model_and_get_positive_probability(model, x_train, y_train, x_holdout)
            status["fit_seconds"] = fit_seconds
            status["model_class"] = model_class
            prediction_export_rows.append(
                write_holdout_prediction_export(
                    holdout_after,
                    pred_proba,
                    analysis_name=analysis_name,
                    submitted_step=submitted_step,
                    resampling_condition=resampling_condition,
                    model_key=model_key,
                    model_label=MODEL_LABELS[model_key],
                    train_source=train_source,
                )
            )
            metric_rows.append(
                {
                    "analysis": analysis_name,
                    "submitted_step": submitted_step,
                    "feature_set": PRIMARY_FEATURE_SET,
                    "resampling_condition": resampling_condition,
                    "parameter_policy": status["parameter_policy"],
                    "best_parameters": status["best_parameters"],
                    "resampling": train_source,
                    "feature_count": len(feature_columns),
                    **metric_row(
                        y_holdout,
                        pred_proba,
                        dataset=HOLDOUT_DATASET_NAME,
                        model_key=model_key,
                    ),
                }
            )
            status["status"] = "completed"
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = repr(exc)
        status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows), pd.DataFrame(prediction_export_rows)


# %%
# Run this stage
def run_analysis() -> None:
    core = load_core_module()
    core.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reset_private_prediction_export_dir()
    core.require_tabpfn_checkpoint_available()

    print(f"[{STEP_DIR.name}] Step 49-02 model rebuild: TabPFN device={TABPFN_DEVICE}, ensembles={TABPFN_ENSEMBLES}", flush=True)

    # Submitted-style reading order:
    # 1. Load RFECV36 features and Step49-01 strict-endpoint train/hold-out tables.
    # 2. Run classical-model hyperparameter tuning on the strict-endpoint training set.
    # 3. Re-run Step07 original CV and Step08 undersampling CV.
    # 4. Re-run Step09 original final hold-out and Step10 undersampling final hold-out.
    # 5. Save the same three reviewer tables: tuning, metrics, and run status.
    rfecv_features = core.read_rfecv36_features()
    train_after, holdout_after = load_prepared_train_holdout()
    analysis_name = core.STRICT_ENDPOINT_LABEL

    # Step05_Hyperparameter_Tuning.ipynb style block.
    step05_tuning = pd.concat(
        [
            run_step05_grid_search(train_after, rfecv_features, analysis_name=analysis_name, use_random_undersampling=False),
            run_step05_grid_search(train_after, rfecv_features, analysis_name=analysis_name, use_random_undersampling=True),
        ],
        ignore_index=True,
    )
    best = best_parameters_by_resampling(step05_tuning)

    # Step07_CV_ORIGINAL.ipynb and Step08_CV_UnderSampling.ipynb style blocks.
    step07_metrics, step07_status = run_step07_cv_original(
        rfecv_features,
        best["no_undersampling"],
        analysis_name=analysis_name,
    )
    step08_metrics, step08_status = run_step08_cv_undersampling(
        rfecv_features,
        best["random_undersampling"],
        analysis_name=analysis_name,
    )

    # Step09_Final_ORIGINAL.ipynb and Step10_Final_UnderSampling.ipynb style blocks.
    step09_metrics, step09_status, step09_prediction_exports = run_final_holdout_step(
        train_after,
        holdout_after,
        rfecv_features,
        best["no_undersampling"],
        analysis_name=analysis_name,
        submitted_step="09_Final_ORIGINAL.ipynb",
        resampling_condition="no_undersampling",
        train_source="train_data_after_imputation.csv",
    )
    step10_metrics, step10_status, step10_prediction_exports = run_final_holdout_step(
        train_after,
        holdout_after,
        rfecv_features,
        best["random_undersampling"],
        analysis_name=analysis_name,
        submitted_step="10_Final_UnderSampling.ipynb",
        resampling_condition="random_undersampling",
        train_source="train_data_after_imputation_undersampled.csv from Step06",
    )

    cv_metrics = pd.concat([step07_metrics, step08_metrics], ignore_index=True)
    holdout_metrics = pd.concat([step09_metrics, step10_metrics], ignore_index=True)
    run_status = pd.concat([step07_status, step08_status, step09_status, step10_status], ignore_index=True)
    prediction_exports = pd.concat([step09_prediction_exports, step10_prediction_exports], ignore_index=True)

    cv_metrics = cv_metrics.copy()
    holdout_metrics = holdout_metrics.copy()
    cv_metrics.insert(0, "evaluation_stage", "cross_validation")
    holdout_metrics.insert(0, "evaluation_stage", "holdout")
    model_metrics = pd.concat([cv_metrics, holdout_metrics], ignore_index=True, sort=False)

    core.write_csv(step05_tuning, core.TUNING_RESULTS_CSV)
    core.write_csv(model_metrics, core.MODEL_METRICS_CSV)
    core.write_csv(run_status, core.MODEL_RUN_STATUS_CSV)
    prediction_exports.to_csv(PREDICTION_EXPORT_INDEX_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run_analysis()
