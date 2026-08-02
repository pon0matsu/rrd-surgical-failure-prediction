from __future__ import annotations

from pathlib import Path
import contextlib
import io
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_common import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    prepare_output_dir,
    wide_metric_summary,
    write_csv,
)
from review_modeling import (  # noqa: E402
    OLD_MODEL_LABELS,
    OLD_MODEL_RANDOM_STATE,
    OLD_UNDERSAMPLING_RANDOM_STATE,
    SUBMITTED_GRID_SEARCH_RANDOM_STATE,
    ID_COLUMN,
    RIDGE_LOGISTIC_C_GRID,
    RIDGE_LOGISTIC_N_JOBS,
    TARGET_COLUMN,
    TABPFN_DEVICE,
    TABPFN_ENSEMBLES,
    all46_features_from_submitted_after_imputation,
    binary_target,
    compact26_features,
    feature_policy_table,
    fit_model,
    load_submitted_after_imputation_fold_files,
    load_submitted_train_holdout_after_imputation,
    make_tabpfn_model,
    numeric_features,
    positive_probability,
    prepare_private_intermediate_dir,
    read_rfecv36_features,
    require_tabpfn_checkpoint_available,
    validate_features_exist,
)


# %%
# Paths, feature policies, models, and analysis flags
STEP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STEP_DIR / "local_outputs"

POLICY_CSV = OUTPUT_DIR / "feature_selection_policy.csv"
TUNING_RESULTS_CSV = OUTPUT_DIR / "feature_selection_hyperparameter_tuning.csv"
MODEL_METRICS_CSV = OUTPUT_DIR / "feature_selection_metrics.csv"
MODEL_RUN_STATUS_CSV = OUTPUT_DIR / "feature_selection_model_run_status.csv"

POSITIVE_CLASS_LABEL = 1
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = BOOTSTRAP_SEED
CV_FOLDS = 5
RESAMPLING_CONDITIONS = {
    "no_undersampling": False,
    "random_undersampling": True,
}
CLASSICAL_MODELS = ["LR", "rf", "xgb", "lgb"]
MODEL_SEQUENCE = ["tabpfn", "LR", "ridge_logistic", "rf", "xgb", "lgb"]
MODEL_LABELS_FOR_THIS_STEP = {
    **OLD_MODEL_LABELS,
    "LR": "Logistic regression without ridge penalty",
    "ridge_logistic": "Ridge logistic regression",
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
LOGISTIC_MODEL_VARIANTS = {
    "LR": "submitted-style logistic regression without ridge penalty",
    "ridge_logistic": "ridge-penalized logistic regression with internal C selection",
}

MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT = True
FEATURE_SELECTION_RERUN_IN_THIS_SCRIPT = False
IMPUTATION_RERUN_IN_THIS_SCRIPT = False
UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT = True
CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = True
RIDGE_LOGISTIC_C_SELECTION_RERUN_IN_THIS_SCRIPT = True
TABPFN_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = False
CALIBRATION_RERUN_IN_THIS_SCRIPT = False


# %%
# Feature-set definitions
def feature_sets() -> dict[str, list[str]]:
    return {
        "all46": all46_features_from_submitted_after_imputation(),
        "compact26": compact26_features(),
        "rfecv36": read_rfecv36_features(),
    }


def feature_count_table(feature_sets_by_name: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "feature_count": len(columns),
            }
            for feature_set, columns in feature_sets_by_name.items()
        ]
    )


def logistic_model_variant_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_key,
                "model_label": MODEL_LABELS_FOR_THIS_STEP[model_key],
                "definition": definition,
                "ridge_penalty_used": model_key == "ridge_logistic",
                "regularization_strength_selected_inside_training_data": model_key == "ridge_logistic",
            }
            for model_key, definition in LOGISTIC_MODEL_VARIANTS.items()
        ]
    )


def feature_selection_policy_table(
    feature_sets_by_name: dict[str, list[str]],
    feature_policy: pd.DataFrame,
    checkpoint_status: dict[str, object],
) -> pd.DataFrame:
    feature_policy = feature_policy.copy()
    feature_policy.insert(0, "policy_section", "feature_policy")
    feature_policy["feature_count"] = feature_policy["feature_set"].map(
        {feature_set: len(columns) for feature_set, columns in feature_sets_by_name.items()}
    )

    counts = feature_count_table(feature_sets_by_name)
    counts.insert(0, "policy_section", "feature_set_counts")

    logistic_variants = logistic_model_variant_table()
    logistic_variants.insert(0, "policy_section", "logistic_model_variants")

    checkpoint = pd.DataFrame(
        [
            {
                "policy_section": "tabpfn_checkpoint_status",
                "model": "tabpfn",
                "tabpfn_device": TABPFN_DEVICE,
                "tabpfn_ensembles": TABPFN_ENSEMBLES,
                **checkpoint_status,
            }
        ]
    )

    analysis_flags = pd.DataFrame(
        [
            {
                "policy_section": "analysis_flags",
                "item": "model_refit_performed",
                "value": MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT,
                "note": "Models are refit for each candidate feature set and imbalance condition.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "feature_selection_rerun",
                "value": FEATURE_SELECTION_RERUN_IN_THIS_SCRIPT,
                "note": "Feature sets are fixed for sensitivity analysis; RFECV is not rerun.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "imputation_rerun",
                "value": IMPUTATION_RERUN_IN_THIS_SCRIPT,
                "note": "Submitted imputed train/hold-out and fold files are reused.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "under_sampling_rerun",
                "value": UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT,
                "note": "Random under-sampling is rerun inside each relevant training set.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "ridge_logistic_c_selection_rerun",
                "value": RIDGE_LOGISTIC_C_SELECTION_RERUN_IN_THIS_SCRIPT,
                "note": "Ridge logistic C is selected inside each training set.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "classical_model_hyperparameter_tuning_rerun",
                "value": CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT,
                "note": "LR/RF/XGB/LGB grid search is run for each feature set and resampling condition.",
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

    return pd.concat(
        [feature_policy, counts, logistic_variants, checkpoint, analysis_flags],
        ignore_index=True,
        sort=False,
    )


# %%
# Submitted fold files
def write_private_submitted_fold_files_for_each_feature_set(
    submitted_folds: list[dict[str, object]],
    feature_sets_by_name: dict[str, list[str]],
) -> pd.DataFrame:
    private_dir = prepare_private_intermediate_dir(STEP_DIR)
    rows = []
    for feature_set, columns in feature_sets_by_name.items():
        for fold in submitted_folds:
            fold_index = int(fold["fold"])
            train = fold["train"].copy()
            validation = fold["validation"].copy()
            validate_features_exist(train, columns, f"{feature_set} submitted fold {fold_index} train")
            validate_features_exist(validation, columns, f"{feature_set} submitted fold {fold_index} validation")
            selected_columns = [ID_COLUMN, TARGET_COLUMN, *columns]
            train_private = train[selected_columns].copy()
            validation_private = validation[selected_columns].copy()
            train_path = private_dir / f"{feature_set}_submitted_fold_{fold_index}_train_data_after_imputation.csv"
            validation_path = private_dir / f"{feature_set}_submitted_fold_{fold_index}_validation_data_after_imputation.csv"
            train_private.to_csv(train_path, index=False, encoding="utf-8-sig")
            validation_private.to_csv(validation_path, index=False, encoding="utf-8-sig")
            rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold_index,
                    "train_file": train_path.name,
                    "validation_file": validation_path.name,
                    "submitted_train_source_file": fold["train_source_file"],
                    "submitted_validation_source_file": fold["validation_source_file"],
                    "train_n": len(train_private),
                    "train_events": int(binary_target(train_private).sum()),
                    "validation_n": len(validation_private),
                    "validation_events": int(binary_target(validation_private).sum()),
                    "feature_count": len(columns),
                }
            )
    return pd.DataFrame(rows)


# %%
# Model fitting utilities
def make_step05_base_models() -> dict[str, object]:
    return {
        "LR": LogisticRegression(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE),
        "rf": RandomForestClassifier(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE),
        "xgb": XGBClassifier(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE, verbosity=0),
        "lgb": LGBMClassifier(random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE, verbosity=-1),
    }


def run_step05_grid_search_for_feature_set(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    feature_set_name: str,
    analysis_name: str,
    use_random_undersampling: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    x_all = numeric_features(train, feature_columns, f"{feature_set_name} Step05 tuning train")
    y_all = binary_target(train)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.30,
        random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE,
        stratify=y_all,
    )

    if use_random_undersampling:
        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_fit, y_fit = sampler.fit_resample(x_train, y_train)
        resampling = f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE})"
        resampling_condition = "random_undersampling"
    else:
        x_fit, y_fit = x_train, y_train
        resampling = "none"
        resampling_condition = "no_undersampling"

    rows = []
    best_parameters: dict[str, dict[str, object]] = {}
    models = make_step05_base_models()
    for model_key in CLASSICAL_MODELS:
        grid_search = GridSearchCV(
            models[model_key],
            STEP05_PARAM_GRIDS[model_key],
            cv=5,
            scoring="roc_auc",
            n_jobs=1,
            verbose=0,
        )
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            warnings.simplefilter("ignore")
            grid_search.fit(x_fit, y_fit)
        best_parameters[model_key] = dict(grid_search.best_params_)
        rows.append(
            {
                "analysis": analysis_name,
                "feature_set": feature_set_name,
                "resampling_condition": resampling_condition,
                "model": model_key,
                "model_label": GRID_SEARCH_MODEL_LABELS[model_key],
                "grid_source": "submitted_05_Hyperparameter_Tuning.ipynb",
                "grid_search_cv": "GridSearchCV(cv=5, scoring='roc_auc')",
                "resampling_inside_internal_training_split": resampling,
                "base_model_random_state": SUBMITTED_GRID_SEARCH_RANDOM_STATE,
                "internal_train_n_before_resampling": len(y_train),
                "internal_train_events_before_resampling": int(np.sum(y_train)),
                "internal_train_n_used_for_grid_search": len(y_fit),
                "internal_train_events_used_for_grid_search": int(np.sum(y_fit)),
                "internal_test_n": len(y_test),
                "internal_test_events": int(np.sum(y_test)),
                "best_parameters": str(best_parameters[model_key]),
                "best_cv_auroc": float(grid_search.best_score_),
                "internal_test_auroc": float(grid_search.score(x_test, y_test)),
            }
        )
    return pd.DataFrame(rows), best_parameters


def apply_resampling_if_needed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    use_random_undersampling: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if use_random_undersampling:
        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_fit, y_fit = sampler.fit_resample(x_train, y_train)
        return x_fit, y_fit, f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE})"
    return x_train, y_train, "none"


def make_classical_model_from_step05_best(model_key: str, best_parameters: dict[str, object]) -> object:
    params = dict(best_parameters)
    if model_key == "LR":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)
    if model_key == "rf":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        return RandomForestClassifier(**params)
    if model_key == "xgb":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        params.setdefault("verbosity", 0)
        return XGBClassifier(**params)
    if model_key == "lgb":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        params.setdefault("verbosity", -1)
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown classical model: {model_key}")


def make_ridge_logistic_model_for_training(y_train: np.ndarray) -> object:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    folds = max(2, min(5, positives, negatives))
    inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    classifier = LogisticRegressionCV(
        Cs=RIDGE_LOGISTIC_C_GRID,
        cv=inner_cv,
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        class_weight=None,
        random_state=RANDOM_STATE,
        max_iter=5000,
        n_jobs=RIDGE_LOGISTIC_N_JOBS,
        refit=True,
    )
    return make_pipeline(StandardScaler(), classifier)


def selected_ridge_c_from_model(model: object) -> float:
    return float(np.asarray(model.named_steps["logisticregressioncv"].C_).ravel()[0])


def fit_score_one_model(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    best_parameters: dict[str, dict[str, object]],
) -> tuple[np.ndarray, float]:
    if model_key == "tabpfn":
        model = make_tabpfn_model()
    elif model_key == "ridge_logistic":
        model = make_ridge_logistic_model_for_training(y_train)
    else:
        model = make_classical_model_from_step05_best(model_key, best_parameters[model_key])
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        warnings.simplefilter("ignore")
        fit_model(model, x_train, y_train)
    score = positive_probability(model, x_validation)
    selected_c = selected_ridge_c_from_model(model) if model_key == "ridge_logistic" else np.nan
    return score, selected_c


def evaluate_submitted_folds(
    submitted_folds: list[dict[str, object]],
    feature_columns: list[str],
    *,
    feature_set_name: str,
    resampling_condition: str,
    use_random_undersampling: bool,
    best_parameters: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    status_rows = []
    for fold in submitted_folds:
        fold_index = int(fold["fold"])
        train = fold["train"]
        validation = fold["validation"]
        x_train = numeric_features(train, feature_columns, f"{feature_set_name} fold {fold_index} train")
        y_train = binary_target(train)
        x_validation = numeric_features(validation, feature_columns, f"{feature_set_name} fold {fold_index} validation")
        y_validation = binary_target(validation)
        x_fit, y_fit, resampling = apply_resampling_if_needed(x_train, y_train, use_random_undersampling)
        for model_key in MODEL_SEQUENCE:
            status = {
                "analysis": "submitted_fold_cv",
                "feature_set": feature_set_name,
                "resampling_condition": resampling_condition,
                "fold": fold_index,
                "model": model_key,
                "model_label": MODEL_LABELS_FOR_THIS_STEP[model_key],
                "status": "started",
                "resampling": resampling,
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_used": len(y_fit),
                "train_events_used": int(np.sum(y_fit)),
                "validation_n": len(y_validation),
                "validation_events": int(y_validation.sum()),
                "feature_count": len(feature_columns),
                "fit_seconds": np.nan,
                "tabpfn_device": TABPFN_DEVICE if model_key == "tabpfn" else "",
                "tabpfn_ensembles": TABPFN_ENSEMBLES if model_key == "tabpfn" else "",
                "ridge_logistic_selected_c": np.nan,
                "error": "",
            }
            try:
                start = time.perf_counter()
                score, selected_c = fit_score_one_model(model_key, x_fit, y_fit, x_validation, best_parameters)
                status["fit_seconds"] = time.perf_counter() - start
                status["ridge_logistic_selected_c"] = selected_c
                metric = wide_metric_summary(
                    y_validation,
                    score,
                    dataset=f"submitted_validation_fold_{fold_index}",
                    model=model_key,
                    model_label=MODEL_LABELS_FOR_THIS_STEP[model_key],
                    iterations=BOOTSTRAP_ITERATIONS,
                    seed=RANDOM_STATE,
                )
                metric_rows.append(
                    {
                        "analysis": "submitted_fold_cv",
                        "feature_set": feature_set_name,
                        "resampling_condition": resampling_condition,
                        "resampling": resampling,
                        "fold": fold_index,
                        "feature_count": len(feature_columns),
                        **metric,
                    }
                )
                status["status"] = "completed"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = repr(exc)
            status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows)


def evaluate_holdout(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    *,
    feature_set_name: str,
    resampling_condition: str,
    use_random_undersampling: bool,
    best_parameters: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = numeric_features(train, feature_columns, f"{feature_set_name} holdout train")
    y_train = binary_target(train)
    x_holdout = numeric_features(holdout, feature_columns, f"{feature_set_name} holdout")
    y_holdout = binary_target(holdout)
    x_fit, y_fit, resampling = apply_resampling_if_needed(x_train, y_train, use_random_undersampling)
    metric_rows = []
    status_rows = []
    for model_key in MODEL_SEQUENCE:
        status = {
            "analysis": "submitted_holdout",
            "feature_set": feature_set_name,
            "resampling_condition": resampling_condition,
            "model": model_key,
            "model_label": MODEL_LABELS_FOR_THIS_STEP[model_key],
            "status": "started",
            "resampling": resampling,
            "train_n_before_resampling": len(y_train),
            "train_events_before_resampling": int(y_train.sum()),
            "train_n_used": len(y_fit),
            "train_events_used": int(np.sum(y_fit)),
            "holdout_n": len(y_holdout),
            "holdout_events": int(y_holdout.sum()),
            "feature_count": len(feature_columns),
            "fit_seconds": np.nan,
            "tabpfn_device": TABPFN_DEVICE if model_key == "tabpfn" else "",
            "tabpfn_ensembles": TABPFN_ENSEMBLES if model_key == "tabpfn" else "",
            "ridge_logistic_selected_c": np.nan,
            "error": "",
        }
        try:
            start = time.perf_counter()
            score, selected_c = fit_score_one_model(model_key, x_fit, y_fit, x_holdout, best_parameters)
            status["fit_seconds"] = time.perf_counter() - start
            status["ridge_logistic_selected_c"] = selected_c
            metric = wide_metric_summary(
                y_holdout,
                score,
                dataset="submitted_ppv_holdout",
                model=model_key,
                model_label=MODEL_LABELS_FOR_THIS_STEP[model_key],
                iterations=BOOTSTRAP_ITERATIONS,
                seed=RANDOM_STATE,
            )
            metric_rows.append(
                {
                    "analysis": "submitted_holdout",
                    "feature_set": feature_set_name,
                    "resampling_condition": resampling_condition,
                    "resampling": resampling,
                    "feature_count": len(feature_columns),
                    **metric,
                }
            )
            status["status"] = "completed"
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = repr(exc)
        status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows)


def add_checkpoint_status_to_run_status(run_status: pd.DataFrame, checkpoint_status: dict[str, object]) -> pd.DataFrame:
    checkpoint_row = {
        "analysis": "tabpfn_checkpoint_status",
        "model": "tabpfn",
        "model_label": "TabPFN",
        "status": "completed" if checkpoint_status.get("checkpoint_exists") else "failed",
        "tabpfn_device": TABPFN_DEVICE,
        "tabpfn_ensembles": TABPFN_ENSEMBLES,
        "error": "" if checkpoint_status.get("checkpoint_exists") else "TabPFN checkpoint not found",
        **checkpoint_status,
    }
    return pd.concat([run_status, pd.DataFrame([checkpoint_row])], ignore_index=True, sort=False)


# %%
# Run the complete analysis
def run_analysis() -> None:
    print(f"[{STEP_DIR.name}] Start Step 47: feature-selection sensitivity.", flush=True)
    prepare_output_dir(STEP_DIR)
    checkpoint_status = require_tabpfn_checkpoint_available()

    # Reader map:
    # 1. Load submitted Step03 train/hold-out tables and submitted fold files.
    # 2. Define the three reviewer feature sets: all46, RFECV36, compact26.
    # 3. For each feature set and resampling condition, run the same submitted-style
    #    sequence: Step05 tuning, CV evaluation, fixed hold-out evaluation.
    # 4. Save policy, tuning, metrics, and run-status tables.
    train, holdout = load_submitted_train_holdout_after_imputation()
    submitted_folds = load_submitted_after_imputation_fold_files(n_splits=CV_FOLDS)
    feature_sets_by_name = feature_sets()

    tuning_tables = []
    cv_metric_tables = []
    holdout_metric_tables = []
    status_tables = []
    policy_tables = []

    for feature_set_name, columns in feature_sets_by_name.items():
        policy_tables.append(feature_policy_table(train, columns, feature_set_name))
        for resampling_condition, use_random_undersampling in RESAMPLING_CONDITIONS.items():
            # Submitted-style block for one feature set and one resampling condition.
            tuning, best_parameters = run_step05_grid_search_for_feature_set(
                train,
                columns,
                feature_set_name=feature_set_name,
                analysis_name=f"feature_selection_sensitivity__{resampling_condition}",
                use_random_undersampling=use_random_undersampling,
            )
            cv_metrics, cv_status = evaluate_submitted_folds(
                submitted_folds,
                columns,
                feature_set_name=feature_set_name,
                resampling_condition=resampling_condition,
                use_random_undersampling=use_random_undersampling,
                best_parameters=best_parameters,
            )
            holdout_metrics, holdout_status = evaluate_holdout(
                train,
                holdout,
                columns,
                feature_set_name=feature_set_name,
                resampling_condition=resampling_condition,
                use_random_undersampling=use_random_undersampling,
                best_parameters=best_parameters,
            )
            tuning_tables.append(tuning)
            cv_metric_tables.append(cv_metrics)
            holdout_metric_tables.append(holdout_metrics)
            status_tables.extend([cv_status, holdout_status])

    _private_fold_index = write_private_submitted_fold_files_for_each_feature_set(submitted_folds, feature_sets_by_name)

    feature_policy = pd.concat(policy_tables, ignore_index=True)
    policy = feature_selection_policy_table(feature_sets_by_name, feature_policy, checkpoint_status)

    cv_metrics = pd.concat(cv_metric_tables, ignore_index=True)
    holdout_metrics = pd.concat(holdout_metric_tables, ignore_index=True)
    cv_metrics.insert(0, "evaluation_stage", "cross_validation")
    holdout_metrics.insert(0, "evaluation_stage", "holdout")
    model_metrics = pd.concat([cv_metrics, holdout_metrics], ignore_index=True, sort=False)
    all_status = pd.concat(status_tables, ignore_index=True)

    write_csv(policy, POLICY_CSV)
    write_csv(pd.concat(tuning_tables, ignore_index=True), TUNING_RESULTS_CSV)
    write_csv(model_metrics, MODEL_METRICS_CSV)
    write_csv(all_status, MODEL_RUN_STATUS_CSV)
    print(f"[{STEP_DIR.name}] Done.", flush=True)


if __name__ == "__main__":
    run_analysis()
