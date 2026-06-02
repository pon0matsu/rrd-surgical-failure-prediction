"""Step 48: class-imbalance strategy sensitivity.

This script answers Reviewer 3 comment 4.  It rebuilds the submitted RFECV36
model family under class-imbalance strategies:

1. no under-sampling;
2. RandomUnderSampler(random_state=42);
3. class weighting for classical models where supported.

The submitted primary result remains the random-undersampling TabPFN workflow.
The no-undersampling and class-weighting conditions are sensitivity analyses.

The public outputs are limited to policy, tuning, combined metrics, and run
status. Private fold files are still written as intermediate files, but their
file-index CSV is not written to `local_outputs`.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
    ID_COLUMN,
    OLD_MODEL_RANDOM_STATE,
    OLD_UNDERSAMPLING_RANDOM_STATE,
    TABPFN_DEVICE,
    TABPFN_ENSEMBLES,
    OLD_MODEL_ORDER,
    OLD_MODEL_LABELS,
    SUBMITTED_GRID_SEARCH_RANDOM_STATE,
    TARGET_COLUMN,
    binary_target,
    feature_policy_table,
    fit_model,
    load_submitted_after_imputation_fold_files,
    load_submitted_train_holdout_after_imputation,
    make_tabpfn_model,
    numeric_features,
    old_hyperparameter_table,
    positive_probability,
    prepare_private_intermediate_dir,
    read_rfecv36_features,
    require_tabpfn_checkpoint_available,
    validate_features_exist,
)


# %%
# Paths, feature set, model settings, and analysis flags
STEP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STEP_DIR / "local_outputs"

POLICY_CSV = OUTPUT_DIR / "class_imbalance_policy.csv"
TUNING_RESULTS_CSV = OUTPUT_DIR / "class_imbalance_hyperparameter_tuning.csv"
MODEL_METRICS_CSV = OUTPUT_DIR / "class_imbalance_metrics.csv"
MODEL_RUN_STATUS_CSV = OUTPUT_DIR / "class_imbalance_model_run_status.csv"

PRIMARY_FEATURE_SET = "submitted_rfecv36"
POSITIVE_CLASS_LABEL = 1
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = BOOTSTRAP_SEED
CV_FOLDS = 5
CLASSICAL_MODELS = ["LR", "rf", "xgb", "lgb"]
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
SUBMITTED_ORIGINAL_FINAL_HYPERPARAMETERS = {
    "tabpfn": {
        "device": TABPFN_DEVICE,
        "N_ensemble_configurations": TABPFN_ENSEMBLES,
    },
    "xgb": {
        "learning_rate": 0.01,
        "max_depth": 3,
        "n_estimators": 200,
        "random_state": OLD_MODEL_RANDOM_STATE,
    },
    "LR": {
        "C": 0.001,
        "penalty": "none",
        "solver": "saga",
        "random_state": OLD_MODEL_RANDOM_STATE,
    },
    "rf": {
        "max_depth": 5,
        "n_estimators": 200,
        "random_state": OLD_MODEL_RANDOM_STATE,
    },
    "lgb": {
        "learning_rate": 0.01,
        "max_depth": 5,
        "n_estimators": 50,
        "random_state": OLD_MODEL_RANDOM_STATE,
    },
}
IMBALANCE_STRATEGIES = {
    "no_undersampling": {
        "use_random_undersampling": False,
        "use_class_weight": False,
        "fit_tabpfn": True,
    },
    "random_undersampling": {
        "use_random_undersampling": True,
        "use_class_weight": False,
        "fit_tabpfn": True,
    },
    "class_weighting": {
        "use_random_undersampling": False,
        "use_class_weight": True,
        "fit_tabpfn": False,
    },
}

MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT = True
FEATURE_SELECTION_RERUN_IN_THIS_SCRIPT = False
IMPUTATION_RERUN_IN_THIS_SCRIPT = False
UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT = True
CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = True
TABPFN_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT = False
CALIBRATION_RERUN_IN_THIS_SCRIPT = False
SUBMITTED_STYLE_NO_UNDERSAMPLING_CV_REBUILT_IN_THIS_SCRIPT = True
SUBMITTED_STYLE_NO_UNDERSAMPLING_FINAL_REBUILT_IN_THIS_SCRIPT = True


# %%
# TabPFN imbalance sensitivity
def fit_tabpfn_with_imbalance_strategy(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    rfecv_features: list[str],
    *,
    resampling_condition: str,
    use_random_undersampling: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = numeric_features(train, rfecv_features, "RFECV36 train")
    y_train = binary_target(train)
    x_holdout = numeric_features(holdout, rfecv_features, "RFECV36 holdout")
    y_holdout = binary_target(holdout)

    if use_random_undersampling:
        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_fit, y_fit = sampler.fit_resample(x_train, y_train)
        resampling = f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE})"
    else:
        x_fit, y_fit = x_train, y_train
        resampling = "none"

    model = make_tabpfn_model()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_model(model, x_fit, y_fit)
    fit_seconds = time.perf_counter() - start
    score = positive_probability(model, x_holdout)

    metrics = pd.DataFrame(
        [
            {
                "analysis": "imbalance_strategy_sensitivity",
                "feature_set": PRIMARY_FEATURE_SET,
                "resampling_condition": resampling_condition,
                "resampling": resampling,
                "feature_count": len(rfecv_features),
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_used": len(y_fit),
                "train_events_used": int(np.sum(y_fit)),
                **wide_metric_summary(
                    y_holdout,
                    score,
                    dataset="submitted_ppv_holdout",
                    model="tabpfn",
                    model_label="TabPFN",
                    iterations=BOOTSTRAP_ITERATIONS,
                    seed=RANDOM_STATE,
                ),
            }
        ]
    )
    status = pd.DataFrame(
        [
            {
                "analysis": "imbalance_strategy_sensitivity",
                "feature_set": PRIMARY_FEATURE_SET,
                "resampling_condition": resampling_condition,
                "model": "tabpfn",
                "model_label": "TabPFN",
                "status": "completed",
                "tabpfn_device": TABPFN_DEVICE,
                "tabpfn_ensembles": TABPFN_ENSEMBLES,
                "resampling": resampling,
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_used": len(y_fit),
                "train_events_used": int(np.sum(y_fit)),
                "holdout_n": len(y_holdout),
                "holdout_events": int(y_holdout.sum()),
                "feature_count": len(rfecv_features),
                "fit_seconds": fit_seconds,
                "error": "",
            }
        ]
    )
    return metrics, status


def tabpfn_class_weight_applicability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": PRIMARY_FEATURE_SET,
                "model": "tabpfn",
                "imbalance_strategy": "class_weighting",
                "class_weight_strategy_tested": False,
                "reason": (
                    "The submitted TabPFN API used for this project does not expose "
                    "a class_weight parameter. The class-imbalance sensitivity is "
                    "therefore evaluated for TabPFN as no under-sampling versus "
                    "random under-sampling. Class weighting is evaluated only for "
                    "classical comparator models where supported."
                ),
            }
        ]
    )


def classical_class_weight_strategy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": PRIMARY_FEATURE_SET,
                "model": "LR",
                "model_label": OLD_MODEL_LABELS["LR"],
                "imbalance_strategy": "class_weighting",
                "implemented": True,
                "implementation": "LogisticRegression(class_weight='balanced')",
                "used_with_random_undersampling": False,
            },
            {
                "feature_set": PRIMARY_FEATURE_SET,
                "model": "rf",
                "model_label": OLD_MODEL_LABELS["rf"],
                "imbalance_strategy": "class_weighting",
                "implemented": True,
                "implementation": "RandomForestClassifier(class_weight='balanced')",
                "used_with_random_undersampling": False,
            },
            {
                "feature_set": PRIMARY_FEATURE_SET,
                "model": "xgb",
                "model_label": OLD_MODEL_LABELS["xgb"],
                "imbalance_strategy": "class_weighting",
                "implemented": True,
                "implementation": "XGBClassifier(scale_pos_weight=negative_n/positive_n within training data)",
                "used_with_random_undersampling": False,
            },
            {
                "feature_set": PRIMARY_FEATURE_SET,
                "model": "lgb",
                "model_label": OLD_MODEL_LABELS["lgb"],
                "imbalance_strategy": "class_weighting",
                "implemented": True,
                "implementation": "LGBMClassifier(class_weight='balanced')",
                "used_with_random_undersampling": False,
            },
        ]
    )


# %%
# Policy and output table assembly
def policy_section(frame: pd.DataFrame, section_name: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "policy_section", section_name)
    return output


def imbalance_strategy_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "imbalance_strategy": strategy_name,
                "use_random_undersampling": bool(strategy["use_random_undersampling"]),
                "use_class_weight": bool(strategy["use_class_weight"]),
                "fit_tabpfn": bool(strategy["fit_tabpfn"]),
                "positive_class_label": POSITIVE_CLASS_LABEL,
                "default_threshold": DEFAULT_THRESHOLD,
            }
            for strategy_name, strategy in IMBALANCE_STRATEGIES.items()
        ]
    )


def class_imbalance_policy_table(train: pd.DataFrame, rfecv_features: list[str]) -> pd.DataFrame:
    feature_policy = feature_policy_table(train, rfecv_features, PRIMARY_FEATURE_SET)
    analysis_flags = pd.DataFrame(
        [
            {
                "policy_section": "analysis_flags",
                "item": "model_refit_performed",
                "value": MODEL_REFIT_PERFORMED_IN_THIS_SCRIPT,
                "note": "Models are refit for each imbalance strategy.",
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
                "note": "Submitted imputed train/hold-out and fold files are reused.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "under_sampling_rerun",
                "value": UNDER_SAMPLING_RERUN_IN_THIS_SCRIPT,
                "note": "Random under-sampling is rerun for the random-under-sampling strategy.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "submitted_style_no_undersampling_cv_rebuilt",
                "value": SUBMITTED_STYLE_NO_UNDERSAMPLING_CV_REBUILT_IN_THIS_SCRIPT,
                "note": "The submitted no-under-sampling CV workflow is rebuilt as a reference.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "submitted_style_no_undersampling_final_rebuilt",
                "value": SUBMITTED_STYLE_NO_UNDERSAMPLING_FINAL_REBUILT_IN_THIS_SCRIPT,
                "note": "The submitted no-under-sampling final workflow is rebuilt as a reference.",
            },
            {
                "policy_section": "analysis_flags",
                "item": "classical_model_hyperparameter_tuning_rerun",
                "value": CLASSICAL_MODEL_HYPERPARAMETER_TUNING_RERUN_IN_THIS_SCRIPT,
                "note": "LR/RF/XGB/LGB Step05-style grid search is rerun for each imbalance strategy.",
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
        [
            policy_section(feature_policy, "submitted_rfecv36_feature_policy"),
            policy_section(imbalance_strategy_definition_table(), "imbalance_strategy_definition"),
            policy_section(tabpfn_class_weight_applicability(), "tabpfn_class_weight_applicability"),
            policy_section(classical_class_weight_strategy_table(), "classical_class_weight_strategy"),
            policy_section(submitted_original_hyperparameter_table(), "submitted_original_hyperparameters"),
            analysis_flags,
        ],
        ignore_index=True,
        sort=False,
    )


def add_output_context(
    frame: pd.DataFrame,
    *,
    evaluation_stage: str,
    imbalance_strategy: str,
) -> pd.DataFrame:
    output = frame.copy()
    if "evaluation_stage" not in output.columns:
        output.insert(1, "evaluation_stage", evaluation_stage)
    else:
        output["evaluation_stage"] = evaluation_stage
    if "imbalance_strategy" not in output.columns:
        output.insert(2, "imbalance_strategy", imbalance_strategy)
    else:
        output["imbalance_strategy"] = imbalance_strategy
    return output


def append_to_existing_output_if_present(new_frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return new_frame
    existing = pd.read_csv(output_path)
    return pd.concat([existing, new_frame], ignore_index=True, sort=False)


# %%
# Submitted fold files
def write_private_submitted_rfecv36_fold_files(
    submitted_folds: list[dict[str, object]],
    rfecv_features: list[str],
) -> pd.DataFrame:
    private_dir = prepare_private_intermediate_dir(STEP_DIR)
    rows = []
    selected_columns = [ID_COLUMN, TARGET_COLUMN, *rfecv_features]
    for fold in submitted_folds:
        fold_index = int(fold["fold"])
        train = fold["train"].copy()
        validation = fold["validation"].copy()
        validate_features_exist(train, rfecv_features, f"submitted fold {fold_index} train")
        validate_features_exist(validation, rfecv_features, f"submitted fold {fold_index} validation")
        train_private = train[selected_columns].copy()
        validation_private = validation[selected_columns].copy()
        train_path = private_dir / f"rfecv36_submitted_fold_{fold_index}_train_data_after_imputation.csv"
        validation_path = private_dir / f"rfecv36_submitted_fold_{fold_index}_validation_data_after_imputation.csv"
        train_private.to_csv(train_path, index=False, encoding="utf-8-sig")
        validation_private.to_csv(validation_path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "feature_set": PRIMARY_FEATURE_SET,
                "fold": fold_index,
                "train_file": train_path.name,
                "validation_file": validation_path.name,
                "submitted_train_source_file": fold["train_source_file"],
                "submitted_validation_source_file": fold["validation_source_file"],
                "train_n": len(train_private),
                "train_events": int(binary_target(train_private).sum()),
                "validation_n": len(validation_private),
                "validation_events": int(binary_target(validation_private).sum()),
                "feature_count": len(rfecv_features),
            }
        )
    return pd.DataFrame(rows)


# %%
# Step05-style classical-model tuning for imbalance strategies
def positive_class_weight_ratio(y_train: np.ndarray) -> float:
    positive_count = int(np.sum(y_train == POSITIVE_CLASS_LABEL))
    negative_count = int(np.sum(y_train != POSITIVE_CLASS_LABEL))
    if positive_count == 0:
        return 1.0
    return float(negative_count / positive_count)


def make_step05_grid_search_base_model(
    model_key: str,
    *,
    use_class_weight: bool,
    y_train: np.ndarray,
) -> object:
    if model_key == "LR":
        return LogisticRegression(
            random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE,
            max_iter=5000,
            class_weight="balanced" if use_class_weight else None,
        )
    if model_key == "rf":
        return RandomForestClassifier(
            random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE,
            class_weight="balanced" if use_class_weight else None,
        )
    if model_key == "xgb":
        return XGBClassifier(
            random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE,
            verbosity=0,
            scale_pos_weight=positive_class_weight_ratio(y_train) if use_class_weight else 1.0,
        )
    if model_key == "lgb":
        return LGBMClassifier(
            random_state=SUBMITTED_GRID_SEARCH_RANDOM_STATE,
            verbosity=-1,
            class_weight="balanced" if use_class_weight else None,
        )
    raise ValueError(f"Step05 grid search is not defined for model: {model_key}")


def make_classical_model_from_step05_best_parameters(
    model_key: str,
    best_parameters: dict[str, object],
    *,
    use_class_weight: bool,
    y_train: np.ndarray,
) -> object:
    params = dict(best_parameters)
    if model_key == "LR":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        params.setdefault("max_iter", 5000)
        if use_class_weight:
            params["class_weight"] = "balanced"
        return LogisticRegression(**params)
    if model_key == "rf":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        if use_class_weight:
            params["class_weight"] = "balanced"
        return RandomForestClassifier(**params)
    if model_key == "xgb":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        params.setdefault("verbosity", 0)
        if use_class_weight:
            params["scale_pos_weight"] = positive_class_weight_ratio(y_train)
        return XGBClassifier(**params)
    if model_key == "lgb":
        params.setdefault("random_state", OLD_MODEL_RANDOM_STATE)
        params.setdefault("verbosity", -1)
        if use_class_weight:
            params["class_weight"] = "balanced"
        return LGBMClassifier(**params)
    raise ValueError(f"Classical model is not defined for: {model_key}")


def run_step05_grid_search_on_training_set(
    train: pd.DataFrame,
    rfecv_features: list[str],
    *,
    analysis_name: str,
    use_random_undersampling: bool,
    use_class_weight: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    if use_random_undersampling and use_class_weight:
        raise ValueError("Use either random under-sampling or class weighting, not both in the same strategy.")

    x_train = numeric_features(train, rfecv_features, f"{analysis_name} training set")
    y_train = binary_target(train)
    if use_random_undersampling:
        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_grid, y_grid = sampler.fit_resample(x_train, y_train)
        resampling = f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE})"
    else:
        x_grid, y_grid = x_train, y_train
        resampling = "none"
    class_weight_strategy = "balanced_or_scale_pos_weight" if use_class_weight else "none"

    rows = []
    best_parameters_by_model: dict[str, dict[str, object]] = {}
    for model_key in CLASSICAL_MODELS:
        base_model = make_step05_grid_search_base_model(
            model_key,
            use_class_weight=use_class_weight,
            y_train=y_grid,
        )
        grid_search = GridSearchCV(
            base_model,
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
            grid_search.fit(x_grid, y_grid)
        best_parameters_by_model[model_key] = dict(grid_search.best_params_)
        rows.append(
            {
                "analysis": analysis_name,
                "feature_set": PRIMARY_FEATURE_SET,
                "model": model_key,
                "model_label": GRID_SEARCH_MODEL_LABELS[model_key],
                "grid_source": "submitted_05_Hyperparameter_Tuning.ipynb",
                "grid_search_cv": "GridSearchCV(cv=5, scoring='roc_auc')",
                "holdout_used_for_tuning": False,
                "resampling_before_grid_search": resampling,
                "class_weight_strategy": class_weight_strategy,
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_used_for_grid_search": len(y_grid),
                "train_events_used_for_grid_search": int(np.sum(y_grid)),
                "best_parameters": str(best_parameters_by_model[model_key]),
                "best_cv_auroc": float(grid_search.best_score_),
            }
        )
    return pd.DataFrame(rows), best_parameters_by_model


def fit_classical_models_for_imbalance_strategy(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    rfecv_features: list[str],
    *,
    analysis_name: str,
    use_random_undersampling: bool,
    use_class_weight: bool,
) -> dict[str, pd.DataFrame]:
    tuning, best_parameters = run_step05_grid_search_on_training_set(
        train,
        rfecv_features,
        analysis_name=analysis_name,
        use_random_undersampling=use_random_undersampling,
        use_class_weight=use_class_weight,
    )
    x_train = numeric_features(train, rfecv_features, f"{analysis_name} final train")
    y_train = binary_target(train)
    x_holdout = numeric_features(holdout, rfecv_features, f"{analysis_name} fixed holdout")
    y_holdout = binary_target(holdout)

    if use_random_undersampling:
        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_fit, y_fit = sampler.fit_resample(x_train, y_train)
        resampling = f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE})"
    else:
        x_fit, y_fit = x_train, y_train
        resampling = "none"
    class_weight_strategy = "balanced_or_scale_pos_weight" if use_class_weight else "none"

    metric_rows = []
    status_rows = []
    for model_key in CLASSICAL_MODELS:
        status = {
            "analysis": analysis_name,
            "feature_set": PRIMARY_FEATURE_SET,
            "model": model_key,
            "model_label": OLD_MODEL_LABELS[model_key],
            "status": "started",
            "tuning_method": "training_set_5fold_cv_with_submitted_05_grid",
            "holdout_used_for_tuning": False,
            "resampling": resampling,
            "class_weight_strategy": class_weight_strategy,
            "train_n_before_resampling": len(y_train),
            "train_events_before_resampling": int(y_train.sum()),
            "train_n_used": len(y_fit),
            "train_events_used": int(np.sum(y_fit)),
            "holdout_n": len(y_holdout),
            "holdout_events": int(y_holdout.sum()),
            "feature_count": len(rfecv_features),
            "fit_seconds": np.nan,
            "error": "",
        }
        try:
            model = make_classical_model_from_step05_best_parameters(
                model_key,
                best_parameters[model_key],
                use_class_weight=use_class_weight,
                y_train=y_fit,
            )
            start = time.perf_counter()
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                warnings.simplefilter("ignore")
                fit_model(model, x_fit, y_fit)
            status["fit_seconds"] = time.perf_counter() - start
            score = positive_probability(model, x_holdout)
            metric = wide_metric_summary(
                y_holdout,
                score,
                dataset="submitted_ppv_holdout",
                model=model_key,
                model_label=OLD_MODEL_LABELS[model_key],
                iterations=BOOTSTRAP_ITERATIONS,
                seed=RANDOM_STATE,
            )
            metric_rows.append(
                {
                    "analysis": analysis_name,
                    "feature_set": PRIMARY_FEATURE_SET,
                    "resampling": resampling,
                    "class_weight_strategy": class_weight_strategy,
                    "feature_count": len(rfecv_features),
                    "train_n_before_resampling": len(y_train),
                    "train_events_before_resampling": int(y_train.sum()),
                    "train_n_used": len(y_fit),
                    "train_events_used": int(np.sum(y_fit)),
                    **metric,
                }
            )
            status["status"] = "completed"
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = repr(exc)
        status_rows.append(status)

    return {
        "tuning": tuning,
        "metrics": pd.DataFrame(metric_rows),
        "run_status": pd.DataFrame(status_rows),
    }


# %%
# Submitted-style no-under-sampling CV and final models
def make_submitted_original_model(model_key: str) -> object:
    params = dict(SUBMITTED_ORIGINAL_FINAL_HYPERPARAMETERS[model_key])
    if model_key == "tabpfn":
        return make_tabpfn_model()
    if model_key == "xgb":
        return XGBClassifier(**params)
    if model_key == "LR":
        return LogisticRegression(**params)
    if model_key == "rf":
        return RandomForestClassifier(**params)
    if model_key == "lgb":
        return LGBMClassifier(**params)
    raise ValueError(f"Submitted original model is not defined for: {model_key}")


def fit_score_submitted_original_model(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    model = make_submitted_original_model(model_key)
    start = time.perf_counter()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        warnings.simplefilter("ignore")
        fit_model(model, x_train, y_train)
    fit_seconds = time.perf_counter() - start
    score = positive_probability(model, x_eval)
    return score, fit_seconds, model.__class__.__name__


def rebuild_submitted_style_original_cv(
    submitted_folds: list[dict[str, object]],
    rfecv_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    status_rows = []
    for fold in submitted_folds:
        fold_index = int(fold["fold"])
        train = fold["train"]
        validation = fold["validation"]
        x_train = numeric_features(train, rfecv_features, f"original CV fold {fold_index} train")
        y_train = binary_target(train)
        x_validation = numeric_features(validation, rfecv_features, f"original CV fold {fold_index} validation")
        y_validation = binary_target(validation)
        for model_key in OLD_MODEL_ORDER:
            status = {
                "analysis": "submitted_style_no_undersampling_cv",
                "feature_set": PRIMARY_FEATURE_SET,
                "fold": fold_index,
                "model": model_key,
                "status": "started",
                "resampling": "none",
                "hyperparameter_source": "submitted_07_CV_ORIGINAL_and_09_Final_ORIGINAL",
                "train_n": len(y_train),
                "train_events": int(y_train.sum()),
                "validation_n": len(y_validation),
                "validation_events": int(y_validation.sum()),
                "feature_count": len(rfecv_features),
                "fit_seconds": np.nan,
                "error": "",
            }
            try:
                score, fit_seconds, model_class = fit_score_submitted_original_model(
                    model_key,
                    x_train,
                    y_train,
                    x_validation,
                )
                status["fit_seconds"] = fit_seconds
                status["model_class"] = model_class
                metric = wide_metric_summary(
                    y_validation,
                    score,
                    dataset=f"submitted_original_validation_fold_{fold_index}",
                    model=model_key,
                    model_label=OLD_MODEL_LABELS[model_key],
                    iterations=BOOTSTRAP_ITERATIONS,
                    seed=RANDOM_STATE,
                )
                metric_rows.append(
                    {
                        "analysis": "submitted_style_no_undersampling_cv",
                        "feature_set": PRIMARY_FEATURE_SET,
                        "fold": fold_index,
                        "resampling": "none",
                        "hyperparameter_source": "submitted_07_CV_ORIGINAL_and_09_Final_ORIGINAL",
                        "feature_count": len(rfecv_features),
                        **metric,
                    }
                )
                status["status"] = "completed"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = repr(exc)
            status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows)


def rebuild_submitted_style_original_final_holdout(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    rfecv_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = numeric_features(train, rfecv_features, "original final train")
    y_train = binary_target(train)
    x_holdout = numeric_features(holdout, rfecv_features, "original final holdout")
    y_holdout = binary_target(holdout)
    metric_rows = []
    status_rows = []
    for model_key in OLD_MODEL_ORDER:
        status = {
            "analysis": "submitted_style_no_undersampling_final_holdout",
            "feature_set": PRIMARY_FEATURE_SET,
            "model": model_key,
            "status": "started",
            "resampling": "none",
            "hyperparameter_source": "submitted_09_Final_ORIGINAL",
            "train_n": len(y_train),
            "train_events": int(y_train.sum()),
            "holdout_n": len(y_holdout),
            "holdout_events": int(y_holdout.sum()),
            "feature_count": len(rfecv_features),
            "fit_seconds": np.nan,
            "error": "",
        }
        try:
            score, fit_seconds, model_class = fit_score_submitted_original_model(
                model_key,
                x_train,
                y_train,
                x_holdout,
            )
            status["fit_seconds"] = fit_seconds
            status["model_class"] = model_class
            metric = wide_metric_summary(
                y_holdout,
                score,
                dataset="submitted_original_ppv_holdout",
                model=model_key,
                model_label=OLD_MODEL_LABELS[model_key],
                iterations=BOOTSTRAP_ITERATIONS,
                seed=RANDOM_STATE,
            )
            metric_rows.append(
                {
                    "analysis": "submitted_style_no_undersampling_final_holdout",
                    "feature_set": PRIMARY_FEATURE_SET,
                    "resampling": "none",
                    "hyperparameter_source": "submitted_09_Final_ORIGINAL",
                    "feature_count": len(rfecv_features),
                    **metric,
                }
            )
            status["status"] = "completed"
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = repr(exc)
        status_rows.append(status)
    return pd.DataFrame(metric_rows), pd.DataFrame(status_rows)


def submitted_original_hyperparameter_table() -> pd.DataFrame:
    table = old_hyperparameter_table()
    return table[table["condition"].eq("original")].copy()


# %%
# Run the complete analysis
def run_analysis() -> None:
    print(f"[{STEP_DIR.name}] Start Step 48: imbalance-strategy sensitivity.", flush=True)
    prepare_output_dir(STEP_DIR)
    require_tabpfn_checkpoint_available()

    # Reader map:
    # 1. Load submitted RFECV36 train/hold-out data and submitted fold files.
    # 2. Walk through the imbalance-strategy table in a fixed order.
    # 3. For each strategy, run Step05-style tuning and fixed hold-out evaluation
    #    for classical models; fit TabPFN where the submitted API supports it.
    # 4. Add the submitted-style no-under-sampling CV/final rebuild.
    # 5. Save policy, tuning, metrics, and run-status tables.
    train, holdout = load_submitted_train_holdout_after_imputation()
    submitted_folds = load_submitted_after_imputation_fold_files(CV_FOLDS)
    rfecv_features = read_rfecv36_features()
    _fold_index = write_private_submitted_rfecv36_fold_files(submitted_folds, rfecv_features)

    tuning_tables = []
    metric_tables = []
    status_tables = []
    for imbalance_strategy, strategy in IMBALANCE_STRATEGIES.items():
        # One strategy at a time keeps the comparison table easy to audit.
        use_random_undersampling = bool(strategy["use_random_undersampling"])
        use_class_weight = bool(strategy["use_class_weight"])
        if strategy["fit_tabpfn"]:
            tabpfn_metrics, tabpfn_status = fit_tabpfn_with_imbalance_strategy(
                train,
                holdout,
                rfecv_features,
                resampling_condition=imbalance_strategy,
                use_random_undersampling=use_random_undersampling,
            )
            metric_tables.append(
                add_output_context(
                    tabpfn_metrics,
                    evaluation_stage="holdout_sensitivity",
                    imbalance_strategy=imbalance_strategy,
                )
            )
            status_tables.append(
                add_output_context(
                    tabpfn_status,
                    evaluation_stage="holdout_sensitivity",
                    imbalance_strategy=imbalance_strategy,
                )
            )

        classical = fit_classical_models_for_imbalance_strategy(
            train,
            holdout,
            rfecv_features,
            analysis_name=f"imbalance_strategy_sensitivity__{imbalance_strategy}",
            use_random_undersampling=use_random_undersampling,
            use_class_weight=use_class_weight,
        )
        tuning = classical["tuning"].copy()
        tuning.insert(2, "imbalance_strategy", imbalance_strategy)
        metrics = add_output_context(
            classical["metrics"],
            evaluation_stage="holdout_sensitivity",
            imbalance_strategy=imbalance_strategy,
        )
        status = add_output_context(
            classical["run_status"],
            evaluation_stage="holdout_sensitivity",
            imbalance_strategy=imbalance_strategy,
        )
        tuning_tables.append(tuning)
        metric_tables.append(metrics)
        status_tables.append(status)

    # Submitted 07_CV_ORIGINAL and 09_Final_ORIGINAL style no-under-sampling branch.
    original_cv_metrics, original_cv_status = rebuild_submitted_style_original_cv(
        submitted_folds,
        rfecv_features,
    )
    original_final_metrics, original_final_status = rebuild_submitted_style_original_final_holdout(
        train,
        holdout,
        rfecv_features,
    )
    metric_tables.extend(
        [
            add_output_context(
                original_cv_metrics,
                evaluation_stage="submitted_style_cv_rebuild",
                imbalance_strategy="no_undersampling",
            ),
            add_output_context(
                original_final_metrics,
                evaluation_stage="submitted_style_holdout_rebuild",
                imbalance_strategy="no_undersampling",
            ),
        ]
    )
    status_tables.extend(
        [
            add_output_context(
                original_cv_status,
                evaluation_stage="submitted_style_cv_rebuild",
                imbalance_strategy="no_undersampling",
            ),
            add_output_context(
                original_final_status,
                evaluation_stage="submitted_style_holdout_rebuild",
                imbalance_strategy="no_undersampling",
            ),
        ]
    )

    write_csv(class_imbalance_policy_table(train, rfecv_features), POLICY_CSV)
    write_csv(pd.concat(tuning_tables, ignore_index=True), TUNING_RESULTS_CSV)
    write_csv(pd.concat(metric_tables, ignore_index=True), MODEL_METRICS_CSV)
    write_csv(pd.concat(status_tables, ignore_index=True), MODEL_RUN_STATUS_CSV)
    print(f"[{STEP_DIR.name}] Done.", flush=True)


if __name__ == "__main__":
    run_analysis()
