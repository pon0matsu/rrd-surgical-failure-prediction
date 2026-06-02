#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_common import write_csv  # noqa: E402
from review_modeling import (  # noqa: E402
    ID_COLUMN,
    OLD_CV_UNDERSAMPLING_HYPERPARAMETERS,
    OLD_MODEL_LABELS,
    OLD_MODEL_RANDOM_STATE,
    OLD_UNDERSAMPLING_RANDOM_STATE,
    RIDGE_LOGISTIC_C_GRID,
    RIDGE_LOGISTIC_N_JOBS,
    TARGET_COLUMN,
    TABPFN_DEVICE,
    TABPFN_ENSEMBLES,
    TABPFN_RANDOM_STATE,
    binary_target,
    fit_model,
    load_submitted_train_holdout_after_imputation,
    make_tabpfn_model,
    numeric_features,
    positive_probability,
    read_rfecv36_features,
    require_tabpfn_checkpoint_available,
)


STEP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STEP_DIR / "local_outputs"

REPEATED_CV_FOLD_LEVEL_CSV = OUTPUT_DIR / "r3_09_repeated_stratified5fold_cv_auc_by_fold.csv"
REPEATED_CV_SUMMARY_CSV = OUTPUT_DIR / "r3_09_repeated_stratified5fold_cv_auc_summary.csv"
REPEATED_CV_LAYOUT_CSV = OUTPUT_DIR / "r3_09_repeated_stratified5fold_cv_auc_manuscript_layout.csv"
REPEATED_CV_POLICY_CSV = OUTPUT_DIR / "r3_09_repeated_stratified5fold_cv_policy.csv"
COVERAGE_CHECK_CSV = OUTPUT_DIR / "r3_09_sample_size_model_complexity_coverage_check.csv"

N_SPLITS = 5
N_REPEATS = int(os.environ.get("R3_09_REPEATED_CV_REPEATS", "10"))
REPEATED_CV_RANDOM_STATE = 20260526
FEATURE_SET_NAME = "submitted_rfecv36"
RESAMPLING_CONDITION = "random_undersampling_inside_training_fold"

MODEL_ORDER = ["tabpfn", "rf", "xgb", "lgb", "LR", "ridge_logistic"]
DISPLAY_MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
    "LR": "Logistic Regression",
    "ridge_logistic": "Ridge Logistic Regression",
}


def make_ridge_logistic_model_for_training(y_train: np.ndarray) -> object:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    folds = max(2, min(5, positives, negatives))
    inner_cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=OLD_MODEL_RANDOM_STATE,
    )
    classifier = LogisticRegressionCV(
        Cs=RIDGE_LOGISTIC_C_GRID,
        cv=inner_cv,
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        class_weight=None,
        random_state=OLD_MODEL_RANDOM_STATE,
        max_iter=5000,
        n_jobs=RIDGE_LOGISTIC_N_JOBS,
        refit=True,
    )
    return make_pipeline(StandardScaler(), classifier)


def make_submitted_cv_undersampling_model(model_key: str, y_train: np.ndarray | None = None) -> object:
    if model_key == "tabpfn":
        return make_tabpfn_model()
    if model_key == "ridge_logistic":
        if y_train is None:
            raise ValueError("Ridge logistic regression requires y_train for inner CV setup.")
        return make_ridge_logistic_model_for_training(y_train)
    params = dict(OLD_CV_UNDERSAMPLING_HYPERPARAMETERS[model_key])
    if model_key == "LR":
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)
    if model_key == "rf":
        return RandomForestClassifier(**params)
    if model_key == "xgb":
        params.setdefault("verbosity", 0)
        return XGBClassifier(**params)
    if model_key == "lgb":
        params.setdefault("verbosity", -1)
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown model key: {model_key}")


def combined_submitted_imputed_ppv() -> pd.DataFrame:
    train, holdout = load_submitted_train_holdout_after_imputation()
    train = train.copy()
    holdout = holdout.copy()
    train["submitted_split"] = "train"
    holdout["submitted_split"] = "holdout"
    combined = pd.concat([train, holdout], ignore_index=True, sort=False)
    if ID_COLUMN in combined.columns and combined[ID_COLUMN].duplicated().any():
        duplicated = int(combined[ID_COLUMN].duplicated().sum())
        raise ValueError(f"Duplicated IDs after combining submitted train/hold-out: {duplicated}")
    return combined


def repeated_cv_policy(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    y = binary_target(frame)
    return pd.DataFrame(
        [
            {
                "analysis": "post_imputation_fixed_feature_repeated_stratified_5fold_cv",
                "dataset": "submitted_imputed_japanese_ppv_train_plus_holdout",
                "n": len(frame),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "feature_set": FEATURE_SET_NAME,
                "feature_count": len(feature_columns),
                "cv_method": "RepeatedStratifiedKFold",
                "n_splits": N_SPLITS,
                "n_repeats": N_REPEATS,
                "total_validation_folds_per_model": N_SPLITS * N_REPEATS,
                "cv_random_state": REPEATED_CV_RANDOM_STATE,
                "resampling": f"RandomUnderSampler(random_state={OLD_UNDERSAMPLING_RANDOM_STATE}) inside each training fold",
                "imputation_rerun": False,
                "feature_selection_rerun": False,
                "hyperparameter_tuning_rerun": False,
                "classical_hyperparameters": (
                    "submitted random-undersampling CV notebook settings for submitted model families; "
                    "ridge logistic regression uses inner StratifiedKFold C selection inside each training fold"
                ),
                "tabpfn_settings": (
                    f"device={TABPFN_DEVICE}; "
                    f"N_ensemble_configurations={TABPFN_ENSEMBLES}; "
                    f"seed={TABPFN_RANDOM_STATE}"
                ),
                "interpretation_scope": (
                    "Stability sensitivity after submitted imputation and fixed RFECV36 feature selection; "
                    "not a full raw-data repeated pipeline."
                ),
            }
        ]
    )


def repeated_cv_fold_rows(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = numeric_features(frame, feature_columns, "submitted imputed Japanese PPV combined cohort")
    y = binary_target(frame)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=REPEATED_CV_RANDOM_STATE,
    )

    rows = []
    completed_keys = set()
    if REPEATED_CV_FOLD_LEVEL_CSV.is_file():
        existing = pd.read_csv(REPEATED_CV_FOLD_LEVEL_CSV, encoding="utf-8-sig")
        rows = existing.to_dict("records")
        completed = existing[existing["status"].eq("completed")]
        completed_keys = {
            (int(row["split_index"]), str(row["model"]))
            for _, row in completed.iterrows()
            if str(row["model"]) in MODEL_ORDER
        }

    for split_index, (train_index, validation_index) in enumerate(splitter.split(x, y)):
        repeat = split_index // N_SPLITS + 1
        fold = split_index % N_SPLITS + 1
        x_train = x[train_index]
        y_train = y[train_index]
        x_validation = x[validation_index]
        y_validation = y[validation_index]

        sampler = RandomUnderSampler(random_state=OLD_UNDERSAMPLING_RANDOM_STATE)
        x_fit, y_fit = sampler.fit_resample(x_train, y_train)

        for model_key in MODEL_ORDER:
            if (split_index, model_key) in completed_keys:
                continue
            row = {
                "analysis": "repeated_stratified5fold_cv",
                "repeat": repeat,
                "fold": fold,
                "split_index": split_index,
                "model": model_key,
                "model_label": DISPLAY_MODEL_LABELS[model_key],
                "feature_set": FEATURE_SET_NAME,
                "feature_count": len(feature_columns),
                "resampling_condition": RESAMPLING_CONDITION,
                "train_n_before_resampling": len(y_train),
                "train_events_before_resampling": int(y_train.sum()),
                "train_n_used": len(y_fit),
                "train_events_used": int(y_fit.sum()),
                "validation_n": len(y_validation),
                "validation_events": int(y_validation.sum()),
                "auroc": np.nan,
                "auprc": np.nan,
                "fit_seconds": np.nan,
                "status": "started",
                "error": "",
            }
            try:
                model = make_submitted_cv_undersampling_model(model_key, y_fit)
                start = time.perf_counter()
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    warnings.simplefilter("ignore")
                    fit_model(model, x_fit, y_fit)
                score = positive_probability(model, x_validation)
                row["fit_seconds"] = time.perf_counter() - start
                row["auroc"] = float(roc_auc_score(y_validation, score))
                row["auprc"] = float(average_precision_score(y_validation, score))
                row["status"] = "completed"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
            rows.append(row)
            pd.DataFrame(rows).sort_values(["split_index", "model"]).to_csv(
                REPEATED_CV_FOLD_LEVEL_CSV,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"[{STEP_DIR.name}] repeat {repeat}/{N_REPEATS}, fold {fold}/{N_SPLITS}, "
                f"{DISPLAY_MODEL_LABELS[model_key]}: {row['status']}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["split_index", "model"]).reset_index(drop=True)


def summarize_repeated_cv(fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed = fold_rows[fold_rows["status"].eq("completed")].copy()
    for model_key in MODEL_ORDER:
        model_rows = completed[completed["model"].eq(model_key)].copy()
        values = pd.to_numeric(model_rows["auroc"], errors="raise").to_numpy(dtype=float)
        auprc_values = pd.to_numeric(model_rows["auprc"], errors="raise").to_numpy(dtype=float)
        rows.append(
            {
                "model": model_key,
                "model_label": DISPLAY_MODEL_LABELS[model_key],
                "completed_folds": len(values),
                "expected_folds": N_SPLITS * N_REPEATS,
                "feature_set": FEATURE_SET_NAME,
                "feature_count": int(model_rows["feature_count"].iloc[0]) if len(model_rows) else "",
                "resampling_condition": RESAMPLING_CONDITION,
                "auroc_mean": float(np.mean(values)),
                "auroc_sd": float(np.std(values, ddof=1)),
                "auroc_median": float(np.median(values)),
                "auroc_iqr_low": float(np.percentile(values, 25)),
                "auroc_iqr_high": float(np.percentile(values, 75)),
                "auroc_percentile_2p5": float(np.percentile(values, 2.5)),
                "auroc_percentile_97p5": float(np.percentile(values, 97.5)),
                "auroc_min": float(np.min(values)),
                "auroc_max": float(np.max(values)),
                "auroc_mean_sd": f"{np.mean(values):.3f} ({np.std(values, ddof=1):.3f})",
                "auroc_median_iqr": (
                    f"{np.median(values):.3f} "
                    f"({np.percentile(values, 25):.3f}-{np.percentile(values, 75):.3f})"
                ),
                "auroc_percentile_interval": (
                    f"{np.mean(values):.3f} "
                    f"({np.percentile(values, 2.5):.3f}-{np.percentile(values, 97.5):.3f})"
                ),
                "auroc_range": f"{np.min(values):.3f}-{np.max(values):.3f}",
                "auprc_mean": float(np.mean(auprc_values)),
                "auprc_sd": float(np.std(auprc_values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def repeated_cv_manuscript_layout(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Models",
        "Validation folds, n",
        "AUROC, mean (SD)",
        "AUROC, median (IQR)",
        "AUROC, empirical 2.5th-97.5th percentile",
        "AUROC range",
        "AUPRC, mean (SD)",
    ]
    rows = []
    for model_key in MODEL_ORDER:
        row = summary[summary["model"].eq(model_key)].iloc[0]
        rows.append(
            {
                "Models": row["model_label"],
                "Validation folds, n": int(row["completed_folds"]),
                "AUROC, mean (SD)": row["auroc_mean_sd"],
                "AUROC, median (IQR)": row["auroc_median_iqr"],
                "AUROC, empirical 2.5th-97.5th percentile": row["auroc_percentile_interval"],
                "AUROC range": row["auroc_range"],
                "AUPRC, mean (SD)": f"{row['auprc_mean']:.3f} ({row['auprc_sd']:.3f})",
            }
        )
    rows.append(
        {
            "Models": (
                "Abbreviations: AUPRC, area under the precision-recall curve; "
                "AUROC, area under the receiver operating characteristic curve; "
                "IQR, interquartile range; SD, standard deviation. Repeated "
                "Stratified 5-Fold CV used the submitted imputed Japanese PPV "
                "cohort and fixed submitted RFECV36 predictors, with random "
                "undersampling applied inside each training fold. Validation "
                f"folds represent {N_REPEATS} repeats of stratified 5-fold CV. "
                "The empirical percentile interval summarizes split-to-split "
                "variability and is not an independent-patient 95% confidence interval."
            ),
            "Validation folds, n": "",
            "AUROC, mean (SD)": "",
            "AUROC, median (IQR)": "",
            "AUROC, empirical 2.5th-97.5th percentile": "",
            "AUROC range": "",
            "AUPRC, mean (SD)": "",
        }
    )
    return pd.DataFrame(rows, columns=columns)


def update_repeated_cv_coverage() -> None:
    if not COVERAGE_CHECK_CSV.is_file():
        return
    coverage = pd.read_csv(COVERAGE_CHECK_CSV, encoding="utf-8-sig")
    mask = coverage["reviewer_request_component"].eq("Repeated 5-fold cross-validation")
    if mask.any():
        coverage.loc[mask, "status_in_step50"] = "covered via Step50-04"
        coverage.loc[mask, "evidence_source"] = "r3_09_repeated_stratified5fold_cv_auc_summary.csv"
        coverage.loc[
            mask,
            "note",
        ] = (
            f"RepeatedStratifiedKFold with {N_REPEATS} repeats and 5 folds was run using "
            "the submitted imputed Japanese PPV cohort and fixed submitted "
            "RFECV36 predictors; random undersampling was applied inside each "
            "training fold. Ridge logistic regression was added as a regularized "
            "comparator with inner-fold C selection."
        )
        write_csv(coverage, COVERAGE_CHECK_CSV)


def run_analysis() -> None:
    print(f"[{STEP_DIR.name}] Start repeated stratified 5-fold CV stability analysis.", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_tabpfn_checkpoint_available()

    frame = combined_submitted_imputed_ppv()
    feature_columns = read_rfecv36_features()
    policy = repeated_cv_policy(frame, feature_columns)
    write_csv(policy, REPEATED_CV_POLICY_CSV)
    fold_rows = repeated_cv_fold_rows(frame, feature_columns)
    summary = summarize_repeated_cv(fold_rows)
    layout = repeated_cv_manuscript_layout(summary)

    write_csv(fold_rows, REPEATED_CV_FOLD_LEVEL_CSV)
    write_csv(summary, REPEATED_CV_SUMMARY_CSV)
    write_csv(layout, REPEATED_CV_LAYOUT_CSV)
    update_repeated_cv_coverage()
    print(f"[{STEP_DIR.name}] Done.", flush=True)


if __name__ == "__main__":
    run_analysis()
