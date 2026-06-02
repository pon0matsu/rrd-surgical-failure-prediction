"""Shared utilities for the compact reviewer-reanalysis workspace.

REVIEWER REVISION ADDITION:
    These helpers keep the Review/ folder focused on code-generated,
    reviewer-facing aggregate outputs.  Scripts may read private row-level
    prediction sources from the internal workspace, but they do not copy those
    rows into Review/local_outputs.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


REVIEW_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REVIEW_ROOT.parent
SUBMITTED_CODE_ROOT = WORKSPACE_ROOT / "20250725_RRD-20260507T074230Z-3-001/20250725_RRD"
PREVIOUS_CODES_ROOT = WORKSPACE_ROOT / "RRD_Analysis/data/previous_codes"

TARGET_COLUMN = "Failure level (0 vs 1 to 3) (6M)"
POSITIVE_CLASS_LABEL = 1
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42

MODEL_COLUMNS = {
    "tabpfn": "tabpfn_pred_proba",
    "xgb": "xgb_pred_proba",
    "LR": "LR_pred_proba",
    "rf": "rf_pred_proba",
    "lgb": "lgb_pred_proba",
}

MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "xgb": "XGBoost",
    "LR": "Logistic regression",
    "rf": "Random forest",
    "lgb": "LightGBM",
}

PRIMARY_POLICY = (
    "The manuscript primary model remains the submitted RFECV36 TabPFN "
    "random-undersampling workflow.  Ridge/logistic models, all46 predictors, "
    "compact26 predictors, no-undersampling, class weighting, recalibration, "
    "strict-endpoint, and subgroup analyses are reviewer-response sensitivity "
    "or comparator evidence."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    resolved = path.resolve()
    for root in (WORKSPACE_ROOT.resolve(), REVIEW_ROOT.resolve()):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(path)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file not found: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    print(
        f"[{utc_now()}] Read CSV: {rel(path)} rows={len(frame)} cols={frame.shape[1]}",
        flush=True,
    )
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_dir(step_dir: Path) -> Path:
    """Clear stale generated files for one Review step and recreate local_outputs."""

    output_dir = step_dir / "local_outputs"
    if output_dir.exists():
        if output_dir.name != "local_outputs" or REVIEW_ROOT.resolve() not in output_dir.resolve().parents:
            raise RuntimeError(f"Refusing to clear unexpected output directory: {output_dir}")
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] Output directory ready: {rel(output_dir)}", flush=True)
    return output_dir


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(
        f"[{utc_now()}] Wrote CSV: {rel(path)} rows={len(frame)} cols={frame.shape[1]}",
        flush=True,
    )


def clean_binary_outcome(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.Series(values).astype(float).round().astype(int).to_numpy()
    unique = set(np.unique(array))
    if not unique.issubset({0, 1}):
        raise ValueError(f"Expected binary outcome, observed values={sorted(unique)}")
    return array


def clean_score(values: pd.Series | np.ndarray) -> np.ndarray:
    score = pd.Series(values).astype(float).to_numpy()
    return np.clip(score, 1e-6, 1 - 1e-6)


def exact_binomial_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    interval = stats.binomtest(successes, total).proportion_ci(confidence_level=1 - alpha, method="exact")
    return float(interval.low), float(interval.high)


def submitted_percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if len(sorted_values) == 0:
        return np.nan, np.nan
    low = float(sorted_values[int(0.025 * len(sorted_values))])
    high = float(sorted_values[int(0.975 * len(sorted_values))])
    return low, high


def point_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    out = {
        "event_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "auprc": float(average_precision_score(y_true, score)) if y_true.sum() > 0 else np.nan,
        "brier_score": float(brier_score_loss(y_true, score)),
    }
    return out


def bootstrap_metric_summary(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    dataset: str,
    model: str,
    model_label: str | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    n = len(y_true)
    events = int(y_true.sum())
    rng = np.random.RandomState(seed)
    boot: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier_score": []}
    for _ in range(iterations):
        idx = rng.randint(0, n, n)
        sample_y = y_true[idx]
        sample_score = score[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        boot["auroc"].append(float(roc_auc_score(sample_y, sample_score)))
        boot["auprc"].append(float(average_precision_score(sample_y, sample_score)))
        boot["brier_score"].append(float(brier_score_loss(sample_y, sample_score)))

    point = point_metrics(y_true, score)
    event_low, event_high = exact_binomial_ci(events, n)
    rows = [
        {
            "dataset": dataset,
            "model": model,
            "model_label": model_label or model,
            "n": n,
            "events": events,
            "metric": "event_rate",
            "estimate": point["event_rate"],
            "ci_low": event_low,
            "ci_high": event_high,
            "bootstrap_valid_iterations": "",
        }
    ]
    for metric in ("auroc", "auprc", "brier_score"):
        values = np.asarray(boot[metric], dtype=float)
        low, high = submitted_percentile_ci(values)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "model_label": model_label or model,
                "n": n,
                "events": events,
                "metric": metric,
                "estimate": point[metric],
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_valid_iterations": len(values),
            }
        )
    return pd.DataFrame(rows)


def wide_metric_summary(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    dataset: str,
    model: str,
    model_label: str | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    long = bootstrap_metric_summary(
        y_true,
        score,
        dataset=dataset,
        model=model,
        model_label=model_label,
        iterations=iterations,
        seed=seed,
    )
    first = long.iloc[0]
    row: dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "model_label": model_label or model,
        "n": int(first["n"]),
        "events": int(first["events"]),
    }
    for _, metric_row in long.iterrows():
        metric = metric_row["metric"]
        row[metric] = float(metric_row["estimate"])
        row[f"{metric}_ci_low"] = float(metric_row["ci_low"])
        row[f"{metric}_ci_high"] = float(metric_row["ci_high"])
        row[f"{metric}_95ci"] = fmt_ci(metric_row["estimate"], metric_row["ci_low"], metric_row["ci_high"])
    row.update(threshold_metric_summary(y_true, score, threshold=DEFAULT_CLASSIFICATION_THRESHOLD))
    return row


def fmt_ci(estimate: float, low: float, high: float, digits: int = 3) -> str:
    if pd.isna(estimate) or pd.isna(low) or pd.isna(high):
        return "NA"
    return f"{estimate:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def fmt_range(estimate: float, low: float, high: float, digits: int = 3) -> str:
    if pd.isna(estimate) or pd.isna(low) or pd.isna(high):
        return "NA"
    return f"{estimate:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def threshold_metric_summary(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict[str, object]:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    positives = int(tp + fn)
    negatives = int(tn + fp)
    flagged = int(tp + fp)
    unflagged = int(tn + fn)
    sensitivity_low, sensitivity_high = exact_binomial_ci(int(tp), positives)
    specificity_low, specificity_high = exact_binomial_ci(int(tn), negatives)
    ppv_low, ppv_high = exact_binomial_ci(int(tp), flagged)
    npv_low, npv_high = exact_binomial_ci(int(tn), unflagged)

    sensitivity = safe_div(tp, positives)
    specificity = safe_div(tn, negatives)
    ppv = safe_div(tp, flagged)
    npv = safe_div(tn, unflagged)
    return {
        "score_threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "sensitivity": sensitivity,
        "sensitivity_ci_low": sensitivity_low,
        "sensitivity_ci_high": sensitivity_high,
        "sensitivity_95ci": fmt_ci(sensitivity, sensitivity_low, sensitivity_high),
        "specificity": specificity,
        "specificity_ci_low": specificity_low,
        "specificity_ci_high": specificity_high,
        "specificity_95ci": fmt_ci(specificity, specificity_low, specificity_high),
        "ppv": ppv,
        "ppv_ci_low": ppv_low,
        "ppv_ci_high": ppv_high,
        "ppv_95ci": fmt_ci(ppv, ppv_low, ppv_high),
        "npv": npv,
        "npv_ci_low": npv_low,
        "npv_ci_high": npv_high,
        "npv_95ci": fmt_ci(npv, npv_low, npv_high),
    }


def threshold_table(y_true: np.ndarray, score: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        predicted = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "score_threshold": threshold,
                "flagged_n": int(predicted.sum()),
                "flagged_percent": float(predicted.mean()),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "sensitivity": safe_div(tp, tp + fn),
                "specificity": safe_div(tn, tn + fp),
                "ppv": safe_div(tp, tp + fp),
                "npv": safe_div(tn, tn + fn),
            }
        )
    return pd.DataFrame(rows)


def risk_enrichment_table(y_true: np.ndarray, score: np.ndarray, fractions: Iterable[float]) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    baseline = float(y_true.mean())
    order = np.argsort(-score)
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        top_n = max(1, int(np.ceil(len(y_true) * fraction)))
        selected = order[:top_n]
        event_rate = float(y_true[selected].mean())
        rows.append(
            {
                "top_score_fraction": fraction,
                "top_score_n": top_n,
                "events": int(y_true[selected].sum()),
                "event_rate": event_rate,
                "baseline_event_rate": baseline,
                "risk_enrichment_ratio": safe_div(event_rate, baseline),
                "minimum_score_in_group": float(score[selected].min()),
            }
        )
    return pd.DataFrame(rows)


def decision_curve_table(y_true: np.ndarray, score: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    n = len(y_true)
    prevalence = float(y_true.mean())
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            continue
        predicted = score >= threshold
        tp = int(((y_true == 1) & predicted).sum())
        fp = int(((y_true == 0) & predicted).sum())
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "threshold_probability": float(threshold),
                "net_benefit_model": (tp / n) - (fp / n) * weight,
                "net_benefit_treat_all": prevalence - (1 - prevalence) * weight,
                "net_benefit_treat_none": 0.0,
                "flagged_percent": float(predicted.mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_bins(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    frame = pd.DataFrame({"y_true": y_true, "score": score})
    unique_scores = frame["score"].nunique()
    q = min(bins, unique_scores)
    if q < 2:
        frame["bin"] = 0
    else:
        frame["bin"] = pd.qcut(frame["score"], q=q, duplicates="drop")
    grouped = frame.groupby("bin", observed=False)
    out = grouped.agg(
        n=("y_true", "size"),
        events=("y_true", "sum"),
        observed_event_rate=("y_true", "mean"),
        mean_predicted_score=("score", "mean"),
        min_predicted_score=("score", "min"),
        max_predicted_score=("score", "max"),
    ).reset_index(drop=True)
    out.insert(0, "bin_index", np.arange(1, len(out) + 1))
    return out


def calibration_diagnostics(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    x = logit(score).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(x, y_true)
    except TypeError:
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        model.fit(x, y_true)
    grouped = calibration_bins(y_true, score, bins=bins)
    expected = grouped["mean_predicted_score"] * grouped["n"]
    observed = grouped["events"]
    denom = np.clip(expected * (1 - grouped["mean_predicted_score"]), 1e-9, None)
    hl_chi2 = float((((observed - expected) ** 2) / denom).sum())
    hl_df = max(int(len(grouped) - 2), 1)
    return pd.DataFrame(
        [
            {
                "n": len(y_true),
                "events": int(y_true.sum()),
                "event_rate": float(y_true.mean()),
                "brier_score": float(brier_score_loss(y_true, score)),
                "calibration_intercept": float(model.intercept_[0]),
                "calibration_slope": float(model.coef_[0][0]),
                "hosmer_lemeshow_chi2": hl_chi2,
                "hosmer_lemeshow_df": hl_df,
                "hosmer_lemeshow_p": float(stats.chi2.sf(hl_chi2, hl_df)),
                "groups": len(grouped),
            }
        ]
    )


def apparent_logistic_recalibration(y_true: np.ndarray, score: np.ndarray) -> np.ndarray:
    y_true = clean_binary_outcome(y_true)
    score = clean_score(score)
    x = logit(score).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(x, y_true)
    except TypeError:
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        model.fit(x, y_true)
    return expit(model.intercept_[0] + model.coef_[0][0] * logit(score))


def paired_bootstrap_differences(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    comparator_score: np.ndarray,
    *,
    reference_model: str,
    comparator_model: str,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    y_true = clean_binary_outcome(y_true)
    reference_score = clean_score(reference_score)
    comparator_score = clean_score(comparator_score)
    if metrics is None:
        metrics = {
            "auroc": lambda y, s: float(roc_auc_score(y, s)),
            "auprc": lambda y, s: float(average_precision_score(y, s)),
            "brier_score": lambda y, s: float(brier_score_loss(y, s)),
        }
    rng = np.random.RandomState(seed)
    n = len(y_true)
    rows: list[dict[str, object]] = []
    for metric, func in metrics.items():
        point_reference = func(y_true, reference_score)
        point_comparator = func(y_true, comparator_score)
        diffs: list[float] = []
        for _ in range(iterations):
            idx = rng.randint(0, n, n)
            sample_y = y_true[idx]
            if len(np.unique(sample_y)) < 2:
                continue
            diffs.append(func(sample_y, reference_score[idx]) - func(sample_y, comparator_score[idx]))
        ci_low, ci_high = submitted_percentile_ci(diffs)
        rows.append(
            {
                "reference_model": reference_model,
                "comparator_model": comparator_model,
                "metric": metric,
                "reference_estimate": point_reference,
                "comparator_estimate": point_comparator,
                "difference_reference_minus_comparator": point_reference - point_comparator,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_valid_iterations": len(diffs),
            }
        )
    return pd.DataFrame(rows)


def compute_midrank(values: np.ndarray) -> np.ndarray:
    """Return 1-based midranks used by the fast DeLong covariance estimate."""

    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    index = 0
    while index < len(values):
        next_index = index
        while next_index < len(values) and sorted_values[next_index] == sorted_values[index]:
            next_index += 1
        ranks[index:next_index] = 0.5 * (index + next_index - 1) + 1
        index = next_index
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def fast_delong(predictions_sorted: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for one or more correlated ROC curves."""

    if predictions_sorted.ndim == 1:
        predictions_sorted = predictions_sorted.reshape(1, -1)
    negative_count = predictions_sorted.shape[1] - positive_count
    positive_predictions = predictions_sorted[:, :positive_count]
    negative_predictions = predictions_sorted[:, positive_count:]
    model_count = predictions_sorted.shape[0]

    positive_midranks = np.empty((model_count, positive_count), dtype=float)
    negative_midranks = np.empty((model_count, negative_count), dtype=float)
    all_midranks = np.empty((model_count, positive_count + negative_count), dtype=float)

    for model_index in range(model_count):
        positive_midranks[model_index, :] = compute_midrank(positive_predictions[model_index, :])
        negative_midranks[model_index, :] = compute_midrank(negative_predictions[model_index, :])
        all_midranks[model_index, :] = compute_midrank(predictions_sorted[model_index, :])

    aucs = (
        all_midranks[:, :positive_count].sum(axis=1)
        / positive_count
        / negative_count
        - (positive_count + 1.0) / (2.0 * negative_count)
    )
    v01 = (all_midranks[:, :positive_count] - positive_midranks) / negative_count
    v10 = 1.0 - (all_midranks[:, positive_count:] - negative_midranks) / positive_count
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = np.atleast_2d(sx / positive_count + sy / negative_count)
    return aucs, covariance


def delong_two_model_test(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    comparator_score: np.ndarray,
    *,
    reference_model: str,
    comparator_model: str,
) -> dict[str, object]:
    """Paired DeLong test for AUROC difference: reference minus comparator."""

    y_true = clean_binary_outcome(y_true)
    reference_score = clean_score(reference_score)
    comparator_score = clean_score(comparator_score)
    if len(np.unique(y_true)) != 2:
        raise ValueError("DeLong test requires both outcome classes.")
    positive_count = int(y_true.sum())
    if positive_count <= 0 or positive_count >= len(y_true):
        raise ValueError("DeLong test requires at least one positive and one negative case.")

    order = np.argsort(-y_true)
    predictions_sorted = np.vstack([reference_score, comparator_score])[:, order]
    aucs, covariance = fast_delong(predictions_sorted, positive_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())
    difference = float(aucs[0] - aucs[1])
    if variance <= 0 or not math.isfinite(variance):
        z_value = np.nan
        p_value = np.nan
    else:
        z_value = difference / math.sqrt(variance)
        p_value = 2.0 * stats.norm.sf(abs(z_value))
    return {
        "reference_model": reference_model,
        "comparator_model": comparator_model,
        "reference_auroc": float(aucs[0]),
        "comparator_auroc": float(aucs[1]),
        "auroc_difference_reference_minus_comparator": difference,
        "delong_z": float(z_value),
        "delong_p": float(p_value),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def add_metric_95ci(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"estimate", "ci_low", "ci_high"}.issubset(out.columns):
        out["estimate_95ci"] = [
            fmt_ci(row.estimate, row.ci_low, row.ci_high) for row in out.itertuples(index=False)
        ]
    return out
