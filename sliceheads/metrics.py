"""Module 3: evaluation metrics with 95% bootstrap CI for AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from sliceheads.constants import OPERATING_POINT, DEFAULT_RANDOM_SEED


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """Compute the full benchmark metric suite.

    Parameters
    ----------
    y_true : [B] int array, values in {0, 1}.
    proba  : [B, 2] float array, column 1 = positive-class probability.
    n_bootstraps : number of bootstrap resamples for AUC CI.
    ci : confidence interval level.
    random_seed : for reproducible bootstrap.

    Returns
    -------
    dict with all metrics and 'operating_point' metadata.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float32)
    scores = proba[:, 1]

    y_pred = (scores >= OPERATING_POINT).astype(int)

    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    roc_boots = []
    pr_boots = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        roc_boots.append(float(roc_auc_score(yt, ys)))
        pr_boots.append(float(average_precision_score(yt, ys)))

    alpha = (1.0 - ci) / 2.0
    roc_ci = (
        (float(np.percentile(roc_boots, 100 * alpha)),
         float(np.percentile(roc_boots, 100 * (1 - alpha))))
        if roc_boots else (float("nan"), float("nan"))
    )
    pr_ci = (
        (float(np.percentile(pr_boots, 100 * alpha)),
         float(np.percentile(pr_boots, 100 * (1 - alpha))))
        if pr_boots else (float("nan"), float("nan"))
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_class_0": float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
        "roc_auc_ci_lower": roc_ci[0],
        "roc_auc_ci_upper": roc_ci[1],
        "pr_auc_ci_lower": pr_ci[0],
        "pr_auc_ci_upper": pr_ci[1],
        "operating_point": OPERATING_POINT,
        "n_bootstraps": n_bootstraps,
        "ci": ci,
        "n_samples": int(n),
    }
