"""Module 4 (runner half): hyperparameter optimisation over classification heads."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

from sliceheads.config import iter_grid, iter_random
from sliceheads.heads.base import BaseHead
from sliceheads.metrics import compute_metrics


# Mapping head name -> fully-qualified class path
_HEAD_CLASSES: dict[str, str] = {
    "MeanPoolClassifier":        "sliceheads.heads.pooling.MeanPoolClassifier",
    "MaxPoolClassifier":         "sliceheads.heads.pooling.MaxPoolClassifier",
    "GeMPoolClassifier":         "sliceheads.heads.pooling.GeMPoolClassifier",
    "ABMILClassifier":           "sliceheads.heads.mil.ABMILClassifier",
    "GatedABMILClassifier":      "sliceheads.heads.mil.GatedABMILClassifier",
    "DSMILClassifier":           "sliceheads.heads.mil.DSMILClassifier",
    "TransformerMILClassifier":  "sliceheads.heads.transformer.TransformerMILClassifier",
    "MultiRocketClassifier":     "sliceheads.heads.timeseries.MultiRocketClassifier",
    "InceptionTimeClassifier":   "sliceheads.heads.timeseries.InceptionTimeClassifier",
    "ALSTMFCNClassifier":        "sliceheads.heads.timeseries.ALSTMFCNClassifier",
}


def _resolve_head_class(name: str):
    fqn = _HEAD_CLASSES[name]
    module, cls = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def run_hpo(
    head_name: str,
    head_cfg: dict[str, Any],
    embeddings_train: list[np.ndarray],
    labels_train: np.ndarray,
    embeddings_val: list[np.ndarray],
    labels_val: np.ndarray,
    *,
    search_strategy: str = "grid",
    selection_metric: str = "roc_auc",
    training_cfg: dict[str, Any] | None = None,
    cancelled: Callable[[], bool] | None = None,
    n_random: int = 50,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run HPO for a single head and return best params + fitted head.

    Parameters
    ----------
    cancelled : optional callable; if it returns True the search stops early.
    Returns dict with keys: best_params, best_score, best_head, all_results.
    """
    if not head_cfg.get("enabled", True):
        return {"skipped": True}

    training_cfg = training_cfg or {}
    device = training_cfg.get("device", "cpu")
    seed = training_cfg.get("random_seed", random_seed)

    hparams = dict(head_cfg.get("hyperparameters", {}))
    # Ensure all values are lists for iter_grid / iter_random
    for k, v in hparams.items():
        if not isinstance(v, list):
            hparams[k] = [v]

    if search_strategy == "grid":
        combos = iter_grid(hparams)
    elif search_strategy == "random":
        combos = iter_random(hparams, n=n_random, seed=seed)
    elif search_strategy == "bayesian":
        # Bayesian not yet implemented; fall back to random with a larger budget
        combos = iter_random(hparams, n=max(n_random, 100), seed=seed)
    else:
        raise ValueError(f"Unknown search_strategy: {search_strategy}")

    head_cls = _resolve_head_class(head_name)
    best_score = -float("inf")
    best_params: dict[str, Any] = {}
    best_head: BaseHead | None = None
    all_results = []

    for params in combos:
        if cancelled is not None and cancelled():
            break

        # Filter to params the head constructor accepts
        try:
            head = head_cls(**{**params, "device": device, "random_seed": seed})
        except TypeError:
            # If device/random_seed not accepted (e.g. sklearn-only heads), omit them
            clean = {k: v for k, v in params.items() if k not in ("device", "random_seed")}
            try:
                head = head_cls(**{**clean, "random_seed": seed})
            except TypeError:
                head = head_cls(**clean)

        try:
            head.fit(embeddings_train, labels_train)
            proba = head.predict_proba(embeddings_val)
            m = compute_metrics(labels_val, proba, n_bootstraps=200)
            score = m[selection_metric]
        except Exception as exc:
            all_results.append({"params": params, "score": None, "error": str(exc)})
            continue

        all_results.append({"params": params, "score": score})
        if score > best_score:
            best_score = score
            best_params = params
            best_head = head

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_head": best_head,
        "all_results": all_results,
    }


def run_benchmark(
    config: dict[str, Any],
    embeddings_train: list[np.ndarray],
    labels_train: np.ndarray,
    embeddings_val: list[np.ndarray],
    labels_val: np.ndarray,
    embeddings_test: list[np.ndarray],
    labels_test: np.ndarray,
    sample_ids_test: list[str],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Full benchmark run: HPO + test evaluation for all enabled heads.

    Returns a dict suitable for passing to results.save_results().
    """
    strategy = config.get("search_strategy", "grid")
    selection_metric = config.get("training", {}).get("selection_metric", "roc_auc")
    training_cfg = config.get("training", {})

    metrics_per_head: dict[str, dict] = {}
    predictions_per_head: dict[str, dict] = {}
    hyperparameters_per_head: dict[str, Any] = {}
    best_heads_per_head: dict[str, BaseHead] = {}
    hpo_detail_per_head: dict[str, list] = {}

    for head_name, head_cfg in config.get("heads", {}).items():
        if not head_cfg.get("enabled", True):
            continue
        if cancelled is not None and cancelled():
            break

        hpo_result = run_hpo(
            head_name, head_cfg,
            embeddings_train, labels_train,
            embeddings_val, labels_val,
            search_strategy=strategy,
            selection_metric=selection_metric,
            training_cfg=training_cfg,
            cancelled=cancelled,
        )

        if hpo_result.get("skipped"):
            continue

        best_head: BaseHead = hpo_result["best_head"]
        if best_head is None:
            continue

        hyperparameters_per_head[head_name] = hpo_result["best_params"]
        best_heads_per_head[head_name] = best_head
        hpo_detail_per_head[head_name] = hpo_result["all_results"]

        proba = best_head.predict_proba(embeddings_test)
        metrics_per_head[head_name] = compute_metrics(labels_test, proba)

        preds_labels = proba.argmax(axis=1)
        preds_dict = {}
        for i, sid in enumerate(sample_ids_test):
            preds_dict[sid] = {
                "true_label": int(labels_test[i]),
                "pred_label": int(preds_labels[i]),
                "prob_class_0": float(proba[i, 0]),
                "prob_class_1": float(proba[i, 1]),
            }
        predictions_per_head[head_name] = preds_dict

    return {
        "metrics_per_head": metrics_per_head,
        "predictions_per_head": predictions_per_head,
        "hyperparameters_per_head": hyperparameters_per_head,
        "best_heads_per_head": best_heads_per_head,
        "hpo_detail_per_head": hpo_detail_per_head,
    }
