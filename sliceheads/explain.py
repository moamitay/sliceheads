"""Module 5: LOO slice importance, native attention, and localization metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from sliceheads.heads.base import BaseHead


def loo_importance(
    head: BaseHead,
    embeddings: np.ndarray,
    mask: np.ndarray,
    mean_embedding: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute leave-one-out slice importance for a single sample.

    Parameters
    ----------
    head : fitted BaseHead.
    embeddings : [N, D] float32.
    mask : [N] float32 — 1=real, 0=padding.
    mean_embedding : [D] float32 — dataset training mean (from HDF5 root attr).

    Returns
    -------
    dict with keys:
        importance        : float32 [N], non-negative, sums to 1.
        signed_importance : float32 [N], raw signed delta p_0 - p_k.
    """
    N, D = embeddings.shape
    mean_emb = mean_embedding.astype(np.float32)
    valid = mask.astype(bool)

    def _positive_proba(emb: np.ndarray) -> float:
        return head.predict_proba([emb[valid]])[0, 1]

    p0 = _positive_proba(embeddings)  # positive-class prob

    deltas = np.zeros(N, dtype=np.float32)
    for k in range(N):
        if not valid[k]:
            continue  # padded/invalid slice contributes no importance
        X_k = embeddings.copy()
        X_k[k] = mean_emb
        deltas[k] = p0 - _positive_proba(X_k)  # signed delta

    abs_delta = np.abs(deltas)
    importance = abs_delta / (abs_delta.sum() + 1e-9)

    return {
        "importance": importance.astype(np.float32),
        "signed_importance": deltas.astype(np.float32),
    }


def compute_importance_dataset(
    head: BaseHead,
    embeddings_list: list[np.ndarray],
    sample_ids: list[str],
    mean_embedding: np.ndarray,
    *,
    mask_list: list[np.ndarray] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute LOO importance for every sample in a list.

    Returns {sample_id: {'importance': ..., 'signed_importance': ...,
                         'native_attention': ...}} .
    """
    results: dict[str, dict[str, np.ndarray]] = {}
    for i, (sample_id, emb) in enumerate(zip(sample_ids, embeddings_list)):
        N = emb.shape[0]
        if mask_list is not None:
            mask = mask_list[i]
        else:
            mask = np.ones(N, dtype=np.float32)

        imp = loo_importance(head, emb, mask, mean_embedding)

        # Native attention over the real (non-padded) slices, scattered back
        # into a full-length [N] array so downstream indexing lines up with
        # the original slice positions.
        native_attn = None
        if head.supports_native_attention:
            valid = mask.astype(bool)
            attn_valid = head.native_attention([emb[valid]])
            if attn_valid is not None:
                native_attn = np.zeros(N, dtype=np.float32)
                native_attn[valid] = attn_valid[0]

        results[sample_id] = {
            "importance": imp["importance"],
            "signed_importance": imp["signed_importance"],
            "native_attention": native_attn,
        }
    return results


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

def localization_metrics(
    importance: np.ndarray,
    important_slices: np.ndarray,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    """Spearman correlation + top-k% recall and precision vs ground-truth annotations.

    Parameters
    ----------
    importance       : [N] float32 predicted importance.
    important_slices : [N] uint8 binary ground-truth (1 = important).
    top_fraction     : fraction of top-ranked slices to use for recall/precision.

    Metrics
    -------
    spearman_correlation     : rank correlation between importance and annotation mask.
    top_k_pct_recall         : of all annotated slices, fraction that appear in top-k%.
    top_k_pct_precision      : of slices in top-k%, fraction that are truly annotated.

    Returns empty dict if ground-truth has no positive slices.
    """
    if important_slices.shape[0] == 0 or important_slices.sum() == 0:
        return {}

    N = len(importance)
    spearman_r, _ = spearmanr(importance, important_slices.astype(float))

    k = max(1, int(np.ceil(top_fraction * N)))
    top_idx = np.argsort(-importance)[:k]
    n_true_in_top = int(important_slices[top_idx].sum())
    n_true_total = int(important_slices.sum())
    recall_at_k    = n_true_in_top / max(n_true_total, 1)
    precision_at_k = n_true_in_top / k

    pct = int(top_fraction * 100)
    return {
        "spearman_correlation": float(spearman_r),
        f"top_{pct}_pct_recall":    recall_at_k,
        f"top_{pct}_pct_precision": precision_at_k,
    }
