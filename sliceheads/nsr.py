"""Module 7: Noise-to-Signal Ratio (NSR) Robustness Analysis.

Measures how each classification head's discrimination degrades when
task-irrelevant "background" slice embeddings are appended to nested-CV
outer-fold test scans. "Noise" here means appended background slice
embeddings drawn from class-0 (negative) scans in the corresponding outer
training partition — not Gaussian numerical noise. Reuses already-fitted
(frozen) models; nothing in this module trains anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sliceheads.metrics import compute_metrics


@dataclass
class BackgroundPool:
    """Every individual slice embedding from class-0 scans in one outer-train fold."""

    embeddings: np.ndarray  # [P, D]
    source_scan_id: list[str] = field(default_factory=list)  # len P
    source_slice_index: list[int] = field(default_factory=list)  # len P

    def __len__(self) -> int:
        return self.embeddings.shape[0]


def build_background_pool(
    embeddings_train: list[np.ndarray],
    labels_train: np.ndarray,
    ids_train: list[str],
) -> BackgroundPool:
    """Flatten every slice from class-0 (negative) scans in the outer-train
    partition into one pool of individually addressable background slices.

    Class-1 scans and anything outside the outer-train partition (in
    particular the outer-test fold) never contribute — the caller must pass
    only the outer-train subset.
    """
    labels_train = np.asarray(labels_train)
    pool_chunks: list[np.ndarray] = []
    source_scan_id: list[str] = []
    source_slice_index: list[int] = []

    for emb, label, sid in zip(embeddings_train, labels_train, ids_train):
        if int(label) != 0:
            continue
        n = emb.shape[0]
        pool_chunks.append(emb)
        source_scan_id.extend([sid] * n)
        source_slice_index.extend(range(n))

    if not pool_chunks:
        raise ValueError("No class-0 scans found in the outer-train partition to build a background pool from.")

    embeddings = np.concatenate(pool_chunks, axis=0)
    return BackgroundPool(
        embeddings=embeddings,
        source_scan_id=source_scan_id,
        source_slice_index=source_slice_index,
    )


def spawn_scan_seeds(seed: int, n_scans: int) -> list[np.random.SeedSequence]:
    """Independent, reproducible per-scan child seeds derived from one
    repeat-level seed, via numpy's SeedSequence spawning.

    Used so that every test scan draws its own independent background set
    for a given (outer_fold, repeat) instead of sharing one set across the
    whole fold/repeat — a shared background set can artificially favor
    aggregation/pooling heads, since they'd all see literally identical
    injected slices. Spawning is deterministic in (seed, n_scans, position):
    the same seed always produces the same n_scans children in the same
    order, and each child is independent of its siblings. Pass a spawned
    child directly as the `seed` argument to sample_background_set (or to
    np.random.default_rng) — both accept a SeedSequence natively.
    """
    return np.random.SeedSequence(seed).spawn(n_scans)


def sample_background_set(
    pool: BackgroundPool, n: int, seed: int | np.random.SeedSequence
) -> tuple[np.ndarray, list[dict]]:
    """Draw n background slices without replacement, uniformly over the pool.

    Call this once per (outer_fold, repeat, scan) with n =
    max(dilution_levels) — each test scan should get its own seed (see
    spawn_scan_seeds) so its background draw is independent of every other
    scan's, rather than one shared set applied to the whole fold/repeat. The
    returned draw order is then used directly for every b by slicing a
    prefix of the single result (background[:50], background[:100], ...) —
    this is what makes B_50 subset of B_100 subset of B_150 hold for that
    scan. Do NOT call this again with a smaller n for the same seed: numpy's
    replace=False sampling is not prefix-stable across independent calls at
    different sizes, even with an identical seed, so re-drawing per b would
    silently break the nesting guarantee.

    Returns
    -------
    background_embeddings : [n, D] array, in draw order.
    manifest_rows : list of {"background_index", "source_scan_id",
        "source_slice_index"} dicts, in draw order (background_index is the
        position in this draw, 0-indexed).
    """
    if len(pool) < n:
        raise ValueError(
            f"Background pool has only {len(pool)} eligible slices, need {n}."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)

    background_embeddings = pool.embeddings[idx]
    manifest_rows = [
        {
            "background_index": i,
            "source_scan_id": pool.source_scan_id[j],
            "source_slice_index": pool.source_slice_index[j],
        }
        for i, j in enumerate(idx)
    ]
    return background_embeddings, manifest_rows


def dilute_bag(original_embedding: np.ndarray, background_embeddings: np.ndarray, b: int) -> np.ndarray:
    """Append the first b background embeddings after the scan's own slices.

    b=0 returns original_embedding unchanged (same object — never copied or
    mutated). b>0 returns a new concatenated array; original_embedding is
    never modified in place.
    """
    if b == 0:
        return original_embedding
    return np.concatenate([original_embedding, background_embeddings[:b]], axis=0)


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clipped logit: log(p / (1 - p)), clipping p to [eps, 1-eps] first."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def pooled_metrics(y_true: np.ndarray, proba: np.ndarray, n_bootstraps: int = 0) -> dict:
    """Pooled-OOF metrics for one (head, repeat, b) condition.

    Thin wrapper around sliceheads.metrics.compute_metrics — same
    conventions (prob_class_1 positive, 0.5 operating point) as the rest of
    the classification pipeline. n_bootstraps=0 by default: the NSR summary
    draws its uncertainty from repeat-to-repeat variance across the 10
    random background draws, not from a per-repeat bootstrap CI, so the
    bootstrap is skipped here for speed.
    """
    return compute_metrics(y_true, proba, n_bootstraps=n_bootstraps)
