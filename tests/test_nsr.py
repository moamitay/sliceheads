"""Tests for Module 7: NSR (Noise-to-Signal Ratio) Robustness Analysis."""

import numpy as np
import pytest

from sliceheads.metrics import compute_metrics
from sliceheads.nsr import (
    BackgroundPool,
    build_background_pool,
    dilute_bag,
    logit,
    pooled_metrics,
    sample_background_set,
    spawn_scan_seeds,
)


@pytest.fixture
def ids20():
    return [f"scan_{i}" for i in range(20)]


def test_pool_only_contains_class0_slices(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    pool = build_background_pool(embeddings_list, labels, ids20)

    class0_ids = {ids20[i] for i in range(len(labels)) if labels[i] == 0}
    class1_ids = {ids20[i] for i in range(len(labels)) if labels[i] == 1}

    assert set(pool.source_scan_id) <= class0_ids
    assert not (set(pool.source_scan_id) & class1_ids)

    expected_pool_size = sum(
        embeddings_list[i].shape[0] for i in range(len(labels)) if labels[i] == 0
    )
    assert len(pool) == expected_pool_size


def test_pool_raises_when_no_class0_scans(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    all_ones = np.ones_like(labels)
    with pytest.raises(ValueError):
        build_background_pool(embeddings_list, all_ones, ids20)


def test_sample_background_set_deterministic(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    pool = build_background_pool(embeddings_list, labels, ids20)

    bg1, manifest1 = sample_background_set(pool, n=30, seed=42)
    bg2, manifest2 = sample_background_set(pool, n=30, seed=42)
    bg3, _ = sample_background_set(pool, n=30, seed=43)

    np.testing.assert_array_equal(bg1, bg2)
    assert manifest1 == manifest2
    assert not np.array_equal(bg1, bg3)


def test_sample_background_set_no_replacement(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    pool = build_background_pool(embeddings_list, labels, ids20)

    _, manifest = sample_background_set(pool, n=30, seed=42)
    pairs = [(row["source_scan_id"], row["source_slice_index"]) for row in manifest]
    assert len(pairs) == len(set(pairs))


def test_sample_background_set_raises_when_pool_too_small(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    pool = build_background_pool(embeddings_list, labels, ids20)
    with pytest.raises(ValueError):
        sample_background_set(pool, n=len(pool) + 1, seed=42)


def test_background_sets_are_prefix_nested(synthetic_data, ids20):
    """The pipeline draws once at the max dilution level per (fold, repeat)
    and slices prefixes of that single draw for smaller b — it never calls
    sample_background_set again per b (numpy's replace=False sampling isn't
    prefix-stable across independent calls at different sizes, even with the
    same seed, so re-drawing per b would silently break the nesting
    guarantee)."""
    embeddings_list, labels, D = synthetic_data
    pool = build_background_pool(embeddings_list, labels, ids20)
    seed = 42

    bg_full, manifest_full = sample_background_set(pool, n=30, seed=seed)
    for b in (10, 20, 30):
        bg_b = bg_full[:b]
        manifest_b = manifest_full[:b]
        assert bg_b.shape[0] == b
        assert [row["background_index"] for row in manifest_b] == list(range(b))


def test_dilute_bag_b0_returns_same_object(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    original = embeddings_list[0]
    background = np.zeros((10, D), dtype=np.float32)
    diluted = dilute_bag(original, background, 0)
    assert diluted is original


def test_dilute_bag_appends_and_preserves_original(synthetic_data, ids20):
    embeddings_list, labels, D = synthetic_data
    original = embeddings_list[0].copy()
    rng = np.random.default_rng(0)
    background = rng.standard_normal((10, D)).astype(np.float32)

    diluted = dilute_bag(original, background, 5)

    assert diluted.shape[0] == original.shape[0] + 5
    np.testing.assert_array_equal(diluted[: original.shape[0]], original)
    np.testing.assert_array_equal(diluted[original.shape[0] :], background[:5])
    # original array itself must be untouched
    np.testing.assert_array_equal(embeddings_list[0], original)


def test_spawn_scan_seeds_reproducible():
    seeds1 = spawn_scan_seeds(42, 5)
    seeds2 = spawn_scan_seeds(42, 5)
    draws1 = [np.random.default_rng(s).integers(0, 10_000, 4) for s in seeds1]
    draws2 = [np.random.default_rng(s).integers(0, 10_000, 4) for s in seeds2]
    for d1, d2 in zip(draws1, draws2):
        np.testing.assert_array_equal(d1, d2)


def test_spawn_scan_seeds_gives_independent_draws_per_scan():
    seeds = spawn_scan_seeds(42, 5)
    draws = [tuple(np.random.default_rng(s).integers(0, 10_000, 8)) for s in seeds]
    assert len(set(draws)) == len(draws), "different scans must get independent background draws"


def test_scan_backgrounds_are_independent_not_shared(synthetic_data, ids20):
    """Two test scans in the same (fold, repeat) must draw different
    background sets — this is the behavior the per-scan redesign requires,
    replacing the old shared-background-per-fold/repeat design."""
    embeddings_list, labels, D = synthetic_data
    pool = build_background_pool(embeddings_list, labels, ids20)

    scan_seeds = spawn_scan_seeds(seed=42, n_scans=3)
    draws = [sample_background_set(pool, n=10, seed=s)[1] for s in scan_seeds]
    keyed = [
        tuple((row["source_scan_id"], row["source_slice_index"]) for row in manifest)
        for manifest in draws
    ]
    assert len(set(keyed)) == len(keyed), "each scan's background draw should be independent"


def test_logit_clips_and_inverts_sigmoid():
    p = np.array([0.0, 0.5, 1.0])
    l = logit(p, eps=1e-7)
    assert np.all(np.isfinite(l))
    assert l[1] == pytest.approx(0.0, abs=1e-9)
    assert l[0] < l[1] < l[2]


def test_pooled_metrics_matches_compute_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    proba = rng.dirichlet([1.0, 1.0], size=100).astype(np.float32)

    got = pooled_metrics(y, proba, n_bootstraps=0)
    expected = compute_metrics(y, proba, n_bootstraps=0)

    assert got["roc_auc"] == pytest.approx(expected["roc_auc"])
    assert got["pr_auc"] == pytest.approx(expected["pr_auc"])
    assert got["balanced_accuracy"] == pytest.approx(expected["balanced_accuracy"])
