"""Tests for Module 5: LOO importance and localization metrics."""

import numpy as np
import pytest

from sliceheads.explain import loo_importance, localization_metrics
from sliceheads.heads.pooling import MeanPoolClassifier


def _fit_head(D=16, n_train=20, seed=0):
    rng = np.random.default_rng(seed)
    N_list = [5, 8, 6, 10, 7, 9, 5, 8, 6, 10, 7, 9, 5, 8, 6, 10, 7, 9, 5, 8]
    labels = np.array([i % 2 for i in range(n_train)], dtype=np.int64)
    embs = [rng.standard_normal((n, D)).astype(np.float32) for n in N_list]
    head = MeanPoolClassifier(C=0.1)
    head.fit(embs, labels)
    return head, embs[0], np.ones(embs[0].shape[0], dtype=np.float32), rng.standard_normal(D).astype(np.float32)


def test_loo_importance_sums_to_one():
    head, emb, mask, mean_emb = _fit_head()
    result = loo_importance(head, emb, mask, mean_emb)
    assert result["importance"].shape == (emb.shape[0],)
    assert abs(result["importance"].sum() - 1.0) < 1e-5


def test_loo_signed_importance_dtype():
    head, emb, mask, mean_emb = _fit_head()
    result = loo_importance(head, emb, mask, mean_emb)
    assert result["signed_importance"].dtype == np.float32
    assert result["importance"].dtype == np.float32


def test_localization_empty_annotation():
    importance = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    annotation = np.array([], dtype=np.uint8)
    result = localization_metrics(importance, annotation)
    assert result == {}


def test_localization_with_annotation():
    importance = np.array([0.1, 0.5, 0.2, 0.7, 0.4], dtype=np.float32)
    annotation = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    result = localization_metrics(importance, annotation, top_fraction=0.4)
    assert "spearman_correlation" in result
    assert "top_40_pct_recall" in result
    assert 0.0 <= result["top_40_pct_recall"] <= 1.0
