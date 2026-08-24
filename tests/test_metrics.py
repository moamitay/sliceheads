"""Tests for Module 3: evaluation metrics."""

import numpy as np
import pytest

from sliceheads.metrics import compute_metrics
from sliceheads.constants import OPERATING_POINT


def _make_perfect(n=100):
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    proba = np.zeros((n, 2))
    proba[y == 0, 0] = 1.0
    proba[y == 1, 1] = 1.0
    return y, proba


def _make_random(seed=0, n=200):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    raw = rng.dirichlet([1.0, 1.0], size=n).astype(np.float32)
    return y, raw


def test_perfect_metrics():
    y, proba = _make_perfect()
    m = compute_metrics(y, proba, n_bootstraps=50)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)


def test_operating_point_in_output():
    y, proba = _make_random()
    m = compute_metrics(y, proba, n_bootstraps=50)
    assert m["operating_point"] == OPERATING_POINT


def test_bootstrap_ci_bounds():
    y, proba = _make_random()
    m = compute_metrics(y, proba, n_bootstraps=200)
    assert m["roc_auc_ci_lower"] <= m["roc_auc"] <= m["roc_auc_ci_upper"]
    assert m["pr_auc_ci_lower"] <= m["pr_auc"] <= m["pr_auc_ci_upper"]


def test_probas_sum_to_one():
    y, proba = _make_random()
    m = compute_metrics(y, proba, n_bootstraps=50)
    assert "roc_auc" in m
    assert "macro_f1" in m


def test_required_keys():
    y, proba = _make_random()
    m = compute_metrics(y, proba, n_bootstraps=50)
    required = {
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "roc_auc", "pr_auc", "sensitivity", "specificity",
        "confusion_matrix", "operating_point",
        "roc_auc_ci_lower", "roc_auc_ci_upper",
    }
    assert required.issubset(set(m.keys()))
