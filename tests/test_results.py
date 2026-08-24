"""Tests for Module 6: results/ persistence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sliceheads.results import save_results
from sliceheads.constants import IMPORTANCE_FILENAME


def _make_fake_metrics():
    return {
        "accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": 0.74,
        "weighted_f1": 0.78, "roc_auc": 0.88, "pr_auc": 0.82,
        "sensitivity": 0.85, "specificity": 0.70,
        "confusion_matrix": [[10, 2], [3, 15]],
        "operating_point": 0.5, "n_bootstraps": 200, "ci": 0.95, "n_samples": 30,
        "roc_auc_ci_lower": 0.82, "roc_auc_ci_upper": 0.93,
        "pr_auc_ci_lower": 0.76, "pr_auc_ci_upper": 0.89,
        "f1_class_0": 0.71, "f1_class_1": 0.77,
    }


def test_save_creates_required_files(tmp_path):
    metrics = {"MeanPool": _make_fake_metrics()}
    preds = {"MeanPool": {"s0": {"true_label": 0, "pred_label": 0, "prob_class_0": 0.7, "prob_class_1": 0.3}}}
    hparams = {"MeanPool": {"C": 1.0}}
    save_results(tmp_path, metrics_per_head=metrics, predictions_per_head=preds,
                 hyperparameters_per_head=hparams, config={"experiment": {"name": "test"}})

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "hyperparameters.json").exists()
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "config_used.yaml").exists()


def test_metrics_csv_head_column(tmp_path):
    metrics = {"HeadA": _make_fake_metrics(), "HeadB": _make_fake_metrics()}
    preds = {
        "HeadA": {"s0": {"true_label": 0, "pred_label": 0, "prob_class_0": 0.8, "prob_class_1": 0.2}},
        "HeadB": {"s0": {"true_label": 0, "pred_label": 0, "prob_class_0": 0.6, "prob_class_1": 0.4}},
    }
    save_results(tmp_path, metrics_per_head=metrics, predictions_per_head=preds,
                 hyperparameters_per_head={})
    df = pd.read_csv(tmp_path / "metrics.csv")
    assert set(df["head_name"]) == {"HeadA", "HeadB"}


def test_importance_h5_filename(tmp_path):
    imp = {"MeanPool": {"s0": {
        "importance": np.array([0.3, 0.7], dtype=np.float32),
        "signed_importance": np.array([-0.1, 0.2], dtype=np.float32),
        "native_attention": None,
    }}}
    save_results(tmp_path, metrics_per_head={}, predictions_per_head={},
                 hyperparameters_per_head={}, importance_per_head=imp)
    assert (tmp_path / IMPORTANCE_FILENAME).exists()
    assert not (tmp_path / "results_importance.h5").exists()


def test_run_metadata_has_version(tmp_path):
    save_results(tmp_path, metrics_per_head={}, predictions_per_head={},
                 hyperparameters_per_head={})
    with open(tmp_path / "run_metadata.json") as f:
        meta = json.load(f)
    assert "sliceheads_version" in meta
    assert "python_version" in meta
