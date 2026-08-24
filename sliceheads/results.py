"""Module 6: results/ directory persistence."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from sliceheads.constants import IMPORTANCE_FILENAME, SLICEHEADS_VERSION, SCHEMA_VERSION


def _run_metadata() -> dict[str, Any]:
    """Collect software/hardware environment info."""
    return {
        "sliceheads_version": SLICEHEADS_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_results(
    results_dir: str | Path,
    *,
    metrics_per_head: dict[str, dict],
    predictions_per_head: dict[str, dict],
    hyperparameters_per_head: dict[str, Any],
    importance_per_head: dict[str, dict[str, dict]] | None = None,
    config: dict | None = None,
    sample_metadata: dict[str, dict] | None = None,
) -> None:
    """Write the full results/ directory.

    Parameters
    ----------
    results_dir : destination directory (created if needed).
    metrics_per_head : {head_name: metrics_dict} from compute_metrics().
    predictions_per_head : {head_name: {sample_id: {'true_label', 'pred_label',
                             'prob_class_0', 'prob_class_1', 'patient_id'}}}.
    hyperparameters_per_head : {head_name: best_params_dict}.
    importance_per_head : {head_name: {sample_id: {'importance', 'signed_importance',
                             'native_attention' (optional)}}}.
    config : the experiment YAML config dict.
    sample_metadata : {sample_id: {'patient_id', ...}} for test_predictions.csv.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    # metrics.csv
    rows = []
    for head_name, m in metrics_per_head.items():
        row = {"head_name": head_name}
        row.update({k: v for k, v in m.items() if not isinstance(v, list)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    # test_predictions.csv
    pred_rows = []
    for head_name, preds in predictions_per_head.items():
        for sample_id, p in preds.items():
            meta = (sample_metadata or {}).get(sample_id, {})
            pred_rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": meta.get("patient_id", ""),
                    "true_label": p["true_label"],
                    "pred_label": p["pred_label"],
                    "prob_class_0": p["prob_class_0"],
                    "prob_class_1": p["prob_class_1"],
                    "head_name": head_name,
                }
            )
    pd.DataFrame(pred_rows).to_csv(out / "test_predictions.csv", index=False)

    # hyperparameters.json
    with open(out / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters_per_head, f, indent=2, default=str)

    # importance.h5
    if importance_per_head:
        imp_path = out / IMPORTANCE_FILENAME
        with h5py.File(imp_path, "w") as f:
            for head_name, samples in importance_per_head.items():
                head_grp = f.require_group(head_name)
                for sample_id, arrays in samples.items():
                    samp_grp = head_grp.require_group(sample_id)
                    samp_grp.create_dataset(
                        "importance",
                        data=np.asarray(arrays["importance"], dtype=np.float32),
                    )
                    samp_grp.create_dataset(
                        "signed_importance",
                        data=np.asarray(arrays["signed_importance"], dtype=np.float32),
                    )
                    if "native_attention" in arrays and arrays["native_attention"] is not None:
                        samp_grp.create_dataset(
                            "native_attention",
                            data=np.asarray(arrays["native_attention"], dtype=np.float32),
                        )

    # config_used.yaml
    if config is not None:
        with open(out / "config_used.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    # run_metadata.json
    with open(out / "run_metadata.json", "w") as f:
        json.dump(_run_metadata(), f, indent=2)
