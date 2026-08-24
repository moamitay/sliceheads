"""Module 4 (config half): YAML experiment configuration parsing and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_TOP_LEVEL = {"experiment", "data", "training", "heads"}
_VALID_STRATEGIES = {"grid", "random", "bayesian"}
_KNOWN_HEADS = {
    "MeanPoolClassifier",
    "MaxPoolClassifier",
    "GeMPoolClassifier",
    "ABMILClassifier",
    "GatedABMILClassifier",
    "DSMILClassifier",
    "TransformerMILClassifier",
    "MultiRocketClassifier",
    "InceptionTimeClassifier",
    "ALSTMFCNClassifier",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a sliceheads experiment YAML config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    missing = _REQUIRED_TOP_LEVEL - set(cfg)
    if missing:
        raise ValueError(f"Config is missing required top-level keys: {sorted(missing)}")

    strategy = cfg.get("search_strategy", "grid")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"search_strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'."
        )

    heads_cfg = cfg.get("heads", {})
    if not isinstance(heads_cfg, dict) or not heads_cfg:
        raise ValueError("Config 'heads' section must be a non-empty mapping.")

    for head_name, head_cfg in heads_cfg.items():
        if head_name not in _KNOWN_HEADS:
            raise ValueError(
                f"Unknown head '{head_name}'. Known heads: {sorted(_KNOWN_HEADS)}."
            )
        if not isinstance(head_cfg, dict):
            raise ValueError(f"Head config for '{head_name}' must be a mapping.")


def iter_grid(hyperparameters: dict[str, list]) -> list[dict[str, Any]]:
    """Return all combinations of hyperparameter values (cartesian product)."""
    import itertools

    keys = list(hyperparameters.keys())
    values = [hyperparameters[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def iter_random(hyperparameters: dict[str, list], n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Return n random hyperparameter combinations (with replacement)."""
    import random

    rng = random.Random(seed)
    keys = list(hyperparameters.keys())
    combos = []
    for _ in range(n):
        combos.append({k: rng.choice(hyperparameters[k]) for k in keys})
    return combos
