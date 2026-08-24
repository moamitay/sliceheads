"""BaseHead: abstract interface that every sliceheads classifier must implement."""

from __future__ import annotations

import importlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from sliceheads.constants import SCHEMA_VERSION, SLICEHEADS_VERSION

InputPolicy = Literal["native_mask", "adapter_mask"]


def _pad_batch(embeddings_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of ragged [N_i, D] arrays to [B, N_max, D] + binary mask [B, N_max].

    Internal to the heads package: every head pads/masks inside its own fit /
    predict_proba, so no caller outside `sliceheads.heads` should need this.
    """
    B = len(embeddings_list)
    N_max = max(e.shape[0] for e in embeddings_list)
    D = embeddings_list[0].shape[1]
    X = np.zeros((B, N_max, D), dtype=np.float32)
    mask = np.zeros((B, N_max), dtype=np.float32)
    for i, e in enumerate(embeddings_list):
        n = e.shape[0]
        X[i, :n, :] = e
        mask[i, :n] = 1.0
    return X, mask


class BaseHead(ClassifierMixin, BaseEstimator, ABC):
    """Abstract base for all sliceheads classification heads.

    Public, sklearn-compatible contract — works with GridSearchCV,
    RandomizedSearchCV, cross_val_score, and clone() with no wrapper:

        X: a sequence (list or object-dtype ndarray) of length B, each element
           a ragged [N_i, D] float array — one bag of slice embeddings per scan.
        y: [B] binary labels {0, 1}.

    `fit(X, y)` / `predict_proba(X)` / `predict(X)` accept and return plain
    numpy arrays; padding to a dense [B, N_max, D] tensor + [B, N_max] mask
    happens internally via `_pad_batch` and is never visible to the caller.
    Heads that early-stop on a validation loss (`uses_validation_split =
    True`) carve a stratified validation split out of the given `X, y`
    themselves — the caller never passes a separate validation set.

    `input_policy` is an implementation detail, not part of the public
    contract: "native_mask" heads consume the padding mask directly inside
    their forward pass (masked attention/pooling); "adapter_mask" heads strip
    padding via an adapter before a backend that expects an unpadded /
    fixed-length input (e.g. the aeon ROCKET transform).
    """

    input_policy: InputPolicy = "native_mask"
    supports_native_attention: bool = False
    uses_validation_split: bool = False
    validation_fraction: float = 0.15

    # ------------------------------------------------------------------
    # Public sklearn-facing API — the only place padding/masking happens
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "BaseHead":
        """Train the head. X is ragged per-sample embeddings; y is binary labels."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        X = list(X)
        if self.uses_validation_split:
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y,
                    test_size=self.validation_fraction,
                    stratify=y,
                    random_state=self.random_seed,
                )
            except ValueError:
                # Too few samples per class for a stratified split (e.g. a
                # small inner CV fold) — train on everything, no early stopping.
                X_tr, y_tr, X_val, y_val = X, y, None, None

            X_tr_pad, mask_tr = _pad_batch(X_tr)
            if X_val is not None:
                X_val_pad, mask_val = _pad_batch(X_val)
            else:
                X_val_pad, mask_val = None, None
            self._fit_padded(X_tr_pad, y_tr, mask_tr, X_val_pad, y_val, mask_val)
        else:
            X_pad, mask = _pad_batch(X)
            self._fit_padded(X_pad, y, mask)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities of shape [B, 2] for ragged input X."""
        X_pad, mask = _pad_batch(list(X))
        return self._predict_proba_padded(X_pad, mask)

    def predict(self, X) -> np.ndarray:
        """Hard labels via argmax over predict_proba (0.5 operating point)."""
        return self.predict_proba(X).argmax(axis=1)

    def native_attention(self, X) -> np.ndarray | None:
        """Per-slice native attention weights for ragged input X, or None if unsupported."""
        if not self.supports_native_attention:
            return None
        X_pad, mask = _pad_batch(list(X))
        return self._native_attention_padded(X_pad, mask)

    # ------------------------------------------------------------------
    # Backend hooks — subclasses implement these against padded tensors.
    # Heads with uses_validation_split = True receive the extra
    # (X_val, y_val, mask_val) positional args; heads without it don't.
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit_padded(self, X, y, mask, *args) -> None:
        """Train against a padded [B, N, D] tensor + [B, N] mask."""

    @abstractmethod
    def _predict_proba_padded(self, X, mask) -> np.ndarray:
        """Predict against a padded [B, N, D] tensor + [B, N] mask."""

    def _native_attention_padded(self, X, mask) -> np.ndarray | None:
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write a self-describing head directory with manifest.json + backend artifact."""
        os.makedirs(path, exist_ok=True)
        manifest = {
            "head_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "params": self.get_params(),
            "input_policy": self.input_policy,
            "sliceheads_version": SLICEHEADS_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        self._save_backend(path)

    @classmethod
    def load(cls, path: str) -> "BaseHead":
        """Load a head from a previously saved directory."""
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Head was saved under schema_version {manifest['schema_version']}, "
                f"but this build expects {SCHEMA_VERSION}. Refusing to load."
            )
        module_name, class_name = manifest["head_class"].rsplit(".", 1)
        mod = importlib.import_module(module_name)
        head_cls = getattr(mod, class_name)
        head = head_cls(**manifest["params"])
        head.input_policy = manifest["input_policy"]
        head._load_backend(path)
        return head

    @abstractmethod
    def _save_backend(self, path: str) -> None:
        """Write the fitted artifact into path (joblib / .pt / aeon-native)."""

    @abstractmethod
    def _load_backend(self, path: str) -> None:
        """Restore the fitted artifact written by _save_backend."""
