"""Fixed-pooling classification heads: Mean, Max, GeM + logistic regression."""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from sliceheads.heads.base import BaseHead


def _masked_mean(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise masked mean. X: [B, N, D], mask: [B, N] -> [B, D]."""
    mask_f = mask.astype(np.float32)[:, :, None]  # [B, N, 1]
    return (X * mask_f).sum(axis=1) / mask_f.sum(axis=1).clip(min=1e-9)


def _masked_max(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise masked max. X: [B, N, D], mask: [B, N] -> [B, D]."""
    mask_bool = mask.astype(bool)
    out = np.full((X.shape[0], X.shape[2]), -np.inf, dtype=np.float32)
    for i in range(X.shape[0]):
        valid = X[i][mask_bool[i]]
        if valid.shape[0] > 0:
            out[i] = valid.max(axis=0)
        else:
            out[i] = 0.0
    return out


def _masked_gem(X: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    """Row-wise masked generalized mean pooling. X: [B, N, D] -> [B, D]."""
    eps = 1e-6
    X_clamped = np.maximum(X, eps)
    mask_f = mask.astype(np.float32)[:, :, None]
    powered = X_clamped ** p
    pooled = (powered * mask_f).sum(axis=1) / mask_f.sum(axis=1).clip(min=1e-9)
    return pooled ** (1.0 / p)


def _make_lr(C: float, class_weight: str | None, random_seed: int) -> LogisticRegression:
    cw = "balanced" if class_weight == "balanced" else None
    return LogisticRegression(
        C=C,
        class_weight=cw,
        max_iter=1000,
        random_state=random_seed,
        solver="lbfgs",
    )


class MeanPoolClassifier(BaseHead):
    """Masked mean pooling followed by logistic regression."""

    input_policy = "native_mask"

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        random_seed: int = 42,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.random_seed = random_seed

    def _pool(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _masked_mean(X, mask)

    def _fit_padded(self, X, y, mask, *args) -> None:
        feats = self._pool(X, mask)
        self.clf_ = _make_lr(self.C, self.class_weight, self.random_seed)
        self.clf_.fit(feats, y)

    def _predict_proba_padded(self, X, mask):
        feats = self._pool(X, mask)
        return self.clf_.predict_proba(feats).astype(np.float32)

    def _save_backend(self, path: str) -> None:
        joblib.dump(self.clf_, os.path.join(path, "model.joblib"))

    def _load_backend(self, path: str) -> None:
        self.clf_ = joblib.load(os.path.join(path, "model.joblib"))


class MaxPoolClassifier(BaseHead):
    """Masked max pooling followed by logistic regression."""

    input_policy = "native_mask"

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        random_seed: int = 42,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.random_seed = random_seed

    def _pool(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _masked_max(X, mask)

    def _fit_padded(self, X, y, mask, *args) -> None:
        feats = self._pool(X, mask)
        self.clf_ = _make_lr(self.C, self.class_weight, self.random_seed)
        self.clf_.fit(feats, y)

    def _predict_proba_padded(self, X, mask):
        feats = self._pool(X, mask)
        return self.clf_.predict_proba(feats).astype(np.float32)

    def _save_backend(self, path: str) -> None:
        joblib.dump(self.clf_, os.path.join(path, "model.joblib"))

    def _load_backend(self, path: str) -> None:
        self.clf_ = joblib.load(os.path.join(path, "model.joblib"))


class GeMPoolClassifier(BaseHead):
    """Masked generalized mean (GeM) pooling followed by logistic regression."""

    input_policy = "native_mask"

    def __init__(
        self,
        p: float = 3.0,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        random_seed: int = 42,
    ) -> None:
        self.p = p
        self.C = C
        self.class_weight = class_weight
        self.random_seed = random_seed

    def _pool(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _masked_gem(X, mask, self.p)

    def _fit_padded(self, X, y, mask, *args) -> None:
        feats = self._pool(X, mask)
        self.clf_ = _make_lr(self.C, self.class_weight, self.random_seed)
        self.clf_.fit(feats, y)

    def _predict_proba_padded(self, X, mask):
        feats = self._pool(X, mask)
        return self.clf_.predict_proba(feats).astype(np.float32)

    def _save_backend(self, path: str) -> None:
        joblib.dump(self.clf_, os.path.join(path, "model.joblib"))

    def _load_backend(self, path: str) -> None:
        self.clf_ = joblib.load(os.path.join(path, "model.joblib"))
