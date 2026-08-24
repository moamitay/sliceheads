"""Adapter-mask input adapters for InceptionTime and MultiRocket."""

from __future__ import annotations

import numpy as np


def apply_native_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out padded positions in-place (used for debugging; not for training)."""
    out = X.copy()
    out[mask == 0] = 0.0
    return out


def inception_time_adapter(
    X: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Prepare [B, D, N] tensor for InceptionTime.

    Pads within the batch to the max real-slice count, then transposes.
    Returns float32 numpy array [B, D, N_max].
    """
    B = X.shape[0]
    seq_lens = mask.sum(axis=1).astype(int)  # [B]
    N_max = int(seq_lens.max())
    D = X.shape[2]

    out = np.zeros((B, D, N_max), dtype=np.float32)
    for i in range(B):
        n = seq_lens[i]
        # X[i, :n, :] are the real slices
        out[i, :, :n] = X[i, :n, :].T
    return out


def multirocket_adapter(
    X: np.ndarray, mask: np.ndarray
) -> list[np.ndarray]:
    """Strip padding per sample and transpose to [D, N_i] for aeon.

    Returns a list of length B, each element [D, N_i].
    """
    B = X.shape[0]
    result = []
    for i in range(B):
        n = int(mask[i].sum())
        result.append(X[i, :n, :].T.astype(np.float32))
    return result
