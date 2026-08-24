"""Shared synthetic fixtures for sliceheads tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng):
    """Small synthetic dataset: 20 samples, varied N, D=16."""
    D = 16
    n_slices_list = [5, 8, 12, 6, 10, 7, 9, 11, 5, 8, 6, 12, 10, 7, 9, 11, 8, 6, 10, 5]
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    embeddings_list = [rng.standard_normal((n, D)).astype(np.float32) for n in n_slices_list]
    return embeddings_list, labels, D


@pytest.fixture
def padded_batch(synthetic_data):
    """Padded [B, N_max, D] + mask [B, N_max] from synthetic data."""
    embeddings_list, labels, D = synthetic_data
    B = len(embeddings_list)
    N_max = max(e.shape[0] for e in embeddings_list)
    X = np.zeros((B, N_max, D), dtype=np.float32)
    mask = np.zeros((B, N_max), dtype=np.float32)
    for i, e in enumerate(embeddings_list):
        n = e.shape[0]
        X[i, :n, :] = e
        mask[i, :n] = 1.0
    return X, mask, labels


@pytest.fixture
def train_val_split(synthetic_data):
    """Split synthetic data into 12 train / 8 val."""
    embeddings_list, labels, D = synthetic_data
    train_emb = embeddings_list[:12]
    val_emb = embeddings_list[12:]
    train_labels = labels[:12]
    val_labels = labels[12:]

    def pad(lst):
        N_max = max(e.shape[0] for e in lst)
        D_ = lst[0].shape[1]
        X = np.zeros((len(lst), N_max, D_), dtype=np.float32)
        m = np.zeros((len(lst), N_max), dtype=np.float32)
        for i, e in enumerate(lst):
            n = e.shape[0]
            X[i, :n, :] = e
            m[i, :n] = 1.0
        return X, m

    Xt, mt = pad(train_emb)
    Xv, mv = pad(val_emb)
    return Xt, mt, train_labels, Xv, mv, val_labels
