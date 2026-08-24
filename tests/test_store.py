"""Tests for EmbedStore and validate_h5 (Module 1)."""

import numpy as np
import pytest
import h5py

from sliceheads.store import EmbedStore, validate_h5


def _make_store(tmp_path, n_samples=6):
    """Helper: create a minimal valid HDF5 file."""
    h5 = tmp_path / "test.h5"
    store = EmbedStore.create(
        h5, backbone_name="test_enc", backbone_source="local", embedding_dim=8, dataset_name="test"
    )
    rng = np.random.default_rng(0)
    splits = ["train", "train", "train", "val", "test", "test"]
    for i in range(n_samples):
        emb = rng.standard_normal((5 + i, 8)).astype(np.float32)
        store.write_sample(
            f"s{i:03d}",
            embeddings=emb,
            label=i % 2,
            split=splits[i],
            patient_id=f"p{i:03d}",
        )
    return store, h5


def test_roundtrip(tmp_path):
    store, h5 = _make_store(tmp_path)
    ids = store.sample_ids()
    assert len(ids) == 6
    sample = store.load_sample("s000")
    assert sample["label"] in (0, 1)
    assert sample["embeddings"].shape[1] == 8


def test_validate_passes(tmp_path):
    _, h5 = _make_store(tmp_path)
    validate_h5(h5)  # should not raise


def test_validate_missing_label(tmp_path):
    _, h5 = _make_store(tmp_path)
    with h5py.File(h5, "a") as f:
        del f["s000"].attrs["label"]
    with pytest.raises(ValueError, match="missing 'label'"):
        validate_h5(h5)


def test_validate_non_binary(tmp_path):
    h5 = tmp_path / "nb.h5"
    store = EmbedStore.create(h5, backbone_name="x", backbone_source="x", embedding_dim=4)
    rng = np.random.default_rng(1)
    for i, (lbl, sp) in enumerate([(0, "train"), (1, "train"), (2, "test")]):
        store.write_sample(f"s{i}", embeddings=rng.standard_normal((3, 4)).astype(np.float32),
                           label=lbl, split=sp)
    with pytest.raises(ValueError, match="binary-only"):
        validate_h5(h5)


def test_validate_important_slices_length_mismatch(tmp_path):
    h5 = tmp_path / "imp.h5"
    store = EmbedStore.create(h5, backbone_name="x", backbone_source="x", embedding_dim=4)
    rng = np.random.default_rng(2)
    store.write_sample(
        "s0", embeddings=rng.standard_normal((5, 4)).astype(np.float32),
        label=0, split="train",
        important_slices=np.array([1, 0, 1], dtype=np.uint8),  # wrong length
    )
    store.write_sample(
        "s1", embeddings=rng.standard_normal((5, 4)).astype(np.float32),
        label=1, split="test",
    )
    with pytest.raises(ValueError, match="important_slices"):
        validate_h5(h5)


def test_mean_embedding(tmp_path):
    store, h5 = _make_store(tmp_path)
    mean = store.compute_and_write_mean_embedding()
    assert mean.shape == (8,)
    assert store.mean_embedding() is not None


def test_load_split(tmp_path):
    store, h5 = _make_store(tmp_path)
    emb_list, labels, ids = store.load_split("train")
    assert len(emb_list) == 3
    assert set(labels).issubset({0, 1})
