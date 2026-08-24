"""Module 1 (storage half): HDF5 read/write and validation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sliceheads.constants import SCHEMA_VERSION, SLICEHEADS_VERSION


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_h5(h5_path: str | Path) -> None:
    """Validate an HDF5 embeddings file against the sliceheads schema.

    Raises
    ------
    ValueError
        If any hard constraint is violated (missing datasets, wrong shapes,
        non-binary labels, patient-id leakage, schema mismatch).
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        sample_keys = list(f.keys())
        if not sample_keys:
            raise ValueError("HDF5 file contains no samples.")

        embedding_dim: int | None = f.attrs.get("embedding_dim", None)

        labels_seen: dict[str, set] = {"train": set(), "val": set(), "test": set()}
        patients_seen: dict[str, set] = {"train": set(), "val": set(), "test": set()}
        all_labels: list[int] = []

        for key in sample_keys:
            grp = f[key]

            # --- embeddings ---
            if "embeddings" not in grp:
                raise ValueError(f"Sample '{key}' is missing 'embeddings' dataset.")
            emb = grp["embeddings"]
            if emb.ndim != 2:
                raise ValueError(
                    f"Sample '{key}': embeddings must be 2-D [N, D], got shape {emb.shape}."
                )
            n_slices, d = emb.shape
            if embedding_dim is not None and d != embedding_dim:
                raise ValueError(
                    f"Sample '{key}': embedding dim {d} != file-level embedding_dim {embedding_dim}."
                )

            # --- label ---
            if "label" not in grp.attrs:
                raise ValueError(f"Sample '{key}' is missing 'label' attribute.")
            label = int(grp.attrs["label"])
            all_labels.append(label)

            # --- split ---
            if "split" not in grp.attrs:
                raise ValueError(f"Sample '{key}' is missing 'split' attribute.")
            split = str(grp.attrs["split"])

            if split in labels_seen:
                labels_seen[split].add(label)

            # --- patient_id (optional, warn if absent) ---
            patient_id = grp.attrs.get("patient_id", None)
            if patient_id is None:
                warnings.warn(
                    f"Sample '{key}' has no 'patient_id' attribute — leakage check skipped for this sample.",
                    stacklevel=2,
                )
            else:
                pid = str(patient_id)
                if split in patients_seen:
                    patients_seen[split].add(pid)

            # --- important_slices (optional dataset) ---
            if "important_slices" in grp:
                imp = grp["important_slices"]
                if imp.shape not in ((), (0,)) and imp.shape[0] not in (0, n_slices):
                    raise ValueError(
                        f"Sample '{key}': important_slices has length {imp.shape[0]} "
                        f"but embeddings has {n_slices} slices."
                    )

        # --- binary-only invariant ---
        distinct = set(all_labels)
        if distinct != {0, 1}:
            raise ValueError(
                f"sliceheads v0.1 is binary-only. Expected label set {{0, 1}}, "
                f"found {len(distinct)} distinct class(es): {sorted(distinct)}."
            )

        # --- active splits exist ---
        active = {split for key in sample_keys for split in [str(f[key].attrs.get("split", ""))]}
        for required in ("train", "test"):
            if required not in active:
                raise ValueError(f"No samples with split='{required}' found in the file.")

        # --- patient_id leakage check ---
        all_have_pid = all(
            f[k].attrs.get("patient_id") is not None for k in sample_keys
        )
        if all_have_pid:
            for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
                overlap = patients_seen[s1] & patients_seen[s2]
                if overlap:
                    raise ValueError(
                        f"patient_id leakage between '{s1}' and '{s2}': {sorted(overlap)[:5]} …"
                    )


# ---------------------------------------------------------------------------
# EmbedStore: read/write interface
# ---------------------------------------------------------------------------

class EmbedStore:
    """Thin wrapper around an HDF5 embeddings file."""

    def __init__(self, h5_path: str | Path) -> None:
        self.h5_path = Path(h5_path)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        h5_path: str | Path,
        *,
        backbone_name: str,
        backbone_source: str,
        embedding_dim: int,
        dataset_name: str = "",
        hu_window_min: float = -1000.0,
        hu_window_max: float = 400.0,
        slice_axis: str = "z",
        feature_type: str = "cls_token",
        dataset_version: str = "",
        dataset_license: str = "",
        backbone_revision: str = "",
        orientation: str = "RAS",
    ) -> "EmbedStore":
        """Create a new HDF5 file with file-level metadata."""
        from datetime import datetime, timezone

        h5_path = Path(h5_path)
        with h5py.File(h5_path, "w") as f:
            f.attrs["sliceheads_version"] = SLICEHEADS_VERSION
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
            f.attrs["backbone_name"] = backbone_name
            f.attrs["backbone_source"] = backbone_source
            f.attrs["backbone_revision"] = backbone_revision
            f.attrs["embedding_dim"] = embedding_dim
            f.attrs["feature_type"] = feature_type
            f.attrs["hu_window_min"] = hu_window_min
            f.attrs["hu_window_max"] = hu_window_max
            f.attrs["slice_axis"] = slice_axis
            f.attrs["orientation"] = orientation
            f.attrs["dataset_name"] = dataset_name
            f.attrs["dataset_version"] = dataset_version
            f.attrs["dataset_license"] = dataset_license
        return cls(h5_path)

    def write_sample(
        self,
        sample_id: str,
        *,
        embeddings: np.ndarray,
        label: int,
        split: str,
        patient_id: str | None = None,
        original_shape: tuple[int, int, int] | None = None,
        slice_spacing_mm: tuple[float, float, float] | None = None,
        scanner_manufacturer: str | None = None,
        scanner_model: str | None = None,
        important_slices: np.ndarray | None = None,
    ) -> None:
        """Append or overwrite a single sample in the HDF5 file."""
        with h5py.File(self.h5_path, "a") as f:
            if sample_id in f:
                del f[sample_id]
            grp = f.create_group(sample_id)
            grp.create_dataset("embeddings", data=embeddings.astype(np.float32))
            grp.attrs["label"] = int(label)
            grp.attrs["split"] = str(split)
            if patient_id is not None:
                grp.attrs["patient_id"] = str(patient_id)
            if original_shape is not None:
                grp.attrs["original_shape_h"] = original_shape[0]
                grp.attrs["original_shape_w"] = original_shape[1]
                grp.attrs["original_shape_z"] = original_shape[2]
            if slice_spacing_mm is not None:
                grp.attrs["slice_spacing_mm_h"] = slice_spacing_mm[0]
                grp.attrs["slice_spacing_mm_w"] = slice_spacing_mm[1]
                grp.attrs["slice_spacing_mm_z"] = slice_spacing_mm[2]
            if scanner_manufacturer is not None:
                grp.attrs["scanner_manufacturer"] = scanner_manufacturer
            if scanner_model is not None:
                grp.attrs["scanner_model"] = scanner_model
            if important_slices is not None:
                grp.create_dataset(
                    "important_slices", data=important_slices.astype(np.uint8)
                )
            else:
                grp.create_dataset(
                    "important_slices", data=np.array([], dtype=np.uint8)
                )

    def write_mean_embedding(self, mean_embedding: np.ndarray) -> None:
        """Store the training-set mean embedding as a root-level attribute."""
        with h5py.File(self.h5_path, "a") as f:
            f.attrs["mean_embedding"] = mean_embedding.astype(np.float32)

    def compute_and_write_mean_embedding(self) -> np.ndarray:
        """Compute mean embedding from train samples and persist it."""
        with h5py.File(self.h5_path, "r") as f:
            train_embs = []
            for key in f.keys():
                if str(f[key].attrs.get("split", "")) == "train":
                    train_embs.append(f[key]["embeddings"][:])
        all_embs = np.concatenate(train_embs, axis=0)  # [total_slices, D]
        mean_emb = all_embs.mean(axis=0)
        self.write_mean_embedding(mean_emb)
        return mean_emb

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def sample_ids(self) -> list[str]:
        with h5py.File(self.h5_path, "r") as f:
            return list(f.keys())

    def load_split(
        self, split: str
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        """Return (embeddings_list, labels, sample_ids) for the given split.

        embeddings_list[i] has shape [N_i, D] (unpadded).
        """
        embeddings_list: list[np.ndarray] = []
        labels: list[int] = []
        ids: list[str] = []
        with h5py.File(self.h5_path, "r") as f:
            for key in f.keys():
                grp = f[key]
                if str(grp.attrs.get("split", "")) == split:
                    embeddings_list.append(grp["embeddings"][:].astype(np.float32))
                    labels.append(int(grp.attrs["label"]))
                    ids.append(key)
        return embeddings_list, np.array(labels, dtype=np.int64), ids

    def load_sample(self, sample_id: str) -> dict[str, Any]:
        """Load a single sample as a dict."""
        with h5py.File(self.h5_path, "r") as f:
            grp = f[sample_id]
            result: dict[str, Any] = {
                "embeddings": grp["embeddings"][:].astype(np.float32),
                "label": int(grp.attrs["label"]),
                "split": str(grp.attrs["split"]),
            }
            for key in ("patient_id", "scanner_manufacturer", "scanner_model"):
                if key in grp.attrs:
                    result[key] = grp.attrs[key]
            if "important_slices" in grp:
                result["important_slices"] = grp["important_slices"][:].astype(np.uint8)
            else:
                result["important_slices"] = np.array([], dtype=np.uint8)
            return result

    def file_attrs(self) -> dict[str, Any]:
        with h5py.File(self.h5_path, "r") as f:
            return dict(f.attrs)

    def mean_embedding(self) -> np.ndarray | None:
        with h5py.File(self.h5_path, "r") as f:
            val = f.attrs.get("mean_embedding", None)
            return np.array(val, dtype=np.float32) if val is not None else None
