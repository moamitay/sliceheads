"""Module 1 (extraction half): NIfTI → 2D slices → encoder → HDF5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _preprocess_slice(
    slice_2d: np.ndarray,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    """Clip, rescale to [0,255], convert to uint8. Returns [H, W] uint8."""
    arr = np.clip(slice_2d, window_min, window_max)
    arr = (arr - window_min) / max(window_max - window_min, 1e-9)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def extract_volume_embeddings(
    nifti_path: str | Path,
    encoder_name_or_path: str,
    *,
    window_min: float = -1000.0,
    window_max: float = 400.0,
    slice_axis: int = 2,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Extract slice embeddings from a NIfTI volume using a HuggingFace encoder.

    Parameters
    ----------
    nifti_path : path to .nii or .nii.gz file.
    encoder_name_or_path : HuggingFace model ID or local directory.
    window_min / window_max : HU clipping window.
    slice_axis : axis to slice along (0=sagittal, 1=coronal, 2=axial).
    batch_size : slices per forward pass.
    device : 'cpu' or 'cuda'.

    Returns
    -------
    float32 [N, D] array of embeddings, one per slice.
    """
    import nibabel as nib
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image

    volume = nib.load(str(nifti_path))
    data = volume.get_fdata(dtype=np.float32)

    # Move the slice axis to position 0 for easy iteration
    data = np.moveaxis(data, slice_axis, 0)  # [N, H, W]
    N = data.shape[0]

    processor = AutoImageProcessor.from_pretrained(encoder_name_or_path)
    model = AutoModel.from_pretrained(encoder_name_or_path).to(device).eval()

    embeddings = []
    for start in range(0, N, batch_size):
        batch_slices = data[start : start + batch_size]  # [B, H, W]
        pil_images = []
        for sl in batch_slices:
            sl_u8 = _preprocess_slice(sl, window_min, window_max)
            # Convert to PIL; replicate to RGB if processor expects 3 channels
            img = Image.fromarray(sl_u8, mode="L")
            pil_images.append(img)

        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token or pooled output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0, :]
        embeddings.append(feats.cpu().float().numpy())

    return np.concatenate(embeddings, axis=0)  # [N, D]


def build_h5_from_nifti_dir(
    nifti_dir: str | Path,
    output_h5: str | Path,
    encoder_name_or_path: str,
    label_csv: str | Path,
    *,
    window_min: float = -1000.0,
    window_max: float = 400.0,
    slice_axis: int = 2,
    batch_size: int = 32,
    device: str = "cpu",
    backbone_name: str = "",
    dataset_name: str = "",
) -> None:
    """Build a sliceheads HDF5 file from a directory of NIfTI files + a CSV label file.

    CSV columns: sample_id, label (0 or 1), split (train|val|test),
                 patient_id (optional), nifti_filename (optional; defaults to sample_id.nii.gz).
    """
    import pandas as pd
    from sliceheads.store import EmbedStore

    df = pd.read_csv(label_csv)
    required = {"sample_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"label_csv is missing columns: {sorted(missing)}")

    nifti_dir = Path(nifti_dir)

    store = EmbedStore.create(
        output_h5,
        backbone_name=backbone_name or encoder_name_or_path,
        backbone_source=encoder_name_or_path,
        embedding_dim=0,  # will be updated after first sample
        dataset_name=dataset_name,
        hu_window_min=window_min,
        hu_window_max=window_max,
    )

    embedding_dim_set = False
    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        fname = str(row.get("nifti_filename", f"{sample_id}.nii.gz"))
        nifti_path = nifti_dir / fname
        if not nifti_path.exists():
            nifti_path = nifti_dir / f"{sample_id}.nii"
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI not found for sample '{sample_id}': {nifti_path}")

        emb = extract_volume_embeddings(
            nifti_path, encoder_name_or_path,
            window_min=window_min, window_max=window_max,
            slice_axis=slice_axis, batch_size=batch_size, device=device,
        )

        if not embedding_dim_set:
            import h5py
            with h5py.File(store.h5_path, "a") as f:
                f.attrs["embedding_dim"] = emb.shape[1]
            embedding_dim_set = True

        store.write_sample(
            sample_id,
            embeddings=emb,
            label=int(row["label"]),
            split=str(row["split"]),
            patient_id=str(row["patient_id"]) if "patient_id" in row else None,
        )

    store.compute_and_write_mean_embedding()
