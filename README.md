# sliceheads

A benchmarking toolkit for comparing classification **heads** on 2D slice
embeddings extracted from 3D medical volumes.

A frozen 2D vision encoder (e.g. DINOv2 / RAD-DINO) produces one embedding
per slice of a scan; `sliceheads` aggregates the per-slice embeddings of a
volume into a single binary prediction and lets you benchmark ten different
aggregation strategies — from simple pooling to attention-based multiple
instance learning (MIL) to time-series classifiers — against each other on
the same data, with a shared `fit` / `predict_proba` API, shared metrics,
shared hyperparameter search, and shared explainability tooling.

Every head is a genuine `scikit-learn` estimator: it works directly with
`GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, and `clone()`.

```python
from sliceheads.heads import GatedABMILClassifier

head = GatedABMILClassifier(input_dim=768, hidden_dim=128)
head.fit(train_embeddings, train_labels)   # X: ragged [N_i, D] bags, one per scan
proba = head.predict_proba(val_embeddings)  # [B, 2]
```

## Why

Comparing aggregation strategies for slice-level embeddings usually means
rewriting the same padding, masking, save/load, metrics, and explainability
code for every new head. `sliceheads` implements that plumbing once so a new
head is just a `fit` / `predict_proba` pair — see
[`notebooks/02_add_a_head.ipynb`](notebooks/02_add_a_head.ipynb).

## Head catalogue

| Head | Backend | Description |
|---|---|---|
| `MeanPoolClassifier` | scikit-learn | Mean-pool slice embeddings, logistic regression |
| `MaxPoolClassifier` | scikit-learn | Max-pool slice embeddings, logistic regression |
| `GeMPoolClassifier` | scikit-learn | Generalized-mean pooling, logistic regression |
| `ABMILClassifier` | PyTorch | Attention-based MIL |
| `GatedABMILClassifier` | PyTorch | Gated attention-based MIL |
| `DSMILClassifier` | PyTorch | Dual-stream MIL |
| `TransformerMILClassifier` | PyTorch | Transformer encoder over the slice bag |
| `MultiRocketClassifier` | aeon | MultiRocket time-series kernels |
| `InceptionTimeClassifier` | PyTorch (native) | InceptionTime, treating the slice bag as a sequence |
| `ALSTMFCNClassifier` | PyTorch (native) | Attention-LSTM-FCN |

All heads accept the same input: `X` is a sequence of length `B`, one ragged
`[N_i, D]` array of slice embeddings per scan (`N_i` varies — different scans
have different slice counts); `y` is a `[B]` array of binary labels.
Padding to a dense batch and masking is handled internally and is never
visible to the caller.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `h5py`, `scikit-learn`,
`torch`, `aeon`, `nibabel`, `transformers`, `pyyaml`, `pandas`, `scipy`,
`pyarrow`.

## Quickstart

**1. Extract slice embeddings** from a directory of NIfTI volumes into an
HDF5 store (`sliceheads.extract` + `sliceheads.store.EmbedStore`) — see
[`notebooks/prepare_hdf5_embeddings.ipynb`](notebooks/prepare_hdf5_embeddings.ipynb).

**2. Run a full benchmark** across heads via a YAML config:

```bash
python run_experiment.py --config experiment.yaml --results results/
```

This loads the embedding store, runs hyperparameter search per head (grid,
random, or Bayesian — see `experiment.yaml`), evaluates the best
configuration on the held-out test split, computes leave-one-out (LOO) slice
importance, and writes everything to `results/`: `metrics.csv`,
`test_predictions.csv`, `hyperparameters.json`, `importance.h5`,
`config_used.yaml`, `run_metadata.json`.

**3. Or drive it from Python directly:**

```python
from sliceheads.store import EmbedStore
from sliceheads.heads import GatedABMILClassifier
from sliceheads.metrics import compute_metrics

store = EmbedStore("dataset.h5")
X_train, y_train, _ = store.load_split("train")
X_test, y_test, _ = store.load_split("test")

embedding_dim = store.file_attrs()["embedding_dim"]
head = GatedABMILClassifier(input_dim=embedding_dim, hidden_dim=128)
head.fit(X_train, y_train)
metrics = compute_metrics(y_test, head.predict_proba(X_test))
```

## Notebooks

| Notebook | What it shows |
|---|---|
| [`01_full_pipeline.ipynb`](notebooks/01_full_pipeline.ipynb) | End-to-end: embeddings → train → evaluate |
| [`02_add_a_head.ipynb`](notebooks/02_add_a_head.ipynb) | Implementing a new head against the `BaseHead` contract |
| [`03_explainability.ipynb`](notebooks/03_explainability.ipynb) | LOO importance and native attention maps |
| [`04_compare_classification_heads.ipynb`](notebooks/04_compare_classification_heads.ipynb) | Benchmarking all ten heads on one dataset |
| [`05_nested_cross_validation.ipynb`](notebooks/05_nested_cross_validation.ipynb) | Nested CV for unbiased head comparison |
| [`05_nested_cross_validation_dilution.ipynb`](notebooks/05_nested_cross_validation_dilution.ipynb) | Nested CV robustness under label-noise dilution |
| [`nested_cross_validation_rad_dino.ipynb`](notebooks/nested_cross_validation_rad_dino.ipynb) | Nested CV using RAD-DINO embeddings |
| [`prepare_hdf5_embeddings.ipynb`](notebooks/prepare_hdf5_embeddings.ipynb) | Extracting slice embeddings into the HDF5 store |
| [`prepare_hdf5_embeddings_rad_dino.ipynb`](notebooks/prepare_hdf5_embeddings_rad_dino.ipynb) | Same, using the RAD-DINO encoder |
| [`slices_dilution.ipynb`](notebooks/slices_dilution.ipynb) | Robustness of heads to diluted/noisy slice bags |
| [`xai_localization.ipynb`](notebooks/xai_localization.ipynb) | Localization quality of importance maps vs. ground-truth slices |
| [`xai_nested_cross_validation.ipynb`](notebooks/xai_nested_cross_validation.ipynb) | Explainability inside a nested CV loop |
| [`xai_positive_only.ipynb`](notebooks/xai_positive_only.ipynb) | Importance analysis restricted to positive-class scans |

## Design contracts

- **Binary classification only.** `validate_h5` enforces exactly two label
  values `{0, 1}`; every `predict_proba` returns shape `[B, 2]`.
- **`predict()` = `argmax(predict_proba())`**, fixed 0.5 operating point — no
  threshold tuning.
- **Uniform save/load.** `BaseHead.save(path)` writes a `manifest.json`
  (class, params, `sliceheads` + schema version) plus a backend-native
  artifact; `BaseHead.load(path)` reconstructs the head and refuses to load
  across a schema-version mismatch.
- **No slice resampling.** Slice order and indices are never
  resized/interpolated, so per-slice importance always maps back to a real
  slice in the original volume.
- **Validation split is internal.** Heads that early-stop
  (`uses_validation_split = True`) carve their own stratified validation
  split out of the training data passed to `fit`; test data is never seen
  during fitting.

See [`experiment.yaml`](experiment.yaml) for a complete config example and
[`notebooks/02_add_a_head.ipynb`](notebooks/02_add_a_head.ipynb) for the full
`BaseHead` contract when implementing a new head.

## Robustness analysis

`sliceheads.nsr` measures how a fitted head's discrimination degrades as
task-irrelevant background slices (drawn from same-fold negative scans) are
diluted into an outer-fold test bag — reusing already-fitted, frozen models
rather than retraining. See
[`notebooks/05_nested_cross_validation_dilution.ipynb`](notebooks/05_nested_cross_validation_dilution.ipynb).

## Development

```bash
pip install -e ".[dev]"
pytest -q                      # run the test suite
ruff check . && ruff format .  # lint + format
```

## License

MIT — see [LICENSE](LICENSE).
