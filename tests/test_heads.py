"""Tests for all classification heads: sklearn estimator contract, padding
internals, and save/load round-trip.

The public contract under test: every head's fit(X, y) / predict_proba(X) /
predict(X) accepts a plain sequence of ragged [N_i, D] arrays and a 1-D label
array — no mask, no padding, no wrapper — and every head is clone()-able and
usable directly inside GridSearchCV / RandomizedSearchCV / cross_val_score.
"""

import os
import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from sliceheads.heads.pooling import MeanPoolClassifier, MaxPoolClassifier, GeMPoolClassifier
from sliceheads.heads.mil import ABMILClassifier, GatedABMILClassifier, DSMILClassifier
from sliceheads.heads.transformer import TransformerMILClassifier
from sliceheads.heads.timeseries import (
    MultiRocketClassifier, InceptionTimeClassifier, ALSTMFCNClassifier,
)
from sliceheads.heads.base import BaseHead


# ---------------------------------------------------------------------------
# The full ten-head catalogue: one tiny/fast instance per head, plus a
# 2-value hyperparameter grid used by the GridSearchCV / nested-CV tests.
# ---------------------------------------------------------------------------

HEAD_CATALOGUE = [
    (MeanPoolClassifier(C=0.1), {"C": [0.1, 1.0]}),
    (MaxPoolClassifier(C=0.1), {"C": [0.1, 1.0]}),
    (GeMPoolClassifier(p=2.0, C=0.1), {"p": [1.0, 2.0]}),
    (MultiRocketClassifier(n_kernels=84, n_groups=2), {"n_kernels": [84, 168]}),
    (ABMILClassifier(input_dim=16, hidden_dim=8, max_epochs=3, patience=3), {"hidden_dim": [8, 16]}),
    (GatedABMILClassifier(input_dim=16, hidden_dim=8, max_epochs=3, patience=3), {"hidden_dim": [8, 16]}),
    (DSMILClassifier(input_dim=16, hidden_dim=8, max_epochs=3, patience=3), {"hidden_dim": [8, 16]}),
    (TransformerMILClassifier(input_dim=16, d_model=16, num_layers=1, n_heads=2, max_epochs=3, patience=3), {"num_layers": [1, 2]}),
    (InceptionTimeClassifier(input_dim=16, n_filters=4, depth=1, max_epochs=3, patience=3), {"n_filters": [4, 8]}),
    (ALSTMFCNClassifier(input_dim=16, lstm_units=8, conv_filters=8, max_epochs=3, patience=3), {"lstm_units": [8, 16]}),
]
HEAD_IDS = [head.__class__.__name__ for head, _ in HEAD_CATALOGUE]

# Name of the trailing-underscore attribute holding each head's fitted backend.
_LEARNED_ATTR = {
    "MeanPoolClassifier": "clf_",
    "MaxPoolClassifier": "clf_",
    "GeMPoolClassifier": "clf_",
    "MultiRocketClassifier": "clf_",
    "ABMILClassifier": "model_",
    "GatedABMILClassifier": "model_",
    "DSMILClassifier": "model_",
    "TransformerMILClassifier": "model_",
    "InceptionTimeClassifier": "model_",
    "ALSTMFCNClassifier": "model_",
}


def _simple_heads():
    return [pair[0] for pair in HEAD_CATALOGUE]


def _object_array(embeddings_list):
    """Wrap a ragged list as a 1-D object ndarray sklearn's CV splitters can index."""
    X = np.empty(len(embeddings_list), dtype=object)
    for i, e in enumerate(embeddings_list):
        X[i] = e
    return X


# ---------------------------------------------------------------------------
# Fixture sanity: the data must actually be ragged to exercise padding
# ---------------------------------------------------------------------------

def test_synthetic_data_is_ragged(synthetic_data):
    embeddings_list, labels, D = synthetic_data
    lengths = {e.shape[0] for e in embeddings_list}
    assert len(lengths) > 1, "fixture must contain varying slice counts to exercise padding"


# ---------------------------------------------------------------------------
# Public contract: predict_proba shape, predict = argmax, classes_
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head", _simple_heads(), ids=HEAD_IDS)
def test_predict_proba_shape(head, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head.fit(embeddings_list[:12], labels[:12])
    proba = head.predict_proba(embeddings_list[12:])
    n_test = len(embeddings_list[12:])
    assert proba.shape == (n_test, 2), f"{head.__class__.__name__}: expected shape ({n_test}, 2), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1"
    assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.parametrize("head", _simple_heads(), ids=HEAD_IDS)
def test_predict_is_argmax(head, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head.fit(embeddings_list[:12], labels[:12])
    proba = head.predict_proba(embeddings_list[12:])
    preds = head.predict(embeddings_list[12:])
    assert np.array_equal(preds, proba.argmax(axis=1))


@pytest.mark.parametrize("head", _simple_heads(), ids=HEAD_IDS)
def test_classes_from_fit_data(head, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head.fit(embeddings_list, labels)
    assert list(head.classes_) == [0, 1]


# ---------------------------------------------------------------------------
# sklearn estimator contract: clone / get_params / set_params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head, grid", HEAD_CATALOGUE, ids=HEAD_IDS)
def test_clone_round_trip(head, grid):
    cloned = clone(head)
    assert cloned is not head
    assert type(cloned) is type(head)
    assert cloned.get_params() == head.get_params()


@pytest.mark.parametrize("head, grid", HEAD_CATALOGUE, ids=HEAD_IDS)
def test_set_params_round_trip(head, grid):
    param_name, values = next(iter(grid.items()))
    new_value = values[-1] if getattr(head, param_name) != values[-1] else values[0]
    head.set_params(**{param_name: new_value})
    assert getattr(head, param_name) == new_value
    assert head.get_params()[param_name] == new_value


@pytest.mark.parametrize("head, grid", HEAD_CATALOGUE, ids=HEAD_IDS)
def test_refit_after_set_params(head, grid, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    param_name, values = next(iter(grid.items()))
    head.set_params(**{param_name: values[-1]})
    head.fit(embeddings_list, labels)
    assert getattr(head, param_name) == values[-1]
    proba = head.predict_proba(embeddings_list)
    assert proba.shape == (len(embeddings_list), 2)


# ---------------------------------------------------------------------------
# Clone independence — the dangerous silent-sharing failure mode: a clone
# must not share its fitted backend (torch module / sklearn estimator) with
# the original, or refitting one would corrupt the other across CV folds.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head", _simple_heads(), ids=HEAD_IDS)
def test_clone_independence(head, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    attr = _LEARNED_ATTR[head.__class__.__name__]

    head.fit(embeddings_list[:12], labels[:12])
    proba_before = head.predict_proba(embeddings_list[12:])

    clone_b = clone(head)
    assert not hasattr(clone_b, attr), "clone() must not carry over fitted state"

    # Fit the clone on different (reversed) data; label pattern alternates so
    # the reversed slice still has both classes.
    clone_b.fit(embeddings_list[:12][::-1], labels[:12][::-1])

    assert getattr(head, attr) is not getattr(clone_b, attr), (
        f"{head.__class__.__name__}: clone shares the fitted backend with the original"
    )
    proba_after = head.predict_proba(embeddings_list[12:])
    np.testing.assert_array_equal(
        proba_before, proba_after,
        err_msg=f"{head.__class__.__name__}: fitting a clone mutated the original estimator's predictions",
    )


# ---------------------------------------------------------------------------
# GridSearchCV over each head (2 configs x 2 inner folds) — the exact
# no-wrapper usage this refactor exists to make possible.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head, grid", HEAD_CATALOGUE, ids=HEAD_IDS)
def test_grid_search_cv(head, grid, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    X = _object_array(embeddings_list)

    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    gs = GridSearchCV(head, grid, cv=inner_cv, scoring="roc_auc", n_jobs=1)
    gs.fit(X, labels)

    param_name = next(iter(grid))
    assert gs.best_params_[param_name] in grid[param_name]
    proba = gs.predict_proba(X)
    assert proba.shape == (len(X), 2)


# ---------------------------------------------------------------------------
# Nested CV: cross_val_score(GridSearchCV(...), cv=2) for every head.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head, grid", HEAD_CATALOGUE, ids=HEAD_IDS)
def test_nested_cross_val_score(head, grid, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    X = _object_array(embeddings_list)

    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    gs = GridSearchCV(head, grid, cv=inner_cv, scoring="roc_auc", n_jobs=1)
    scores = cross_val_score(gs, X, labels, cv=outer_cv, scoring="roc_auc", n_jobs=1)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head", _simple_heads(), ids=HEAD_IDS)
def test_save_load_roundtrip(tmp_path, head, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head.fit(embeddings_list[:12], labels[:12])
    proba_before = head.predict_proba(embeddings_list[12:])

    save_dir = str(tmp_path / head.__class__.__name__)
    head.save(save_dir)

    loaded = BaseHead.load(save_dir)
    proba_after = loaded.predict_proba(embeddings_list[12:])

    # Exact equality for deterministic sklearn heads; tolerance for torch heads
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-4,
                                err_msg=f"{head.__class__.__name__} round-trip failed")


# ---------------------------------------------------------------------------
# Save / load onto a GPU device: BaseHead.load() reconstructs the model on
# CPU (state_dict loaded with map_location="cpu") and must move it back onto
# self.device — a head built with device="cuda" must actually predict on
# CUDA-resident weights after a save/load round-trip, not silently stay on
# CPU (which breaks the first time input tensors are also moved to CUDA).
# ---------------------------------------------------------------------------

_TORCH_HEAD_FACTORIES = {
    "ABMILClassifier": lambda dev: ABMILClassifier(input_dim=16, hidden_dim=8, max_epochs=3, patience=3, device=dev),
    "GatedABMILClassifier": lambda dev: GatedABMILClassifier(input_dim=16, hidden_dim=8, max_epochs=3, patience=3, device=dev),
    "DSMILClassifier": lambda dev: DSMILClassifier(input_dim=16, hidden_dim=8, max_epochs=3, patience=3, device=dev),
    "TransformerMILClassifier": lambda dev: TransformerMILClassifier(
        input_dim=16, d_model=16, num_layers=1, n_heads=2, max_epochs=3, patience=3, device=dev
    ),
    "InceptionTimeClassifier": lambda dev: InceptionTimeClassifier(input_dim=16, n_filters=4, depth=1, max_epochs=3, patience=3, device=dev),
    "ALSTMFCNClassifier": lambda dev: ALSTMFCNClassifier(input_dim=16, lstm_units=8, conv_filters=8, max_epochs=3, patience=3, device=dev),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("head_name", list(_TORCH_HEAD_FACTORIES), ids=list(_TORCH_HEAD_FACTORIES))
def test_save_load_roundtrip_moves_model_to_device(tmp_path, head_name, synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head = _TORCH_HEAD_FACTORIES[head_name]("cuda")
    head.fit(embeddings_list[:12], labels[:12])

    save_dir = str(tmp_path / head_name)
    head.save(save_dir)

    loaded = BaseHead.load(save_dir)
    assert next(loaded.model_.parameters()).device.type == "cuda", (
        f"{head_name}: model still on CPU after loading a device='cuda' head"
    )
    # Would previously raise: "Expected all tensors to be on the same
    # device" — input is moved to cuda by _predict_proba_torch, but the
    # reconstructed model stayed on CPU without the .to(device) fix above.
    proba = loaded.predict_proba(embeddings_list[12:])
    assert proba.shape == (len(embeddings_list[12:]), 2)


# ---------------------------------------------------------------------------
# Test: schema_version mismatch is refused
# ---------------------------------------------------------------------------

def test_load_wrong_schema_version(tmp_path, synthetic_data):
    import json
    embeddings_list, labels, D = synthetic_data
    head = MeanPoolClassifier()
    head.fit(embeddings_list, labels)
    save_dir = str(tmp_path / "mean")
    head.save(save_dir)

    manifest_path = os.path.join(save_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["schema_version"] = 999
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError, match="schema_version"):
        BaseHead.load(save_dir)


# ---------------------------------------------------------------------------
# Native attention for attention-capable heads
# ---------------------------------------------------------------------------

def test_abmil_native_attention(synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head = ABMILClassifier(input_dim=D, hidden_dim=8, max_epochs=3, patience=3)
    head.fit(embeddings_list[:12], labels[:12])

    test_emb = embeddings_list[12:]
    attn = head.native_attention(test_emb)
    assert attn is not None
    N_max = max(e.shape[0] for e in test_emb)
    assert attn.shape == (len(test_emb), N_max)
    # Attention over real slices should sum to ~1 per sample
    for i, e in enumerate(test_emb):
        real_sum = attn[i, : e.shape[0]].sum()
        assert abs(real_sum - 1.0) < 1e-4


def test_transformer_cls_no_attention(synthetic_data):
    embeddings_list, labels, D = synthetic_data
    head = TransformerMILClassifier(
        input_dim=D, d_model=16, num_layers=1, n_heads=2, pooling="cls", max_epochs=3, patience=3
    )
    head.fit(embeddings_list[:12], labels[:12])
    attn = head.native_attention(embeddings_list[12:])
    assert attn is None  # cls pooling has no attention map
