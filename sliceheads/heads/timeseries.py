"""Time-series classification heads: MultiRocket (aeon), InceptionTime, ALSTMFCN (PyTorch)."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sliceheads.heads.base import BaseHead
from sliceheads.heads.adapters import inception_time_adapter, multirocket_adapter
from sliceheads.heads.mil import _get_device, _train_loop


# ---------------------------------------------------------------------------
# Helpers shared by InceptionTime & ALSTMFCN
# ---------------------------------------------------------------------------

def _torch_predict_proba(
    model: nn.Module,
    X: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    adapter_fn,
) -> np.ndarray:
    model.eval()
    if adapter_fn is not None:
        inp = adapter_fn(X, mask)
    else:
        inp = X
    Xt = torch.from_numpy(inp).float().to(device)
    with torch.no_grad():
        logits = model(Xt).squeeze(-1)
        pos_prob = torch.sigmoid(logits).cpu().numpy()
    return np.stack([1.0 - pos_prob, pos_prob], axis=1).astype(np.float32)


def _adapter_train_loop(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    mask_val: np.ndarray | None,
    adapter_fn,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    class_weight: str | None,
    device: torch.device,
    random_seed: int,
) -> nn.Module:
    torch.manual_seed(random_seed)
    model = model.to(device)

    pos_weight = None
    if class_weight == "balanced":
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.from_numpy(adapter_fn(X_train, mask_train)).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)

    has_val = X_val is not None
    if has_val:
        Xv = torch.from_numpy(adapter_fn(X_val, mask_val)).float().to(device)
        yv = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt).squeeze(-1), yt)
        loss.backward()
        optimizer.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xv).squeeze(-1), yv).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# MultiRocket (aeon backend, adapter_mask)
# ---------------------------------------------------------------------------

class MultiRocketClassifier(BaseHead):
    """MultiRocket convolutional sequence features (aeon) + linear classifier."""

    input_policy = "adapter_mask"
    supports_native_attention = False

    def __init__(
        self,
        n_kernels: int = 100,
        n_groups: int = 64,
        class_weight: str | None = "balanced",
        random_seed: int = 42,
    ) -> None:
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.class_weight = class_weight
        self.random_seed = random_seed

    def _build_transform(self):
        from aeon.transformations.collection.convolution_based import MultiRocket
        return MultiRocket(
            n_kernels=self.n_kernels,
            random_state=self.random_seed,
        )

    def _to_aeon(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Convert [B, N, D] + mask to aeon format [B, D, train_n_max_] (3D numpy array).

        Used only at fit time, where every sample is <= train_n_max_ by
        construction (train_n_max_ is defined as that max) — a single
        zero-padded window per sample is always sufficient here.
        """
        samples = multirocket_adapter(X, mask)
        max_n = self.train_n_max_
        D = samples[0].shape[0]
        out = np.zeros((len(samples), D, max_n), dtype=np.float32)
        for i, s in enumerate(samples):
            n = min(s.shape[1], max_n)
            out[i, :, :n] = s[:, :n]
        return out

    def _windowed_features(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Feature-extract samples of any length, never dropping slices.

        aeon's collection transforms require every call to `transform()` to
        use the exact same fixed length seen at `fit()` time
        (`train_n_max_`), and the mask rules forbid resizing/interpolating
        along the slice axis. So instead of truncating samples longer than
        `train_n_max_` (which silently discards slices beyond the cap), each
        sample is split into ceil(N_i / train_n_max_) non-overlapping,
        untouched `train_n_max_`-length windows (the last zero-padded like
        any other short sample); every slice appears in exactly one window.
        `transform_` is applied once per window and a sample's windows are
        mean-pooled into its single feature vector. Samples with
        N_i <= train_n_max_ take the single-window path used at fit time —
        byte-identical behavior to before this fix.
        """
        samples = multirocket_adapter(X, mask)  # list of [D, N_i]
        max_n = self.train_n_max_
        D = samples[0].shape[0]

        window_batches = []
        n_windows_per_sample = []
        for s in samples:
            n = s.shape[1]
            n_windows = max(1, -(-n // max_n))  # ceil division
            windows = np.zeros((n_windows, D, max_n), dtype=np.float32)
            for w in range(n_windows):
                start = w * max_n
                end = min(start + max_n, n)
                windows[w, :, : end - start] = s[:, start:end]
            window_batches.append(windows)
            n_windows_per_sample.append(n_windows)

        all_windows = np.concatenate(window_batches, axis=0)
        window_feats = self.transform_.transform(all_windows)

        feats = np.zeros((len(samples), window_feats.shape[1]), dtype=np.float32)
        idx = 0
        for i, n_windows in enumerate(n_windows_per_sample):
            feats[i] = window_feats[idx : idx + n_windows].mean(axis=0)
            idx += n_windows
        return feats

    def _fit_padded(self, X, y, mask, *args) -> None:
        from sklearn.linear_model import LogisticRegression

        # Record training max-N before converting so predict windows consistently
        samples = multirocket_adapter(X, mask)
        self.train_n_max_ = max(s.shape[1] for s in samples)

        X_aeon = self._to_aeon(X, mask)
        self.transform_ = self._build_transform()
        feats = self.transform_.fit_transform(X_aeon)

        cw = "balanced" if self.class_weight == "balanced" else None
        self.clf_ = LogisticRegression(
            C=1.0, class_weight=cw, max_iter=1000, random_state=self.random_seed, solver="lbfgs"
        )
        self.clf_.fit(feats, y)

    def _predict_proba_padded(self, X, mask):
        feats = self._windowed_features(X, mask)
        return self.clf_.predict_proba(feats).astype(np.float32)

    def _save_backend(self, path: str) -> None:
        import joblib
        joblib.dump(self.clf_, os.path.join(path, "model.joblib"))
        joblib.dump(self.transform_, os.path.join(path, "transform.joblib"))
        joblib.dump({"train_n_max": self.train_n_max_}, os.path.join(path, "meta.joblib"))

    def _load_backend(self, path: str) -> None:
        import joblib
        self.clf_ = joblib.load(os.path.join(path, "model.joblib"))
        self.transform_ = joblib.load(os.path.join(path, "transform.joblib"))
        self.train_n_max_ = joblib.load(os.path.join(path, "meta.joblib"))["train_n_max"]


# ---------------------------------------------------------------------------
# InceptionTime (native PyTorch, adapter_mask)
# ---------------------------------------------------------------------------

class _InceptionModule(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, kernel_sizes: list[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.branches.append(
                nn.Conv1d(in_channels, n_filters, kernel_size=ks, padding=ks // 2, bias=False)
            )
        self.bottleneck = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        self.maxpool_branch = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [b(x) for b in self.branches]
        mp_out = self.mp_conv(self.maxpool_branch(x))
        out = torch.cat(branch_outs + [mp_out], dim=1)
        return F.relu(self.bn(out))


class _InceptionTimeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        depth: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        n_out_per_block = n_filters * (len(kernel_sizes) + 1)
        in_ch = in_channels
        for _ in range(depth):
            self.blocks.append(_InceptionModule(in_ch, n_filters, kernel_sizes))
            in_ch = n_out_per_block
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_out_per_block, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, N]
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


class InceptionTimeClassifier(BaseHead):
    """Multi-scale 1D CNN (InceptionTime), native PyTorch, adapter_mask."""

    input_policy = "adapter_mask"
    supports_native_attention = False
    uses_validation_split = True

    def __init__(
        self,
        input_dim: int = 768,
        n_filters: int = 32,
        depth: int = 3,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = device
        self.random_seed = random_seed

    def _resolved_kernel_sizes(self) -> list[int]:
        return self.kernel_sizes if self.kernel_sizes is not None else [9, 19, 39]

    def _fit_padded(self, X, y, mask, X_val, y_val, mask_val) -> None:
        dev = _get_device(self.device)
        net = _InceptionTimeNet(self.input_dim, self.n_filters, self.depth, self._resolved_kernel_sizes(), self.dropout)
        self.model_ = _adapter_train_loop(
            net, X, y, mask, X_val, y_val, mask_val,
            inception_time_adapter,
            lr=self.lr, weight_decay=self.weight_decay, max_epochs=self.max_epochs,
            patience=self.patience, class_weight=self.class_weight,
            device=dev, random_seed=self.random_seed,
        )

    def _predict_proba_padded(self, X, mask):
        return _torch_predict_proba(
            self.model_, X, mask, _get_device(self.device), inception_time_adapter
        )

    def _save_backend(self, path: str) -> None:
        torch.save(self.model_.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path: str) -> None:
        self.model_ = _InceptionTimeNet(
            self.input_dim, self.n_filters, self.depth, self._resolved_kernel_sizes(), self.dropout
        )
        self.model_.load_state_dict(
            torch.load(os.path.join(path, "model_state.pt"), map_location="cpu")
        )
        self.model_ = self.model_.to(_get_device(self.device))
        self.model_.eval()


# ---------------------------------------------------------------------------
# ALSTMFCN (native PyTorch, native_mask — LSTM uses seq lengths, CNN uses masked pooling)
# ---------------------------------------------------------------------------

class _ALSTMFCNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lstm_units: int,
        conv_filters: int,
        dropout: float,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.lstm_drop = nn.Dropout(dropout)
        if use_attention:
            self.attn = nn.Linear(lstm_units, 1)

        self.conv1 = nn.Conv1d(input_dim, conv_filters, kernel_size=8, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=5, padding="same", bias=False)
        self.bn2 = nn.BatchNorm1d(conv_filters * 2)
        self.conv3 = nn.Conv1d(conv_filters * 2, conv_filters, kernel_size=3, padding="same", bias=False)
        self.bn3 = nn.BatchNorm1d(conv_filters)
        self.conv_drop = nn.Dropout(dropout)

        self.classifier = nn.Linear(lstm_units + conv_filters, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X: [B, N, D], mask: [B, N]
        B, N, D = X.shape
        seq_lens = mask.sum(dim=1).long().clamp(min=1)

        # LSTM branch
        packed = nn.utils.rnn.pack_padded_sequence(
            X, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (h_n, _) = self.lstm(packed)
        lstm_out_pad, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=N)

        if self.use_attention:
            a = self.attn(lstm_out_pad).squeeze(-1)  # [B, N]
            a = a.masked_fill(mask == 0, -1e9)
            a = F.softmax(a, dim=1)
            lstm_feat = (a.unsqueeze(-1) * lstm_out_pad).sum(dim=1)
        else:
            lstm_feat = h_n.squeeze(0)  # [B, lstm_units]
        lstm_feat = self.lstm_drop(lstm_feat)

        # CNN branch: [B, D, N]
        xc = X.permute(0, 2, 1)
        xc = F.relu(self.bn1(self.conv1(xc)))
        xc = F.relu(self.bn2(self.conv2(xc)))
        xc = F.relu(self.bn3(self.conv3(xc)))
        # masked global avg pool
        m = mask.unsqueeze(1).float()  # [B, 1, N]
        conv_feat = (xc * m).sum(dim=2) / m.sum(dim=2).clamp(min=1e-9)
        conv_feat = self.conv_drop(conv_feat)

        fused = torch.cat([lstm_feat, conv_feat], dim=1)
        return self.classifier(fused)


class ALSTMFCNClassifier(BaseHead):
    """LSTM + FCN hybrid with optional attention reweighting (native PyTorch, native_mask)."""

    input_policy = "native_mask"
    supports_native_attention = True  # when attention=True
    uses_validation_split = True

    def __init__(
        self,
        input_dim: int = 768,
        lstm_units: int = 64,
        conv_filters: int = 128,
        dropout: float = 0.25,
        attention: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.conv_filters = conv_filters
        self.dropout = dropout
        self.attention = attention
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = device
        self.random_seed = random_seed

    def _fit_padded(self, X, y, mask, X_val, y_val, mask_val) -> None:
        net = _ALSTMFCNNet(self.input_dim, self.lstm_units, self.conv_filters, self.dropout, self.attention)
        dev = _get_device(self.device)
        self.model_ = _train_loop(
            net, X, y, mask, X_val, y_val, mask_val,
            lr=self.lr, weight_decay=self.weight_decay, max_epochs=self.max_epochs,
            patience=self.patience, class_weight=self.class_weight,
            device=dev, random_seed=self.random_seed,
        )

    def _predict_proba_padded(self, X, mask):
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(Xt, mt).squeeze(-1)
            pos_prob = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1.0 - pos_prob, pos_prob], axis=1).astype(np.float32)

    def _native_attention_padded(self, X, mask):
        if not self.attention:
            return None
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        B, N, D = Xt.shape
        self.model_.eval()
        with torch.no_grad():
            seq_lens = mt.sum(dim=1).long().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(Xt, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.model_.lstm(packed)
            lstm_out_pad, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=N)
            a = self.model_.attn(lstm_out_pad).squeeze(-1)
            a = a.masked_fill(mt == 0, -1e9)
            a = F.softmax(a, dim=1)
        return a.cpu().numpy()

    def _save_backend(self, path: str) -> None:
        torch.save(self.model_.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path: str) -> None:
        self.model_ = _ALSTMFCNNet(
            self.input_dim, self.lstm_units, self.conv_filters, self.dropout, self.attention
        )
        self.model_.load_state_dict(
            torch.load(os.path.join(path, "model_state.pt"), map_location="cpu")
        )
        self.model_ = self.model_.to(_get_device(self.device))
        self.model_.eval()
