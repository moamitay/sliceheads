"""Attention-based MIL heads: ABMIL, GatedABMIL, DSMIL (PyTorch, native_mask)."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sliceheads.heads.base import BaseHead


# ---------------------------------------------------------------------------
# Shared training loop helper
# ---------------------------------------------------------------------------

def _train_loop(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    mask_val: np.ndarray | None,
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

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    mt = torch.from_numpy(mask_train).float().to(device)

    has_val = X_val is not None
    if has_val:
        Xv = torch.from_numpy(X_val).float().to(device)
        yv = torch.from_numpy(y_val).float().to(device)
        mv = torch.from_numpy(mask_val).float().to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xt, mt)
        loss = criterion(logits.squeeze(-1), yt)
        loss.backward()
        optimizer.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv, mv)
                val_loss = criterion(val_logits.squeeze(-1), yv).item()
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


def _predict_proba_torch(model: nn.Module, X: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    Xt = torch.from_numpy(X).float().to(device)
    mt = torch.from_numpy(mask).float().to(device)
    with torch.no_grad():
        logits = model(Xt, mt).squeeze(-1)  # [B]
        pos_prob = torch.sigmoid(logits).cpu().numpy()
    proba = np.stack([1.0 - pos_prob, pos_prob], axis=1).astype(np.float32)
    return proba


def _get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ABMIL network
# ---------------------------------------------------------------------------

class _ABMILNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X: [B, N, D], mask: [B, N]
        a = self.attention(X).squeeze(-1)  # [B, N]
        a = a.masked_fill(mask == 0, -1e9)
        a = F.softmax(a, dim=1)  # [B, N]
        pooled = (a.unsqueeze(-1) * X).sum(dim=1)  # [B, D]
        return self.classifier(pooled)  # [B, 1]

    def get_attention(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.attention(X).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e9)
        return F.softmax(a, dim=1)  # [B, N]


class ABMILClassifier(BaseHead):
    """Attention-based MIL (ABMIL) with masked softmax attention."""

    input_policy = "native_mask"
    supports_native_attention = True

    uses_validation_split = True

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = device
        self.random_seed = random_seed

    def _fit_padded(self, X, y, mask, X_val, y_val, mask_val) -> None:
        net = _ABMILNet(self.input_dim, self.hidden_dim, self.dropout)
        dev = _get_device(self.device)
        self.model_ = _train_loop(
            net, X, y, mask, X_val, y_val, mask_val,
            lr=self.lr, weight_decay=self.weight_decay, max_epochs=self.max_epochs,
            patience=self.patience, class_weight=self.class_weight,
            device=dev, random_seed=self.random_seed,
        )

    def _predict_proba_padded(self, X, mask):
        return _predict_proba_torch(self.model_, X, mask, _get_device(self.device))

    def _native_attention_padded(self, X, mask):
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        self.model_.eval()
        with torch.no_grad():
            return self.model_.get_attention(Xt, mt).cpu().numpy()

    def _save_backend(self, path: str) -> None:
        torch.save(self.model_.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path: str) -> None:
        self.model_ = _ABMILNet(self.input_dim, self.hidden_dim, self.dropout)
        self.model_.load_state_dict(
            torch.load(os.path.join(path, "model_state.pt"), map_location="cpu")
        )
        self.model_ = self.model_.to(_get_device(self.device))
        self.model_.eval()


# ---------------------------------------------------------------------------
# GatedABMIL
# ---------------------------------------------------------------------------

class _GatedABMILNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.tanh_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.sigmoid_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.attention_head = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def _attn(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.attention_head(self.tanh_branch(X) * self.sigmoid_branch(X)).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e9)
        return F.softmax(a, dim=1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self._attn(X, mask)
        pooled = (a.unsqueeze(-1) * X).sum(dim=1)
        return self.classifier(pooled)

    def get_attention(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._attn(X, mask)


class GatedABMILClassifier(BaseHead):
    """Gated attention MIL (tanh branch × sigmoid gate)."""

    input_policy = "native_mask"
    supports_native_attention = True

    uses_validation_split = True

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = device
        self.random_seed = random_seed

    def _fit_padded(self, X, y, mask, X_val, y_val, mask_val) -> None:
        net = _GatedABMILNet(self.input_dim, self.hidden_dim, self.dropout)
        dev = _get_device(self.device)
        self.model_ = _train_loop(
            net, X, y, mask, X_val, y_val, mask_val,
            lr=self.lr, weight_decay=self.weight_decay, max_epochs=self.max_epochs,
            patience=self.patience, class_weight=self.class_weight,
            device=dev, random_seed=self.random_seed,
        )

    def _predict_proba_padded(self, X, mask):
        return _predict_proba_torch(self.model_, X, mask, _get_device(self.device))

    def _native_attention_padded(self, X, mask):
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        self.model_.eval()
        with torch.no_grad():
            return self.model_.get_attention(Xt, mt).cpu().numpy()

    def _save_backend(self, path: str) -> None:
        torch.save(self.model_.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path: str) -> None:
        self.model_ = _GatedABMILNet(self.input_dim, self.hidden_dim, self.dropout)
        self.model_.load_state_dict(
            torch.load(os.path.join(path, "model_state.pt"), map_location="cpu")
        )
        self.model_ = self.model_.to(_get_device(self.device))
        self.model_.eval()


# ---------------------------------------------------------------------------
# DSMIL
# ---------------------------------------------------------------------------

class _DSMILNet(nn.Module):
    """Dual-stream MIL: instance classifier + bag-level attention classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.instance_clf = nn.Linear(input_dim, 1)
        self.bag_key = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.bag_query = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.bag_clf = nn.Linear(input_dim, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor, lambda_instance: float = 0.5) -> torch.Tensor:
        inst_logits = self.instance_clf(X).squeeze(-1)  # [B, N]
        inst_logits_masked = inst_logits.masked_fill(mask == 0, -1e9)

        # critical instance: highest instance score
        crit_idx = inst_logits_masked.argmax(dim=1, keepdim=True)  # [B, 1]
        crit_emb = X.gather(1, crit_idx.unsqueeze(-1).expand(-1, -1, X.shape[-1])).squeeze(1)  # [B, D]

        # attention between critical instance and bag
        q = self.bag_query(crit_emb).unsqueeze(1)  # [B, 1, H]
        k = self.bag_key(X)  # [B, N, H]
        attn = (k * q).sum(-1)  # [B, N]
        attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        pooled = (attn.unsqueeze(-1) * X).sum(dim=1)  # [B, D]
        bag_logit = self.bag_clf(pooled)  # [B, 1]

        # instance loss term (max over real slices)
        crit_logit = inst_logits_masked.max(dim=1, keepdim=True)[0]  # [B, 1]
        return bag_logit + lambda_instance * crit_logit

    def get_attention(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inst_logits = self.instance_clf(X).squeeze(-1)
        inst_logits_masked = inst_logits.masked_fill(mask == 0, -1e9)
        crit_idx = inst_logits_masked.argmax(dim=1, keepdim=True)
        crit_emb = X.gather(1, crit_idx.unsqueeze(-1).expand(-1, -1, X.shape[-1])).squeeze(1)
        q = self.bag_query(crit_emb).unsqueeze(1)
        k = self.bag_key(X)
        attn = (k * q).sum(-1)
        attn = attn.masked_fill(mask == 0, -1e9)
        return F.softmax(attn, dim=1)


class DSMILClassifier(BaseHead):
    """Dual-stream MIL (instance classifier + bag-level attention)."""

    input_policy = "native_mask"
    supports_native_attention = True

    uses_validation_split = True

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        lambda_instance: float = 0.5,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lambda_instance = lambda_instance
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = device
        self.random_seed = random_seed

    def _fit_padded(self, X, y, mask, X_val, y_val, mask_val) -> None:
        net = _DSMILNet(self.input_dim, self.hidden_dim, self.dropout)
        dev = _get_device(self.device)
        lam = self.lambda_instance

        # Custom train loop (not the shared _train_loop) to pass lambda_instance
        torch.manual_seed(self.random_seed)
        net = net.to(dev)
        pos_weight = None
        if self.class_weight == "balanced":
            n_pos = float((y == 1).sum())
            n_neg = float((y == 0).sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xt = torch.from_numpy(X).float().to(dev)
        yt = torch.from_numpy(y).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        has_val = X_val is not None
        if has_val:
            Xv = torch.from_numpy(X_val).float().to(dev)
            yv = torch.from_numpy(y_val).float().to(dev)
            mv = torch.from_numpy(mask_val).float().to(dev)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        for _ in range(self.max_epochs):
            net.train()
            optimizer.zero_grad()
            logits = net(Xt, mt, lam).squeeze(-1)
            loss = criterion(logits, yt)
            loss.backward()
            optimizer.step()
            if has_val:
                net.eval()
                with torch.no_grad():
                    vl = criterion(net(Xv, mv, lam).squeeze(-1), yv).item()
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if has_val and best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model_ = net

    def _predict_proba_padded(self, X, mask):
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(Xt, mt, self.lambda_instance).squeeze(-1)
            pos_prob = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1.0 - pos_prob, pos_prob], axis=1).astype(np.float32)

    def _native_attention_padded(self, X, mask):
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        self.model_.eval()
        with torch.no_grad():
            return self.model_.get_attention(Xt, mt).cpu().numpy()

    def _save_backend(self, path: str) -> None:
        torch.save(self.model_.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path: str) -> None:
        self.model_ = _DSMILNet(self.input_dim, self.hidden_dim, self.dropout)
        self.model_.load_state_dict(
            torch.load(os.path.join(path, "model_state.pt"), map_location="cpu")
        )
        self.model_ = self.model_.to(_get_device(self.device))
        self.model_.eval()
