"""TransformerMIL classification head (native PyTorch, native_mask)."""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sliceheads.heads.base import BaseHead
from sliceheads.heads.mil import _get_device, _predict_proba_torch, _train_loop


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class _TransformerMILNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if pooling == "attention":
            self.attn_pool = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape
        x = self.input_proj(X)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [B, N+1, d]
            # mask: prepend a 1 for the cls token
            cls_mask = torch.ones(B, 1, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, mask], dim=1) == 0  # True = ignore
        else:
            key_padding_mask = mask == 0  # [B, N], True where padding

        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.pooling == "cls":
            pooled = x[:, 0, :]
        elif self.pooling == "mean":
            m = (~key_padding_mask).float().unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "attention":
            a = self.attn_pool(x).squeeze(-1)
            a = a.masked_fill(key_padding_mask, -1e9)
            a = F.softmax(a, dim=1)
            pooled = (a.unsqueeze(-1) * x).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.classifier(pooled)

    def get_attention(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if self.pooling != "attention":
            return None
        B, N, D = X.shape
        x = self.input_proj(X)
        key_padding_mask = mask == 0
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        a = self.attn_pool(x).squeeze(-1)
        a = a.masked_fill(key_padding_mask, -1e9)
        return F.softmax(a, dim=1)


class TransformerMILClassifier(BaseHead):
    """Transformer encoder over slice embeddings (positional encoding + pooling)."""

    input_policy = "native_mask"
    supports_native_attention = True  # only when pooling="attention"
    uses_validation_split = True

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "cls",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.pooling = pooling
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = device
        self.random_seed = random_seed

    def _fit_padded(self, X, y, mask, X_val, y_val, mask_val) -> None:
        net = _TransformerMILNet(
            self.input_dim, self.d_model, self.num_layers, self.n_heads,
            self.dropout, self.pooling,
        )
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
        if self.pooling != "attention":
            return None
        dev = _get_device(self.device)
        Xt = torch.from_numpy(X).float().to(dev)
        mt = torch.from_numpy(mask).float().to(dev)
        self.model_.eval()
        with torch.no_grad():
            result = self.model_.get_attention(Xt, mt)
            return result.cpu().numpy() if result is not None else None

    def _save_backend(self, path: str) -> None:
        torch.save(self.model_.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path: str) -> None:
        self.model_ = _TransformerMILNet(
            self.input_dim, self.d_model, self.num_layers, self.n_heads,
            self.dropout, self.pooling,
        )
        self.model_.load_state_dict(
            torch.load(os.path.join(path, "model_state.pt"), map_location="cpu")
        )
        self.model_ = self.model_.to(_get_device(self.device))
        self.model_.eval()
