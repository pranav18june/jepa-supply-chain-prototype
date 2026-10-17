"""Operational history context encoder for SupplyChain-JEPA."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class OperationalHistoryEncoder(nn.Module):
    """Encodes partially observed multi-echelon history into a latent operating state z_t."""

    def __init__(
        self,
        num_features: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_projection = nn.Linear(num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history: Tensor) -> Tensor:
        """Args:
            history: [batch_size, seq_len, num_features]
        Returns:
            latent_state: [batch_size, d_model]
        """
        batch_size, seq_len, _ = history.shape
        # Input normalization across sequence
        mean = history.mean(dim=1, keepdim=True)
        std = history.std(dim=1, keepdim=True).clamp_min(1e-4)
        normalized = (history - mean) / std

        proj = self.feature_projection(normalized)
        pos = self.pos_embedding[:, :seq_len, :]
        tokens = proj + pos

        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)

        # Average pool across sequence time dimension to produce summary latent operating state
        latent_state = encoded.mean(dim=1)
        return latent_state
