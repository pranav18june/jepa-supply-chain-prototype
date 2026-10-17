"""Conventional learned dynamics baseline (Direct MLP without JEPA representation)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple

from ..env.simulator import SimulatedWorld


class ConventionalDynamicsModel(nn.Module):
    """Direct end-to-end MLP dynamics that maps raw history features and action directly
    to predicted future cost, without learning a self-supervised JEPA state representation.
    """

    def __init__(self, raw_feature_dim: int = 16 * 16, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 25.0)

    def forward(self, raw_history: Tensor, action: Tensor) -> Tensor:
        """Args:
            raw_history: [batch_size, raw_feature_dim]
            action: [batch_size, action_dim]
        Returns:
            predicted_cost: [batch_size]
        """
        combined = torch.cat([raw_history, action], dim=-1)
        return self.net(combined).squeeze(-1)
