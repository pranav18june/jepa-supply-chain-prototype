"""Cost and risk readouts mapping latent states to operational quantities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class CostReadout(nn.Module):
    """Predicts expected operational cost and tail-risk exposure from future latent state."""

    def __init__(self, d_model: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [expected_cost, tail_risk_estimate]
        )
        # Initialize readout bias close to reasonable mean scale
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[0], 25.0)
        nn.init.constant_(self.net[-1].bias[1], 35.0)

    def forward(self, future_latent: Tensor) -> Tuple[Tensor, Tensor]:
        """Args:
            future_latent: [batch_size, d_model]
        Returns:
            expected_cost: [batch_size]
            tail_cost: [batch_size]
        """
        out = self.net(future_latent)
        expected_cost = out[:, 0]
        tail_cost = out[:, 1]
        return expected_cost, tail_cost
