"""Action-conditioned dynamics predictor for JEPA world model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ActionConditionedDynamics(nn.Module):
    """Predicts future latent state z_{t+H} conditioned on state z_t and candidate action plan."""

    def __init__(
        self,
        d_model: int = 64,
        action_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim + 1, hidden_dim),  # action_dim + horizon
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        layers = []
        in_dim = d_model * 2  # State + Action
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        
        self.transition_net = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, state_latent: Tensor, action: Tensor, horizon: Tensor) -> Tensor:
        """Args:
            state_latent: [batch_size, d_model]
            action: [batch_size, action_dim]
            horizon: [batch_size, 1]
        Returns:
            future_latent: [batch_size, d_model]
        """
        action_input = torch.cat([action, horizon], dim=-1)
        action_emb = self.action_encoder(action_input)

        combined = torch.cat([state_latent, action_emb], dim=-1)
        delta = self.transition_net(combined)

        # Residual latent transition: z_{t+H} = z_t + scale * delta
        future_latent = state_latent + self.residual_scale * delta
        return future_latent
