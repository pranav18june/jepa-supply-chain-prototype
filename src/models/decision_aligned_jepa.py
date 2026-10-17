"""Risk-Sensitive Decision-Aligned JEPA World Model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional

from .jepa_encoder import OperationalHistoryEncoder
from .dynamics import ActionConditionedDynamics
from .readouts import CostReadout
from .uncertainty import EnsembleUncertaintyGate


class RiskSensitiveJEPAWorldModel(nn.Module):
    """Integrated Action-Conditioned JEPA World Model with Risk-Sensitive Decision Alignment."""

    def __init__(
        self,
        num_features: int = 16,
        d_model: int = 64,
        action_dim: int = 8,
        risk_beta: float = 0.25,
        enable_uncertainty_gate: bool = False,
        uncertainty_threshold: float = 15.0,
    ):
        super().__init__()
        self.encoder = OperationalHistoryEncoder(num_features=num_features, d_model=d_model)
        self.dynamics = ActionConditionedDynamics(d_model=d_model, action_dim=action_dim)
        self.readout = CostReadout(d_model=d_model)
        self.risk_beta = risk_beta
        self.enable_uncertainty_gate = enable_uncertainty_gate
        self.uncertainty_gate = EnsembleUncertaintyGate(uncertainty_threshold)

    def encode(self, history: Tensor) -> Tensor:
        """Encode history into latent state z_t."""
        return self.encoder(history)

    def forward(
        self,
        history: Tensor,
        actions: Tensor,
        horizons: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Args:
            history: [batch_size, seq_len, num_features]
            actions: [batch_size, num_candidates, action_dim] or [batch_size, action_dim]
            horizons: [batch_size, 1]
        Returns:
            future_latents: [batch_size, num_candidates, d_model]
            pred_costs: [batch_size, num_candidates]
            pred_tail_costs: [batch_size, num_candidates]
        """
        batch_size = history.shape[0]
        z_t = self.encode(history)  # [batch_size, d_model]

        if actions.dim() == 2:
            # Single action per batch element
            future_z = self.dynamics(z_t, actions, horizons)
            pred_cost, pred_tail = self.readout(future_z)
            return future_z, pred_cost, pred_tail
        elif actions.dim() == 3:
            # Multiple candidate actions per batch element [batch_size, num_actions, action_dim]
            num_actions = actions.shape[1]
            z_t_expanded = z_t.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, self.encoder.d_model)
            actions_flat = actions.reshape(-1, actions.shape[-1])
            horizons_expanded = horizons.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, 1)

            future_z_flat = self.dynamics(z_t_expanded, actions_flat, horizons_expanded)
            pred_cost_flat, pred_tail_flat = self.readout(future_z_flat)

            future_z = future_z_flat.reshape(batch_size, num_actions, self.encoder.d_model)
            pred_cost = pred_cost_flat.reshape(batch_size, num_actions)
            pred_tail = pred_tail_flat.reshape(batch_size, num_actions)
            return future_z, pred_cost, pred_tail
        else:
            raise ValueError(f"Unexpected action tensor dimension {actions.dim()}")

    def rank_plans(
        self,
        history: Tensor,
        action_tensor: Tensor,
        horizon_tensor: Tensor,
        plan_names: List[str],
    ) -> Tuple[str, Dict[str, float]]:
        """Rank candidate plans for a single decision instance and select the best action.
        
        Selection rule: a* = argmin [ E[C | s, a] + beta * Tail[C | s, a] ]
        """
        self.eval()
        with torch.no_grad():
            _, pred_costs, pred_tails = self.forward(history, action_tensor, horizon_tensor)
            
            # Risk-aware decision score
            risk_scores = pred_costs[0] + self.risk_beta * pred_tails[0]
            scores_dict = {
                name: float(risk_scores[i].item())
                for i, name in enumerate(plan_names)
            }
            
            best_plan = min(scores_dict.keys(), key=lambda p: scores_dict[p])
            return best_plan, scores_dict
