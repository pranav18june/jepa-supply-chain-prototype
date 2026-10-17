"""Pairwise and risk-sensitive decision-alignment losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from .risk_weights import RiskWeightingScheme


class RiskSensitiveDecisionAlignmentLoss(nn.Module):
    """Loss function combining predictive reconstruction loss and risk-weighted pairwise decision alignment."""

    def __init__(
        self,
        alignment_weight: float = 1.0,
        risk_weighting_scheme: Optional[RiskWeightingScheme] = None,
        weight_mode: str = "tail_cost",
        tie_threshold: float = 0.25,
        margin: float = 0.0,
    ):
        super().__init__()
        self.alignment_weight = alignment_weight
        self.risk_scheme = risk_weighting_scheme or RiskWeightingScheme()
        self.weight_mode = weight_mode
        self.tie_threshold = tie_threshold
        self.margin = margin

    def forward(
        self,
        pred_costs: Tensor,       # [batch_size, num_candidates]
        true_costs: Tensor,       # [batch_size, num_candidates] (centered)
        pred_tail: Optional[Tensor] = None, # [batch_size, num_candidates]
        true_tail: Optional[Tensor] = None, # [batch_size, num_candidates]
        raw_costs: Optional[Tensor] = None, # [batch_size, num_candidates] (uncentered for risk weights)
    ) -> Tuple[Tensor, dict[str, float]]:
        """Compute the joint predictive regression and risk-sensitive ranking loss."""
        # 1. Base predictive loss (Smooth L1 regression on centered cost scale)
        pred_loss = F.smooth_l1_loss(pred_costs, true_costs)
        if pred_tail is not None and true_tail is not None:
            tail_reg_loss = F.smooth_l1_loss(pred_tail, true_tail)
            pred_loss = pred_loss + 0.5 * tail_reg_loss

        # 2. Risk-sensitive pairwise decision-alignment loss
        batch_size, num_candidates = pred_costs.shape
        if num_candidates < 2 or self.alignment_weight <= 0.0:
            return pred_loss, {"loss_pred": float(pred_loss.item()), "loss_rank": 0.0, "loss_total": float(pred_loss.item())}

        # Form all unique candidate pairs within each scenario instance (i < j)
        pair_losses = []
        pair_weights = []

        cost_ref = raw_costs if raw_costs is not None else true_costs

        for b in range(batch_size):
            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    c_i = true_costs[b, i]
                    c_j = true_costs[b, j]
                    cost_diff = c_j - c_i  # positive if plan i is cheaper than plan j

                    # Prespecified tie handling rule: ignore negligible differences
                    if torch.abs(cost_diff) < self.tie_threshold:
                        continue

                    # Ground truth preference: +1 if plan i is better (cheaper), -1 if j is better
                    y_ij = torch.sign(cost_diff)

                    # Model predicted cost difference: q_i - q_j = pred_c_j - pred_c_i
                    pred_q_diff = pred_costs[b, j] - pred_costs[b, i]

                    # Pairwise logistic ranking loss: log(1 + exp(-y_ij * (q_i - q_j - margin)))
                    pair_loss = F.softplus(-y_ij * pred_q_diff + self.margin)

                    # Calculate risk-sensitive weight w_ij using absolute cost reference
                    raw_i = cost_ref[b, i]
                    raw_j = cost_ref[b, j]
                    w_ij = self.risk_scheme.compute_pair_weights(
                        raw_i.unsqueeze(0), raw_j.unsqueeze(0), scheme=self.weight_mode
                    ).squeeze(0)

                    pair_losses.append(pair_loss)
                    pair_weights.append(w_ij)

        if len(pair_losses) > 0:
            losses_tensor = torch.stack(pair_losses)
            weights_tensor = torch.stack(pair_weights)
            # Weighted average: sum(w_ij * l_ij) / sum(w_ij)
            rank_loss = (weights_tensor * losses_tensor).sum() / weights_tensor.sum().clamp_min(1e-6)
        else:
            rank_loss = torch.tensor(0.0, device=pred_costs.device)

        total_loss = pred_loss + self.alignment_weight * rank_loss

        metrics = {
            "loss_pred": float(pred_loss.item()),
            "loss_rank": float(rank_loss.item()),
            "loss_total": float(total_loss.item()),
        }
        return total_loss, metrics
