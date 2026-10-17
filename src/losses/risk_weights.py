"""Risk-sensitive scenario and pair weighting schemes."""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np


class RiskWeightingScheme:
    """Computes risk-sensitive weights w_ij based strictly on training-set statistics."""

    def __init__(
        self,
        cost_median: float = 25.0,
        cost_iqr: float = 15.0,
        gamma: float = 2.0,
        tail_quantile_threshold: float = 0.75,
    ):
        self.cost_median = cost_median
        self.cost_iqr = max(cost_iqr, 1e-4)
        self.gamma = gamma
        self.tail_threshold = tail_quantile_threshold

    @classmethod
    def fit_from_training_costs(cls, training_costs: np.ndarray, gamma: float = 2.0) -> RiskWeightingScheme:
        """Calibrate baseline statistics strictly on the training partition."""
        median_val = float(np.median(training_costs))
        q75, q25 = np.percentile(training_costs, [75, 25])
        iqr_val = float(max(q75 - q25, 1.0))
        return cls(cost_median=median_val, cost_iqr=iqr_val, gamma=gamma, tail_quantile_threshold=float(q75))

    def compute_pair_weights(
        self,
        cost_i: Tensor,
        cost_j: Tensor,
        scheme: str = "tail_cost",
    ) -> Tensor:
        """Args:
            cost_i: Realized cost of plan i [num_pairs]
            cost_j: Realized cost of plan j [num_pairs]
            scheme: "tail_cost", "linear_severity", or "uniform"
        Returns:
            weights: [num_pairs] non-negative weighting tensor
        """
        if scheme == "uniform":
            # Generic unweighted decision alignment (D-JEPA baseline)
            return torch.ones_like(cost_i)

        max_cost = torch.maximum(cost_i, cost_j)

        if scheme == "tail_cost":
            # Tail cost weighting: heavily penalize ranking errors when either candidate leads to severe tail costs
            tail_excess = torch.clamp((max_cost - self.cost_median) / self.cost_iqr, min=0.0)
            weights = 1.0 + self.gamma * tail_excess
            # Clamp weights to prevent gradient explosions in extreme outliers
            return torch.clamp(weights, min=1.0, max=10.0)

        elif scheme == "tail_indicator":
            # Binary tail indicator: extra weight if max_cost is above 75th percentile
            is_tail = (max_cost >= self.tail_threshold).float()
            return 1.0 + self.gamma * is_tail

        elif scheme == "cost_difference":
            # Weight proportional to regret stakes |C_i - C_j|
            diff = torch.abs(cost_i - cost_j) / self.cost_iqr
            return 1.0 + self.gamma * torch.clamp(diff, max=5.0)

        else:
            raise ValueError(f"Unknown risk weighting scheme {scheme}")
