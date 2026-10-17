"""Unit tests for risk-sensitive weighting and pairwise ranking losses."""

import pytest
import torch
import numpy as np
from prototype_new.src.losses.risk_weights import RiskWeightingScheme
from prototype_new.src.losses.ranking_losses import RiskSensitiveDecisionAlignmentLoss


def test_risk_weighting_scheme():
    costs = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 80.0])
    scheme = RiskWeightingScheme.fit_from_training_costs(costs, gamma=2.0)

    # Low cost pair
    c1 = torch.tensor([12.0])
    c2 = torch.tensor([14.0])
    w_low = scheme.compute_pair_weights(c1, c2, scheme="tail_cost")
    assert float(w_low.item()) == 1.0  # Normal weight

    # Severe high cost pair
    c3 = torch.tensor([75.0])
    c4 = torch.tensor([85.0])
    w_high = scheme.compute_pair_weights(c3, c4, scheme="tail_cost")
    assert float(w_high.item()) > 1.0  # Amplified risk-sensitive weight


def test_ranking_loss_direction():
    loss_fn = RiskSensitiveDecisionAlignmentLoss(alignment_weight=1.0)

    # 1 batch, 2 candidates
    # Ground truth: candidate 0 cost = 10, candidate 1 cost = 30 (candidate 0 is better!)
    true_costs = torch.tensor([[10.0, 30.0]])

    # Case A: Correct model prediction (predicts 0 is cheaper than 1)
    pred_correct = torch.tensor([[10.0, 30.0]])
    loss_a, m_a = loss_fn(pred_correct, true_costs)

    # Case B: Inverted model prediction (predicts 1 is cheaper than 0)
    pred_wrong = torch.tensor([[30.0, 10.0]])
    loss_b, m_b = loss_fn(pred_wrong, true_costs)

    # Ranking loss for wrong prediction should be strictly higher
    assert m_b["loss_rank"] > m_a["loss_rank"]
