"""Unit tests for JEPA neural components and world model."""

import pytest
import torch
from prototype_new.src.models.jepa_encoder import OperationalHistoryEncoder
from prototype_new.src.models.dynamics import ActionConditionedDynamics
from prototype_new.src.models.readouts import CostReadout
from prototype_new.src.models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from prototype_new.src.models.uncertainty import EnsembleUncertaintyGate


def test_encoder_forward():
    encoder = OperationalHistoryEncoder(num_features=16, d_model=64)
    dummy_history = torch.randn(4, 16, 16)
    z_t = encoder(dummy_history)
    assert z_t.shape == (4, 64)


def test_dynamics_forward():
    dynamics = ActionConditionedDynamics(d_model=64, action_dim=8)
    dummy_z = torch.randn(4, 64)
    dummy_action = torch.randn(4, 8)
    dummy_horizon = torch.full((4, 1), 4.0)

    future_z = dynamics(dummy_z, dummy_action, dummy_horizon)
    assert future_z.shape == (4, 64)


def test_cost_readout():
    readout = CostReadout(d_model=64)
    future_z = torch.randn(4, 64)
    exp_c, tail_c = readout(future_z)
    assert exp_c.shape == (4,)
    assert tail_c.shape == (4,)


def test_full_jepa_world_model():
    model = RiskSensitiveJEPAWorldModel(num_features=16, d_model=64, action_dim=8)
    dummy_history = torch.randn(2, 16, 16)
    dummy_actions = torch.randn(2, 5, 8)  # 5 candidate plans
    dummy_horizon = torch.full((2, 1), 4.0)

    future_z, pred_cost, pred_tail = model(dummy_history, dummy_actions, dummy_horizon)
    assert future_z.shape == (2, 5, 64)
    assert pred_cost.shape == (2, 5)
    assert pred_tail.shape == (2, 5)

    best_plan, scores = model.rank_plans(
        dummy_history[:1], dummy_actions[:1], dummy_horizon[:1],
        plan_names=["plan_0", "plan_1", "plan_2", "plan_3", "plan_4"]
    )
    assert best_plan in scores
    assert len(scores) == 5


def test_uncertainty_gate():
    gate = EnsembleUncertaintyGate(uncertainty_threshold=5.0)
    preds = [torch.tensor([10.0, 20.0]), torch.tensor([11.0, 35.0]), torch.tensor([10.5, 25.0])]
    unc = gate.compute_uncertainty(preds)
    assert unc.shape == (2,)
    assert not gate.should_fallback(float(unc[0]))  # low variance
    assert gate.should_fallback(float(unc[1]))      # high variance
