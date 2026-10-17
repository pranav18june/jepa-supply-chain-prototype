"""Unit tests for GatedEnsemblePolicy and threshold calibration."""

import numpy as np
import pytest
import torch

from prototype_new.src.env.simulator import MultiEchelonSimulator, SimulationConfig
from prototype_new.src.env.disruptions import DisruptionRegime
from prototype_new.src.models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from prototype_new.src.models.uncertainty import EnsembleUncertaintyGate
from prototype_new.src.baselines.rule_based import RuleBasedPlanner
from prototype_new.src.baselines.robust_optim import RobustOptimizationPlanner
from prototype_new.src.evaluation.gating import (
    GatedEnsemblePolicy,
    evaluate_policy_with_traces,
)


def test_gated_ensemble_policy_routing():
    sim = MultiEchelonSimulator()
    config = SimulationConfig(seed=9601, future_replications=4)
    world = sim.generate_world(0, DisruptionRegime.IN_DISTRIBUTION, config)

    m1 = RiskSensitiveJEPAWorldModel(num_features=16, d_model=32, action_dim=8)
    m2 = RiskSensitiveJEPAWorldModel(num_features=16, d_model=32, action_dim=8)
    fb = RuleBasedPlanner()

    # Case 1: Threshold = inf -> never fallback
    gate_never = EnsembleUncertaintyGate(threshold=float("inf"))
    policy_never = GatedEnsemblePolicy([m1, m2], fb, gate_never)
    plan_never, trace_never = policy_never.predict_and_select(world)
    assert not trace_never.fallback_triggered
    assert trace_never.policy_type == "ensemble"

    # Case 2: Threshold = -1.0 -> always fallback
    gate_always = EnsembleUncertaintyGate(threshold=-1.0)
    policy_always = GatedEnsemblePolicy([m1, m2], fb, gate_always)
    plan_always, trace_always = policy_always.predict_and_select(world)
    assert trace_always.fallback_triggered
    assert trace_always.policy_type == "fallback"
    assert plan_always == fb.select_plan(world)


def test_evaluate_policy_with_traces():
    sim = MultiEchelonSimulator()
    config = SimulationConfig(seed=9601, future_replications=4)
    worlds = [sim.generate_world(i, DisruptionRegime.IN_DISTRIBUTION, config) for i in range(3)]

    m = RiskSensitiveJEPAWorldModel(num_features=16, d_model=32, action_dim=8)
    gate = EnsembleUncertaintyGate(threshold=float("inf"))
    policy = GatedEnsemblePolicy([m], RuleBasedPlanner(), gate)

    traces, metrics = evaluate_policy_with_traces(policy, worlds, "test_policy")
    assert len(traces) == 3
    assert metrics.mean_regret >= 0.0
    for t in traces:
        assert t.evaluation.regret >= 0.0
        assert t.evaluation.oracle_cost <= t.evaluation.chosen_cost + 1e-4
