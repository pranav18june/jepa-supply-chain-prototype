"""Unit tests for multi-echelon network simulator and CRN branching."""

import pytest
import numpy as np
from prototype_new.src.env.network import NetworkTopology, NetworkState
from prototype_new.src.env.costs import CostParameters, CostBreakdown
from prototype_new.src.env.actions import generate_candidate_plans
from prototype_new.src.env.disruptions import DisruptionRegime
from prototype_new.src.env.simulator import MultiEchelonSimulator, SimulationConfig


def test_topology_creation():
    topo = NetworkTopology.standard_retail()
    assert len(topo.suppliers) == 2
    assert topo.dc.name == "central_dc"
    assert len(topo.stores) == 3


def test_candidate_plans_generation():
    plans = generate_candidate_plans(
        dc_inventory=50.0,
        store_inventories={"store_A": 10.0, "store_B": 15.0, "store_C": 8.0},
        store_backorders={"store_A": 0.0, "store_B": 2.0, "store_C": 0.0},
        recent_store_demands={"store_A": 12.0, "store_B": 16.0, "store_C": 10.0},
    )
    assert len(plans) == 5
    assert "regular" in plans
    assert "split" in plans
    assert "expedited" in plans
    assert "reserve_central" in plans
    assert "priority_store" in plans

    for name, plan in plans.items():
        feat = plan.to_feature_vector()
        assert feat.shape == (8,)
        assert not np.isnan(feat).any()


def test_simulator_crn_world_generation():
    sim = MultiEchelonSimulator()
    config = SimulationConfig(lookback=16, horizon=4, future_replications=4, seed=123)
    world = sim.generate_world(0, DisruptionRegime.IN_DISTRIBUTION, config)

    assert world.public_history.shape == (16, 16)
    assert len(world.candidate_plans) == 5
    assert len(world.outcomes_by_plan) == 5

    # Check that each plan has 4 replications
    for name, outcomes in world.outcomes_by_plan.items():
        assert len(outcomes) == 4
        for out in outcomes:
            assert out.total_cost > 0.0
            assert 0.0 <= out.fill_rate <= 1.0

    # Check oracle cost logic
    oracle_plan = world.oracle_plan()
    assert oracle_plan in world.candidate_plans
    exp_costs = [world.expected_cost(p) for p in world.candidate_plans]
    assert world.oracle_cost() == min(exp_costs)
