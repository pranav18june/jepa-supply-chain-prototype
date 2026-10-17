"""Unit tests for world split disjointness and seed determinism."""

import pytest
from prototype_new.src.env.simulator import MultiEchelonSimulator, SimulationConfig
from prototype_new.src.env.disruptions import DisruptionRegime


def test_world_pools_disjointness():
    sim = MultiEchelonSimulator()
    config = SimulationConfig(seed=9601, future_replications=4)

    train_ids = set(range(40))
    val_ids = set(range(1000, 1010))
    calib_ids = set(range(2000, 2015))
    test_ids = set(range(3000, 3020))

    # Assert all world ID sets are completely mutually disjoint
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(calib_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(calib_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(calib_ids.intersection(test_ids)) == 0


def test_seed_determinism():
    sim = MultiEchelonSimulator()
    c1 = SimulationConfig(seed=1234, future_replications=4)
    c2 = SimulationConfig(seed=1234, future_replications=4)

    w1 = sim.generate_world(0, DisruptionRegime.IN_DISTRIBUTION, c1)
    w2 = sim.generate_world(0, DisruptionRegime.IN_DISTRIBUTION, c2)

    assert (w1.public_history == w2.public_history).all()
    assert abs(w1.oracle_cost() - w2.oracle_cost()) < 1e-4
