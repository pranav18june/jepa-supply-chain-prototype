"""Unit tests for disruption taxonomy and shift regimes."""

import pytest
import numpy as np
from prototype_new.src.env.disruptions import DisruptionRegime, generate_scenario


def test_disruption_regimes_generation():
    rng = np.random.default_rng(42)

    # In-Distribution
    s_id = generate_scenario(DisruptionRegime.IN_DISTRIBUTION, rng)
    assert s_id.regime == DisruptionRegime.IN_DISTRIBUTION
    assert s_id.primary_capacity_factor >= 0.60
    assert not s_id.corridor_failure

    # Severity Shift
    s_sev = generate_scenario(DisruptionRegime.SEVERITY_SHIFT, rng)
    assert s_sev.regime == DisruptionRegime.SEVERITY_SHIFT
    assert s_sev.primary_capacity_factor <= 0.40  # Deep capacity drop
    assert s_sev.lead_time_delay_primary >= 2

    # Compound Shift
    s_comp = generate_scenario(DisruptionRegime.COMPOUND_SHIFT, rng)
    assert s_comp.regime == DisruptionRegime.COMPOUND_SHIFT
    assert s_comp.demand_surge_factor >= 1.40     # Surge concurrent with delay
    assert s_comp.transport_congestion is True

    # Mechanism Shift
    s_mech = generate_scenario(DisruptionRegime.MECHANISM_SHIFT, rng)
    assert s_mech.regime == DisruptionRegime.MECHANISM_SHIFT
    assert s_mech.primary_capacity_factor == 0.0   # Complete primary outage
    assert s_mech.corridor_failure is True
