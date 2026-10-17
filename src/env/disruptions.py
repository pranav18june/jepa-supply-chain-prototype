"""Disruption regimes and scenario generators following the proposal taxonomy."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


class DisruptionRegime(str, Enum):
    IN_DISTRIBUTION = "in_distribution"
    SEVERITY_SHIFT = "severity_shift"
    COMPOUND_SHIFT = "compound_shift"
    MECHANISM_SHIFT = "mechanism_shift"


@dataclass(frozen=True)
class DisruptionScenario:
    """Scenario realization affecting supplier capacity, lead times, demand, and transport."""
    regime: DisruptionRegime
    primary_capacity_factor: float
    secondary_capacity_factor: float
    lead_time_delay_primary: int
    lead_time_delay_secondary: int
    demand_surge_factor: float
    transport_congestion: bool
    corridor_failure: bool = False  # Mechanism shift: complete route or primary supplier failure


def generate_scenario(
    regime: DisruptionRegime,
    rng: np.random.Generator,
    training_mode: bool = False,
) -> DisruptionScenario:
    """Generate disruption scenario parameters conditioned on regime.
    
    In training mode, disruptions only occur individually (e.g. demand spike alone OR
    mild supplier delay alone), maintaining the compound-shift holdout guarantee.
    """
    if training_mode or regime == DisruptionRegime.IN_DISTRIBUTION:
        # Familiar mild variations
        disruption_type = rng.choice(["mild_supplier", "mild_demand", "normal"], p=[0.35, 0.35, 0.30])
        if disruption_type == "mild_supplier":
            cap_factor = float(rng.uniform(0.65, 0.90))
            delay = int(rng.choice([0, 1], p=[0.7, 0.3]))
            surge = float(rng.uniform(0.95, 1.10))
            congestion = bool(rng.random() < 0.15)
        elif disruption_type == "mild_demand":
            cap_factor = float(rng.uniform(0.90, 1.0))
            delay = 0
            surge = float(rng.uniform(1.15, 1.35))
            congestion = bool(rng.random() < 0.15)
        else:
            cap_factor = float(rng.uniform(0.95, 1.05))
            delay = 0
            surge = 1.0
            congestion = False

        return DisruptionScenario(
            regime=DisruptionRegime.IN_DISTRIBUTION,
            primary_capacity_factor=cap_factor,
            secondary_capacity_factor=float(rng.uniform(0.90, 1.05)),
            lead_time_delay_primary=delay,
            lead_time_delay_secondary=0,
            demand_surge_factor=surge,
            transport_congestion=congestion,
            corridor_failure=False,
        )

    elif regime == DisruptionRegime.SEVERITY_SHIFT:
        # Severity shift: Familiar supplier capacity loss but at MUCH higher magnitude than training
        # (e.g. 70-90% drop vs 10-35% in training, plus prolonged delays)
        cap_factor = float(rng.uniform(0.10, 0.35))
        delay = int(rng.choice([2, 3], p=[0.6, 0.4]))
        return DisruptionScenario(
            regime=DisruptionRegime.SEVERITY_SHIFT,
            primary_capacity_factor=cap_factor,
            secondary_capacity_factor=float(rng.uniform(0.60, 0.85)),
            lead_time_delay_primary=delay,
            lead_time_delay_secondary=int(rng.choice([0, 1], p=[0.6, 0.4])),
            demand_surge_factor=float(rng.uniform(0.95, 1.10)),  # Demand stays normal
            transport_congestion=bool(rng.random() < 0.30),
            corridor_failure=False,
        )

    elif regime == DisruptionRegime.COMPOUND_SHIFT:
        # Compound shift: Severe supplier capacity loss + demand surge + transport congestion TOGETHER!
        # These were only seen individually during training.
        cap_factor = float(rng.uniform(0.35, 0.60))
        delay = int(rng.choice([1, 2], p=[0.5, 0.5]))
        surge = float(rng.uniform(1.45, 1.85))  # Large demand surge simultaneously
        return DisruptionScenario(
            regime=DisruptionRegime.COMPOUND_SHIFT,
            primary_capacity_factor=cap_factor,
            secondary_capacity_factor=float(rng.uniform(0.70, 0.90)),
            lead_time_delay_primary=delay,
            lead_time_delay_secondary=1,
            demand_surge_factor=surge,
            transport_congestion=True,  # Severe congestion concurrent with demand surge & supply shock
            corridor_failure=False,
        )

    elif regime == DisruptionRegime.MECHANISM_SHIFT:
        # Mechanism shift: Complete primary supplier shutdown + corridor transport failure
        return DisruptionScenario(
            regime=DisruptionRegime.MECHANISM_SHIFT,
            primary_capacity_factor=0.0,  # 100% shutdown
            secondary_capacity_factor=float(rng.uniform(0.40, 0.70)),
            lead_time_delay_primary=4,    # Never arrives in horizon
            lead_time_delay_secondary=2,
            demand_surge_factor=float(rng.uniform(1.20, 1.50)),
            transport_congestion=True,
            corridor_failure=True,
        )

    raise ValueError(f"Unknown disruption regime {regime}")
