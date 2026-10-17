"""Rule-based operational heuristics baseline (Base-stock and Safety Buffer)."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ..env.actions import InventoryAction
from ..env.simulator import SimulatedWorld


class RuleBasedPlanner:
    """Conventional operational heuristic: conservative base-stock with risk buffering."""

    def __init__(self, buffer_safety_factor: float = 1.25):
        self.buffer_safety_factor = buffer_safety_factor

    def select_plan(self, world: SimulatedWorld) -> str:
        """Select plan based on classical inventory rule:
        - If current DC inventory is depleted or backorders exist, prefer 'split' or 'expedited' to buffer risk.
        - Otherwise, stick to 'regular' replenishment.
        """
        history = world.public_history
        # Check current stockout indicators from history
        recent_backorders = history[-1, [3, 6, 9]]  # Store A, B, C backorders
        dc_inventory = history[-1, 0]
        congestion = history[-1, 15]

        total_backorder = float(np.sum(recent_backorders))

        if congestion > 0.5 and total_backorder > 5.0:
            return "expedited"
        elif total_backorder > 8.0 or dc_inventory < 30.0:
            return "split"
        elif dc_inventory > 100.0:
            return "reserve_central"
        else:
            return "regular"
