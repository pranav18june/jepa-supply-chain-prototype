"""Robust / Stochastic optimization baseline over uncertainty sets with validation tuning."""

from __future__ import annotations

from typing import Dict, List, Tuple, Sequence
import numpy as np

from ..env.network import NetworkTopology
from ..env.costs import CostParameters
from ..env.actions import InventoryAction
from ..env.simulator import SimulatedWorld


class RobustOptimizationPlanner:
    """Robust inventory optimization: evaluates candidate plans against an uncertainty set
    of plausible disruption scenarios (minimax formulation):
        a* = argmin_{a} [ (1 - alpha) * Mean(C(a, xi)) + alpha * Max(C(a, xi)) ]
    """

    def __init__(
        self,
        risk_aversion: float = 0.40,
        topology: NetworkTopology | None = None,
        costs: CostParameters | None = None,
    ):
        self.risk_aversion = risk_aversion
        self.topology = topology or NetworkTopology.standard_retail()
        self.costs = costs or CostParameters()

    def tune_on_validation(self, val_worlds: Sequence[SimulatedWorld]) -> float:
        """Grid search over risk aversion alpha on validation split."""
        best_alpha = self.risk_aversion
        best_regret = float("inf")
        for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
            self.risk_aversion = alpha
            regrets = []
            for w in val_worlds:
                chosen = self.select_plan(w)
                regrets.append(w.expected_cost(chosen) - w.oracle_cost())
            mean_reg = float(np.mean(regrets))
            if mean_reg < best_regret:
                best_regret = mean_reg
                best_alpha = alpha
        self.risk_aversion = best_alpha
        return best_alpha

    def select_plan(self, world: SimulatedWorld) -> str:
        """Evaluate each candidate plan against an uncertainty set of scenarios."""
        history = world.public_history
        recent_demands = history[-4:, [4, 7, 10]].mean(axis=0)  # Stores A, B, C
        current_dc = float(history[-1, 0])
        congestion = float(history[-1, 15])

        # Define 3 plausible operational scenarios in the uncertainty set:
        # Scenario 1: Nominal (1.0x demand, 1.0x capacity, 0 delay)
        # Scenario 2: Supply Breakdown (0.9x demand, 0.3x primary capacity, +2 delay)
        # Scenario 3: Compound Shock (1.5x demand, 0.4x primary capacity, +2 delay, congestion)
        scenarios = [
            {"demand_mult": 1.0, "prim_cap": 1.0, "sec_cap": 1.0, "prim_del": 0, "sec_del": 0, "cong": False},
            {"demand_mult": 0.95, "prim_cap": 0.30, "sec_cap": 0.85, "prim_del": 2, "sec_del": 0, "cong": False},
            {"demand_mult": 1.50, "prim_cap": 0.40, "sec_cap": 0.80, "prim_del": 2, "sec_del": 1, "cong": True},
        ]

        candidate_scores = {}
        horizon = 4

        for name, plan in world.candidate_plans.items():
            scen_costs = []
            primary_q = plan.supplier_orders.get("supplier_primary", 0.0)
            secondary_q = plan.supplier_orders.get("supplier_secondary", 0.0)

            for sc in scenarios:
                c = 0.0
                if plan.total_order_quantity > 0:
                    c += self.costs.fixed_order_cost
                if primary_q > 0 and secondary_q > 0:
                    c += self.costs.dual_source_split_penalty
                if plan.is_expedited:
                    c += self.costs.expedited_shipping_markup * secondary_q
                if sc["cong"]:
                    c += self.costs.transport_congestion_penalty * horizon

                # Lead times
                lt_p = self.topology.suppliers[0].nominal_lead_time + sc["prim_del"]
                lt_s = self.topology.suppliers[1].nominal_lead_time + sc["sec_del"]
                if plan.is_expedited:
                    lt_s = max(1, lt_s - 1)

                p_arr = primary_q * sc["prim_cap"] if lt_p <= horizon else 0.0
                s_arr = secondary_q * sc["sec_cap"] if lt_s <= horizon else 0.0
                total_inflow = p_arr + s_arr

                # Available supply for horizon
                total_supply = current_dc + total_inflow
                projected_demand = sum(recent_demands) * sc["demand_mult"] * horizon

                shortage = max(0.0, projected_demand - total_supply)
                excess = max(0.0, total_supply - projected_demand)

                c += shortage * self.costs.store_shortage_cost
                if shortage > sum(recent_demands):
                    c += (shortage - sum(recent_demands)) * self.costs.severe_shortage_multiplier
                c += excess * self.costs.dc_holding_cost

                scen_costs.append(c)

            worst_case = max(scen_costs)
            expected = float(np.mean(scen_costs))
            candidate_scores[name] = (1.0 - self.risk_aversion) * expected + self.risk_aversion * worst_case

        return min(candidate_scores.keys(), key=lambda p: candidate_scores[p])
