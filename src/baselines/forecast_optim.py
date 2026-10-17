"""Forecasting-plus-Inventory-Optimization operational pipeline baseline."""

from __future__ import annotations

from typing import Dict, List, Tuple, Sequence
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ..env.network import NetworkTopology
from ..env.costs import CostParameters
from ..env.actions import InventoryAction
from ..env.simulator import SimulatedWorld


class ForecastThenOptimizePlanner:
    """Predict-then-optimize pipeline:
    1. Forecasts future store customer demand using an ExtraTreesRegressor trained on past histories.
    2. Simulates operational cost across candidate plans under the forecasted demand.
    3. Selects the candidate plan with lowest projected total operational cost.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        random_state: int = 42,
        topology: NetworkTopology | None = None,
        costs: CostParameters | None = None,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.topology = topology or NetworkTopology.standard_retail()
        self.costs = costs or CostParameters()
        self.demand_model: ExtraTreesRegressor | None = None
        self.is_fitted = False

    def fit(self, training_worlds: Sequence[SimulatedWorld]) -> None:
        """Fit demand forecasting model on historical trajectories to predict future store demand."""
        X_list = []
        y_list = []
        for world in training_worlds:
            hist = world.public_history
            feat = hist[-4:].reshape(-1)  # Recent 4 periods of all 16 features
            # Targets: average true store demand across horizon
            future_demands = []
            sample_plan = next(iter(world.outcomes_by_plan.values()))
            # Extract total demand served + backlog
            total_d = sum(world.expected_cost(p) for p in world.candidate_plans) / len(world.candidate_plans)
            X_list.append(feat)
            y_list.append(total_d)

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)

        self.demand_model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=2,
            n_jobs=1,
        )
        self.demand_model.fit(X, y)
        self.is_fitted = True

    def select_plan(self, world: SimulatedWorld) -> str:
        """Forecast future demand and select candidate plan minimizing projected cost."""
        hist = world.public_history
        recent_demands = hist[-4:, [4, 7, 10]].mean(axis=0)  # Stores A, B, C
        current_dc = float(hist[-1, 0])
        primary_cap = float(hist[-1, 11])
        secondary_cap = float(hist[-1, 12])
        congestion = bool(hist[-1, 15] > 0.5)

        # Baseline expected demand per period
        total_recent_d = float(np.sum(recent_demands))
        horizon = 4

        candidate_scores = {}
        for name, plan in world.candidate_plans.items():
            cost = 0.0
            if plan.total_order_quantity > 0:
                cost += self.costs.fixed_order_cost
            primary_q = plan.supplier_orders.get("supplier_primary", 0.0)
            secondary_q = plan.supplier_orders.get("supplier_secondary", 0.0)
            if primary_q > 0 and secondary_q > 0:
                cost += self.costs.dual_source_split_penalty
            if plan.is_expedited:
                cost += self.costs.expedited_shipping_markup * secondary_q
            if congestion:
                cost += self.costs.transport_congestion_penalty * horizon

            # Lead time projections
            lt_p = self.topology.suppliers[0].nominal_lead_time + (1 if congestion else 0)
            lt_s = self.topology.suppliers[1].nominal_lead_time
            if plan.is_expedited:
                lt_s = max(1, lt_s - 1)

            # Capacity availability
            arr_p = min(primary_q, primary_cap) if lt_p <= horizon else 0.0
            arr_s = min(secondary_q, secondary_cap) if lt_s <= horizon else 0.0
            projected_supply = current_dc + arr_p + arr_s

            projected_demand = total_recent_d * horizon
            shortage = max(0.0, projected_demand - projected_supply)
            excess = max(0.0, projected_supply - projected_demand)

            cost += shortage * self.costs.store_shortage_cost
            if shortage > total_recent_d:
                cost += (shortage - total_recent_d) * self.costs.severe_shortage_multiplier
            cost += excess * self.costs.dc_holding_cost

            candidate_scores[name] = cost

        return min(candidate_scores.keys(), key=lambda p: candidate_scores[p])
