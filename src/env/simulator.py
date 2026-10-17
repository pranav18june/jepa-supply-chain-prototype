"""Multi-echelon simulator with counterfactual branching via Common Random Numbers (CRN)."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .network import NetworkTopology, NetworkState, StoreNode, SupplierNode
from .costs import CostParameters, CostBreakdown
from .actions import InventoryAction, generate_candidate_plans
from .disruptions import DisruptionRegime, DisruptionScenario, generate_scenario


HISTORY_FEATURE_NAMES = (
    "dc_inventory",
    "dc_pipeline",
    "store_A_inventory",
    "store_A_backorder",
    "store_A_demand",
    "store_B_inventory",
    "store_B_backorder",
    "store_B_demand",
    "store_C_inventory",
    "store_C_backorder",
    "store_C_demand",
    "supplier_primary_capacity_reported",
    "supplier_secondary_capacity_reported",
    "supplier_primary_report_age",
    "supplier_secondary_report_age",
    "congestion_reported",
)


@dataclass(frozen=True)
class SimulationConfig:
    lookback: int = 16
    horizon: int = 4
    future_replications: int = 8
    seed: int = 9601
    missing_report_rate: float = 0.15


@dataclass
class FutureOutcome:
    replication: int
    plan_name: str
    total_cost: float
    breakdown: CostBreakdown
    terminal_backorders: float
    stockout_periods: int
    fill_rate: float


@dataclass
class SimulatedWorld:
    world_id: str
    regime: DisruptionRegime
    scenario: DisruptionScenario
    public_history: np.ndarray  # Shape [lookback, len(HISTORY_FEATURE_NAMES)]
    state_at_decision: NetworkState
    candidate_plans: Dict[str, InventoryAction]
    outcomes_by_plan: Dict[str, List[FutureOutcome]]

    def expected_cost(self, plan_name: str) -> float:
        outcomes = self.outcomes_by_plan[plan_name]
        return float(np.mean([o.total_cost for o in outcomes]))

    def cvar_cost(self, plan_name: str, alpha: float = 0.95) -> float:
        """Conditional Value at Risk (CVaR) over future replications."""
        costs = np.array([o.total_cost for o in self.outcomes_by_plan[plan_name]])
        cutoff = np.quantile(costs, alpha)
        tail = costs[costs >= cutoff]
        return float(np.mean(tail)) if len(tail) > 0 else float(cutoff)

    def oracle_plan(self) -> str:
        """Hindsight expected optimal candidate plan."""
        return min(self.candidate_plans.keys(), key=lambda p: self.expected_cost(p))

    def oracle_cost(self) -> float:
        return self.expected_cost(self.oracle_plan())


class MultiEchelonSimulator:
    """Simulator modeling supplier ordering, DC handling, transportation, and store retail."""

    def __init__(
        self,
        topology: Optional[NetworkTopology] = None,
        costs: Optional[CostParameters] = None,
    ):
        self.topology = topology or NetworkTopology.standard_retail()
        self.costs = costs or CostParameters()

    def generate_world(
        self,
        world_index: int,
        regime: DisruptionRegime,
        config: SimulationConfig,
        training_mode: bool = False,
    ) -> SimulatedWorld:
        """Generate a complete world with operational history and CRN counterfactual branches."""
        regime_seed = zlib.crc32(regime.value.encode("utf-8"))
        seed_seq = np.random.SeedSequence([config.seed, world_index, regime_seed])
        rng = np.random.default_rng(seed_seq)

        world_id = f"{regime.value[:3]}-w{world_index:04d}-{rng.integers(10000, 99999)}"
        scenario = generate_scenario(regime, rng, training_mode=training_mode)

        # 1. Warm up and simulate history (lookback periods)
        history_rows, current_state, recent_store_demands = self._simulate_history(
            rng, config, scenario
        )

        # 2. Generate feasible candidate action plans at decision time
        candidate_plans = generate_candidate_plans(
            dc_inventory=current_state.dc_inventory,
            store_inventories=current_state.store_inventories,
            store_backorders=current_state.store_backorders,
            recent_store_demands=recent_store_demands,
            nominal_horizon=config.horizon,
        )

        # 3. Simulate counterfactual branches with Common Random Numbers (CRN)
        outcomes_by_plan: Dict[str, List[FutureOutcome]] = {p: [] for p in candidate_plans}

        for replication in range(config.future_replications):
            rep_seed_seq = np.random.SeedSequence([config.seed, world_index, replication, 7719])
            rep_rng = np.random.default_rng(rep_seed_seq)

            # Pre-generate exogenous shock sequence for this replication
            exogenous_sequence = self._sample_exogenous_future(
                rep_rng, config.horizon, scenario, recent_store_demands
            )

            # Evaluate each candidate action from identical starting state and identical exogenous draws
            for plan_name, plan in candidate_plans.items():
                outcome = self._simulate_future_rollout(
                    current_state=current_state.copy(),
                    plan=plan,
                    horizon=config.horizon,
                    scenario=scenario,
                    exogenous=exogenous_sequence,
                    replication=replication,
                )
                outcomes_by_plan[plan_name].append(outcome)

        return SimulatedWorld(
            world_id=world_id,
            regime=regime,
            scenario=scenario,
            public_history=np.asarray(history_rows, dtype=np.float32),
            state_at_decision=current_state,
            candidate_plans=candidate_plans,
            outcomes_by_plan=outcomes_by_plan,
        )

    def _simulate_history(
        self,
        rng: np.random.Generator,
        config: SimulationConfig,
        scenario: DisruptionScenario,
    ) -> Tuple[List[List[float]], NetworkState, Dict[str, float]]:
        """Simulate pre-decision operational history with partial observation and report lags."""
        total_base_demand = sum(s.base_demand for s in self.topology.stores)

        # Initial steady-state buffers
        dc_inventory = float(rng.uniform(2.0, 3.5) * total_base_demand)
        dc_pipeline = float(rng.uniform(1.5, 3.0) * total_base_demand)
        store_inventories = {s.name: float(rng.uniform(1.5, 2.5) * s.base_demand) for s in self.topology.stores}
        store_backorders = {s.name: 0.0 for s in self.topology.stores}
        store_pipelines = {s.name: float(rng.uniform(0.8, 1.2) * s.base_demand) for s in self.topology.stores}
        
        rep_age = {"supplier_primary": 0.0, "supplier_secondary": 0.0}
        reported_caps = {
            "supplier_primary": self.topology.suppliers[0].nominal_capacity,
            "supplier_secondary": self.topology.suppliers[1].nominal_capacity,
        }

        history_rows: List[List[float]] = []
        recent_demands: Dict[str, List[float]] = {s.name: [] for s in self.topology.stores}

        for t in range(config.lookback):
            # Demand realizations
            t_demands = {}
            for s in self.topology.stores:
                seasonal = 1.0 + 0.15 * math.sin(2.0 * math.pi * t / 7.0)
                d = max(1.0, float(rng.normal(s.base_demand * seasonal, 1.5)))
                t_demands[s.name] = d
                recent_demands[s.name].append(d)

            # Pipeline deliveries arriving to DC
            dc_delivered = min(dc_pipeline, float(rng.uniform(0.8, 1.2) * total_base_demand))
            dc_pipeline = max(0.0, dc_pipeline - dc_delivered)
            dc_inventory += dc_delivered

            # DC replenishment order
            if t % 2 == 0:
                target_dc = 3.0 * total_base_demand
                dc_pipeline += max(0.0, target_dc - dc_inventory - dc_pipeline)

            # Store arrivals, demand fulfillment, and sales
            store_demands_reported = {}
            for s in self.topology.stores:
                store_delivered = store_pipelines[s.name]
                store_pipelines[s.name] = 0.0
                available = store_inventories[s.name] + store_delivered
                demand = t_demands[s.name] + store_backorders[s.name]
                sales = min(available, demand)
                store_backorders[s.name] = max(0.0, demand - sales)
                store_inventories[s.name] = max(0.0, available - sales)
                # Censored demand: when stockout occurs, demand is partially masked by sales
                store_demands_reported[s.name] = sales if store_inventories[s.name] <= 1e-4 else t_demands[s.name]

            # Regular dispatches from DC to stores
            for s in self.topology.stores:
                needed = max(0.0, 2.0 * s.base_demand + store_backorders[s.name] - store_inventories[s.name])
                dispatch = min(dc_inventory * (s.base_demand / total_base_demand), needed)
                dc_inventory = max(0.0, dc_inventory - dispatch)
                store_pipelines[s.name] += dispatch

            # Supplier capacity reporting with stochastic delay / missingness
            for sup in self.topology.suppliers:
                missing = rng.random() < config.missing_report_rate
                if not missing:
                    rep_age[sup.name] = 0.0
                    reported_caps[sup.name] = sup.nominal_capacity * scenario.primary_capacity_factor if sup.reliability_tier == "primary" else sup.nominal_capacity * scenario.secondary_capacity_factor
                else:
                    rep_age[sup.name] += 1.0

            row = [
                dc_inventory,
                dc_pipeline,
                store_inventories["store_A"],
                store_backorders["store_A"],
                store_demands_reported["store_A"],
                store_inventories["store_B"],
                store_backorders["store_B"],
                store_demands_reported["store_B"],
                store_inventories["store_C"],
                store_backorders["store_C"],
                store_demands_reported["store_C"],
                reported_caps["supplier_primary"],
                reported_caps["supplier_secondary"],
                rep_age["supplier_primary"],
                rep_age["supplier_secondary"],
                float(scenario.transport_congestion),
            ]
            history_rows.append(row)

        current_state = NetworkState(
            step=config.lookback,
            dc_inventory=dc_inventory,
            dc_pipeline=dc_pipeline,
            store_inventories=store_inventories,
            store_backorders=store_backorders,
            store_pipelines=store_pipelines,
            supplier_capacities=reported_caps,
            supplier_states={"supplier_primary": 1, "supplier_secondary": 2},
            congestion=scenario.transport_congestion,
        )

        avg_recent_demands = {s: float(np.mean(recent_demands[s][-4:])) for s in recent_demands}
        return history_rows, current_state, avg_recent_demands

    def _sample_exogenous_future(
        self,
        rng: np.random.Generator,
        horizon: int,
        scenario: DisruptionScenario,
        recent_store_demands: Dict[str, float],
    ) -> Dict[str, Any]:
        """Draw exogenous shocks across the future horizon for CRN."""
        store_demands_future = {}
        for s in self.topology.stores:
            base = recent_store_demands[s.name] * scenario.demand_surge_factor
            noise = rng.normal(0.0, 0.18 * base, size=horizon)
            store_demands_future[s.name] = np.maximum(1.0, base + noise)

        # Supplier fill rate stochastic variations
        sup_fill_primary = np.clip(scenario.primary_capacity_factor + rng.normal(0.0, 0.05, size=horizon), 0.0, 1.0)
        sup_fill_secondary = np.clip(scenario.secondary_capacity_factor + rng.normal(0.0, 0.05, size=horizon), 0.0, 1.0)

        return {
            "store_demands": store_demands_future,
            "primary_fill": sup_fill_primary,
            "secondary_fill": sup_fill_secondary,
        }

    def _simulate_future_rollout(
        self,
        current_state: NetworkState,
        plan: InventoryAction,
        horizon: int,
        scenario: DisruptionScenario,
        exogenous: Dict[str, Any],
        replication: int,
    ) -> FutureOutcome:
        """Simulate a candidate action forward over the decision horizon."""
        state = current_state.copy()
        breakdown = CostBreakdown()
        stockout_periods = 0
        total_demand_served = 0.0
        total_demand_occurred = 0.0

        # Account for action execution costs at decision time t
        # Fixed order cost
        if plan.total_order_quantity > 0:
            breakdown.ordering += self.costs.fixed_order_cost
        # Sourcing costs & dual-sourcing coordination penalty
        primary_q = plan.supplier_orders.get("supplier_primary", 0.0)
        secondary_q = plan.supplier_orders.get("supplier_secondary", 0.0)
        if primary_q > 0 and secondary_q > 0:
            breakdown.penalties += self.costs.dual_source_split_penalty

        # Expedited shipping markup
        if plan.is_expedited:
            breakdown.transport += self.costs.expedited_shipping_markup * secondary_q

        # Schedule supplier arrivals to DC
        # Primary lead time with delay
        lt_primary = self.topology.suppliers[0].nominal_lead_time + scenario.lead_time_delay_primary
        lt_secondary = self.topology.suppliers[1].nominal_lead_time + scenario.lead_time_delay_secondary
        if plan.is_expedited:
            lt_secondary = max(1, lt_secondary - 1)

        dc_arrivals = np.zeros(horizon, dtype=np.float32)
        if lt_primary - 1 < horizon:
            fill_p = float(exogenous["primary_fill"][min(horizon - 1, lt_primary - 1)])
            dc_arrivals[lt_primary - 1] += primary_q * fill_p
        if lt_secondary - 1 < horizon:
            fill_s = float(exogenous["secondary_fill"][min(horizon - 1, lt_secondary - 1)])
            dc_arrivals[lt_secondary - 1] += secondary_q * fill_s

        # Dispatch allocations from DC to stores at step 0
        total_allocated = plan.total_store_allocation
        available_dc = max(0.0, state.dc_inventory)
        actual_dispatch = min(available_dc, total_allocated)
        scale = actual_dispatch / max(total_allocated, 1e-6) if total_allocated > 0 else 1.0

        state.dc_inventory -= actual_dispatch
        store_arrivals = np.zeros((len(self.topology.stores), horizon), dtype=np.float32)
        store_indices = {s.name: i for i, s in enumerate(self.topology.stores)}
        
        for s in self.topology.stores:
            dispatched_store = plan.dc_allocations.get(s.name, 0.0) * scale
            idx = store_indices[s.name]
            # DC to store transit takes 1 period (or 2 if congested)
            transit_time = 1 + (1 if scenario.transport_congestion else 0)
            if transit_time - 1 < horizon:
                store_arrivals[idx, transit_time - 1] += dispatched_store

        # Roll forward through horizon periods
        for h in range(horizon):
            # 1. DC arrivals from upstream suppliers
            state.dc_inventory += float(dc_arrivals[h])

            # 2. Dynamic DC allocation to stores as shipments arrive
            reserve_buffer = (plan.reserve_ratio * state.dc_inventory) if plan.reserve_ratio > 0 else 0.0
            dispatchable_dc = max(0.0, state.dc_inventory - reserve_buffer)

            if dispatchable_dc > 0.5 and h < horizon - 1:
                shortfalls = {
                    s.name: max(0.0, 1.5 * s.base_demand + state.store_backorders[s.name] - state.store_inventories[s.name])
                    for s in self.topology.stores
                }
                total_sf = sum(shortfalls.values())
                if total_sf > 0.5:
                    disp_amount = min(dispatchable_dc, total_sf)
                    transit_lag = 1 + (1 if scenario.transport_congestion else 0)
                    arrival_period = h + transit_lag
                    if arrival_period < horizon:
                        for s in self.topology.stores:
                            share = shortfalls[s.name] / total_sf
                            disp_s = disp_amount * share
                            state.dc_inventory -= disp_s
                            store_arrivals[store_indices[s.name], arrival_period] += disp_s

            breakdown.holding += state.dc_inventory * self.costs.dc_holding_cost

            # 3. Store arrivals & retail fulfillment
            for s in self.topology.stores:
                idx = store_indices[s.name]
                state.store_inventories[s.name] += float(store_arrivals[idx, h])
                
                d = float(exogenous["store_demands"][s.name][h])
                total_demand_occurred += d
                net_demand = d + state.store_backorders[s.name]
                
                fulfilled = min(state.store_inventories[s.name], net_demand)
                total_demand_served += fulfilled
                
                state.store_backorders[s.name] = max(0.0, net_demand - fulfilled)
                state.store_inventories[s.name] = max(0.0, state.store_inventories[s.name] - fulfilled)

                # Cost accounting
                breakdown.holding += state.store_inventories[s.name] * self.costs.store_holding_cost
                
                backorder = state.store_backorders[s.name]
                if backorder > 0:
                    stockout_periods += 1
                    breakdown.shortage += backorder * self.costs.store_shortage_cost
                    # Non-linear severe shortage penalty if backlog exceeds single-period demand
                    if backorder > d:
                        breakdown.penalties += (backorder - d) * self.costs.severe_shortage_multiplier

            # Transport congestion demurrage
            if scenario.transport_congestion:
                breakdown.transport += self.costs.transport_congestion_penalty

        terminal_backorders = sum(state.store_backorders.values())
        fill_rate = min(1.0, max(0.0, total_demand_served / max(total_demand_occurred, 1e-6)))

        return FutureOutcome(
            replication=replication,
            plan_name=plan.name,
            total_cost=breakdown.total,
            breakdown=breakdown,
            terminal_backorders=terminal_backorders,
            stockout_periods=stockout_periods,
            fill_rate=fill_rate,
        )
