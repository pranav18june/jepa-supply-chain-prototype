"""Legacy five-template baselines; not the active CEM planner action space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class InventoryAction:
    """Action plan at planning anchor time t."""
    name: str
    supplier_orders: Dict[str, float]  # Orders from each supplier to Central DC
    dc_allocations: Dict[str, float]   # Dispatches from Central DC to each store
    is_expedited: bool = False
    reserve_ratio: float = 0.0          # Portion of DC stock reserved centrally

    @property
    def total_order_quantity(self) -> float:
        return sum(self.supplier_orders.values())

    @property
    def total_store_allocation(self) -> float:
        return sum(self.dc_allocations.values())

    def to_feature_vector(self) -> np.ndarray:
        """Vectorized feature representation of the action plan for model backbones."""
        primary_order = self.supplier_orders.get("supplier_primary", 0.0)
        secondary_order = self.supplier_orders.get("supplier_secondary", 0.0)
        total_order = primary_order + secondary_order
        supplier_mix = secondary_order / max(total_order, 1e-6)
        
        alloc_a = self.dc_allocations.get("store_A", 0.0)
        alloc_b = self.dc_allocations.get("store_B", 0.0)
        alloc_c = self.dc_allocations.get("store_C", 0.0)
        total_alloc = alloc_a + alloc_b + alloc_c
        
        return np.array([
            float(np.log1p(total_order)),
            float(supplier_mix),
            float(1.0 if self.is_expedited else 0.0),
            float(self.reserve_ratio),
            float(np.log1p(total_alloc)),
            float(alloc_a / max(total_alloc, 1e-6)),
            float(alloc_b / max(total_alloc, 1e-6)),
            float(alloc_c / max(total_alloc, 1e-6)),
        ], dtype=np.float32)


def generate_candidate_plans(
    dc_inventory: float,
    store_inventories: Dict[str, float],
    store_backorders: Dict[str, float],
    recent_store_demands: Dict[str, float],
    nominal_horizon: int = 4,
) -> Dict[str, InventoryAction]:
    """Generate a diverse set of feasible candidate action plans."""
    total_store_demand = sum(recent_store_demands.values())
    total_backorder = sum(store_backorders.values())
    total_store_inv = sum(store_inventories.values())

    # Base replenishment quantity covering expected demand + backlog gap
    target_inventory = nominal_horizon * total_store_demand
    base_order = max(10.0, min(120.0, target_inventory + total_backorder - dc_inventory - total_store_inv))

    # Base allocation proportions based on current store net requirements
    store_needs = {
        store: max(0.0, 2.0 * recent_store_demands[store] + store_backorders[store] - store_inventories[store])
        for store in store_inventories
    }
    total_need = max(1e-6, sum(store_needs.values()))
    alloc_shares = {store: need / total_need for store, need in store_needs.items()}

    available_dc = max(0.0, dc_inventory)
    base_dispatch = min(available_dc, sum(store_needs.values()))

    # 1. Regular plan: single-source from primary supplier, balanced allocation
    regular = InventoryAction(
        name="regular",
        supplier_orders={"supplier_primary": base_order, "supplier_secondary": 0.0},
        dc_allocations={s: base_dispatch * alloc_shares[s] for s in store_inventories},
        is_expedited=False,
        reserve_ratio=0.0,
    )

    # 2. Split plan: dual-sourcing 60/40 between primary and secondary to mitigate supply disruption
    split = InventoryAction(
        name="split",
        supplier_orders={"supplier_primary": 0.55 * base_order, "supplier_secondary": 0.45 * base_order},
        dc_allocations={s: base_dispatch * alloc_shares[s] for s in store_inventories},
        is_expedited=False,
        reserve_ratio=0.0,
    )

    # 3. Expedited plan: fast-tracked shipping from secondary supplier (shorter lead time)
    expedited = InventoryAction(
        name="expedited",
        supplier_orders={"supplier_primary": 0.30 * base_order, "supplier_secondary": 0.70 * base_order},
        dc_allocations={s: base_dispatch * alloc_shares[s] for s in store_inventories},
        is_expedited=True,
        reserve_ratio=0.0,
    )

    # 4. Reserve central: retain 40% of DC stock centrally as buffer against impending disruption
    reserve = InventoryAction(
        name="reserve_central",
        supplier_orders={"supplier_primary": 0.60 * base_order, "supplier_secondary": 0.40 * base_order},
        dc_allocations={s: (0.60 * base_dispatch) * alloc_shares[s] for s in store_inventories},
        is_expedited=False,
        reserve_ratio=0.40,
    )

    # 5. Priority store: concentrate available stock on the store with worst stockout risk
    most_vulnerable_store = max(store_needs.items(), key=lambda item: item[1])[0]
    priority_allocations = {s: 0.15 * base_dispatch for s in store_inventories}
    priority_allocations[most_vulnerable_store] = base_dispatch - 0.15 * base_dispatch * (len(store_inventories) - 1)
    
    priority = InventoryAction(
        name="priority_store",
        supplier_orders={"supplier_primary": 0.50 * base_order, "supplier_secondary": 0.50 * base_order},
        dc_allocations=priority_allocations,
        is_expedited=False,
        reserve_ratio=0.0,
    )

    return {
        "regular": regular,
        "split": split,
        "expedited": expedited,
        "reserve_central": reserve,
        "priority_store": priority,
    }
