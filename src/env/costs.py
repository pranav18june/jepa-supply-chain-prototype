"""Operational cost accounting for multi-echelon inventory planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CostParameters:
    """Unit cost parameters for inventory, transportation, ordering and shortage."""
    dc_holding_cost: float = 0.08
    store_holding_cost: float = 0.15
    store_shortage_cost: float = 1.80
    severe_shortage_multiplier: float = 1.50  # Extra penalty when backlog > demand
    fixed_order_cost: float = 1.50
    dual_source_split_penalty: float = 0.40   # Coordination cost for splitting across suppliers
    expedited_shipping_markup: float = 3.00   # Cost premium per unit for expedited orders
    transport_congestion_penalty: float = 0.50 # Extra freight demurrage during congestion


@dataclass
class CostBreakdown:
    holding: float = 0.0
    ordering: float = 0.0
    shortage: float = 0.0
    transport: float = 0.0
    penalties: float = 0.0

    @property
    def total(self) -> float:
        return self.holding + self.ordering + self.shortage + self.transport + self.penalties

    def add(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            holding=self.holding + other.holding,
            ordering=self.ordering + other.ordering,
            shortage=self.shortage + other.shortage,
            transport=self.transport + other.transport,
            penalties=self.penalties + other.penalties,
        )
