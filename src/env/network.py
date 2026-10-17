"""Multi-echelon retail supply chain network topology and state definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SupplierNode:
    name: str
    nominal_capacity: float
    nominal_lead_time: int
    unit_cost: float
    reliability_tier: str = "primary"  # "primary" or "secondary"


@dataclass(frozen=True)
class StoreNode:
    name: str
    base_demand: float
    holding_cost: float
    stockout_penalty: float
    lead_time_from_dc: int = 1


@dataclass(frozen=True)
class DCNode:
    name: str
    capacity: float
    holding_cost: float


@dataclass(frozen=True)
class NetworkTopology:
    """Topology of a multi-echelon supply chain: Suppliers -> Central DC -> Stores."""
    suppliers: Tuple[SupplierNode, ...]
    dc: DCNode
    stores: Tuple[StoreNode, ...]

    @classmethod
    def standard_retail(cls) -> NetworkTopology:
        """Create standard 2-supplier, 1-DC, 3-store retail topology."""
        return cls(
            suppliers=(
                SupplierNode("supplier_primary", nominal_capacity=50.0, nominal_lead_time=2, unit_cost=1.0, reliability_tier="primary"),
                SupplierNode("supplier_secondary", nominal_capacity=30.0, nominal_lead_time=1, unit_cost=1.35, reliability_tier="secondary"),
            ),
            dc=DCNode("central_dc", capacity=200.0, holding_cost=0.08),
            stores=(
                StoreNode("store_A", base_demand=12.0, holding_cost=0.15, stockout_penalty=1.8, lead_time_from_dc=1),
                StoreNode("store_B", base_demand=16.0, holding_cost=0.15, stockout_penalty=2.0, lead_time_from_dc=1),
                StoreNode("store_C", base_demand=10.0, holding_cost=0.15, stockout_penalty=1.6, lead_time_from_dc=1),
            )
        )


@dataclass
class NetworkState:
    """Dynamic operational state of the multi-echelon network at time t."""
    step: int
    dc_inventory: float
    dc_pipeline: float  # In-transit orders to DC
    store_inventories: Dict[str, float]
    store_backorders: Dict[str, float]
    store_pipelines: Dict[str, float]  # In-transit shipments from DC to stores
    supplier_capacities: Dict[str, float]
    supplier_states: Dict[str, int]  # 0=disrupted, 1=strained, 2=reliable
    congestion: bool
    
    def copy(self) -> NetworkState:
        return NetworkState(
            step=self.step,
            dc_inventory=self.dc_inventory,
            dc_pipeline=self.dc_pipeline,
            store_inventories=dict(self.store_inventories),
            store_backorders=dict(self.store_backorders),
            store_pipelines=dict(self.store_pipelines),
            supplier_capacities=dict(self.supplier_capacities),
            supplier_states=dict(self.supplier_states),
            congestion=self.congestion,
        )
