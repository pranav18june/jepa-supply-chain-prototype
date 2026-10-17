"""ISOMORPH day-level adapter. Dispatch requests are actions, shipments outcomes.

Observation is end-of-previous-day inventory/backlog, outstanding order quantity
and days to arrival, demand EMA, destination pipeline by remaining arrival day.
No future demand signal is exposed. Fixed topology and catalog ordering.
"""
from __future__ import annotations
import contextlib
import io
import random
from dataclasses import dataclass, asdict
import numpy as np
from ..vendor.isomorph import simulator as upstream


@dataclass(frozen=True)
class CostWeights:
    """Assumed accounting units, NOT monetary costs supplied by ISOMORPH."""
    holding: float = 0.001
    backlog: float = 1.0
    transport: float = 0.01

    def __post_init__(self):
        if not all(np.isfinite(x) and x >= 0 for x in asdict(self).values()):
            raise ValueError('Costs must be finite and nonnegative')


@dataclass(frozen=True)
class DispatchSpace:
    size: int
    maximum: int = 1000

    def __post_init__(self):
        if self.size < 1 or self.maximum < 1:
            raise ValueError('Positive action size and maximum required')

    def project(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape[-1:] != (self.size,) or not np.isfinite(a).all():
            raise ValueError('Invalid dispatch action shape or nonfinite values')
        return np.rint(np.clip(a, 0, self.maximum)).astype(np.int64)

    def validate(self, action):
        a = np.asarray(action)
        if a.shape != (self.size,) or not np.array_equal(a, self.project(a)):
            raise ValueError('Action must be an integer request within bounds')


def build_simulation(seed, days, scenario, n_items=3, module=upstream):
    """Catalog restriction matches the supported v2 first-N catalog convention.

    Network capacities are those of the original 50-item builder (not rescaled).
    Source seeds retain upstream semantics, including fixed network seed 2025.
    """
    if not 1 <= n_items <= 50 or days < 1:
        raise ValueError('Invalid catalog size or horizon')
    state = random.getstate()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sim, net, items, _ = module.build_example_simulation_from_adjacency(
                seed=seed, horizon_days=days, pipeline_multiplier=7., scenario=scenario)
    finally:
        random.setstate(state)
    selected = sorted(items)[:n_items]
    sim.items = {i: items[i] for i in selected}
    sim.item_order = selected
    for field in ('demand_ema', 'item_intransit', 'svc_demand', 'svc_served', 'svc_backlog'):
        setattr(sim, field, {i: getattr(sim, field)[i] for i in selected})
    for node in net.nodes.values():
        for field in ('inventory', 'backlog', 's_levels', 'S_levels', 'lead_time_mean', 'outstanding_orders'):
            setattr(node, field, {i: v for i, v in getattr(node, field).items() if i in selected})
    return sim


class IsomorphEnv:
    def __init__(self, days=32, n_items=3, scenario=None, maximum=1000,
                 costs=CostWeights()):
        self.days, self.n_items = days, n_items
        self.scenario = dict(scenario or {})
        self.space = DispatchSpace(n_items, maximum)
        self.costs = costs
        self.sim = None

    def reset(self, seed=0):
        self.seed = seed
        self.sim = build_simulation(seed, self.days, self.scenario, self.n_items)
        self.day = 0
        self.records = []
        # Bound representation of destination pipeline by maximum original route.
        self.pipeline_horizon = max(1, max(int(np.ceil(self.sim._total_tt(pe)))
            for _, _, pe in self.sim.network.paths_to_dest.values()))
        return self.observe()

    def observe(self):
        values = []
        for nid in sorted(self.sim.network.nodes):
            node = self.sim.network.nodes[nid]
            for iid in self.sim.item_order:
                order = node.outstanding_orders.get(iid)
                values.extend([node.inventory.get(iid, 0), node.backlog.get(iid, 0),
                               order[1] if order else 0,
                               max(0, order[0] - self.day) if order else 0])
        for iid in self.sim.item_order:
            values.append(self.sim.demand_ema[iid])
            values.extend(self.sim.dest_in_transit.get(self.day + k, {}).get(iid, 0)
                          for k in range(self.pipeline_horizon + 1))
        # Observable calendar; no future demand or RNG state.
        values.extend([np.sin(2*np.pi*self.day/7), np.cos(2*np.pi*self.day/7)])
        return np.asarray(values, dtype=np.float32)

    def step(self, action=None):
        if self.sim is None or self.day >= self.days:
            raise RuntimeError('reset required or episode finished')
        before = self.observe()
        if action is not None:
            self.space.validate(action)
            chosen = np.asarray(action, dtype=np.int64).copy()
            self.sim.external_dispatch_targets = dict(zip(self.sim.item_order, chosen))
        else:
            chosen = None
            self.sim.external_dispatch_targets = None
        start = len(self.sim.shipments_log)
        self.sim.step(self.day)
        shipments = self.sim.shipments_log[start:]
        daily = self.sim.daily_records[-self.n_items:]
        holding = sum(sum(n.inventory.values()) for n in self.sim.network.nodes.values())
        backlog = sum(self.sim.network.nodes[self.sim.destination_id].backlog.values())
        # Unit-edge count, explicitly an assumed transport accounting model.
        transport = sum(r['units'] * (len(r['path_nodes'])-1) for r in shipments)
        components = dict(holding=self.costs.holding*holding,
                          backlog=self.costs.backlog*backlog,
                          transport=self.costs.transport*transport)
        self.day += 1
        after = self.observe()
        info = dict(day=self.day-1, action=None if chosen is None else chosen.tolist(),
                    action_kind='native_policy' if chosen is None else 'dispatch_target',
                    realized_dispatch=[sum(r['units'] for r in shipments
                        if r['to']==self.sim.destination_id and r['item']==i)
                        for i in self.sim.item_order],
                    components=components, cost=sum(components.values()),
                    demand=sum(r['demand'] for r in daily),
                    served=sum(r['served_from_stock'] for r in daily),
                    shipments=shipments)
        self.records.append(dict(state=before.tolist(), next_state=after.tolist(), **info))
        return after, info['cost'], self.day == self.days, info


# ---------------------------------------------------------------------------
# Decision-time protocol (round_claude, 2026-10-17)
# ---------------------------------------------------------------------------

class _KeyedLeadTimeNode(upstream.Node):
    """Upstream Node whose source lead-time draw is keyed, not sequential.

    Same (s,S) logic and Normal(mean, 0.2*mean) lead-time distribution as the MIT
    upstream ``maybe_place_orders``, but the standard-normal draw is indexed by
    (day, item) per node, so changing actions cannot shift the lead-time stream of
    other orders (common random numbers). Deep copies remain independent.
    """

    def maybe_place_orders(self, day, rng):
        if self.is_destination or not self.is_source:
            return
        for item_id, s in self.s_levels.items():
            on_hand = self.inventory.get(item_id, 0)
            if on_hand < s and self.outstanding_orders.get(item_id) is None:
                qty = max(self.S_levels.get(item_id, on_hand) - on_hand, 0)
                if qty <= 0:
                    continue
                mean_lt = max(self.lead_time_mean.get(item_id, 1.0), 0.1)
                z = self.lead_z[day, self.lead_item_index[item_id]]
                sampled = mean_lt + self.lead_time_std_frac * mean_lt * z
                self.outstanding_orders[item_id] = (day + max(1, int(np.ceil(sampled))), qty)


class DecisionEnv:
    """ISOMORPH dispatch control with post-demand decision timing.

    Day t: arrivals and demand of day t are realized first; the controller then
    observes the state and chooses per-item destination dispatch requests; the
    shipping, replenishment and source-ordering of day t follow; then arrivals
    and demand of day t+1 are realized and the next observation is formed. This
    is the information set used by the built-in rule, so native and external
    controllers are information-matched.

    Exogenous randomness: demand is tabulated at reset from the upstream
    generator (identical values to sequential upstream calls) and lead-time
    draws are keyed by (day, node, item) when ``crn=True``. Both are therefore
    independent of the action history, and ``clone()`` gives exact paired
    counterfactual branches.

    Last-mile capacity in the upstream builder is sized from the realized
    horizon-mean demand intensity of the 50-item catalog. It is exposed in the
    observation as known exogenous lane-capacity metadata for every controller.
    """

    def __init__(self, days=60, n_items=3, scenario=None, maximum=1000,
                 costs=CostWeights(), crn=True):
        self.days, self.n_items = days, n_items
        self.scenario = dict(scenario or {})
        self.space = DispatchSpace(n_items, maximum)
        self.costs, self.crn = costs, crn
        self.sim = None

    def reset(self, seed=0):
        self.seed = seed
        # One extra day so the final transition's next-day demand exists.
        sim = build_simulation(seed, self.days + 1, self.scenario, self.n_items)
        table = [sim.demand_fn(d) for d in range(self.days + 1)]
        sim.demand_fn = lambda day, _t=tuple(table): dict(_t[day])
        self.demand_table = table
        if self.crn:
            names = sorted(sim.network.nodes)
            draws = np.random.default_rng([seed, 7919]).standard_normal(
                (self.days + 1, len(names), len(sim.item_order)))
            index = {i: b for b, i in enumerate(sim.item_order)}
            for a, name in enumerate(names):
                node = sim.network.nodes[name]
                node.__class__ = _KeyedLeadTimeNode
                node.lead_z, node.lead_item_index = draws[:, a, :], index
        self.sim = sim
        self.pipeline_horizon = max(1, max(int(np.ceil(sim._total_tt(pe)))
            for _, _, pe in sim.network.paths_to_dest.values()))
        self.last_mile = [(u, v) for (u, v) in sim.network.edges if v == sim.destination_id]
        self.day = 0
        self.records = []
        sim.step_before_dispatch(0)
        return self.observe()

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def state_names(self):
        names = []
        for nid in sorted(self.sim.network.nodes):
            for iid in self.sim.item_order:
                names += [f'{nid}/{iid}/{k}' for k in ('inventory', 'backlog', 'order_qty', 'order_eta')]
        for iid in self.sim.item_order:
            names += [f'item/{iid}/demand_ema', f'item/{iid}/demand_today']
            names += [f'pipeline/{iid}/{k}' for k in range(1, self.pipeline_horizon + 1)]
        names += ['calendar/sin7', 'calendar/cos7']
        names += [f'capacity/{u}->{v}' for u, v in self.last_mile]
        return names

    def observe(self):
        sim, day = self.sim, self.day
        values = []
        for nid in sorted(sim.network.nodes):
            node = sim.network.nodes[nid]
            for iid in sim.item_order:
                order = node.outstanding_orders.get(iid)
                values += [node.inventory.get(iid, 0), node.backlog.get(iid, 0),
                           order[1] if order else 0, max(0, order[0] - day) if order else 0]
        today = sim.daily_records[-self.n_items:]
        demand_today = {r['item']: r['demand'] for r in today if r['day'] == day}
        for iid in sim.item_order:
            values += [sim.demand_ema[iid], demand_today.get(iid, 0)]
            values += [sim.dest_in_transit.get(day + k, {}).get(iid, 0)
                       for k in range(1, self.pipeline_horizon + 1)]
        values += [np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7)]
        values += [sim.network.edges[e].daily_total_capacity for e in self.last_mile]
        return np.asarray(values, dtype=np.float32)

    def native_action(self):
        """Built-in rule evaluated on the current (post-demand) state, unclipped."""
        sim = self.sim
        dest = sim.network.nodes[sim.destination_id]
        out = []
        for iid in sim.item_order:
            cb, it, oh = dest.backlog.get(iid, 0), sim.item_intransit.get(iid, 0), dest.inventory.get(iid, 0)
            if sim.pipeline_multiplier > 0:
                target = max(0, int(np.ceil(cb + sim.demand_ema[iid]*sim.pipeline_multiplier - it - oh)))
            else:
                target = max(0, cb + max(1, int(sim.demand_ema[iid]*3)) - oh - it)
            out.append(target)
        return np.asarray(out, dtype=np.int64)

    def step(self, action):
        if self.sim is None or self.day >= self.days:
            raise RuntimeError('reset required or episode finished')
        self.space.validate(action)
        sim, chosen = self.sim, np.asarray(action, dtype=np.int64).copy()
        before = self.observe()
        sim.external_dispatch_targets = dict(zip(sim.item_order, chosen))
        start = len(sim.shipments_log)
        sim.step_dispatch_and_after(self.day)
        shipments = sim.shipments_log[start:]
        self.day += 1
        sim.step_before_dispatch(self.day)
        after = self.observe()
        dest = sim.network.nodes[sim.destination_id]
        holding = sum(sum(n.inventory.values()) for n in sim.network.nodes.values())
        backlog = sum(dest.backlog.values())
        transport = sum(r['units'] * (len(r['path_nodes']) - 1) for r in shipments)
        components = dict(holding=self.costs.holding*holding, backlog=self.costs.backlog*backlog,
                          transport=self.costs.transport*transport)
        today = [r for r in sim.daily_records[-self.n_items:] if r['day'] == self.day]
        used = {e: 0. for e in self.last_mile}
        for r in shipments:
            edge = tuple(r['path_nodes'][-2:])
            if edge in used:
                used[edge] += r['units'] * sim.items[r['item']].volume
        info = dict(day=self.day-1, seed=self.seed, action=chosen.tolist(),
                    lastmile_utilization=[used[e]/max(sim.network.edges[e].daily_total_capacity, 1e-9)
                                          for e in self.last_mile],
                    realized_dispatch=[sum(r['units'] for r in shipments
                        if r['to'] == sim.destination_id and r['item'] == i) for i in sim.item_order],
                    transport_units=transport, holding_units=holding, backlog_units=backlog,
                    components=components, cost=sum(components.values()),
                    demand=sum(r['demand'] for r in today),
                    served=sum(r['served_from_stock'] for r in today))
        self.records.append(dict(state=before.tolist(), next_state=after.tolist(), **info))
        return after, info['cost'], self.day == self.days, info
