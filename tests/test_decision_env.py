"""Decision-time ISOMORPH protocol: timing, pairing and clone semantics."""
import numpy as np
import pytest
from prototype_new.src.env.isomorph import DecisionEnv, build_simulation

SC = {'ss_scale': .08, 'base_lambda_lo': 32., 'base_lambda_hi': 48.}


@pytest.mark.parametrize('seed', [3, 41])
def test_native_rule_via_external_matches_upstream_builtin(seed):
    days = 30
    env = DecisionEnv(days, 3, SC, maximum=5000, crn=False); env.reset(seed)
    ref = build_simulation(seed, days + 1, SC, 3)
    for day in range(days):
        action = env.native_action()
        env.step(action)
        ref.step(day)
    ref.step_before_dispatch(days)
    for field in ('daily_records', 'shipments_log', 'demand_ema', 'dest_in_transit'):
        assert getattr(ref, field) == getattr(env.sim, field), field
    for nid in ref.network.nodes:
        assert ref.network.nodes[nid].inventory == env.sim.network.nodes[nid].inventory


def test_observation_contains_today_but_no_future_demand():
    env = DecisionEnv(20, 3, SC); obs = env.reset(5)
    names = env.state_names()
    today = [obs[names.index(f'item/{i}/demand_today')] for i in env.sim.item_order]
    assert today == [env.demand_table[0][i] for i in env.sim.item_order]
    other = env.clone()
    other.sim.demand_fn = lambda day: {i: 10**6 for i in other.sim.item_order}
    np.testing.assert_array_equal(other.observe(), obs)
    assert len(names) == len(obs)


def test_clone_is_independent_and_branches_share_exogenous_streams():
    env = DecisionEnv(40, 3, {**SC, 'ss_scale': .04}); env.reset(11)
    for _ in range(5):
        env.step(env.space.project(env.native_action()))
    snapshot = env.observe().copy()
    branches = []
    for q in (0, 900):
        b = env.clone()
        infos = [b.step(np.full(3, q))[3]]
        for _ in range(20):
            infos.append(b.step(b.space.project(b.native_action()))[3])
        branches.append((b, infos))
    np.testing.assert_array_equal(env.observe(), snapshot)   # original untouched
    assert env.day == 5
    (b0, i0), (b1, i1) = branches
    assert [r['demand'] for r in i0] == [r['demand'] for r in i1]
    assert i0[0]['realized_dispatch'] == [0, 0, 0] and sum(i1[0]['realized_dispatch']) > 0
    # Keyed lead-time draws are identical across branches but owned per copy.
    lead = lambda b: b.sim.network.nodes['SanFrancisco'].lead_z
    np.testing.assert_array_equal(lead(b0), lead(b1))
    assert lead(b0) is not lead(env) and not np.shares_memory(lead(b0), lead(env))


def test_step_accounting_and_projection():
    env = DecisionEnv(10, 3, SC); env.reset(2)
    _, cost, _, info = env.step(np.array([100, 0, 50]))
    assert cost == pytest.approx(sum(info['components'].values()))
    assert info['components']['holding'] == pytest.approx(env.costs.holding*info['holding_units'])
    assert env.records[0]['action'] == [100, 0, 50]
    assert all(r <= a for r, a in zip(info['realized_dispatch'], [100, 0, 50]))
    with pytest.raises(ValueError):
        env.step(np.array([1.5, 0, 0]))
