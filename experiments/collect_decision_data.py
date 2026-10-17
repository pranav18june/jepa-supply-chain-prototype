"""Collect action-labeled ISOMORPH transitions under the decision_v3 protocol.

Every transition stores the executed (validated, projected) request, the
post-demand state before it, the next post-demand state, realized dispatch,
cost components and exogenous identifiers. Behavior policies only use the
observation and their own RNG; none reads the demand table or lead-time draws.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from prototype_new.src.env.isomorph import DecisionEnv
from prototype_new.src.data.protocols import configuration_hash, validate_rollout_splits


def sample_scenario(spec, rng):
    draw = lambda v: float(rng.uniform(*v)) if isinstance(v, list) else float(v)
    sc = {k: draw(v) for k, v in spec.items() if k != 'lambda_width'}
    sc['base_lambda_hi'] = sc['base_lambda_lo'] + draw(spec['lambda_width'])
    return sc


def basestock(env, k):
    sim = env.sim; dest = sim.network.nodes[sim.destination_id]
    return np.array([max(0., k[j]*sim.demand_ema[i] + dest.backlog.get(i, 0) - dest.inventory.get(i, 0)
                         - sim.item_intransit[i]) for j, i in enumerate(sim.item_order)])


class Behavior:
    """Randomized observation-only behavior policy families."""

    def __init__(self, family, n, maximum, rng, extreme_prob=0., starve_range=(4, 15)):
        self.family, self.n, self.maximum, self.rng = family, n, maximum, rng
        self.extreme_prob, self.starve_range = extreme_prob, tuple(starve_range)
        r = rng
        self.params = dict(sigma=float(r.uniform(.1, .7)), p_zero=float(r.uniform(0, .2)),
                           k=r.uniform(0, 14, n).tolist(), period=int(r.integers(6, 20)),
                           on=int(r.integers(1, 6)), block=int(r.integers(5, 15)))
        if family == 'starve_recover':  # extra draws only for this family keep v3a streams unchanged
            lo, hi = self.starve_range
            self.params.update(starve=int(r.integers(lo, hi)), recover=int(r.integers(5, 20)))
        self.current, self.hold = None, 0
        self.item_family = [None]*n

    def _family_action(self, family, env, day):
        r, p, n, m = self.rng, self.params, self.n, self.maximum
        if family == 'native_noise':
            a = env.native_action() * r.lognormal(0, p['sigma'], n)
            return a * (r.random(n) > p['p_zero'])
        if family == 'basestock':
            return basestock(env, np.asarray(p['k'])) * r.lognormal(0, .15, n)
        if family == 'uniform':
            a = r.uniform(0, m, n)
            if self.extreme_prob > 0:
                a = np.where(r.random(n) < self.extreme_prob, m*(r.random(n) < .5), a)
            return a
        if family == 'intermittent':
            return r.uniform(0, m, n) if day % p['period'] < p['on'] else np.zeros(n)
        if family == 'starve_recover':
            cycle = p['starve'] + p['recover']
            return np.zeros(n) if day % cycle < p['starve'] else basestock(env, np.asarray(p['k'])/2 + 3)
        if family == 'piecewise_constant':
            if self.hold <= 0:
                self.current, self.hold = r.uniform(0, .6*m, n), int(r.integers(3, 12))
            self.hold -= 1
            return self.current
        raise ValueError(family)

    def __call__(self, env, day):
        if self.family != 'mixed_item':
            return self._family_action(self.family, env, day)
        if day % self.params['block'] == 0:
            options = ['native_noise', 'basestock', 'uniform', 'intermittent'] + (['starve_recover'] if 'starve' in self.params else [])
            self.item_family = [options[i] for i in self.rng.integers(0, len(options), self.n)]
        full = {f: self._family_action(f, env, day) for f in set(self.item_family)}
        return np.array([full[f][j] for j, f in enumerate(self.item_family)])


def collect(cfg):
    manifest, episodes, count = [], [], 0
    for split, number in cfg['splits'].items():
        ood = split == 'test_ood'
        families = cfg['behaviors_ood' if ood else 'behaviors_iid']
        for j in range(number):
            seed = cfg['seed_base'] + count; count += 1
            rng = np.random.default_rng([seed, 17])
            scenario = sample_scenario(cfg['scenario_ood' if ood else 'scenario_iid'], rng)
            family = families[j % len(families)]
            behavior = Behavior(family, cfg['n_items'], cfg['maximum'], rng, cfg.get('uniform_extreme_prob', 0.),
                                cfg.get('starve_range', (4, 15)))
            env = DecisionEnv(cfg['days'], cfg['n_items'], scenario, cfg['maximum'], crn=True)
            env.reset(seed)
            for day in range(cfg['days']):
                env.step(env.space.project(behavior(env, day)))
            manifest.append(dict(rollout=f'{split}_{j}', split=split, seed=seed, scenario=scenario,
                                 config_hash=configuration_hash(dict(scenario=scenario, seed=seed)),
                                 behavior=family, behavior_params=behavior.params,
                                 demand_table=[[d[i] for i in env.sim.item_order] for d in env.demand_table]))
            episodes.append(env.records)
    validate_rollout_splits([dict(m, split='test' if m['split'] == 'test_ood' else m['split']) for m in manifest])
    return manifest, episodes, env.state_names()


def to_arrays(manifest, episodes):
    keys = ['state', 'action', 'next_state', 'realized_dispatch', 'lastmile_utilization']
    scalars = ['transport_units', 'holding_units', 'backlog_units', 'demand', 'served', 'cost', 'day', 'seed']
    out = {k: [] for k in keys + scalars + ['episode']}
    for e, records in enumerate(episodes):
        for r in records:
            for k in keys + scalars:
                out[k].append(r[k])
            out['episode'].append(e)
    return {k: np.asarray(v, dtype=np.float64 if k in ('cost',) else np.float32 if k not in ('episode', 'day', 'seed') else np.int64)
            for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='prototype_new/configs/decision_v3.json')
    parser.add_argument('--output', default='prototype_new/results/decision_v3/data')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text()); out = Path(args.output)
    if (out/'transitions.npz').exists():
        raise FileExistsError('Use a new output directory to preserve results')
    out.mkdir(parents=True, exist_ok=True)
    manifest, episodes, names = collect(cfg)
    arrays = to_arrays(manifest, episodes)
    np.savez_compressed(out/'transitions.npz', **arrays)
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg, config_hash=configuration_hash(cfg),
                                                     state_names=names, episodes=manifest), indent=1))
    print({s: sum(m['split'] == s for m in manifest) for s in cfg['splits']}, len(arrays['cost']), 'transitions')


if __name__ == '__main__':
    main()
