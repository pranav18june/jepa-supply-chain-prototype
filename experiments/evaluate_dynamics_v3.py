"""Pre-registered decision_v3 evaluation: rollouts (P3), paired action effects (P1), ranking (P2).

Test episodes are replayed from manifest seeds and logged actions; replay must
reproduce every logged next state exactly (CRN determinism check). Paired
branches are deep copies of the replayed environment, so every branch shares
demand and lead-time randomness and differs only in the intervention.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from prototype_new.src.env.isomorph import DecisionEnv
from prototype_new.src.models.controlled import (DeltaDynamics, LatentDynamics, RidgeDynamics, ConstantDelta, Ensemble)
from prototype_new.experiments.coverage_report import groups

PROBE_DAYS, H_COST, HORIZONS, ROLL_STARTS = (8, 20, 32, 44), 7, (1, 3, 7, 14), range(0, 46, 5)


def load_model(path):
    p = torch.load(path, map_location='cpu', weights_only=False); D, A, cfg = p['D'], p['A'], p['cfg']
    if p['kind'] == 'delta_mlp': m = DeltaDynamics(D, A, cfg['width'])
    elif p['kind'] == 'latent': m = LatentDynamics(D, A, p['objective'], cfg['latent_dim'], cfg['width'])
    elif p['kind'] == 'ridge': m = RidgeDynamics(D, A)
    else: m = ConstantDelta(D, A, p['use_mean'])
    m.load_state_dict(p['state_dict']); return m.eval()


def forecast_clone(env):
    """Known simulator mechanics; future demand = current EMA, lead-time noise = 0 (mean)."""
    c = env.clone(); sim = c.sim
    table = [dict(t) for t in c.demand_table]
    for day in range(c.day + 1, len(table)):
        table[day] = {i: int(round(sim.demand_ema[i])) for i in sim.item_order}
    sim.demand_fn = lambda day, _t=tuple(table): dict(_t[day])
    for node in sim.network.nodes.values():
        if hasattr(node, 'lead_z'):
            node.lead_z = node.lead_z.copy(); node.lead_z[c.day + 1:] = 0.
    return c


def sim_rollout(env, actions):
    states, transport = [], []
    for a in actions:
        s, _, _, info = env.step(np.asarray(a, dtype=np.int64)); states.append(s); transport.append(info['transport_units'])
    return np.asarray(states), np.asarray(transport, dtype=np.float32)


class Cost:
    def __init__(self, names, costs):
        self.inv = [i for i, n in enumerate(names) if n.endswith('/inventory')]
        self.bl = [i for i, n in enumerate(names) if n.startswith('NewYork/') and n.endswith('/backlog')]
        self.w = costs

    def __call__(self, states, transport):  # [..., H, D], [..., H] -> [..., H]
        return (self.w['holding'] * states[..., self.inv].sum(-1) + self.w['backlog'] * states[..., self.bl].sum(-1)
                + self.w['transport'] * transport)


def replay(m, actions, cfg):
    env = DecisionEnv(cfg['days'], cfg['n_items'], m['scenario'], cfg['maximum'], crn=True); env.reset(m['seed'])
    snaps = {}
    for t, a in enumerate(actions):
        snaps[t] = env.clone(); env.step(a.astype(np.int64))
    return env, snaps


def candidates(env, rng, maximum):
    nat = env.native_action().astype(float)
    c = [np.zeros_like(nat), nat, .5*nat, 2*nat, np.full_like(nat, maximum)] + [rng.uniform(0, maximum, len(nat)) for _ in range(3)]
    return np.stack([env.space.project(x) for x in c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='prototype_new/results/decision_v3d/data')
    ap.add_argument('--models', default='prototype_new/results/decision_v3d/models')
    ap.add_argument('--output', default='prototype_new/results/decision_v3d/evaluation')
    ap.add_argument('--splits', nargs='*', default=['test', 'test_ood'])
    ap.add_argument('--max-episodes', type=int, default=None)
    ap.add_argument('--model-names', nargs='*', default=None)
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    d = dict(np.load(Path(args.data)/'transitions.npz')); man = json.loads((Path(args.data)/'manifest.json').read_text())
    cfg, names = man['config'], man['state_names']; G = groups(names)
    costs = dict(holding=.001, backlog=1., transport=.01); cost = Cost(names, costs)
    mdir = Path(args.models)
    singles = sorted(p.stem for p in mdir.glob('*.pt') if args.model_names is None or p.stem in args.model_names)
    models = {n: load_model(mdir/f'{n}.pt') for n in singles}
    if any(n.startswith('delta_mlp_') for n in singles):
        models['delta_mlp_ensemble'] = Ensemble([models[n] for n in singles if n.startswith('delta_mlp_')])
    item_pipe = [[i for i, n in enumerate(names) if n.startswith(f'pipeline/{it}/')] for it in sorted({n.split('/')[1] for n in names if n.startswith('item/')})]
    results = {}
    for split in args.splits:
        eps = [(e, m) for e, m in enumerate(man['episodes']) if m['split'] == split][:args.max_episodes]
        R = {k: {} for k in ('rollout', 'p1', 'p2', 'mono', 'unc')}
        mismatches = 0
        for e, m in eps:
            sel = d['episode'] == e; S, Aa, N, T = d['state'][sel], d['action'][sel], d['next_state'][sel], d['transport_units'][sel]
            env, snaps = replay(m, Aa, cfg)
            mismatches += int(not np.array_equal(np.asarray([r['next_state'] for r in env.records], dtype=np.float32), N))
            # ---- P3 open-loop rollouts on logged actions
            st = torch.tensor(S[list(ROLL_STARTS)]); acts = torch.tensor(np.stack([Aa[t:t+14] for t in ROLL_STARTS]))
            truth = np.stack([N[t:t+14] for t in ROLL_STARTS]); ttruth = np.stack([T[t:t+14] for t in ROLL_STARTS])
            preds = {}
            with torch.no_grad():
                for name, mod in models.items():
                    ps, pt = mod.rollout(st, acts); preds[name] = (ps.numpy(), pt.numpy())
            hs, ht = zip(*[sim_rollout(forecast_clone(snaps[t]), Aa[t:t+14]) for t in ROLL_STARTS])
            preds['sim_forecast'] = (np.stack(hs), np.stack(ht))
            for name, (ps, pt) in preds.items():
                row = R['rollout'].setdefault(name, [])
                ent = {f'{g}_h{h}': float(np.abs(ps[:, h-1, ix] - truth[:, h-1, ix]).mean()) for g, ix in G.items() for h in HORIZONS}
                pc, tc = cost(ps[:, :H_COST], pt[:, :H_COST]).sum(-1), cost(truth[:, :H_COST], ttruth[:, :H_COST]).sum(-1)
                ent['cost7_abs_err'] = float(np.abs(pc - tc).mean()); ent['cost7_true'] = float(tc.mean())
                row.append(ent)
            # ---- P1/P2 paired probes
            rng = np.random.default_rng([m['seed'], 991])
            for t in PROBE_DAYS:
                base = snaps[t]; C = candidates(base, rng, cfg['maximum']); cont = Aa[t+1:t+H_COST]
                seqs = np.stack([np.vstack([c[None], cont]) for c in C])
                real = [sim_rollout(base.clone(), s) for s in seqs]
                rs = np.stack([r[0] for r in real]); rt = np.stack([r[1] for r in real])
                rcost = cost(rs, rt).sum(-1)
                s0 = torch.tensor(S[t]).expand(len(C), -1)
                probe_preds = {}
                with torch.no_grad():
                    for name, mod in models.items():
                        if name == 'delta_mlp_ensemble':
                            ps, pt, mst, mtr = mod.rollout(s0, torch.tensor(seqs, dtype=torch.float32), return_members=True)
                            mc = np.stack([cost(a.numpy(), b.numpy()).sum(-1) for a, b in zip(mst, mtr)])
                            R['unc'].setdefault(name, []).append(dict(std=mc.std(0).tolist(), err=np.abs(mc.mean(0) - rcost).tolist()))
                        else:
                            ps, pt = mod.rollout(s0, torch.tensor(seqs, dtype=torch.float32))
                        probe_preds[name] = (ps.numpy(), pt.numpy())
                fc = [sim_rollout(forecast_clone(base), s) for s in seqs]
                probe_preds['sim_forecast'] = (np.stack([f[0] for f in fc]), np.stack([f[1] for f in fc]))
                true_delta = rs[1:, 0] - rs[0, 0]
                for name, (ps, pt) in probe_preds.items():
                    pd = ps[1:, 0] - ps[0, 0]
                    ent = {f'{g}_mae': float(np.abs(pd[:, ix] - true_delta[:, ix]).mean()) for g, ix in G.items()}
                    ent.update({f'{g}_zero_mae': float(np.abs(true_delta[:, ix]).mean()) for g, ix in G.items()})
                    aff = G['pipeline'] + G['lastmile_wh'] + G['dest_inventory'] + G['dest_backlog']
                    ent['affected_mae'] = float(np.abs(pd[:, aff] - true_delta[:, aff]).mean())
                    ent['affected_zero_mae'] = float(np.abs(true_delta[:, aff]).mean())
                    tv, pv = true_delta[:, aff].ravel(), pd[:, aff].ravel(); nz = np.abs(tv) > .5
                    ent['sign_agree_nonzero'] = float((np.sign(pv[nz]) == np.sign(tv[nz])).mean()) if nz.any() else None
                    R['p1'].setdefault(name, []).append(dict(episode=e, day=t, **ent))
                    pcost = cost(ps, pt).sum(-1); rho = spearmanr(pcost, rcost)[0]
                    R['p2'].setdefault(name, []).append(dict(episode=e, day=t, spearman=None if np.isnan(rho) else float(rho),
                        regret=float(rcost[np.argmin(pcost)] - rcost.min()), regret_vs_native=float(rcost[np.argmin(pcost)] - rcost[1]),
                        cost_mae=float(np.abs(pcost - rcost).mean()), best_true=float(rcost.min())))
                # ---- monotonicity / cross-item sanity on item 0 request sweep
                grid = np.linspace(0, cfg['maximum'], 11)
                nat = base.native_action().clip(0, cfg['maximum'])
                sweep = np.stack([np.r_[q, nat[1:]] for q in grid]).astype(np.int64)
                true1 = np.stack([sim_rollout(base.clone(), s[None])[0][0] for s in sweep])
                tp0 = true1[:, item_pipe[0]].sum(1); tcross = true1[:, item_pipe[1] + item_pipe[2]].sum(1)
                with torch.no_grad():
                    for name, mod in models.items():
                        ps = mod.rollout(torch.tensor(S[t]).expand(len(grid), -1), torch.tensor(sweep, dtype=torch.float32)[:, None])[0][:, 0].numpy()
                        pp0 = ps[:, item_pipe[0]].sum(1); pcross = ps[:, item_pipe[1] + item_pipe[2]].sum(1)
                        R['mono'].setdefault(name, []).append(dict(
                            true_monotone=bool((np.diff(tp0) >= -1e-6).all()),
                            pred_monotone_rate=float((np.diff(pp0) >= -1e-3).mean()),
                            true_range=float(tp0.max() - tp0.min()), pred_range=float(pp0.max() - pp0.min()),
                            true_cross_range=float(tcross.max() - tcross.min()), pred_cross_range=float(pcross.max() - pcross.min())))
            print(split, e, flush=True)
        R['replay_mismatched_episodes'] = mismatches
        results[split] = R
    (out/'raw.json').write_text(json.dumps(results))
    print('replay mismatches', {s: results[s]['replay_mismatched_episodes'] for s in results})


if __name__ == '__main__':
    main()
