"""Aggregate decision_v3 evaluation: seed-averaged method metrics with episode-clustered bootstrap CIs."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def method_of(name):
    return re.sub(r'_\d+$', '', name) if re.search(r'_\d+$', name) and not name.endswith('ensemble') else name


def per_episode(rows, key):
    by = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by[r['episode']].append(r[key])
    return {e: float(np.mean(v)) for e, v in by.items()}


def boot(diff, n=4000, seed=0):
    diff = np.asarray(diff); rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--evaluation', default='prototype_new/results/decision_v3d/evaluation')
    args = ap.parse_args(); root = Path(args.evaluation); raw = json.loads((root/'raw.json').read_text())
    summary, md = {}, []
    for split, R in raw.items():
        S = summary[split] = dict(replay_mismatched_episodes=R['replay_mismatched_episodes'], methods={})
        names = sorted(R['p1']); methods = defaultdict(list)
        for n in names: methods[method_of(n)].append(n)
        # rollout rows carry no episode id; they are appended once per episode in fixed order
        n_eps = len(R['rollout'][names[0]])
        ep_index = sorted({r['episode'] for r in R['p1'][names[0]]})
        def ep_metric(member_names, section, key):
            vals = []
            for n in member_names:
                rows = R[section][n]
                if section == 'rollout':
                    vals.append({ep_index[i]: r[key] for i, r in enumerate(rows)})
                else:
                    vals.append(per_episode(rows, key))
            eps = sorted(set.intersection(*[set(v) for v in vals]))
            return eps, np.array([[v[e] for e in eps] for v in vals])   # [seeds, episodes]
        zero_eps, zero = ep_metric(['persistence'], 'p1', 'affected_zero_mae')
        ref = {}
        for m, members in methods.items():
            out = dict(members=members)
            for section, key in [('p1', 'affected_mae'), ('p1', 'sign_agree_nonzero'), ('p1', 'pipeline_mae'), ('p1', 'lastmile_wh_mae'),
                                 ('p1', 'dest_inventory_mae'), ('p1', 'dest_backlog_mae'), ('p1', 'upstream_inventory_mae'),
                                 ('p2', 'spearman'), ('p2', 'regret'), ('p2', 'regret_vs_native'), ('p2', 'cost_mae'),
                                 ('rollout', 'cost7_abs_err')] + [('rollout', f'{g}_h{h}') for g in ('dest_inventory', 'dest_backlog', 'pipeline', 'lastmile_wh', 'upstream_inventory') for h in (1, 3, 7, 14)]:
                eps, arr = ep_metric(members, section, key)
                out[key] = dict(mean=float(arr.mean()), seed_sd=float(arr.mean(1).std()) if len(members) > 1 else 0.,
                                per_episode=arr.mean(0).tolist(), episodes=eps)
            ref[m] = out
            S['methods'][m] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('per_episode', 'episodes')}) for k, v in out.items()}
        # paired comparisons
        for m, out in ref.items():
            comp = {}
            for base, key in [('persistence', 'affected_mae'), ('delta_mlp', 'affected_mae'), ('delta_mlp', 'regret'),
                              ('delta_mlp', 'spearman'), ('delta_mlp', 'pipeline_h7'), ('persistence', 'pipeline_h7'), ('delta_mlp', 'cost7_abs_err')]:
                if base not in ref or base == m: continue
                a, b = out[key], ref[base][key]
                common = sorted(set(a['episodes']) & set(b['episodes']))
                da = dict(zip(a['episodes'], a['per_episode'])); db = dict(zip(b['episodes'], b['per_episode']))
                diff = [da[e] - db[e] for e in common]
                comp[f'{key}_minus_{base}'] = dict(mean=float(np.mean(diff)), ci95=boot(diff), rel=float(np.mean(diff)/max(abs(np.mean([db[e] for e in common])), 1e-9)))
            S['methods'][m]['paired'] = comp
        # monotonicity + uncertainty
        for n in names:
            m = method_of(n); mono = R['mono'].get(n)
            if mono:
                S['methods'][m].setdefault('mono_members', []).append(dict(
                    pred_monotone_rate=float(np.mean([x['pred_monotone_rate'] for x in mono])),
                    range_ratio=float(np.mean([x['pred_range']/max(x['true_range'], 1) for x in mono])),
                    pred_cross_range=float(np.mean([x['pred_cross_range'] for x in mono])),
                    true_cross_range=float(np.mean([x['true_cross_range'] for x in mono])),
                    true_monotone_frac=float(np.mean([x['true_monotone'] for x in mono]))))
        for n, rows in R['unc'].items():
            std = np.concatenate([r['std'] for r in rows]); err = np.concatenate([r['err'] for r in rows])
            q = np.quantile(std, .8)
            S['methods'][n]['uncertainty'] = dict(spearman_std_vs_abs_err=float(spearmanr(std, err)[0]),
                                                  err_top20pct_std=float(err[std >= q].mean()), err_rest=float(err[std < q].mean()))
        # markdown
        order = ['persistence', 'train_mean_delta', 'ridge', 'delta_mlp', 'delta_mlp_ensemble', 'latent_sup', 'ema', 'sigreg', 'jepa_only', 'sim_forecast']
        md += [f'\n### {split} (replay mismatches: {R["replay_mismatched_episodes"]})', '',
               '| method | P1 affected Δ-MAE (units) | vs zero-delta (95% CI) | sign agree | P2 Spearman | P2 regret | P3 pipeline MAE h7 | P3 dest backlog MAE h7 | 7-step cost abs err |',
               '|---|---|---|---|---|---|---|---|---|']
        for m in [o for o in order if o in S['methods']]:
            x = S['methods'][m]; p = x.get('paired', {}).get('affected_mae_minus_persistence')
            f = lambda k: f"{x[k]['mean']:.2f}" + (f" ±{x[k]['seed_sd']:.2f}" if x[k]['seed_sd'] else '')
            md.append(f"| {m} | {f('affected_mae')} | " + (f"{p['mean']:+.2f} [{p['ci95'][0]:+.2f},{p['ci95'][1]:+.2f}] ({p['rel']:+.0%})" if p else '—') +
                      f" | {x['sign_agree_nonzero']['mean']:.3f} | {f('spearman')} | {f('regret')} | {f('pipeline_h7')} | {f('dest_backlog_h7')} | {f('cost7_abs_err')} |")
    (root/'summary.json').write_text(json.dumps(summary, indent=1)); (root/'summary.md').write_text('\n'.join(md)); print('\n'.join(md))


if __name__ == '__main__':
    main()
