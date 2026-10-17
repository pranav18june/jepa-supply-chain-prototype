"""Summarize decision_v3 planning: paired (same seed, CRN) episode-level comparisons.

Cost sensitivity re-weights the *executed* trajectories' unit counts; planners were
optimized under the default weights, so this is not re-optimization.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

WEIGHTS = {'default': (.001, 1., .01), 'holding_x10': (.01, 1., .01), 'backlog_x5': (.001, 5., .01), 'transport_x5': (.001, 1., .05)}


def boot(x, n=4000, seed=0):
    x = np.asarray(x); m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='prototype_new/results/decision_v3d/planning_test')
    args = ap.parse_args(); root = Path(args.run)
    eps = defaultdict(dict)
    for line in open(root/'episodes.jsonl'):
        r = json.loads(line); eps[r['method']][r['seed']] = r
    seeds = sorted(set.intersection(*[set(v) for v in eps.values()]))
    def cost(r, w):
        return sum(w[0]*s['holding_units'] + w[1]*s['backlog_units'] + w[2]*s['transport_units'] for s in r['steps'])
    out = dict(episodes=len(seeds), methods={})
    for m, rows in eps.items():
        rs = [rows[s] for s in seeds]
        c = np.array([r['cost'] for r in rs])
        entry = dict(cost_mean=float(c.mean()), cost_ci95=boot(c), cost_p90=float(np.quantile(c, .9)),
                     fill_rate=float(np.mean([r['fill_rate'] for r in rs])), backlog_days=float(np.mean([r['backlog_days'] for r in rs])),
                     components={k: float(np.mean([r['components'][k] for r in rs])) for k in ('holding', 'backlog', 'transport')},
                     fallbacks=int(sum(r['fallbacks'] for r in rs)), seconds_per_episode=float(np.mean([r['seconds'] for r in rs])),
                     sensitivity={k: float(np.mean([cost(r, w) for r in rs])) for k, w in WEIGHTS.items()})
        for base in ('basestock_k9', 'native', 'mpc_sim_oracle'):
            if base in eps and base != m:
                d = c - np.array([eps[base][s]['cost'] for s in seeds])
                entry[f'minus_{base}'] = dict(mean=float(d.mean()), ci95=boot(d), win_rate=float((d < 0).mean()))
        out['methods'][m] = entry
    order = sorted(out['methods'], key=lambda k: out['methods'][k]['cost_mean'])
    lines = [f"Episodes (paired seeds): {len(seeds)}", '', '| method | mean cost [95% CI] | Δ vs base-stock k9 [95% CI] (win rate) | Δ vs native [95% CI] | fill | backlog days | holding/backlog/transport | s/ep |', '|---|---|---|---|---|---|---|---|']
    for m in order:
        e = out['methods'][m]; b = e.get('minus_basestock_k9'); n = e.get('minus_native')
        f = lambda x: f"{x['mean']:+.1f} [{x['ci95'][0]:+.1f},{x['ci95'][1]:+.1f}]" if x else '—'
        lines.append(f"| {m} | {e['cost_mean']:.1f} [{e['cost_ci95'][0]:.1f},{e['cost_ci95'][1]:.1f}] | {f(b)}" + (f" ({b['win_rate']:.0%})" if b else '') +
                     f" | {f(n)} | {e['fill_rate']:.4f} | {e['backlog_days']:.1f} | " + '/'.join(f"{e['components'][k]:.0f}" for k in ('holding', 'backlog', 'transport')) + f" | {e['seconds_per_episode']:.1f} |")
    lines += ['', '| method | ' + ' | '.join(WEIGHTS) + ' |', '|---|' + '---|'*len(WEIGHTS)]
    for m in order:
        lines.append(f'| {m} | ' + ' | '.join(f"{out['methods'][m]['sensitivity'][k]:.1f}" for k in WEIGHTS) + ' |')
    (root/'summary.json').write_text(json.dumps(out, indent=1)); (root/'summary.md').write_text('\n'.join(lines)); print('\n'.join(lines))


if __name__ == '__main__':
    main()
