"""Machine-checkable action/state coverage report for decision_v3 data."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

GATES = {  # declared before inspecting model results; per training split
    'backlog_positive_frac': 0.10, 'truncated_request_frac': 0.10, 'capacity_saturated_frac': 0.03,
    'zero_request_frac': 0.05, 'max_request_frac': 0.02, 'recovery_events': 100,
    'max_behavior_share': 0.40, 'test_mass_in_unseen_cells': 0.05}


def groups(names):
    idx = lambda f: [i for i, n in enumerate(names) if f(n)]
    return dict(
        dest_inventory=idx(lambda n: n.startswith('NewYork/') and n.endswith('/inventory')),
        dest_backlog=idx(lambda n: n.startswith('NewYork/') and n.endswith('/backlog')),
        pipeline=idx(lambda n: n.startswith('pipeline/')),
        lastmile_wh=idx(lambda n: n.split('/')[0] in ('Philadelphia', 'Baltimore') and n.endswith('/inventory')),
        upstream_inventory=idx(lambda n: n.endswith('/inventory') and n.split('/')[0] not in ('NewYork', 'Philadelphia', 'Baltimore')),
        orders=idx(lambda n: n.endswith('/order_qty')))


def per_item(names, prefix):
    items = sorted({n.split('/')[1] for n in names if n.startswith('item/')})
    return [[i for i, n in enumerate(names) if n.startswith(f'{prefix}/{it}/')] for it in items]


def summarize(d, sel, names, maximum):
    g = groups(names)
    S, A, N, R = d['state'][sel], d['action'][sel], d['next_state'][sel], d['realized_dispatch'][sel]
    U = d['lastmile_utilization'][sel]; ep = d['episode'][sel]
    bl_next = N[:, g['dest_backlog']].sum(1); bl_now = S[:, g['dest_backlog']].sum(1)
    trunc = (R < A - .5).any(1); sat = (U >= .98).any(1)
    recov = int(((bl_now > 0) & (bl_next == 0)).sum())
    pipe_items = per_item(names, 'pipeline')
    dpipe = np.stack([N[:, ix].sum(1) - S[:, ix].sum(1) for ix in pipe_items], 1)
    bins = np.digitize(A, [1, 100, 300, 600, 999])
    effect = {f'request_bin_{b}': dict(n=int((bins == b).sum()), mean_abs_dpipeline=float(np.abs(dpipe[bins == b]).mean()) if (bins == b).any() else None)
              for b in range(6)}
    return dict(transitions=int(sel.sum()), episodes=int(len(np.unique(ep))),
                action_quantiles=np.percentile(A, [0, 10, 25, 50, 75, 90, 100]).round(1).tolist(),
                zero_request_frac=float((A == 0).mean()), max_request_frac=float((A >= maximum).mean()),
                truncated_request_frac=float(trunc.mean()),
                truncated_by_capacity_frac=float((trunc & sat).mean()),
                capacity_saturated_frac=float(sat.mean()),
                backlog_positive_frac=float((bl_next > 0).mean()),
                backlog_p90=float(np.percentile(bl_next, 90)), backlog_max=float(bl_next.max()),
                severe_backlog_frac=float((bl_next > N[:, [names.index(f'item/{it}/demand_ema') for it in sorted({n.split('/')[1] for n in names if n.startswith('item/')})]].sum(1)).mean()),
                recovery_events=recov,
                episodes_with_backlog=int(len(np.unique(ep[bl_next > 0]))),
                lastmile_wh_empty_frac=float((S[:, g['lastmile_wh']] <= 0).any(1).mean()),
                effect_by_request_bin=effect)


def cells(d, sel, names):
    g = groups(names)
    S, A = d['state'][sel], d['action'][sel]
    backlog = (S[:, g['dest_backlog']].sum(1) > 0).astype(int)
    cap = np.digitize(S[:, [i for i, n in enumerate(names) if n.startswith('capacity/')]].sum(1), [5000, 9000])
    wh = np.digitize(S[:, g['lastmile_wh']].sum(1), [100, 600, 1500])
    req = np.digitize(A.sum(1), [1, 300, 900, 1800])
    return list(zip(backlog, cap, wh, req))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', default='prototype_new/results/decision_v3/data')
    args = p.parse_args(); root = Path(args.data)
    d = dict(np.load(root/'transitions.npz')); man = json.loads((root/'manifest.json').read_text())
    names, cfg = man['state_names'], man['config']
    split_of = np.array([m['split'] for m in man['episodes']])[d['episode']]
    beh_of = np.array([m['behavior'] for m in man['episodes']])[d['episode']]
    report = dict(gates=GATES, splits={}, behaviors={})
    for s in cfg['splits']:
        report['splits'][s] = summarize(d, split_of == s, names, cfg['maximum'])
        report['splits'][s]['behavior_counts'] = dict(Counter(beh_of[split_of == s].tolist()))
    for b in sorted(set(beh_of)):
        report['behaviors'][b] = summarize(d, (beh_of == b) & (split_of == 'train'), names, cfg['maximum']) if ((beh_of == b) & (split_of == 'train')).any() else None
    train_cells = Counter(cells(d, split_of == 'train', names))
    for s in ('test', 'test_ood'):
        c = cells(d, split_of == s, names)
        report['splits'][s]['test_mass_in_unseen_cells'] = float(np.mean([x not in train_cells for x in c]))
        report['splits'][s]['test_mass_in_rare_cells_lt10'] = float(np.mean([train_cells.get(x, 0) < 10 for x in c]))
    tr = report['splits']['train']
    checks = {k: (tr[k] >= v) for k, v in GATES.items() if k not in ('max_behavior_share', 'test_mass_in_unseen_cells')}
    checks['max_behavior_share'] = max(tr['behavior_counts'].values())/tr['transitions'] <= GATES['max_behavior_share']
    checks['test_mass_in_unseen_cells'] = report['splits']['test']['test_mass_in_unseen_cells'] <= GATES['test_mass_in_unseen_cells']
    report['checks'] = {k: bool(v) for k, v in checks.items()}; report['coverage_gate_pass'] = all(checks.values())
    (root/'coverage_report.json').write_text(json.dumps(report, indent=1))
    keys = ['transitions', 'zero_request_frac', 'max_request_frac', 'truncated_request_frac', 'truncated_by_capacity_frac',
            'capacity_saturated_frac', 'backlog_positive_frac', 'severe_backlog_frac', 'recovery_events', 'lastmile_wh_empty_frac']
    lines = ['| split | ' + ' | '.join(keys) + ' |', '|' + '---|'*(len(keys)+1)]
    for s, r in report['splits'].items():
        lines.append(f'| {s} | ' + ' | '.join(f'{r[k]:.3f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + ' |')
    lines += ['', '| train behavior | ' + ' | '.join(keys) + ' |', '|' + '---|'*(len(keys)+1)]
    for b, r in report['behaviors'].items():
        if r: lines.append(f'| {b} | ' + ' | '.join(f'{r[k]:.3f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + ' |')
    lines += ['', f"Unseen-cell test mass: iid {report['splits']['test']['test_mass_in_unseen_cells']:.4f}, "
              f"ood {report['splits']['test_ood']['test_mass_in_unseen_cells']:.4f}", '', f"Checks: {report['checks']}",
              f"Coverage gate pass: {report['coverage_gate_pass']}"]
    (root/'coverage_report.md').write_text('\n'.join(lines)); print('\n'.join(lines))


if __name__ == '__main__':
    main()
