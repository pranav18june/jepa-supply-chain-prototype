"""Regenerate every figure of PROJECT_COMPREHENSIVE_REPORT.docx from project files.

Run from JEPA/:  PYTHONPATH=. python3 prototype_new/docs/report_assets/make_figures.py
Result charts read result files only; diagrams are drawn schematically and are
labelled 'implemented', 'legacy' or 'conceptual' in the figure itself.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = Path('prototype_new/results'); OUT = Path('prototype_new/docs/report_assets/figures'); OUT.mkdir(parents=True, exist_ok=True)
C = dict(data=('#E6F1FB', '#185FA5'), learn=('#EEEDFE', '#534AB7'), plan=('#E1F5EE', '#0F6E56'),
         eval=('#FAEEDA', '#854F0B'), limit=('#FCEBEB', '#A32D2D'), neutral=('#F1EFE8', '#5F5E5A'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def box(ax, x, y, w, h, title, sub='', kind='neutral', dashed=False, fs=10):
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fill, ec=edge,
                                lw=1.2, ls='--' if dashed else '-'))
    ax.text(x + w/2, y + h/2 + (0.12 if sub else 0), title, ha='center', va='center', fontsize=fs, color=edge, weight='bold')
    if sub:
        ax.text(x + w/2, y + h/2 - 0.17, sub, ha='center', va='center', fontsize=fs-2, color=edge)


def arrow(ax, x1, y1, x2, y2, color='#5F5E5A', style='-', text=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=color, lw=1.2, ls=style))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.08, text, fontsize=8, color=color, ha='center')


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off'); return fig, ax


def legend(ax, items, x, y):
    for i, (kind, label, dashed) in enumerate(items):
        fill, edge = C[kind]
        ax.add_patch(FancyBboxPatch((x + i*2.2, y), 0.3, 0.2, boxstyle='round,pad=0.01', fc=fill, ec=edge, ls='--' if dashed else '-'))
        ax.text(x + i*2.2 + 0.4, y + 0.1, label, va='center', fontsize=8, color='#444441')


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=200, bbox_inches='tight'); plt.close(fig)


# F1 — the real ISOMORPH network used in every controlled experiment
def fig_network():
    import contextlib, io, random
    from prototype_new.src.vendor.isomorph import simulator as up
    st = random.getstate()
    with contextlib.redirect_stdout(io.StringIO()):
        sim, net, _, _ = up.build_example_simulation_from_adjacency(seed=1, horizon_days=10, scenario={})
    random.setstate(st)
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = {n: (v.lon, v.lat) for n, v in net.nodes.items()}
    for (u, v), e in net.edges.items():
        last = v == 'NewYork'
        ax.annotate('', xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle='->', color='#A32D2D' if last else '#888780', lw=1.6 if last else 0.9))
        mx, my = (pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2
        ax.text(mx, my, f'{e.travel_time_days:g}d', fontsize=7, color='#5F5E5A')
    for n, (x, y) in pos.items():
        node = net.nodes[n]
        kind = 'limit' if node.is_destination else 'data' if node.is_source else 'plan'
        ax.scatter(x, y, s=120, color=C[kind][1], zorder=3)
        ax.text(x + 0.4, y + 0.35, n, fontsize=8, color=C[kind][1])
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    for kind, lab in (('data', 'Source (magic replenishment)'), ('plan', 'Intermediate (s,S) warehouse'), ('limit', 'Destination (demand)')):
        ax.scatter([], [], color=C[kind][1], label=lab)
    ax.plot([], [], color='#A32D2D', label='Last-mile lane (capacity can bind)')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    save(fig, 'f01_isomorph_network')


# F2 — conceptual JEPA vs what the project trains
def fig_jepa():
    fig, ax = canvas(10, 4.6)
    box(ax, 0.2, 3.2, 1.8, 0.8, 'State s_t', 'observed numbers', 'data')
    box(ax, 0.2, 1.4, 1.8, 0.8, 'Action a_t', 'dispatch request', 'data')
    box(ax, 2.6, 3.2, 1.8, 0.8, 'Encoder', 'z_t = E(s_t)', 'learn')
    box(ax, 4.9, 2.3, 1.9, 0.8, 'Predictor', 'ẑ(t+1) = P(z_t, a_t)', 'learn')
    box(ax, 0.2, 0.2, 1.8, 0.8, 'Next state', 's(t+1), training only', 'data')
    box(ax, 2.6, 0.2, 1.8, 0.8, 'Target encoder', 'EMA copy or same E', 'learn', dashed=True)
    box(ax, 7.4, 2.3, 2.4, 0.8, 'Latent loss', '‖ẑ − sg(z̄(t+1))‖²', 'eval')
    box(ax, 7.4, 3.5, 2.4, 0.8, 'Decoder + state loss', 'Δŝ vs true Δs', 'plan')
    box(ax, 7.4, 1.1, 2.4, 0.8, 'SIGReg (optional)', 'keeps z spread out', 'eval', dashed=True)
    arrow(ax, 2.0, 3.6, 2.6, 3.6); arrow(ax, 4.4, 3.5, 4.9, 2.9); arrow(ax, 2.0, 1.8, 4.9, 2.5)
    arrow(ax, 2.0, 0.6, 2.6, 0.6); arrow(ax, 4.4, 0.6, 7.4, 2.4); arrow(ax, 6.8, 2.7, 7.4, 2.7); arrow(ax, 6.8, 2.9, 7.4, 3.8)
    arrow(ax, 4.4, 0.4, 7.4, 1.4, style='--')
    ax.text(0.2, 4.35, 'Figure content: latent model as implemented in LatentDynamics (src/models/controlled.py). Dashed = objective-dependent part.', fontsize=8, color='#5F5E5A')
    save(fig, 'f02_jepa_concept')


# F3 — reference comparison
def fig_refs():
    fig, ax = canvas(10, 4.2)
    cols = [('V-JEPA 2-AC', 'vjepa2-main (MIT)', ['Video ViT encoder, frozen in AC stage', 'Transformer predictor + 7-D robot actions', 'L1 teacher-forced + rollout loss', 'CEM on goal-embedding distance'], 'learn'),
            ('LeWorldModel', 'le-wm-main (MIT)', ['Pixel encoder trained end-to-end', 'Action-conditioned AR predictor', 'Next-embedding MSE + SIGReg, no EMA', 'CEM via external solver'], 'learn'),
            ('SupplyChain-JEPA', 'no open-source licence', ['Typed supply-chain tokens', 'Online + frozen/EMA target encoders', 'Masked latent + VICReg/SIGReg options', 'Concepts only; no code copied'], 'limit'),
            ('This project (v3)', 'prototype_new', ['MLP encoder on 214-D numeric state', 'MLP latent predictor, integer dispatch', 'Decoder Δ-state loss ± EMA / SIGReg', 'Integer CEM-MPC on decoded cost'], 'plan')]
    for i, (t, s, rows, k) in enumerate(cols):
        x = 0.15 + i*2.47
        box(ax, x, 3.1, 2.3, 0.8, t, s, k)
        import textwrap
        y = 2.85
        for r in rows:
            lines = textwrap.wrap(r, 26)
            for k, line in enumerate(lines):
                ax.text(x + 0.08, y - k*0.22, ('• ' if k == 0 else '  ') + line, fontsize=7.6, color='#444441')
            y -= 0.22*len(lines) + 0.2
    save(fig, 'f03_reference_comparison')


# F4 — actual system architecture (training vs evaluation/planning)
def fig_arch():
    fig, ax = canvas(11, 6.2)
    ax.add_patch(FancyBboxPatch((0.1, 3.3), 10.8, 2.7, boxstyle='round,pad=0.02', fc='none', ec='#888780', ls=':'))
    ax.text(0.25, 5.8, 'Offline: data generation and training', fontsize=9, color='#5F5E5A')
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 10.8, 3.0, boxstyle='round,pad=0.02', fc='none', ec='#888780', ls=':'))
    ax.text(0.25, 2.9, 'Online: evaluation and receding-horizon planning', fontsize=9, color='#5F5E5A')
    box(ax, 0.3, 4.4, 1.9, 0.9, 'ISOMORPH sim', 'vendored, MIT', 'data')
    box(ax, 2.6, 4.4, 1.9, 0.9, 'DecisionEnv', 'timing, CRN, clone', 'data')
    box(ax, 4.9, 4.4, 1.9, 0.9, 'Behavior policies', '6 families', 'data')
    box(ax, 7.2, 4.4, 1.6, 0.9, 'Coverage gate', 'v3d passes', 'eval')
    box(ax, 9.1, 4.4, 1.6, 0.9, 'Train models', 'val. selection', 'learn')
    for a, b in ((2.2, 2.6), (4.5, 4.9), (6.8, 7.2), (8.8, 9.1)):
        arrow(ax, a, 4.85, b, 4.85)
    box(ax, 0.3, 1.3, 1.9, 0.9, 'Observe s_t', 'post-demand', 'data')
    box(ax, 2.6, 1.3, 1.9, 0.9, 'CEM candidates', '7-day sequences', 'plan')
    box(ax, 4.9, 1.3, 1.9, 0.9, 'World model', 'rollout(s, a)', 'learn')
    box(ax, 7.2, 1.3, 1.6, 0.9, 'Known cost fn', 'on predicted s', 'eval')
    box(ax, 9.1, 1.3, 1.6, 0.9, 'Execute a_t', 'simulator step', 'data')
    for a, b in ((2.2, 2.6), (4.5, 4.9), (6.8, 7.2), (8.8, 9.1)):
        arrow(ax, a, 1.75, b, 1.75)
    arrow(ax, 9.9, 1.3, 9.9, 0.55); arrow(ax, 9.9, 0.55, 1.25, 0.55); arrow(ax, 1.25, 0.55, 1.25, 1.3)
    ax.text(5.5, 0.62, 'next observation (replan every day)', fontsize=8, ha='center', color='#5F5E5A')
    arrow(ax, 9.9, 4.4, 5.85, 2.2, style='--', text='checkpoints')
    save(fig, 'f04_system_architecture')


# F5 — planner loop / paired branches (conceptual mechanics, implemented code)
def fig_planner():
    fig, ax = canvas(10, 3.6)
    steps = [('1. Observe', 'state s_t', 'data'), ('2. Propose', '64 sequences × 3 iters', 'plan'), ('3. Imagine', 'model rollouts, 7 days', 'learn'),
             ('4. Score', 'assumed cost units', 'eval'), ('5. Act', 'first action only', 'plan'), ('6. Repeat', 'next day', 'neutral')]
    for i, (t, s, k) in enumerate(steps):
        box(ax, 0.1 + i*1.65, 1.5, 1.5, 0.9, t, s, k, fs=9)
        if i: arrow(ax, 0.1 + i*1.65 - 0.15, 1.95, 0.1 + i*1.65, 1.95)
    ax.text(0.1, 0.9, 'CEM: sample integer request sequences around a mean (warm-started at the native rule), keep the 8 best, refit mean/std, repeat;', fontsize=8, color='#444441')
    ax.text(0.1, 0.6, 'every candidate is projected to integers in [0,1000]; the simulator then enforces stock and lane capacity on the executed request.', fontsize=8, color='#444441')
    save(fig, 'f05_planner_loop')


# F6 — real paired branches from one replayed test state: truth vs delta-MLP ensemble
def fig_branches():
    import torch
    from prototype_new.experiments.evaluate_dynamics_v3 import replay, load_model, sim_rollout
    from prototype_new.src.models.controlled import Ensemble
    root = R/'decision_v3d'; d = np.load(root/'data/transitions.npz'); man = json.loads((root/'data/manifest.json').read_text())
    names = man['state_names']; e = next(i for i, m in enumerate(man['episodes']) if m['split'] == 'test')
    m = man['episodes'][e]; sel = d['episode'] == e
    _, snaps = replay(m, d['action'][sel], man['config'])
    t = 20; base = snaps[t]; cont = d['action'][sel][t+1:t+7]
    nat = base.native_action().clip(0, 1000)
    cands = {'zero': np.zeros(3), 'native rule': nat, '2× native': np.clip(2*nat, 0, 1000), 'max (1000)': np.full(3, 1000)}
    ens = Ensemble([load_model(root/f'models/delta_mlp_{s}.pt') for s in range(5)])
    inv = [i for i, n in enumerate(names) if n.startswith('NewYork/') and n.endswith('/inventory')]
    bl = [i for i, n in enumerate(names) if n.startswith('NewYork/') and n.endswith('/backlog')]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#A32D2D', '#0F6E56', '#534AB7', '#854F0B']
    s0 = d['state'][sel][t]
    for (label, c), col in zip(cands.items(), colors):
        seq = np.vstack([c[None], cont]).astype(np.int64)
        st, _ = sim_rollout(base.clone(), seq)
        with torch.no_grad():
            ps = ens.rollout(torch.tensor(s0)[None], torch.tensor(seq, dtype=torch.float32)[None])[0][0].numpy()
        net = lambda S: S[:, inv].sum(1) - S[:, bl].sum(1)
        x = np.arange(1, 8)
        ax.plot(x, net(st), color=col, lw=2, label=f'{label}: simulator')
        ax.plot(x, net(ps), color=col, lw=1.2, ls='--', label=f'{label}: model')
    ax.axhline(0, color='#888780', lw=0.6)
    ax.set_xlabel('days after decision'); ax.set_ylabel('destination inventory − backlog (units)')
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_title(f'Paired branches from test episode {m["rollout"]}, day {t} (same demand and lead times)', fontsize=9)
    save(fig, 'f06_paired_branches')


# F7 — data splits: v3 episode generation and FreshRetailNet chronology
def fig_splits():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    man = json.loads((R/'decision_v3d/data/manifest.json').read_text())
    counts = {}
    for m in man['episodes']:
        counts.setdefault(m['split'], {}).setdefault(m['behavior'], 0); counts[m['split']][m['behavior']] += 1
    fams = sorted({b for v in counts.values() for b in v}); splits = ['train', 'validation', 'test', 'test_ood']
    bottom = np.zeros(4); pal = plt.cm.tab10.colors
    for i, f in enumerate(fams):
        vals = np.array([counts[s].get(f, 0) for s in splits])
        axes[0].bar(splits, vals, bottom=bottom, color=pal[i], label=f); bottom += vals
    axes[0].set_ylabel('episodes (60 days each)'); axes[0].legend(fontsize=7, frameon=False); axes[0].set_title('decision_v3d episodes by behavior family', fontsize=9)
    man2 = json.loads((R/'retail_attribution/data_manifest.json').read_text())
    sc = man2['split_counts']; labels = ['train\n≤ 2024-05-31', 'validation\n06-01 … 06-12', 'test\n06-13 … 06-25']
    axes[1].bar(labels, [sc['train'], sc['validation'], sc['test']], color=[C['learn'][1], C['eval'][1], C['limit'][1]])
    axes[1].set_ylabel('next-day windows'); axes[1].set_title(f"FreshRetailNet sample: {man2['series']} series, {man2['stores']} stores", fontsize=9)
    save(fig, 'f07_splits')


# F8 — coverage gate across data versions
def fig_coverage():
    keys = ['backlog_positive_frac', 'truncated_request_frac', 'capacity_saturated_frac', 'max_request_frac']
    vers = ['v3a', 'v3b', 'v3c', 'v3d']; reps = [json.loads((R/f'decision_{v}/data/coverage_report.json').read_text()) for v in vers]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    x = np.arange(len(keys)); w = 0.2
    for i, (v, r) in enumerate(zip(vers, reps)):
        axes[0].bar(x + (i-1.5)*w, [r['splits']['train'][k] for k in keys], w, label=v + (' (pass)' if r['coverage_gate_pass'] else ' (fail)'))
    axes[0].scatter(x, [reps[0]['gates'][k] for k in keys], marker='_', s=400, color='k', label='gate threshold', zorder=3)
    axes[0].set_xticks(x, ['backlog>0', 'truncated', 'lane saturated', 'at max request'], fontsize=8); axes[0].set_ylabel('fraction of train transitions')
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].bar(vers, [r['splits']['train']['recovery_events'] for r in reps], color=C['plan'][1]); axes[1].axhline(100, color='k', lw=1, ls='--')
    axes[1].set_ylabel('recovery events (backlog → 0)'); axes[1].set_title('threshold 100', fontsize=9)
    save(fig, 'f08_coverage')


# F9 — legacy controlled development planning (pre-demand protocol, 8 episodes)
def fig_legacy_planning():
    s = json.loads((R/'controlled_development/summary.json').read_text())['methods']
    fc = json.loads((R/'controlled_development/forecast_baseline.json').read_text())
    rows = [(k, v['cost_mean'], v['cost_bootstrap95']) for k, v in s.items()] + [('forecast_dispatch', fc['cost_mean'], None)]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = [r[0].replace('no_action', 'action-blind model') for r in rows]
    vals = [r[1] for r in rows]
    err = np.array([[r[1]-r[2][0], r[2][1]-r[1]] if r[2] else [0, 0] for r in rows]).T
    cols = [C['learn'][1] if r[0] in ('ema_jepa', 'sigreg_jepa', 'supervised', 'no_action') else C['neutral'][1] for r in rows]
    ax.barh(labels, vals, xerr=err, color=cols, capsize=3); ax.invert_yaxis()
    ax.set_xlabel('mean realized cost over 48 days (assumed units), 8 test episodes; bars: 95% bootstrap CI')
    save(fig, 'f09_legacy_planning')


# F10 — v3 action-effect (P1) and action sensitivity
def fig_p1():
    s = json.loads((R/'decision_v3d/evaluation/summary.json').read_text())
    order = ['ridge', 'delta_mlp', 'delta_mlp_ensemble', 'latent_sup', 'ema', 'sigreg', 'jepa_only', 'sim_forecast']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6)); fig.subplots_adjust(wspace=0.55)
    for j, split in enumerate(['test', 'test_ood']):
        M = s[split]['methods']; zero = M['persistence']['affected_mae']['mean']
        vals = [M[m]['affected_mae']['mean'] for m in order]
        rel = [M[m]['paired']['affected_mae_minus_persistence'] for m in order]
        err = np.array([[r['mean']-r['ci95'][0], r['ci95'][1]-r['mean']] for r in rel]).T
        cols = [C['plan'][1] if m.startswith('delta') else C['learn'][1] if m in ('latent_sup', 'ema', 'sigreg', 'jepa_only') else C['neutral'][1] for m in order]
        axes[j].barh(order, vals, xerr=err, color=cols, capsize=3); axes[j].axvline(zero, color='#A32D2D', ls='--', lw=1)
        axes[j].text(zero, len(order)-0.4, ' zero-delta', color='#A32D2D', fontsize=8)
        axes[j].invert_yaxis(); axes[j].set_title(f'{split}: one-step paired Δ error (units)', fontsize=9)
        axes[j].set_xlabel('MAE on action-affected features (lower is better)')
    save(fig, 'f10_action_effect')
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ms = ['ridge', 'delta_mlp', 'delta_mlp_ensemble', 'latent_sup', 'ema', 'sigreg', 'jepa_only']
    for j, split in enumerate(['test', 'test_ood']):
        vals = [np.mean([a['range_ratio'] for a in s[split]['methods'][m]['mono_members']]) for m in ms]
        ax.bar(np.arange(len(ms)) + (j-0.5)*0.38, vals, 0.38, label=split, color=[C['plan'][1], C['eval'][1]][j])
    ax.axhline(1, color='k', ls='--', lw=1); ax.set_yscale('log'); ax.set_xticks(range(len(ms)), ms, fontsize=8)
    ax.set_ylabel('predicted / true response range'); ax.legend(frameon=False, fontsize=8)
    ax.set_title('Response of item-1 pipeline to sweeping its request 0→1000 (1 = correct magnitude)', fontsize=9)
    save(fig, 'f11_action_sensitivity')


# F12 — v3 decision ranking (P2) and rollout error (P3)
def fig_p2p3():
    s = json.loads((R/'decision_v3d/evaluation/summary.json').read_text())['test']['methods']
    ms = ['persistence', 'ridge', 'delta_mlp', 'delta_mlp_ensemble', 'latent_sup', 'ema', 'sigreg', 'jepa_only', 'sim_forecast']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].barh(ms, [s[m]['regret']['mean'] for m in ms], color=C['eval'][1]); axes[0].invert_yaxis()
    axes[0].set_xlabel('mean 7-day regret of chosen first action (units, lower better)'); axes[0].set_title('P2 decision ranking (iid test)', fontsize=9)
    for m in ['persistence', 'ridge', 'delta_mlp_ensemble', 'ema', 'sim_forecast']:
        axes[1].plot([1, 3, 7, 14], [s[m][f'dest_backlog_h{h}']['mean'] for h in (1, 3, 7, 14)], marker='o', label=m)
    axes[1].set_xlabel('horizon (days)'); axes[1].set_ylabel('destination backlog MAE (units)'); axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title('P3 open-loop rollout error (iid test)', fontsize=9)
    save(fig, 'f12_ranking_rollout')


# F13 — FreshRetailNet attribution
def fig_retail():
    m = json.loads((R/'retail_attribution/metrics.json').read_text())['results']
    order = ['last_value', 'history_mean', 'extratrees', 'supervised', 'ema_jepa', 'sigreg_probe', 'random_probe']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, k, lab in ((axes[0], 'wape', 'sales WAPE (lower better)'), (axes[1], 'auroc', 'stockout AUROC (higher better)')):
        ax.barh(order, [m[o][k]['mean'] for o in order], xerr=[m[o][k]['seed_sd'] for o in order], color=[C['neutral'][1]]*3 + [C['learn'][1]]*4, capsize=3)
        ax.invert_yaxis(); ax.set_xlabel(lab)
        lo = min(m[o][k]['mean'] for o in order); hi = max(m[o][k]['mean'] for o in order); ax.set_xlim(lo - 0.05*(hi-lo) - 0.01, hi + 0.02)
    fig.suptitle('FreshRetailNet test (42,769 windows, 3,338 series; error bars = SD over 5 seeds)', fontsize=9)
    save(fig, 'f13_retail')


# F14 — v3 planning (if available)
def fig_planning():
    p = R/'decision_v3d/planning_test/summary.json'
    if not p.exists(): return
    s = json.loads(p.read_text())['methods']
    ms = [k for k in sorted(s, key=lambda k: s[k]['cost_mean']) if k not in ('zero',)]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [s[k]['cost_mean'] for k in ms]; err = np.array([[s[k]['cost_mean']-s[k]['cost_ci95'][0], s[k]['cost_ci95'][1]-s[k]['cost_mean']] for k in ms]).T
    cols = [C['eval'][1] if 'oracle' in k else C['plan'][1] if 'sim_forecast' in k else C['learn'][1] if k.startswith('mpc') else C['neutral'][1] for k in ms]
    ax.barh(ms, vals, xerr=err, color=cols, capsize=3); ax.invert_yaxis()
    ax.set_xlabel(f"mean realized cost, 60 controlled + 14 tail days (assumed units), {json.loads(p.read_text())['episodes']} paired test episodes")
    ax.set_xscale('log')
    save(fig, 'f14_v3_planning')


# F15 — experimental tracks and the evidence each can produce
def fig_tracks():
    fig, ax = canvas(10, 3.8)
    tr = [('Track A', 'native-policy trajectory', 'prediction (legacy)', 'neutral', 'state forecasting only'),
          ('Track B (legacy)', 'pre-demand controlled', 'smoke + development', 'neutral', 'superseded protocol'),
          ('Track B (v3)', 'decision_v3d: P1/P2/P3', '+ MPC planning', 'plan', 'action effects, decisions'),
          ('Track C (retail)', 'FreshRetailNet sample', 'attribution, 5 seeds', 'learn', 'observational prediction')]
    for i, (t, a, b, k, ev) in enumerate(tr):
        x = 0.15 + i*2.47
        box(ax, x, 2.2, 2.3, 1.1, t, a + '\n' + b, k, fs=9)
        box(ax, x, 0.6, 2.3, 0.8, 'Evidence', ev, 'eval', fs=9)
        arrow(ax, x + 1.15, 2.2, x + 1.15, 1.42)
    ax.text(0.15, 0.2, 'Simulator tracks use assumed cost units; no track establishes real-world operational effectiveness.', fontsize=8, color='#A32D2D')
    save(fig, 'f15_tracks')


# F16 — failure analysis: links supported by measured evidence in this report
def fig_failure():
    fig, ax = canvas(10.5, 4.4)
    box(ax, 0.2, 3.1, 3.0, 0.9, 'Legacy data coverage', '0/288 backlog transitions (smoke)', 'limit', fs=9)
    box(ax, 0.2, 1.7, 3.0, 0.9, 'Pre-demand timing', 'native saw same-day demand', 'limit', fs=9)
    box(ax, 0.2, 0.3, 3.0, 0.9, 'Action-blind latents', 'sweep response 12–35% of true', 'limit', fs=9)
    box(ax, 3.9, 2.4, 3.0, 0.9, 'Weak action-effect accuracy', 'legacy: worse than zero-delta', 'eval', fs=9)
    box(ax, 3.9, 0.3, 3.0, 0.9, 'OOD shift', 'v3 learned ≈ zero-delta on OOD', 'eval', fs=9)
    box(ax, 7.4, 1.4, 2.9, 0.9, 'Planner quality', 'see planning results', 'plan', fs=9)
    arrow(ax, 3.2, 3.5, 3.9, 2.95); arrow(ax, 3.2, 0.75, 3.9, 2.5); arrow(ax, 6.9, 2.85, 7.4, 2.1)
    arrow(ax, 6.9, 0.75, 7.4, 1.6); arrow(ax, 3.2, 2.1, 7.4, 1.9, style='--')
    ax.text(0.2, 4.15, 'Solid: relationship measured in this project. Dashed: fairness confound, corrected in v3 by post-demand timing.', fontsize=8, color='#5F5E5A')
    save(fig, 'f16_failure_analysis')


# F17 — roadmap: completed / partial / open
def fig_roadmap():
    fig, ax = canvas(10.5, 3.2)
    items = [('Inventory + backup', 'plan'), ('Timing / CRN fix', 'plan'), ('Coverage gate', 'plan'), ('Dynamics comparison', 'plan'),
             ('Planning (iid)', 'eval'), ('OOD robustness', 'limit'), ('Wider actions / topology', 'limit'), ('Real-data transfer', 'limit')]
    for i, (t, k) in enumerate(items):
        x = 0.1 + i*1.3
        box(ax, x, 1.4, 1.2, 0.9, t, {'plan': 'done', 'eval': 'partial', 'limit': 'open'}[k], k, dashed=k == 'limit', fs=8)
        if i: arrow(ax, x - 0.1, 1.85, x, 1.85)
    legend(ax, [('plan', 'completed with evidence', False), ('eval', 'partial', False), ('limit', 'open / proposed', True)], 0.1, 0.6)
    save(fig, 'f17_roadmap')


if __name__ == '__main__':
    for f in (fig_network, fig_jepa, fig_refs, fig_arch, fig_planner, fig_branches, fig_splits, fig_coverage,
              fig_legacy_planning, fig_p1, fig_p2p3, fig_retail, fig_planning, fig_tracks, fig_failure, fig_roadmap):
        f(); print('ok', f.__name__)
