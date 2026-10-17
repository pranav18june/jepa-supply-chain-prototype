"""Train decision_v3 dynamics comparators. Selection uses validation episodes only."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from prototype_new.src.models.controlled import (DeltaDynamics, LatentDynamics, RidgeDynamics, ConstantDelta)

NONNEG = ('/inventory', '/backlog', '/order_qty', '/order_eta', '/demand_ema', '/demand_today')


def load(data_dir):
    d = dict(np.load(Path(data_dir)/'transitions.npz'))
    man = json.loads((Path(data_dir)/'manifest.json').read_text())
    split = np.array([m['split'] for m in man['episodes']])[d['episode']]
    return d, man, split


def nonneg_mask(names):
    return [float(n.endswith(NONNEG) or n.startswith('pipeline/') or n.startswith('capacity/')) for n in names]


def windows(d, split, which, H):
    """Contiguous H-step windows inside single episodes of the requested split."""
    S, A, N, T, ep, day = d['state'], d['action'], d['next_state'], d['transport_units'], d['episode'], d['day']
    idx = np.flatnonzero(split == which)
    starts = [i for i in idx if i + H - 1 < len(ep) and ep[i + H - 1] == ep[i] and day[i + H - 1] == day[i] + H - 1]
    st = np.asarray(starts)
    off = st[:, None] + np.arange(H)
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    return t(S[st]), t(A[off]), t(N[off]), t(T[off])


def build(kind, D, A, cfg, objective=None):
    if kind == 'delta_mlp':
        return DeltaDynamics(D, A, cfg['width'])
    return LatentDynamics(D, A, objective, cfg['latent_dim'], cfg['width'])


def val_loss(model, val, bs=2048):
    s, a, f, t = val; total = 0.
    with torch.no_grad():
        for i in range(0, len(s), bs):
            pred, tp = model.rollout(s[i:i+bs], a[i:i+bs])
            total += sum(float(model.norm.step_loss(None, pred[:, k], f[i:i+bs, k], tp[:, k], t[i:i+bs, k]))
                         for k in range(a.shape[1])) / a.shape[1] * len(s[i:i+bs])
    return total / len(s)


def train(model, train_w, val_w, cfg, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'])
    best, best_state, log = float('inf'), None, []
    s, a, f, t = train_w
    for epoch in range(cfg['epochs']):
        model.train(); t0 = time.perf_counter()
        for idx in torch.randperm(len(s), generator=g).split(cfg['batch_size']):
            loss, stats = model.loss(s[idx], a[idx], f[idx], t[idx])
            opt.zero_grad(); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])); opt.step()
            if hasattr(model, 'update_target'):
                model.update_target(cfg['ema_momentum'])
        model.eval(); v = val_loss(model, val_w)
        log.append(dict(epoch=epoch, validation=v, grad_norm=gn, seconds=time.perf_counter()-t0, **stats))
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    return model, log, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='prototype_new/configs/dynamics_v3.json')
    p.add_argument('--output', default='prototype_new/results/decision_v3d/models')
    p.add_argument('--only', nargs='*', default=None)
    args = p.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(cfg['torch_threads'])
    d, man, split = load(cfg['data']); names = man['state_names']
    H = cfg['train_horizon']; D, A = d['state'].shape[1], d['action'].shape[1]
    tr, va = windows(d, split, 'train', H), windows(d, split, 'validation', H)
    one = lambda w: (w[0], w[1][:, 0], w[2][:, 0], w[3][:, 0])
    s1, a1, n1, t1 = one(windows(d, split, 'train', 1))
    fit_norm = lambda m: m.norm.fit(s1, a1, n1, t1, man['config']['maximum'], nonneg_mask(names))
    summary = json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    save = lambda name, model, extra: torch.save(dict(state_dict=model.state_dict(), kind=model.kind, D=D, A=A,
                                                      cfg=cfg, **extra), out/f'{name}.pt')
    want = lambda name: args.only is None or name in args.only
    for name, use_mean in (('persistence', False), ('train_mean_delta', True)):
        if want(name):
            m = ConstantDelta(D, A, use_mean); fit_norm(m); save(name, m, dict(use_mean=use_mean))
            summary[name] = dict(validation=val_loss(m, va), params=0)
    if want('ridge'):
        scores = {}
        for alpha in cfg['ridge_alphas']:
            m = RidgeDynamics(D, A); fit_norm(m); m.fit(s1, a1, n1, t1, alpha); scores[alpha] = val_loss(m, va)
        alpha = min(scores, key=scores.get)
        m = RidgeDynamics(D, A); fit_norm(m); m.fit(s1, a1, n1, t1, alpha); save('ridge', m, dict(alpha=alpha))
        summary['ridge'] = dict(validation=scores[alpha], alpha=alpha, alpha_scores=scores, params=int(m.W.numel()))
    jobs = [('delta_mlp', None, s) for s in cfg['delta_mlp_seeds']]
    jobs += [('latent', o, s) for o in cfg['latent_objectives'] for s in cfg['latent_seeds']]
    for kind, objective, seed in jobs:
        name = f"{objective or kind}_{seed}"
        if not want(name) and not want(objective or kind):
            continue
        torch.manual_seed(seed); m = build(kind, D, A, cfg, objective); fit_norm(m)
        t0 = time.perf_counter(); m, log, best = train(m, tr, va, cfg, seed)
        save(name, m, dict(objective=objective, seed=seed))
        (out/f'{name}_training.json').write_text(json.dumps(log))
        summary[name] = dict(validation=best, best_epoch=int(np.argmin([r['validation'] for r in log])), seed=seed,
                             params=sum(p.numel() for p in m.parameters() if p.requires_grad), seconds=time.perf_counter()-t0)
        print(name, summary[name], flush=True)
        (out/'summary.json').write_text(json.dumps(summary, indent=1))
    (out/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: round(v['validation'], 4) for k, v in summary.items()}, indent=1))


if __name__ == '__main__':
    main()
