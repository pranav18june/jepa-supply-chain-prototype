"""Track D (FreshRetailNet) attribution: does a JEPA latent objective add value?

Identical encoder/predictor/heads under four regimes, each with 5 seeds:
  supervised        sales MSE + stockout BCE
  ema_jepa          + MSE(predicted latent, EMA-target encoding of next-day context)
  sigreg_probe      encoder/predictor pretrained with latent + SIGReg only, frozen; heads fitted after
  random_probe      random frozen encoder/predictor; heads fitted after
plus last-value, history-mean and ExtraTrees on the same features. Observed sales
are stockout-censored; no latent-demand or policy claim follows.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from prototype_new.src.data.protocols import retail_windows
from prototype_new.src.models.controlled import sigreg


class Net(nn.Module):
    def __init__(self, inputs, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, latent))
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(latent, 64), nn.SiLU(), nn.Linear(64, latent))
        self.sales, self.stock = nn.Linear(latent, 1), nn.Linear(latent, 1)

    def latent(self, x):
        return self.predictor(self.encoder(x))

    def heads(self, z):
        return nn.functional.softplus(self.sales(z)[:, 0]), self.stock(z)[:, 0]

    @torch.no_grad()
    def ema(self, m=.99):
        for p, q in zip(self.target.parameters(), self.encoder.parameters()): p.lerp_(q, 1 - m)


def metrics(y, c, p, prob):
    return dict(mae=float(np.abs(y - p).mean()), wape=float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9)),
                auroc=float(roc_auc_score(c, prob)) if len(np.unique(c)) == 2 else None,
                auprc=float(average_precision_score(c, prob)), brier=float(brier_score_loss(c, np.clip(prob, 0, 1))))


def fit(regime, X, ctx_next, y, c, V, cfg, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = Net(X['train'].shape[1], cfg['latent'])
    yt, ct = torch.tensor(y['train']), torch.tensor(c['train'])
    sup = lambda zb, idx: (nn.functional.mse_loss(net.heads(zb)[0], yt[idx]) +
                           nn.functional.binary_cross_entropy_with_logits(net.heads(zb)[1], ct[idx]))
    if regime in ('sigreg_probe', 'random_probe'):
        if regime == 'sigreg_probe':
            opt = torch.optim.Adam(list(net.encoder.parameters()) + list(net.predictor.parameters()), lr=cfg['lr'])
            for _ in range(cfg['pretrain_epochs']):
                for idx in torch.randperm(len(yt), generator=g).split(cfg['batch_size']):
                    zp = net.latent(X['train'][idx]); zt = net.encoder(ctx_next[idx])
                    loss = nn.functional.mse_loss(zp, zt) + 0.1 * sigreg(torch.stack((net.encoder(X['train'][idx]), zt)))
                    opt.zero_grad(); loss.backward(); opt.step()
        net.encoder.requires_grad_(False); net.predictor.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg['lr']); best, state = np.inf, None
    for _ in range(cfg['epochs']):
        for idx in torch.randperm(len(yt), generator=g).split(cfg['batch_size']):
            z = net.latent(X['train'][idx]); loss = sup(z, idx)
            if regime == 'ema_jepa':
                with torch.no_grad(): target = net.target(ctx_next[idx])
                loss = loss + nn.functional.mse_loss(z, target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 10); opt.step()
            if regime == 'ema_jepa': net.ema()
        with torch.no_grad():
            ps, pc = net.heads(net.latent(X['validation']))
            v = float(nn.functional.mse_loss(ps, torch.tensor(y['validation'])) +
                      nn.functional.binary_cross_entropy_with_logits(pc, torch.tensor(c['validation'])))
        if v < best: best, state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        ps, pc = net.heads(net.latent(X['test']))
    return ps.numpy(), torch.sigmoid(pc).numpy(), best


def boot_diff(a, b, groups, n=4000, seed=0):
    """Entity-clustered bootstrap of mean(a-b) (a, b per-window losses)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, a - b); counts = np.bincount(inv)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(uniq), (n, len(uniq)))
    m = sums[idx].sum(1) / counts[idx].sum(1)
    return float((a - b).mean()), [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='prototype_new/configs/retail_attribution.json')
    ap.add_argument('--output', default='prototype_new/results/retail_attribution'); args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text()); out = Path(args.output)
    if (out/'metrics.json').exists(): raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(8)
    path = Path(cfg['dataset_path']); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    side = json.loads(path.with_suffix('.manifest.json').read_text())
    if side['subset_sha256'] != digest or side['dataset_revision'] != cfg['dataset_revision']:
        raise ValueError('Subset checksum/revision mismatch')
    f = pq.read_table(path).to_pandas()
    W = retail_windows(f, cfg['history'], cfg['train_end'], cfg['validation_end'], cfg['test_end'])
    raw = {s: np.stack([r['x'] for r in W[s]]) for s in W}
    y = {s: np.array([r['sales'] for r in W[s]], dtype=np.float32) for s in W}
    c = {s: np.array([r['stockout'] for r in W[s]], dtype=np.float32) for s in W}
    mean, scale = raw['train'].mean((0, 1)), raw['train'].std((0, 1)).clip(1e-4)
    X = {s: torch.tensor(((raw[s] - mean) / scale).reshape(len(raw[s]), -1), dtype=torch.float32) for s in W}
    # Next-day context (target branch only): shift window by one day, append observed day-t values.
    nxt = raw['train'].copy(); nxt[:, :-1] = raw['train'][:, 1:]
    hours = {(int(a), int(b), str(pd.Timestamp(d).date())): float(h) for a, b, d, h in
             zip(f.store_id, f.product_id, f.dt, f.stock_hour6_22_cnt)}
    nxt[:, -1, 0] = y['train']; nxt[:, -1, 1] = [hours[(*r['entity'], r['date'])] for r in W['train']]
    ctx_next = torch.tensor(((nxt - mean) / scale).reshape(len(nxt), -1), dtype=torch.float32)
    ty, tc = y['test'], c['test']; groups = np.array([f"{r['entity'][0]}_{r['entity'][1]}" for r in W['test']])
    preds = {'last_value': [(raw['test'][:, -1, 0], (raw['test'][:, -1, 1] > 0).astype(float))],
             'history_mean': [(raw['test'][:, :, 0].mean(1), (raw['test'][:, :, 1] > 0).mean(1))]}
    flat = lambda s: raw[s].reshape(len(raw[s]), -1)
    preds['extratrees'] = []
    for seed in cfg['seeds']:
        reg = ExtraTreesRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=seed, n_jobs=8).fit(flat('train'), y['train'])
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=seed, n_jobs=8).fit(flat('train'), c['train'])
        preds['extratrees'].append((reg.predict(flat('test')), clf.predict_proba(flat('test'))[:, 1]))
    sales_scale = float(np.sqrt(np.mean(y['train']**2)))
    ys = {s: v / sales_scale for s, v in y.items()}; val_scores = {}
    for regime in ('supervised', 'ema_jepa', 'sigreg_probe', 'random_probe'):
        preds[regime], val_scores[regime] = [], []
        for seed in cfg['seeds']:
            p, prob, v = fit(regime, X, ctx_next, ys, c, None, cfg, seed)
            preds[regime].append((p * sales_scale, prob)); val_scores[regime].append(v)
        print(regime, np.mean(val_scores[regime]), flush=True)
    results, per_window = {}, {}
    for name, runs in preds.items():
        ms = [metrics(ty, tc, p, pr) for p, pr in runs]
        results[name] = {k: dict(mean=float(np.mean([m[k] for m in ms])), seed_sd=float(np.std([m[k] for m in ms]))) for k in ms[0]}
        p = np.mean([r[0] for r in runs], 0); pr = np.mean([r[1] for r in runs], 0)
        per_window[name] = dict(ae=np.abs(ty - p), brier=(np.clip(pr, 0, 1) - tc)**2)
    paired = {}
    for name in per_window:
        for base in ('extratrees', 'supervised'):
            if name == base: continue
            paired[f'{name}_minus_{base}'] = {k: dict(zip(('mean', 'ci95'), boot_diff(per_window[name][k], per_window[base][k], groups)))
                                              for k in ('ae', 'brier')}
    manifest = dict(track='D FreshRetailNet observational prediction', dataset_revision=cfg['dataset_revision'], subset_sha256=digest,
                    row_groups=side['row_groups'], sampling_seed=side.get('sampling_seed'), config=cfg,
                    split_counts={s: len(W[s]) for s in W}, test_entities=int(len(np.unique(groups))),
                    stores=int(f.store_id.nunique()), products=int(f.product_id.nunique()), series=int(f.groupby(['store_id', 'product_id']).ngroups),
                    stockout_rate={s: float(c[s].mean()) for s in c}, normalization=dict(mean=mean.tolist(), scale=scale.tolist(), sales_scale=sales_scale),
                    validation_scores=val_scores,
                    censoring='Observed sales are censored by stockouts; target is observed next-day sales and stock_hour6_22_cnt>0.')
    (out/'data_manifest.json').write_text(json.dumps(manifest, indent=1))
    (out/'metrics.json').write_text(json.dumps(dict(results=results, paired_seed_mean_predictions=paired), indent=1))
    np.savez_compressed(out/'test_predictions.npz', y=ty, stockout=tc, groups=groups,
                        **{f'{k}_sales': np.mean([r[0] for r in v], 0) for k, v in preds.items()},
                        **{f'{k}_prob': np.mean([r[1] for r in v], 0) for k, v in preds.items()})
    lines = ['| method | MAE | WAPE | AUROC | AUPRC | Brier |', '|---|---|---|---|---|---|']
    for n, r in results.items():
        lines.append(f"| {n} | " + ' | '.join(f"{r[k]['mean']:.4f}±{r[k]['seed_sd']:.4f}" if r[k]['mean'] is not None else '—' for k in ('mae', 'wape', 'auroc', 'auprc', 'brier')) + ' |')
    lines += ['', '| paired (seed-mean predictions) | ΔMAE [95% entity-cluster CI] | ΔBrier [95% CI] |', '|---|---|---|']
    for k, v in paired.items():
        lines.append(f"| {k} | {v['ae']['mean']:+.4f} [{v['ae']['ci95'][0]:+.4f},{v['ae']['ci95'][1]:+.4f}] | {v['brier']['mean']:+.4f} [{v['brier']['ci95'][0]:+.4f},{v['brier']['ci95'][1]:+.4f}] |")
    (out/'report.md').write_text('\n'.join(lines)); print('\n'.join(lines))


if __name__ == '__main__':
    main()
