"""decision_v3 receding-horizon planning comparison (paired by scenario seed, CRN).

Every method acts on the same post-demand observation, with identical CEM budget
for all planners. Learned models never see the demand table or lead-time draws;
`mpc_sim_forecast` uses known simulator mechanics with EMA demand forecasts;
`mpc_sim_oracle` uses the true exogenous future (diagnostic ceiling, not deployable).
Each episode runs `decision_days` controlled days, then `tail_days` of the native
rule for all methods so that terminal state is charged.
"""
import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from prototype_new.src.env.isomorph import DecisionEnv
from prototype_new.src.planning.cem import CEMPlanner, Goal
from prototype_new.experiments.evaluate_dynamics_v3 import load_model, forecast_clone, Cost
from prototype_new.experiments.collect_decision_data import basestock
from prototype_new.src.models.controlled import Ensemble


def components(cost, states, transport):
    w = cost.w
    return np.stack([w['holding'] * states[..., cost.inv].sum(-1), w['backlog'] * states[..., cost.bl].sum(-1),
                     w['transport'] * transport], -1)


def run_episode(job, cfg, names, model_names):
    method, scenario, seed = job
    torch.set_num_threads(1)
    cost = Cost(names, cfg['costs'])
    D, T = cfg['decision_days'], cfg['tail_days']
    env = DecisionEnv(D + T, 3, scenario, 1000, crn=True); env.reset(seed)
    planner = CEMPlanner(env.space, cfg['horizon'], cfg['samples'], cfg['elites'], cfg['iterations'], cfg['init_std'])
    model = None
    goal = Goal(uncertainty=1.0) if method.endswith('_unc') else Goal()   # ensemble-spread penalty (heuristic)
    if method.startswith('mpc_') and not method.startswith('mpc_sim'):
        key = method[4:].removesuffix('_unc')
        if key == 'delta_mlp_ensemble':
            model = Ensemble([load_model(Path(cfg['models'])/f'delta_mlp_{s}.pt') for s in range(5)])
        else:
            model = load_model(Path(cfg['models'])/f'{key}.pt')
    rng = np.random.default_rng([seed, 4242]); steps = []; t0 = time.perf_counter()
    for day in range(D + T):
        obs = env.observe(); plan_info = {}
        if day >= D or method == 'native':
            a = env.space.project(env.native_action())
        elif method.startswith('basestock_k'):
            a = env.space.project(basestock(env, np.full(3, float(method.split('k')[-1]))))
        elif method == 'zero':
            a = np.zeros(3, dtype=np.int64)
        elif method == 'random':
            a = env.space.project(rng.uniform(0, 1000, 3))
        else:
            nat = env.space.project(env.native_action()); init = np.repeat(nat[None], cfg['horizon'], 0)
            if model is not None:
                def predict(o, cand):
                    with torch.no_grad():
                        s = torch.tensor(o).expand(len(cand), -1); a_ = torch.tensor(cand, dtype=torch.float32)
                        if isinstance(model, Ensemble):
                            _, _, st, tr = model.rollout(s, a_, return_members=True)
                            return components(cost, st.numpy(), tr.numpy())
                        st, tr = model.rollout(s, a_)
                        return components(cost, st.numpy(), tr.numpy())[None]
            else:
                make = forecast_clone if method == 'mpc_sim_forecast' else (lambda e: e.clone())
                def predict(o, cand):
                    out = []
                    for seq in cand:
                        c = make(env); st, tr = [], []
                        for a_ in seq[:max(0, min(len(seq), c.days - c.day))]:
                            s, _, _, info = c.step(a_); st.append(s); tr.append(info['transport_units'])
                        st, tr = np.asarray(st), np.asarray(tr, dtype=np.float32)
                        comp = components(cost, st, tr)
                        out.append(np.pad(comp, ((0, len(seq) - len(comp)), (0, 0))))
                    return np.asarray(out)[None]
            res = planner.plan(obs, predict, goal, seed=seed * 1000 + day, init=init, extra=[init])
            a = res.actions[0]; plan_info = dict(predicted=res.score, fallback=res.fallback, latency=res.elapsed)
        _, c, _, info = env.step(a)
        steps.append(dict(day=day, action=info['action'], cost=c, components=info['components'],
                          holding_units=info['holding_units'], backlog_units=info['backlog_units'],
                          transport_units=info['transport_units'], demand=info['demand'], served=info['served'], **plan_info))
    total = sum(s['cost'] for s in steps); controlled = sum(s['cost'] for s in steps[:D])
    return dict(method=method, seed=seed, cost=total, cost_controlled=controlled,
                components={k: sum(s['components'][k] for s in steps) for k in ('holding', 'backlog', 'transport')},
                fill_rate=sum(s['served'] for s in steps[:D]) / max(1, sum(s['demand'] for s in steps[:D])),
                backlog_days=int(sum(s['backlog_units'] > 0 for s in steps[:D])),
                fallbacks=int(sum(s.get('fallback', False) for s in steps)), seconds=time.perf_counter() - t0, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='prototype_new/configs/planning_v3.json')
    ap.add_argument('--output', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--max-episodes', type=int, default=None)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(cfg['data'])/'manifest.json').read_text()); names = man['state_names']
    eps = [m for m in man['episodes'] if m['split'] == args.split][:args.max_episodes]
    done = {(r['method'], r['seed']) for r in map(json.loads, open(out/'episodes.jsonl'))} if (out/'episodes.jsonl').exists() else set()
    jobs = [(meth, m['scenario'], m['seed']) for meth in args.methods for m in eps if (meth, m['seed']) not in done]
    with Pool(cfg['workers']) as pool, (out/'episodes.jsonl').open('a') as f:
        for r in pool.imap_unordered(partial(run_episode, cfg=cfg, names=names, model_names=None), jobs):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(r['method'], r['seed'], round(r['cost'], 1), round(r['seconds'], 1), flush=True)
    (out/'config.json').write_text(json.dumps(dict(cfg, split=args.split), indent=1))


if __name__ == '__main__':
    main()
