"""Audited controlled ISOMORPH smoke. Outputs are engineering evidence, not a paper benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from prototype_new.src.env.isomorph import IsomorphEnv
from prototype_new.src.data.protocols import configuration_hash, validate_rollout_splits
from prototype_new.src.models.controlled import ControlledWorldModel
from prototype_new.src.planning.cem import CEMPlanner, Goal, RecedingHorizonController


def collect(cfg):
    groups={s: [] for s in ('train','validation','test')}; manifest=[]
    count=0
    for split,number in zip(groups,[cfg['train_rollouts'],cfg['validation_rollouts'],cfg['test_rollouts']]):
        for j in range(number):
            seed=cfg['seed']+count; count+=1
            rng=np.random.default_rng(seed)
            scenario={'ss_scale':float(rng.uniform(.04,.12)),
                      'base_lambda_lo':32.,'base_lambda_hi':48.,
                      'burst_rate_scale':0.,'shock_count_scale':0.}
            manifest.append(dict(rollout=f'{split}_{j}',seed=seed,config_hash=configuration_hash(scenario),
                                 scenario=scenario,split=split,behavior=(['uniform','demand_scaled','low_dispatch','intermittent'][j%4] if cfg.get('diverse_behavior') else ['uniform','demand_scaled'][j%2])))
            env=IsomorphEnv(cfg['days'],cfg['n_items'],scenario,cfg['maximum']); env.reset(seed)
            for day in range(cfg['days']):
                if cfg.get('diverse_behavior') and j%4==2: request=rng.uniform(0,.5,cfg['n_items'])*np.array(list(env.sim.demand_ema.values()))
                elif cfg.get('diverse_behavior') and j%4==3: request=np.zeros(cfg['n_items']) if day%16<12 else rng.uniform(0,cfg['maximum'],cfg['n_items'])
                elif j%2: request=rng.uniform(0,4,cfg['n_items'])*np.array(list(env.sim.demand_ema.values()))
                else: request=rng.uniform(0,cfg['maximum'],cfg['n_items'])
                env.step(env.space.project(request))
            groups[split].append(env.records)
    validate_rollout_splits(manifest)
    return groups,manifest


def windows(episodes,horizon):
    arrays=[[],[],[],[]]
    for episode in episodes:
        for t in range(len(episode)-horizon+1):
            chunk=episode[t:t+horizon]
            for bucket,value in zip(arrays,[chunk[0]['state'],[r['action'] for r in chunk],
                [r['next_state'] for r in chunk],[[r['components'][k] for k in ('holding','backlog','transport')] for r in chunk]]):
                bucket.append(value)
    return tuple(torch.tensor(np.asarray(a),dtype=torch.float32) for a in arrays)


def train(cfg, data, objective, actions, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=ControlledWorldModel(data['train'][0].shape[-1],cfg['n_items'],cfg['latent_dim'],objective,actions)
    s,a,f,c=data['train']; model.fit_preprocessing(torch.cat((s,f.flatten(0,1))),c.flatten(0,1),cfg['maximum'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    best=float('inf'); best_state=None; log=[]
    for epoch in range(cfg['epochs']):
        model.train()
        for idx in torch.randperm(len(s)).split(64):
            loss,stats=model.loss(s[idx],a[idx],f[idx],c[idx]); optimizer.zero_grad(); loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10)); optimizer.step(); model.update_target()
        model.eval()
        with torch.no_grad():
            vs,va,vf,vc=data['validation']; _,ps,pc=model(vs,va)
            val=float(((ps-model.normalize(vf))**2).mean()+((pc-vc/model.cost_scale)**2).mean())
        log.append(dict(epoch=epoch,validation=val,gradient_norm=norm,**stats))
        if val<best:
            best=val; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    return model,log


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='prototype_new/configs/controlled_smoke.json')
    parser.add_argument('--output',default='prototype_new/results/controlled_smoke'); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); out=Path(args.output)
    if (out/'summary.json').exists(): raise FileExistsError('Use a new output directory to preserve results')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1); start=time.perf_counter()
    groups,manifest=collect(cfg)
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2)); (out/'config.json').write_text(json.dumps(cfg,indent=2))
    with (out/'transitions.jsonl').open('w') as f:
        for m,episode in zip(manifest,[e for g in groups.values() for e in g]):
            for r in episode: f.write(json.dumps(dict(rollout=m['rollout'],split=m['split'],**r))+'\n')
    data={s:windows(e,cfg['horizon']) for s,e in groups.items()}
    models={}; training={}; predictions={}
    for method,objective,actions in [('ema_jepa','ema',True),('sigreg_jepa','sigreg',True),
                                     ('supervised','supervised',True),('no_action','ema',False)]:
        models[method]=[]; training[method]=[]
        for member in range(cfg['ensemble_members']):
            model,log=train(cfg,data,objective,actions,cfg['seed']+1000+member)
            models[method].append(model); training[method].append(log)
            torch.save({'model':model.state_dict(),'config':cfg,'objective':objective,'use_actions':actions,
                        'state_dim':data['train'][0].shape[-1], 'split_hash':configuration_hash(manifest)},out/f'{method}_{member}.pt')
        s,a,f,c=data['test']
        with torch.no_grad():
            pred=torch.stack([m(s,a)[1]*m.state_scale+m.state_mean for m in models[method]]).mean(0)
            pred_cost=torch.stack([m(s,a)[2]*m.cost_scale for m in models[method]]).mean(0)
        predictions[method]={'state_rmse_by_horizon':((pred-f).square().mean((0,2)).sqrt()).tolist(),
                             'component_mae':(pred_cost-c).abs().mean((0,1)).tolist()}
    s,a,f,c=data['test']; predictions['persistence']={'state_rmse_by_horizon':((s[:,None]-f).square().mean((0,2)).sqrt()).tolist()}
    (out/'training.json').write_text(json.dumps(training,indent=2))
    (out/'prediction.json').write_text(json.dumps(dict(track='B controlled action-labeled prediction',metrics=predictions),indent=2))
    rows=[]; detailed=[]
    for m in [m for m in manifest if m['split']=='test']:
        for method in ['native_policy','forecast_dispatch','random_feasible',*models]:
            env=IsomorphEnv(cfg['days'],cfg['n_items'],m['scenario'],cfg['maximum']); obs=env.reset(m['seed'])
            if method in models:
                def predict(s,a): return np.stack([model.predict_components(s,a) for model in models[method]])
                planner=CEMPlanner(env.space,cfg['horizon'],cfg['samples'],cfg['elites'],cfg['iterations'])
                records=RecedingHorizonController(planner,predict,Goal(**cfg['goal']),m['seed']).run(env,obs)
            else:
                rng=np.random.default_rng(m['seed']+2000); records=[]
                for _ in range(cfg['days']):
                    if method=='native_policy': action=None
                    elif method=='forecast_dispatch':
                        dest=env.sim.network.nodes[env.sim.destination_id]
                        action=env.space.project([max(0, 7*env.sim.demand_ema[i]+dest.backlog.get(i,0)-dest.inventory.get(i,0)-env.sim.item_intransit[i]) for i in env.sim.item_order])
                    else: action=env.space.project(rng.uniform(0,cfg['maximum'],cfg['n_items']))
                    _,_,_,info=env.step(action); records.append(dict(**info,fallback=False,latency=0.))
            demand=sum(r['demand'] for r in records)
            rows.append(dict(method=method,seed=m['seed'],cost=sum(r['cost'] for r in records),
                             fill_rate=sum(r['served'] for r in records)/max(demand,1),
                             fallback_count=sum(r['fallback'] for r in records),
                             latency=float(np.mean([r['latency'] for r in records])),
                             **{k:sum(r['components'][k] for r in records) for k in ('holding','backlog','transport')}))
            detailed.extend(dict(method=method,seed=m['seed'],**r) for r in records)
    with (out/'planning_steps.jsonl').open('w') as file:
        for row in detailed: file.write(json.dumps(row)+'\n')
    (out/'episodes.json').write_text(json.dumps(rows,indent=2))
    summary={}
    for method in ['native_policy','forecast_dispatch','random_feasible',*models]:
        subset=[r for r in rows if r['method']==method]; costs=np.array([r['cost'] for r in subset])
        rng=np.random.default_rng(cfg['seed']); boots=rng.choice(costs,(2000,len(costs)),replace=True).mean(1)
        summary[method]=dict(cost_mean=float(costs.mean()),cost_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),
                             fill_rate_mean=float(np.mean([r['fill_rate'] for r in subset])),episodes=len(subset),
                             fallback_count=sum(r['fallback_count'] for r in subset))
    payload=dict(track='B: locally adapted ISOMORPH; assumed costs; engineering smoke',methods=summary,
                 elapsed_seconds=time.perf_counter()-start,source_commit='9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7',
                 limitations=['Small sample; not a confirmatory benchmark','Fixed topology with 3-item catalog',
                              'Source lead-time RNG consumption can differ by policy; paired seeds are not full CRN',
                              'No calibrated uncertainty claim','Final holdout not used for checkpoint selection'])
    (out/'summary.json').write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
