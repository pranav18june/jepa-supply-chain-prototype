"""Track A: generated native-policy trajectories, persistence and ridge baselines.
No actions are invented for the native-policy records.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from prototype_new.src.env.isomorph import IsomorphEnv
from prototype_new.src.data.protocols import configuration_hash,validate_rollout_splits


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='prototype_new/results/isomorph_prediction')
    args=parser.parse_args(); out=Path(args.output)
    if (out/'metrics.json').exists(): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    data={s:[] for s in ['train','validation','test']}; manifest=[]; seed=4700
    for split,n in [('train',12),('validation',4),('test',4)]:
        for i in range(n):
            seed+=1; rng=np.random.default_rng(seed)
            scenario={'ss_scale':float(rng.uniform(.03,.15)),'base_lambda_lo':32.,'base_lambda_hi':48.}
            env=IsomorphEnv(days=40,n_items=3,scenario=scenario); obs=env.reset(seed); states=[obs]
            for _ in range(env.days): states.append(env.step(None)[0])
            data[split].append(np.array(states))
            manifest.append(dict(rollout=f'{split}_{i}',seed=seed,config_hash=configuration_hash(scenario),split=split,scenario=scenario))
    validate_rollout_splits(manifest)
    x=np.concatenate([s[:-1] for s in data['train']]); y=np.concatenate([s[1:] for s in data['train']]); norm=StandardScaler().fit(x)
    selected=None; best=np.inf
    for alpha in [.1,1.,10.,100.]:
        model=Ridge(alpha=alpha).fit(norm.transform(x),norm.transform(y))
        vx=np.concatenate([s[:-1] for s in data['validation']]); vy=np.concatenate([s[1:] for s in data['validation']])
        score=np.mean((model.predict(norm.transform(vx))-norm.transform(vy))**2)
        if score<best: selected=model; best=score
    rows=[]
    for i,s in enumerate(data['test']):
        for horizon in [1,3,8]:
            context=s[:-horizon]; targets=s[horizon:]; pred=norm.transform(context)
            for _ in range(horizon): pred=selected.predict(pred)
            pred=norm.inverse_transform(pred)
            for method,p in [('persistence',context),('ridge',pred)]:
                rows.append(dict(rollout=i,method=method,horizon=horizon,rmse=float(np.sqrt(np.mean((p-targets)**2))),
                                 per_feature_mae=np.mean(abs(p-targets),axis=0).tolist()))
    payload=dict(track='A generated ISOMORPH native-policy prediction',source_commit='9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7',
                 days=40,n_items=3,selected_alpha=selected.alpha,rows=rows,
                 limitation='Mixed-unit overall RMSE is diagnostic; per-feature MAE supplied. No causal or action-conditioned claim.')
    (out/'metrics.json').write_text(json.dumps(payload,indent=2)); (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Track A complete:',len(rows),'baseline evaluations; selected alpha',selected.alpha)

if __name__=='__main__': main()
