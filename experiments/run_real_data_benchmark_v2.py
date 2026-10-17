"""Track C: chronological observed-sales/stockout prediction, no policy claims.

Replaces the legacy v2 runner whose validation labels were training targets.
The original source is preserved in execution_audit/pre_edit_source.tar.gz.
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


class RetailJEPA(nn.Module):
    def __init__(self,history,latent=32):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(history*2,latent),nn.SiLU(),nn.Linear(latent,latent))
        self.target=copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor=nn.Sequential(nn.Linear(latent,latent),nn.SiLU(),nn.Linear(latent,latent))
        self.sales=nn.Linear(latent,1); self.stock=nn.Linear(latent,1)

    def forward(self,x):
        z=self.predictor(self.encoder(x.flatten(1)))
        return nn.functional.softplus(self.sales(z)[:,0]),self.stock(z)[:,0],z

    @torch.no_grad()
    def ema(self):
        for p,q in zip(self.target.parameters(),self.encoder.parameters()): p.lerp_(q,.01)


def load_subset(path,count):
    parquet=pq.ParquetFile(path)
    columns=['store_id','product_id','dt','sale_amount','hours_sale','hours_stock_status','stock_hour6_22_cnt']
    keys=[]
    for b in parquet.iter_batches(batch_size=4096,columns=['store_id','product_id']):
        for pair in b.to_pandas().drop_duplicates().itertuples(index=False,name=None):
            if pair not in keys: keys.append(pair)
            if len(keys)>=count: break
        if len(keys)>=count: break
    selected=set(keys); pieces=[]
    for b in parquet.iter_batches(batch_size=16384,columns=columns):
        f=b.to_pandas(); mask=[k in selected for k in zip(f.store_id,f.product_id)]
        pieces.append(f.loc[mask])
    return pd.concat(pieces,ignore_index=True)


def metrics(y,stock,p,prob):
    denom=float(np.abs(y).sum())
    result=dict(n=len(y),mae=float(abs(y-p).mean()),rmse=float(np.sqrt(((y-p)**2).mean())),
                wape=None if denom==0 else float(abs(y-p).sum()/denom),
                brier=float(brier_score_loss(stock,prob)),
                auroc=float(roc_auc_score(stock,prob)) if len(np.unique(stock))==2 else None,
                auprc=float(average_precision_score(stock,prob)) if stock.sum()>0 else None)
    for flag in [0,1]:
        mask=stock==flag
        result['stockout' if flag else 'in_stock']=dict(n=int(mask.sum()),mae=float(abs(y[mask]-p[mask]).mean()) if mask.any() else None)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='prototype_new/configs/retail_audited.json')
    parser.add_argument('--output-dir',default='prototype_new/results/retail_audited'); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); out=Path(args.output_dir)
    if (out/'metrics.json').exists(): raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    path=Path(cfg['dataset_path'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar=path.with_suffix('.manifest.json')
    if sidecar.exists():
        provenance=json.loads(sidecar.read_text())
        if provenance['dataset_revision']!=cfg['dataset_revision'] or provenance['subset_sha256']!=digest:
            raise ValueError('Subset revision/checksum mismatch')
    else:
        expected={'eval.parquet':'1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e',
                  'train.parquet':'6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d'}
        if digest!=expected.get(path.name) or cfg['dataset_revision']!='c374e85975c82816c495acfa904cc55ab14a47d3':
            raise ValueError('Pinned dataset checksum/revision mismatch')
        provenance={'full_file_sha256_verified':True}
    f=load_subset(path,cfg['series'])
    splits=retail_windows(f,cfg['history'],cfg['train_end'],cfg['validation_end'],cfg.get('test_end'))
    if any(len(rows)==0 for rows in splits.values()): raise ValueError('Empty chronological split')
    arrays={s:(np.stack([r['x'] for r in rows]),np.array([r['sales'] for r in rows]),
               np.array([r['stockout'] for r in rows])) for s,rows in splits.items()}
    x,y,stock=arrays['train']; mean=x.mean((0,1)); scale=x.std((0,1)).clip(1e-4)
    sales_scale=max(float(np.sqrt(np.mean(y*y))),1e-4)
    normalized={s:torch.tensor((v[0]-mean)/scale,dtype=torch.float32) for s,v in arrays.items()}
    model=RetailJEPA(cfg['history']); opt=torch.optim.Adam(model.parameters(),lr=.002)
    labels=torch.tensor(y/sales_scale,dtype=torch.float32); binary=torch.tensor(stock,dtype=torch.float32)
    next_context=x.copy(); next_context[:,:-1]=x[:,1:]
    # Real next observation, used exclusively in the training target branch.
    next_context[:,-1,0]=y
    label_hours={ (int(r.store_id),int(r.product_id),str(pd.Timestamp(r.dt).date())):float(r.stock_hour6_22_cnt) for r in f.itertuples() }
    next_context[:,-1,1]=[label_hours[(*r['entity'],r['date'])] for r in splits['train']]
    targets=torch.tensor((next_context-mean)/scale,dtype=torch.float32).flatten(1)
    best=np.inf; best_state=None; logs=[]
    for epoch in range(cfg['epochs']):
        model.train()
        for idx in torch.randperm(len(x)).split(128):
            ps,pc,z=model(normalized['train'][idx])
            with torch.no_grad(): target=model.target(targets[idx])
            loss=nn.functional.mse_loss(ps,labels[idx])+nn.functional.binary_cross_entropy_with_logits(pc,binary[idx])+nn.functional.mse_loss(z,target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),10); opt.step(); model.ema()
        model.eval()
        with torch.no_grad():
            ps,pc,_=model(normalized['validation']); vy,vc=arrays['validation'][1:]
            val=float(nn.functional.mse_loss(ps,torch.tensor(vy/sales_scale,dtype=torch.float32))+nn.functional.binary_cross_entropy_with_logits(pc,torch.tensor(vc,dtype=torch.float32)))
        logs.append(dict(epoch=epoch,validation=val))
        if val<best: best=val; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    reg=ExtraTreesRegressor(n_estimators=64,max_depth=12,random_state=cfg['seed'],n_jobs=1).fit(x.reshape(len(x),-1),y)
    clf=ExtraTreesClassifier(n_estimators=64,max_depth=12,random_state=cfg['seed'],n_jobs=1).fit(x.reshape(len(x),-1),stock)
    tx,ty,tc=arrays['test']; flat=tx.reshape(len(tx),-1)
    prob=clf.predict_proba(flat)[:,list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(tx))
    with torch.no_grad(): ps,pc,_=model(normalized['test'])
    predictions={'last_value':(tx[:,-1,0],(tx[:,:,1]>0).mean(1)),
                 'extratrees':(reg.predict(flat),prob),
                 'ema_latent_supervised':(ps.numpy()*sales_scale,torch.sigmoid(pc).numpy())}
    results={name:metrics(ty,tc,*v) for name,v in predictions.items()}
    manifest=dict(track='C observational prediction',dataset_revision=cfg['dataset_revision'],
                  file_sha256=digest,source_provenance=provenance,config=cfg,
                  split_counts={s:len(r) for s,r in splits.items()},
                  split_dates={s:sorted({r['date'] for r in rows}) for s,rows in splits.items()},
                  preprocessing={'mean':mean.tolist(),'scale':scale.tolist(),'sales_scale':sales_scale},
                  selection='First series in physical file order: convenience subset, not representative sample',
                  censoring='Observed sales remain censored. Stockout target is next-day stock_hour6_22_cnt > 0.',
                  holdout_status=cfg['holdout_status'])
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)); (out/'metrics.json').write_text(json.dumps(results,indent=2)); (out/'training.json').write_text(json.dumps(logs,indent=2))
    rows=[]
    for method,(p,prob) in predictions.items():
        for r,a,b,ys,cs in zip(splits['test'],p,prob,ty,tc):
            rows.append(dict(method=method,entity=r['entity'],date=r['date'],observed_sales=float(ys),stockout=int(cs),predicted_sales=float(a),stockout_probability=float(b)))
    with (out/'predictions.jsonl').open('w') as file:
        for r in rows:file.write(json.dumps(r)+'\n')
    torch.save(dict(model=model.state_dict(),preprocessing=manifest['preprocessing'],config=cfg),out/'model.pt')
    (out/'report.md').write_text('# Track C predictive results\n\n'+cfg['holdout_status']+'\n\nObserved-sales prediction only; no latent-demand recovery or policy claim.\n\n| Method | MAE | Stockout AUROC |\n|---|---:|---:|\n'+'\n'.join(f"| {k} | {v['mae']:.6g} | {v['auroc']} |" for k,v in results.items())+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
