"""Paired one-step action-effect audit of a frozen model on held-out scenarios."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prototype_new.src.env.isomorph import IsomorphEnv
from prototype_new.src.models.controlled import ControlledWorldModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('prototype_new/results/controlled_development'));args=parser.parse_args()
    cfg=json.loads((args.run/'config.json').read_text());manifest=json.loads((args.run/'split_manifest.json').read_text())
    rows=[];torch.set_num_threads(1)
    for method in ['ema_jepa','sigreg_jepa','supervised','no_action']:
        payload=torch.load(args.run/f'{method}_0.pt',map_location='cpu',weights_only=True)
        model=ControlledWorldModel(payload['state_dim'],cfg['n_items'],cfg['latent_dim'],payload['objective'],payload['use_actions']);model.load_state_dict(payload['model']);model.eval()
        for row in [r for r in manifest if r['split']=='test']:
            pair=[]
            for quantity in [0,cfg['maximum']]:
                env=IsomorphEnv(cfg['days'],cfg['n_items'],row['scenario'],cfg['maximum']); state=env.reset(row['seed'])
                for _ in range(12): state=env.step(np.zeros(cfg['n_items'],dtype=int))[0]
                action=np.full(cfg['n_items'],quantity,dtype=int)
                future,cost,_,_=env.step(action)
                with torch.no_grad():
                    _,prediction,pc=model(torch.tensor(state)[None],torch.tensor(action,dtype=torch.float32)[None,None])
                pair.append((future,(prediction[0,0]*model.state_scale+model.state_mean).detach().numpy()))
            true=pair[1][0]-pair[0][0];pred=pair[1][1]-pair[0][1]
            denominator=np.linalg.norm(true)*np.linalg.norm(pred)
            rows.append(dict(method=method,seed=row['seed'],true_delta_norm=float(np.linalg.norm(true)),
                             predicted_delta_norm=float(np.linalg.norm(pred)),delta_mae=float(abs(true-pred).mean()),
                             cosine=None if denominator==0 else float(np.dot(true,pred)/denominator)))
    (args.run/'action_effects.json').write_text(json.dumps(dict(scope='Paired zero vs maximum dispatch after 12 zero-dispatch days; one member only; diagnostic',rows=rows),indent=2))
    print({m:float(np.mean([r['delta_mae'] for r in rows if r['method']==m])) for m in ['ema_jepa','sigreg_jepa','supervised','no_action']})

if __name__=='__main__':main()
