"""Add a separately stored matched-observation forecast dispatch baseline."""
import json
from pathlib import Path
import numpy as np
from prototype_new.src.env.isomorph import IsomorphEnv


def main():
    run=Path('prototype_new/results/controlled_development')
    cfg=json.loads((run/'config.json').read_text()); manifest=json.loads((run/'split_manifest.json').read_text()); rows=[]
    for r in [r for r in manifest if r['split']=='test']:
        env=IsomorphEnv(cfg['days'],cfg['n_items'],r['scenario'],cfg['maximum']);env.reset(r['seed'])
        info=[]
        for _ in range(env.days):
            d=env.sim.network.nodes[env.sim.destination_id]
            action=env.space.project([max(0,7*env.sim.demand_ema[i]+d.backlog.get(i,0)-d.inventory.get(i,0)-env.sim.item_intransit[i]) for i in env.sim.item_order])
            info.append(env.step(action)[3])
        rows.append(dict(seed=r['seed'],cost=sum(x['cost'] for x in info),fill_rate=sum(x['served'] for x in info)/sum(x['demand'] for x in info),steps=info))
    result=dict(method='forecast_dispatch',scope='Matched pre-day observations; 7-day EMA inventory-position target, no learned model; fixed coefficient inherited from native pipeline multiplier',
                cost_mean=float(np.mean([r['cost'] for r in rows])),fill_rate_mean=float(np.mean([r['fill_rate'] for r in rows])),episodes=rows)
    (run/'forecast_baseline.json').write_text(json.dumps(result,indent=2));print('Forecast baseline',result['cost_mean'],result['fill_rate_mean'])

if __name__=='__main__': main()
