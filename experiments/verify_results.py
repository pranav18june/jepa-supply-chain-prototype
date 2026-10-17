"""Recompute reported values from raw artifacts, failing on mismatches."""
import argparse
import json
from pathlib import Path
import numpy as np
from prototype_new.experiments.run_real_data_benchmark_v2 import metrics


def verify_controlled(path):
    summary=json.loads((path/'summary.json').read_text())
    episodes=json.loads((path/'episodes.json').read_text())
    steps=[json.loads(line) for line in (path/'planning_steps.jsonl').read_text().splitlines()]
    for e in episodes:
        rows=[r for r in steps if r['method']==e['method'] and r['seed']==e['seed']]
        assert np.isclose(sum(r['cost'] for r in rows),e['cost'])
        assert all(np.isclose(r['cost'],sum(r['components'].values())) for r in rows)
    for method,stats in summary['methods'].items():
        selected=[e for e in episodes if e['method']==method]
        assert len(selected)==stats['episodes']
        assert np.isclose(np.mean([e['cost'] for e in selected]),stats['cost_mean'])
        assert sum(e['fallback_count'] for e in selected)==stats['fallback_count']
    return len(episodes)


def verify_retail(path):
    summary=json.loads((path/'metrics.json').read_text())
    rows=[json.loads(line) for line in (path/'predictions.jsonl').read_text().splitlines()]
    for method,expected in summary.items():
        r=[x for x in rows if x['method']==method]
        actual=metrics(*(np.array([x[k] for x in r]) for k in ('observed_sales','stockout','predicted_sales','stockout_probability')))
        for k in ('mae','rmse','wape','auroc','auprc','brier'):
            assert (actual[k] is None and expected[k] is None) or np.isclose(actual[k],expected[k])
    return len(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('path',type=Path); args=parser.parse_args()
    count=verify_controlled(args.path) if (args.path/'summary.json').exists() else verify_retail(args.path)
    print('Verified raw-to-summary consistency:',count,'records')

if __name__=='__main__': main()
