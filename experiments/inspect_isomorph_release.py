"""Read immutable manifest, Parquet footers and a 16-row topology; no full tables."""
import json
import urllib.request
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REVISION='478ee4d339eeab4a4208b32c6a4b4a2881c999a5'


def main():
    out=Path('prototype_new/results/release_audit'); out.mkdir(parents=True,exist_ok=True)
    base=f'https://huggingface.co/datasets/Isomorph2026/isomorph-supply-chain-benchmark/resolve/{REVISION}'
    raw=urllib.request.urlopen(base+'/manifest.csv',timeout=30).read(); (out/'manifest.csv').write_bytes(raw)
    manifest=pd.read_csv(out/'manifest.csv')
    config_cols=[c for c in manifest if c not in ('rollout','family','scenario','path')]
    duplicate=manifest[manifest.duplicated(config_cols,keep=False)].rollout.tolist()
    result={'revision':REVISION,'rollouts':len(manifest),'seeds':manifest.seed.unique().tolist(),
            'horizons':manifest.horizon.unique().tolist(),'duplicate_config_rollouts':duplicate,'files':{}}
    fs=HfFileSystem()
    for name in ('daily_records','shipments','inventory_history','backlog_history','intransit_history','edge_list'):
        with fs.open(f'datasets/Isomorph2026/isomorph-supply-chain-benchmark@{REVISION}/output_item50/{name}.parquet','rb',block_size=65536) as f:
            p=pq.ParquetFile(f)
            result['files'][name]={'rows':p.metadata.num_rows,'row_groups':p.metadata.num_row_groups,'schema':str(p.schema_arrow)}
            if name=='edge_list': result['files'][name]['sample']=p.read().to_pylist()
    (out/'metadata.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
