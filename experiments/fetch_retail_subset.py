"""Fetch bounded row groups from pinned train data, preserving metadata.

Default: first 24 row groups (historical subset). --sample N --seed S draws N
row groups uniformly without replacement (round_claude representative sample).
"""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from prototype_new.src.data.http_range import HTTPRangeReader


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample',type=int,default=None); ap.add_argument('--seed',type=int,default=None)
    ap.add_argument('--output',default='prototype_new/data/fresh_retail_net/data/train_subset.parquet'); args=ap.parse_args()
    revision='c374e85975c82816c495acfa904cc55ab14a47d3'
    url=f'https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/{revision}/data/train.parquet?download=true'
    reader=HTTPRangeReader(url,106436287,max_transfer=(110 if args.sample else 32)*1024*1024,retries=4 if args.sample else 0)
    p=pq.ParquetFile(reader)
    columns=['store_id','product_id','dt','sale_amount','hours_sale','hours_stock_status','stock_hour6_22_cnt']
    if args.sample:
        import numpy as np
        groups=sorted(np.random.default_rng(args.seed).choice(p.metadata.num_row_groups,args.sample,replace=False).tolist())
    else:
        groups=list(range(24))
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Refusing to overwrite an existing subset')
    table=p.read_row_groups(groups,columns=columns); pq.write_table(table,out)
    manifest=dict(dataset_revision=revision,source_url=url,source_bytes=106436287,
                  source_sha256='6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d',
                  source_hash_verified=False,subset_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  row_groups=groups,sampling_seed=args.seed,columns=columns,rows=table.num_rows,
                  source_rows=p.metadata.num_rows,transferred_bytes=reader.transferred)
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    f=table.to_pandas(); print(json.dumps(manifest,indent=2)); print('dates',f.dt.min(),f.dt.max())

if __name__=='__main__': main()
