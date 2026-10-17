"""Fetch small immutable source/metadata fixtures; never the full ISOMORPH data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ISOMORPH_SOURCE='9ab2689cbbe02ef5e9bb614f5e9ea60df159abc7'
ISOMORPH_DATA='478ee4d339eeab4a4208b32c6a4b4a2881c999a5'
FRESHRETAIL_DATA='c374e85975c82816c495acfa904cc55ab14a47d3'


def fetch(url, path, sha256=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest()==sha256):
        return
    temporary=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=60) as source, temporary.open('wb') as dest:
        while chunk:=source.read(1024*1024): dest.write(chunk)
    if sha256 and hashlib.sha256(temporary.read_bytes()).hexdigest()!=sha256:
        raise ValueError(f'Checksum failed for {url}')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--retail-train',action='store_true')
    args=parser.parse_args(); out=Path('prototype_new/results/execution_audit')
    for f in ['simulator/Supplychaingeo_item50.py','simulator/Supplychaingeo_item50_v2.py','LICENSE','README.md']:
        fetch(f'https://raw.githubusercontent.com/tuhinsahai/ISOMORPH/{ISOMORPH_SOURCE}/{f}',out/('upstream_'+Path(f).name))
    for f in ['README.md','LICENSE','manifest.csv','output_item50/scenario.json']:
        fetch(f'https://huggingface.co/datasets/Isomorph2026/isomorph-supply-chain-benchmark/resolve/{ISOMORPH_DATA}/{f}',out/('isomorph_'+f.replace('/','_')))
    if args.retail_train:
        fetch(f'https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/{FRESHRETAIL_DATA}/data/train.parquet?download=true',
              'prototype_new/data/fresh_retail_net/data/train.parquet',
              '6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d')
    print('Pinned source/metadata ready; full ISOMORPH corpus not downloaded.')

if __name__=='__main__': main()
