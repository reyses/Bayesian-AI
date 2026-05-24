"""Pre-compute GBM P_timing for every 1m bar in IS and OOS.

Output: parquet cached lookup table {day, timestamp: p_timing}.
Strategy at run-time does dict lookup → no per-bar inference cost.
"""
from __future__ import annotations
import argparse
import pickle
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    help='golden_entry_dataset parquet (IS or OOS)')
    ap.add_argument('--model-pkl', required=True,
                    help='golden_entry_clf_gbm.pkl or _lr.pkl')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.model_pkl, 'rb') as f:
        pl = pickle.load(f)
    clf = pl['clf']
    v2_cols = pl['v2_cols']
    kind = pl.get('model_kind', 'lr')
    print(f'Loaded {kind}; cols={len(v2_cols)}')

    df = pd.read_parquet(args.dataset)
    print(f'Dataset: {len(df)} rows')

    X = df[v2_cols].fillna(0).values.astype(np.float32)
    if kind == 'lr':
        scaler = pl['scaler']
        Xs = scaler.transform(X)
        p = clf.predict_proba(Xs)[:, 1]
    else:
        # GBM: batch predict (fast vs per-row)
        p = clf.predict_proba(X)[:, 1]

    out = df[['timestamp', 'day']].copy()
    out['p_timing'] = p.astype(np.float32)
    # Carry through whatever label/direction columns the dataset has
    label_col = 'is_pivot' if 'is_pivot' in df.columns else 'is_golden'
    dir_col = 'pivot_dir' if 'pivot_dir' in df.columns else 'oracle_dir'
    px_col = 'pivot_price' if 'pivot_price' in df.columns else 'oracle_mfe'
    out['is_golden'] = df[label_col].astype(np.int8)   # rename for backward compat
    if dir_col in df.columns:
        out['oracle_dir'] = df[dir_col]
    if px_col in df.columns:
        out['oracle_mfe'] = df[px_col]
    out.to_parquet(args.out, index=False)
    print(f'Wrote: {args.out}')
    print(f'  p_timing range: [{p.min():.4f}, {p.max():.4f}]  mean={p.mean():.4f}')


if __name__ == '__main__':
    main()
