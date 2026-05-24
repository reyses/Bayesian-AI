"""Pre-cache P(LONG) per 1m bar using the direction classifier.
Same shape as golden_timing_cache for visualizer overlay.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model-pkl', default='training_iso_v2/output/direction_clf.pkl')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.model_pkl, 'rb') as f:
        pl = pickle.load(f)
    clf = pl['clf']; scaler = pl['scaler']; v2_cols = list(pl['v2_cols'])

    df = pd.read_parquet(args.dataset)
    print(f'Dataset: {len(df)} rows, cols: {len(v2_cols)}')
    X = df[v2_cols].fillna(0).values.astype(np.float32)
    Xs = scaler.transform(X)
    p = clf.predict_proba(Xs)[:, 1]

    out = df[['timestamp','day']].copy()
    out['p_long'] = p.astype(np.float32)
    out.to_parquet(args.out, index=False)
    print(f'Wrote: {args.out}')
    print(f'  p_long range: [{p.min():.4f}, {p.max():.4f}]  mean={p.mean():.4f}')


if __name__ == '__main__':
    main()
