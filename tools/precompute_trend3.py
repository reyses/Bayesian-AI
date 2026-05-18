"""Pre-cache trend3 classifier predictions per bar for inspector overlay.

Output columns:
  timestamp, day, p_long, p_short, p_neutral, argmax_class
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
    ap.add_argument('--model-pkl', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.model_pkl, 'rb') as f:
        pl = pickle.load(f)
    clf = pl['clf']
    v2_cols = list(pl['v2_cols'])
    classes = list(pl['classes'])
    kind = pl.get('model_kind', 'gbm')
    print(f'Model: {kind}, classes: {classes}, cols: {len(v2_cols)}')

    df = pd.read_parquet(args.dataset)
    print(f'Dataset: {len(df)} rows')

    X = df[v2_cols].fillna(0).values.astype(np.float32)
    if kind == 'lr':
        scaler = pl['scaler']
        proba = clf.predict_proba(scaler.transform(X))
    else:
        proba = clf.predict_proba(X)

    long_i  = classes.index('LONG')
    short_i = classes.index('SHORT')
    neut_i  = classes.index('NEUTRAL')

    out = df[['timestamp','day']].copy()
    out['p_long']    = proba[:, long_i].astype(np.float32)
    out['p_short']   = proba[:, short_i].astype(np.float32)
    out['p_neutral'] = proba[:, neut_i].astype(np.float32)
    argmax = proba.argmax(1)
    out['pred_class'] = [classes[i] for i in argmax]
    out.to_parquet(args.out, index=False)
    print(f'Wrote: {args.out}')
    print(f'  Predicted class distribution:')
    print(out['pred_class'].value_counts())


if __name__ == '__main__':
    main()
