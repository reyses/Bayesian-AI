"""Pre-compute B1 (pivot-imminent) and B2 (fakeout) probabilities for the
NT8 OOS days so the inspector can overlay them without re-running the model.

Output parquets:
  reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet
    columns: timestamp, day, p_pivot_1m, p_pivot_3m, p_pivot_5m, p_pivot_10m
  reports/findings/regret_oracle/b2_proba_OOS_NT8.parquet
    columns: timestamp, day, p_fakeout_3m, p_fakeout_5m, p_fakeout_10m
    (one row per pivot event — not per bar)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_b1_pivot_imminent import K_MINUTES as B1_K
from train_b2_fakeout import build_pivot_dataset, K_MINUTES as B2_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1-pkl', default='reports/findings/regret_oracle/b1_pivot_imminent.pkl')
    ap.add_argument('--b2-pkl', default='reports/findings/regret_oracle/b2_fakeout.pkl')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-b1', default='reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
    ap.add_argument('--out-b2', default='reports/findings/regret_oracle/b2_proba_OOS_NT8.parquet')
    args = ap.parse_args()

    print('Loading OOS:', args.oos_dataset)
    oos = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos)} bars / {oos["day"].nunique()} days')

    with open(args.b1_pkl, 'rb') as f:
        b1 = pickle.load(f)
    with open(args.b2_pkl, 'rb') as f:
        b2 = pickle.load(f)

    # --- B1: per-bar probabilities ---
    print('Computing B1 per-bar probabilities...')
    cols = b1[B1_K[0]]['v2_cols']
    X = oos[cols].fillna(0.0).values.astype(np.float32)
    out_b1 = oos[['timestamp', 'day']].copy()
    for K in B1_K:
        out_b1[f'p_pivot_{K}m'] = b1[K]['model'].predict_proba(X)[:, 1]
    out_b1.to_parquet(args.out_b1, index=False)
    print(f'  -> {args.out_b1}  ({len(out_b1)} rows)')

    # --- B2: per-pivot probabilities ---
    print('Computing B2 per-pivot probabilities...')
    piv, v2_cols = build_pivot_dataset(oos, B2_K)
    X_piv = piv[v2_cols].values.astype(np.float32)
    out_b2 = piv[['timestamp', 'day']].copy()
    for K in B2_K:
        out_b2[f'p_fakeout_{K}m'] = b2[K]['model'].predict_proba(X_piv)[:, 1]
    out_b2.to_parquet(args.out_b2, index=False)
    print(f'  -> {args.out_b2}  ({len(out_b2)} rows)')


if __name__ == '__main__':
    main()
