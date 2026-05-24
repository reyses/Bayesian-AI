"""Fit direction classifier on velocity-filtered oracle bars only.

Hypothesis: high-velocity moves have stronger directional signal, so a
classifier trained on JUST velocity-filtered bars (vel > $5/min) should
have higher accuracy than the all-oracle baseline.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='reports/findings/regret_oracle/daisy_with_v2_features_IS_full.parquet')
    ap.add_argument('--oracle-csv', default='reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv')
    ap.add_argument('--velocity-threshold', type=float, default=5.0)
    ap.add_argument('--out', default='training_iso_v2/output/direction_clf_vel5.pkl')
    args = ap.parse_args()

    # Join V2 features with velocity from oracle CSV
    df = pd.read_parquet(args.input)
    if 'mfe_velocity' not in df.columns:
        oracle = pd.read_csv(args.oracle_csv)
        df = df.merge(oracle[['oracle_idx','mfe_velocity']], on='oracle_idx', how='left')

    print(f'Total trades: {len(df)}')
    high_vel = df[df['mfe_velocity'] > args.velocity_threshold].copy()
    print(f'High-velocity (>${args.velocity_threshold}/min): {len(high_vel)} ({100*len(high_vel)/len(df):.1f}%)')

    v2_cols = [c for c in df.columns if c.startswith(('L1_','L2_','L3_'))]

    X = high_vel[v2_cols].fillna(0).values.astype(np.float32)
    y = (high_vel['direction'].values == 'LONG').astype(np.int8)

    rng = np.random.default_rng(42)
    idx = np.arange(len(X)); rng.shuffle(idx)
    n_test = int(0.2 * len(X))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[train_idx])
    X_te_s = scaler.transform(X[test_idx])

    clf = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
    clf.fit(X_tr_s, y[train_idx])
    p_te = clf.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y[test_idx], p_te)
    acc = ((p_te > 0.5).astype(np.int8) == y[test_idx]).mean()
    print(f'  AUC test: {auc:.4f}    argmax acc: {acc:.4f}')

    # Refit on full
    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
    clf_full.fit(X_all_s, y)
    print(f'  Refit on full ({len(X)} trades)')

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler_full,
            'clf': clf_full,
            'v2_cols': v2_cols,
            'velocity_threshold': args.velocity_threshold,
            'auc_test': float(auc),
            'argmax_acc_test': float(acc),
        }, f)
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
