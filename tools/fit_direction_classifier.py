"""Fit the LR direction classifier on full IS daisy trades and save the
model + scaler as a single pickle so the ticker-engine strategy can load it
at startup.

Outputs:
  training_iso_v2/output/direction_clf.pkl
    {
      'scaler': StandardScaler,
      'clf':    LogisticRegression,
      'v2_cols': list of 184 V2 feature names in trained order,
      'is_test_auc': float (informational),
    }
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
    ap.add_argument('--out', default='training_iso_v2/output/direction_clf.pkl')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    v2_cols = [c for c in df.columns if c.startswith(('L1_','L2_','L3_'))]
    print(f'Loaded {len(df)} trades x {len(v2_cols)} V2 features')

    X = df[v2_cols].fillna(0).values.astype(np.float32)
    y = (df['direction'].values == 'LONG').astype(np.int8)

    # Train/test split for AUC reporting (same as direction classifier — seed 42)
    rng = np.random.default_rng(args.seed)
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

    # Refit on FULL IS so deployment uses all training data
    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
    clf_full.fit(X_all_s, y)
    print(f'  Refit on full IS: {len(X)} trades')

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler_full,
            'clf': clf_full,
            'v2_cols': v2_cols,
            'is_test_auc': float(auc),
            'is_test_argmax_acc': float(acc),
            'n_train_full': int(len(X)),
        }, f)
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
