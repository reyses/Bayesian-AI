"""Train a DEDICATED direction classifier on zigzag-pivot bars only.

Hypothesis: V2 features at zigzag pivot bars (extremes/inflections) have
different direction signal characteristics than at average daisy oracle bars.
The current direction_clf.pkl was trained on ALL oracle bars (smooth setups
dominate → 81% acc in training distribution). But at high-timing-confidence
bars (= pivot bars), that acc degrades to ~30% (anti-predictive).

This trainer uses only is_pivot=1 rows, with pivot_dir as the target.

Inputs:
  IS dataset:  zz_pivot_IS_extended_atr4.parquet
  OOS dataset: zz_pivot_OOS_NT8_atr4.parquet
Output:
  training_iso_v2/output/direction_clf_pivot.pkl
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset',  required=True)
    ap.add_argument('--oos-dataset', required=True)
    ap.add_argument('--out-pkl', default='training_iso_v2/output/direction_clf_pivot.pkl')
    ap.add_argument('--model', choices=['lr','gbm'], default='lr')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    is_df  = pd.read_parquet(args.is_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'IS  rows: {len(is_df)}, pivots: {(is_df["is_pivot"]==1).sum()}')
    print(f'OOS rows: {len(oos_df)}, pivots: {(oos_df["is_pivot"]==1).sum()}')

    # Filter to pivot rows; build (X, y) where y = 1 if LONG else 0
    pos_is  = is_df[is_df['is_pivot'] == 1].copy()
    pos_oos = oos_df[oos_df['is_pivot'] == 1].copy()
    pos_is  = pos_is[pos_is['pivot_dir'].isin(['LONG','SHORT'])]
    pos_oos = pos_oos[pos_oos['pivot_dir'].isin(['LONG','SHORT'])]
    print(f'Filtered pivots — IS: {len(pos_is)}, OOS: {len(pos_oos)}')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_','L2_','L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    X_is  = pos_is[v2_cols].fillna(0).values.astype(np.float32)
    y_is  = (pos_is['pivot_dir'].values == 'LONG').astype(np.int8)
    X_oos = pos_oos[v2_cols].fillna(0).values.astype(np.float32)
    y_oos = (pos_oos['pivot_dir'].values == 'LONG').astype(np.int8)
    print(f'IS LONG rate: {y_is.mean():.3f}    OOS LONG rate: {y_oos.mean():.3f}')

    # 80/20 split for IS-test reporting
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X_is)); rng.shuffle(idx)
    n_test = int(0.2 * len(X_is))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_is[train_idx])
    X_te_s = scaler.transform(X_is[test_idx])
    X_oos_s = scaler.transform(X_oos)

    if args.model == 'lr':
        clf = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
        clf.fit(X_tr_s, y_is[train_idx])
        p_te  = clf.predict_proba(X_te_s)[:, 1]
        p_oos = clf.predict_proba(X_oos_s)[:, 1]
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                              learning_rate=0.1, random_state=42)
        clf.fit(X_is[train_idx], y_is[train_idx])
        p_te  = clf.predict_proba(X_is[test_idx])[:, 1]
        p_oos = clf.predict_proba(X_oos)[:, 1]

    auc_te  = roc_auc_score(y_is[test_idx], p_te)
    brier_te = brier_score_loss(y_is[test_idx], p_te)
    acc_te  = ((p_te > 0.5).astype(np.int8) == y_is[test_idx]).mean()
    auc_oos  = roc_auc_score(y_oos, p_oos)
    brier_oos = brier_score_loss(y_oos, p_oos)
    acc_oos  = ((p_oos > 0.5).astype(np.int8) == y_oos).mean()

    print(f'\n=== Headline ===')
    print(f'  IS-test  AUC: {auc_te:.4f}   acc: {acc_te:.4f}   Brier: {brier_te:.4f}')
    print(f'  NT8 OOS  AUC: {auc_oos:.4f}   acc: {acc_oos:.4f}   Brier: {brier_oos:.4f}')

    # Threshold sweep — what's the precision at each confidence cutoff?
    print(f'\n=== NT8 OOS threshold sweep ===')
    print(f'  {"thr":>5} {"cov":>5} {"acc":>5} {"LONG_acc":>9} {"SHORT_acc":>9}')
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        long_mask = p_oos > thr
        short_mask = p_oos < (1 - thr)
        fire_mask = long_mask | short_mask
        if fire_mask.sum() == 0:
            print(f'  {thr:.2f} {0:.2f} (no fires)')
            continue
        long_acc = (y_oos[long_mask] == 1).mean() if long_mask.any() else float('nan')
        short_acc = (y_oos[short_mask] == 0).mean() if short_mask.any() else float('nan')
        correct = np.where(long_mask, y_oos == 1, np.where(short_mask, y_oos == 0, False))
        overall = correct[fire_mask].mean()
        cov = fire_mask.mean()
        print(f'  {thr:.2f} {cov:.2f}  {overall:.3f}  {long_acc:.3f}    {short_acc:.3f}')

    # Refit on full IS
    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X_is)
    if args.model == 'lr':
        clf_full = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
        clf_full.fit(X_all_s, y_is)
    else:
        clf_full = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                   learning_rate=0.1, random_state=42)
        clf_full.fit(X_is, y_is)

    out_pkl = Path(args.out_pkl); out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'scaler': scaler_full if args.model == 'lr' else None,
            'clf': clf_full,
            'v2_cols': v2_cols,
            'model_kind': args.model,
            'auc_oos': float(auc_oos),
            'acc_oos': float(acc_oos),
            'is_pivot_only': True,
        }, f)
    print(f'\nWrote: {out_pkl}')


if __name__ == '__main__':
    main()
