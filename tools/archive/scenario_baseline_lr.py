"""Baseline: LogisticRegression on V2 entry features (no sequence) predicting
the same 4 scenario buckets the LSTM predicts. Apples-to-apples comparison.

If LR matches LSTM accuracy, the sequence input adds nothing — the model is
just learning the entry-bar features.

For each of (direction, duration, speed, trajectory):
  - Train LR on entry features only
  - Eval on IS-test (20% holdout, seed 42 for parity)
  - Eval on OOS (2026)
  - Report per-head accuracy and OOS gap
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-input', required=True,
                    help='daisy_with_v2_features parquet (IS)')
    ap.add_argument('--is-buckets', required=True,
                    help='daisy_IS_buckets.csv')
    ap.add_argument('--oos-input', required=True,
                    help='daisy_with_v2_features parquet (OOS)')
    ap.add_argument('--oos-buckets', required=True,
                    help='daisy_OOS_buckets.csv')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='scenario_lr_baseline')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load IS
    print(f'Loading IS: {args.is_input}')
    is_df = pd.read_parquet(args.is_input)
    is_b = pd.read_csv(args.is_buckets).set_index('oracle_idx')
    bucket_cols = ['bucket_direction','bucket_duration','bucket_speed','bucket_trajectory']
    # Drop any pre-existing bucket cols from V2 parquet to avoid merge suffixes
    is_df = is_df.drop(columns=[c for c in bucket_cols if c in is_df.columns])
    is_df = is_df.merge(
        is_b[bucket_cols],
        left_on='oracle_idx', right_index=True, how='inner'
    )
    v2_cols = [c for c in is_df.columns if c.startswith(('L1_','L2_','L3_'))]
    print(f'  IS: {len(is_df)} trades, {len(v2_cols)} V2 features')

    X_is = is_df[v2_cols].fillna(0).values.astype(np.float32)
    y_is = {
        'dir': is_df['bucket_direction'].values.astype(np.int64),
        'dur': is_df['bucket_duration'].values.astype(np.int64),
        'spd': is_df['bucket_speed'].values.astype(np.int64),
        'traj': is_df['bucket_trajectory'].values.astype(np.int64),
    }

    # Train/val (test) split — same as LSTM and direction classifier
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X_is)); rng.shuffle(idx)
    n_test = int(0.2 * len(X_is))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_tr, X_te = X_is[train_idx], X_is[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Load OOS
    print(f'Loading OOS: {args.oos_input}')
    oos_df = pd.read_parquet(args.oos_input)
    oos_b = pd.read_csv(args.oos_buckets).set_index('oracle_idx')
    oos_df = oos_df.drop(columns=[c for c in bucket_cols if c in oos_df.columns])
    oos_df = oos_df.merge(
        oos_b[bucket_cols],
        left_on='oracle_idx', right_index=True, how='inner'
    )
    X_oos = oos_df[v2_cols].fillna(0).values.astype(np.float32)
    X_oos_s = scaler.transform(X_oos)
    y_oos = {
        'dir': oos_df['bucket_direction'].values.astype(np.int64),
        'dur': oos_df['bucket_duration'].values.astype(np.int64),
        'spd': oos_df['bucket_speed'].values.astype(np.int64),
        'traj': oos_df['bucket_trajectory'].values.astype(np.int64),
    }
    print(f'  OOS: {len(oos_df)} trades')

    summary = {}
    for head in ('dir', 'dur', 'spd', 'traj'):
        y_tr = y_is[head][train_idx]
        y_te = y_is[head][test_idx]
        y_oos_h = y_oos[head]
        n_classes = max(y_tr.max(), y_te.max(), y_oos_h.max()) + 1
        clf = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs',
                                 class_weight='balanced')
        clf.fit(X_tr_s, y_tr)
        acc_tr = clf.score(X_tr_s, y_tr)
        acc_te = clf.score(X_te_s, y_te)
        acc_oos = clf.score(X_oos_s, y_oos_h)
        base_te = np.bincount(y_te).max() / len(y_te)
        base_oos = np.bincount(y_oos_h).max() / len(y_oos_h)
        print(f'\n{head}: n_classes={n_classes}')
        print(f'  Train acc: {acc_tr:.4f}')
        print(f'  IS test:   {acc_te:.4f}   (baseline {base_te:.4f})')
        print(f'  OOS:       {acc_oos:.4f}   (baseline {base_oos:.4f})')
        print(f'  IS-OOS delta: {acc_oos-acc_te:+.4f}')
        summary[head] = {
            'train_acc': float(acc_tr),
            'is_test_acc': float(acc_te),
            'oos_acc': float(acc_oos),
            'is_test_baseline': float(base_te),
            'oos_baseline': float(base_oos),
            'n_classes': int(n_classes),
        }

    with open(out_dir / f'{args.name}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote: {out_dir / f"{args.name}_summary.json"}')


if __name__ == '__main__':
    main()
