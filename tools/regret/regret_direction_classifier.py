"""Binary direction classifier with calibrated P(LONG | entry features).

Per user 2026-05-16: instead of predicting cluster ID (failed at R²=−0.05)
or magnitude (also weak), train a direct binary classifier whose output is
the *confidence* of the direction call. That confidence becomes the
selector's fire/no-fire dial.

For each trade, predict direction (LONG=1, SHORT=0) from V2 entry features.
Output a calibrated P(LONG). At inference:
    P(LONG) > 0.5  → fire LONG
    P(LONG) < 0.5  → fire SHORT
    |P − 0.5|     → confidence; gate the selector on this

This is what the L4 selector uses. We measure:
  - AUC-ROC: overall discrimination
  - Brier score: calibration (lower = better-calibrated probabilities)
  - Threshold sweep: at P-threshold = T, what's the coverage + accuracy?
  - Reliability diagram: do our probabilities match observed rates?

Optionally include lead-in PCA signature alongside entry features for
extra context.

Output:
    direction_classifier_<name>.npz   per-trade P(LONG) + meta
    direction_classifier_<name>.csv   threshold sweep table
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='daisy_with_v2_features parquet (must have direction + V2 features)')
    ap.add_argument('--leadin-signatures', default=None,
                    help='Optional: lead-in signatures npz to concatenate as extra features')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='IS_full_v2_direction')
    ap.add_argument('--model', default='lr', choices=['lr', 'gbm', 'gbm_calibrated'])
    ap.add_argument('--test-frac', type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.input}')
    df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)

    # V2 features at entry
    v2_cols = [c for c in df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'  {len(df)} trades x {len(v2_cols)} V2 entry features')

    # Optional: lead-in centroid + direction
    extra_cols = 0
    if args.leadin_signatures:
        print(f'Loading lead-in signatures: {args.leadin_signatures}')
        li = np.load(args.leadin_signatures, allow_pickle=True)
        li_oidx = li['oracle_idx']
        li_centroids = li['centroids']
        li_directions = li['directions']
        li_unstable = li['pca_unstable']
        # Map by oracle_idx
        n_li_feats = li_centroids.shape[1]
        li_lookup = {int(o): i for i, o in enumerate(li_oidx)}
        extra = np.full((len(df), 2 * n_li_feats), np.nan, dtype=np.float32)
        skipped_li = 0
        for ri, oid in enumerate(df['oracle_idx'].values):
            i = li_lookup.get(int(oid))
            if i is None or li_unstable[i]:
                skipped_li += 1
                continue
            extra[ri, :n_li_feats] = li_centroids[i]
            extra[ri, n_li_feats:] = li_directions[i]
        extra_cols = extra.shape[1]
        print(f'  Added {extra_cols} lead-in features ({skipped_li} trades without stable lead-in)')

    # Build X, y
    X_v2 = df[v2_cols].fillna(0).values.astype(np.float32)
    if extra_cols > 0:
        X = np.hstack([X_v2, np.nan_to_num(extra, nan=0.0).astype(np.float32)])
    else:
        X = X_v2
    print(f'Full feature matrix: {X.shape}')

    y = (df['direction'].values == 'LONG').astype(np.int8)
    print(f'  LONG: {y.sum()}    SHORT: {(y==0).sum()}    base rate LONG: {y.mean():.3f}')

    # Train/test split (random)
    rng = np.random.default_rng(42)
    idx = np.arange(len(X)); rng.shuffle(idx)
    n_test = int(args.test_frac * len(X))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f'  Train: {len(X_tr)}    Test: {len(X_te)}')

    # Standardize for LR
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)

    # Fit
    print(f'\nFitting {args.model}...')
    import time
    t0 = time.time()
    if args.model == 'lr':
        clf = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
        clf.fit(Xs_tr, y_tr)
        proba_te = clf.predict_proba(Xs_te)[:, 1]
        proba_tr = clf.predict_proba(Xs_tr)[:, 1]
    elif args.model == 'gbm':
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42)
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]
        proba_tr = clf.predict_proba(X_tr)[:, 1]
    else:  # gbm_calibrated
        base = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42)
        clf = CalibratedClassifierCV(base, method='isotonic', cv=3)
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]
        proba_tr = clf.predict_proba(X_tr)[:, 1]
    print(f'  Fit in {time.time()-t0:.1f}s')

    # Metrics
    auc_te = roc_auc_score(y_te, proba_te)
    brier_te = brier_score_loss(y_te, proba_te)
    auc_tr = roc_auc_score(y_tr, proba_tr)
    brier_tr = brier_score_loss(y_tr, proba_tr)
    print(f'\n=== Headline metrics ===')
    print(f'  Train  AUC: {auc_tr:.4f}    Brier: {brier_tr:.4f}')
    print(f'  Test   AUC: {auc_te:.4f}    Brier: {brier_te:.4f}')
    print(f'  (AUC: 0.5=chance, 1.0=perfect.  Brier: lower better, 0.25=random)')

    # Threshold sweep (on test set)
    print(f'\n=== Threshold sweep (test set) ===')
    print(f'  Fire LONG  if P(LONG) > T        (high confidence LONG)')
    print(f'  Fire SHORT if P(LONG) < 1-T      (high confidence SHORT)')
    print(f'  Skip if  1-T <= P(LONG) <= T     (uncertain — selector abstains)')
    print()
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep_rows = []
    for th in thresholds:
        long_mask = proba_te > th
        short_mask = proba_te < (1 - th)
        fire_mask = long_mask | short_mask
        cov = fire_mask.mean()
        n_fired = int(fire_mask.sum())
        if n_fired == 0:
            sweep_rows.append({'threshold': th, 'coverage': cov, 'n_fired': 0,
                              'long_acc': float('nan'), 'short_acc': float('nan'),
                              'overall_acc': float('nan')})
            continue
        # Direction accuracy when we fire
        # For LONG fires: correct = y_te == 1
        # For SHORT fires: correct = y_te == 0
        correct = np.zeros(len(y_te), dtype=bool)
        correct[long_mask] = (y_te[long_mask] == 1)
        correct[short_mask] = (y_te[short_mask] == 0)
        overall_acc = correct[fire_mask].mean()
        long_acc = correct[long_mask].mean() if long_mask.any() else float('nan')
        short_acc = correct[short_mask].mean() if short_mask.any() else float('nan')
        sweep_rows.append({
            'threshold': th,
            'coverage': round(float(cov), 4),
            'n_fired': n_fired,
            'long_acc': round(float(long_acc), 4) if not np.isnan(long_acc) else None,
            'short_acc': round(float(short_acc), 4) if not np.isnan(short_acc) else None,
            'overall_acc': round(float(overall_acc), 4),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))

    # Reliability table (binned)
    print(f'\n=== Reliability (calibration) — predicted P-bin vs actual LONG rate ===')
    bins = np.linspace(0, 1, 11)
    rel_rows = []
    for i in range(len(bins) - 1):
        mask = (proba_te >= bins[i]) & (proba_te < bins[i+1])
        if mask.sum() < 10:
            continue
        rel_rows.append({
            'p_bin': f'[{bins[i]:.1f}, {bins[i+1]:.1f})',
            'n': int(mask.sum()),
            'mean_pred_P': round(float(proba_te[mask].mean()), 4),
            'actual_long_rate': round(float(y_te[mask].mean()), 4),
            'delta': round(float(proba_te[mask].mean() - y_te[mask].mean()), 4),
        })
    rel_df = pd.DataFrame(rel_rows)
    print(rel_df.to_string(index=False))

    # Save outputs
    summary_path = out_dir / f'direction_classifier_{args.name}_summary.csv'
    sweep_df.to_csv(summary_path, index=False)
    rel_path = out_dir / f'direction_classifier_{args.name}_reliability.csv'
    rel_df.to_csv(rel_path, index=False)

    # Per-trade probabilities for all trades (re-predict on full set)
    if args.model == 'lr':
        proba_all = clf.predict_proba(scaler.transform(X))[:, 1]
    else:
        proba_all = clf.predict_proba(X)[:, 1]
    out_npz = out_dir / f'direction_classifier_{args.name}.npz'
    is_test = np.zeros(len(X), dtype=bool)
    is_test[test_idx] = True
    np.savez_compressed(
        out_npz,
        oracle_idx=df['oracle_idx'].values,
        p_long=proba_all.astype(np.float32),
        direction_true=y,
        is_test=is_test,
        auc_test=auc_te,
        brier_test=brier_te,
    )
    print(f'\nWrote: {summary_path}')
    print(f'Wrote: {rel_path}')
    print(f'Wrote: {out_npz}')


if __name__ == '__main__':
    main()
