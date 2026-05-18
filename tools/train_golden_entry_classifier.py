"""Train + evaluate the entry-timing classifier.

Goal: P(this 1m bar is a GOLDEN entry moment | V2 features)
Where golden = oracle bar with mfe_dollars > $X.

Train on IS dataset (250+ days, ~360K bars, ~1.5K positives).
Evaluate on OOS dataset (68 days, ~100K bars, ~430 positives).

Outputs:
  - direction-classifier-style pickle for the trained model
  - threshold sweep CSV: at P > T, precision / recall / fires-per-day
  - reliability table
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset', required=True)
    ap.add_argument('--oos-dataset', required=True)
    ap.add_argument('--out-pkl', default='training_iso_v2/output/golden_entry_clf.pkl')
    ap.add_argument('--out-summary', default='reports/findings/regret_oracle/golden_entry_summary.csv')
    ap.add_argument('--model', choices=['lr', 'gbm'], default='lr')
    args = ap.parse_args()

    out_pkl = Path(args.out_pkl); out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary); out_summary.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading IS: {args.is_dataset}')
    is_df = pd.read_parquet(args.is_dataset)
    # Auto-detect label column (legacy 'is_golden' or new 'is_pivot')
    label_col = 'is_pivot' if 'is_pivot' in is_df.columns else 'is_golden'
    print(f'  Label column: {label_col}')
    print(f'  {len(is_df)} rows, {is_df[label_col].sum()} positives '
          f'({100*is_df[label_col].mean():.2f}%)')
    print(f'Loading OOS: {args.oos_dataset}')
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} rows, {oos_df[label_col].sum()} positives '
          f'({100*oos_df[label_col].mean():.2f}%)')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    X_is = is_df[v2_cols].fillna(0).values.astype(np.float32)
    y_is = is_df[label_col].values.astype(np.int8)
    X_oos = oos_df[v2_cols].fillna(0).values.astype(np.float32)
    y_oos = oos_df[label_col].values.astype(np.int8)

    scaler = StandardScaler()
    X_is_s = scaler.fit_transform(X_is)
    X_oos_s = scaler.transform(X_oos)

    print(f'\nFitting {args.model}...')
    import time
    t0 = time.time()
    if args.model == 'lr':
        clf = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs',
                                 class_weight='balanced')
        clf.fit(X_is_s, y_is)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                             learning_rate=0.1, class_weight='balanced',
                                             random_state=42)
        clf.fit(X_is, y_is)
    print(f'  Fit in {time.time()-t0:.1f}s')

    if args.model == 'lr':
        p_is = clf.predict_proba(X_is_s)[:, 1]
        p_oos = clf.predict_proba(X_oos_s)[:, 1]
    else:
        p_is = clf.predict_proba(X_is)[:, 1]
        p_oos = clf.predict_proba(X_oos)[:, 1]

    auc_is = roc_auc_score(y_is, p_is)
    auc_oos = roc_auc_score(y_oos, p_oos)
    ap_is = average_precision_score(y_is, p_is)
    ap_oos = average_precision_score(y_oos, p_oos)
    print(f'\n=== Headline ===')
    print(f'  IS  AUC: {auc_is:.4f}    PR-AUC: {ap_is:.4f}')
    print(f'  OOS AUC: {auc_oos:.4f}   PR-AUC: {ap_oos:.4f}')

    # Threshold sweep
    print(f'\n=== OOS threshold sweep ===')
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    n_days_oos = oos_df['day'].nunique()
    n_pos_oos = int(y_oos.sum())
    rows = []
    for t in thresholds:
        fires = p_oos > t
        n_fired = int(fires.sum())
        if n_fired == 0:
            continue
        n_tp = int(((fires) & (y_oos == 1)).sum())
        n_fp = n_fired - n_tp
        precision = n_tp / max(n_fired, 1)
        recall = n_tp / max(n_pos_oos, 1)
        fires_per_day = n_fired / max(n_days_oos, 1)
        row = {
            'threshold': t,
            'n_fired': n_fired, 'n_TP': n_tp, 'n_FP': n_fp,
            'precision': precision, 'recall': recall,
            'fires_per_day': fires_per_day,
            'fires_per_day_TP': n_tp / max(n_days_oos, 1),
        }
        rows.append(row)
        print(f'  T={t:.2f}  fires={n_fired:5d}  TP={n_tp:4d} FP={n_fp:5d}  '
              f'precision={precision:.3f}  recall={recall:.3f}  '
              f'fires/day={fires_per_day:.2f}')
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(out_summary, index=False)
    print(f'\nWrote: {out_summary}')

    # Save model
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'scaler': scaler if args.model == 'lr' else None,
            'clf': clf,
            'v2_cols': v2_cols,
            'model_kind': args.model,
            'auc_is': float(auc_is),
            'auc_oos': float(auc_oos),
            'mfe_threshold_label': '$200',
        }, f)
    print(f'Wrote: {out_pkl}')


if __name__ == '__main__':
    main()
