"""B1 — Pivot-imminent classifier.

Question: at each 1m bar inside a leg, can the V2 features predict whether
a zigzag pivot will occur within the next K minutes?

If yes, this gives us an EARLY signal that the current leg is ending —
something the indicator can't produce (it confirms AFTER the reversal).

Per user 2026-05-17: "predict next bar with confidence for next leg with
at least 1s lead time" = pivot-imminent classification.

Labels (built inline from pivot timestamps):
  pivot_within_1m  : pivot in (t, t+60s]
  pivot_within_3m  : pivot in (t, t+180s]
  pivot_within_5m  : pivot in (t, t+300s]
  pivot_within_10m : pivot in (t, t+600s]

Training: GBM (HistGradientBoosting) with class_weight balanced — positives
are rare (~5-20% of bars depending on K).

Evaluation (NT8 OOS):
  - AUC, base rate, lift at top-decile threshold
  - Precision/recall at thresholds {0.3, 0.5, 0.7, 0.85}
  - Per-day stability check
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve


# K values (in minutes) to label and train at
K_MINUTES = [1, 3, 5, 10]


def pivot_centroid_ts(day_df: pd.DataFrame) -> np.ndarray:
    """Collapse consecutive is_pivot==1 runs to centroid timestamps.

    Bars within 90s of each other belong to the same pivot zone.
    Returns sorted array of pivot centroid timestamps (int64 seconds).
    """
    piv_rows = day_df[day_df['is_pivot'] == 1].sort_values('timestamp')
    if len(piv_rows) == 0:
        return np.array([], dtype=np.int64)
    ts = piv_rows['timestamp'].values.astype(np.int64)
    groups = [[ts[0]]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([ts[i]])
        else:
            groups[-1].append(ts[i])
    return np.array([int(np.median(g)) for g in groups], dtype=np.int64)


def build_labels(df: pd.DataFrame, k_minutes_list) -> pd.DataFrame:
    """For each 1m bar add boolean columns 'pivot_within_{K}m' for each K.

    Label = 1 iff any pivot centroid timestamp falls in (t, t + K*60].
    """
    df = df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    out_cols = {f'pivot_within_{K}m': np.zeros(len(df), dtype=np.int8)
                for K in k_minutes_list}
    for day, g in df.groupby('day'):
        piv_ts = pivot_centroid_ts(g)
        if len(piv_ts) == 0:
            continue
        ts_arr = g['timestamp'].values.astype(np.int64)
        idx_offset = g.index[0]
        for K in k_minutes_list:
            window_s = K * 60
            # For each bar t, is there a pivot in (t, t+window_s]?
            # searchsorted gives the index of first pivot strictly > t.
            # Then check whether that pivot is within the window.
            pos = np.searchsorted(piv_ts, ts_arr, side='right')
            mask = (pos < len(piv_ts)) & (piv_ts[np.clip(pos, 0, len(piv_ts)-1)]
                                           - ts_arr <= window_s)
            out_cols[f'pivot_within_{K}m'][g.index] = mask.astype(np.int8)
    for k, v in out_cols.items():
        df[k] = v
    return df


def evaluate(model, X, y, name='OOS'):
    """Return dict of metrics for a binary classifier output."""
    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float('nan')
    base_rate = float(y.mean())
    metrics = {'n': len(y), 'base_rate': base_rate, 'auc': float(auc)}
    for thr in [0.30, 0.50, 0.70, 0.85]:
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        cov  = (pred == 1).mean()
        metrics[f'thr_{thr:.2f}'] = {
            'prec': prec, 'rec': rec, 'coverage': cov,
            'tp': tp, 'fp': fp, 'fn': fn,
            'lift_over_base': prec / max(base_rate, 1e-9),
        }
    return metrics, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b1_pivot_imminent.pkl')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b1_pivot_imminent.txt')
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading IS:', args.is_dataset)
    is_df = pd.read_parquet(args.is_dataset)
    print(f'  {len(is_df)} bars / {is_df["day"].nunique()} days')
    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    print('Building labels (IS)...')
    is_df = build_labels(is_df, K_MINUTES)
    print('Building labels (OOS)...')
    oos_df = build_labels(oos_df, K_MINUTES)

    # Base rate summary
    for K in K_MINUTES:
        col = f'pivot_within_{K}m'
        print(f'  base rate {col}: IS={is_df[col].mean()*100:.2f}%  '
              f'OOS={oos_df[col].mean()*100:.2f}%')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    X_is  = is_df[v2_cols].fillna(0.0).values.astype(np.float32)
    X_oos = oos_df[v2_cols].fillna(0.0).values.astype(np.float32)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 72)
    out('B1 PIVOT-IMMINENT CLASSIFIER')
    out('truth = zigzag (ATR x4) pivots from build_zigzag_pivot_dataset.py')
    out('=' * 72)
    out(f'IS rows: {len(is_df):,}   OOS rows: {len(oos_df):,}   features: {len(v2_cols)}')

    models = {}
    for K in K_MINUTES:
        col = f'pivot_within_{K}m'
        y_is  = is_df[col].values
        y_oos = oos_df[col].values

        out(f'\n--- K={K} min --- (positives IS {y_is.mean()*100:.2f}%, OOS {y_oos.mean()*100:.2f}%) ---')
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.05,
            max_depth=6, min_samples_leaf=50,
            l2_regularization=0.5,
            class_weight='balanced',
            random_state=args.seed,
        )
        model.fit(X_is, y_is)

        is_metrics,  _      = evaluate(model, X_is,  y_is,  name='IS')
        oos_metrics, p_oos  = evaluate(model, X_oos, y_oos, name='OOS')

        out(f'IS  AUC = {is_metrics["auc"]:.4f}')
        out(f'OOS AUC = {oos_metrics["auc"]:.4f}  '
            f'(base rate {oos_metrics["base_rate"]*100:.2f}%)')
        for thr in [0.30, 0.50, 0.70, 0.85]:
            m = oos_metrics[f'thr_{thr:.2f}']
            out(f'  thr={thr:.2f}:  prec={m["prec"]*100:.1f}%  rec={m["rec"]*100:.1f}%  '
                f'cov={m["coverage"]*100:.2f}%  lift={m["lift_over_base"]:.2f}x  '
                f'(tp={m["tp"]} fp={m["fp"]} fn={m["fn"]})')

        models[K] = {
            'model': model, 'v2_cols': v2_cols, 'K': K,
            'is_metrics': is_metrics, 'oos_metrics': oos_metrics,
        }

    # Persist
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(models, f)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
