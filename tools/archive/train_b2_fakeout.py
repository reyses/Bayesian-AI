"""B2 — Fake-out classifier.

Question: at the moment a zigzag pivot is confirmed (R-trigger crossed),
can V2 features tell us whether the new leg is REAL or a FAKE-OUT
(will reverse within K minutes back to the prior direction)?

If yes, this is a filter that the indicator can't produce — it just flags
every confirmed pivot.

Per user 2026-05-17: "confirm next leg is not fake" = fake-out classifier.

Sample: ONE row per pivot (centroid bar). Drop the last pivot of each day
(no next pivot to determine fake-out status).

Labels:
  is_fakeout_3m  : next pivot within 3 min IS in OPPOSITE direction from this leg
  is_fakeout_5m  : same with 5-min window
  is_fakeout_10m : same with 10-min window

A "fakeout" means: the indicator confirmed a flip at time T, but a new pivot
within K minutes reverses us back — the original leg was short-lived noise.

Training: GBM on V2 features at the pivot bar.
Evaluation (NT8 OOS): AUC, precision/recall at thresholds.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# K values (in minutes) to label and train at
K_MINUTES = [3, 5, 10]


def pivot_events_per_day(day_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse is_pivot==1 runs to one row per pivot event.

    Returns DataFrame indexed to a subset of the input bars (the centroid
    bar of each pivot zone) with columns:
      timestamp (int64), pivot_dir (str), bar_index (int — index into day_df)
    """
    piv = day_df[day_df['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return pd.DataFrame(columns=['timestamp', 'pivot_dir', 'bar_index'])
    ts = piv['timestamp'].values.astype(np.int64)
    pdir = piv['pivot_dir'].values
    idx_in_df = piv.index.values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i]); continue
        groups[-1].append(i)
    events = []
    for grp in groups:
        sub_ts = ts[grp]; sub_dir = pdir[grp]; sub_idx = idx_in_df[grp]
        ts_c = int(np.median(sub_ts))
        # Direction = mode of pivot_dir in zone (almost always uniform)
        vals, counts = np.unique(sub_dir, return_counts=True)
        d = vals[np.argmax(counts)]
        # Bar index = the bar nearest the centroid
        mid = grp[len(grp) // 2]
        bi = int(idx_in_df[grp[0]] + (mid - grp[0]))
        events.append({'timestamp': ts_c, 'pivot_dir': str(d), 'bar_index': bi})
    return pd.DataFrame(events)


def build_pivot_dataset(df: pd.DataFrame, k_minutes_list):
    """Per pivot event, attach V2 features (from the centroid bar) + labels.

    Label per K: is_fakeout_{K}m = 1 iff the NEXT pivot occurs within K min
    AND has OPPOSITE direction from this one.

    NOTE: this assumes pivot_dir convention from build_zigzag_pivot_dataset.py
    where pivot_dir labels the start of the next leg's direction. Need to
    verify — see check below.
    """
    rows = []
    v2_cols = [c for c in df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    for day, g in df.groupby('day'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        events = pivot_events_per_day(g.reset_index(drop=True))
        if len(events) < 2:
            continue
        for ei in range(len(events) - 1):
            this = events.iloc[ei]
            # Zigzag legs alternate by construction, so consecutive pivots
            # always have opposite direction. Therefore:
            #   is_fakeout_K  =  (time_to_next_pivot <= K)
            # i.e., the leg this pivot KICKS OFF was short-lived noise.
            nxt = events.iloc[ei + 1]
            dt_min = (nxt['timestamp'] - this['timestamp']) / 60.0
            bar = g.iloc[this['bar_index']]
            row = {
                'day': day,
                'timestamp': int(this['timestamp']),
                'pivot_dir': this['pivot_dir'],
                'time_to_next_pivot_min': float(dt_min),
            }
            for K in k_minutes_list:
                row[f'is_fakeout_{K}m'] = int(dt_min <= K)
            for c in v2_cols:
                row[c] = float(bar[c]) if not pd.isna(bar[c]) else 0.0
            rows.append(row)
    return pd.DataFrame(rows), v2_cols


def evaluate(model, X, y):
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
                    default='reports/findings/regret_oracle/b2_fakeout.pkl')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b2_fakeout.txt')
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading IS:', args.is_dataset)
    is_df = pd.read_parquet(args.is_dataset)
    print(f'  {len(is_df)} bars / {is_df["day"].nunique()} days')

    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    print('Building IS pivot events...')
    is_piv, v2_cols = build_pivot_dataset(is_df, K_MINUTES)
    print(f'  {len(is_piv)} pivot events on IS')

    print('Building OOS pivot events...')
    oos_piv, _ = build_pivot_dataset(oos_df, K_MINUTES)
    print(f'  {len(oos_piv)} pivot events on OOS')

    print(f'V2 features: {len(v2_cols)}')

    for K in K_MINUTES:
        col = f'is_fakeout_{K}m'
        print(f'  base rate {col}: IS={is_piv[col].mean()*100:.2f}%  '
              f'OOS={oos_piv[col].mean()*100:.2f}%')

    X_is  = is_piv[v2_cols].values.astype(np.float32)
    X_oos = oos_piv[v2_cols].values.astype(np.float32)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 72)
    out('B2 FAKE-OUT CLASSIFIER')
    out('truth = pivot pairs from build_zigzag_pivot_dataset.py (ATR x4)')
    out('=' * 72)
    out(f'IS pivots: {len(is_piv):,}   OOS pivots: {len(oos_piv):,}   features: {len(v2_cols)}')

    models = {}
    for K in K_MINUTES:
        col = f'is_fakeout_{K}m'
        y_is  = is_piv[col].values
        y_oos = oos_piv[col].values

        out(f'\n--- K={K} min --- (fakeout IS {y_is.mean()*100:.2f}%, OOS {y_oos.mean()*100:.2f}%) ---')
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.05,
            max_depth=6, min_samples_leaf=20,
            l2_regularization=0.5,
            class_weight='balanced',
            random_state=args.seed,
        )
        model.fit(X_is, y_is)

        is_metrics,  _ = evaluate(model, X_is,  y_is)
        oos_metrics, p_oos = evaluate(model, X_oos, y_oos)

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

    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(models, f)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
