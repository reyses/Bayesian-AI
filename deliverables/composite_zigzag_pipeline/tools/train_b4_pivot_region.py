"""B4 — Pivot-region classifier.

Per user 2026-05-17: reframe the question from "WHEN is the pivot?" to
"AM I CURRENTLY INSIDE the region around a pivot?".

Label: in_region_Ws  =  1 iff |bar_ts - nearest_pivot_ts| <= W seconds
Symmetric around pivot — captures both pre-pivot approach AND post-pivot
echo (volume residuals, mean-reversion lookback, etc.).

Why this should outperform B1's forward-only formulation:
  - The event study showed P_K=1 peaks at [+0m,+1m] (POST-pivot) almost
    as much as [-1m,+0m] (PRE-pivot). Forward-only labels miss the
    post-pivot signal half.
  - Symmetric label = 2x positive bars per pivot = easier to learn.
  - "Region identification" is more actionable for trade management:
    "you're in a pivot zone, prepare" beats "pivot in 4.5±5 min".

Trains binary GBM at multiple W:
  W=30s    : very tight (only bars AT the pivot moment)
  W=60s    : 2-min total region
  W=120s   : 4-min total region
  W=300s   : 10-min total region (similar to B1 K=10 but symmetric)

Train: full IS (282k bars)
Eval:  NT8 OOS (32 days)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


W_SECONDS = [30, 60, 120, 300]


def pivot_centroid_ts(day_df: pd.DataFrame) -> np.ndarray:
    """Collapse is_pivot==1 runs to centroid timestamps."""
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


def build_region_labels(df: pd.DataFrame, W_list) -> pd.DataFrame:
    """For each bar add 'in_region_Ws' binary column.
    1 iff |bar_ts - nearest_pivot_ts| <= W."""
    df = df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    cols = {f'in_region_{W}s': np.zeros(len(df), dtype=np.int8) for W in W_list}
    for day, g in df.groupby('day'):
        pivots = pivot_centroid_ts(g)
        if len(pivots) == 0:
            continue
        ts_arr = g['timestamp'].values.astype(np.int64)
        # Vectorized: for each bar, find nearest pivot (left or right)
        idx_right = np.searchsorted(pivots, ts_arr, side='left')
        for W in W_list:
            col = cols[f'in_region_{W}s']
            for i, ts in enumerate(ts_arr):
                k = idx_right[i]
                d_right = (pivots[k] - ts) if k < len(pivots) else np.inf
                d_left  = (ts - pivots[k-1]) if k > 0 else np.inf
                if min(d_left, d_right) <= W:
                    col[g.index[i]] = 1
    for k, v in cols.items():
        df[k] = v
    return df


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float('nan')
    base = float(y.mean())
    metrics = {'n': len(y), 'base_rate': base, 'auc': float(auc)}
    for thr in [0.30, 0.50, 0.70, 0.85, 0.95]:
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
        rec  = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0
        cov  = float((pred == 1).mean())
        metrics[f'thr_{thr:.2f}'] = {
            'prec': prec, 'rec': rec, 'coverage': cov,
            'tp': tp, 'fp': fp, 'fn': fn,
            'lift_over_base': prec / max(base, 1e-9),
        }
    return metrics, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b4_pivot_region.pkl')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b4_pivot_region.txt')
    ap.add_argument('--out-cache',
                    default='reports/findings/regret_oracle/b4_proba_OOS_NT8.parquet')
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading IS:', args.is_dataset)
    is_df = pd.read_parquet(args.is_dataset)
    print(f'  {len(is_df)} bars / {is_df["day"].nunique()} days')
    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    print('Building region labels (IS)...')
    is_df = build_region_labels(is_df, W_SECONDS)
    print('Building region labels (OOS)...')
    oos_df = build_region_labels(oos_df, W_SECONDS)

    for W in W_SECONDS:
        col = f'in_region_{W}s'
        print(f'  base rate {col}: IS={is_df[col].mean()*100:.2f}%  '
              f'OOS={oos_df[col].mean()*100:.2f}%')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    X_is  = is_df[v2_cols].fillna(0.0).values.astype(np.float32)
    X_oos = oos_df[v2_cols].fillna(0.0).values.astype(np.float32)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B4 PIVOT-REGION CLASSIFIER')
    out('  Label: in_region_W = 1 iff |bar_ts - nearest_pivot| <= W  (symmetric)')
    out('=' * 78)
    out(f'IS rows: {len(is_df):,}   OOS rows: {len(oos_df):,}   features: {len(v2_cols)}')

    models = {}
    for W in W_SECONDS:
        col = f'in_region_{W}s'
        y_is  = is_df[col].values
        y_oos = oos_df[col].values

        out(f'\n--- W={W}s --- positives IS {y_is.mean()*100:.2f}%, OOS {y_oos.mean()*100:.2f}%')
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.05,
            max_depth=6, min_samples_leaf=50,
            l2_regularization=0.5,
            class_weight='balanced',
            random_state=args.seed,
        )
        model.fit(X_is, y_is)

        is_metrics,  _      = evaluate(model, X_is,  y_is)
        oos_metrics, p_oos  = evaluate(model, X_oos, y_oos)

        out(f'IS  AUC = {is_metrics["auc"]:.4f}')
        out(f'OOS AUC = {oos_metrics["auc"]:.4f}  '
            f'(base rate {oos_metrics["base_rate"]*100:.2f}%)')
        for thr in [0.30, 0.50, 0.70, 0.85, 0.95]:
            m = oos_metrics[f'thr_{thr:.2f}']
            out(f'  thr={thr:.2f}:  prec={m["prec"]*100:5.1f}%  rec={m["rec"]*100:5.1f}%  '
                f'cov={m["coverage"]*100:5.2f}%  lift={m["lift_over_base"]:5.2f}x  '
                f'(tp={m["tp"]} fp={m["fp"]} fn={m["fn"]})')

        models[W] = {
            'model': model, 'v2_cols': v2_cols, 'W': W,
            'is_metrics': is_metrics, 'oos_metrics': oos_metrics,
        }

    # Per-bar predictions for inspector overlay
    print('\nWriting per-bar OOS predictions cache...')
    cache = oos_df[['timestamp', 'day']].copy()
    for W in W_SECONDS:
        cache[f'p_region_{W}s'] = models[W]['model'].predict_proba(X_oos)[:, 1]
        cache[f'in_region_{W}s_truth'] = oos_df[f'in_region_{W}s'].values
    cache.to_parquet(args.out_cache, index=False)

    out('')
    out('=' * 78)
    out('COMPARISON to B1 K=1 (forward-only label, ~1-min ahead)')
    out('=' * 78)
    out('  B1 K=1 thr=0.70 (NT8 OOS):  prec 13.6%  cov 4.48%   lift 2.61x')
    out('  B1 K=10 thr=0.85 (NT8 OOS): prec 78.1%  cov 3.88%   lift 1.89x')
    out('  --> B4 at comparable coverage should beat these if symmetric')
    out('       label exposes more signal.')

    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(models, f)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_cache}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
