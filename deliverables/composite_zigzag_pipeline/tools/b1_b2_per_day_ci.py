"""Per-day bootstrap CI for B1 (pivot-imminent) and B2 (fakeout) OOS metrics.

Loads the trained pickles, reapplies models to the OOS dataset, computes
per-day metrics (AUC + precision at threshold), and bootstraps the daily
distribution for 95% CIs on the mean.

Per CLAUDE.md: "Bootstrap each population independently, 4,000 resamples,
take CI of mean". This is the rigorous version of the headline numbers
from train_b1_pivot_imminent.py / train_b2_fakeout.py.

Output:
  - reports/findings/regret_oracle/b1_per_day.csv
  - reports/findings/regret_oracle/b2_per_day.csv
  - reports/findings/regret_oracle/b1_b2_per_day_ci.txt  (summary)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import label builders from the training scripts
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_b1_pivot_imminent import build_labels as build_b1_labels
from train_b1_pivot_imminent import K_MINUTES as B1_K
from train_b2_fakeout import build_pivot_dataset
from train_b2_fakeout import K_MINUTES as B2_K


def bootstrap_ci_mean(values, n=4000, seed=42):
    """95% bootstrap CI on the mean of a 1D array."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for i in range(n):
        boots[i] = arr[rng.integers(0, len(arr), len(arr))].mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_day_b1_metrics(model, oos_df, K_list, thresholds=(0.30, 0.50, 0.70, 0.85)):
    """For each day in OOS, compute per-K AUC + per-(K, thr) precision/recall."""
    v2_cols = [c for c in oos_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    X = oos_df[v2_cols].fillna(0.0).values.astype(np.float32)
    p_pred = model.predict_proba(X)[:, 1]
    rows = []
    for day, g in oos_df.groupby('day'):
        idx = g.index.values
        p_day = p_pred[idx]
        for K in K_list:
            y_day = g[f'pivot_within_{K}m'].values
            row = {'day': day, 'K': K, 'n_bars': len(y_day),
                   'pos_rate': float(y_day.mean())}
            if len(np.unique(y_day)) == 2:
                row['auc'] = float(roc_auc_score(y_day, p_day))
            else:
                row['auc'] = float('nan')
            for thr in thresholds:
                pred = (p_day >= thr).astype(int)
                tp = int(((pred == 1) & (y_day == 1)).sum())
                fp = int(((pred == 1) & (y_day == 0)).sum())
                fn = int(((pred == 0) & (y_day == 1)).sum())
                row[f'prec_{thr:.2f}'] = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
                row[f'rec_{thr:.2f}']  = tp / max(tp + fn, 1) if (tp + fn) > 0 else float('nan')
                row[f'cov_{thr:.2f}']  = float((pred == 1).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def per_day_b2_metrics(model, oos_piv, K_list, v2_cols, thresholds=(0.30, 0.50, 0.70, 0.85)):
    """For each day in OOS, compute per-K AUC + precision on pivot-level data."""
    X = oos_piv[v2_cols].values.astype(np.float32)
    p_pred = model.predict_proba(X)[:, 1]
    rows = []
    for day, g in oos_piv.groupby('day'):
        idx = g.index.values
        p_day = p_pred[idx]
        for K in K_list:
            y_day = g[f'is_fakeout_{K}m'].values
            row = {'day': day, 'K': K, 'n_pivots': len(y_day),
                   'pos_rate': float(y_day.mean())}
            if len(np.unique(y_day)) == 2:
                row['auc'] = float(roc_auc_score(y_day, p_day))
            else:
                row['auc'] = float('nan')
            for thr in thresholds:
                pred = (p_day >= thr).astype(int)
                tp = int(((pred == 1) & (y_day == 1)).sum())
                fp = int(((pred == 1) & (y_day == 0)).sum())
                fn = int(((pred == 0) & (y_day == 1)).sum())
                row[f'prec_{thr:.2f}'] = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
                row[f'rec_{thr:.2f}']  = tp / max(tp + fn, 1) if (tp + fn) > 0 else float('nan')
                row[f'cov_{thr:.2f}']  = float((pred == 1).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_day(df_per_day, metric_cols):
    """Compute mean + 95% CI per K for given metric columns."""
    summary = []
    for K, g in df_per_day.groupby('K'):
        row = {'K': K, 'n_days': len(g)}
        for m in metric_cols:
            mean, lo, hi = bootstrap_ci_mean(g[m].values)
            row[f'{m}_mean'] = mean
            row[f'{m}_ci_lo'] = lo
            row[f'{m}_ci_hi'] = hi
        summary.append(row)
    return pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1-pkl', default='reports/findings/regret_oracle/b1_pivot_imminent.pkl')
    ap.add_argument('--b2-pkl', default='reports/findings/regret_oracle/b2_fakeout.pkl')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle/')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)

    # --- B1 ---
    print('Loading B1 pickle:', args.b1_pkl)
    with open(args.b1_pkl, 'rb') as f:
        b1_models = pickle.load(f)
    print('  models for K =', sorted(b1_models.keys()))

    print('Building B1 labels on OOS...')
    oos_b1 = build_b1_labels(oos_df.copy(), B1_K)

    print('Computing B1 per-day metrics...')
    # All K share the same model? No — separate per K. Run each.
    all_b1 = []
    for K in B1_K:
        model = b1_models[K]['model']
        # build_labels added pivot_within_Km cols for all K; we'll filter to this K below
        per_day = per_day_b1_metrics(model, oos_b1, [K])
        all_b1.append(per_day)
    b1_per_day = pd.concat(all_b1, ignore_index=True)
    b1_per_day.to_csv(out_dir / 'b1_per_day.csv', index=False)

    metric_cols = ['auc', 'prec_0.70', 'prec_0.85', 'rec_0.70', 'cov_0.70', 'cov_0.85']
    b1_summary = summarize_per_day(b1_per_day, metric_cols)
    b1_summary.to_csv(out_dir / 'b1_per_day_summary.csv', index=False)

    # --- B2 ---
    print('Loading B2 pickle:', args.b2_pkl)
    with open(args.b2_pkl, 'rb') as f:
        b2_models = pickle.load(f)
    print('  models for K =', sorted(b2_models.keys()))

    print('Building B2 pivot events on OOS...')
    oos_piv, v2_cols = build_pivot_dataset(oos_df, B2_K)
    print(f'  {len(oos_piv)} pivot events')

    print('Computing B2 per-day metrics...')
    all_b2 = []
    for K in B2_K:
        model = b2_models[K]['model']
        per_day = per_day_b2_metrics(model, oos_piv, [K], v2_cols)
        all_b2.append(per_day)
    b2_per_day = pd.concat(all_b2, ignore_index=True)
    b2_per_day.to_csv(out_dir / 'b2_per_day.csv', index=False)
    b2_summary = summarize_per_day(b2_per_day, metric_cols)
    b2_summary.to_csv(out_dir / 'b2_per_day_summary.csv', index=False)

    # --- Report ---
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B1 + B2 PER-DAY BOOTSTRAP CI (95% on mean, 4000 resamples)')
    out('=' * 78)

    out('\n[B1 PIVOT-IMMINENT] -- per-day means with 95% CI:')
    out(f'  {"K":>3}  {"days":>4}  {"auc":>22}  {"prec@0.70":>22}  {"prec@0.85":>22}')
    for _, r in b1_summary.iterrows():
        out(f'  {int(r["K"]):>3}  {int(r["n_days"]):>4}  '
            f'{r["auc_mean"]:.3f} [{r["auc_ci_lo"]:.3f},{r["auc_ci_hi"]:.3f}]  '
            f'{r["prec_0.70_mean"]*100:5.1f}%[{r["prec_0.70_ci_lo"]*100:5.1f},{r["prec_0.70_ci_hi"]*100:5.1f}]  '
            f'{r["prec_0.85_mean"]*100:5.1f}%[{r["prec_0.85_ci_lo"]*100:5.1f},{r["prec_0.85_ci_hi"]*100:5.1f}]')

    out('\n[B2 FAKEOUT] -- per-day means with 95% CI (per-pivot eval inside days):')
    out(f'  {"K":>3}  {"days":>4}  {"auc":>22}  {"prec@0.70":>22}  {"prec@0.85":>22}')
    for _, r in b2_summary.iterrows():
        out(f'  {int(r["K"]):>3}  {int(r["n_days"]):>4}  '
            f'{r["auc_mean"]:.3f} [{r["auc_ci_lo"]:.3f},{r["auc_ci_hi"]:.3f}]  '
            f'{r["prec_0.70_mean"]*100:5.1f}%[{r["prec_0.70_ci_lo"]*100:5.1f},{r["prec_0.70_ci_hi"]*100:5.1f}]  '
            f'{r["prec_0.85_mean"]*100:5.1f}%[{r["prec_0.85_ci_lo"]*100:5.1f},{r["prec_0.85_ci_hi"]*100:5.1f}]')

    # Significance check: AUC CI strictly above 0.50 = better than random
    out('\nSignificance vs random (AUC CI > 0.50):')
    for name, df in [('B1', b1_summary), ('B2', b2_summary)]:
        for _, r in df.iterrows():
            sig = 'YES' if r['auc_ci_lo'] > 0.50 else 'no'
            out(f'  {name} K={int(r["K"]):>2}: AUC CI [{r["auc_ci_lo"]:.3f}, {r["auc_ci_hi"]:.3f}] -> {sig}')

    # Write
    (out_dir / 'b1_b2_per_day_ci.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {out_dir / "b1_per_day.csv"}')
    print(f'Wrote: {out_dir / "b2_per_day.csv"}')
    print(f'Wrote: {out_dir / "b1_b2_per_day_ci.txt"}')


if __name__ == '__main__':
    main()
