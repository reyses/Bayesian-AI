"""B9 -- During-trade remaining-amplitude regressor.

Predicts remaining_pnl_usd = exit_pnl_usd - pnl_usd_so_far at K bars
after entry. Operational use: continuous sizing signal -- if predicted
remaining is strongly negative, reduce position; if strongly positive,
hold/scale.

Different from B9 (binary cut): regression output allows GRADIENT-based
actions:
  - pred < -$50  ->  cut completely (= B9-style cut)
  - pred in [-$50, -$10]  ->  reduce 50%
  - pred in [-$10, +$10]  ->  hold (model uncertain)
  - pred in [+$10, +$50]  ->  hold full
  - pred > +$50  ->  scale up 1.5x (pyramid)

Methodology: walk-forward 4 folds, per-K models, Pearson + MAE on val,
then continuous sizing rule evaluation (honest accounting: 'reduce X%'
realizes pnl_so_far * (1 - X) immediately, holds (X) at original size
to exit).

Why this might survive where B9 failed:
  - Avoids binary cut/hold trade-off
  - Variable action calibrated to prediction magnitude
  - On HIGH-confidence negative predictions, action = cut (replicates B9
    in that regime)
  - On low-magnitude predictions, action = hold (doesn't pay Type 1 cost)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr
from tqdm import tqdm


K_HORIZONS = [5, 10, 30, 60, 120]
N_FOLDS = 4
N_BOOTSTRAP = 4000


def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {'leg_id', 'day', 'entry_ts', 'leg_dir', 'K', 'bar_ts',
            'r_price', 'exit_ts', 'exit_pnl_pts', 'exit_pnl_usd',
            'bars_since_entry'}
    return [c for c in df.columns
            if c not in skip and df[c].dtype != object]


def walk_forward_folds(days, n_folds):
    n_days = len(days)
    val_size = n_days // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = val_size * (i + 1)
        val_end = val_size * (i + 2)
        if val_end > n_days:
            val_end = n_days
        folds.append((set(days[:train_end]), set(days[train_end:val_end])))
    return folds


def size_from_pred(pred_remaining):
    """Continuous sizing rule based on predicted remaining $."""
    if pred_remaining > 50:   return 1.5     # pyramid
    if pred_remaining > 10:   return 1.0     # hold full
    if pred_remaining > -10:  return 1.0     # uncertain, hold
    if pred_remaining > -50:  return 0.5     # reduce 50%
    return 0.0                                # cut


def evaluate_sizing(val_df, y_pred):
    """Honest delta accounting with variable sizing.

    Action at K: size_factor in {0, 0.5, 1.0, 1.5} based on pred.
    Realized P&L per leg =
      pnl_usd_so_far + size_factor * (exit_pnl_usd - pnl_usd_so_far)

    Baseline (no action) realizes exit_pnl_usd (= size_factor 1.0 case).
    Delta = realized - exit_pnl_usd
          = (size_factor - 1) * (exit_pnl_usd - pnl_usd_so_far)
    """
    sizes = np.array([size_from_pred(p) for p in y_pred])
    pnl_at_cut = val_df['pnl_usd_so_far'].values
    pnl_at_exit = val_df['exit_pnl_usd'].values
    realized = pnl_at_cut + sizes * (pnl_at_exit - pnl_at_cut)
    delta = realized - pnl_at_exit
    val_df = val_df.copy()
    val_df['delta'] = delta
    val_df['size_factor'] = sizes
    per_day = val_df.groupby('day')['delta'].sum().values
    size_dist = pd.Series(sizes).value_counts().sort_index().to_dict()
    return per_day, size_dist


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    print(f'Loading: {args.traj}')
    df = pd.read_parquet(args.traj)
    feat_cols = get_feature_cols(df)
    days = sorted(df['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)

    # Target: remaining $ from K to exit
    df['remaining_pnl_usd'] = df['exit_pnl_usd'] - df['pnl_usd_so_far']
    print(f'  rows {len(df):,}   features {len(feat_cols)}')
    print(f'  remaining_pnl mean ${df["remaining_pnl_usd"].mean():+.2f}  '
          f'std ${df["remaining_pnl_usd"].std():.2f}  '
          f'min ${df["remaining_pnl_usd"].min():+.2f}  '
          f'max ${df["remaining_pnl_usd"].max():+.2f}')

    wf_rows = []
    for fold_idx, (tr_days, va_days) in enumerate(folds):
        print(f'\n=== Fold {fold_idx + 1}/{N_FOLDS} ===')
        for K in K_HORIZONS:
            sub = df[df['K'] == K]
            train = sub[sub['day'].isin(tr_days)]
            val = sub[sub['day'].isin(va_days)]
            if len(train) < 100 or len(val) < 20:
                continue
            X_tr = train[feat_cols].fillna(0.0).values
            y_tr = train['remaining_pnl_usd'].values
            X_va = val[feat_cols].fillna(0.0).values
            y_va = val['remaining_pnl_usd'].values

            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42, l2_regularization=1.0,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            try:
                pearson, _ = pearsonr(y_va, y_pred)
            except Exception:
                pearson = float('nan')
            mae = float(np.mean(np.abs(y_va - y_pred)))

            per_day_delta, size_dist = evaluate_sizing(val, y_pred)
            ci_lo, ci_hi = bootstrap_ci(per_day_delta)
            wf_rows.append({
                'fold': fold_idx + 1,
                'K': K,
                'pearson': pearson,
                'mae': mae,
                'val_legs': len(val),
                'val_days': val['day'].nunique(),
                'mean_delta_per_day': float(per_day_delta.mean()),
                'median_delta_per_day': float(np.median(per_day_delta)),
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'significant': ci_lo > 0,
                'n_cut': int(size_dist.get(0.0, 0)),
                'n_half': int(size_dist.get(0.5, 0)),
                'n_full': int(size_dist.get(1.0, 0)),
                'n_pyramid': int(size_dist.get(1.5, 0)),
            })

    wf_df = pd.DataFrame(wf_rows)
    wf_csv = out_dir / 'b9_remaining_amplitude_walk_forward.csv'
    wf_df.to_csv(wf_csv, index=False)
    print(f'\nWrote: {wf_csv}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('B9 REMAINING-AMPLITUDE REGRESSOR  walk-forward summary')
    out('=' * 100)
    out('Target: remaining_pnl_usd = exit_pnl_usd - pnl_usd_so_far at K bars')
    out('Sizing rule:')
    out('  pred > +$50    : size 1.5  (pyramid)')
    out('  pred > +$10    : size 1.0  (hold full)')
    out('  pred > -$10    : size 1.0  (uncertain, hold)')
    out('  pred > -$50    : size 0.5  (reduce 50%)')
    out('  else           : size 0.0  (cut)')
    out(f'Folds: {N_FOLDS} expanding-window, day-based')
    out('')

    for K in K_HORIZONS:
        sub = wf_df[wf_df['K'] == K]
        if len(sub) == 0:
            continue
        out(f'K={K:>3}  Pearson {sub["pearson"].mean():.3f}  MAE ${sub["mae"].mean():.2f}   '
            f'mean delta ${sub["mean_delta_per_day"].mean():+.0f}/day   '
            f'avg CI [${sub["ci_lo"].mean():+.0f}, ${sub["ci_hi"].mean():+.0f}]   '
            f'sig {int(sub["significant"].sum())}/{len(sub)}')
        out(f'        size dist (avg): cut {sub["n_cut"].mean():.0f}  '
            f'half {sub["n_half"].mean():.0f}  '
            f'full {sub["n_full"].mean():.0f}  '
            f'pyramid {sub["n_pyramid"].mean():.0f}')

    out('')
    out('--- Qualified: sig >= 2/4 folds AND mean delta > 0 ---')
    sdf = wf_df.groupby('K').agg(
        mean_pearson=('pearson', 'mean'),
        mean_delta=('mean_delta_per_day', 'mean'),
        n_sig=('significant', 'sum'),
        n_folds=('significant', 'count'),
    ).reset_index()
    sdf['sig_frac'] = sdf['n_sig'] / sdf['n_folds']
    qualified = sdf[(sdf['sig_frac'] >= 0.5) & (sdf['mean_delta'] > 0)].sort_values(
        'mean_delta', ascending=False)
    if len(qualified) == 0:
        out('  NONE qualified. B10 also fails walk-forward with significance.')
        out('  Per-fold deltas may be positive but not reliably across folds.')
    else:
        out(f'  {"K":>4}  {"Pearson":>8}  {"delta_$/day":>11}  {"sig":>6}')
        for _, r in qualified.iterrows():
            out(f'  {int(r["K"]):>4}  {r["mean_pearson"]:>8.3f}  '
                f'${r["mean_delta"]:>+9.0f}    {int(r["n_sig"])}/{int(r["n_folds"])}')

    # Train production models on FULL IS regardless of walk-forward outcome --
    # for OOS test (which is the honest measurement, not IS walk-forward)
    out('')
    out('--- Training production models on FULL IS (for OOS test) ---')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        X = sub[feat_cols].fillna(0.0).values
        y = sub['remaining_pnl_usd'].values
        if len(X) < 100:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42, l2_regularization=1.0,
        )
        model.fit(X, y)
        prod_path = out_dir / f'b9_remaining_amplitude_K{K}.pkl'
        with open(prod_path, 'wb') as f:
            pickle.dump({'model': model, 'feat_cols': feat_cols,
                          'K': K, 'target': 'remaining_pnl_usd',
                          'n_train': len(X)}, f)
        out(f'  K={K:>3}  trained on {len(X):,} rows  -> {prod_path.name}')

    summary_path = out_dir / 'b9_remaining_amplitude_summary.txt'
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {summary_path}')


if __name__ == '__main__':
    main()
