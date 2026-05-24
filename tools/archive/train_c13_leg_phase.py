"""C13 candidate -- during-trade leg-phase regressor (B5 retarget).

Target: leg_phase = bar_ts_within_leg / total_leg_duration (continuous [0, 1])
  0 = trade just entered
  1 = trade exits at this bar
  Mid values = where we are in the leg lifecycle

Operational use: NOT a cut signal -- B9 already covers cut/hold via
continuous sizing. C13 is informational/constraint signal:
  - phase < 0.3: leg is YOUNG -> allow B9 to pyramid aggressively
  - phase 0.3-0.7: mid-leg -> use B9 default sizing
  - phase > 0.7: late-leg -> cap B9 pyramid at 1.0x (no aggressive scaling)

The interaction with B9 is a future engineering question. This trainer
just establishes whether the phase signal is learnable.

Validation: Pearson correlation on val set + per-bucket calibration.
Quality bar: Pearson > 0.25 to be considered useful.

If C13 hits Pearson > 0.25 AND we figure out an operational use case
that produces +$/day, promote to B10 or B11.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr


K_HORIZONS = [5, 10, 30, 60, 120]
N_FOLDS = 4


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
    print(f'  rows {len(df):,}   features {len(feat_cols)}')

    # Target: leg phase = (bar_ts - entry_ts) / (exit_ts - entry_ts), in [0, 1]
    leg_duration = (df['exit_ts'] - df['entry_ts']).astype(float)
    bar_offset = (df['bar_ts'] - df['entry_ts']).astype(float)
    df['leg_phase'] = bar_offset / leg_duration.clip(lower=1)
    df['leg_phase'] = df['leg_phase'].clip(0, 1)

    print(f'  leg_phase by K:')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        print(f'    K={K:>3}  median phase {sub["leg_phase"].median():.3f}   '
              f'mean {sub["leg_phase"].mean():.3f}   n={len(sub):,}')

    days = sorted(df['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)

    wf_rows = []
    for fold_idx, (tr_days, va_days) in enumerate(folds):
        for K in K_HORIZONS:
            sub = df[df['K'] == K]
            train = sub[sub['day'].isin(tr_days)]
            val = sub[sub['day'].isin(va_days)]
            if len(train) < 100 or len(val) < 20:
                continue
            X_tr = train[feat_cols].fillna(0.0).values
            y_tr = train['leg_phase'].values
            X_va = val[feat_cols].fillna(0.0).values
            y_va = val['leg_phase'].values

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
            wf_rows.append({
                'fold': fold_idx + 1,
                'K': K,
                'pearson': pearson,
                'mae': mae,
                'val_legs': len(val),
                'val_days': val['day'].nunique(),
            })

    wf_df = pd.DataFrame(wf_rows)
    wf_csv = out_dir / 'c13_leg_phase_walk_forward.csv'
    wf_df.to_csv(wf_csv, index=False)
    print(f'\nWrote: {wf_csv}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('C13 CANDIDATE  --  leg-phase regressor (B5 retarget)')
    out('=' * 100)
    out('Target: leg_phase = (bar_ts - entry_ts) / (exit_ts - entry_ts) in [0, 1]')
    out(f'Folds: {N_FOLDS} expanding-window, day-based')
    out(f'IS days: {len(days)}')
    out('Quality bar: Pearson > 0.25 to be considered useful informational signal')
    out('')

    for K in K_HORIZONS:
        sub = wf_df[wf_df['K'] == K]
        if len(sub) == 0:
            continue
        out(f'K={K:>3}  Pearson {sub["pearson"].mean():.3f} '
            f'(folds: {sub["pearson"].tolist()})   '
            f'MAE {sub["mae"].mean():.3f}   '
            f'val_legs avg {sub["val_legs"].mean():.0f}')

    out('')
    sdf = wf_df.groupby('K').agg(mean_pearson=('pearson', 'mean')).reset_index()
    qualified = sdf[sdf['mean_pearson'] > 0.25]
    if len(qualified) == 0:
        out('--- C13 walk-forward: NO K horizon clears Pearson > 0.25 bar. ---')
        out('Leg-phase signal too weak for operational use. Stays as C13 (failed).')
    else:
        out('--- C13 qualified K horizons (Pearson > 0.25) ---')
        out(f'  {"K":>4}  {"Pearson":>8}')
        for _, r in qualified.iterrows():
            out(f'  {int(r["K"]):>4}  {r["mean_pearson"]:>8.3f}')
        out('')
        out('NOTE: a positive Pearson does not automatically mean +$/day operational')
        out('value. Need to define a sizing/cut rule that uses leg_phase prediction')
        out('and walk-forward validate that rule. Pearson alone is necessary not sufficient.')

    # Train production models for completeness
    out('')
    out('--- Training C13 candidate models on FULL IS ---')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        X = sub[feat_cols].fillna(0.0).values
        y = sub['leg_phase'].values
        if len(X) < 100:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42, l2_regularization=1.0,
        )
        model.fit(X, y)
        prod_path = out_dir / f'c13_leg_phase_K{K}.pkl'
        with open(prod_path, 'wb') as f:
            pickle.dump({'model': model, 'feat_cols': feat_cols,
                          'K': K, 'target': 'leg_phase',
                          'n_train': len(X)}, f)
        out(f'  K={K:>3}  trained on {len(X):,} rows  -> {prod_path.name}')

    summary_path = out_dir / 'c13_leg_phase_summary.txt'
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {summary_path}')


if __name__ == '__main__':
    main()
