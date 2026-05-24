"""Evaluate B9 + C13 combined rule: phase-gated pyramid.

Rule: at each K, use B9 to predict remaining_pnl_usd AND C13 to predict
leg_phase. Apply phase constraint on B9's pyramid action:
  - If C13_phase > 0.7 (late leg): cap size at 1.0 (no pyramid)
  - Otherwise: B9 sizing as-is

Tests whether C13 adds incremental $/day over B9 alone.

Walk-forward with 4 expanding folds. Both B9 and C13 retrained per
fold to avoid lookahead (each fold trains both models on training days
only, applies to val days).

Promotion criterion: combined Delta > B9-alone Delta with CI strictly
positive on the difference, sig >= 2/4 folds.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


K_HORIZONS = [5, 10, 30, 60, 120]
N_FOLDS = 4
N_BOOTSTRAP = 4000
PHASE_LATE_THRESHOLD = 0.7


def get_feature_cols(df):
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


def size_from_b9_pred(pred_remaining):
    if pred_remaining > 50:   return 1.5
    if pred_remaining > 10:   return 1.0
    if pred_remaining > -10:  return 1.0
    if pred_remaining > -50:  return 0.5
    return 0.0


def evaluate_combined(val_df, b9_pred, c13_pred):
    """Combined rule: B9 sizing, but cap at 1.0 if C13 predicts late phase."""
    sizes_b9_alone = np.array([size_from_b9_pred(p) for p in b9_pred])
    sizes_combined = sizes_b9_alone.copy()
    late_mask = (c13_pred > PHASE_LATE_THRESHOLD)
    # In late phase, cap pyramid (1.5) -> 1.0; leave cut/reduce/full unchanged
    sizes_combined[late_mask & (sizes_combined > 1.0)] = 1.0

    pnl_at_K = val_df['pnl_usd_so_far'].values
    pnl_at_exit = val_df['exit_pnl_usd'].values
    realized_b9 = pnl_at_K + sizes_b9_alone * (pnl_at_exit - pnl_at_K)
    realized_combined = pnl_at_K + sizes_combined * (pnl_at_exit - pnl_at_K)
    delta_b9 = realized_b9 - pnl_at_exit
    delta_combined = realized_combined - pnl_at_exit

    val_df = val_df.copy()
    val_df['delta_b9'] = delta_b9
    val_df['delta_combined'] = delta_combined

    per_day_b9 = val_df.groupby('day')['delta_b9'].sum().values
    per_day_combined = val_df.groupby('day')['delta_combined'].sum().values
    per_day_delta = per_day_combined - per_day_b9   # incremental over B9
    return per_day_b9, per_day_combined, per_day_delta, sizes_combined, sizes_b9_alone


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    traj_path = 'reports/findings/regret_oracle/trade_trajectory_IS.parquet'
    print(f'Loading: {traj_path}')
    df = pd.read_parquet(traj_path)
    feat_cols = get_feature_cols(df)
    leg_duration = (df['exit_ts'] - df['entry_ts']).astype(float)
    bar_offset = (df['bar_ts'] - df['entry_ts']).astype(float)
    df['leg_phase'] = bar_offset / leg_duration.clip(lower=1)
    df['leg_phase'] = df['leg_phase'].clip(0, 1)
    df['remaining_pnl_usd'] = df['exit_pnl_usd'] - df['pnl_usd_so_far']

    days = sorted(df['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)

    print(f'Walk-forward: {N_FOLDS} folds, retraining B9 + C13 per fold')

    out_lines = []
    def out(s=''):
        print(s); out_lines.append(s)

    out('=' * 100)
    out('B9 + C13 COMBINED RULE EVALUATION')
    out('=' * 100)
    out(f'Combined rule: B9 sizing; if C13 phase > {PHASE_LATE_THRESHOLD}, cap at 1.0 (no pyramid)')
    out(f'Folds: {N_FOLDS} expanding-window, both models retrained per fold')
    out('')
    out(f'{"K":>4}  {"fold":>4}  {"B9-alone $/day":>14}  {"Combined $/day":>14}  '
        f'{"Incremental":>11}  {"95% CI on incremental":>22}  {"sig":>5}')

    summary = []
    for K in K_HORIZONS:
        for fold_idx, (tr_days, va_days) in enumerate(folds):
            sub = df[df['K'] == K]
            train = sub[sub['day'].isin(tr_days)]
            val = sub[sub['day'].isin(va_days)]
            if len(train) < 100 or len(val) < 20:
                continue

            X_tr = train[feat_cols].fillna(0.0).values
            X_va = val[feat_cols].fillna(0.0).values

            # B9 (remaining-amplitude)
            b9 = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42, l2_regularization=1.0,
            )
            b9.fit(X_tr, train['remaining_pnl_usd'].values)
            b9_pred = b9.predict(X_va)

            # C13 (leg-phase)
            c13 = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42, l2_regularization=1.0,
            )
            c13.fit(X_tr, train['leg_phase'].values)
            c13_pred = c13.predict(X_va)

            per_day_b9, per_day_combined, per_day_delta, sizes_combined, sizes_b9 = \
                evaluate_combined(val, b9_pred, c13_pred)

            ci_lo, ci_hi = bootstrap_ci(per_day_delta)
            sig = ci_lo > 0
            n_capped = int((sizes_combined < sizes_b9).sum())

            out(f'  {K:>3}  {fold_idx+1:>4}  '
                f'${per_day_b9.mean():>+11.0f}     '
                f'${per_day_combined.mean():>+11.0f}     '
                f'${per_day_delta.mean():>+9.0f}    '
                f'[${ci_lo:>+5.0f}, ${ci_hi:>+5.0f}]    '
                f'{str(sig):>5}  (capped {n_capped})')
            summary.append({
                'K': K, 'fold': fold_idx + 1,
                'b9_alone_per_day': float(per_day_b9.mean()),
                'combined_per_day': float(per_day_combined.mean()),
                'incremental': float(per_day_delta.mean()),
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'significant': sig, 'n_capped': n_capped,
            })

    sdf = pd.DataFrame(summary)
    out('')
    out('=== Aggregate by K ===')
    agg = sdf.groupby('K').agg(
        mean_b9=('b9_alone_per_day', 'mean'),
        mean_combined=('combined_per_day', 'mean'),
        mean_incr=('incremental', 'mean'),
        n_sig=('significant', 'sum'),
        n_folds=('significant', 'count'),
    ).reset_index()
    out(f'{"K":>4}  {"B9-alone":>10}  {"Combined":>10}  {"Incremental":>11}  {"sig":>6}')
    for _, r in agg.iterrows():
        out(f'  {int(r["K"]):>3}  ${r["mean_b9"]:>+7.0f}    '
            f'${r["mean_combined"]:>+7.0f}    '
            f'${r["mean_incr"]:>+9.0f}    '
            f'{int(r["n_sig"])}/{int(r["n_folds"])}')

    out('')
    qualified = agg[(agg['n_sig'] >= 2) & (agg['mean_incr'] > 0)]
    if len(qualified) == 0:
        out('--- C13 phase-gated pyramid: NO K horizon shows incremental + sig ---')
        out('C13 stays as failed candidate. The leg-phase signal is learnable but the')
        out('pyramid-gating action does not produce measurable $/day improvement.')
    else:
        out('--- C13 phase-gated pyramid: QUALIFIED K horizons ---')
        out(f'{"K":>4}  {"Incremental":>11}  {"sig":>6}')
        for _, r in qualified.iterrows():
            out(f'  {int(r["K"]):>3}  ${r["mean_incr"]:>+9.0f}    '
                f'{int(r["n_sig"])}/{int(r["n_folds"])}')
        out('')
        out('-> C13 promotion candidate. Run OOS test before final promotion.')

    sdf.to_csv('reports/findings/regret_oracle/b9_c13_combined_walk_forward.csv', index=False)
    Path('reports/findings/regret_oracle/b9_c13_combined_summary.txt').write_text(
        '\n'.join(out_lines), encoding='utf-8')
    print('\nWrote: reports/findings/regret_oracle/b9_c13_combined_summary.txt')


if __name__ == '__main__':
    main()
