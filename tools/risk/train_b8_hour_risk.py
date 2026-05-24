"""B8 — Intraday hour-level risk classifier.

Predicts forward-60-min total leg P&L at each 1m bar. Use the prediction
as a multiplier on per-leg sizing:
   if hour P&L predicted >> 0  -> full sizing (good window ahead)
   if hour P&L predicted near 0 -> reduced sizing (uncertain)
   if hour P&L predicted < 0   -> skip / minimal sizing (bad window)

This complements B7 (per-leg amplitude prediction) by adding a
time-window dimension. B7 says "this leg looks big"; B8 says "the
next hour looks favorable/risky."

Target: forward_60min_pnl_usd =
  sum over legs with pivot timestamp in [t, t+3600] of
       (leg_amp_usd - 2*r_price*$2 - $6 friction)
  [this approximates the R-trigger-entry P&L per leg]

Features: V2 184 features at 1m close bar t
  + hour_of_day (sin/cos)
  + day_of_week (one-hot, optional)

Training: full IS (2025, 277 days)
Eval: NT8 OOS (32 days)

NOTE: this uses oracle leg P&L (leg_amp - 2R - friction) as label, NOT
R-trigger-detected leg P&L. Labels are slightly optimistic. For full
honesty, redo with R-trigger-detected legs on IS days.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


DOLLAR_PER_POINT = 2.0
FRICTION_PER_LEG = 6.00   # commission + slippage
FORWARD_WINDOW_S = 3600   # 60 minutes


def build_dataset_for_split(truth_df, legs_df, v2_cols, label):
    """For each 1m bar in truth_df, compute forward-60min total leg P&L."""
    rows = []
    for day in tqdm(sorted(truth_df['day'].unique()), desc=f'{label} days'):
        day_truth = truth_df[truth_df['day'] == day].sort_values('timestamp').reset_index(drop=True)
        day_legs = legs_df[legs_df['day'] == day].sort_values('entry_ts').reset_index(drop=True)
        if len(day_legs) == 0:
            continue
        # Per-leg realistic P&L approximation
        # leg_amp - 2R - friction  (uses leg's actual amplitude, simulates R-trigger entry/exit)
        leg_pnl_usd = (day_legs['leg_amp_usd'].values
                         - 2 * day_legs['r_price'].values * DOLLAR_PER_POINT
                         - FRICTION_PER_LEG)
        leg_entry_ts = day_legs['entry_ts'].values.astype(np.int64)

        # For each 1m bar, forward window sum
        bar_ts = day_truth['timestamp'].values.astype(np.int64)
        # Use searchsorted for fast window aggregation
        fwd_pnl = np.zeros(len(bar_ts), dtype=np.float64)
        for i, t in enumerate(bar_ts):
            mask = (leg_entry_ts >= t) & (leg_entry_ts < t + FORWARD_WINDOW_S)
            fwd_pnl[i] = float(leg_pnl_usd[mask].sum())

        # Drop bars near end of day (forward window incomplete)
        # Keep only bars with at least 60 min of forward day data
        end_ts = bar_ts.max()
        keep = bar_ts + FORWARD_WINDOW_S <= end_ts
        sub = day_truth[keep].copy()
        sub['forward_60min_pnl_usd'] = fwd_pnl[keep]
        sub['n_legs_in_window'] = [int(((leg_entry_ts >= t) &
                                          (leg_entry_ts < t + FORWARD_WINDOW_S)).sum())
                                     for t in bar_ts[keep]]

        # Time-of-day features
        dt = pd.to_datetime(sub['timestamp'], unit='s')
        sub['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        sub['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
        sub['minute_of_session'] = (sub['timestamp'] - sub['timestamp'].min()).astype('int64') / 60

        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--is-legs',
                    default='reports/findings/regret_oracle/b7_leg_sizer_IS.parquet')
    ap.add_argument('--oos-legs',
                    default='reports/findings/regret_oracle/b7_leg_sizer_OOS.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b8_hour_risk.pkl')
    ap.add_argument('--out-oos-cache',
                    default='reports/findings/regret_oracle/b8_hour_risk_OOS.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b8_hour_risk.txt')
    ap.add_argument('--max-iter', type=int, default=300)
    args = ap.parse_args()

    print('Loading truth + legs...')
    is_truth = pd.read_parquet(args.is_truth)
    oos_truth = pd.read_parquet(args.oos_truth)
    is_legs = pd.read_parquet(args.is_legs)
    oos_legs = pd.read_parquet(args.oos_legs)
    print(f'  IS truth bars: {len(is_truth):,}   legs: {len(is_legs):,}')
    print(f'  OOS truth bars: {len(oos_truth):,}   legs: {len(oos_legs):,}')

    v2_cols = [c for c in is_truth.columns if c.startswith(('L1_', 'L2_', 'L3_'))]

    print('Building IS dataset...')
    is_df = build_dataset_for_split(is_truth, is_legs, v2_cols, 'IS')
    print(f'  IS rows: {len(is_df):,}')
    print(f'  IS forward_60min_pnl: median ${is_df["forward_60min_pnl_usd"].median():+.0f}   '
          f'mean ${is_df["forward_60min_pnl_usd"].mean():+.0f}   '
          f'p25 ${is_df["forward_60min_pnl_usd"].quantile(0.25):+.0f}   '
          f'p75 ${is_df["forward_60min_pnl_usd"].quantile(0.75):+.0f}')

    print('Building OOS dataset...')
    oos_df = build_dataset_for_split(oos_truth, oos_legs, v2_cols, 'OOS')
    print(f'  OOS rows: {len(oos_df):,}')
    print(f'  OOS forward_60min_pnl: median ${oos_df["forward_60min_pnl_usd"].median():+.0f}   '
          f'mean ${oos_df["forward_60min_pnl_usd"].mean():+.0f}')

    extra_features = ['hour_sin', 'hour_cos', 'minute_of_session']
    feat_cols = v2_cols + extra_features

    X_is = is_df[feat_cols].fillna(0.0).values.astype(np.float32)
    y_is = is_df['forward_60min_pnl_usd'].values.astype(np.float32)
    X_oos = oos_df[feat_cols].fillna(0.0).values.astype(np.float32)
    y_oos = oos_df['forward_60min_pnl_usd'].values.astype(np.float32)

    print('Training HistGradientBoostingRegressor (MAE loss)...')
    model = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=args.max_iter, learning_rate=0.05,
        max_depth=6, min_samples_leaf=100,
        l2_regularization=0.5,
        random_state=42,
    )
    model.fit(X_is, y_is)

    print('Predicting...')
    p_is = model.predict(X_is)
    p_oos = model.predict(X_oos)

    mae_is = mean_absolute_error(y_is, p_is)
    mae_oos = mean_absolute_error(y_oos, p_oos)
    mae_baseline = mean_absolute_error(y_oos, np.full_like(y_oos, np.median(y_is)))

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B8 INTRADAY HOUR RISK CLASSIFIER (forward 60min total P&L)')
    out('=' * 78)
    out(f'IS rows:  {len(is_df):,}   OOS rows: {len(oos_df):,}   features: {len(feat_cols)}')
    out(f'IS  MAE: ${mae_is:.2f}')
    out(f'OOS MAE: ${mae_oos:.2f}')
    out(f'Baseline (always-median): ${mae_baseline:.2f}')
    if mae_baseline > 0:
        out(f'OOS lift: {(1 - mae_oos/mae_baseline)*100:.1f}% MAE reduction')
    rho = float(np.corrcoef(p_oos, y_oos)[0, 1])
    out(f'OOS Pearson(pred, truth): {rho:.4f}')

    # Calibration: pred bucket -> actual mean
    out('')
    out('--- CALIBRATION (predicted hour P&L bucket -> actual mean) ---')
    bins = [-1e9, -500, -200, 0, 200, 500, 1000, 2000, 1e9]
    labels = ['<-500', '-500--200', '-200-0', '0-200', '200-500', '500-1000',
              '1000-2000', '>2000']
    oos_df['pred_bucket'] = pd.cut(p_oos, bins=bins, labels=labels)
    out(f'  {"bucket":<12}  {"n":>6}  {"actual_mean":>12}  {"actual_median":>14}  '
        f'{"% pos":>6}')
    for b in labels:
        sub = oos_df[oos_df['pred_bucket'] == b]
        if len(sub) == 0: continue
        actual = sub['forward_60min_pnl_usd'].values
        out(f'  {b:<12}  {len(sub):>6}  ${actual.mean():>+10.2f}   '
            f'${np.median(actual):>+11.2f}    {(actual > 0).mean()*100:>5.1f}%')

    # Save model
    with open(args.out_pkl, 'wb') as f:
        pickle.dump({
            'model': model, 'feat_cols': feat_cols,
            'is_median': float(np.median(y_is)),
            'is_mean': float(np.mean(y_is)),
        }, f)

    # Save OOS cache
    cache = oos_df[['timestamp', 'day']].copy()
    cache['pred_hour_pnl_usd'] = p_oos
    cache['actual_hour_pnl_usd'] = y_oos
    cache.to_parquet(args.out_oos_cache, index=False)

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_oos_cache}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
