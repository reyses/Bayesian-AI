"""B7 — GBM leg-amplitude regressor for trade sizing.

User insight (2026-05-17): "direction accuracy is the first step everything
else is position management" + "GBM for trade sizing."

This trains a GBM to predict, at entry time (pivot bar), the FORWARD LEG
AMPLITUDE in R units. The prediction is then used as a sizing multiplier:
larger predicted amplitude -> bigger position.

Target: leg_amplitude_R = leg_amplitude_pts / r_price
  Values near 1.0 = leg barely makes the R-trigger threshold
  Values near 2.0 = "double" leg, hit 2R favorable
  Values near 3.0 = monster leg
  Median target across all OOS legs ~ 2.3 R

Features: 184 V2 features at entry bar (L1/L2/L3 × 5s-1D)

Train: full IS (2025, 277 days, ~14k legs)
Eval:  NT8 OOS (32 days, 1827 legs)

Outputs:
  - model pickle
  - per-leg OOS predictions cache (for sizing simulator)
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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')
ATLAS_5S_DIR = Path('DATA/ATLAS/5s')
ATLAS_1M_DIR = Path('DATA/ATLAS/1m')


def pivot_centroid_events(day_df: pd.DataFrame):
    piv = day_df[day_df['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return []
    ts = piv['timestamp'].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    pp_ = piv['pivot_price'].values
    bar_idx = piv.index.values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for grp in groups:
        ts_c = int(np.median(ts[grp]))
        vals, counts = np.unique(pd_[grp], return_counts=True)
        d = str(vals[np.argmax(counts)])
        p = float(np.mean(pp_[grp]))
        # Pick the centroid bar index in day_df
        mid_grp = grp[len(grp) // 2]
        bi = int(bar_idx[mid_grp])
        out.append((ts_c, d, p, bi))
    return out


def build_leg_dataset(truth_df, atlas_5s_dir, atlas_1m_dir, v2_cols, label='IS'):
    """Build per-leg dataset: entry features + forward amplitude in R units."""
    rows = []
    days_all = sorted(truth_df['day'].unique())
    for day in tqdm(days_all, desc=f'{label} days'):
        bars5s_path = atlas_5s_dir / f'{day}.parquet'
        bars1m_path = atlas_1m_dir / f'{day}.parquet'
        if not bars5s_path.exists() or not bars1m_path.exists():
            continue
        bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE

        truth_day = truth_df[truth_df['day'] == day].sort_values('timestamp').reset_index(drop=True)
        events = pivot_centroid_events(truth_day)
        if len(events) < 2:
            continue
        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        for k in range(len(events) - 1):
            entry_ts, leg_dir, entry_price, entry_bar_idx = events[k]
            next_ts, _, _, _ = events[k + 1]
            mask = (ts5s >= entry_ts) & (ts5s <= next_ts)
            leg_closes = closes5s[mask]
            if len(leg_closes) == 0:
                continue
            if leg_dir == 'LONG':
                leg_extreme = float(leg_closes.max())
            else:
                leg_extreme = float(leg_closes.min())
            leg_amp_pts = abs(leg_extreme - entry_price)
            leg_amp_R = leg_amp_pts / r_price
            pnl_at_R_pts = leg_amp_pts - r_price
            pnl_at_R_usd = pnl_at_R_pts * DOLLAR_PER_POINT

            # Entry V2 features (bar at entry_bar_idx of truth_day)
            entry_bar = truth_day.iloc[entry_bar_idx]
            feat_dict = {c: float(entry_bar[c]) if not pd.isna(entry_bar[c]) else 0.0
                          for c in v2_cols}
            feat_dict.update({
                'day': day,
                'entry_ts': entry_ts,
                'leg_dir': leg_dir,
                'entry_price': entry_price,
                'leg_amp_pts': leg_amp_pts,
                'leg_amp_usd': leg_amp_pts * DOLLAR_PER_POINT,
                'leg_amp_R': leg_amp_R,
                'pnl_at_R_usd': pnl_at_R_usd,
                'r_price': r_price,
                'atr_pts': atr_pts,
                'leg_duration_s': int(next_ts - entry_ts),
            })
            rows.append(feat_dict)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b7_leg_sizer.pkl')
    ap.add_argument('--out-cache',
                    default='reports/findings/regret_oracle/b7_leg_sizer_OOS.parquet')
    ap.add_argument('--out-is-cache',
                    default='reports/findings/regret_oracle/b7_leg_sizer_IS.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b7_leg_sizer.txt')
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading truth datasets...')
    is_df = pd.read_parquet(args.is_truth)
    oos_df = pd.read_parquet(args.oos_truth)
    print(f'  IS bars  {len(is_df):,} / {is_df["day"].nunique()} days')
    print(f'  OOS bars {len(oos_df):,} / {oos_df["day"].nunique()} days')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    print('\nBuilding IS per-leg dataset...')
    is_legs = build_leg_dataset(is_df, ATLAS_5S_DIR, ATLAS_1M_DIR, v2_cols, label='IS')
    print(f'  IS legs:  {len(is_legs):,}')
    print(f'  IS amp_R: median {is_legs["leg_amp_R"].median():.2f}   '
          f'mean {is_legs["leg_amp_R"].mean():.2f}   '
          f'p25 {is_legs["leg_amp_R"].quantile(0.25):.2f}   '
          f'p75 {is_legs["leg_amp_R"].quantile(0.75):.2f}')

    print('\nBuilding OOS per-leg dataset...')
    oos_legs = build_leg_dataset(oos_df, NT8_5S_DIR, NT8_1M_DIR, v2_cols, label='OOS')
    print(f'  OOS legs: {len(oos_legs):,}')
    print(f'  OOS amp_R: median {oos_legs["leg_amp_R"].median():.2f}   '
          f'mean {oos_legs["leg_amp_R"].mean():.2f}')

    # Save IS leg dataset for completeness
    Path(args.out_is_cache).parent.mkdir(parents=True, exist_ok=True)
    is_legs.to_parquet(args.out_is_cache, index=False)

    # Train regressor on IS, eval on OOS
    X_is  = is_legs[v2_cols].values.astype(np.float32)
    y_is  = is_legs['leg_amp_R'].values.astype(np.float32)
    X_oos = oos_legs[v2_cols].values.astype(np.float32)
    y_oos = oos_legs['leg_amp_R'].values.astype(np.float32)

    print('\nTraining HistGradientBoostingRegressor (absolute_error loss)...')
    model = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=args.max_iter, learning_rate=0.05,
        max_depth=6, min_samples_leaf=30,
        l2_regularization=0.5,
        random_state=args.seed,
    )
    model.fit(X_is, y_is)

    print('Predicting...')
    p_is  = model.predict(X_is)
    p_oos = model.predict(X_oos)

    mae_is  = mean_absolute_error(y_is, p_is)
    mae_oos = mean_absolute_error(y_oos, p_oos)
    median_baseline = float(np.median(y_is))
    mae_baseline = mean_absolute_error(y_oos, np.full_like(y_oos, median_baseline))

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B7 LEG-AMPLITUDE REGRESSOR — predicts leg_amp / R at entry')
    out('=' * 78)
    out(f'IS legs:  {len(is_legs):,}   OOS legs: {len(oos_legs):,}   features: {len(v2_cols)}')
    out('')
    out(f'IS  MAE (amp_R units): {mae_is:.3f}')
    out(f'OOS MAE (amp_R units): {mae_oos:.3f}')
    out(f'Baseline (always-median predictor): {mae_baseline:.3f}')
    if mae_baseline > 0:
        out(f'OOS lift over baseline: {(1 - mae_oos/mae_baseline)*100:.1f}% MAE reduction')

    # Correlation
    rho_oos = float(np.corrcoef(p_oos, y_oos)[0, 1])
    out(f'OOS Pearson(pred, truth): {rho_oos:.4f}')

    # Bucket OOS predictions and see if mean actual amp_R follows
    out('')
    out('--- CALIBRATION: predicted amp_R bucket -> actual amp_R distribution ---')
    bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 100.0]
    labels = ['<1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '>=4.0']
    oos_legs['pred_amp_R'] = p_oos
    oos_legs['pred_bucket'] = pd.cut(p_oos, bins=bins, labels=labels)
    out(f'  {"pred bucket":<12}  {"n":>5}  {"%legs":>6}  '
        f'{"actual_mean":>12}  {"actual_median":>14}  {"actual_p25":>11}  {"actual_p75":>11}')
    for b in labels:
        sub = oos_legs[oos_legs['pred_bucket'] == b]
        if len(sub) == 0:
            continue
        actual = sub['leg_amp_R'].values
        out(f'  {b:<12}  {len(sub):>5}  {len(sub)/len(oos_legs)*100:>5.1f}%  '
            f'{actual.mean():>11.2f}   {np.median(actual):>13.2f}   '
            f'{np.percentile(actual,25):>10.2f}   {np.percentile(actual,75):>10.2f}')

    # Save outputs
    with open(args.out_pkl, 'wb') as f:
        pickle.dump({'model': model, 'v2_cols': v2_cols,
                     'target_mean_R': float(y_is.mean()),
                     'target_median_R': float(np.median(y_is)),
                     'is_mae': float(mae_is), 'oos_mae': float(mae_oos)}, f)
    cache = oos_legs[['day', 'entry_ts', 'leg_dir', 'entry_price',
                       'leg_amp_pts', 'leg_amp_usd', 'leg_amp_R',
                       'pnl_at_R_usd', 'r_price', 'atr_pts',
                       'pred_amp_R']].copy()
    cache.to_parquet(args.out_cache, index=False)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_cache}')
    print(f'Wrote: {args.out_is_cache}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
