"""Build per-trade trajectory dataset on IS only.

For each IS hardened leg, log:
  - leg metadata (day, entry_ts, leg_dir, exit_pnl) -- constant per leg
  - trajectory features at K bars after entry (5s units):
      mae_pts_so_far, mfe_pts_so_far, pnl_pts_so_far,
      bars_since_entry, has_reached_R_against
  - V2 features at the 1m close <= bar_ts (denormalized into row)

K horizons (5s units):
  5   -> +25 seconds  (very early -- baseline check)
  10  -> +50 seconds
  30  -> +2.5 minutes
  60  -> +5 minutes   (typical "is this trade going bad?" decision point)
  120 -> +10 minutes  (late -- diminishing returns for an early-cut)

OUTPUT (long format):
  reports/findings/regret_oracle/trade_trajectory_IS.parquet
  Schema: leg_id, day, entry_ts, leg_dir, K, bar_ts,
          exit_pnl_usd, exit_pnl_pts, exit_ts,
          mae_pts_so_far, mfe_pts_so_far, pnl_pts_so_far,
          bars_since_entry, has_reached_R_against,
          + ~190 V2 feature cols at bar_ts.

Rows: ~17,767 legs x 5 K horizons = ~88,000 (only rows where bar_ts is
before exit_ts and within the trading day -- legs that exit early get
fewer K samples).

IS ONLY. OOS dataset built separately at final backtest time only.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


K_HORIZONS = [5, 10, 30, 60, 120]   # 5s bar units (25s..10min)
DOLLAR_PER_POINT = 2.0


def get_v2_cols(truth: pd.DataFrame) -> list:
    skip = {'timestamp', 'day', 'is_pivot', 'pivot_dir', 'pivot_price',
            'pivot_idx', 'leg_direction', 'leg_amplitude_pts',
            'leg_amplitude_R', 'leg_duration_bars', 'atr_pts', 'target_split'}
    # Also drop any non-numeric column defensively
    return [c for c in truth.columns
            if c not in skip and truth[c].dtype != object]


def process_day(day: str, day_legs: pd.DataFrame, truth: pd.DataFrame,
                bars5s_dir: Path, v2_cols: list) -> list:
    bars5s_path = bars5s_dir / f'{day}.parquet'
    if not bars5s_path.exists():
        return []

    bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
    ts5s = bars5s['timestamp'].values.astype(np.int64)
    close5s = bars5s['close'].values.astype(np.float64)
    high5s = bars5s['high'].values.astype(np.float64)
    low5s = bars5s['low'].values.astype(np.float64)
    last_ts5s = int(ts5s[-1]) if len(ts5s) > 0 else 0

    truth_day = truth[truth['day'] == day].sort_values('timestamp').reset_index(drop=True)
    truth_ts = truth_day['timestamp'].values.astype(np.int64)
    if len(truth_ts) == 0:
        return []
    v2_arr = truth_day[v2_cols].astype(np.float32).fillna(0.0).values

    out_rows = []
    for _, leg in day_legs.iterrows():
        entry_ts = int(leg['entry_ts'])
        exit_ts = int(leg['exit_ts'])
        entry_price = float(leg['entry_price'])
        leg_dir = str(leg['leg_dir'])
        leg_sign = 1.0 if leg_dir == 'LONG' else -1.0
        r_price = float(leg['r_price'])
        exit_pnl_pts = float(leg['pnl_pts'])
        exit_pnl_usd = float(leg['pnl_usd'])
        leg_id = int(leg['leg_id'])

        # Entry index in 5s bars
        entry_idx = int(np.searchsorted(ts5s, entry_ts, side='left'))
        if entry_idx >= len(ts5s) or ts5s[entry_idx] != entry_ts:
            # Fall back to nearest
            entry_idx = int(np.searchsorted(ts5s, entry_ts, side='right') - 1)
            if entry_idx < 0:
                continue

        # Cumulative MAE / MFE walking forward
        for K in K_HORIZONS:
            bar_ts = entry_ts + K * 5
            # Skip if bar past exit (leg already closed)
            if bar_ts > exit_ts:
                continue
            # Skip if bar past end of available 5s data
            if bar_ts > last_ts5s:
                continue

            end_idx = int(np.searchsorted(ts5s, bar_ts, side='right') - 1)
            if end_idx <= entry_idx:
                continue

            # Slice from entry to end_idx inclusive
            slc_high = high5s[entry_idx:end_idx + 1]
            slc_low = low5s[entry_idx:end_idx + 1]
            slc_close = close5s[entry_idx:end_idx + 1]

            # Signed P&L progression vs entry
            if leg_dir == 'LONG':
                # Favorable = high - entry, adverse = entry - low
                mfe_pts = float((slc_high - entry_price).max())
                mae_pts = float((entry_price - slc_low).max())
            else:  # SHORT
                mfe_pts = float((entry_price - slc_low).max())
                mae_pts = float((slc_high - entry_price).max())
            # Clamp at zero (MAE/MFE are non-negative)
            mfe_pts = max(0.0, mfe_pts)
            mae_pts = max(0.0, mae_pts)
            pnl_pts_so_far = float(leg_sign * (slc_close[-1] - entry_price))
            has_reached_R_against = mae_pts >= r_price

            # V2 features at latest 1m close <= bar_ts
            v_idx = int(np.searchsorted(truth_ts, bar_ts, side='right') - 1)
            if v_idx < 0:
                v_idx = 0
            v2_row = v2_arr[v_idx]

            row = {
                'leg_id': leg_id,
                'day': day,
                'entry_ts': entry_ts,
                'leg_dir': leg_dir,
                'K': K,
                'bar_ts': bar_ts,
                'r_price': r_price,
                'exit_ts': exit_ts,
                'exit_pnl_pts': exit_pnl_pts,
                'exit_pnl_usd': exit_pnl_usd,
                'mae_pts_so_far': mae_pts,
                'mfe_pts_so_far': mfe_pts,
                'pnl_pts_so_far': pnl_pts_so_far,
                'pnl_usd_so_far': pnl_pts_so_far * DOLLAR_PER_POINT,
                'bars_since_entry': K,
                'has_reached_R_against': has_reached_R_against,
            }
            # Add V2 features (denormalized)
            for i, c in enumerate(v2_cols):
                row[c] = float(v2_row[i])
            out_rows.append(row)

    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs',
                    default='reports/findings/regret_oracle/is_hardened_legs.csv',
                    help='IS hardened leg list CSV')
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet',
                    help='IS pivot/feature dataset (V2 per 1m bar)')
    ap.add_argument('--bars5s-dir', default='DATA/ATLAS/5s',
                    help='Directory of 5s bar parquets')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.parquet',
                    help='Output trajectory dataset parquet')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.txt',
                    help='Output summary report')
    args = ap.parse_args()

    print(f'Loading legs: {args.legs}')
    legs = pd.read_csv(args.legs)
    legs = legs.reset_index().rename(columns={'index': 'leg_id'})
    print(f'  legs {len(legs):,}   days {legs["day"].nunique()}')

    print(f'Loading truth: {args.truth}')
    truth = pd.read_parquet(args.truth)
    v2_cols = get_v2_cols(truth)
    print(f'  truth rows {len(truth):,}   V2 cols {len(v2_cols)}')

    bars5s_dir = Path(args.bars5s_dir)

    days = sorted(legs['day'].unique())
    all_rows = []
    for day in tqdm(days, desc='days'):
        day_legs = legs[legs['day'] == day]
        rows = process_day(day, day_legs, truth, bars5s_dir, v2_cols)
        all_rows.extend(rows)

    if not all_rows:
        print('No rows produced.')
        return

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('TRADE TRAJECTORY DATASET (IS only)')
    out('=' * 78)
    out(f'Source legs: {args.legs}')
    out(f'Source features: {args.truth}')
    out(f'Total trajectory rows: {len(df):,}')
    out(f'Unique legs: {df["leg_id"].nunique():,} '
        f'(input legs: {len(legs):,}; missing trajectory rows = '
        f'{len(legs) - df["leg_id"].nunique():,} legs whose K bars '
        f'exit early)')
    out(f'Days: {df["day"].nunique()}')
    out(f'V2 feature cols: {len(v2_cols)}')
    out('')
    out('Rows per K horizon:')
    for K in K_HORIZONS:
        n = int((df['K'] == K).sum())
        out(f'  K={K:>3}  ({K*5}s)   n={n:>6,}')
    out('')
    out('Leg outcome distribution (one row per leg, K=5 horizon):')
    leg5 = df[df['K'] == 5]
    pnl = leg5['exit_pnl_usd'].values
    out(f'  Legs with K=5 row: {len(leg5):,}')
    out(f'  Mean exit pnl:    ${pnl.mean():+.2f}')
    out(f'  Median exit pnl:  ${np.median(pnl):+.2f}')
    out(f'  Frac losing:      {(pnl < 0).mean()*100:.1f}%')
    out(f'  Frac < -$50:      {(pnl < -50).mean()*100:.1f}%')
    out(f'  Frac < -$100:     {(pnl < -100).mean()*100:.1f}%')
    out(f'  Frac < -$200:     {(pnl < -200).mean()*100:.1f}%')
    out('')
    out('MAE at each K (vs entry, in points):')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        if len(sub) == 0:
            continue
        out(f'  K={K:>3}  median {sub["mae_pts_so_far"].median():.2f}  '
            f'90th {sub["mae_pts_so_far"].quantile(0.9):.2f}  '
            f'max {sub["mae_pts_so_far"].max():.2f}')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
