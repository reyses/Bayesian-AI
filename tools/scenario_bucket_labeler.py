"""Label each daisy-chain trade with scenario buckets.

Per user 2026-05-16 late-late: predict scenario = (direction, duration,
speed, trajectory) buckets from V2 entry features via LSTM. This tool
computes the bucket labels for training/eval.

Buckets:
  direction:   LONG / SHORT  (2 classes — already in daisy CSV)
  duration:    quartiles of time_to_mfe_min (4 classes: D0..D3)
  speed:       quartiles of mfe_velocity  ($/min, 4 classes: V0..V3)
  trajectory:  quartiles of MAE/MFE ratio (4 classes: T0..T3)
    T0 = MONOTONIC  (MAE/MFE in [0, 0.25))
    T1 = SMALL_PULL (MAE/MFE in [0.25, 0.50))
    T2 = BIG_PULL   (MAE/MFE in [0.50, 0.75))
    T3 = CHURN      (MAE/MFE in [0.75, 1.0+])

We use OPERATIONAL boundaries for trajectory (semantically meaningful)
and QUANTILE boundaries for duration + speed (data-balanced).

MAE definition: for LONG trade, MAE = entry_price - min(low) within
[oracle_ts, exit_ts]. For SHORT, MAE = max(high) - entry_price. Positive
by construction. MAE_dollars = MAE_ticks * $0.50/tick.

Input:  daisy CSV
Output:
  daisy_with_buckets_<name>.csv  — all original cols + 4 bucket cols + MAE
  bucket_boundaries_<name>.json  — quartile boundaries (use IS for OOS)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


RAW_5S_DIR = Path('DATA/ATLAS/5s')
TICK = 0.25
TICK_VALUE = 0.50  # dollars per tick (MNQ)
TF_S = 5


def date_key(ts_unix: int) -> str:
    """Get YYYY_MM_DD key for a UTC unix timestamp (CME session day == calendar day after 22:00 UTC offset; we use simple UTC date for file lookup with ±1 pad)."""
    return pd.to_datetime(ts_unix, unit='s').strftime('%Y_%m_%d')


def load_5s_window(start_ts: int, end_ts: int, cache: dict) -> pd.DataFrame:
    """Load 5s OHLC bars covering [start_ts, end_ts]. Cache by date key."""
    keys = set()
    for k_ts in (start_ts, end_ts):
        d = pd.to_datetime(k_ts, unit='s')
        for off in (-1, 0, 1):
            keys.add((d + pd.Timedelta(days=off)).strftime('%Y_%m_%d'))
    frames = []
    for k in sorted(keys):
        if k not in cache:
            p = RAW_5S_DIR / f'{k}.parquet'
            cache[k] = pd.read_parquet(p) if p.exists() else None
        if cache[k] is not None:
            frames.append(cache[k])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates('timestamp')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_mae(direction: str, entry_price: float,
                bars: pd.DataFrame, start_ts: int, end_ts: int) -> float:
    """Compute MAE in DOLLARS for a trade. Returns max adverse move."""
    win = bars[(bars['timestamp'] >= start_ts) & (bars['timestamp'] <= end_ts)]
    if len(win) == 0:
        return np.nan
    if direction == 'LONG':
        # MAE = entry - lowest low within window (in price units), $0.50/tick
        adverse_price = entry_price - win['low'].min()
    else:  # SHORT
        adverse_price = win['high'].max() - entry_price
    adverse_ticks = max(0.0, adverse_price / TICK)
    return float(adverse_ticks * TICK_VALUE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='daisy-chain CSV')
    ap.add_argument('--out', required=True, help='labeled CSV output')
    ap.add_argument('--boundaries-in', default=None,
                    help='Use saved boundaries.json (for OOS labeling)')
    ap.add_argument('--boundaries-out', default=None,
                    help='Save boundaries.json (for IS labeling)')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f'Loading {len(df)} trades from {args.input}')
    print(f'Date range: {df["session_date"].min()} to {df["session_date"].max()}')

    # Compute MAE per trade
    print('\n--- Computing MAE per trade ---')
    cache = {}
    mae_dollars = np.zeros(len(df), dtype=np.float32)
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        start_ts = int(row.oracle_ts)
        end_ts = int(row.exit_ts) if not pd.isna(row.exit_ts) else \
            start_ts + int(row.time_to_mfe_min * 60)
        if end_ts <= start_ts:
            end_ts = start_ts + int(row.time_to_mfe_min * 60)
        bars = load_5s_window(start_ts, end_ts, cache)
        if bars is None:
            mae_dollars[i] = np.nan
            continue
        mae_dollars[i] = compute_mae(row.direction, float(row.entry_price),
                                     bars, start_ts, end_ts)
        if (i+1) % max(1, len(df)//20) == 0:
            print(f'  {i+1}/{len(df)} ({time.time()-t0:.0f}s)', flush=True)
    df['mae_dollars'] = mae_dollars

    # MAE/MFE ratio for trajectory bucketing
    # mfe_dollars is always positive (favorable). mae_dollars is also positive
    # (adverse). Ratio in [0, ...]; cap at 1.0 because adverse > MFE means
    # the trade actually lost money in the window before resolving — rare for
    # daisy oracle but possible at boundaries.
    safe_mfe = np.where(df['mfe_dollars'] > 1e-6, df['mfe_dollars'], 1e-6)
    df['mae_mfe_ratio'] = (df['mae_dollars'] / safe_mfe).clip(0, 2.0)

    n_nan_mae = df['mae_dollars'].isna().sum()
    print(f'\nMAE stats: median ${df["mae_dollars"].median():.0f}, '
          f'mean ${df["mae_dollars"].mean():.0f}; '
          f'{n_nan_mae} NaN')
    print(f'MFE stats: median ${df["mfe_dollars"].median():.0f}, '
          f'mean ${df["mfe_dollars"].mean():.0f}')
    print(f'MAE/MFE ratio percentiles: '
          f'p25={df["mae_mfe_ratio"].quantile(0.25):.3f} '
          f'p50={df["mae_mfe_ratio"].quantile(0.50):.3f} '
          f'p75={df["mae_mfe_ratio"].quantile(0.75):.3f} '
          f'p90={df["mae_mfe_ratio"].quantile(0.90):.3f}')

    # === Buckets ===
    if args.boundaries_in:
        print(f'\nUsing boundaries from {args.boundaries_in}')
        with open(args.boundaries_in) as f:
            bnd = json.load(f)
        dur_b = bnd['duration_boundaries']  # 3 cut points for 4 classes
        spd_b = bnd['speed_boundaries']
        traj_b = bnd['trajectory_boundaries']
    else:
        # Compute quartile boundaries from this dataset (IS)
        dur_b = list(df['time_to_mfe_min'].quantile([0.25, 0.50, 0.75]).values)
        spd_b = list(df['mfe_velocity'].quantile([0.25, 0.50, 0.75]).values)
        # Trajectory: operational boundaries (MAE/MFE)
        traj_b = [0.25, 0.50, 0.75]

    print(f'\n--- Bucket boundaries ---')
    print(f'Duration (min):   {dur_b[0]:.2f} | {dur_b[1]:.2f} | {dur_b[2]:.2f}')
    print(f'Speed ($/min):    {spd_b[0]:.2f} | {spd_b[1]:.2f} | {spd_b[2]:.2f}')
    print(f'Trajectory (MAE/MFE): {traj_b[0]:.2f} | {traj_b[1]:.2f} | {traj_b[2]:.2f}')

    def assign_bucket(vals, cuts):
        """Return integer bucket 0..3 from 3 cut points."""
        return (np.searchsorted(np.asarray(cuts), vals, side='right')).astype(np.int8)

    df['bucket_direction'] = (df['direction'] == 'LONG').astype(np.int8)  # 1=LONG, 0=SHORT
    df['bucket_duration']  = assign_bucket(df['time_to_mfe_min'].values, dur_b)
    df['bucket_speed']     = assign_bucket(df['mfe_velocity'].values, spd_b)
    df['bucket_trajectory'] = assign_bucket(df['mae_mfe_ratio'].values, traj_b)

    print(f'\n--- Bucket distributions ---')
    for col in ['bucket_direction', 'bucket_duration', 'bucket_speed', 'bucket_trajectory']:
        vc = df[col].value_counts().sort_index()
        print(f'  {col}: {dict(vc)}')

    # Cross-tab: are these axes independent?
    print(f'\n--- Cross-tabs (top-level dependence) ---')
    print('\nDirection x Duration:')
    print(pd.crosstab(df['bucket_direction'], df['bucket_duration']))
    print('\nDirection x Speed:')
    print(pd.crosstab(df['bucket_direction'], df['bucket_speed']))
    print('\nDuration x Speed:')
    print(pd.crosstab(df['bucket_duration'], df['bucket_speed']))
    print('\nTrajectory x Duration:')
    print(pd.crosstab(df['bucket_trajectory'], df['bucket_duration']))

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')

    if args.boundaries_out:
        bnd_out = {
            'duration_boundaries': [float(x) for x in dur_b],
            'speed_boundaries': [float(x) for x in spd_b],
            'trajectory_boundaries': [float(x) for x in traj_b],
            'n_trades': int(len(df)),
            'date_range': [str(df['session_date'].min()), str(df['session_date'].max())],
        }
        with open(args.boundaries_out, 'w') as f:
            json.dump(bnd_out, f, indent=2)
        print(f'Wrote: {args.boundaries_out}')


if __name__ == '__main__':
    main()
