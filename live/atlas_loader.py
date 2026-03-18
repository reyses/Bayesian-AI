"""
ATLAS parquet loader for compressed history replay.

Usage:
    python -m live.atlas_loader --days 5
"""

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd


def load_atlas_range(atlas_root: str, tf: str, n_days: int,
                     end_date: datetime = None) -> pd.DataFrame:
    """Load N trading days of data from ATLAS parquet files.

    Args:
        atlas_root: Path to DATA/ATLAS/
        tf: Timeframe label (e.g. '15s', '4h')
        n_days: Number of trading days to load
        end_date: End date (default: last available date in data)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    tf_dir = os.path.join(atlas_root, tf)
    parquet_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {tf_dir}")

    # Read and concat all files (parquet read is fast)
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    if end_date is not None:
        end_ts = end_date.timestamp()
        df = df[df['timestamp'] <= end_ts]

    # Identify trading days by gaps > 30 min (maintenance window)
    ts = df['timestamp'].values
    gaps = np.diff(ts)
    gap_mask = gaps > 1800  # 30 min in seconds
    day_boundaries = np.where(gap_mask)[0] + 1  # indices where new days start

    # Build day start indices
    day_starts = np.concatenate([[0], day_boundaries])
    n_available = len(day_starts)

    if n_available < n_days:
        n_days = n_available

    # Take last n_days
    start_idx = day_starts[-n_days]
    df = df.iloc[start_idx:].reset_index(drop=True)

    return df


def load_multi_tf(atlas_root: str, n_days: int = 5) -> dict:
    """Load multi-TF data for replay.

    Returns:
        {tf_label: pd.DataFrame} for all needed TFs.
    """
    result = {}

    # Primary: 15s (always required)
    result['15s'] = load_atlas_range(atlas_root, '15s', n_days)
    print(f"  ATLAS 15s: {len(result['15s']):,} bars ({n_days} days)")

    # Sub-resolution (optional)
    for tf in ['5s', '1s']:
        try:
            result[tf] = load_atlas_range(atlas_root, tf, n_days)
            print(f"  ATLAS {tf}: {len(result[tf]):,} bars")
        except FileNotFoundError:
            pass

    # Supra-resolution: 4h needs more history for meaningful states
    try:
        result['4h'] = load_atlas_range(atlas_root, '4h', n_days * 4)
        print(f"  ATLAS 4h: {len(result['4h']):,} bars ({n_days * 4} days)")
    except FileNotFoundError:
        pass

    return result


def split_trading_days(df: pd.DataFrame) -> list:
    """Split continuous bars into trading days.

    A trading day starts at ~5PM CT and ends at ~4PM CT next day.
    Detected by gaps > 30 minutes in timestamps.

    Returns:
        List of DataFrames, one per day, sorted chronologically.
    """
    ts = df['timestamp'].values
    gaps = np.diff(ts)
    gap_mask = gaps > 1800  # 30 min
    day_boundaries = np.where(gap_mask)[0] + 1

    starts = np.concatenate([[0], day_boundaries])
    ends = np.concatenate([day_boundaries, [len(df)]])

    days = []
    for s, e in zip(starts, ends):
        day_df = df.iloc[s:e].reset_index(drop=True)
        if len(day_df) > 100:  # skip tiny fragments
            days.append(day_df)

    return days


def _slice_day(df_tf: pd.DataFrame, day_df: pd.DataFrame) -> pd.DataFrame:
    """Slice a TF dataframe to match a trading day's time range."""
    if df_tf is None or len(df_tf) == 0:
        return pd.DataFrame()
    t_start = day_df['timestamp'].iloc[0]
    t_end = day_df['timestamp'].iloc[-1]
    mask = (df_tf['timestamp'] >= t_start) & (df_tf['timestamp'] <= t_end)
    return df_tf[mask].reset_index(drop=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ATLAS loader test')
    parser.add_argument('--days', type=int, default=5)
    parser.add_argument('--atlas', default='DATA/ATLAS')
    args = parser.parse_args()

    tf_data = load_multi_tf(args.atlas, args.days)
    df = tf_data['15s']
    days = split_trading_days(df)
    print(f"\n  Split into {len(days)} trading days:")
    for i, d in enumerate(days):
        t0 = datetime.fromtimestamp(d['timestamp'].iloc[0])
        t1 = datetime.fromtimestamp(d['timestamp'].iloc[-1])
        print(f"    Day {i+1}: {t0:%Y-%m-%d %H:%M} -> {t1:%H:%M}  ({len(d):,} bars)")
