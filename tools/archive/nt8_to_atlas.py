"""
Convert NT8 tick export to ATLAS 1s parquet format.

Reads: DATA/MNQ 06-26.Last.txt (NT8 tick export)
Writes: DATA/ATLAS/1s/YYYY_MM.parquet (ATLAS format)

Also aggregates to higher TFs: 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1D

Usage:
  python tools/nt8_to_atlas.py
  python tools/nt8_to_atlas.py --input "DATA/MNQ 06-26.Last.txt"
  python tools/nt8_to_atlas.py --only-missing   # only fill gaps in existing ATLAS
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

INPUT = sys.argv[1] if len(sys.argv) > 1 else 'DATA/MNQ 06-26.Last.txt'
ATLAS_ROOT = 'DATA/ATLAS'
TICK = 0.25

# TF aggregation periods in seconds
TF_PERIODS = {
    '1s': 1, '5s': 5, '15s': 15, '30s': 30,
    '1m': 60, '3m': 180, '5m': 300, '15m': 900,
    '30m': 1800, '1h': 3600, '4h': 14400, '1D': 86400,
}


def parse_nt8_ticks(path):
    """Parse NT8 tick export into DataFrame with unix timestamp."""
    print(f'  Reading {path}...')
    df = pd.read_csv(path, sep=';', header=None,
                      names=['datetime_raw', 'last', 'bid', 'ask', 'volume'])

    # Parse datetime: "YYYYMMDD HHMMSS NNNNNNN" -> unix timestamp
    print(f'  Parsing {len(df):,} ticks...')

    # Fast vectorized parse
    dt_str = df['datetime_raw'].str[:15]  # "YYYYMMDD HHMMSS"
    timestamps = pd.to_datetime(dt_str, format='%Y%m%d %H%M%S').values.astype(np.int64) // 10**9

    df['timestamp'] = timestamps.astype(np.float64)
    df['price'] = df['last']

    return df[['timestamp', 'price', 'bid', 'ask', 'volume']]


def aggregate_to_bars(ticks, period_seconds):
    """Aggregate tick data to OHLCV bars at given period."""
    ticks = ticks.sort_values('timestamp')

    # Floor timestamp to period boundary
    ticks['bar_ts'] = (ticks['timestamp'] // period_seconds) * period_seconds

    bars = ticks.groupby('bar_ts').agg(
        timestamp=('bar_ts', 'first'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('volume', 'sum'),
    ).reset_index(drop=True)

    return bars


def save_atlas(bars, tf_label):
    """Save bars to ATLAS format: {atlas}/{tf}/YYYY_MM.parquet"""
    tf_dir = os.path.join(ATLAS_ROOT, tf_label)
    os.makedirs(tf_dir, exist_ok=True)

    bars['_month'] = bars['timestamp'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m'))

    for month, group in bars.groupby('_month'):
        month_path = os.path.join(tf_dir, f'{month}.parquet')
        out = group.drop(columns=['_month'])

        # Merge with existing (dedup by timestamp)
        if os.path.exists(month_path):
            old = pd.read_parquet(month_path)
            out = pd.concat([old, out]).drop_duplicates(
                subset='timestamp', keep='last').sort_values('timestamp')

        out.to_parquet(month_path, index=False)

    return len(bars)


def main():
    print(f'NT8 Tick Export -> ATLAS Converter')
    print(f'  Input: {INPUT}')
    print()

    ticks = parse_nt8_ticks(INPUT)
    print(f'  Ticks: {len(ticks):,}')
    print(f'  Range: {datetime.utcfromtimestamp(ticks["timestamp"].iloc[0]).strftime("%Y-%m-%d %H:%M")} '
          f'to {datetime.utcfromtimestamp(ticks["timestamp"].iloc[-1]).strftime("%Y-%m-%d %H:%M")}')
    print()

    for tf_label, period in tqdm(TF_PERIODS.items(), desc='Aggregating'):
        bars = aggregate_to_bars(ticks, period)
        n = save_atlas(bars, tf_label)
        tqdm.write(f'  {tf_label:>4}: {n:>10,} bars')

    print(f'\nDone. ATLAS updated at {ATLAS_ROOT}/')


if __name__ == '__main__':
    main()
