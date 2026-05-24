"""
Convert NT8 exported data to ATLAS parquet format.

Reads NT8 txt exports (semicolon-separated) and saves as
DATA/ATLAS/{tf}/YYYY_MM_DD.parquet (one file per day).

Supports:
  - 1m bars: YYYYMMDD HHMMSS;open;high;low;close;volume
  - Tick data: YYYYMMDD HHMMSS FFFFFFF;price;bid;ask;volume

Usage:
    python tools/nt8_export_to_atlas.py "DATA/MNQ 06-26min .Last.txt" --tf 1m
    python tools/nt8_export_to_atlas.py "DATA/MNQ 06-26.Last.txt" --tf tick
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

ATLAS_OUT = 'DATA/ATLAS'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='NT8 export to ATLAS')
    p.add_argument('input', type=str, help='Path to NT8 export txt file')
    p.add_argument('--tf', type=str, default='1m', choices=['1m', 'tick', '1s'],
                   help='Timeframe of the export')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    return p.parse_args()


def parse_1m_line(line):
    """Parse: YYYYMMDD HHMMSS;open;high;low;close;volume"""
    parts = line.strip().split(';')
    if len(parts) != 6:
        return None
    ts_str = parts[0]  # "20260101 230100"
    try:
        dt = datetime.strptime(ts_str, '%Y%m%d %H%M%S')
        timestamp = dt.timestamp()
        return {
            'timestamp': timestamp,
            'open': float(parts[1]),
            'high': float(parts[2]),
            'low': float(parts[3]),
            'close': float(parts[4]),
            'volume': int(parts[5]),
        }
    except (ValueError, IndexError):
        return None


def parse_tick_line(line):
    """Parse: YYYYMMDD HHMMSS FFFFFFF;price;bid;ask;volume"""
    parts = line.strip().split(';')
    if len(parts) != 4:
        return None
    ts_str = parts[0]  # "20260101 230000 0320000"
    try:
        # Parse date+time, ignore fractional seconds for 1s aggregation
        dt_str = ts_str[:15]  # "20260101 230000"
        dt = datetime.strptime(dt_str, '%Y%m%d %H%M%S')
        timestamp = dt.timestamp()
        price = float(parts[1])
        return {
            'timestamp': timestamp,
            'price': price,
            'bid': float(parts[2]),
            'ask': float(parts[3]),
            'volume': int(parts[4]),
        }
    except (ValueError, IndexError):
        return None


def ticks_to_1s_bars(ticks):
    """Aggregate ticks into 1-second OHLCV bars."""
    if not ticks:
        return []

    df = pd.DataFrame(ticks)
    df['second'] = df['timestamp'].astype(int)

    bars = []
    for sec, group in df.groupby('second'):
        bars.append({
            'timestamp': float(sec),
            'open': group['price'].iloc[0],
            'high': group['price'].max(),
            'low': group['price'].min(),
            'close': group['price'].iloc[-1],
            'volume': group['volume'].sum(),
        })

    return sorted(bars, key=lambda b: b['timestamp'])


def save_by_day(bars, tf_label):
    """Save bars grouped by day to ATLAS parquet format."""
    df = pd.DataFrame(bars)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y_%m_%d')

    out_dir = os.path.join(ATLAS_OUT, tf_label)
    os.makedirs(out_dir, exist_ok=True)

    days = df['date'].unique()
    saved = 0
    skipped = 0

    for day in tqdm(sorted(days), desc=f'Saving {tf_label}', unit='day'):
        day_df = df[df['date'] == day].drop(columns=['date']).sort_values('timestamp').reset_index(drop=True)
        out_path = os.path.join(out_dir, f'{day}.parquet')

        if os.path.exists(out_path):
            skipped += 1
            continue

        day_df.to_parquet(out_path, index=False)
        saved += 1

    print(f'  Saved: {saved} days | Skipped (exists): {skipped} | Total bars: {len(df):,}')
    return saved


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f'File not found: {args.input}')
        return

    print(f'Converting NT8 export to ATLAS')
    print(f'  Input: {args.input}')
    print(f'  TF: {args.tf}')

    # Read and parse
    print(f'  Reading...')
    with open(args.input, 'r') as f:
        lines = f.readlines()
    print(f'  Lines: {len(lines):,}')

    if args.tf == '1m':
        bars = []
        for line in tqdm(lines, desc='Parsing 1m'):
            bar = parse_1m_line(line)
            if bar:
                bars.append(bar)
        print(f'  Parsed: {len(bars):,} bars')
        save_by_day(bars, '1m')

    elif args.tf in ('tick', '1s'):
        ticks = []
        for line in tqdm(lines, desc='Parsing ticks'):
            tick = parse_tick_line(line)
            if tick:
                ticks.append(tick)
        print(f'  Parsed: {len(ticks):,} ticks')

        # Aggregate to 1s bars
        print(f'  Aggregating to 1s bars...')
        bars_1s = ticks_to_1s_bars(ticks)
        print(f'  1s bars: {len(bars_1s):,}')
        save_by_day(bars_1s, '1s')

    print(f'\nDone. ATLAS output: {ATLAS_OUT}/')


if __name__ == '__main__':
    main()
