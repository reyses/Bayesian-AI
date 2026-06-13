"""
Databento to ATLAS — reads any Databento download folder and creates
ATLAS parquet structure.

Auto-detects the schema (trades, ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d)
from filenames and processes accordingly.

Filters to front-month MNQ contract only.
Saves as DATA/ATLAS/{tf}/YYYY_MM_DD.parquet (one file per day).

Usage:
  python tools/databento_to_atlas.py "C:/Users/reyse/OneDrive/Desktop/RAW/GLBX-20260402-DD6HDFKMA9"
  python tools/databento_to_atlas.py all     # process all folders in RAW
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import os
import sys
import glob
import gc
import numpy as np
import pandas as pd
import databento as db
from datetime import datetime, timezone
from tqdm import tqdm

RAW_ROOT = 'C:/Users/reyse/OneDrive/Desktop/RAW'
ATLAS_OUT = 'DATA/ATLAS'

# Front-month symbol mapping
FRONT_MONTH = {
    '2024-01': 'MNQH4', '2024-02': 'MNQH4', '2024-03': 'MNQH4',
    '2024-04': 'MNQM4', '2024-05': 'MNQM4', '2024-06': 'MNQM4',
    '2024-07': 'MNQU4', '2024-08': 'MNQU4', '2024-09': 'MNQU4',
    '2024-10': 'MNQZ4', '2024-11': 'MNQZ4', '2024-12': 'MNQZ4',
    '2025-01': 'MNQH5', '2025-02': 'MNQH5', '2025-03': 'MNQH5',
    '2025-04': 'MNQM5', '2025-05': 'MNQM5', '2025-06': 'MNQM5',
    '2025-07': 'MNQU5', '2025-08': 'MNQU5', '2025-09': 'MNQU5',
    '2025-10': 'MNQZ5', '2025-11': 'MNQZ5', '2025-12': 'MNQZ5',
    '2026-01': 'MNQH6', '2026-02': 'MNQH6', '2026-03': 'MNQH6',
    '2026-04': 'MNQM6', '2026-05': 'MNQM6', '2026-06': 'MNQM6',
}

# Schema to TF mapping
SCHEMA_TO_TF = {
    'ohlcv-1s': '1s',
    'ohlcv-1m': '1m',
    'ohlcv-1h': '1h',
    'ohlcv-1d': '1D',
    'trades': 'trades',  # raw trades, need aggregation
}


def detect_schema(folder):
    """Auto-detect the Databento schema from filenames."""
    files = glob.glob(os.path.join(folder, '*.dbn.zst'))
    files = [f for f in files if 'condition' not in os.path.basename(f)]
    if not files:
        return None, []

    # Check first filename for schema
    fname = os.path.basename(files[0])
    for schema_key in SCHEMA_TO_TF:
        if schema_key in fname:
            return schema_key, files

    return None, files


def read_dbn(fpath):
    """Read a .dbn.zst file into DataFrame."""
    data = db.DBNStore.from_file(fpath)
    df = data.to_df()
    if len(df) == 0:
        return pd.DataFrame()

    df = df.reset_index()

    # Convert timestamp
    if 'ts_event' in df.columns:
        df['timestamp'] = df['ts_event'].astype(np.int64) // 10**9
    elif 'ts_recv' in df.columns:
        df['timestamp'] = df['ts_recv'].astype(np.int64) // 10**9

    return df


def filter_front_month(df):
    """Filter to front-month contract only."""
    if 'symbol' not in df.columns:
        return df

    df['_ym'] = df['timestamp'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m'))
    df['_front'] = df['_ym'].map(FRONT_MONTH)
    df = df[df['symbol'] == df['_front']].copy()
    df = df.drop(columns=['_ym', '_front'], errors='ignore')
    return df


def normalize_ohlcv(df):
    """Normalize OHLCV columns — handle Databento fixed-point if needed."""
    if len(df) == 0:
        return df

    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns and len(df) > 0 and df[col].iloc[0] > 1e6:
            df[col] = df[col] / 1e9

    # Keep only standard columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def aggregate_trades_to_1s(df):
    """Aggregate raw trades to 1s OHLCV bars."""
    if 'price' not in df.columns:
        return pd.DataFrame()

    df['bar_ts'] = (df['timestamp'] // 1).astype(int)  # floor to second

    bars = df.groupby('bar_ts').agg(
        timestamp=('bar_ts', 'first'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('size', 'sum') if 'size' in df.columns else ('price', 'count'),
    ).reset_index(drop=True)

    return bars


def save_daily(df, tf_label):
    """Save DataFrame to daily parquet files."""
    tf_dir = os.path.join(ATLAS_OUT, tf_label)
    os.makedirs(tf_dir, exist_ok=True)

    df['_day'] = df['timestamp'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m_%d'))

    total = 0
    for day, group in df.groupby('_day'):
        day_path = os.path.join(tf_dir, f'{day}.parquet')
        out = group.drop(columns=['_day']).sort_values('timestamp').reset_index(drop=True)

        # Merge with existing
        if os.path.exists(day_path):
            old = pd.read_parquet(day_path)
            out = pd.concat([old, out]).drop_duplicates(
                subset='timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)

        out.to_parquet(day_path, index=False)
        total += len(group)

    return total


def process_folder(folder):
    """Process a single Databento download folder."""
    name = os.path.basename(folder)
    schema, files = detect_schema(folder)

    if schema is None:
        print(f'  {name}: unknown schema, skipping')
        return

    if schema == 'trades':
        print(f'  {name}: raw trades — skipping (use ohlcv-1s instead)')
        return

    tf = SCHEMA_TO_TF[schema]
    print(f'  {name}: {schema} -> {tf} ({len(files)} files)')

    if len(files) == 1 and tf in ('1s', 'trades'):
        # Single large file — process in one go
        print(f'    Reading single file...')
        df = read_dbn(files[0])
        df = filter_front_month(df)

        if tf == 'trades':
            df = aggregate_trades_to_1s(df)
            tf = '1s'
        else:
            df = normalize_ohlcv(df)

        n = save_daily(df, tf)
        print(f'    Saved {n:,} bars to {tf}/')
        del df; gc.collect()

    else:
        # Daily files — process one at a time
        all_bars = []
        for fpath in tqdm(files, desc=f'    {tf}'):
            df = read_dbn(fpath)
            if len(df) == 0:
                continue
            df = filter_front_month(df)
            if len(df) == 0:
                continue

            if tf == 'trades':
                df = aggregate_trades_to_1s(df)
            else:
                df = normalize_ohlcv(df)

            all_bars.append(df)

        if all_bars:
            combined = pd.concat(all_bars, ignore_index=True)
            combined = combined.drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp')
            actual_tf = '1s' if schema == 'trades' else tf
            n = save_daily(combined, actual_tf)
            print(f'    Saved {n:,} bars to {actual_tf}/')
            del combined; gc.collect()


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print(f'DATABENTO TO ATLAS')
    print(f'  Output: {ATLAS_OUT}/{{tf}}/YYYY_MM_DD.parquet')
    print()

    if target == 'all':
        folders = sorted(glob.glob(os.path.join(RAW_ROOT, 'GLBX-*')))
        folders = [f for f in folders if os.path.isdir(f)]
    else:
        folders = [target]

    for folder in folders:
        process_folder(folder)
        gc.collect()

    # Summary
    print(f'\nATLAS SUMMARY:')
    for tf in ['1s', '1m', '1h', '1D']:
        tf_dir = os.path.join(ATLAS_OUT, tf)
        if os.path.exists(tf_dir):
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if files:
                total = sum(len(pd.read_parquet(f)) for f in files[:3])  # sample first 3
                avg = total // min(3, len(files))
                est_total = avg * len(files)
                print(f'  {tf:>4}: {len(files)} days, ~{est_total:,} bars')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
