"""
Rebuild ATLAS from Databento source files.

Reads Databento .dbn.zst files, filters to front-month MNQ,
saves to DATA/ATLAS/{tf}/YYYY_MM.parquet format.

Databento sources:
  1s:  GLBX-20260210 (single file, Jan 2025 - Feb 2026)
  1m:  GLBX-20260402-DD6H (daily files)
  1h:  GLBX-20260402-MUD4 (daily files)
  1D:  GLBX-20260403-9ACN (monthly files)

Usage:
  python tools/rebuild_atlas_databento.py
  python tools/rebuild_atlas_databento.py --tf 1m     # specific TF only
  python tools/rebuild_atlas_databento.py --skip-1s    # skip 1s (slow)
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
ATLAS_OUT = 'DATA/ATLAS'  # replace existing ATLAS

# Front-month symbol mapping (MNQ and NQ quarterly contracts)
# H = March, M = June, U = September, Z = December
# NQ (full-size) and MNQ (micro) share identical OHLC prices
FRONT_MONTH_MNQ = {
    '2025-01': 'MNQH5', '2025-02': 'MNQH5', '2025-03': 'MNQH5',
    '2025-04': 'MNQM5', '2025-05': 'MNQM5', '2025-06': 'MNQM5',
    '2025-07': 'MNQU5', '2025-08': 'MNQU5', '2025-09': 'MNQU5',
    '2025-10': 'MNQZ5', '2025-11': 'MNQZ5', '2025-12': 'MNQZ5',
    '2026-01': 'MNQH6', '2026-02': 'MNQH6', '2026-03': 'MNQH6',
    '2026-04': 'MNQM6',
}
FRONT_MONTH_NQ = {
    '2025-01': 'NQH5', '2025-02': 'NQH5', '2025-03': 'NQH5',
    '2025-04': 'NQM5', '2025-05': 'NQM5', '2025-06': 'NQM5',
    '2025-07': 'NQU5', '2025-08': 'NQU5', '2025-09': 'NQU5',
    '2025-10': 'NQZ5', '2025-11': 'NQZ5', '2025-12': 'NQZ5',
    '2026-01': 'NQH6', '2026-02': 'NQH6', '2026-03': 'NQH6',
    '2026-04': 'NQM6',
}

# Databento paths
PATHS = {
    '1s': os.path.join(RAW_ROOT, 'GLBX-20260210-4K6G5MC7B6'),
    '1m': os.path.join(RAW_ROOT, 'GLBX-20260402-DD6HDFKMA9'),
    '1h': os.path.join(RAW_ROOT, 'GLBX-20260402-MUD4KLQKRA'),
    '1D': os.path.join(RAW_ROOT, 'GLBX-20260403-9ACNW3VK4H'),
}

SKIP_1S = '--skip-1s' in sys.argv
TF_FILTER = None
if '--tf' in sys.argv:
    idx = sys.argv.index('--tf')
    TF_FILTER = sys.argv[idx + 1]


def process_dbn_file(fpath, tf_label):
    """Read a .dbn.zst file, filter to front month, return OHLCV DataFrame."""
    data = db.DBNStore.from_file(fpath)
    df = data.to_df()

    if len(df) == 0:
        return pd.DataFrame()

    # Convert index to timestamp
    df = df.reset_index()
    df['timestamp'] = df['ts_event'].astype(np.int64) // 10**9

    # Auto-detect NQ vs MNQ from first symbol
    first_sym = df['symbol'].iloc[0] if 'symbol' in df.columns else ''
    if first_sym.startswith('MNQ'):
        front_month_map = FRONT_MONTH_MNQ
    else:
        front_month_map = FRONT_MONTH_NQ
    print(f'    Symbol prefix: {"MNQ" if first_sym.startswith("MNQ") else "NQ"} (first: {first_sym})')

    # Filter to front-month symbol
    df['_month'] = df['timestamp'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m'))
    df['_front'] = df['_month'].map(front_month_map)

    # Also filter out spread symbols (contain '-')
    df = df[~df['symbol'].str.contains('-', na=False)]
    df = df[df['symbol'] == df['_front']]

    if len(df) == 0:
        print(f'    WARNING: Front month filter returned 0 rows')
        return pd.DataFrame()

    # Databento prices are in fixed-point (multiply by 1e-9 for some schemas)
    # Check if prices look right
    if df['close'].iloc[0] > 1e6:
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] / 1e9

    # Select OHLCV columns
    out = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out


def save_to_atlas(bars, tf_label):
    """Save bars to ATLAS format by day (YYYY_MM_DD.parquet)."""
    tf_dir = os.path.join(ATLAS_OUT, tf_label)
    os.makedirs(tf_dir, exist_ok=True)

    bars['_day'] = bars['timestamp'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m_%d'))

    total = 0
    for day, group in bars.groupby('_day'):
        day_path = os.path.join(tf_dir, f'{day}.parquet')
        out = group.drop(columns=['_day']).sort_values('timestamp').reset_index(drop=True)

        # Merge with existing (dedup)
        if os.path.exists(day_path):
            old = pd.read_parquet(day_path)
            out = pd.concat([old, out]).drop_duplicates(
                subset='timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)

        out.to_parquet(day_path, index=False)
        total += len(group)

    return total


def process_daily_files(tf_label, raw_dir):
    """Process daily .dbn.zst files for a TF."""
    pattern = f'*.ohlcv-{tf_label}.dbn.zst' if tf_label != '1D' else '*.ohlcv-1d.dbn.zst'
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not files:
        # Try alternate pattern
        files = sorted(glob.glob(os.path.join(raw_dir, '*.dbn.zst')))
        files = [f for f in files if 'condition' not in f]

    print(f'  {tf_label}: {len(files)} files from {raw_dir}')

    all_bars = []
    for fpath in tqdm(files, desc=f'    {tf_label}'):
        df = process_dbn_file(fpath, tf_label)
        if len(df) > 0:
            all_bars.append(df)

    if not all_bars:
        print(f'    No data for {tf_label}')
        return

    combined = pd.concat(all_bars, ignore_index=True)
    combined = combined.drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp')
    n = save_to_atlas(combined, tf_label)
    print(f'    Saved {n:,} bars')


def process_1s():
    """Process the single 1s .dbn.zst file."""
    raw_dir = PATHS['1s']
    files = sorted(glob.glob(os.path.join(raw_dir, '*.ohlcv-1s.dbn.zst')))
    if not files:
        print('  1s: no files found')
        return

    print(f'  1s: {files[0]} (this may take a while...)')
    df = process_dbn_file(files[0], '1s')
    if len(df) > 0:
        n = save_to_atlas(df, '1s')
        print(f'    Saved {n:,} bars')
    del df; gc.collect()


def main():
    print(f'REBUILD ATLAS FROM DATABENTO')
    print(f'  Output: {ATLAS_OUT}/')
    print(f'  Front-month filter: active')
    print()

    # Backup existing ATLAS, then wipe and rebuild
    import shutil
    backup = 'DATA/ATLAS_BACKUP'
    if os.path.exists(ATLAS_OUT) and not os.path.exists(backup):
        print(f'  Backing up existing ATLAS -> {backup}/')
        shutil.copytree(ATLAS_OUT, backup)

    # Only wipe TFs being rebuilt — controls (1m, 1h, 1D) are separate Databento sources
    tfs_to_wipe = []
    if TF_FILTER:
        tfs_to_wipe = [TF_FILTER]
    else:
        tfs_to_wipe = ['1s', '1m', '1h', '1D']  # all Databento sources

    for tf in tfs_to_wipe:
        tf_dir = os.path.join(ATLAS_OUT, tf)
        if os.path.exists(tf_dir):
            # Delete individual files (rmtree can fail on OneDrive/locked dirs)
            for f in glob.glob(os.path.join(tf_dir, '*.parquet')):
                os.remove(f)
            print(f'  Cleared {tf_dir}/ ({len(glob.glob(os.path.join(tf_dir, "*.parquet")))} files remain)')

    os.makedirs(ATLAS_OUT, exist_ok=True)

    tfs_to_process = []
    if TF_FILTER:
        tfs_to_process = [TF_FILTER]
    else:
        tfs_to_process = ['1m', '1h', '1D']
        if not SKIP_1S:
            tfs_to_process = ['1s'] + tfs_to_process

    for tf in tfs_to_process:
        if tf == '1s':
            process_1s()
        else:
            process_daily_files(tf, PATHS[tf])
        gc.collect()

    # Summary
    print(f'\nATLAS_DATABENTO SUMMARY:')
    for tf in ['1s', '1m', '1h', '1D']:
        tf_dir = os.path.join(ATLAS_OUT, tf)
        if os.path.exists(tf_dir):
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            total_bars = sum(len(pd.read_parquet(f)) for f in files)
            first_file = os.path.basename(files[0]) if files else '?'
            last_file = os.path.basename(files[-1]) if files else '?'
            print(f'  {tf:>4}: {len(files)} files, {total_bars:,} bars ({first_file} to {last_file})')

    print(f'\nDone. Clean ATLAS at {ATLAS_OUT}/')


if __name__ == '__main__':
    main()
