"""
Convert NT8 history dump (CSV) to ATLAS parquet.

Reads:  DATA/ATLAS_NT8/5s/{contract}/*.csv
Writes: DATA/ATLAS/5s/YYYY_MM_DD.parquet

Handles:
  - Multiple contracts (stitch at rollover — newer contract wins)
  - Gap detection and reporting
  - Duplicate bar removal
  - OHLC sanity checks
  - Schema validation (timestamp int64, OHLC float64, volume uint64)

Usage:
    python tools/convert_nt8_atlas.py                    # convert all
    python tools/convert_nt8_atlas.py --contract MNQ_06-26
    python tools/convert_nt8_atlas.py --validate-only    # report without writing
    python tools/convert_nt8_atlas.py --backup           # backup existing ATLAS/5s first
"""
import os
import sys
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

NT8_ROOT = 'DATA/ATLAS_NT8/5s'
ATLAS_OUT = 'DATA/ATLAS_NT8/5s'  # Output stays in NT8 dataset — never overwrites Databento
ATLAS_BACKUP = 'DATA/ATLAS_NT8_5s_BACKUP'

# Expected bars per full trading day (23h session at 5s = ~16,560)
MIN_BARS_PER_DAY = 10000
BAR_PERIOD_S = 5
MAX_GAP_S = 30  # flag gaps > 30s during session


def parse_args():
    p = argparse.ArgumentParser(description='Convert NT8 CSV dump to ATLAS parquet')
    p.add_argument('--contract', type=str, default=None,
                   help='Process only this contract (e.g. MNQ_06-26)')
    p.add_argument('--validate-only', action='store_true',
                   help='Report validation without writing parquets')
    p.add_argument('--backup', action='store_true',
                   help='Backup existing ATLAS/5s before overwriting')
    return p.parse_args()


def discover_contracts():
    """Find all contract subdirectories in NT8_ROOT, sorted chronologically."""
    if not os.path.exists(NT8_ROOT):
        print(f'ERROR: {NT8_ROOT}/ does not exist. Run NT8 history dump first.')
        return []

    contracts = []
    for d in sorted(os.listdir(NT8_ROOT)):
        full = os.path.join(NT8_ROOT, d)
        if os.path.isdir(full):
            csvs = glob.glob(os.path.join(full, '*.csv'))
            if csvs:
                contracts.append(d)

    return contracts


def load_contract(contract_dir):
    """Load all CSVs for a contract into a single DataFrame."""
    path = os.path.join(NT8_ROOT, contract_dir)
    csvs = sorted(glob.glob(os.path.join(path, '*.csv')))

    if not csvs:
        print(f'  WARNING: No CSVs in {path}/')
        return pd.DataFrame()

    dfs = []
    for csv_path in tqdm(csvs, desc=f'  Reading {contract_dir}', unit='file', leave=False):
        df = pd.read_csv(csv_path)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full = full.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)

    print(f'  {contract_dir}: {len(full):,} bars from {len(csvs)} files')
    return full


def stitch_contracts(contract_dfs):
    """Stitch multiple contracts — newer contract wins on overlap days.

    Args:
        contract_dfs: list of (contract_name, DataFrame) sorted chronologically

    Returns:
        Single stitched DataFrame
    """
    if len(contract_dfs) == 1:
        return contract_dfs[0][1]

    print(f'\n  Stitching {len(contract_dfs)} contracts...')

    # Union all bars, tag with contract
    parts = []
    for i, (name, df) in enumerate(contract_dfs):
        df = df.copy()
        df['_contract_idx'] = i  # higher index = newer contract
        parts.append(df)
        ts_min = datetime.utcfromtimestamp(df['timestamp'].min()).strftime('%Y-%m-%d')
        ts_max = datetime.utcfromtimestamp(df['timestamp'].max()).strftime('%Y-%m-%d')
        print(f'    {name}: {ts_min} to {ts_max} ({len(df):,} bars)')

    combined = pd.concat(parts, ignore_index=True)
    # On duplicate timestamps, keep the newer contract (higher _contract_idx)
    combined = (combined.sort_values(['timestamp', '_contract_idx'])
                .drop_duplicates(subset='timestamp', keep='last')
                .drop(columns=['_contract_idx'])
                .sort_values('timestamp')
                .reset_index(drop=True))

    print(f'    Stitched: {len(combined):,} bars')
    return combined


def validate(df):
    """Validate the stitched data. Returns (issues_list, stats_dict)."""
    issues = []
    stats = {
        'total_bars': len(df),
        'total_days': 0,
        'short_days': 0,
        'gaps_found': 0,
        'ohlc_violations': 0,
    }

    if len(df) == 0:
        issues.append('EMPTY: No bars to validate')
        return issues, stats

    # Group by day
    df['_day'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y_%m_%d')
    days = df.groupby('_day')
    stats['total_days'] = len(days)

    for day_name, day_df in days:
        n = len(day_df)

        # Short day check
        if n < MIN_BARS_PER_DAY:
            stats['short_days'] += 1
            issues.append(f'SHORT: {day_name} has {n} bars (min {MIN_BARS_PER_DAY})')

        # Gap check (within day)
        ts = day_df['timestamp'].values
        diffs = np.diff(ts)
        big_gaps = diffs[diffs > MAX_GAP_S]
        if len(big_gaps) > 0:
            stats['gaps_found'] += len(big_gaps)
            max_gap = big_gaps.max()
            issues.append(f'GAP: {day_name} has {len(big_gaps)} gaps > {MAX_GAP_S}s (max {max_gap:.0f}s)')

        # OHLC sanity
        h = day_df['high'].values
        l = day_df['low'].values
        o = day_df['open'].values
        c = day_df['close'].values
        bad = ((h < l) | (h < o) | (h < c) | (l > o) | (l > c)).sum()
        if bad > 0:
            stats['ohlc_violations'] += bad
            issues.append(f'OHLC: {day_name} has {bad} bars with high < low or similar')

    df.drop(columns=['_day'], inplace=True)
    return issues, stats


def write_atlas(df, validate_only=False):
    """Write daily parquet files to ATLAS/5s/."""
    if validate_only:
        return

    os.makedirs(ATLAS_OUT, exist_ok=True)

    df['_day'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y_%m_%d')
    days = df.groupby('_day')
    written = 0

    for day_name, day_df in tqdm(days, desc='Writing parquets', unit='day'):
        out_df = day_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Enforce ATLAS schema
        out_df['timestamp'] = out_df['timestamp'].astype(np.int64)
        out_df['open'] = out_df['open'].astype(np.float64)
        out_df['high'] = out_df['high'].astype(np.float64)
        out_df['low'] = out_df['low'].astype(np.float64)
        out_df['close'] = out_df['close'].astype(np.float64)
        out_df['volume'] = out_df['volume'].astype(np.uint64)

        out_df = out_df.sort_values('timestamp').reset_index(drop=True)

        out_path = os.path.join(ATLAS_OUT, f'{day_name}.parquet')
        out_df.to_parquet(out_path, index=False)
        written += 1

    df.drop(columns=['_day'], inplace=True)
    print(f'  Written: {written} daily parquets to {ATLAS_OUT}/')


def main():
    args = parse_args()

    print(f'{"="*60}')
    print(f'NT8 CSV -> ATLAS PARQUET CONVERTER')
    print(f'{"="*60}')

    # Discover contracts
    contracts = discover_contracts()
    if not contracts:
        return

    if args.contract:
        contracts = [c for c in contracts if c == args.contract]
        if not contracts:
            print(f'Contract "{args.contract}" not found in {NT8_ROOT}/')
            return

    print(f'Contracts found: {contracts}')

    # Load all contracts
    contract_dfs = []
    for contract in contracts:
        df = load_contract(contract)
        if len(df) > 0:
            contract_dfs.append((contract, df))

    if not contract_dfs:
        print('No data loaded.')
        return

    # Stitch contracts
    stitched = stitch_contracts(contract_dfs)

    # Validate
    print(f'\nValidating...')
    issues, stats = validate(stitched)

    print(f'\n  Total bars:   {stats["total_bars"]:,}')
    print(f'  Total days:   {stats["total_days"]}')
    print(f'  Short days:   {stats["short_days"]}')
    print(f'  Gaps > {MAX_GAP_S}s:  {stats["gaps_found"]}')
    print(f'  OHLC issues:  {stats["ohlc_violations"]}')

    if issues:
        print(f'\n  Issues ({len(issues)}):')
        for issue in issues[:20]:
            print(f'    {issue}')
        if len(issues) > 20:
            print(f'    ... and {len(issues) - 20} more')

    if args.validate_only:
        print(f'\n  --validate-only: No files written.')
        return

    # Backup existing ATLAS/5s
    if args.backup and os.path.exists(ATLAS_OUT):
        existing = glob.glob(os.path.join(ATLAS_OUT, '*.parquet'))
        if existing:
            os.makedirs(ATLAS_BACKUP, exist_ok=True)
            for f in existing:
                shutil.copy2(f, os.path.join(ATLAS_BACKUP, os.path.basename(f)))
            print(f'\n  Backed up {len(existing)} files to {ATLAS_BACKUP}/')

    # Write
    print(f'\nWriting ATLAS parquets...')
    write_atlas(stitched)

    print(f'\n{"="*60}')
    print(f'DONE')
    print(f'  Next steps:')
    print(f'    rm -rf DATA/FEATURES_79D_5s/*')
    print(f'    python training/build_dataset.py --resolution 5s')
    print(f'    python training/run.py blended')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
