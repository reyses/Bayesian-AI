#!/usr/bin/env python3
"""
ATLAS Monthly-to-Daily Splitter
Converts existing DATA/ATLAS/{tf}/YYYY_MM.parquet monthly files into
per-trading-day YYYYMMDD.parquet files, in-place.

Run this once after an atlas build that used the monthly writer format.
Subsequent atlas builds will use the daily writer format directly.

Usage:
    python scripts/split_atlas_to_daily.py
    python scripts/split_atlas_to_daily.py --atlas DATA/ATLAS --dry-run
"""

import os
import sys
import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Pattern that matches monthly files: YYYY_MM.parquet (7 char stem)
_MONTHLY_RE = re.compile(r'^\d{4}_\d{2}$')

# Pattern that matches already-split daily files: YYYYMMDD.parquet (8 char stem)
_DAILY_RE = re.compile(r'^\d{8}$')

# Minimum bars in a trading day to be worth writing (avoids empty holiday files)
_MIN_BARS_PER_DAY = 10


def split_monthly_file(monthly_path: Path, dry_run: bool = False) -> int:
    """
    Reads one monthly parquet file, splits by UTC trading day, and writes
    per-day files alongside the monthly file (same directory).

    Returns number of day files written.
    """
    df = pd.read_parquet(monthly_path)

    if df.empty:
        print(f"    SKIP (empty): {monthly_path.name}")
        return 0

    # Ensure timestamp column is present and numeric (Unix seconds)
    if 'timestamp' not in df.columns:
        print(f"    SKIP (no timestamp): {monthly_path.name}")
        return 0

    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9
    elif not pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Extract UTC date for each bar
    dates = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.date

    out_dir = monthly_path.parent
    n_written = 0

    for day_date, day_df in df.groupby(dates):
        if len(day_df) < _MIN_BARS_PER_DAY:
            continue   # skip holiday stubs / partial days with no data

        day_str = day_date.strftime('%Y%m%d')
        out_path = out_dir / f"{day_str}.parquet"

        if out_path.exists():
            # Already split — skip to avoid clobbering existing daily files
            continue

        if not dry_run:
            day_df = day_df.reset_index(drop=True)
            table = pa.Table.from_pandas(day_df, preserve_index=False)
            pq.write_table(table, out_path, compression='snappy')

        n_written += 1

    return n_written


def main():
    parser = argparse.ArgumentParser(
        description="Split monthly ATLAS parquet files into per-day files."
    )
    parser.add_argument(
        '--atlas', default=os.path.join('DATA', 'ATLAS'),
        help="Path to ATLAS root directory (default: DATA/ATLAS)"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Show what would be done without writing any files"
    )
    parser.add_argument(
        '--delete-monthly', action='store_true',
        help="Delete monthly files after splitting (use only after verifying output)"
    )
    args = parser.parse_args()

    atlas_root = Path(args.atlas)
    if not atlas_root.exists():
        print(f"ERROR: ATLAS directory not found: {atlas_root}")
        sys.exit(1)

    tf_dirs = sorted([d for d in atlas_root.iterdir() if d.is_dir()])
    if not tf_dirs:
        print(f"ERROR: No timeframe subdirectories found in {atlas_root}")
        sys.exit(1)

    print(f"ATLAS Monthly-to-Daily Splitter")
    print(f"  ATLAS root:     {atlas_root}")
    print(f"  Dry run:        {args.dry_run}")
    print(f"  Delete monthly: {args.delete_monthly}")
    print()

    total_monthly   = 0
    total_days_out  = 0
    monthly_to_del  = []

    for tf_dir in tf_dirs:
        tf_name = tf_dir.name
        monthly_files = sorted([
            f for f in tf_dir.glob('*.parquet')
            if _MONTHLY_RE.match(f.stem)
        ])

        if not monthly_files:
            continue

        print(f"  [{tf_name}]  {len(monthly_files)} monthly files")

        for mf in monthly_files:
            n = split_monthly_file(mf, dry_run=args.dry_run)
            if n:
                action = "(dry-run)" if args.dry_run else "written"
                print(f"    {mf.name}  ->  {n} day files {action}")
            else:
                print(f"    {mf.name}  ->  already split / empty, skipped")

            total_monthly += 1
            total_days_out += n
            if n > 0:
                monthly_to_del.append(mf)

    print()
    print(f"Summary: {total_monthly} monthly files processed, {total_days_out} day files {'would be ' if args.dry_run else ''}written")

    if args.delete_monthly and not args.dry_run and monthly_to_del:
        print(f"\nDeleting {len(monthly_to_del)} monthly source files...")
        for mf in monthly_to_del:
            mf.unlink()
            print(f"  Deleted: {mf}")
    elif args.delete_monthly:
        print("  (--dry-run active: no files deleted)")

    print("\nDone.")


if __name__ == '__main__':
    main()
