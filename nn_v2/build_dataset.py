"""
Build 79D Feature Dataset — one-time computation, run overnight.

Processes all 1s bars (Jan 2025 → Feb 2026) through the ticker + aggregator.
At every 5s close: computes 79D features from closed + partial bars.
Saves to DATA/FEATURES_79D/YYYY_MM_DD.parquet

Each row: timestamp + 79D features (80 columns)
~2.5M rows total across 311 days.

After this runs once, no SFE recomputation needed — everything reads from disk.

Usage:
  python nn_v2/build_dataset.py                    # all days
  python nn_v2/build_dataset.py --start 2025-06-01 # from date
  python nn_v2/build_dataset.py --days 5           # first N days
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.ticker import FileTicker
from nn_v2.aggregator import Aggregator, TF_ORDER
from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import (
    extract_79d, FEATURE_NAMES_79D, N_FEATURES, TF_ORDER as FEAT_TF_ORDER
)

# Paths
ATLAS_1S = 'DATA/ATLAS/1s'
OUTPUT_DIR = 'DATA/FEATURES_79D'

# SFE needs this many bars minimum
SFE_MIN_BARS = 21

# Compute 79D at this resolution
COMPUTE_TF = '5s'  # every 5s close


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Build 79D feature dataset (one-time)')
    p.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    p.add_argument('--days', type=int, default=None, help='Process first N days only')
    return p.parse_args()


def compute_79d_from_aggregator(agg: Aggregator, sfe: StatisticalFieldEngine,
                                  prev_velocities: dict) -> tuple:
    """Compute 79D features from aggregator's current state.

    Uses closed bars + partial bars for each TF.
    Returns (features_79d, updated_prev_velocities) or (None, prev_velocities) if not enough data.
    """
    states_by_tf = {}
    ohlcv_by_tf = {}

    for tf in FEAT_TF_ORDER:
        # Get closed bars for this TF
        closed_df = agg.get_closed_bars_df(tf)

        # Add partial bar if available
        partial = agg.get_partial_bar(tf)
        if partial is not None:
            partial_df = pd.DataFrame([partial])
            if len(closed_df) > 0:
                full_df = pd.concat([closed_df, partial_df], ignore_index=True)
            else:
                full_df = partial_df
        else:
            full_df = closed_df

        if len(full_df) < SFE_MIN_BARS:
            continue

        ohlcv_by_tf[tf] = full_df

        # SFE on tail (300 bars max for speed — acceptable for dataset build)
        sfe_input = full_df.tail(300).reset_index(drop=True) if len(full_df) > 300 else full_df
        states = sfe.batch_compute_states(sfe_input)
        if states:
            states_by_tf[tf] = states[-1]

    if '1m' not in states_by_tf:
        return None, prev_velocities

    # Get timestamp from latest 5s bar
    bars_5s = agg.get_closed_bars('5s')
    ts = bars_5s[-1]['timestamp'] if bars_5s else 0

    feat, prev_velocities = extract_79d(
        states_by_tf, ohlcv_by_tf, prev_velocities, ts
    )
    return feat, prev_velocities


def process_day(day_file: str) -> pd.DataFrame:
    """Process one day of 1s bars → 79D features at 5s resolution."""
    sfe = StatisticalFieldEngine()
    agg = Aggregator(history_limit=2000)
    prev_velocities = {}

    rows = []
    last_5s_count = 0

    # Track which TF triggers the 79D computation
    def on_bar_close(tf, bar):
        nonlocal last_5s_count, prev_velocities

        if tf == COMPUTE_TF:
            # 5s bar closed → compute 79D
            feat, prev_velocities = compute_79d_from_aggregator(agg, sfe, prev_velocities)
            if feat is not None:
                rows.append({
                    'timestamp': bar['timestamp'],
                    **{name: feat[i] for i, name in enumerate(FEATURE_NAMES_79D)}
                })

    agg.on_bar_close = on_bar_close

    # Feed all 1s bars
    ticker = FileTicker(day_file)
    for bar in ticker:
        agg.feed(bar)

    del sfe
    gc.collect()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all 1s files
    all_files = sorted(glob.glob(os.path.join(ATLAS_1S, '*.parquet')))
    print(f'Total 1s files: {len(all_files)}')

    # Filter by date
    def file_date(f):
        return os.path.basename(f).replace('.parquet', '').replace('_', '-')

    if args.start:
        all_files = [f for f in all_files if file_date(f) >= args.start]
    if args.end:
        all_files = [f for f in all_files if file_date(f) <= args.end]
    if args.days:
        all_files = all_files[:args.days]

    print(f'Processing: {len(all_files)} days')
    if all_files:
        print(f'  From: {file_date(all_files[0])}')
        print(f'  To:   {file_date(all_files[-1])}')
    print(f'  Output: {OUTPUT_DIR}/')
    print(f'  Resolution: every {COMPUTE_TF} close')
    print()

    total_rows = 0
    skipped = 0

    for day_file in tqdm(all_files, desc='Days', unit='day'):
        day_name = os.path.basename(day_file).replace('.parquet', '')
        out_path = os.path.join(OUTPUT_DIR, f'{day_name}.parquet')

        # Skip if already computed
        if os.path.exists(out_path):
            skipped += 1
            continue

        df = process_day(day_file)
        if len(df) == 0:
            continue

        df.to_parquet(out_path, index=False)
        total_rows += len(df)

        gc.collect()

    print(f'\nDone: {total_rows:,} rows across {len(all_files) - skipped} days')
    print(f'Skipped (already exists): {skipped}')
    print(f'Output: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
