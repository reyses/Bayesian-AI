"""
Build 79D Feature Dataset — bulk GPU processing, no lookahead protection needed.

This is TRAINING DATA preprocessing. It loads full days, runs SFE in batch on GPU,
and extracts 79D features for every bar. Fast because there's no sequential constraint.

The zero-lookahead protection is in the LIVE pipeline (ticker → aggregator).
Here we're just labeling history — seeing the whole day is fine.

Output: DATA/FEATURES_79D_1m/YYYY_MM_DD.parquet (at 1m resolution)
        DATA/FEATURES_79D/YYYY_MM_DD.parquet    (at 5s resolution)

Usage:
  python nn_v2/build_dataset.py                         # all days, 1m
  python nn_v2/build_dataset.py --resolution 5s         # all days, 5s
  python nn_v2/build_dataset.py --days 5                # first 5 days
  python nn_v2/build_dataset.py --start 2026-01-01      # from date
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import (
    extract_79d, aggregate_partial_bar,
    FEATURE_NAMES_79D, N_FEATURES, TF_ORDER, TF_SECONDS,
)

# Paths
ATLAS_ROOT = 'DATA/ATLAS'
OUTPUT_DIR = 'DATA/FEATURES_79D'
SFE_MIN_BARS = 21


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Build 79D feature dataset (bulk GPU)')
    p.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    p.add_argument('--days', type=int, default=None, help='Process first N days only')
    p.add_argument('--resolution', type=str, default='1m', choices=['5s', '15s', '1m'],
                   help='Output resolution (default: 1m)')
    return p.parse_args()


def load_tf_data(day_name: str, tf: str) -> pd.DataFrame:
    """Load a TF's parquet for a given day."""
    path = os.path.join(ATLAS_ROOT, tf, f'{day_name}.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    return pd.DataFrame()


def process_day_bulk(day_name: str, resolution: str = '1m') -> pd.DataFrame:
    """Process one day in bulk — load all TFs, run SFE once per TF on GPU, extract 79D.

    No sequential bar feeding. Full day loaded at once. GPU batch processing.
    """
    sfe = StatisticalFieldEngine()

    # Load the anchor TF (what we output one row per bar for)
    anchor_df = load_tf_data(day_name, resolution)
    if len(anchor_df) < SFE_MIN_BARS:
        return pd.DataFrame()

    anchor_ts = anchor_df['timestamp'].values
    n_bars = len(anchor_df)

    # Load and run SFE on each TF — ONE batch call per TF (GPU)
    all_states = {}   # {tf: list of state dicts}
    all_ohlcv = {}    # {tf: DataFrame}
    tf_timestamps = {}  # {tf: np.array of timestamps}

    for tf in TF_ORDER:
        df = load_tf_data(day_name, tf)
        if len(df) < SFE_MIN_BARS:
            continue

        all_ohlcv[tf] = df
        tf_timestamps[tf] = df['timestamp'].values

        # Single GPU batch call for the entire day
        states = sfe.batch_compute_states(df)
        if states:
            all_states[tf] = states

    if '1m' not in all_states:
        del sfe
        return pd.DataFrame()

    # Extract 79D for each anchor bar
    rows = []
    prev_velocities = {}

    for bar_idx in range(n_bars):
        ts = anchor_ts[bar_idx]

        # For each TF, find the state at this timestamp
        states_this_bar = {}
        for tf in TF_ORDER:
            if tf not in all_states:
                continue
            tf_states = all_states[tf]
            tf_ts = tf_timestamps[tf]

            if tf == resolution:
                # Direct index
                if bar_idx < len(tf_states):
                    states_this_bar[tf] = tf_states[bar_idx]
            else:
                # Align: latest TF bar <= anchor timestamp
                tf_bar_idx = np.searchsorted(tf_ts, ts, side='right') - 1
                if 0 <= tf_bar_idx < len(tf_states):
                    states_this_bar[tf] = tf_states[tf_bar_idx]

        # Extract 79D
        feat, prev_velocities = extract_79d(
            states_this_bar, all_ohlcv, prev_velocities, ts
        )

        rows.append({
            'timestamp': ts,
            **{name: feat[i] for i, name in enumerate(FEATURE_NAMES_79D)}
        })

    del sfe, all_states
    gc.collect()

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    resolution = args.resolution
    out_dir = OUTPUT_DIR if resolution == '5s' else f'{OUTPUT_DIR}_{resolution}'
    os.makedirs(out_dir, exist_ok=True)

    # Find all days from 1m (the anchor)
    anchor_dir = os.path.join(ATLAS_ROOT, resolution)
    all_files = sorted(glob.glob(os.path.join(anchor_dir, '*.parquet')))
    print(f'Total {resolution} files: {len(all_files)}')

    # Filter
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
    print(f'  Output: {out_dir}/')
    print(f'  Resolution: {resolution}')
    print(f'  Mode: BULK GPU (no sequential constraint)')
    print()

    total_rows = 0
    skipped = 0

    for fpath in tqdm(all_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        out_path = os.path.join(out_dir, f'{day_name}.parquet')

        if os.path.exists(out_path):
            skipped += 1
            continue

        df = process_day_bulk(day_name, resolution=resolution)
        if len(df) == 0:
            continue

        df.to_parquet(out_path, index=False)
        total_rows += len(df)

        gc.collect()

    print(f'\nDone: {total_rows:,} rows across {len(all_files) - skipped} days')
    print(f'Skipped (already exists): {skipped}')
    print(f'Output: {out_dir}/')


if __name__ == '__main__':
    main()
