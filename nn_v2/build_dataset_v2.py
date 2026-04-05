"""
Build 79D Feature Dataset V2 — per-TF computation, no aggregator.

Computes each TF's features independently from ATLAS parquets.
No partial bars. Each TF updates only when its bar closes.
Higher TFs carry history across days naturally (loaded from ATLAS).

For each day:
  1. Load anchor TF bars for this day
  2. For each higher TF: load all bars UP TO this day (cross-day history)
  3. Run SFE on tail(300) per TF — gets states for latest bars
  4. For each anchor bar: find latest closed bar per TF, assemble 79D

Usage:
  python nn_v2/build_dataset_v2.py                          # all days, 1m anchor
  python nn_v2/build_dataset_v2.py --resolution 5s          # 5s anchor
  python nn_v2/build_dataset_v2.py --days 5                 # first 5 days
  python nn_v2/build_dataset_v2.py --start 2025-02-01       # from date
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
    extract_79d, FEATURE_NAMES_79D, N_FEATURES, TF_ORDER, TF_SECONDS,
)

ATLAS_ROOT = 'DATA/ATLAS'
OUTPUT_DIR = 'DATA/FEATURES_79D'
SFE_MIN_BARS = 21
SFE_TAIL = 300  # max bars to feed SFE


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Build 79D features V2 (per-TF, no aggregator)')
    p.add_argument('--start', type=str, default=None)
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--days', type=int, default=None)
    p.add_argument('--resolution', type=str, default='1m', choices=['5s', '15s', '1m'])
    return p.parse_args()


def get_all_days(tf: str) -> list:
    """Get sorted day names from an ATLAS TF directory."""
    d = os.path.join(ATLAS_ROOT, tf)
    files = sorted(glob.glob(os.path.join(d, '*.parquet')))
    return [os.path.basename(f).replace('.parquet', '') for f in files]


def load_tf_cumulative(tf: str, days_up_to: list) -> pd.DataFrame:
    """Load all bars for a TF for the given days, sorted by timestamp."""
    dfs = []
    for day in days_up_to:
        path = os.path.join(ATLAS_ROOT, tf, f'{day}.parquet')
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)


def process_one_day(day_name: str, anchor_tf: str, days_up_to: list,
                    sfe, prev_velocities: dict):
    """Process one day: for each anchor bar, assemble 79D from all TFs.

    Args:
        day_name: the day to produce output for
        anchor_tf: resolution TF (e.g. '1m')
        days_up_to: all days up to and including this day (for history)
        sfe: StatisticalFieldEngine instance (reused)
        prev_velocities: velocity state from previous day

    Returns:
        (DataFrame, updated prev_velocities)
    """
    # Load anchor bars for this day only
    anchor_path = os.path.join(ATLAS_ROOT, anchor_tf, f'{day_name}.parquet')
    if not os.path.exists(anchor_path):
        return pd.DataFrame(), prev_velocities

    anchor_df = pd.read_parquet(anchor_path).sort_values('timestamp').reset_index(drop=True)
    if len(anchor_df) < SFE_MIN_BARS:
        return pd.DataFrame(), prev_velocities

    anchor_ts = anchor_df['timestamp'].values

    # For each TF: load cumulative history, run SFE on tail
    tf_data = {}  # {tf: (timestamps, states, bars_df)}

    for tf in TF_ORDER:
        cumul = load_tf_cumulative(tf, days_up_to)
        if len(cumul) < SFE_MIN_BARS:
            continue

        ts_arr = cumul['timestamp'].values

        # SFE on tail
        if len(cumul) > SFE_TAIL:
            sfe_input = cumul.tail(SFE_TAIL).reset_index(drop=True)
            tail_offset = len(cumul) - SFE_TAIL
        else:
            sfe_input = cumul
            tail_offset = 0

        states = sfe.batch_compute_states(sfe_input)
        if not states:
            continue

        tf_data[tf] = {
            'timestamps': ts_arr,
            'states': states,
            'tail_offset': tail_offset,
            'bars': cumul,
        }

    if anchor_tf not in tf_data:
        return pd.DataFrame(), prev_velocities

    rows = []

    for bar_idx in range(len(anchor_df)):
        ts = anchor_ts[bar_idx]

        states_by_tf = {}
        ohlcv_by_tf = {}

        for tf, data in tf_data.items():
            tf_ts = data['timestamps']
            states = data['states']
            offset = data['tail_offset']

            # Find latest bar <= current timestamp
            idx = int(np.searchsorted(tf_ts, ts, side='right')) - 1
            if idx < 0:
                continue

            # Map to states array (states covers bars from offset onward)
            state_idx = idx - offset
            if state_idx < 0 or state_idx >= len(states):
                continue

            states_by_tf[tf] = states[state_idx]

            # OHLCV up to this bar
            ohlcv = data['bars'].iloc[:idx + 1]
            if len(ohlcv) > SFE_TAIL:
                ohlcv = ohlcv.tail(SFE_TAIL).reset_index(drop=True)
            ohlcv_by_tf[tf] = ohlcv

        if anchor_tf not in states_by_tf:
            continue

        feat, prev_velocities = extract_79d(
            states_by_tf, ohlcv_by_tf, prev_velocities, ts
        )

        rows.append({
            'timestamp': ts,
            **{name: feat[i] for i, name in enumerate(FEATURE_NAMES_79D)}
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(), prev_velocities


def main():
    args = parse_args()
    anchor_tf = args.resolution
    out_dir = f'{OUTPUT_DIR}_{anchor_tf}_v2'
    os.makedirs(out_dir, exist_ok=True)

    all_days = get_all_days(anchor_tf)
    print(f'Total days in ATLAS/{anchor_tf}: {len(all_days)}')

    # Filter
    def d2date(d):
        return d.replace('_', '-')
    if args.start:
        all_days = [d for d in all_days if d2date(d) >= args.start]
    if args.end:
        all_days = [d for d in all_days if d2date(d) <= args.end]
    if args.days:
        all_days = all_days[:args.days]

    # Skip existing
    to_build = []
    skipped = 0
    for day in all_days:
        if os.path.exists(os.path.join(out_dir, f'{day}.parquet')):
            skipped += 1
        else:
            to_build.append(day)

    print(f'To build: {len(to_build)} | Skipping: {skipped}')
    if not to_build:
        print('Nothing to build.')
        return
    print(f'  From: {to_build[0]} | To: {to_build[-1]}')
    print(f'  Anchor: {anchor_tf} | Output: {out_dir}/')
    print()

    sfe = StatisticalFieldEngine()
    prev_velocities = {}
    total_rows = 0

    for i, day_name in enumerate(tqdm(to_build, desc='Days', unit='day')):
        # All days up to and including this one (for higher TF history)
        day_idx_in_all = all_days.index(day_name) if day_name in all_days else i
        days_up_to = all_days[:day_idx_in_all + 1]

        df, prev_velocities = process_one_day(
            day_name, anchor_tf, days_up_to, sfe, prev_velocities)

        if len(df) > 0:
            out_path = os.path.join(out_dir, f'{day_name}.parquet')
            df.to_parquet(out_path, index=False)
            total_rows += len(df)

        # Periodic GC
        if i % 10 == 0:
            gc.collect()

    del sfe
    gc.collect()

    print(f'\nDone: {total_rows:,} rows across {len(to_build)} days')
    print(f'Output: {out_dir}/')

    # Verify: are 1h/1D alive?
    if to_build:
        last_path = os.path.join(out_dir, f'{to_build[-1]}.parquet')
        if os.path.exists(last_path):
            df = pd.read_parquet(last_path)
            for tf in ['1h', '1D']:
                col = f'{tf}_z_se'
                if col in df.columns:
                    v = df[col].var()
                    print(f'  {col}: var={v:.6f} {"ALIVE" if v > 1e-6 else "DEAD"}')


if __name__ == '__main__':
    main()
