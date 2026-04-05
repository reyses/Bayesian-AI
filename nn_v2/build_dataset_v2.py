"""
Build 79D Feature Dataset V2 — per-TF computation, no aggregator.

Computes each TF's features independently from ATLAS parquets.
No partial bars. Each TF updates only when its bar closes.
Higher TFs carry history across days naturally (loaded from ATLAS).

Advantages over v1:
  - Modular: rebuild one TF without touching others
  - Fast: SFE runs once per TF per day, not per-bar
  - Cross-day context: 1h/1D load from ATLAS (multi-day history built in)
  - No aggregator: simpler, fewer bugs

Output resolution is the ANCHOR TF. For each anchor bar, the 79D uses
the latest closed bar from every higher TF.

Usage:
  python nn_v2/build_dataset_v2.py                          # all days, 1m anchor
  python nn_v2/build_dataset_v2.py --resolution 5s          # all days, 5s anchor
  python nn_v2/build_dataset_v2.py --days 5                 # first 5 days
  python nn_v2/build_dataset_v2.py --start 2025-02-01       # from date
  python nn_v2/build_dataset_v2.py --tf-only 1h,1D          # rebuild only these TFs
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
    extract_tf_features, extract_79d,
    FEATURE_NAMES_79D, N_FEATURES, N_CORE, N_HELPER,
    TF_ORDER, TF_SECONDS, CORE_FEATURE_NAMES, HELPER_FEATURE_NAMES,
    CORE_START, HELPER_START, GLOBAL_START,
)

ATLAS_ROOT = 'DATA/ATLAS'
OUTPUT_DIR = 'DATA/FEATURES_79D'
SFE_MIN_BARS = 21
SFE_TAIL_LIMIT = 300  # max bars to feed SFE (speed vs accuracy tradeoff)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Build 79D features V2 (per-TF, no aggregator)')
    p.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    p.add_argument('--days', type=int, default=None, help='Process first N days only')
    p.add_argument('--resolution', type=str, default='1m',
                   choices=['5s', '15s', '1m'],
                   help='Anchor TF / output resolution (default: 1m)')
    p.add_argument('--tf-only', type=str, default=None,
                   help='Only rebuild these TFs (comma-separated, e.g. 1h,1D)')
    return p.parse_args()


def get_all_days(anchor_tf: str) -> list:
    """Get sorted list of day names from anchor TF ATLAS directory."""
    anchor_dir = os.path.join(ATLAS_ROOT, anchor_tf)
    files = sorted(glob.glob(os.path.join(anchor_dir, '*.parquet')))
    return [os.path.basename(f).replace('.parquet', '') for f in files]


def load_tf_bars(tf: str, day_names: list) -> pd.DataFrame:
    """Load ALL bars for a TF across multiple days. Sorted by timestamp."""
    dfs = []
    for day in day_names:
        path = os.path.join(ATLAS_ROOT, tf, f'{day}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)


def compute_tf_states(sfe, tf_bars: pd.DataFrame) -> list:
    """Run SFE on a TF's bars. Returns list of MarketState per bar."""
    if len(tf_bars) < SFE_MIN_BARS:
        return []
    # Use tail for speed if many bars
    if len(tf_bars) > SFE_TAIL_LIMIT:
        sfe_input = tf_bars.tail(SFE_TAIL_LIMIT).reset_index(drop=True)
        offset = len(tf_bars) - SFE_TAIL_LIMIT
    else:
        sfe_input = tf_bars
        offset = 0
    states = sfe.batch_compute_states(sfe_input)
    return states, offset


def process_days(day_names: list, anchor_tf: str, all_days_before: list):
    """Process multiple days. For each anchor bar, assemble 79D from all TFs.

    Higher TFs use accumulated history from all_days_before + current day.
    This gives 1h and 1D proper multi-day context.

    Args:
        day_names: days to process (output for these)
        anchor_tf: resolution TF (e.g. '1m')
        all_days_before: all days before the first day_name (for history)
    """
    sfe = StatisticalFieldEngine()
    prev_velocities = {}

    # Pre-load higher TF history (all days up to and including current batch)
    all_days = all_days_before + day_names
    tf_all_bars = {}
    tf_all_states = {}
    tf_all_ts = {}

    print(f'  Loading TF data...')
    for tf in TF_ORDER:
        bars = load_tf_bars(tf, all_days)
        if len(bars) == 0:
            continue
        tf_all_bars[tf] = bars
        tf_all_ts[tf] = bars['timestamp'].values

        # Run SFE on full history
        if len(bars) >= SFE_MIN_BARS:
            states, offset = compute_tf_states(sfe, bars)
            if states:
                tf_all_states[tf] = (states, offset)

    results = {}  # {day_name: DataFrame}

    for day_name in day_names:
        # Load anchor bars for this day
        anchor_path = os.path.join(ATLAS_ROOT, anchor_tf, f'{day_name}.parquet')
        if not os.path.exists(anchor_path):
            continue
        anchor_df = pd.read_parquet(anchor_path).sort_values('timestamp').reset_index(drop=True)
        if len(anchor_df) == 0:
            continue

        rows = []

        for bar_idx in range(len(anchor_df)):
            ts = anchor_df.iloc[bar_idx]['timestamp']

            # For each TF: find the latest closed bar AT OR BEFORE this timestamp
            states_by_tf = {}
            ohlcv_by_tf = {}

            for tf in TF_ORDER:
                if tf not in tf_all_ts or tf not in tf_all_states:
                    continue

                tf_ts = tf_all_ts[tf]
                states, offset = tf_all_states[tf]

                # Find latest bar <= current timestamp
                tf_bar_idx = int(np.searchsorted(tf_ts, ts, side='right')) - 1
                if tf_bar_idx < 0:
                    continue

                # Map to states index (states may be offset from tail)
                state_idx = tf_bar_idx - offset
                if state_idx < 0 or state_idx >= len(states):
                    continue

                states_by_tf[tf] = states[state_idx]

                # OHLCV up to this bar (for variance_ratio, vol_rel, etc.)
                ohlcv_slice = tf_all_bars[tf].iloc[:tf_bar_idx + 1]
                # Limit to tail for speed
                if len(ohlcv_slice) > SFE_TAIL_LIMIT:
                    ohlcv_slice = ohlcv_slice.tail(SFE_TAIL_LIMIT).reset_index(drop=True)
                ohlcv_by_tf[tf] = ohlcv_slice

            # Need at least the anchor TF
            if anchor_tf not in states_by_tf:
                continue

            # Extract 79D
            feat, prev_velocities = extract_79d(
                states_by_tf, ohlcv_by_tf, prev_velocities, ts
            )

            rows.append({
                'timestamp': ts,
                **{name: feat[i] for i, name in enumerate(FEATURE_NAMES_79D)}
            })

        if rows:
            results[day_name] = pd.DataFrame(rows)

    del sfe
    gc.collect()
    return results


def main():
    args = parse_args()
    anchor_tf = args.resolution
    out_dir = f'{OUTPUT_DIR}_{anchor_tf}_v2'
    os.makedirs(out_dir, exist_ok=True)

    # Get all days from anchor TF
    all_days = get_all_days(anchor_tf)
    print(f'Total days in ATLAS/{anchor_tf}: {len(all_days)}')

    # Filter
    def day_to_date(d):
        return d.replace('_', '-')

    if args.start:
        all_days = [d for d in all_days if day_to_date(d) >= args.start]
    if args.end:
        all_days = [d for d in all_days if day_to_date(d) <= args.end]
    if args.days:
        all_days = all_days[:args.days]

    # Skip already built
    to_build = []
    skipped = 0
    for day in all_days:
        out_path = os.path.join(out_dir, f'{day}.parquet')
        if os.path.exists(out_path):
            skipped += 1
        else:
            to_build.append(day)

    print(f'To build: {len(to_build)} days (skipping {skipped} existing)')
    if not to_build:
        print('Nothing to build.')
        return

    print(f'  From: {to_build[0]}')
    print(f'  To:   {to_build[-1]}')
    print(f'  Anchor TF: {anchor_tf}')
    print(f'  Output: {out_dir}/')
    print()

    # Process in batches (to manage memory for higher TF history)
    BATCH_SIZE = 20  # days per batch
    total_rows = 0

    for batch_start in tqdm(range(0, len(to_build), BATCH_SIZE),
                            desc='Batches', unit='batch'):
        batch_days = to_build[batch_start:batch_start + BATCH_SIZE]

        # All days before this batch (for higher TF history)
        first_day_idx = all_days.index(batch_days[0]) if batch_days[0] in all_days else 0
        history_days = all_days[:first_day_idx]

        results = process_days(batch_days, anchor_tf, history_days)

        for day_name, df in results.items():
            out_path = os.path.join(out_dir, f'{day_name}.parquet')
            df.to_parquet(out_path, index=False)
            total_rows += len(df)

        gc.collect()

    # Verify 1h/1D are alive
    print(f'\nDone: {total_rows:,} rows across {len(to_build)} days')
    print(f'Output: {out_dir}/')

    # Quick check: are 1D features alive?
    if to_build:
        last_day = to_build[-1]
        last_path = os.path.join(out_dir, f'{last_day}.parquet')
        if os.path.exists(last_path):
            df = pd.read_parquet(last_path)
            for tf in ['1h', '1D']:
                z_col = f'{tf}_z_se'
                if z_col in df.columns:
                    var = df[z_col].var()
                    mean = df[z_col].mean()
                    print(f'  {z_col}: mean={mean:.4f}, var={var:.4f} '
                          f'{"ALIVE" if var > 1e-6 else "DEAD"}')


if __name__ == '__main__':
    main()
