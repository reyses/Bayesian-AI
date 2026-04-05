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
    p.add_argument('--resolution', type=str, default='1m', choices=['1s', '5s', '15s', '1m'],
                   help='Output resolution (default: 1m)')
    p.add_argument('--sequential', action='store_true',
                   help='Sheet music mode: sequential 1s through aggregator (honest, slow)')
    return p.parse_args()


def load_tf_data(day_name: str, tf: str) -> pd.DataFrame:
    """Load a TF's parquet for a given day."""
    path = os.path.join(ATLAS_ROOT, tf, f'{day_name}.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    return pd.DataFrame()


def process_day_bulk(day_name: str, resolution: str = '1m') -> pd.DataFrame:
    """Process one day in bulk with partial higher TF bars.

    1. Load anchor TF (1m) bars for the day
    2. For each anchor bar, aggregate 1m bars up to that point into higher TFs
       (partial bars — same as live would see at that moment)
    3. Run SFE on each TF's accumulated bars, extract 79D

    Higher TFs are re-aggregated at each step but SFE is only called when
    the TF has a new bar (cached otherwise). Fast: ~6 SFE calls per TF per day
    for higher TFs, 1 call per bar for anchor TF.
    """
    sfe = StatisticalFieldEngine()

    # Load anchor TF (also used to build higher TFs via aggregation)
    anchor_df = load_tf_data(day_name, resolution)
    if len(anchor_df) < SFE_MIN_BARS:
        return pd.DataFrame()

    # Also load sub-anchor TFs directly if available (15s, 5s)
    sub_tfs = {}
    for tf in ['15s', '5s']:
        df = load_tf_data(day_name, tf)
        if len(df) >= SFE_MIN_BARS:
            sub_tfs[tf] = df

    anchor_ts = anchor_df['timestamp'].values
    n_bars = len(anchor_df)

    # Run SFE on full anchor TF once — all states at once
    anchor_states = sfe.batch_compute_states(anchor_df)
    if not anchor_states:
        del sfe
        return pd.DataFrame()

    # For sub-anchor TFs, run SFE once too
    sub_states = {}
    sub_ts = {}
    for tf, df in sub_tfs.items():
        states = sfe.batch_compute_states(df)
        if states:
            sub_states[tf] = states
            sub_ts[tf] = df['timestamp'].values

    # For higher TFs: aggregate from anchor bars with partials
    # Cache SFE results per TF — only recompute when bar count changes
    higher_tf_cache = {}   # {tf: (n_bars_at_compute, states_list)}
    higher_tfs = ['5m', '15m', '1h', '1D']

    rows = []
    prev_velocities = {}

    for bar_idx in range(n_bars):
        ts = anchor_ts[bar_idx]

        # Anchor bars up to this point (partial TF = all bars through current)
        bars_so_far = anchor_df.iloc[:bar_idx + 1]

        # Build states_by_tf for this bar
        states_this_bar = {}
        ohlcv_this_bar = {}

        # Anchor TF state (direct index from pre-computed)
        states_this_bar[resolution] = anchor_states[bar_idx]
        ohlcv_this_bar[resolution] = bars_so_far

        # Sub-anchor TFs (aligned by timestamp)
        for tf in sub_tfs:
            if tf in sub_states and tf in sub_ts:
                tf_bar_idx = np.searchsorted(sub_ts[tf], ts, side='right') - 1
                if 0 <= tf_bar_idx < len(sub_states[tf]):
                    states_this_bar[tf] = sub_states[tf][tf_bar_idx]
                    ohlcv_this_bar[tf] = sub_tfs[tf].iloc[:tf_bar_idx + 1]

        # Higher TFs: aggregate from anchor bars up to this point
        for tf in higher_tfs:
            tf_sec = TF_SECONDS[tf]
            tf_bars = aggregate_partial_bar(bars_so_far, tf_sec)
            n_tf = len(tf_bars)

            if n_tf < SFE_MIN_BARS:
                continue

            ohlcv_this_bar[tf] = tf_bars

            # Only rerun SFE if this TF got a new bar
            cached_n, cached_states = higher_tf_cache.get(tf, (0, None))
            if n_tf != cached_n:
                tf_states = sfe.batch_compute_states(tf_bars)
                if tf_states:
                    higher_tf_cache[tf] = (n_tf, tf_states)
                    states_this_bar[tf] = tf_states[-1]
            elif cached_states is not None:
                states_this_bar[tf] = cached_states[-1]

        # Extract 79D
        feat, prev_velocities = extract_79d(
            states_this_bar, ohlcv_this_bar, prev_velocities, ts
        )

        rows.append({
            'timestamp': ts,
            **{name: feat[i] for i, name in enumerate(FEATURE_NAMES_79D)}
        })

    del sfe, anchor_states, sub_states, higher_tf_cache
    gc.collect()

    return pd.DataFrame(rows)


def process_day_sequential(day_name: str, resolution: str = '1m',
                           agg=None, sfe=None, prev_velocities=None):
    """Sheet music mode: feed 1s bars through aggregator, compute 79D honestly.

    Zero lookahead. Each bar only sees data up to that point.
    Slow (~9 min/day) but honest — same as live would produce.

    If agg/sfe/prev_velocities are passed, CARRIES HISTORY from previous days.
    This is critical for 1h and 1D features which need multi-day context.

    Returns: (DataFrame of rows, agg, sfe, prev_velocities) for chaining.
    """
    from nn_v2.ticker import FileTicker
    from nn_v2.aggregator import Aggregator

    # Create fresh if not carrying from previous day
    if agg is None:
        agg = Aggregator(history_limit=2000)
    if sfe is None:
        sfe = StatisticalFieldEngine()
    if prev_velocities is None:
        prev_velocities = {}

    rows = []

    # Load 1s bars for this day
    path_1s = os.path.join(ATLAS_ROOT, '1s', f'{day_name}.parquet')
    if not os.path.exists(path_1s):
        return pd.DataFrame(), agg, sfe, prev_velocities

    def on_bar_close(tf, bar):
        nonlocal prev_velocities
        if tf != resolution:
            return

        # Compute 79D from aggregator's accumulated state
        states_by_tf = {}
        ohlcv_by_tf = {}
        for _tf in TF_ORDER:
            df = agg.get_closed_bars_df(_tf)
            partial = agg.get_partial_bar(_tf)
            if partial is not None:
                partial_df = pd.DataFrame([partial])
                full_df = pd.concat([df, partial_df], ignore_index=True) if len(df) > 0 else partial_df
            else:
                full_df = df
            if len(full_df) < SFE_MIN_BARS:
                continue
            ohlcv_by_tf[_tf] = full_df
            sfe_input = full_df.tail(300).reset_index(drop=True) if len(full_df) > 300 else full_df
            states = sfe.batch_compute_states(sfe_input)
            if states:
                states_by_tf[_tf] = states[-1]

        if '1m' not in states_by_tf:
            return

        feat, prev_velocities = extract_79d(
            states_by_tf, ohlcv_by_tf, prev_velocities, bar['timestamp']
        )
        rows.append({
            'timestamp': bar['timestamp'],
            **{name: feat[i] for i, name in enumerate(FEATURE_NAMES_79D)}
        })

    agg.on_bar_close = on_bar_close
    ticker = FileTicker(path_1s)
    for bar in ticker:
        agg.feed(bar)

    return pd.DataFrame(rows) if rows else pd.DataFrame(), agg, sfe, prev_velocities


def main():
    args = parse_args()
    resolution = args.resolution
    sequential = args.sequential

    if sequential:
        out_dir = f'{OUTPUT_DIR}_{resolution}_seq'
    else:
        out_dir = OUTPUT_DIR if resolution == '5s' else f'{OUTPUT_DIR}_{resolution}'
    os.makedirs(out_dir, exist_ok=True)

    # Find source files
    if sequential:
        # Sequential needs 1s source
        anchor_dir = os.path.join(ATLAS_ROOT, '1s')
    else:
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
    print(f'  Mode: {"SEQUENTIAL (sheet music, honest)" if sequential else "BULK GPU (fast)"}')
    print()

    total_rows = 0
    skipped = 0

    # Sequential mode: carry aggregator + SFE across days for 1h/1D context
    carry_agg = None
    carry_sfe = None
    carry_vels = None

    for fpath in tqdm(all_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        out_path = os.path.join(out_dir, f'{day_name}.parquet')

        if os.path.exists(out_path):
            # Still need to process for carry-forward even if output exists
            if sequential and carry_agg is not None:
                # Feed through aggregator to maintain history, discard output
                _, carry_agg, carry_sfe, carry_vels = process_day_sequential(
                    day_name, resolution=resolution,
                    agg=carry_agg, sfe=carry_sfe, prev_velocities=carry_vels)
            skipped += 1
            continue

        if sequential:
            df, carry_agg, carry_sfe, carry_vels = process_day_sequential(
                day_name, resolution=resolution,
                agg=carry_agg, sfe=carry_sfe, prev_velocities=carry_vels)
        else:
            df = process_day_bulk(day_name, resolution=resolution)
        if len(df) == 0:
            continue

        df.to_parquet(out_path, index=False)
        total_rows += len(df)

        gc.collect()

    if sequential and carry_agg is not None:
        # Report 1h/1D bar counts at end
        from nn_v2.aggregator import Aggregator
        for tf in ['1h', '1D']:
            tf_df = carry_agg.get_closed_bars_df(tf)
            print(f'  {tf} accumulated bars: {len(tf_df)}')

    print(f'\nDone: {total_rows:,} rows across {len(all_files) - skipped} days')
    print(f'Skipped (already exists): {skipped}')
    print(f'Output: {out_dir}/')


if __name__ == '__main__':
    main()
