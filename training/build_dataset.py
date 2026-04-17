"""
Build 79D Feature Dataset — sequential, per-TF, zero lookahead.

Computes each TF's features independently from ATLAS parquets.
No partial bars. Each TF updates only when its bar closes.
Higher TFs carry history across days naturally (loaded from ATLAS).
SFE window = 300 bars (matches live via compute_features.SFE_WINDOW).

For each day:
  1. Load anchor TF bars for this day
  2. For each higher TF: load all bars UP TO this day (cross-day history)
  3. Run SFE on tail(300) per TF — gets states for latest bars
  4. For each anchor bar: find latest closed bar per TF, assemble 79D

Usage:
  python training/build_dataset.py                          # all days, 1m anchor
  python training/build_dataset.py --resolution 5s          # 5s anchor
  python training/build_dataset.py --days 5                 # first 5 days
  python training/build_dataset.py --start 2025-02-01       # from date
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
from core.features import (
    extract_features, FEATURE_NAMES, N_FEATURES, TF_ORDER, TF_SECONDS,
)

ATLAS_ROOT_DEFAULT = 'DATA/ATLAS'
OUTPUT_DIR_DEFAULT = 'DATA/FEATURES_79D'

# Set by main() from args — modules use these
ATLAS_ROOT = ATLAS_ROOT_DEFAULT
OUTPUT_DIR = OUTPUT_DIR_DEFAULT
SFE_MIN_BARS = 21
SFE_WINDOW = 300  # max bars to feed SFE — matches compute_features.SFE_WINDOW (live parity)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Build 79D features (sequential, per-TF, live parity)')
    p.add_argument('--start', type=str, default=None)
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--days', type=int, default=None)
    p.add_argument('--resolution', type=str, default='1m', choices=['5s', '15s', '1m'])
    p.add_argument('--atlas', type=str, default=None,
                   help='ATLAS root dir (default: DATA/ATLAS). Use DATA/ATLAS_NT8 for NT8 data.')
    p.add_argument('--incremental', action='store_true',
                   help='Use FeatureProcessor: load checkpoint, process only new days')
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


class AtlasCache:
    """Pre-loads all ATLAS data per TF once. Slices by day index — no re-reading.

    Eliminates the O(days² × TFs) parquet read bottleneck.
    Each TF's full history is loaded once, then day boundaries are used
    to slice "all bars up to day N" in O(1).
    """

    def __init__(self, tfs: list):
        self._data = {}        # {tf: DataFrame} — full history, sorted
        self._day_ends = {}    # {tf: {day_name: end_idx}} — cumulative bar count per day

        for tf in tqdm(tfs, desc='Loading ATLAS', unit='tf'):
            tf_dir = os.path.join(ATLAS_ROOT, tf)
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if not files:
                continue

            dfs = []
            day_ends = {}
            cumul_len = 0
            for f in files:
                day_name = os.path.basename(f).replace('.parquet', '')
                df = pd.read_parquet(f)
                dfs.append(df)
                cumul_len += len(df)
                day_ends[day_name] = cumul_len

            full = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
            self._data[tf] = full
            self._day_ends[tf] = day_ends

    def get_cumulative(self, tf: str, up_to_day: str) -> pd.DataFrame:
        """Get all bars for a TF up to and including the given day. O(1) slice."""
        if tf not in self._data:
            return pd.DataFrame()
        ends = self._day_ends[tf]
        if up_to_day not in ends:
            # Day not in this TF — find the latest day <= up_to_day
            valid = [d for d in ends if d <= up_to_day]
            if not valid:
                return pd.DataFrame()
            up_to_day = valid[-1]
        end_idx = ends[up_to_day]
        return self._data[tf].iloc[:end_idx]

    def get_day_start(self, tf: str, day_name: str) -> int:
        """Get the index where a day starts in the full history."""
        ends = self._day_ends.get(tf, {})
        # Day start = previous day's end
        days_sorted = sorted(ends.keys())
        day_idx = days_sorted.index(day_name) if day_name in days_sorted else -1
        if day_idx <= 0:
            return 0
        prev_day = days_sorted[day_idx - 1]
        return ends[prev_day]

    def memory_mb(self) -> float:
        return sum(df.memory_usage(deep=True).sum() for df in self._data.values()) / 1e6


def process_one_day(day_name: str, anchor_tf: str, cache: 'AtlasCache',
                    sfe, prev_velocities: dict, sfe_cache: dict = None):
    """Process one day: for each anchor bar, assemble 79D from all TFs.

    Args:
        day_name: the day to produce output for
        anchor_tf: resolution TF (e.g. '1m')
        cache: AtlasCache with pre-loaded ATLAS data
        sfe: StatisticalFieldEngine instance (reused)
        prev_velocities: velocity state from previous day
        sfe_cache: {tf: (n_bars, states, tail_offset)} — reuse SFE if bar count unchanged

    Returns:
        (DataFrame, updated prev_velocities, updated sfe_cache)
    """
    if sfe_cache is None:
        sfe_cache = {}

    # Load anchor bars for this day only
    anchor_path = os.path.join(ATLAS_ROOT, anchor_tf, f'{day_name}.parquet')
    if not os.path.exists(anchor_path):
        return pd.DataFrame(), prev_velocities, sfe_cache

    anchor_df = pd.read_parquet(anchor_path).sort_values('timestamp').reset_index(drop=True)
    if len(anchor_df) < SFE_MIN_BARS:
        return pd.DataFrame(), prev_velocities, sfe_cache

    anchor_ts = anchor_df['timestamp'].values

    # For each TF: get cumulative history from cache, run SFE on tail
    tf_data = {}  # {tf: (timestamps, states, bars_df)}

    for tf in TF_ORDER:
        cumul = cache.get_cumulative(tf, day_name)
        if len(cumul) < SFE_MIN_BARS:
            continue

        ts_arr = cumul['timestamp'].values
        n_bars = len(cumul)

        # Check SFE cache — skip recompute if this TF has no new bars
        cached = sfe_cache.get(tf)
        if cached and cached[0] == n_bars:
            tf_data[tf] = {
                'timestamps': ts_arr,
                'states': cached[1],
                'tail_offset': cached[2],
                'bars': cumul,
            }
            continue

        # Find where today's data starts
        today_start_idx = cache.get_day_start(tf, day_name)

        # SFE must cover today's bars + warmup history before today
        warmup = min(SFE_WINDOW, today_start_idx)
        sfe_start = max(0, today_start_idx - warmup)
        sfe_input = cumul.iloc[sfe_start:].reset_index(drop=True)
        tail_offset = sfe_start

        states = sfe.batch_compute_states(sfe_input)
        if not states:
            continue

        # Cache for next day
        sfe_cache[tf] = (n_bars, states, tail_offset)

        tf_data[tf] = {
            'timestamps': ts_arr,
            'states': states,
            'tail_offset': tail_offset,
            'bars': cumul,
        }

    # Need at least the smallest TF in TF_ORDER (15s) or the anchor if it's in TF_ORDER
    min_tf = anchor_tf if anchor_tf in TF_ORDER else TF_ORDER[0]
    if min_tf not in tf_data:
        return pd.DataFrame(), prev_velocities, sfe_cache


    rows = []

    for bar_idx in range(len(anchor_df)):
        ts = anchor_ts[bar_idx]

        states_by_tf = {}
        ohlcv_by_tf = {}

        for tf, data in tf_data.items():
            tf_ts = data['timestamps']
            states = data['states']
            offset = data['tail_offset']

            # Find the LAST CLOSED bar at current timestamp (no lookahead).
            # A bar labeled B covers [B, B+period) — it closes at B+period.
            # Use only bars where B+period <= ts → B <= ts - period.
            # This matches the live feature engine which only has closed bars
            # in its store (aggregator emits on boundary crossing).
            period = TF_SECONDS.get(tf, 0)
            idx = int(np.searchsorted(tf_ts, ts - period, side='right')) - 1
            if idx < 0:
                continue

            # Map to states array (states covers bars from offset onward)
            state_idx = idx - offset
            if state_idx < 0 or state_idx >= len(states):
                continue

            states_by_tf[tf] = states[state_idx]

            # OHLCV up to this bar
            ohlcv = data['bars'].iloc[:idx + 1]
            if len(ohlcv) > SFE_WINDOW:
                ohlcv = ohlcv.tail(SFE_WINDOW).reset_index(drop=True)
            ohlcv_by_tf[tf] = ohlcv

        # Need at least 15s (smallest TF in 79D) or anchor, whichever is in TF_ORDER
        min_required_tf = anchor_tf if anchor_tf in TF_ORDER else '15s'
        if min_required_tf not in states_by_tf:
            continue

        feat, prev_velocities = extract_features(
            states_by_tf, ohlcv_by_tf, prev_velocities, ts
        )

        rows.append({
            'timestamp': ts,
            **{name: feat[i] for i, name in enumerate(FEATURE_NAMES)}
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(), prev_velocities, sfe_cache


WARMUP_DAYS_CHECKPOINT = 10  # days of 5s bars to include in checkpoint


def _save_checkpoint_for_live(all_days: list, anchor_tf: str, prev_velocities: dict):
    """Build aggregator from last N days and save checkpoint for live engine."""
    from training.aggregator import Aggregator

    checkpoint_path = os.path.join(ATLAS_ROOT, 'checkpoint.json')
    days_to_load = all_days[-WARMUP_DAYS_CHECKPOINT:]
    if not days_to_load:
        return

    print(f'\nBuilding checkpoint for live engine...')
    agg = Aggregator(history_limit=2000)

    total_bars = 0
    for day_name in tqdm(days_to_load, desc='  Checkpoint', unit='day', leave=False):
        # Load 5s bars for this day
        fpath = os.path.join(ATLAS_ROOT, anchor_tf, f'{day_name}.parquet')
        if not os.path.exists(fpath):
            continue
        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        for _, row in df.iterrows():
            agg.feed({
                'timestamp': row['timestamp'],
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row.get('volume', 0),
            })
            total_bars += 1

    agg.save_checkpoint(checkpoint_path, velocities=prev_velocities)
    bar_counts = {tf: len(bars) for tf, bars in agg.history.items() if bars}
    print(f'  Checkpoint saved: {checkpoint_path}')
    print(f'  {total_bars:,} bars from {len(days_to_load)} days: {bar_counts}')


def main():
    global ATLAS_ROOT, OUTPUT_DIR
    args = parse_args()
    anchor_tf = args.resolution

    # Switch data source if --atlas specified
    if args.atlas:
        ATLAS_ROOT = args.atlas
        atlas_name = os.path.basename(ATLAS_ROOT.rstrip('/'))
        feat_name = atlas_name.replace('ATLAS', 'FEATURES')
        OUTPUT_DIR = os.path.join('DATA', feat_name)

    # Incremental mode: use FeatureProcessor (same path as live)
    if args.incremental:
        from training.feature_processor import FeatureProcessor
        fp = FeatureProcessor(atlas_root=ATLAS_ROOT)
        fp.process_new_days()
        return

    out_dir = f'{OUTPUT_DIR}_{anchor_tf}'
    os.makedirs(out_dir, exist_ok=True)

    # Full day list (unfiltered) — needed for higher TF cross-day history
    all_days_full = get_all_days(anchor_tf)
    print(f'Total days in ATLAS/{anchor_tf}: {len(all_days_full)}')

    # Filter to build range
    all_days = list(all_days_full)
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

    # Pre-load all ATLAS data per TF (one-time cost, eliminates O(days²) I/O)
    cache = AtlasCache(TF_ORDER)
    print(f'  ATLAS loaded: {cache.memory_mb():.0f} MB in memory')

    sfe = StatisticalFieldEngine()
    prev_velocities = {}
    sfe_cache = {}  # {tf: (n_bars, states, offset)} — skip SFE if no new bars
    total_rows = 0

    for i, day_name in enumerate(tqdm(to_build, desc='Days', unit='day')):
        df, prev_velocities, sfe_cache = process_one_day(
            day_name, anchor_tf, cache, sfe, prev_velocities, sfe_cache)

        if len(df) > 0:
            out_path = os.path.join(out_dir, f'{day_name}.parquet')
            df.to_parquet(out_path, index=False)
            total_rows += len(df)

        # Periodic GC
        if i % 10 == 0:
            gc.collect()

    del sfe, cache
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

    # Save aggregator checkpoint for live engine startup
    # Build aggregator from last WARMUP_DAYS of 5s bars
    _save_checkpoint_for_live(all_days_full, anchor_tf, prev_velocities)


if __name__ == '__main__':
    main()
