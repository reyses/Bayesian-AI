"""
Maintenance — run before market open to warm up the system.

Downloads latest data, warms aggregator with 30 days of 1s bars,
saves warm state for live engine to load on startup.

Run during maintenance window (before market open):
    python -m live.maintenance                          # full warmup
    python -m live.maintenance --days 10                # quick warmup
    python -m live.maintenance --skip-download          # use existing ATLAS

Produces:
    live/state/aggregator.pkl   — warm aggregator state
    live/state/velocities.pkl   — prev_velocities for 79D
    live/state/warmup_info.json — metadata (last bar ts, bar counts per TF)
"""
import asyncio
import os
import sys
import glob
import json
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import extract_79d, FEATURE_NAMES_79D, TF_ORDER, N_FEATURES
from nn_v2.aggregator import Aggregator

ATLAS_1S = 'DATA/ATLAS/1s'
STATE_DIR = 'live/state'
SFE_MIN_BARS = 21


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Maintenance — warm up for live trading')
    p.add_argument('--days', type=int, default=30, help='Days of history to warm up')
    p.add_argument('--skip-download', action='store_true', help='Use existing ATLAS data')
    return p.parse_args()


def get_recent_days(n_days: int) -> list:
    """Get the most recent N days of 1s ATLAS data."""
    files = sorted(glob.glob(os.path.join(ATLAS_1S, '*.parquet')))
    if not files:
        print(f'ERROR: No 1s data in {ATLAS_1S}/')
        return []
    return files[-n_days:]


def warm_aggregator(day_files: list):
    """Feed 1s bars through aggregator to warm up all TFs."""
    agg = Aggregator(history_limit=2000)
    sfe = StatisticalFieldEngine()
    prev_velocities = {}

    total_bars = 0
    last_ts = 0
    last_79d = None

    print(f'Warming aggregator with {len(day_files)} days...')

    for fpath in tqdm(day_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)

        for _, row in df.iterrows():
            bar = {
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0),
            }
            agg.feed(bar)
            total_bars += 1
            last_ts = row['timestamp']

        # Compute one 79D at end of day to verify warmup
        states_by_tf = {}
        ohlcv_by_tf = {}
        for tf in TF_ORDER:
            tf_df = agg.get_closed_bars_df(tf)
            if len(tf_df) < SFE_MIN_BARS:
                continue
            ohlcv_by_tf[tf] = tf_df
            sfe_input = tf_df.tail(300).reset_index(drop=True) if len(tf_df) > 300 else tf_df
            states = sfe.batch_compute_states(sfe_input)
            if states:
                states_by_tf[tf] = states[-1]

        if '1m' in states_by_tf:
            last_79d, prev_velocities = extract_79d(
                states_by_tf, ohlcv_by_tf, prev_velocities, last_ts)

    # Report bar counts per TF
    print(f'\nWarmup complete:')
    print(f'  Total 1s bars fed: {total_bars:,}')
    for tf in ['1s', '15s', '1m', '5m', '15m', '1h', '1D']:
        n = agg.bar_count(tf) if hasattr(agg, 'bar_count') else len(agg.history.get(tf, []))
        print(f'  {tf:>4}: {n} bars')

    if last_79d is not None:
        # Check 1h and 1D are alive
        from core.features_79d import FEATURE_NAMES_79D as FN
        for check_tf in ['1h', '1D']:
            z_idx = FN.index(f'{check_tf}_z_se')
            z_val = last_79d[z_idx]
            status = 'ALIVE' if abs(z_val) > 0.001 else 'DEAD (fallback)'
            print(f'  {check_tf}_z_se = {z_val:.4f} — {status}')

    return agg, sfe, prev_velocities, last_ts, total_bars


def save_state(agg, prev_velocities, last_ts, total_bars):
    """Save warm aggregator state to disk."""
    os.makedirs(STATE_DIR, exist_ok=True)

    # Save aggregator history
    agg_state = {
        'history': agg.history,
        'accumulators': {},
    }
    # Save accumulator state per TF
    for tf in TF_ORDER:
        acc = agg._accumulators.get(tf)
        if acc and hasattr(acc, 'current_bar'):
            agg_state['accumulators'][tf] = {
                'current_bar': acc.current_bar,
                'bar_count': acc._bar_count if hasattr(acc, '_bar_count') else 0,
            }

    with open(os.path.join(STATE_DIR, 'aggregator.pkl'), 'wb') as f:
        pickle.dump(agg_state, f)

    with open(os.path.join(STATE_DIR, 'velocities.pkl'), 'wb') as f:
        pickle.dump(prev_velocities, f)

    info = {
        'last_ts': last_ts,
        'total_bars': total_bars,
        'warmup_time': datetime.utcnow().isoformat(),
        'bar_counts': {tf: len(agg.history.get(tf, [])) for tf in ['1s', '15s', '1m', '5m', '15m', '1h', '1D']},
    }
    with open(os.path.join(STATE_DIR, 'warmup_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f'\nState saved to {STATE_DIR}/')
    print(f'  aggregator.pkl')
    print(f'  velocities.pkl')
    print(f'  warmup_info.json')


def load_state():
    """Load warm aggregator state from disk. Returns (agg, prev_velocities, info) or None."""
    agg_path = os.path.join(STATE_DIR, 'aggregator.pkl')
    vel_path = os.path.join(STATE_DIR, 'velocities.pkl')
    info_path = os.path.join(STATE_DIR, 'warmup_info.json')

    if not all(os.path.exists(p) for p in [agg_path, vel_path, info_path]):
        return None

    with open(agg_path, 'rb') as f:
        agg_state = pickle.load(f)

    with open(vel_path, 'rb') as f:
        prev_velocities = pickle.load(f)

    with open(info_path, 'r') as f:
        info = json.load(f)

    # Reconstruct aggregator
    agg = Aggregator(history_limit=2000)
    agg.history = agg_state['history']

    print(f'Loaded warm state from {STATE_DIR}/')
    print(f'  Last bar: ts={info["last_ts"]}')
    print(f'  Warmup time: {info["warmup_time"]}')
    for tf, n in info['bar_counts'].items():
        print(f'  {tf:>4}: {n} bars')

    return agg, prev_velocities, info


def merge_atlas_live():
    """Merge ATLAS_LIVE data into main ATLAS (extends the dataset)."""
    import shutil

    live_root = 'DATA/ATLAS_LIVE'
    atlas_root = 'DATA/ATLAS'

    if not os.path.exists(live_root):
        return

    merged = 0
    for tf in os.listdir(live_root):
        tf_live = os.path.join(live_root, tf)
        tf_atlas = os.path.join(atlas_root, tf)
        if not os.path.isdir(tf_live):
            continue
        os.makedirs(tf_atlas, exist_ok=True)

        for f in sorted(os.listdir(tf_live)):
            if not f.endswith('.parquet'):
                continue
            src = os.path.join(tf_live, f)
            dst = os.path.join(tf_atlas, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                merged += 1
                print(f'  Merged: {tf}/{f}')

    # Also merge FEATURES_79D_5s_live into FEATURES_79D_5s_v2
    feat_live = 'DATA/FEATURES_79D_5s_live'
    feat_main = 'DATA/FEATURES_79D_5s_v2'
    if os.path.exists(feat_live):
        os.makedirs(feat_main, exist_ok=True)
        for f in sorted(os.listdir(feat_live)):
            if not f.endswith('.parquet'):
                continue
            src = os.path.join(feat_live, f)
            dst = os.path.join(feat_main, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                merged += 1
                print(f'  Merged: features/{f}')

    if merged:
        print(f'  Total merged: {merged} files')
    else:
        print(f'  No new live data to merge')


def build_features_for_new_days():
    """Build 79D features for any ATLAS days missing from FEATURES_79D_5s_v2."""
    import subprocess
    feat_dir = 'DATA/FEATURES_79D_5s_v2'
    atlas_1m = 'DATA/ATLAS/1m'

    if not os.path.exists(atlas_1m):
        print('  No ATLAS 1m data')
        return

    existing = set()
    if os.path.exists(feat_dir):
        existing = {f.replace('.parquet', '') for f in os.listdir(feat_dir) if f.endswith('.parquet')}

    atlas_days = {f.replace('.parquet', '') for f in os.listdir(atlas_1m) if f.endswith('.parquet')}
    missing = sorted(atlas_days - existing)

    if not missing:
        print('  All ATLAS days have features — nothing to build')
        return

    print(f'  {len(missing)} days need 79D features: {missing[0]} to {missing[-1]}')
    print(f'  Running build_dataset_v2...')

    # Run the feature builder for missing days
    result = subprocess.run(
        ['python', 'nn_v2/build_dataset_v2.py', '--resolution', '1m',
         '--start', missing[0].replace('_', '-'),
         '--end', missing[-1].replace('_', '-')],
        capture_output=True, text=True, timeout=3600)

    if result.returncode == 0:
        print(f'  Features built successfully')
    else:
        print(f'  Feature build failed: {result.stderr[-200:] if result.stderr else "unknown"}')


async def download_from_nt8():
    """Connect to NT8, download bars from last ATLAS to now, save to ATLAS."""
    from live.config import LiveConfig
    from live.nt8_client import NT8Client
    from live.protocol import MsgType

    # Find last ATLAS 1m timestamp
    atlas_1m = 'DATA/ATLAS/1m'
    if os.path.exists(atlas_1m):
        files = sorted(glob.glob(os.path.join(atlas_1m, '*.parquet')))
        if files:
            last_df = pd.read_parquet(files[-1])
            last_ts = last_df['timestamp'].max()
            last_day = os.path.basename(files[-1]).replace('.parquet', '')
            print(f'  Last ATLAS 1m: {last_day} (ts={last_ts:.0f})')
        else:
            last_ts = 0
    else:
        last_ts = 0

    if last_ts == 0:
        print(f'  No ATLAS data — requesting full history')

    config = LiveConfig()
    client = NT8Client(config)

    if last_ts > 0:
        client.set_resume_timestamp(last_ts)

    print(f'  Connecting to NT8...')
    if not await client.connect():
        print(f'  Failed to connect to NT8 — skipping download')
        return

    print(f'  Connected. Receiving bars...')

    bars_1s = []
    history_done = False
    timeout_start = time.time()

    while not history_done and (time.time() - timeout_start) < 300:  # 5 min timeout
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=10.0)
        except asyncio.TimeoutError:
            if bars_1s:
                # Got some bars, no more coming
                history_done = True
            continue

        msg_type = msg.get('type', '')
        if msg_type == MsgType.BAR:
            bars_1s.append({
                'timestamp': msg.get('timestamp', 0),
                'open': msg.get('open', 0),
                'high': msg.get('high', 0),
                'low': msg.get('low', 0),
                'close': msg.get('close', 0),
                'volume': msg.get('volume', 0),
            })
            if len(bars_1s) % 10000 == 0:
                print(f'    {len(bars_1s):,} bars received...')
        elif msg_type == MsgType.HISTORY_DONE:
            history_done = True

    await client.disconnect()
    print(f'  Received {len(bars_1s):,} bars')

    if not bars_1s:
        print(f'  No new bars')
        return

    # Save 1s bars to ATLAS
    df = pd.DataFrame(bars_1s)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y_%m_%d')

    for day, day_df in df.groupby('date'):
        # Save 1s
        out_dir = os.path.join('DATA', 'ATLAS', '1s')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{day}.parquet')
        if not os.path.exists(out_path):
            day_df.drop(columns=['date']).to_parquet(out_path, index=False)
            print(f'    ATLAS 1s: {day} ({len(day_df)} bars)')

        # Aggregate to 1m and save
        day_sorted = day_df.sort_values('timestamp')
        day_sorted['minute'] = (day_sorted['timestamp'] // 60).astype(int) * 60
        bars_1m = day_sorted.groupby('minute').agg({
            'timestamp': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).reset_index(drop=True)

        out_dir = os.path.join('DATA', 'ATLAS', '1m')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{day}.parquet')
        if not os.path.exists(out_path):
            bars_1m.to_parquet(out_path, index=False)
            print(f'    ATLAS 1m: {day} ({len(bars_1m)} bars)')

    print(f'  Download complete')


def main():
    args = parse_args()

    print(f'{"="*60}')
    print(f'MAINTENANCE — Full Data Pipeline + Warmup')
    print(f'{"="*60}')

    # Step 0: Merge live session data into main ATLAS
    print(f'\n--- Step 0: Merge ATLAS_LIVE → ATLAS ---')
    merge_atlas_live()

    # Step 1: Build 79D features for any new ATLAS days
    print(f'\n--- Step 1: Build 79D features for new days ---')
    build_features_for_new_days()

    # Step 1b: Download missing days from NT8
    if not args.skip_download:
        print(f'\n--- Step 1b: Download missing data from NT8 ---')
        asyncio.run(download_from_nt8())

        # Rebuild features for any new days
        print(f'\n--- Step 1c: Build features for newly downloaded days ---')
        build_features_for_new_days()

    print(f'\n--- Step 2: Warm aggregator ---')
    day_files = get_recent_days(args.days)
    if not day_files:
        print('No data available. Download data first.')
        return

    print(f'Using {len(day_files)} days: {os.path.basename(day_files[0])} to {os.path.basename(day_files[-1])}')

    t0 = time.perf_counter()
    agg, sfe, prev_vel, last_ts, total_bars = warm_aggregator(day_files)
    elapsed = time.perf_counter() - t0
    print(f'Warmup took {elapsed:.0f}s')

    # Step 3: Save state
    print(f'\n--- Step 3: Save state ---')
    save_state(agg, prev_vel, last_ts, total_bars)

    print(f'\n{"="*60}')
    print(f'READY FOR LIVE')
    print(f'  Run: python -m live.live_blended')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
