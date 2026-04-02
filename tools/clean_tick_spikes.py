"""
Clean tick spikes from 1s ATLAS data.

Detects single-bar price spikes that deviate more than THRESHOLD ticks
from the surrounding bars, replaces with interpolated values.

Method:
  1. For each bar, compute distance from rolling median of neighbors
  2. If distance > THRESHOLD ticks, flag as spike
  3. Replace spike OHLC with linear interpolation from prev/next bars
  4. Save cleaned data back to ATLAS

Usage:
  python tools/clean_tick_spikes.py                    # clean all 1s
  python tools/clean_tick_spikes.py --threshold 100    # custom threshold (ticks)
  python tools/clean_tick_spikes.py --dry-run           # report only, don't modify
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import glob
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

ATLAS_ROOT = 'DATA/ATLAS'
TICK = 0.25
THRESHOLD = 400  # ticks — 400 ticks = 100 points on MNQ
WINDOW = 5       # bars on each side for median calculation

if '--threshold' in sys.argv:
    idx = sys.argv.index('--threshold')
    THRESHOLD = int(sys.argv[idx + 1])

DRY_RUN = '--dry-run' in sys.argv


def clean_spikes(df):
    """Detect and interpolate spike bars. Returns cleaned df + spike count."""
    closes = df['close'].values.copy()
    opens = df['open'].values.copy()
    highs = df['high'].values.copy()
    lows = df['low'].values.copy()
    n = len(df)
    spikes = 0

    for i in range(WINDOW, n - WINDOW):
        # Rolling median of surrounding bars (exclude current)
        neighbors = np.concatenate([
            closes[i - WINDOW:i],
            closes[i + 1:i + WINDOW + 1]
        ])
        median_price = np.median(neighbors)

        # Check all OHLC for spikes
        for arr_name, arr in [('open', opens), ('high', highs), ('low', lows), ('close', closes)]:
            deviation_ticks = abs(arr[i] - median_price) / TICK
            if deviation_ticks > THRESHOLD:
                # Interpolate from prev and next bar
                prev_val = arr[i - 1]
                next_val = arr[i + 1]
                interp = (prev_val + next_val) / 2

                if not DRY_RUN:
                    arr[i] = interp

                if arr_name == 'close':  # only count once per bar
                    spikes += 1
                    if spikes <= 20:
                        from datetime import datetime
                        ts = df['timestamp'].iloc[i]
                        t_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                        print(f'    Spike at {t_str}: {arr_name}={df[arr_name].iloc[i]:.2f} '
                              f'-> {interp:.2f} (median={median_price:.2f}, dev={deviation_ticks:.0f}t)')

    if not DRY_RUN and spikes > 0:
        df['open'] = opens
        df['high'] = highs
        df['low'] = lows
        df['close'] = closes

        # Fix: high must be >= open, close, and low must be <= open, close
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df, spikes


def main():
    print(f'Tick Spike Cleaner')
    print(f'  Threshold: {THRESHOLD} ticks ({THRESHOLD * TICK:.1f} points)')
    print(f'  Window: {WINDOW} bars each side')
    print(f'  Mode: {"DRY RUN (report only)" if DRY_RUN else "ARCHIVE + CLEAN"}')
    print()

    # Archive original 1s data before modifying
    RAW_ROOT = 'DATA/ATLAS_RAW'
    if not DRY_RUN:
        raw_1s = os.path.join(RAW_ROOT, '1s')
        os.makedirs(raw_1s, exist_ok=True)

    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1s', '*.parquet')))
    total_spikes = 0

    for fpath in tqdm(files, desc='Cleaning 1s'):
        fname = os.path.basename(fpath)
        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)

        df_clean, spikes = clean_spikes(df)
        total_spikes += spikes

        if spikes > 0:
            tqdm.write(f'  {fname}: {spikes} spikes {"(would fix)" if DRY_RUN else "fixed"}')
            if not DRY_RUN:
                # Archive original
                raw_path = os.path.join(RAW_ROOT, '1s', fname)
                if not os.path.exists(raw_path):
                    # Save original untouched copy
                    df_orig = pd.read_parquet(fpath)
                    df_orig.to_parquet(raw_path, index=False)
                    tqdm.write(f'    Archived original -> {raw_path}')
                # Write cleaned version
                df_clean.to_parquet(fpath, index=False)

    print(f'\nTotal spikes: {total_spikes}')
    if not DRY_RUN:
        print(f'Originals archived to: {RAW_ROOT}/1s/')

    if not DRY_RUN and total_spikes > 0:
        print(f'\nRun atlas_rebuild.py --skip-clean to validate + rebuild features')


if __name__ == '__main__':
    main()
