"""
Build intermediate timeframes from base TFs and validate against controls.

Aggregation chain:
  1s  → 5s, 15s, 30s     (validate: 60 × 1s = 1 × 1m control)
  1m  → 3m, 5m, 15m, 30m (validate: 60 × 1m = 1 × 1h control)
  1h  → 4h               (validate against 1D control)

Usage:
  python tools/build_timeframes.py           # build all + validate
  python tools/build_timeframes.py --skip-1s # skip 1s-based TFs (if 1s not available)
  python tools/build_timeframes.py --validate-only  # just validate, don't build
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import os
import sys
import glob
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

ATLAS = 'DATA/ATLAS'
TICK = 0.25

SKIP_1S = '--skip-1s' in sys.argv
VALIDATE_ONLY = '--validate-only' in sys.argv

# Aggregation rules: (source_tf, target_tf, bars_per_target)
AGG_FROM_1S = [
    ('1s', '5s', 5),
    ('1s', '15s', 15),
    ('1s', '30s', 30),
]
AGG_FROM_1M = [
    ('1m', '5m', 5),
    ('1m', '15m', 15),
    ('1m', '30m', 30),
]
AGG_FROM_1H = [
    ('1h', '4h', 4),
]

# Validation pairs: (fine_tf, coarse_tf, fine_per_coarse)
VALIDATE_PAIRS = [
    ('1s', '1m', 60, 'Databento 1m control'),
    ('5s', '1m', 12, '1m control'),
    ('15s', '1m', 4, '1m control'),
    ('1m', '1h', 60, 'Databento 1h control'),
    ('5m', '1h', 12, '1h control'),
    ('15m', '1h', 4, '1h control'),
]


def aggregate_bars(source_df, period_seconds):
    """Aggregate OHLCV bars to a coarser timeframe."""
    source_df = source_df.sort_values('timestamp')
    source_df['bar_ts'] = (source_df['timestamp'] // period_seconds) * period_seconds

    agg = source_df.groupby('bar_ts').agg(
        timestamp=('bar_ts', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index(drop=True)

    return agg


def build_tf(source_tf, target_tf, period_seconds):
    """Build a target TF from a source TF."""
    source_dir = os.path.join(ATLAS, source_tf)
    target_dir = os.path.join(ATLAS, target_tf)
    os.makedirs(target_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(source_dir, '*.parquet')))
    if not files:
        print(f'  {source_tf} -> {target_tf}: no source files')
        return

    total = 0
    for fpath in tqdm(files, desc=f'  {source_tf}->{target_tf}'):
        fname = os.path.basename(fpath)
        target_path = os.path.join(target_dir, fname)

        if os.path.exists(target_path):
            continue  # skip if already built

        df = pd.read_parquet(fpath)
        if len(df) < 2:
            continue

        agg = aggregate_bars(df, period_seconds)
        agg.to_parquet(target_path, index=False)
        total += len(agg)

    print(f'    {total:,} bars written')


def validate_pair(fine_tf, coarse_tf, fine_per_coarse, label):
    """Validate that fine TF bars aggregate correctly to coarse TF."""
    fine_dir = os.path.join(ATLAS, fine_tf)
    coarse_dir = os.path.join(ATLAS, coarse_tf)

    fine_files = sorted(glob.glob(os.path.join(fine_dir, '*.parquet')))
    coarse_files = sorted(glob.glob(os.path.join(coarse_dir, '*.parquet')))

    if not fine_files or not coarse_files:
        print(f'  {fine_tf} vs {coarse_tf}: missing data')
        return

    coarse_lookup = {}
    for f in coarse_files:
        day = os.path.basename(f).replace('.parquet', '')
        coarse_lookup[day] = f

    errors = 0
    checked = 0
    total_bars = 0

    # Get TF period in seconds
    tf_seconds = {'1s': 1, '5s': 5, '15s': 15, '30s': 30,
                  '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                  '1h': 3600, '4h': 14400}
    coarse_period = tf_seconds.get(coarse_tf, 60)

    for fpath in fine_files[:30]:  # sample first 30 days
        day = os.path.basename(fpath).replace('.parquet', '')
        if day not in coarse_lookup:
            continue

        fine_df = pd.read_parquet(fpath)
        coarse_df = pd.read_parquet(coarse_lookup[day])

        if len(fine_df) == 0 or len(coarse_df) == 0:
            continue

        # Aggregate fine to coarse period
        fine_agg = aggregate_bars(fine_df, coarse_period)

        # Compare OHLCV
        for _, c_row in coarse_df.iterrows():
            checked += 1
            ts = c_row['timestamp']
            f_match = fine_agg[fine_agg['timestamp'] == ts]

            if len(f_match) == 0:
                continue

            f_row = f_match.iloc[0]
            for col in ['high', 'low']:
                diff = abs(f_row[col] - c_row[col])
                if diff > TICK:
                    errors += 1
                    if errors <= 5:
                        t_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                        print(f'    MISMATCH {t_str}: {fine_tf} {col}={f_row[col]:.2f} vs {coarse_tf} {col}={c_row[col]:.2f} (diff={diff:.2f})')

        total_bars += len(coarse_df)

    pct = errors / max(checked, 1) * 100
    status = 'PASS' if pct < 1.0 else 'WARN' if pct < 5.0 else 'FAIL'
    print(f'  {fine_tf} vs {coarse_tf} ({label}): {status} — {errors}/{checked} mismatches ({pct:.1f}%)')


def main():
    print(f'BUILD TIMEFRAMES + VALIDATE')
    print(f'  ATLAS: {ATLAS}/')
    print()

    if not VALIDATE_ONLY:
        # Build from 1s
        if not SKIP_1S:
            print('Aggregating from 1s:')
            for src, tgt, n in AGG_FROM_1S:
                tf_secs = {'5s': 5, '15s': 15, '30s': 30}
                build_tf(src, tgt, tf_secs[tgt])

        # Build from 1m
        print('Aggregating from 1m:')
        for src, tgt, n in AGG_FROM_1M:
            tf_secs = {'3m': 180, '5m': 300, '15m': 900, '30m': 1800}
            build_tf(src, tgt, tf_secs[tgt])

        # Build from 1h
        print('Aggregating from 1h:')
        for src, tgt, n in AGG_FROM_1H:
            build_tf(src, tgt, 14400)

    # Validate
    print(f'\nVALIDATION:')
    for fine, coarse, n, label in VALIDATE_PAIRS:
        fine_dir = os.path.join(ATLAS, fine)
        if os.path.exists(fine_dir) and glob.glob(os.path.join(fine_dir, '*.parquet')):
            validate_pair(fine, coarse, n, label)
        else:
            print(f'  {fine} vs {coarse}: {fine} not available, skipping')

    # Summary
    print(f'\nATLAS SUMMARY:')
    for tf in ['1s', '5s', '15s', '30s', '1m', '3m', '5m', '15m', '30m', '1h', '4h', '1D']:
        tf_dir = os.path.join(ATLAS, tf)
        if os.path.exists(tf_dir):
            files = glob.glob(os.path.join(tf_dir, '*.parquet'))
            print(f'  {tf:>4}: {len(files)} days')
        else:
            print(f'  {tf:>4}: --')


if __name__ == '__main__':
    main()
