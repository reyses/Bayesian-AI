"""Hourly regime labeler — computes regime label per HOUR using only
data from that hour. Removes the lookahead bias of the daily labeler.

For each completed 1-hour window of trading, this tool computes the
SAME 2D classification (direction × variation) as
atlas_regime_labeler_2d, but on the hour's own bars rather than the
whole day. The output is a per-hour CSV with columns:

    day, hour_start_ts, hour_end_ts, regime_2d, direction, variation,
    net_move, range, directional_strength, efficiency_ratio,
    range_expansion, n_bars

OPERATIONAL USE:

At any 5s bar with timestamp `t`, the regime in use is the regime
LABEL for the most recently COMPLETED hour, i.e., the hour ending at
the largest `hour_end_ts <= t`. So during 10:00-11:00, the regime in
use is the label computed from the 9:00-10:00 hour. NO LOOKAHEAD.

For the FIRST hour of any day (no preceding hour in session), the
regime is `WARMUP` — strategies and exits that gate on regime should
treat WARMUP as "regime unknown" and either skip or use a permissive
default.

CLASSIFICATION RULES (same as atlas_regime_labeler_2d):

    Direction (dir_threshold = 0.5):
        UP    if directional_strength >= dir_thr AND net_move > 0
        DOWN  if directional_strength >= dir_thr AND net_move < 0
        FLAT  otherwise

    Variation:
        SMOOTH  if range_expansion < smooth_range_thr (0.7)
                OR efficiency_ratio >= smooth_eff_thr (0.05)
        CHOPPY  otherwise

ATR reference: rolling 24-hour mean of the hourly range (≈ 1 trading day
of hourly windows). range_expansion = current_hour_range / ATR_24h.

USAGE:
    python tools/atlas_regime_labeler_hourly.py
    python tools/atlas_regime_labeler_hourly.py --atr-window 24
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ATLAS_5S_DIR = 'DATA/ATLAS/5s'
DEFAULT_OUT_CSV = 'DATA/ATLAS/regime_labels_hourly.csv'

ATR_HOURS_DEFAULT = 24    # ~1 trading day of hours
DIR_THRESHOLD = 0.5
SMOOTH_RANGE_THRESHOLD = 0.7
SMOOTH_EFF_THRESHOLD = 0.05


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df


def compute_hourly_metrics(day: str, ohlcv_5s: pd.DataFrame) -> list:
    """Aggregate 5s bars into hour buckets; return per-hour metrics."""
    df = ohlcv_5s.copy()
    df['hour_ts'] = (df['timestamp'] // 3600) * 3600

    rows = []
    for hour_start, sub in df.groupby('hour_ts'):
        if len(sub) < 60:    # < 5 minutes of bars; skip
            continue
        opn = float(sub.iloc[0]['open'])
        cls = float(sub.iloc[-1]['close'])
        hi = float(sub['high'].max())
        lo = float(sub['low'].min())
        rng = hi - lo
        net_move = cls - opn
        ds = abs(net_move) / rng if rng > 0 else np.nan
        close_diff_sum = float(sub['close'].diff().abs().sum())
        ef = abs(net_move) / close_diff_sum if close_diff_sum > 1e-9 else np.nan
        rows.append({
            'day': day,
            'hour_start_ts': int(hour_start),
            'hour_end_ts': int(hour_start + 3600),
            'open': opn, 'close': cls, 'high': hi, 'low': lo,
            'range': rng, 'net_move': net_move,
            'directional_strength': float(ds) if not pd.isna(ds) else np.nan,
            'efficiency_ratio': float(ef) if not pd.isna(ef) else np.nan,
            'n_bars': int(len(sub)),
        })
    return rows


def classify_hourly(row: dict,
                          dir_thr: float = DIR_THRESHOLD,
                          smooth_range_thr: float = SMOOTH_RANGE_THRESHOLD,
                          smooth_eff_thr: float = SMOOTH_EFF_THRESHOLD) -> tuple:
    ds = row['directional_strength']
    nm = row['net_move']
    re_ = row.get('range_expansion', np.nan)
    ef = row.get('efficiency_ratio', np.nan)

    if pd.isna(ds) or pd.isna(re_):
        return 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

    # Direction
    if ds >= dir_thr and nm > 0:
        direction = 'UP'
    elif ds >= dir_thr and nm < 0:
        direction = 'DOWN'
    else:
        direction = 'FLAT'

    # Variation
    smooth = (re_ < smooth_range_thr) or (
        not pd.isna(ef) and ef >= smooth_eff_thr)
    variation = 'SMOOTH' if smooth else 'CHOPPY'

    return direction, variation, f'{direction}_{variation}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas-5s', default=ATLAS_5S_DIR)
    ap.add_argument('--out', default=DEFAULT_OUT_CSV)
    ap.add_argument('--atr-window', type=int, default=ATR_HOURS_DEFAULT,
                          help='Rolling N-hour ATR proxy window')
    ap.add_argument('--dir-threshold', type=float, default=DIR_THRESHOLD)
    ap.add_argument('--smooth-range', type=float,
                          default=SMOOTH_RANGE_THRESHOLD)
    ap.add_argument('--smooth-eff', type=float, default=SMOOTH_EFF_THRESHOLD)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.atlas_5s, '*.parquet')))
    if not files:
        print(f'!!! No 5s parquets at {args.atlas_5s}')
        sys.exit(1)

    # Walk every day, extract per-hour metrics
    all_rows = []
    print(f'Processing {len(files)} day files...')
    for path in tqdm(files, desc='hourly metrics'):
        day = os.path.basename(path).replace('.parquet', '')
        df = pd.read_parquet(path)
        df = _normalize_ts(df).sort_values('timestamp').reset_index(drop=True)
        if len(df) == 0:
            continue
        rows = compute_hourly_metrics(day, df)
        all_rows.extend(rows)

    if not all_rows:
        print('!!! No hourly metrics computed; aborting')
        sys.exit(1)

    df = pd.DataFrame(all_rows).sort_values('hour_start_ts').reset_index(drop=True)

    # Rolling ATR proxy: mean range over last N hours (excluding current)
    df['atr_proxy'] = df['range'].rolling(args.atr_window,
                                                          min_periods=4).mean().shift(1)
    df['range_expansion'] = df['range'] / df['atr_proxy'].replace(0, np.nan)

    # Classify
    cls_results = df.apply(lambda r: classify_hourly(
        r.to_dict(), args.dir_threshold,
        args.smooth_range, args.smooth_eff), axis=1)
    df['direction'] = [t[0] for t in cls_results]
    df['variation'] = [t[1] for t in cls_results]
    df['regime_2d'] = [t[2] for t in cls_results]

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_cols = ['day', 'hour_start_ts', 'hour_end_ts',
                      'open', 'close', 'high', 'low', 'range', 'net_move',
                      'directional_strength', 'efficiency_ratio',
                      'atr_proxy', 'range_expansion',
                      'direction', 'variation', 'regime_2d', 'n_bars']
    df[out_cols].to_csv(args.out, index=False)
    print(f'\nHourly regime labels -> {args.out}')

    # Distribution diagnostics
    print(f'\n{"=" * 80}')
    print(f'HOURLY REGIME DISTRIBUTION  (n_hours = {len(df)})')
    print(f'{"=" * 80}')
    counts = df['regime_2d'].value_counts()
    total = len(df)
    for label, n in counts.items():
        print(f'  {label:<14} {n:>5}  ({n/total*100:>4.1f}%)')

    # Vs daily labels — how often does the hour-end agree with the day's label?
    daily_csv = 'DATA/ATLAS/regime_labels_2d.csv'
    if os.path.exists(daily_csv):
        daily = pd.read_csv(daily_csv)
        # Last hour of each day vs the full-day label
        last_hour_per_day = df.groupby('day').tail(1).set_index('day')[
            'regime_2d']
        if 'day' in daily.columns:
            daily.set_index('day', inplace=True)
        else:
            daily.set_index(daily.columns[0], inplace=True)
        if 'regime_2d' in daily.columns:
            joined = pd.concat([
                daily['regime_2d'].rename('daily'),
                last_hour_per_day.rename('hourly_lasthr'),
            ], axis=1).dropna()
            joined['daily'] = joined['daily'].astype(str)
            joined['hourly_lasthr'] = joined['hourly_lasthr'].astype(str)
            agree = (joined['daily'] == joined['hourly_lasthr']).mean()
            print(f'\nAgreement (daily label == last hourly label): '
                      f'{agree:.1%} of {len(joined)} days')


if __name__ == '__main__':
    main()
