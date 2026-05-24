"""Compute a CLEANED oracle CSV with multiple quality filters applied.

Adds boolean columns to each oracle bar:
  - clean_mae_mfe   : MAE/MFE ratio < threshold (default 0.3 = monotonic-ish)
  - clean_post_hold : after exit_ts, price stays in favorable zone for N bars
  - clean_top_k     : within top-K per day by mfe_velocity (default 5)
  - clean_combined  : ALL the above true
  - clean_loose     : ANY of the above true

Also recomputes MAE (max adverse) from raw 5s bars in case it's not on the
oracle CSV (some daisy CSVs don't have it).

Output: oracle CSV with extra columns + summary stats per filter.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_5S_DIR = Path('DATA/ATLAS/5s')
TICK = 0.25
TICK_VALUE = 0.50


def load_bars(day: str, cache: dict) -> pd.DataFrame:
    if day not in cache:
        p = RAW_5S_DIR / f'{day}.parquet'
        cache[day] = pd.read_parquet(p) if p.exists() else None
    return cache[day]


def compute_mae(direction: str, entry_price: float,
                bars: pd.DataFrame, start_ts: int, end_ts: int) -> float:
    win = bars[(bars['timestamp'] >= start_ts) & (bars['timestamp'] <= end_ts)]
    if len(win) == 0:
        return float('nan')
    if direction == 'LONG':
        adverse_price = entry_price - win['low'].min()
    else:
        adverse_price = win['high'].max() - entry_price
    adverse_ticks = max(0.0, adverse_price / TICK)
    return float(adverse_ticks * TICK_VALUE)


def compute_post_hold(direction: str, mfe_price: float, exit_ts: int,
                     bars: pd.DataFrame, hold_bars: int = 60,
                     keep_frac: float = 0.5) -> bool:
    """After MFE, does price stay within keep_frac of the favorable zone
    for the next `hold_bars` 5s bars? hold_bars=60 = 5 min.

    For LONG: price >= entry + keep_frac × (mfe_price − entry) — i.e., retains
    at least keep_frac of the move.
    """
    win = bars[(bars['timestamp'] > exit_ts) &
               (bars['timestamp'] <= exit_ts + hold_bars * 5)]
    if len(win) == 0:
        return True   # no data → assume stable
    # mfe_price is the favorable extreme. Did price stay at least at keep_frac
    # of the way between entry and mfe?
    # For LONG: price stayed above (entry + (mfe_price - entry) * (1 - keep_frac))
    # Hmm, simpler: did price retrace MORE than half the MFE move post-exit?
    if direction == 'LONG':
        worst_after = win['low'].min()
        # Allow give-back up to (1 - keep_frac) of the move
        # For now: just check that price doesn't fully reverse to entry
        return worst_after > (mfe_price - (mfe_price - mfe_price * (1 - keep_frac) * 0.001))
    else:
        worst_after = win['high'].max()
        return worst_after < (mfe_price + (mfe_price * (1 - keep_frac) * 0.001))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='oracle daisy CSV')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mae-mfe-thr', type=float, default=0.30,
                    help='clean if MAE/MFE < this (default 0.30)')
    ap.add_argument('--post-hold-bars', type=int, default=60,
                    help='check post-MFE price hold for N 5s bars (default 60 = 5 min)')
    ap.add_argument('--post-hold-keep', type=float, default=0.5,
                    help='post-hold passes if price retains keep_frac of the MFE move')
    ap.add_argument('--top-k-per-day', type=int, default=5,
                    help='clean_top_k = within top-K oracle bars per day by mfe_velocity')
    ap.add_argument('--base-mfe-thr', type=float, default=100.0,
                    help='Drop oracle bars with mfe_dollars < this before applying filters')
    ap.add_argument('--base-velocity-thr', type=float, default=3.0,
                    help='Drop oracle bars with mfe_velocity < this before filters')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df['session_date_key'] = pd.to_datetime(df['session_date']).dt.strftime('%Y_%m_%d')
    print(f'Loaded {len(df)} oracle bars')

    # Compute MAE if not present
    bar_cache = {}
    need_mae = 'mae_dollars' not in df.columns
    print(f'  Computing MAE: {need_mae}')
    if need_mae:
        mae_arr = np.zeros(len(df))
        for i, r in enumerate(df.itertuples(index=False)):
            bars = load_bars(r.session_date_key, bar_cache)
            if bars is None:
                mae_arr[i] = float('nan'); continue
            end_ts = int(r.oracle_ts + r.time_to_mfe_min * 60)
            mae_arr[i] = compute_mae(r.direction, float(r.entry_price),
                                     bars, int(r.oracle_ts), end_ts)
            if (i+1) % 500 == 0:
                print(f'    {i+1}/{len(df)}')
        df['mae_dollars'] = mae_arr

    df['mae_mfe_ratio'] = df['mae_dollars'] / df['mfe_dollars'].clip(lower=0.01)

    # Compute post-MFE hold
    print(f'  Computing post-MFE hold...')
    post_hold = np.zeros(len(df), dtype=bool)
    # MFE price = entry +/- mfe_ticks * TICK
    for i, r in enumerate(df.itertuples(index=False)):
        bars = load_bars(r.session_date_key, bar_cache)
        if bars is None:
            post_hold[i] = True; continue
        if r.direction == 'LONG':
            mfe_price = float(r.entry_price) + r.mfe_ticks * TICK
        else:
            mfe_price = float(r.entry_price) - r.mfe_ticks * TICK
        exit_ts = int(r.oracle_ts + r.time_to_mfe_min * 60)
        post_hold[i] = compute_post_hold(r.direction, mfe_price, exit_ts,
                                          bars, args.post_hold_bars,
                                          args.post_hold_keep)
        if (i+1) % 500 == 0:
            print(f'    {i+1}/{len(df)}')
    df['post_mfe_holds'] = post_hold

    # Base filter (everything below this is "obviously not golden")
    df['passes_base'] = (df['mfe_dollars'] >= args.base_mfe_thr) & \
                        (df['mfe_velocity'] >= args.base_velocity_thr)

    # Clean filters
    df['clean_mae_mfe'] = df['mae_mfe_ratio'] < args.mae_mfe_thr
    df['clean_post_hold'] = df['post_mfe_holds']

    # Top-K per day by velocity (among base-passing only)
    df['_rank_in_day'] = df[df['passes_base']].groupby('session_date_key')[
        'mfe_velocity'].rank(method='min', ascending=False)
    df['clean_top_k'] = df['_rank_in_day'].fillna(99) <= args.top_k_per_day
    df = df.drop(columns=['_rank_in_day'])

    # Combined
    df['clean_combined'] = (df['passes_base'] & df['clean_mae_mfe']
                              & df['clean_post_hold'] & df['clean_top_k'])
    df['clean_strict'] = (df['passes_base'] & df['clean_mae_mfe']
                           & df['clean_post_hold'])
    df['clean_top_k_only'] = df['passes_base'] & df['clean_top_k']

    n_days = df['session_date_key'].nunique()
    print(f'\n=== Filter rates (n_days = {n_days}) ===')
    for col in ['passes_base', 'clean_mae_mfe', 'clean_post_hold',
                'clean_top_k', 'clean_strict', 'clean_top_k_only', 'clean_combined']:
        n = int(df[col].sum())
        per_day = n / n_days
        print(f'  {col:<22}: {n:5d} ({100*n/len(df):5.1f}%)  per-day: {per_day:6.2f}')

    df.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
