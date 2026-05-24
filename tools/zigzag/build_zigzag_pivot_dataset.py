"""Build entry-timing classifier dataset using ZIGZAG PIVOTS as the oracle.

Replaces the daisy-chain "high mfe_velocity" labels with ATR-adaptive zigzag
swing pivots. The user's visual calibration: ATR(14) × 4 gives swings that
track price closely without hugging every micro-wiggle.

Per 1m bar: label is_pivot = 1 if any zigzag pivot timestamp falls within
that 1m window. Pivot direction (LONG = uptrend starts, SHORT = downtrend
starts) recorded too.

Pivot SOURCE: 5s closes (precise extreme detection) — same as inspector.
ATR SOURCE: 1m bars (matches inspector display + standard ATR definition).
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.viz.auto_swing_marker import detect_swings, TICK_SIZE


DEFAULT_ATLAS_ROOT = 'DATA/ATLAS'
NT8_ATLAS_ROOT = 'DATA/ATLAS_NT8'

# These get rebound from --root in main()
V2_ROOT = Path(DEFAULT_ATLAS_ROOT) / 'FEATURES_5s_v2'
RAW_5S_DIR = Path(DEFAULT_ATLAS_ROOT) / '5s'
RAW_1M_DIR = Path(DEFAULT_ATLAS_ROOT) / '1m'
LAYER_DIRS = [
    'L1_5s', 'L1_15s', 'L1_1m', 'L1_5m', 'L1_15m', 'L1_1h', 'L1_4h', 'L1_1D',
    'L2_5s', 'L2_15s', 'L2_1m', 'L2_5m', 'L2_15m', 'L2_1h', 'L2_4h', 'L2_1D',
    'L3_5s', 'L3_15s', 'L3_1m', 'L3_5m', 'L3_15m', 'L3_1h', 'L3_4h', 'L3_1D',
]


def compute_atr(bars1m: pd.DataFrame, period: int = 14) -> float:
    h = bars1m['high'].values; l = bars1m['low'].values; c = bars1m['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())


def load_v2_at_1m_closes(day: str) -> pd.DataFrame:
    """Return DataFrame: timestamp + 184 V2 cols, filtered to the per-minute
    boundary the data actually uses (ATLAS=0, NT8=59 — detected from data).
    """
    frames = []
    for layer in LAYER_DIRS:
        p = V2_ROOT / layer / f'{day}.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        df['timestamp'] = df['timestamp'].astype('int64')
        df = df.sort_values('timestamp').reset_index(drop=True)
        frames.append(df.set_index('timestamp'))
    if not frames:
        return None
    merged = pd.concat(frames, axis=1).reset_index()
    # Detect 1m-boundary mod: in ATLAS it's 0, in NT8 it's 59. Use the most
    # common non-trivial mod-60 value (excluding sub-5s tick events).
    mods = merged['timestamp'] % 60
    valid_mods = mods[mods.isin([0, 55, 59])]   # the candidate 1m-close mods
    if len(valid_mods) == 0:
        # No clean candidate; fall back to the most common 5s-aligned value
        candidates = mods[mods % 5 == 0]   # multiples of 5
        if len(candidates) == 0:
            return merged.iloc[0:0]
        target_mod = int(candidates.mode().iloc[0])
    else:
        target_mod = int(valid_mods.mode().iloc[0])
    merged = merged[mods == target_mod].reset_index(drop=True)
    return merged


def detect_day_pivots(day: str, atr_mult: float):
    """Compute zigzag pivots for one day. Returns
    {'atr_pts': float, 'min_rev_ticks': int, 'pivots': [(ts, price, direction)]}.
    Direction: 'LONG' if pivot is a LOW (uptrend starts), 'SHORT' if HIGH.
    """
    # ATR from 1m bars
    bars1m_path = RAW_1M_DIR / f'{day}.parquet'
    if not bars1m_path.exists():
        return None
    bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
    atr_pts = compute_atr(bars1m, period=14)
    min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * atr_mult)))

    # Swings on 5s closes
    bars5s_path = RAW_5S_DIR / f'{day}.parquet'
    if not bars5s_path.exists():
        return None
    bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
    closes = bars5s['close'].values.astype(np.float64)
    ts_unix = bars5s['timestamp'].values.astype(np.int64)
    pivot_idxs = detect_swings(closes, min_reversal=min_rev_ticks,
                                min_bars=36, max_bars=0)
    pivots = []
    for k, i in enumerate(pivot_idxs):
        # Direction of the LEG STARTING at this pivot
        if k + 1 >= len(pivot_idxs):
            break
        next_i = pivot_idxs[k + 1]
        direction = 'LONG' if closes[next_i] > closes[i] else 'SHORT'
        pivots.append((int(ts_unix[i]), float(closes[i]), direction))
    return {
        'atr_pts': atr_pts,
        'min_rev_ticks': min_rev_ticks,
        'pivots': pivots,
        'date': day,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['is', 'oos', 'both'], default='is')
    ap.add_argument('--atr-mult', type=float, default=4.0,
                    help='Zigzag threshold = ATR(14) × this multiplier (default 4)')
    ap.add_argument('--days', nargs='*', default=None)
    ap.add_argument('--root', default=DEFAULT_ATLAS_ROOT,
                    help='ATLAS root: DATA/ATLAS (historical) or DATA/ATLAS_NT8 (live)')
    ap.add_argument('--start', default=None,
                    help='Earliest day YYYY_MM_DD (inclusive)')
    ap.add_argument('--end', default=None,
                    help='Latest day YYYY_MM_DD (inclusive)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Rebind data roots based on --root flag
    global V2_ROOT, RAW_5S_DIR, RAW_1M_DIR
    V2_ROOT = Path(args.root) / 'FEATURES_5s_v2'
    RAW_5S_DIR = Path(args.root) / '5s'
    RAW_1M_DIR = Path(args.root) / '1m'
    print(f'Root: {args.root}')
    print(f'  V2_ROOT: {V2_ROOT}')
    print(f'  RAW_5S: {RAW_5S_DIR}')

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve days
    if args.days:
        days = args.days
    else:
        import glob
        all_p = sorted(glob.glob(str(V2_ROOT / 'L1_5s' / '*.parquet')))
        all_days = [Path(p).stem for p in all_p]
        if args.target == 'is':
            days = [d for d in all_days if d.startswith('2025_')]
        elif args.target == 'oos':
            days = [d for d in all_days if d.startswith('2026_')]
        else:
            days = all_days
        # Apply date range filter
        if args.start:
            days = [d for d in days if d >= args.start]
        if args.end:
            days = [d for d in days if d <= args.end]
    print(f'Building dataset: {len(days)} days, ATR×{args.atr_mult}')

    all_rows = []
    n_pos_total = 0
    skipped = 0
    for day in tqdm(days, desc='days'):
        v2 = load_v2_at_1m_closes(day)
        if v2 is None or len(v2) == 0:
            skipped += 1; continue
        info = detect_day_pivots(day, args.atr_mult)
        if info is None:
            skipped += 1; continue

        v2['is_pivot'] = 0
        v2['pivot_dir'] = ''
        v2['pivot_price'] = 0.0
        v2['leg_direction'] = ''   # direction of the leg CONTAINING this bar
        v2['trend_class']  = 'NEUTRAL'  # 3-class: LONG (in up-leg) / SHORT (in down-leg) / NEUTRAL (transition)

        pivots_sorted = sorted(info['pivots'], key=lambda p: p[0])
        piv_ts_set    = [p[0] for p in pivots_sorted]

        # --- Mark pivot bars (informational only) ---
        for piv_ts, piv_px, piv_dir in pivots_sorted:
            zone_mask = ((v2['timestamp'] >= piv_ts - 60) &
                         (v2['timestamp'] < piv_ts + 60))
            if zone_mask.any():
                v2.loc[zone_mask, 'is_pivot'] = 1
                v2.loc[zone_mask, 'pivot_dir'] = piv_dir
                v2.loc[zone_mask, 'pivot_price'] = piv_px

        # --- LABEL leg_direction for ALL bars (direction of containing leg) ---
        # Each [p0, p1) interval: every bar in it has direction = p0_dir
        # (the direction of the leg starting at p0 and ending at p1).
        for k in range(len(pivots_sorted) - 1):
            p0_ts, _p0_px, p0_dir = pivots_sorted[k]
            p1_ts, _p1_px, _p1_dir = pivots_sorted[k + 1]
            mask = (v2['timestamp'] >= p0_ts) & (v2['timestamp'] < p1_ts)
            v2.loc[mask, 'leg_direction'] = p0_dir
            v2.loc[mask, 'trend_class']   = p0_dir   # provisional

        # --- Override NEUTRAL on transition zones (near pivots) ---
        # Bars within ±NEUTRAL_ZONE_S seconds of any pivot get NEUTRAL.
        # This is where direction is genuinely unclear (trend transitioning).
        NEUTRAL_ZONE_S = 120   # ±2 min around each pivot
        for pts in piv_ts_set:
            near_mask = ((v2['timestamp'] >= pts - NEUTRAL_ZONE_S) &
                         (v2['timestamp'] <= pts + NEUTRAL_ZONE_S))
            v2.loc[near_mask, 'trend_class'] = 'NEUTRAL'
        v2['day'] = day
        v2['atr_pts'] = info['atr_pts']
        v2['min_rev_ticks'] = info['min_rev_ticks']
        v2['target_split'] = args.target.upper()
        n_pos_total += int(v2['is_pivot'].sum())
        all_rows.append(v2)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_parquet(out_path, index=False)

    print(f'\nWrote: {out_path}')
    print(f'  days processed: {len(all_rows)} (skipped {skipped})')
    print(f'  total 1m bars:  {len(out_df)}')
    print(f'  pivot positives: {n_pos_total} ({100*n_pos_total/len(out_df):.2f}%)')
    print(f'  imbalance: {(len(out_df)-n_pos_total)/max(n_pos_total,1):.0f}:1')
    print(f'  pivots per day median: '
          f'{int(np.median([df["is_pivot"].sum() for df in all_rows]))}')


if __name__ == '__main__':
    main()
