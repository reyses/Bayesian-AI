"""
regime_envelope_quality.py -- 1h regime + z-band envelope filter for 1m pivots.
================================================================================

Framework:
  1. Compute 1h zigzag pivots at R_1h. The most recent 1h pivot defines the
     "macro regime direction" -- LONG after a 1h LOW pivot, SHORT after a 1h
     HIGH pivot.
  2. At each 1m pivot, look up `1h_z_se` (already in FEATURES_5s/) at the 1m
     pivot's confirm timestamp. This is "where price sits in 1h's regression
     band" -- the envelope position.
  3. Bucket each 1m pivot by:
        a) WITH-regime  vs AGAINST-regime  (does the 1m flip agree with macro?)
        b) Envelope position: |1h_z_se|  binned by sigma-distance from regression
           mean (the 1h-band depth at 1m pivot moment).
  4. Measure leg outcome (MFE, MAE, ETD, final P&L) per bucket.

Hypothesis (user's framework):
  WITH-regime + outside-envelope (|1h_z|>1.5) = best trades (real continuation).
  AGAINST-regime + inside-envelope = fakeouts to skip.
  Envelope half-width sets the minimax (expected MAE).

Usage:
    python tools/regime_envelope_quality.py --r 30 --r1h 50
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.zigzag_backtest import zigzag_pivots_with_confirmation

DOLLAR_PER_POINT = 2.0
ATLAS_ROOT = 'DATA/ATLAS'
FEATURES_DIR = os.path.join(ATLAS_ROOT, 'FEATURES_5s')


# ── Loaders ───────────────────────────────────────────────────────────────

def load_bars(atlas_root: str, day_label: str, tf: str) -> pd.DataFrame | None:
    p = os.path.join(atlas_root, tf, f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


def load_features(features_dir: str, day_label: str) -> pd.DataFrame | None:
    p = os.path.join(features_dir, f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


# ── Regime computation ────────────────────────────────────────────────────

def regime_at_timestamps(pivots_1h: list,
                          ts_1h: np.ndarray,
                          period_seconds: int,
                          query_ts: np.ndarray) -> np.ndarray:
    """For each query_ts, return +1 (LONG regime) / -1 (SHORT) / 0 (no regime yet).

    Regime flips on each 1h pivot CONFIRMATION:
      - LOW pivot at confirm time T  -> regime LONG from T forward
      - HIGH pivot at confirm time T -> regime SHORT from T forward
    """
    if not pivots_1h:
        return np.zeros(len(query_ts), dtype=np.int8)

    # Build (confirm_ts, regime) timeline from the 1h pivot list.
    # confirm_ts = ts_1h[confirm_idx] + period_seconds (= bar close)
    flips = []
    for ext_idx, ext_price, kind, confirm_idx in pivots_1h:
        if confirm_idx >= len(ts_1h):
            continue
        flip_ts = int(ts_1h[confirm_idx]) + period_seconds
        regime = +1 if kind == 'low' else -1
        flips.append((flip_ts, regime))
    flips.sort()
    if not flips:
        return np.zeros(len(query_ts), dtype=np.int8)

    flip_ts_arr = np.array([f[0] for f in flips], dtype=np.int64)
    flip_reg_arr = np.array([f[1] for f in flips], dtype=np.int8)

    # For each query, find the most recent flip at or before it.
    idx = np.searchsorted(flip_ts_arr, query_ts, side='right') - 1
    out = np.zeros(len(query_ts), dtype=np.int8)
    valid = idx >= 0
    out[valid] = flip_reg_arr[idx[valid]]
    return out


def value_at_timestamp(df_features: pd.DataFrame, col: str, query_ts: np.ndarray) -> np.ndarray:
    if col not in df_features.columns:
        return np.zeros(len(query_ts))
    ts_arr = df_features['timestamp'].values.astype(np.int64)
    vals = df_features[col].values.astype(np.float64)
    idx = np.clip(np.searchsorted(ts_arr, query_ts, side='right') - 1, 0, len(vals) - 1)
    return vals[idx]


# ── Leg outcome ───────────────────────────────────────────────────────────

def measure_leg(closes_1m, ts_1m, lows_1m, highs_1m, entry_idx, exit_idx, direction):
    if entry_idx >= len(closes_1m) or exit_idx >= len(closes_1m) or exit_idx <= entry_idx:
        return None
    entry_p = float(closes_1m[entry_idx])
    exit_p = float(closes_1m[exit_idx])
    held_high = float(highs_1m[entry_idx:exit_idx + 1].max())
    held_low  = float(lows_1m[entry_idx:exit_idx + 1].min())
    if direction > 0:
        mfe = held_high - entry_p
        mae = entry_p - held_low
        leg = exit_p - entry_p
    else:
        mfe = entry_p - held_low
        mae = held_high - entry_p
        leg = entry_p - exit_p
    return dict(
        entry_p=entry_p, exit_p=exit_p,
        mfe_pts=float(mfe), mae_pts=float(mae),
        leg_pts=float(leg), etd_pts=float(mfe - leg),
    )


# ── Per-day pipeline ──────────────────────────────────────────────────────

def process_day(day_label, atlas_root, features_dir, r_1m, r_1h):
    df_1m = load_bars(atlas_root, day_label, '1m')
    df_1h = load_bars(atlas_root, day_label, '1h')
    df_feat = load_features(features_dir, day_label)
    if df_1m is None or df_1h is None or df_feat is None or len(df_1m) < 2 or len(df_1h) < 2:
        return []

    closes_1m = df_1m['close'].values.astype(np.float64)
    ts_1m_arr = df_1m['timestamp'].values.astype(np.int64)
    lows_1m   = df_1m['low'].values.astype(np.float64)
    highs_1m  = df_1m['high'].values.astype(np.float64)

    closes_1h = df_1h['close'].values.astype(np.float64)
    ts_1h_arr = df_1h['timestamp'].values.astype(np.int64)

    pivots_1m = zigzag_pivots_with_confirmation(closes_1m, r_1m)
    pivots_1h = zigzag_pivots_with_confirmation(closes_1h, r_1h)
    if len(pivots_1m) < 2:
        return []

    # Build per-pivot rows
    confirm_ts_list = []
    for ext_idx, ext_price, kind, confirm_idx in pivots_1m[:-1]:
        confirm_ts_list.append(int(ts_1m_arr[confirm_idx]) + 60)
    confirm_ts_arr = np.array(confirm_ts_list, dtype=np.int64)

    # Macro regime at each 1m pivot's confirm moment (period for 1h = 3600s).
    regimes = regime_at_timestamps(pivots_1h, ts_1h_arr, 3600, confirm_ts_arr)

    # 1h envelope position (1h_z_se) at each 1m pivot moment.
    z_1h = value_at_timestamp(df_feat, '1h_z_se', confirm_ts_arr)
    z_15m = value_at_timestamp(df_feat, '15m_z_se', confirm_ts_arr)
    z_5m  = value_at_timestamp(df_feat, '5m_z_se', confirm_ts_arr)
    z_1m  = value_at_timestamp(df_feat, '1m_z_se', confirm_ts_arr)
    z_1D  = value_at_timestamp(df_feat, '1D_z_se', confirm_ts_arr)

    out = []
    for i in range(len(pivots_1m) - 1):
        ext_idx, ext_price, kind, confirm_idx = pivots_1m[i]
        _, _, _, next_confirm_idx = pivots_1m[i + 1]
        entry_idx = confirm_idx + 1
        exit_idx = next_confirm_idx + 1
        if exit_idx >= len(closes_1m):
            continue

        direction = +1 if kind == 'low' else -1
        leg = measure_leg(closes_1m, ts_1m_arr, lows_1m, highs_1m, entry_idx, exit_idx, direction)
        if leg is None:
            continue

        regime = int(regimes[i])
        with_regime = (regime == direction) if regime != 0 else None  # None = no regime yet
        z_envelope = float(z_1h[i])

        # Sigma-distance bin: |z_1h| binned into [<0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, >2.0]
        abs_z = abs(z_envelope)
        if   abs_z < 0.5:  band = 'inside'    # noise band
        elif abs_z < 1.0:  band = 'near'      # mid
        elif abs_z < 1.5:  band = 'edge'      # 1-sigma+
        elif abs_z < 2.0:  band = 'outside'   # 1.5-sigma+
        else:              band = 'extreme'   # 2+ sigma

        out.append(dict(
            day=day_label, confirm_ts=int(confirm_ts_arr[i]),
            kind=kind, direction=direction,
            ext_price=float(ext_price),
            regime=regime,
            with_regime=with_regime,
            z_1h_envelope=z_envelope,
            band=band,
            z_15m=float(z_15m[i]), z_5m=float(z_5m[i]),
            z_1m=float(z_1m[i]), z_1D=float(z_1D[i]),
            **leg,
        ))
    return out


# ── Reporting ─────────────────────────────────────────────────────────────

def is_2025(d): return d.startswith('2025_')
def is_2026(d): return d.startswith('2026_')

BAND_ORDER = ['inside', 'near', 'edge', 'outside', 'extreme']


def print_2x2(label, df):
    """2x2 table: regime alignment x band."""
    print(f'\n=== {label} (N={len(df):,}) ===')
    df = df[df['regime'] != 0]   # require macro regime to exist
    if len(df) == 0:
        print('  (no rows w/ macro regime)')
        return
    print(f'{"Regime":>14} {"Band":>10} {"N":>6} {"WR%":>6} '
          f'{"MFE$":>9} {"MAE$":>9} {"ETD$":>9} {"Final$":>10}')
    for with_regime, label_r in [(True, 'WITH-regime'), (False, 'AGAINST-regime')]:
        for band in BAND_ORDER:
            sub = df[(df['with_regime'] == with_regime) & (df['band'] == band)]
            if len(sub) == 0:
                continue
            wr = (sub['leg_pts'] > 0).mean() * 100
            mfe = sub['mfe_pts'].mean() * DOLLAR_PER_POINT
            mae = sub['mae_pts'].mean() * DOLLAR_PER_POINT
            etd = sub['etd_pts'].mean() * DOLLAR_PER_POINT
            final = sub['leg_pts'].mean() * DOLLAR_PER_POINT
            print(f'{label_r:>14} {band:>10} {len(sub):>6,} {wr:>5.1f}% '
                  f'${mfe:>+7.2f} ${mae:>+7.2f} ${etd:>+7.2f} ${final:>+8.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default=FEATURES_DIR)
    ap.add_argument('--atlas', default=ATLAS_ROOT)
    ap.add_argument('--r', type=float, default=30.0, help='1m zigzag R (points)')
    ap.add_argument('--r1h', type=float, default=50.0, help='1h zigzag R (points)')
    ap.add_argument('--day', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.day:
        days = [args.day.replace('-', '_')]
    else:
        files = sorted(glob.glob(os.path.join(args.features_dir, '*.parquet')))
        days = [os.path.splitext(os.path.basename(p))[0] for p in files]

    print('=' * 100)
    print(f'REGIME-ENVELOPE PIVOT QUALITY  R_1m={args.r:g}  R_1h={args.r1h:g}')
    print(f'Days: {len(days)}  |  envelope = |1h_z_se|  bands {BAND_ORDER}')
    print('=' * 100)

    rows = []
    for d in (tqdm(days, desc='days') if len(days) > 1 else days):
        rows.extend(process_day(d, args.atlas, args.features_dir, args.r, args.r1h))

    if not rows:
        print('No data.')
        return

    df = pd.DataFrame(rows)
    is_df  = df[df['day'].apply(is_2025)]
    oos_df = df[df['day'].apply(is_2026)]
    print_2x2(f'IS  2025  R_1m={args.r:g} R_1h={args.r1h:g}', is_df)
    print_2x2(f'OOS 2026  R_1m={args.r:g} R_1h={args.r1h:g}', oos_df)

    # Save
    out_csv = args.out or f'reports/findings/regime_envelope_R1m{args.r:g}_R1h{args.r1h:g}.csv'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv} ({len(df):,} rows)')


if __name__ == '__main__':
    main()
