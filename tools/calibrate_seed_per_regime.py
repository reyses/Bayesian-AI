"""Per-regime calibration of the NMP seed (z_threshold, r_threshold).

User insight: a universal `|z_se|>=1.8 AND rprob>=0.55` threshold STARVES
calm regimes (z rarely reaches 1.8) and over-fires choppy ones. Each regime
has its own z-volatility distribution and its own forward-edge curve.

Method:
    1. Walk the V2 ticker on IS days. At every 1m close, capture
       (regime, z_se_1m, rprob_1m, entry_price, day, ts).
    2. Per day, pre-load 5s OHLCV. For each captured bar, compute forward
       fade-direction PnL at 30 min and forward MFE (max favorable
       excursion) within 60 min.
    3. Per regime, sweep (z_threshold × r_threshold) grids. For each
       (z_thr, r_thr) combo:
         - subset = bars where |z|>=z_thr AND rprob>=r_thr
         - mean_fwd30, mean_mfe60, n
       Pick the combo per regime that maximizes mean_fwd30 (with min-n
       guard to avoid micro-cells).

Outputs:
    training_iso_v2/output/seed_thresholds_per_regime.json
        {regime: {'z_thr': float, 'r_thr': float, 'mean_fwd30': float,
                  'mean_mfe60': float, 'n': int}}
    reports/findings/seed_calibration_per_regime/sweep.csv  (full grid)
    reports/findings/seed_calibration_per_regime/raw_bars.parquet  (per-bar)

Usage:
    python tools/calibrate_seed_per_regime.py
    python tools/calibrate_seed_per_regime.py --n-days 50
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.ticker import MultiDayV2Ticker
from training_iso_v2.state import REGIME_VOCAB
from training_iso_v2.v2_cols import z_se_w, reversion_prob_w
from training_iso_v2.ledger import TICK, TICK_VALUE
from training_iso_v2.regret import _load_5s_ohlcv


FORWARD_BARS_30M = 360         # 30 min at 5s anchor
HORIZON_BARS_60M = 720         # 60 min — used for MFE

Z_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
R_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_N_PER_CELL = 50            # sample-size guard for grid sweep


def collect_bars(days: List[str]) -> pd.DataFrame:
    """Walk the ticker; record every 1m-close bar's (regime, z, rprob, price, day, ts)."""
    multi = MultiDayV2Ticker(days=days)

    z_col = z_se_w('1m')
    r_col = reversion_prob_w('1m')

    by_day = defaultdict(list)
    for state in tqdm(multi, total=10000, desc='collect 1m bars'):
        if not state.is_1m_close:
            continue
        z = state.get(z_col, np.nan)
        if not np.isfinite(z):
            continue
        r = state.get(r_col, np.nan)
        if not np.isfinite(r):
            continue
        by_day[state.day].append({
            'ts': state.timestamp,
            'price': state.price,
            'regime': state.regime_idx,
            'z': float(z),
            'rprob': float(r),
        })

    print(f'\nCollected raw 1m-close bars across {len(by_day)} days, '
              f'{sum(len(v) for v in by_day.values())} total')
    return by_day


def add_forward_pnl(by_day: dict) -> pd.DataFrame:
    """For each day, load 5s OHLCV; compute fwd 30m PnL + MFE 60m per bar."""
    rows = []
    for day, recs in tqdm(by_day.items(), desc='forward pnl'):
        ohlcv = _load_5s_ohlcv(day)
        if ohlcv is None or len(ohlcv) == 0:
            continue
        ts_arr = ohlcv['timestamp'].values.astype(np.int64)
        close = ohlcv['close'].values

        for r in recs:
            idx = int(np.searchsorted(ts_arr, r['ts']))
            if idx >= len(ts_arr):
                continue
            entry_price = float(close[idx]) if idx < len(close) else r['price']
            end = min(idx + HORIZON_BARS_60M, len(close))
            if end - idx < 60:           # need at least 5 min of lookahead
                continue
            future = close[idx:end]

            # NMP fade direction
            direction = 'short' if r['z'] > 0 else 'long'
            if direction == 'long':
                pnl_path = (future - entry_price) / TICK * TICK_VALUE
            else:
                pnl_path = (entry_price - future) / TICK * TICK_VALUE

            fwd_30 = float(pnl_path[min(FORWARD_BARS_30M, len(pnl_path) - 1)])
            mfe_60 = float(pnl_path.max())

            rows.append({
                'day': day, 'ts': r['ts'], 'regime': r['regime'],
                'absz': abs(r['z']), 'z': r['z'], 'rprob': r['rprob'],
                'direction': direction,
                'fwd_pnl_30m': fwd_30, 'mfe_60m': mfe_60,
            })

    df = pd.DataFrame(rows)
    print(f'Computed forward-PnL for {len(df)} bars')
    return df


def sweep_thresholds(df: pd.DataFrame) -> tuple:
    """Per regime, sweep z_thr × r_thr grids; return per-regime optimum and full grid."""
    rows = []
    by_regime_best = {}

    for ridx in sorted(df['regime'].unique()):
        regime_label = (REGIME_VOCAB[ridx]
                              if 0 <= ridx < len(REGIME_VOCAB) else f'IDX{ridx}')
        sub = df[df['regime'] == ridx]
        best = None
        for z_thr in Z_GRID:
            for r_thr in R_GRID:
                qual = sub[(sub['absz'] >= z_thr) & (sub['rprob'] >= r_thr)]
                n = len(qual)
                if n < MIN_N_PER_CELL:
                    continue
                mean_fwd = float(qual['fwd_pnl_30m'].mean())
                mean_mfe = float(qual['mfe_60m'].mean())
                rec = {
                    'regime': regime_label, 'regime_idx': int(ridx),
                    'z_thr': z_thr, 'r_thr': r_thr, 'n': n,
                    'mean_fwd30': mean_fwd, 'mean_mfe60': mean_mfe,
                }
                rows.append(rec)
                if best is None or mean_fwd > best['mean_fwd30']:
                    best = rec
        if best is not None:
            by_regime_best[regime_label] = best

    return pd.DataFrame(rows), by_regime_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-days', type=int, default=80,
                          help='Number of IS days to sample (evenly spaced)')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--out-json',
                          default='training_iso_v2/output/seed_thresholds_per_regime.json')
    ap.add_argument('--out-dir',
                          default='reports/findings/seed_calibration_per_regime')
    args = ap.parse_args()

    # Sample IS days evenly
    l0_dir = os.path.join(args.features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    is_days = [os.path.basename(f).replace('.parquet', '') for f in files
                    if os.path.basename(f).startswith('2025_')]
    if len(is_days) > args.n_days:
        idx = np.linspace(0, len(is_days) - 1, args.n_days, dtype=int)
        sample_days = [is_days[i] for i in idx]
    else:
        sample_days = is_days
    print(f'Sampling {len(sample_days)} IS days: {sample_days[:3]} ... {sample_days[-3:]}')

    by_day = collect_bars(sample_days)
    df = add_forward_pnl(by_day)
    if df.empty:
        print('!!! No bars with valid forward PnL collected. Possible causes: '
                  'missing 5s OHLCV in DATA/ATLAS/5s/, schema drift in V2 features, '
                  'or all sampled days outside trading hours. Aborting non-zero so '
                  'the pipeline catches it.')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_parquet(os.path.join(args.out_dir, 'raw_bars.parquet'), index=False)
    print(f'Raw per-bar dataframe -> {os.path.join(args.out_dir, "raw_bars.parquet")}')

    # ── Per-regime z-distribution diagnostic (the user's question) ──────
    print(f'\n{"=" * 90}')
    print(f'PER-REGIME |z| DISTRIBUTION  (does each regime even REACH the universal 1.8?)')
    print(f'{"=" * 90}')
    print(f'{"regime":<14} {"n":>7} {"|z|.q50":>8} {"|z|.q75":>8} {"|z|.q90":>8} '
              f'{"|z|.q95":>8}  {"%>=1.8":>8} {"%>=1.5":>8} {"%>=1.2":>8}')
    print('-' * 90)
    for ridx in sorted(df['regime'].unique()):
        regime_label = (REGIME_VOCAB[ridx]
                              if 0 <= ridx < len(REGIME_VOCAB) else f'IDX{ridx}')
        sub = df[df['regime'] == ridx]
        absz = sub['absz']
        n = len(sub)
        if n < 50: continue
        q = absz.quantile
        pct18 = (absz >= 1.8).mean() * 100
        pct15 = (absz >= 1.5).mean() * 100
        pct12 = (absz >= 1.2).mean() * 100
        print(f'{regime_label:<14} {n:>7} {q(0.50):>8.2f} {q(0.75):>8.2f} '
                  f'{q(0.90):>8.2f} {q(0.95):>8.2f}  {pct18:>7.1f}% '
                  f'{pct15:>7.1f}% {pct12:>7.1f}%')

    # ── Threshold sweep ─────────────────────────────────────────────────
    sweep_df, best_per_regime = sweep_thresholds(df)
    sweep_path = os.path.join(args.out_dir, 'sweep.csv')
    sweep_df.to_csv(sweep_path, index=False)
    print(f'\nFull sweep CSV -> {sweep_path}')

    print(f'\n{"=" * 110}')
    print(f'PER-REGIME OPTIMAL SEED THRESHOLDS  (max mean_fwd30 with n>={MIN_N_PER_CELL})')
    print(f'{"=" * 110}')
    print(f'{"regime":<14} {"z_thr":>6} {"r_thr":>6} {"n":>7} '
              f'{"$/30m mean":>11} {"$mfe60 mean":>12}   '
              f'{"Δ vs (1.8, 0.55)":>20}')
    print('-' * 110)
    for regime_label, best in best_per_regime.items():
        # Default-cell stats (1.8, 0.55) for comparison
        ridx = best['regime_idx']
        ref = sweep_df[(sweep_df['regime_idx'] == ridx)
                              & (sweep_df['z_thr'] == 1.8)
                              & (sweep_df['r_thr'] == 0.55)]
        if len(ref) > 0:
            ref_fwd = float(ref['mean_fwd30'].iloc[0])
            ref_n = int(ref['n'].iloc[0])
            delta_str = f'+${best["mean_fwd30"] - ref_fwd:>+5.2f} (n {ref_n}->{best["n"]})'
        else:
            delta_str = f'(no n>={MIN_N_PER_CELL} at default 1.8/0.55)'
        print(f'{regime_label:<14} {best["z_thr"]:>6.1f} {best["r_thr"]:>6.2f} '
                  f'{best["n"]:>7} '
                  f'${best["mean_fwd30"]:>+10.2f} ${best["mean_mfe60"]:>+11.2f}   '
                  f'{delta_str:>20}')

    # ── Save JSON ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        '_meta': {
            'method': ('per-regime sweep of z_thr × r_thr; objective = '
                            'max mean fwd-30m fade-direction PnL; min-n guard.'),
            'horizon_30m_bars': FORWARD_BARS_30M,
            'horizon_mfe_bars': HORIZON_BARS_60M,
            'min_n_per_cell': MIN_N_PER_CELL,
            'z_grid': Z_GRID, 'r_grid': R_GRID,
            'n_days_sampled': len(sample_days),
        },
    }
    for regime_label, best in best_per_regime.items():
        out[regime_label] = {
            'z_thr': float(best['z_thr']),
            'r_thr': float(best['r_thr']),
            'n_qualifying': int(best['n']),
            'mean_fwd30_pnl': float(best['mean_fwd30']),
            'mean_mfe60_pnl': float(best['mean_mfe60']),
        }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nPer-regime JSON -> {args.out_json}')


if __name__ == '__main__':
    main()
