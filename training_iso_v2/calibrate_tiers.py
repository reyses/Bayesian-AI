"""Calibrate V2-native tier thresholds from actual IS feature distributions.

Walks the V2 ticker on a sample of IS days. At each bar where the NMP seed
qualifies (`|z_se_w| >= 1.8 + reversion_prob_w >= 0.55`), captures the
features each tier filters on:

    velocity_1m  : |L2_1m_price_velocity_w|       (FADE/RIDE CALM/MOMENTUM, FREIGHT_TRAIN)
    velocity_1h  : L2_1h_price_velocity_w (signed) (FADE/RIDE_AGAINST, CASCADE)
    swing_noise  : L3_1m_swing_noise_15            (FREIGHT_TRAIN)
    hurst        : L3_1m_hurst_15                  (FREIGHT_TRAIN)
    wick_5m_dir  : directional_wick(5m bar, dir)   (KILL_SHOT, CASCADE)
    wick_15m_dir : directional_wick(15m bar, dir)  (KILL_SHOT, CASCADE)

Outputs `training_iso_v2/output/tier_thresholds.json` with quantile-derived
thresholds per tier. The legacy defaults were V1-units and need V2-unit
recalibration; this script does it from observed data.

Quantile assignment philosophy:
  - "low" / "calm"     -> Q30 of |feature|
  - "moderate" / "high"-> Q80 of |feature|
  - "extreme"          -> Q95
  - "compressed"       -> Q20 (low end of swing_noise)
  - "wick rejection"   -> Q70 (top 30% of directional wicks)

Usage:
    python -m training_iso_v2.calibrate_tiers
    python -m training_iso_v2.calibrate_tiers --n-days 50 --out path/to.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.ticker import MultiDayV2Ticker
from training_iso_v2.v2_cols import (price_velocity_w, swing_noise_w, hurst_w,
                                                z_se_w, reversion_prob_w)
from training_iso_v2.wicks import directional_wick


def collect_qualifying_bars(days: List[str]) -> pd.DataFrame:
    """At each NMP-qualifying bar (1m close), capture tier-relevant features."""
    multi = MultiDayV2Ticker(days=days)

    rows = []
    z_col = z_se_w('1m')
    r_col = reversion_prob_w('1m')
    v1m_col = price_velocity_w('1m')
    v1h_col = price_velocity_w('1h')
    sn_col = swing_noise_w('1m')
    h_col = hurst_w('1m')

    for state in tqdm(multi, total=10000, desc='calibrate'):
        if not state.is_1m_close:
            continue
        z = state.get(z_col, 0.0)
        r = state.get(r_col, 0.0)
        if abs(z) < 1.8 or r < 0.55:
            continue
        direction = 'short' if z > 0 else 'long'

        # Wick at most-recent CLOSED 5m / 15m bar (already lookahead-free in ticker)
        wick_5m = 0.0
        if state.ohlcv_5m is not None:
            o, h, l, c = (state.ohlcv_5m['open'], state.ohlcv_5m['high'],
                                 state.ohlcv_5m['low'], state.ohlcv_5m['close'])
            wick_5m = directional_wick(o, h, l, c, direction)
        wick_15m = 0.0
        if state.ohlcv_15m is not None:
            o, h, l, c = (state.ohlcv_15m['open'], state.ohlcv_15m['high'],
                                 state.ohlcv_15m['low'], state.ohlcv_15m['close'])
            wick_15m = directional_wick(o, h, l, c, direction)

        rows.append({
            'day': state.day, 'ts': state.timestamp,
            'direction': direction,
            'regime_idx': state.regime_idx,
            'z_se': z,
            'reversion_prob': r,
            'vel_1m_abs': abs(state.get(v1m_col, 0.0)),
            'vel_1m_signed': state.get(v1m_col, 0.0),
            'vel_1h_signed': state.get(v1h_col, 0.0),
            'swing_noise_1m': state.get(sn_col, 0.0),
            'hurst_1m': state.get(h_col, 0.0),
            'wick_5m': wick_5m,
            'wick_15m': wick_15m,
        })
    return pd.DataFrame(rows)


def derive_thresholds(df: pd.DataFrame) -> Dict:
    """Quantile-based threshold extraction per tier."""
    if len(df) < 100:
        raise RuntimeError(f'Too few qualifying bars: {len(df)}. Need more sample days.')

    # NaN-safe quantile (warmup periods can leave higher-TF columns blank)
    def q(col, p):
        vals = df[col].dropna()
        return float(vals.quantile(p)) if len(vals) > 10 else 0.0

    # Velocity quantiles (1m, abs)
    v_q30 = q('vel_1m_abs', 0.30)
    v_q80 = q('vel_1m_abs', 0.80)
    v_q95 = q('vel_1m_abs', 0.95)

    # Wick quantiles (directional, at NMP-qualifying bars only)
    w5_q60 = q('wick_5m', 0.60)
    w5_q75 = q('wick_5m', 0.75)
    w15_q60 = q('wick_15m', 0.60)
    w15_q75 = q('wick_15m', 0.75)

    # Swing noise (low = compressed)
    sn_q20 = q('swing_noise_1m', 0.20)

    # Hurst (low = mean-reverting)
    h_q40 = q('hurst_1m', 0.40)

    # 1h velocity threshold for "non-trivial" 1h movement
    # Use the magnitude at Q60 as the "must exceed this to count as opposing/aligned"
    h1_q60 = q('vel_1h_signed', 0.60)
    h1_q40 = q('vel_1h_signed', 0.40)
    # For threshold of "opposing", use the 60th absolute percentile
    h1_abs_clean = np.abs(df['vel_1h_signed'].dropna().values)
    h1_thr = float(np.quantile(h1_abs_clean, 0.60)) if len(h1_abs_clean) > 10 else 0.0

    thresholds = {
        '_meta': {
            'n_qualifying_bars': int(len(df)),
            'method': 'quantile-based; calibrated from NMP-qualifying entry bars',
            'note': 'velocity is |L2_1m_price_velocity_w|; wicks are directional at NMP entry direction',
            'distributions': {
                'vel_1m_abs': {
                    'q10': q('vel_1m_abs', 0.10),
                    'q30': v_q30, 'q50': q('vel_1m_abs', 0.50),
                    'q80': v_q80, 'q95': v_q95,
                    'mean': float(df['vel_1m_abs'].mean()),
                },
                'wick_5m': {
                    'q30': q('wick_5m', 0.30), 'q60': w5_q60,
                    'q75': w5_q75, 'q90': q('wick_5m', 0.90),
                    'mean': float(df['wick_5m'].mean()),
                },
                'wick_15m': {
                    'q30': q('wick_15m', 0.30), 'q60': w15_q60,
                    'q75': w15_q75, 'q90': q('wick_15m', 0.90),
                    'mean': float(df['wick_15m'].mean()),
                },
                'swing_noise_1m': {
                    'q20': sn_q20, 'q50': q('swing_noise_1m', 0.50),
                    'q80': q('swing_noise_1m', 0.80),
                    'mean': float(df['swing_noise_1m'].mean()),
                },
                'hurst_1m': {
                    'q20': q('hurst_1m', 0.20), 'q40': h_q40,
                    'q60': q('hurst_1m', 0.60), 'q80': q('hurst_1m', 0.80),
                    'mean': float(df['hurst_1m'].mean()),
                },
            },
        },
        'FADE_CALM':     {'calm_velocity': v_q30},
        'FADE_MOMENTUM': {'momentum_velocity': v_q80},
        'RIDE_CALM':     {'calm_velocity': v_q30},
        'RIDE_MOMENTUM': {'momentum_velocity': v_q80},
        'FADE_AGAINST':  {'h1_vel_threshold': h1_thr},
        'RIDE_AGAINST':  {'h1_vel_threshold': h1_thr},
        'KILL_SHOT': {
            'wick_5m_min': w5_q60,
            'wick_15m_min': w15_q60,
        },
        'CASCADE': {
            'wick_5m_min': w5_q60,
            'wick_15m_min': w15_q60,
            'h1_vel_align_min': h1_thr * 0.5,  # easier alignment criterion
        },
        'FREIGHT_TRAIN': {
            'extreme_velocity': v_q95,
            'low_noise': sn_q20,
            'hurst_revert': h_q40,
        },
    }
    return thresholds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-days', type=int, default=30,
                       help='Number of IS days to sample (evenly spaced)')
    p.add_argument('--out', default='training_iso_v2/output/tier_thresholds.json')
    p.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    args = p.parse_args()

    # Sample IS days evenly
    l0_dir = os.path.join(args.features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    is_days = [os.path.basename(f).replace('.parquet', '') for f in files
                    if os.path.basename(f).startswith('2025_')]
    if len(is_days) > args.n_days:
        # Evenly-spaced sample
        idx = np.linspace(0, len(is_days) - 1, args.n_days, dtype=int)
        sample_days = [is_days[i] for i in idx]
    else:
        sample_days = is_days
    print(f'Sampling {len(sample_days)} IS days: {sample_days[:5]} ... {sample_days[-3:]}')

    df = collect_qualifying_bars(sample_days)
    print(f'\nCollected {len(df)} NMP-qualifying bars')
    if len(df) < 100:
        print('Too few bars to derive stable thresholds. Increase --n-days.')
        return

    thresholds = derive_thresholds(df)

    print(f'\nFeature distributions (NMP-qualifying bars):')
    print(f'  vel_1m_abs: Q30={thresholds["_meta"]["distributions"]["vel_1m_abs"]["q30"]:.2f}, '
              f'Q80={thresholds["_meta"]["distributions"]["vel_1m_abs"]["q80"]:.2f}, '
              f'Q95={thresholds["_meta"]["distributions"]["vel_1m_abs"]["q95"]:.2f}')
    print(f'  wick_5m: Q60={thresholds["_meta"]["distributions"]["wick_5m"]["q60"]:.3f}, '
              f'Q75={thresholds["_meta"]["distributions"]["wick_5m"]["q75"]:.3f}')
    print(f'  wick_15m: Q60={thresholds["_meta"]["distributions"]["wick_15m"]["q60"]:.3f}, '
              f'Q75={thresholds["_meta"]["distributions"]["wick_15m"]["q75"]:.3f}')
    print(f'  swing_noise_1m: Q20={thresholds["_meta"]["distributions"]["swing_noise_1m"]["q20"]:.1f}')
    print(f'  hurst_1m: Q40={thresholds["_meta"]["distributions"]["hurst_1m"]["q40"]:.3f}')

    print(f'\nTier thresholds:')
    for tier, thr in thresholds.items():
        if tier.startswith('_'): continue
        print(f'  {tier:<16}: {thr}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
