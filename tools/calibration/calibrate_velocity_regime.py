"""Calibrate velocity-regime thresholds from IS feature distributions.

The velocity-based regime classifier reads:
    L2_1h_price_velocity_12   (or chosen TF) — direction signal
    L3_1h_swing_noise_12      (or chosen TF) — variation signal

Per-bar regime = (UP/DOWN/FLAT × SMOOTH/CHOPPY) classified by thresholds:
    direction:   |vel| >= vel_thr  → UP/DOWN by sign; else FLAT
    variation:   sn < sn_thr        → SMOOTH; else CHOPPY

This tool walks IS days, samples values at every 1h close, and picks
thresholds by quantile so the distribution lands at a meaningful split:

    vel_thr = q67(|vel|)        → top ~33% of |vel| qualify as directional
    sn_thr  = q50(swing_noise)  → median split for SMOOTH vs CHOPPY

Output: training_iso_v2/output/velocity_regime_thresholds.json

Usage:
    python tools/calibrate_velocity_regime.py
    python tools/calibrate_velocity_regime.py --tf 4h --n-days 80
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.ticker import MultiDayV2Ticker
from training.utils.v2_cols import price_velocity_w, swing_noise_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='1h',
                          help='TF for velocity + swing_noise (1h default)')
    ap.add_argument('--n-days', type=int, default=80,
                          help='IS days sampled (evenly spaced)')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--out',
                          default='training_iso_v2/output/velocity_regime_thresholds.json')
    ap.add_argument('--vel-quantile', type=float, default=0.67,
                          help='Quantile of |vel| above which counts as directional')
    ap.add_argument('--sn-quantile', type=float, default=0.50,
                          help='Quantile of swing_noise above which counts as CHOPPY')
    ap.add_argument('--fire-on', default='1h',
                          help='Sample at TF close (default every 1h)')
    args = ap.parse_args()

    l0_dir = os.path.join(args.features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    is_days = [os.path.basename(f).replace('.parquet', '') for f in files
                    if os.path.basename(f).startswith('2025_')]
    if len(is_days) > args.n_days:
        idx = np.linspace(0, len(is_days) - 1, args.n_days, dtype=int)
        sample_days = [is_days[i] for i in idx]
    else:
        sample_days = is_days
    print(f'Sampling {len(sample_days)} IS days at {args.fire_on} closes...')

    multi = MultiDayV2Ticker(days=sample_days)
    vel_col = price_velocity_w(args.tf)
    sn_col = swing_noise_w(args.tf)

    fire_check = {
        '1m': lambda s: s.is_1m_close,
        '5m': lambda s: s.is_5m_close,
        '15m': lambda s: s.is_15m_close,
        '1h': lambda s: s.is_1h_close,
    }.get(args.fire_on, lambda s: s.is_1h_close)

    vels, sns = [], []
    for state in tqdm(multi, total=10000, desc='collect'):
        if not fire_check(state):
            continue
        v = state.get(vel_col, np.nan)
        s = state.get(sn_col, np.nan)
        if np.isfinite(v):
            vels.append(float(v))
        if np.isfinite(s):
            sns.append(float(s))

    if not vels or not sns:
        print('!!! No samples; aborting')
        sys.exit(1)

    vels = np.asarray(vels)
    sns = np.asarray(sns)
    abs_vel = np.abs(vels)

    vel_thr = float(np.quantile(abs_vel, args.vel_quantile))
    sn_thr = float(np.quantile(sns, args.sn_quantile))

    out = {
        '_meta': {
            'tf': args.tf, 'fire_on': args.fire_on,
            'n_samples_vel': int(len(vels)), 'n_samples_sn': int(len(sns)),
            'vel_quantile': args.vel_quantile, 'sn_quantile': args.sn_quantile,
            'vel_distribution': {
                'p10': float(np.quantile(abs_vel, 0.10)),
                'p33': float(np.quantile(abs_vel, 0.33)),
                'p50': float(np.quantile(abs_vel, 0.50)),
                'p67': float(np.quantile(abs_vel, 0.67)),
                'p90': float(np.quantile(abs_vel, 0.90)),
                'p95': float(np.quantile(abs_vel, 0.95)),
                'mean': float(abs_vel.mean()),
            },
            'sn_distribution': {
                'p10': float(np.quantile(sns, 0.10)),
                'p25': float(np.quantile(sns, 0.25)),
                'p50': float(np.quantile(sns, 0.50)),
                'p75': float(np.quantile(sns, 0.75)),
                'p90': float(np.quantile(sns, 0.90)),
                'mean': float(sns.mean()),
            },
        },
        'vel_thr': vel_thr,
        'sn_thr': sn_thr,
        'tf': args.tf,
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\n{"=" * 70}')
    print(f'VELOCITY-REGIME THRESHOLDS  (TF={args.tf})')
    print(f'{"=" * 70}')
    print(f'  vel_thr (|vel| ≥ this → directional): {vel_thr:>10.3f}')
    print(f'  sn_thr  (sn  ≥ this → CHOPPY)       : {sn_thr:>10.3f}')
    print(f'\n  |vel| distribution: '
              f'p10={out["_meta"]["vel_distribution"]["p10"]:.2f}  '
              f'p50={out["_meta"]["vel_distribution"]["p50"]:.2f}  '
              f'p90={out["_meta"]["vel_distribution"]["p90"]:.2f}')
    print(f'  sn   distribution: '
              f'p10={out["_meta"]["sn_distribution"]["p10"]:.1f}  '
              f'p50={out["_meta"]["sn_distribution"]["p50"]:.1f}  '
              f'p90={out["_meta"]["sn_distribution"]["p90"]:.1f}')

    # Distribution of resulting regimes
    direction_counts = {'UP': 0, 'DOWN': 0, 'FLAT': 0}
    for v in vels:
        if v >= vel_thr: direction_counts['UP'] += 1
        elif v <= -vel_thr: direction_counts['DOWN'] += 1
        else: direction_counts['FLAT'] += 1
    smooth = sum(1 for s in sns if s < sn_thr)
    choppy = len(sns) - smooth
    total = len(vels)
    print(f'\n  EXPECTED REGIME DISTRIBUTION:')
    for k, v in direction_counts.items():
        print(f'    {k:<8} {v:>6} ({v/total*100:>5.1f}%)')
    print(f'    SMOOTH   {smooth:>6} ({smooth/len(sns)*100:>5.1f}%)')
    print(f'    CHOPPY   {choppy:>6} ({choppy/len(sns)*100:>5.1f}%)')

    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
