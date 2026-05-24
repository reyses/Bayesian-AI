"""3-body state analysis — answers two questions empirically:

Q1: How often is price in the "Crux state" — near M_close but at extremes
    of M_high/M_low?

Q2: Where does M_close ± 3σ_close land relative to M_high and M_low?
    Does the close-mean's ±3σ envelope encompass the high/low regression
    centers, or are they outside it?

Walks IS days at 5s cadence, computes 3-body state per bar, aggregates
and prints distributions + saves a visualization.

Usage:
    python tools/three_body_state_analysis.py
    python tools/three_body_state_analysis.py --tf 15m --n-days 30
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TF_CONFIG = {
    '1m':  ('DATA/ATLAS/1m',  15),
    '5m':  ('DATA/ATLAS/5m',   9),
    '15m': ('DATA/ATLAS/15m', 12),
    '1h':  ('DATA/ATLAS/1h',  12),
}


def _load_ohlcv(tf: str, day: str) -> pd.DataFrame:
    base, _ = TF_CONFIG[tf]
    path = os.path.join(base, f'{day}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _per_day_states(day: str, tf: str, atlas_5s: str) -> pd.DataFrame:
    base, N = TF_CONFIG[tf]
    period_s = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}[tf]

    ohlcv_5s = pd.read_parquet(os.path.join(atlas_5s, f'{day}.parquet'))
    if pd.api.types.is_datetime64_any_dtype(ohlcv_5s['timestamp']):
        ohlcv_5s = ohlcv_5s.copy()
        ohlcv_5s['timestamp'] = (ohlcv_5s['timestamp'].astype('int64') // 10**9)
    ohlcv_5s = ohlcv_5s.sort_values('timestamp').reset_index(drop=True)
    if ohlcv_5s.empty:
        return pd.DataFrame()

    tf_oh = _load_ohlcv(tf, day)
    if tf_oh.empty:
        return pd.DataFrame()

    tf_oh['close_mean']  = tf_oh['close'].rolling(N, min_periods=2).mean()
    tf_oh['close_sigma'] = tf_oh['close'].rolling(N, min_periods=2).std()
    tf_oh['high_mean']   = tf_oh['high'].rolling(N, min_periods=2).mean()
    tf_oh['high_sigma']  = tf_oh['high'].rolling(N, min_periods=2).std()
    tf_oh['low_mean']    = tf_oh['low'].rolling(N, min_periods=2).mean()
    tf_oh['low_sigma']   = tf_oh['low'].rolling(N, min_periods=2).std()

    # Forward-fill to 5s timeline
    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)
    tf_ts = tf_oh['timestamp'].values.astype(np.int64)
    target = oh_ts - period_s
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)

    p = ohlcv_5s['close'].values
    Mc = tf_oh['close_mean'].values[idx]
    Sc = tf_oh['close_sigma'].values[idx]
    Mh = tf_oh['high_mean'].values[idx]
    Sh = tf_oh['high_sigma'].values[idx]
    Ml = tf_oh['low_mean'].values[idx]
    Sl = tf_oh['low_sigma'].values[idx]

    return pd.DataFrame({
        'day': day, 'ts': oh_ts, 'p': p,
        'Mc': Mc, 'Sc': Sc, 'Mh': Mh, 'Sh': Sh, 'Ml': Ml, 'Sl': Sl,
        'd_close': (p - Mc) / Sc,
        'd_high':  (p - Mh) / Sh,
        'd_low':   (p - Ml) / Sl,
        # Where does M_close ±3σ_close land relative to M_high and M_low?
        'plus3_vs_Mh':  (Mc + 3 * Sc) - Mh,    # >0 means +3σ exceeds M_high
        'minus3_vs_Ml': Ml - (Mc - 3 * Sc),    # >0 means -3σ exceeds M_low
        # As σ-units of M_high, M_low respectively
        'plus3_z_high': ((Mc + 3 * Sc) - Mh) / Sh,
        'minus3_z_low': ((Mc - 3 * Sc) - Ml) / Sl,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='15m', choices=list(TF_CONFIG.keys()))
    ap.add_argument('--n-days', type=int, default=80,
                          help='Sample N IS days evenly')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='reports/findings/three_body')
    ap.add_argument('--crux-near-close', type=float, default=0.5,
                          help='|d_close| < this → "near M_close"')
    ap.add_argument('--crux-extreme-hl', type=float, default=2.0,
                          help='|d_high| > this OR |d_low| > this → "extreme of HL"')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.features_root, 'L0', '*.parquet')))
    is_days = [os.path.basename(f).replace('.parquet', '') for f in files
                    if os.path.basename(f).startswith('2025_')]
    if len(is_days) > args.n_days:
        idx = np.linspace(0, len(is_days) - 1, args.n_days, dtype=int)
        sample_days = [is_days[i] for i in idx]
    else:
        sample_days = is_days
    print(f'Sampling {len(sample_days)} IS days, TF={args.tf}')

    frames = []
    for day in tqdm(sample_days, desc='days'):
        df = _per_day_states(day, args.tf, args.atlas_5s)
        if not df.empty:
            frames.append(df)
    if not frames:
        print('No data'); sys.exit(1)
    full = pd.concat(frames, ignore_index=True)
    full = full.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        'd_close', 'd_high', 'd_low'])
    print(f'\nTotal bars analyzed: {len(full):,}')

    # Q1 — Crux state
    crux_mask = (
        (full['d_close'].abs() < args.crux_near_close)
        & ((full['d_high'].abs() > args.crux_extreme_hl)
              | (full['d_low'].abs() > args.crux_extreme_hl))
    )
    n_crux = int(crux_mask.sum())
    pct_crux = n_crux / len(full) * 100
    print(f'\n{"=" * 80}')
    print(f'Q1 — CRUX STATE FREQUENCY')
    print(f'  near M_close (|d_close| < {args.crux_near_close})')
    print(f'  AND extreme of HL (|d_high| > {args.crux_extreme_hl} OR |d_low| > {args.crux_extreme_hl})')
    print(f'  Bars in crux state: {n_crux:>9,}  ({pct_crux:.3f}% of all bars)')

    # Sub-cases
    crux_high = (full['d_close'].abs() < args.crux_near_close) & (full['d_high'].abs() > args.crux_extreme_hl)
    crux_low = (full['d_close'].abs() < args.crux_near_close) & (full['d_low'].abs() > args.crux_extreme_hl)
    print(f'    near close + extreme of M_high: {int(crux_high.sum()):>9,}  ({crux_high.mean()*100:.3f}%)')
    print(f'    near close + extreme of M_low : {int(crux_low.sum()):>9,}  ({crux_low.mean()*100:.3f}%)')

    # Reference distributions
    print(f'\n  REFERENCE: |d_close| distribution (all bars):')
    print(f'    median={float(full["d_close"].abs().median()):.2f}  '
              f'q90={float(full["d_close"].abs().quantile(0.90)):.2f}')
    print(f'  REFERENCE: |d_high| distribution (all bars):')
    print(f'    median={float(full["d_high"].abs().median()):.2f}  '
              f'q90={float(full["d_high"].abs().quantile(0.90)):.2f}')

    # Q2 — Where does M_close ±3σ land vs M_high/M_low?
    print(f'\n{"=" * 80}')
    print(f'Q2 — DOES M_close ±3σ_close ENCOMPASS M_high / M_low?')
    print(f'\n  M_close + 3σ_close vs M_high (price points):')
    print(f'    median: {float(full["plus3_vs_Mh"].median()):+.2f} pts  '
              f'(>0 means +3σ exceeds M_high)')
    print(f'    Q10:    {float(full["plus3_vs_Mh"].quantile(0.10)):+.2f}')
    print(f'    Q90:    {float(full["plus3_vs_Mh"].quantile(0.90)):+.2f}')
    print(f'    %above (+3σ > M_high):   '
              f'{float((full["plus3_vs_Mh"] > 0).mean()*100):.1f}%')
    print(f'  Same in σ_high units (M_close+3σ vs M_high in z_high):')
    print(f'    median: {float(full["plus3_z_high"].median()):+.2f} σ_high')
    print(f'    Q10:    {float(full["plus3_z_high"].quantile(0.10)):+.2f}')
    print(f'    Q90:    {float(full["plus3_z_high"].quantile(0.90)):+.2f}')

    print(f'\n  M_close − 3σ_close vs M_low (price points):')
    print(f'    median: {float(full["minus3_vs_Ml"].median()):+.2f} pts  '
              f'(>0 means -3σ exceeds M_low going DOWN)')
    print(f'    Q10:    {float(full["minus3_vs_Ml"].quantile(0.10)):+.2f}')
    print(f'    Q90:    {float(full["minus3_vs_Ml"].quantile(0.90)):+.2f}')
    print(f'    %above (-3σ exceeds M_low):  '
              f'{float((full["minus3_vs_Ml"] > 0).mean()*100):.1f}%')
    print(f'  Same in σ_low units:')
    print(f'    median: {float(full["minus3_z_low"].median()):+.2f} σ_low')
    print(f'    Q10:    {float(full["minus3_z_low"].quantile(0.10)):+.2f}')
    print(f'    Q90:    {float(full["minus3_z_low"].quantile(0.90)):+.2f}')

    # ── Visualization ──────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Distribution of d_close, d_high, d_low
    ax = axes[0, 0]
    ax.hist(full['d_close'].clip(-5, 5), bins=80, alpha=0.5, label='d_close', color='royalblue')
    ax.hist(full['d_high'].clip(-5, 5), bins=80, alpha=0.5, label='d_high', color='tab:green')
    ax.hist(full['d_low'].clip(-5, 5), bins=80, alpha=0.5, label='d_low', color='tab:red')
    ax.axvline(0, color='black', lw=0.6)
    ax.set_title('σ-distance distributions per anchor')
    ax.set_xlabel('σ from anchor')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D heatmap: d_close vs max(|d_high|, |d_low|)
    ax = axes[0, 1]
    hl_extreme = np.maximum(full['d_high'].abs(), full['d_low'].abs()).clip(0, 5)
    h, xedges, yedges = np.histogram2d(
        full['d_close'].clip(-3, 3), hl_extreme,
        bins=[60, 50])
    im = ax.imshow(h.T, origin='lower', aspect='auto', cmap='hot',
                          extent=[-3, 3, 0, 5])
    ax.axvline(args.crux_near_close, color='cyan', lw=1.0, linestyle='--',
                     label=f'|d_close| = {args.crux_near_close}')
    ax.axvline(-args.crux_near_close, color='cyan', lw=1.0, linestyle='--')
    ax.axhline(args.crux_extreme_hl, color='magenta', lw=1.0, linestyle='--',
                     label=f'max(|d_HL|) = {args.crux_extreme_hl}')
    ax.set_title(f'CRUX STATE map  ({pct_crux:.2f}% of bars)\n'
                       f'(top-left or top-right of dashed quadrants)')
    ax.set_xlabel('d_close (σ from M_close)')
    ax.set_ylabel('max(|d_high|, |d_low|) σ')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax)

    # M_close + 3σ vs M_high distribution (in σ_high units)
    ax = axes[1, 0]
    ax.hist(full['plus3_z_high'].clip(-5, 5), bins=80, color='tab:green', alpha=0.7)
    ax.axvline(0, color='black', lw=0.8,
                     label='zero = M_close+3σ exactly equals M_high')
    median_val = float(full['plus3_z_high'].median())
    ax.axvline(median_val, color='red', lw=1.0, linestyle='--',
                     label=f'median = {median_val:+.2f}σ_high')
    ax.set_title('Where does M_close + 3σ_close land vs M_high?')
    ax.set_xlabel('(M_close + 3σ_close − M_high) / σ_high')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # M_close - 3σ vs M_low (in σ_low units)
    ax = axes[1, 1]
    ax.hist(full['minus3_z_low'].clip(-5, 5), bins=80, color='tab:red', alpha=0.7)
    ax.axvline(0, color='black', lw=0.8,
                     label='zero = M_close−3σ exactly equals M_low')
    median_val = float(full['minus3_z_low'].median())
    ax.axvline(median_val, color='blue', lw=1.0, linestyle='--',
                     label=f'median = {median_val:+.2f}σ_low')
    ax.set_title('Where does M_close − 3σ_close land vs M_low?')
    ax.set_xlabel('(M_close − 3σ_close − M_low) / σ_low')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out, f'three_body_state_tf{args.tf}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nChart -> {out_png}')

    # Save the per-bar parquet for downstream
    pq = os.path.join(args.out, f'three_body_state_tf{args.tf}.parquet')
    full[['day', 'ts', 'p', 'd_close', 'd_high', 'd_low',
              'plus3_z_high', 'minus3_z_low']].to_parquet(pq, index=False)
    print(f'Per-bar data -> {pq}')


if __name__ == '__main__':
    main()
