"""
Chart bands at any TF on a single day, with multiple window sizes.

Generalization of chart_5s_1h_bands_multi_window.py — works for any band TF.

Usage:
    python tools/chart_tf_bands_multi_window.py --day 2025_06_09 --band-tf 1m --windows 15 30 60 180
    python tools/chart_tf_bands_multi_window.py --day 2025_06_09 --band-tf 5m --windows 9 18 36 72
    python tools/chart_tf_bands_multi_window.py --day 2025_06_09 --band-tf 15m --windows 12 24 48 96
    python tools/chart_tf_bands_multi_window.py --day 2025_06_09 --band-tf 1h --windows 12 24 60 168

Underlying chart granularity: always 5s (consistent visual scale across TFs).

Output:
    charts/<band_tf>_bands_multi_window_<day>.png
"""
import os
import sys
import glob
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_ROOT = 'DATA/ATLAS'
CHARTS_DIR = 'charts'


def rolling_rm_se(closes, window):
    n = len(closes)
    rm = np.full(n, np.nan, dtype=np.float64)
    se = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom < 1e-12:
        return rm, se
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        ym = y.mean()
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = ym - slope * xm
        fit_last = intercept + slope * (window - 1)
        fits = intercept + slope * x
        resid = y - fits
        sigma = float(np.sqrt((resid ** 2).sum() / max(window - 2, 1)))
        rm[i] = fit_last
        se[i] = sigma
    return rm, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--band-tf', default='1h',
                    help='TF for the bands: 1m, 5m, 15m, 1h, etc.')
    ap.add_argument('--windows', type=int, nargs='+', default=[12, 24, 60, 168])
    args = ap.parse_args()

    # 5s underlying for visual
    p5s = os.path.join(ATLAS_ROOT, '5s', f'{args.day}.parquet')
    df5s = pd.read_parquet(p5s).sort_values('timestamp').reset_index(drop=True)
    ts_5s = df5s['timestamp'].values.astype(np.int64)
    close_5s = df5s['close'].values.astype(np.float64)
    dts_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]

    # Band-TF data — load ALL available days
    band_dir = os.path.join(ATLAS_ROOT, args.band_tf)
    all_paths = sorted(glob.glob(os.path.join(band_dir, '*.parquet')))
    if not all_paths:
        raise RuntimeError(f'No data in {band_dir}')
    df_band = (pd.concat([pd.read_parquet(p) for p in all_paths],
                         ignore_index=True)
               .drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True))
    ts_band = df_band['timestamp'].values.astype(np.int64)
    close_band = df_band['close'].values.astype(np.float64)

    # TF duration (for label)
    tf_sec_map = {
        '1s': 1, '5s': 5, '15s': 15, '30s': 30,
        '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '4h': 14400, '1d': 86400,
    }
    tf_sec = tf_sec_map.get(args.band_tf, 3600)

    # Precompute RM+SE per window, step-fill onto 5s
    per_window = {}
    for W in args.windows:
        rm, se = rolling_rm_se(close_band, W)
        idx = np.searchsorted(ts_band, ts_5s, side='right') - 1
        idx = np.clip(idx, 0, len(ts_band) - 1)
        per_window[W] = {
            'rm5': rm[idx],
            'se5': se[idx],
        }

    # ── Chart ──────────────────────────────────────────────────────
    n = len(args.windows)
    fig, axes = plt.subplots(n, 1, figsize=(22, 4.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, W in zip(axes, args.windows):
        rm5 = per_window[W]['rm5']
        se5 = per_window[W]['se5']
        valid = ~np.isnan(se5)
        dts_v = [d for d, ok in zip(dts_5s, valid) if ok]
        price_v = close_5s[valid]
        rm_v = rm5[valid]
        se_v = se5[valid]
        if len(dts_v) == 0:
            ax.set_title(f'{args.band_tf} × {W} bars — no valid history')
            continue

        ax.fill_between(dts_v, rm_v - 3*se_v, rm_v + 3*se_v, color='tab:red', alpha=0.07)
        ax.fill_between(dts_v, rm_v - 2*se_v, rm_v + 2*se_v, color='tab:orange', alpha=0.10)
        ax.fill_between(dts_v, rm_v - 1*se_v, rm_v + 1*se_v, color='tab:blue', alpha=0.10)
        ax.plot(dts_v, rm_v + 3*se_v, color='tab:red', linewidth=0.8, alpha=0.7)
        ax.plot(dts_v, rm_v - 3*se_v, color='tab:red', linewidth=0.8, alpha=0.7)
        ax.plot(dts_v, rm_v + 2*se_v, color='tab:orange', linewidth=0.9, alpha=0.8)
        ax.plot(dts_v, rm_v - 2*se_v, color='tab:orange', linewidth=0.9, alpha=0.8)
        ax.plot(dts_v, rm_v + 1*se_v, color='tab:blue', linewidth=1.0, linestyle=':', alpha=0.8)
        ax.plot(dts_v, rm_v - 1*se_v, color='tab:blue', linewidth=1.0, linestyle=':', alpha=0.8)
        ax.plot(dts_v, rm_v, color='black', linewidth=2.0, alpha=0.9)
        ax.plot(dts_5s, close_5s, color='tab:purple', linewidth=0.9, alpha=0.9)

        z = (price_v - rm_v) / np.where(se_v > 0, se_v, np.nan)
        pct1 = (np.abs(z) <= 1).mean() * 100
        pct2 = (np.abs(z) <= 2).mean() * 100
        max_abs_z = np.max(np.abs(z))
        med_se_dollars = float(np.median(se_v)) * 2

        window_hours = W * tf_sec / 3600.0
        ax.set_title(
            f'{args.band_tf} x {W} bars ({window_hours:.1f}h lookback) - '
            f'median +-1sig = \\${med_se_dollars:.0f}  '
            f'| today: {pct1:.0f}% within +-1sig, {pct2:.0f}% within +-2sig, '
            f'max |z| = {max_abs_z:.2f}',
            fontsize=11
        )
        ax.set_ylabel('MNQ', fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    axes[-1].set_xlabel('Time (UTC)', fontsize=12)
    fig.suptitle(f'{args.day} - 5s price with {args.band_tf} bands at multiple windows',
                 fontsize=15, y=0.995)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    out_path = os.path.join(CHARTS_DIR,
                             f'{args.band_tf}_bands_multi_window_{args.day}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {out_path}')
    print(f'Band TF: {args.band_tf} | Windows: {args.windows}')


if __name__ == '__main__':
    main()
