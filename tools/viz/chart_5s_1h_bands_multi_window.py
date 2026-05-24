"""
Chart 5s price on a single day with 1h bands at MULTIPLE window sizes.

One panel per window (stacked vertically, shared X). Visualizes how band
width + coverage change as the 1h OLS window changes.

Usage:
    python tools/chart_5s_1h_bands_multi_window.py --day 2025_06_09
    python tools/chart_5s_1h_bands_multi_window.py --day 2025_06_09 \
        --windows 12 24 60 168

Output:
    charts/5s_multi_window_bands_<day>.png
"""
import os
import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_1H = 'DATA/ATLAS/1h'
ATLAS_5S = 'DATA/ATLAS/5s'
CHARTS_DIR = 'charts'
DEFAULT_WINDOWS = [12, 24, 60, 168]


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
    ap.add_argument('--windows', type=int, nargs='+', default=DEFAULT_WINDOWS)
    args = ap.parse_args()

    # Load 5s for the day
    p5s = os.path.join(ATLAS_5S, f'{args.day}.parquet')
    df5s = pd.read_parquet(p5s).sort_values('timestamp').reset_index(drop=True)
    ts_5s = df5s['timestamp'].values.astype(np.int64)
    close_5s = df5s['close'].values.astype(np.float64)
    dts_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]

    # Load ALL available 1h data (small files, easier than calculating lookback)
    import glob
    all_1h_paths = sorted(glob.glob(os.path.join(ATLAS_1H, '*.parquet')))
    dfs = [pd.read_parquet(p) for p in all_1h_paths]
    if not dfs:
        raise RuntimeError('no 1h data')
    df_1h = (pd.concat(dfs, ignore_index=True)
             .drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True))
    ts_1h = df_1h['timestamp'].values.astype(np.int64)
    close_1h = df_1h['close'].values.astype(np.float64)

    # Precompute RM+SE per window, then step-fill onto 5s
    per_window = {}
    for W in args.windows:
        rm, se = rolling_rm_se(close_1h, W)
        idx = np.searchsorted(ts_1h, ts_5s, side='right') - 1
        idx = np.clip(idx, 0, len(ts_1h) - 1)
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
            ax.set_title(f'Window {W}h — no valid data (need {W}h of history)')
            continue

        # Bands
        ax.fill_between(dts_v, rm_v - 3*se_v, rm_v + 3*se_v, color='tab:red', alpha=0.07)
        ax.fill_between(dts_v, rm_v - 2*se_v, rm_v + 2*se_v, color='tab:orange', alpha=0.10)
        ax.fill_between(dts_v, rm_v - 1*se_v, rm_v + 1*se_v, color='tab:blue', alpha=0.10)
        # Edges
        ax.plot(dts_v, rm_v + 3*se_v, color='tab:red', linewidth=0.8, alpha=0.7)
        ax.plot(dts_v, rm_v - 3*se_v, color='tab:red', linewidth=0.8, alpha=0.7)
        ax.plot(dts_v, rm_v + 2*se_v, color='tab:orange', linewidth=0.9, alpha=0.8)
        ax.plot(dts_v, rm_v - 2*se_v, color='tab:orange', linewidth=0.9, alpha=0.8)
        ax.plot(dts_v, rm_v + 1*se_v, color='tab:blue', linewidth=1.0, linestyle=':', alpha=0.8)
        ax.plot(dts_v, rm_v - 1*se_v, color='tab:blue', linewidth=1.0, linestyle=':', alpha=0.8)
        # RM
        ax.plot(dts_v, rm_v, color='black', linewidth=2.0, alpha=0.9)
        # Price
        ax.plot(dts_s := dts_5s, close_5s, color='tab:purple', linewidth=0.9, alpha=0.9)

        # Interaction metrics for this day + window
        z = (price_v - rm_v) / np.where(se_v > 0, se_v, np.nan)
        pct1 = (np.abs(z) <= 1).mean() * 100
        pct2 = (np.abs(z) <= 2).mean() * 100
        max_abs_z = np.max(np.abs(z))
        med_se_dollars = float(np.median(se_v)) * 2

        ax.set_title(
            f'Window {W}h ({W/24:.1f}d) — '
            f'median ±1σ = \\${med_se_dollars:.0f}  '
            f'| today: {pct1:.0f}% within ±1σ, {pct2:.0f}% within ±2σ, '
            f'max |z| = {max_abs_z:.2f}',
            fontsize=11
        )
        ax.set_ylabel('MNQ', fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    axes[-1].set_xlabel('Time (UTC)', fontsize=12)
    fig.suptitle(f'{args.day} — 5s price with 1h bands at multiple windows',
                 fontsize=15, y=0.995)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    out_path = os.path.join(CHARTS_DIR, f'5s_multi_window_bands_{args.day}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {out_path}')
    print(f'Windows: {args.windows}')


if __name__ == '__main__':
    main()
