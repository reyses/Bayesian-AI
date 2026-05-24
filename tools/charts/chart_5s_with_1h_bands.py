"""
Chart 5s price with 1h regression-mean bands overlay.

Shows how 5s price interacts with the MUCH slower 1h regression structure:
  - 5s close (fast, wiggly)
  - 1h RM (rolling 60-bar OLS, updates every hour → step line on 5s scale)
  - 1h RM ± 1σ / ±2σ / ±3σ bands (same step-constant behavior)

Bands are computed per 1h bar using the 60 preceding 1h bars (no lookahead)
and held constant through the 720 × 5s bars inside that hour.

Usage:
    python tools/chart_5s_with_1h_bands.py --day 2025_06_09
    python tools/chart_5s_with_1h_bands.py --day 2025_06_09 --reg-window 60

Output:
    charts/5s_with_1h_bands_<day>.png
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

ATLAS_1H_DIR = 'DATA/ATLAS/1h'
ATLAS_5S_DIR = 'DATA/ATLAS/5s'
CHARTS_DIR = 'charts'
REG_WINDOW = 60     # 60-bar OLS on 1h


def rolling_rm_and_se(closes, window):
    """Per-bar rolling OLS. Returns (fitted_last, se) arrays, NaN for warm-up.

    fitted_last = value of OLS line AT the current bar (price-level estimate).
    se          = sample std-error of residuals inside the window (price-level).
    """
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


def load_1h_history(target_day, reg_window=REG_WINDOW):
    """Load target day 1h file + enough preceding days to have reg_window
    complete bars before the target day starts."""
    # Need reg_window+ 1h bars of history. 24 hours/day, so ~3 days before
    # is enough even on thinnish sessions.
    target = pd.Timestamp(target_day.replace('_', '-'))
    # Load a generous range: target-7d to target+0d
    path_target = os.path.join(ATLAS_1H_DIR, f'{target_day}.parquet')
    if not os.path.exists(path_target):
        raise FileNotFoundError(f'1h file missing: {path_target}')

    dfs = []
    for back in range(7, -1, -1):
        d = (target - pd.Timedelta(days=back)).strftime('%Y_%m_%d')
        p = os.path.join(ATLAS_1H_DIR, f'{d}.parquet')
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise RuntimeError('No 1h data found')
    df = pd.concat(dfs, ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--reg-window', type=int, default=REG_WINDOW)
    args = ap.parse_args()

    # Load 5s for the day
    p5s = os.path.join(ATLAS_5S_DIR, f'{args.day}.parquet')
    df5s = pd.read_parquet(p5s).sort_values('timestamp').reset_index(drop=True)
    ts_5s = df5s['timestamp'].values.astype(np.int64)
    close_5s = df5s['close'].values.astype(np.float64)
    dts_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]

    # Load 1h + preceding days and compute per-1h-bar RM + SE
    df_1h = load_1h_history(args.day, args.reg_window)
    ts_1h = df_1h['timestamp'].values.astype(np.int64)
    close_1h = df_1h['close'].values.astype(np.float64)
    rm_1h, se_1h = rolling_rm_and_se(close_1h, args.reg_window)

    # Step-fill onto 5s: for each 5s bar, find the 1h bar whose ts <= 5s_ts
    # (using searchsorted). If 1h bar's timestamp > 5s_ts, fall back.
    idx = np.searchsorted(ts_1h, ts_5s, side='right') - 1
    idx = np.clip(idx, 0, len(ts_1h) - 1)
    rm_on_5s = rm_1h[idx]
    se_on_5s = se_1h[idx]

    # Day start/end for context
    day_start = datetime.strptime(args.day, '%Y_%m_%d').replace(tzinfo=timezone.utc)

    # ── Chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 10))

    # Bands
    ax.fill_between(dts_5s, rm_on_5s - 3 * se_on_5s, rm_on_5s + 3 * se_on_5s,
                    color='tab:red', alpha=0.07, label='1h ±3σ')
    ax.fill_between(dts_5s, rm_on_5s - 2 * se_on_5s, rm_on_5s + 2 * se_on_5s,
                    color='tab:orange', alpha=0.10, label='1h ±2σ')
    ax.fill_between(dts_5s, rm_on_5s - 1 * se_on_5s, rm_on_5s + 1 * se_on_5s,
                    color='tab:blue', alpha=0.10, label='1h ±1σ')
    # Band edges (step-style lines)
    ax.plot(dts_5s, rm_on_5s + 3 * se_on_5s, color='tab:red', linewidth=0.8, alpha=0.7)
    ax.plot(dts_5s, rm_on_5s - 3 * se_on_5s, color='tab:red', linewidth=0.8, alpha=0.7)
    ax.plot(dts_5s, rm_on_5s + 2 * se_on_5s, color='tab:orange', linewidth=0.9, alpha=0.8)
    ax.plot(dts_5s, rm_on_5s - 2 * se_on_5s, color='tab:orange', linewidth=0.9, alpha=0.8)
    ax.plot(dts_5s, rm_on_5s + 1 * se_on_5s, color='tab:blue', linewidth=1.0, alpha=0.8, linestyle=':')
    ax.plot(dts_5s, rm_on_5s - 1 * se_on_5s, color='tab:blue', linewidth=1.0, alpha=0.8, linestyle=':')

    # 1h regression mean
    ax.plot(dts_5s, rm_on_5s, color='black', linewidth=2.0, alpha=0.9,
            label=f'1h RM ({args.reg_window}-bar OLS)')

    # 5s price
    ax.plot(dts_5s, close_5s, color='tab:purple', linewidth=0.9, alpha=0.9,
            label='5s close')

    # Hour gridlines (1h bar boundaries)
    hour_boundaries = [datetime.fromtimestamp(int(t), tz=timezone.utc)
                       for t in ts_1h
                       if ts_5s[0] <= t <= ts_5s[-1]]
    for hb in hour_boundaries:
        ax.axvline(hb, color='gray', linewidth=0.3, alpha=0.3)

    # Annotations
    ax.set_title(f'{args.day} — 5s price with 1h regression bands '
                 f'({args.reg_window}-bar OLS = ~{args.reg_window}h history)',
                 fontsize=14)
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_ylabel('MNQ price', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    os.makedirs(CHARTS_DIR, exist_ok=True)
    out_path = os.path.join(CHARTS_DIR, f'5s_with_1h_bands_{args.day}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

    print(f'Wrote: {out_path}')
    print(f'5s bars: {len(df5s)}  1h bars in day: '
          f'{len([t for t in ts_1h if ts_5s[0] <= t <= ts_5s[-1]])}')
    print(f'1h RM range across day: '
          f'{np.nanmin(rm_on_5s):.2f} to {np.nanmax(rm_on_5s):.2f}')
    print(f'1h SE range: {np.nanmin(se_on_5s):.2f} to {np.nanmax(se_on_5s):.2f}')


if __name__ == '__main__':
    main()
