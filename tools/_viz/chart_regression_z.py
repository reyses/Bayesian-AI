"""
Chart regression mean + z overlay on 1m price (zoomed-out view).

Shows the physics of mean-reversion visually:
  - Price line (1m close) — noisier at 1m, gives finer mean-rev dynamics
  - Regression mean (rolling OLS over N 1m bars)
  - Bands at ±1σ, ±2σ, ±3σ
  - Z-score subplot below (1m_z_se from FEATURES_5s)

Usage:
    python tools/chart_regression_z.py --day 2025_06_09
    python tools/chart_regression_z.py --day 2025_04_09 --window 60 --tf 1m
    python tools/chart_regression_z.py --tf 5m                      # legacy

Output: charts/chart_reg_z_<tf>_<day>.png
"""
import os
import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ATLAS_DIR = 'DATA/ATLAS'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'


def rolling_regression(prices, window):
    """For each bar i, fit linear regression on prices[i-window+1:i+1].
    Return (fitted_value_at_i, SE) per bar.

    SE = stderr of residuals in the window.
    """
    n = len(prices)
    fitted = np.full(n, np.nan)
    se = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = prices[i - window + 1:i + 1]
        x = np.arange(window, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            continue
        slope = (dx * (y - ym)).sum() / denom
        intercept = ym - slope * xm
        # Fitted value at last x (x = window - 1)
        fit_last = intercept + slope * (window - 1)
        # Residual std (SE of fit at x_last, simplified)
        fits = intercept + slope * x
        resid = y - fits
        sigma = np.sqrt((resid ** 2).sum() / max(window - 2, 1))
        fitted[i] = fit_last
        se[i] = sigma
    return fitted, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--tf', default='1m', choices=['1m', '5m'],
                    help='Price timeframe (default 1m)')
    ap.add_argument('--window', type=int, default=None,
                    help='Regression window in bars (default: 60 for 1m, 20 for 5m)')
    args = ap.parse_args()

    window = args.window if args.window is not None else (60 if args.tf == '1m' else 20)
    z_feature = f'{args.tf}_z_se'

    price_path = os.path.join(ATLAS_DIR, args.tf, f'{args.day}.parquet')
    feat_path = os.path.join(FEATURES_5S_DIR, f'{args.day}.parquet')
    if not os.path.exists(price_path):
        print(f'Missing: {price_path}')
        sys.exit(1)

    df = pd.read_parquet(price_path).sort_values('timestamp').reset_index(drop=True)
    print(f'Loaded {args.day}: {len(df)} {args.tf} bars')
    closes = df['close'].values.astype(np.float64)
    ts = df['timestamp'].values.astype(np.float64)
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]

    fitted, se = rolling_regression(closes, window)
    u1, u2, u3 = fitted + se, fitted + 2.0 * se, fitted + 3.0 * se
    l1, l2, l3 = fitted - se, fitted - 2.0 * se, fitted - 3.0 * se

    # Load 5s features for z overlay, downsample to chart timeframe
    z_series = np.full(len(df), np.nan)
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path).sort_values('timestamp').reset_index(drop=True)
        f_ts = fdf['timestamp'].values.astype(np.int64)
        if z_feature in fdf.columns:
            z_arr = fdf[z_feature].values.astype(np.float64)
            for i, t in enumerate(ts):
                idx = np.searchsorted(f_ts, int(t), side='right') - 1
                if idx >= 0:
                    z_series[i] = z_arr[idx]
        else:
            print(f'WARN: {z_feature} missing from features')
    else:
        print(f'WARN: features file missing for z overlay')

    # Build chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    gridspec_kw={'height_ratios': [2.5, 1]},
                                    sharex=True)

    # Top: price + regression + σ bands
    ax1.plot(dts, closes, color='black', linewidth=0.9, label=f'Price ({args.tf} close)')
    ax1.plot(dts, fitted, color='tab:blue', linewidth=2.0,
             label=f'Regression mean ({window}-bar OLS)')
    # σ bands — color intensity increases with sigma
    ax1.plot(dts, u1, color='tab:orange', linewidth=0.8, linestyle=':',
             alpha=0.65, label='±1σ')
    ax1.plot(dts, l1, color='tab:orange', linewidth=0.8, linestyle=':', alpha=0.65)
    ax1.plot(dts, u2, color='tab:red', linewidth=1.0, linestyle='--',
             alpha=0.75, label='±2σ')
    ax1.plot(dts, l2, color='tab:red', linewidth=1.0, linestyle='--', alpha=0.75)
    ax1.plot(dts, u3, color='darkred', linewidth=1.2, linestyle='-',
             alpha=0.8, label='±3σ')
    ax1.plot(dts, l3, color='darkred', linewidth=1.2, linestyle='-', alpha=0.8)
    ax1.fill_between(dts, l1, u1, color='tab:orange', alpha=0.04)
    ax1.fill_between(dts, l2, u2, color='tab:red', alpha=0.03)
    ax1.set_ylabel('MNQ price', fontsize=11)
    ax1.set_title(f'{args.day} — {args.tf} price, regression mean, ±1σ ±2σ ±3σ bands, z overlay',
                  fontsize=13)
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.25)

    # Mark extreme-z bars
    extreme_mask = np.abs(z_series) > 2.0
    if extreme_mask.any():
        ax1.scatter([dts[i] for i in np.where(extreme_mask)[0]],
                    closes[extreme_mask],
                    c=np.where(z_series[extreme_mask] > 0, 'tab:red', 'tab:green'),
                    s=25, zorder=5, alpha=0.7, edgecolors='black', linewidths=0.3)

    # Bottom: z score with ±1, ±2, ±3 reference lines
    ax2.axhline(0, color='gray', linewidth=0.8)
    for sigma, color in [(1, 'tab:orange'), (2, 'tab:red'), (3, 'darkred')]:
        ax2.axhline(sigma, color=color, linestyle=':' if sigma == 1 else '--',
                     linewidth=0.8 + sigma * 0.15, alpha=0.7)
        ax2.axhline(-sigma, color=color, linestyle=':' if sigma == 1 else '--',
                     linewidth=0.8 + sigma * 0.15, alpha=0.7)
    ax2.plot(dts, z_series, color='tab:purple', linewidth=1.0, label=z_feature)
    ax2.fill_between(dts, 0, z_series, where=(z_series > 0), color='tab:red', alpha=0.15)
    ax2.fill_between(dts, 0, z_series, where=(z_series < 0), color='tab:green', alpha=0.15)
    ax2.set_ylabel(f'{args.tf} z-score', fontsize=11)
    ax2.set_xlabel('Time (UTC)', fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(-4.5, 4.5)

    # Format x-axis
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
    plt.tight_layout()

    out_path = f'charts/chart_reg_z_{args.tf}_{args.day}.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {out_path}')

    # Stats summary
    valid_z = z_series[~np.isnan(z_series)]
    print(f'\nDay summary:')
    print(f'  Price range: {closes.min():.2f} - {closes.max():.2f} '
          f'({closes.max()-closes.min():.2f} pts)')
    valid_fitted = fitted[~np.isnan(fitted)]
    if valid_fitted.size:
        print(f'  Regression mean drift: {valid_fitted[0]:.1f} -> '
              f'{valid_fitted[-1]:.1f} '
              f'({valid_fitted[-1]-valid_fitted[0]:+.1f} pts)')
    if valid_z.size:
        print(f'  {args.tf} z_se range: {valid_z.min():+.2f} to {valid_z.max():+.2f}')
        print(f'  Bars |z|>1: {int((np.abs(valid_z) > 1).sum())} '
              f'({(np.abs(valid_z) > 1).mean()*100:.1f}%)')
        print(f'  Bars |z|>2: {int((np.abs(valid_z) > 2).sum())} '
              f'({(np.abs(valid_z) > 2).mean()*100:.1f}%)')
        print(f'  Bars |z|>3: {int((np.abs(valid_z) > 3).sum())} '
              f'({(np.abs(valid_z) > 3).mean()*100:.1f}%)')


if __name__ == '__main__':
    main()
