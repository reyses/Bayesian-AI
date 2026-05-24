"""
DMI + SE Bands + Volume overlay on historical price data.
Visualize where all three signals align.

Usage:
    python tools/dmi_se_overlay.py --date 2026-02-05 --data DATA/ATLAS_OOS
    python tools/dmi_se_overlay.py --date 2026-02-05 --tf 1m --data DATA/ATLAS_OOS
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from core_v2.statistical_field_engine import StatisticalFieldEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--tf', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--data', default='DATA/ATLAS_OOS', help='ATLAS root')
    parser.add_argument('--se-window', type=int, default=60, help='SE band window (default: 60)')
    parser.add_argument('--save', action='store_true', help='Save to file instead of show')
    args = parser.parse_args()

    # Map tf to folder
    tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '15s': '15s'}
    tf_folder = tf_map.get(args.tf, args.tf)

    # Load data
    import glob
    month = args.date[:7].replace('-', '_')
    files = sorted(glob.glob(f'{args.data}/{tf_folder}/{month}*.parquet'))
    if not files:
        print(f"No data found: {args.data}/{tf_folder}/{month}*.parquet")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df[df['dt'].dt.strftime('%Y-%m-%d') == args.date].reset_index(drop=True)
    print(f'{args.date} ({args.tf}): {len(df)} bars')

    if len(df) < args.se_window + 10:
        print(f"Not enough bars for SE window {args.se_window}")
        return

    # Compute SFE states for DMI
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)

    # Extract arrays
    times = df['dt'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))

    dmi_p = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_plus', 0) for s in states])
    dmi_m = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_minus', 0) for s in states])
    dmi_diff = dmi_p - dmi_m

    # Smoothed DMI (3-bar MA)
    dmi_smooth = pd.Series(dmi_diff).rolling(3).mean().values

    # SE bands
    W = args.se_window
    mean_line = np.full(len(closes), np.nan)
    se1_up = np.full(len(closes), np.nan)
    se1_lo = np.full(len(closes), np.nan)
    se2_up = np.full(len(closes), np.nan)
    se2_lo = np.full(len(closes), np.nan)
    se3_up = np.full(len(closes), np.nan)
    se3_lo = np.full(len(closes), np.nan)

    for i in range(W, len(closes)):
        chunk = closes[i - W:i]
        mu = chunk.mean()
        se = chunk.std() / np.sqrt(W)
        mean_line[i] = mu
        se1_up[i] = mu + 1 * se
        se1_lo[i] = mu - 1 * se
        se2_up[i] = mu + 2 * se
        se2_lo[i] = mu - 2 * se
        se3_up[i] = mu + 3 * se
        se3_lo[i] = mu - 3 * se

    # DMI cross points
    cross_long = []
    cross_short = []
    for i in range(1, len(dmi_smooth)):
        if np.isnan(dmi_smooth[i]) or np.isnan(dmi_smooth[i-1]):
            continue
        if dmi_smooth[i-1] < 0 and dmi_smooth[i] > 0:
            cross_long.append(i)
        elif dmi_smooth[i-1] > 0 and dmi_smooth[i] < 0:
            cross_short.append(i)

    # Volume average
    vol_avg = pd.Series(volumes).rolling(30).mean().values

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(48, 20), sharex=True,
                             gridspec_kw={'height_ratios': [4, 1.5, 1]})

    # Panel 1: Price + SE bands + DMI crosses
    ax1 = axes[0]
    # Candlesticks (simplified)
    for i in range(len(closes)):
        c = 'green' if closes[i] >= (closes[i-1] if i > 0 else closes[i]) else 'red'
        ax1.plot([times[i], times[i]], [lows[i], highs[i]], color=c, lw=0.8, alpha=0.6)
        ax1.plot(times[i], closes[i], '.', color=c, markersize=2)

    # SE bands
    ax1.fill_between(times, se3_lo, se3_up, alpha=0.05, color='purple', label='3σ SE')
    ax1.fill_between(times, se2_lo, se2_up, alpha=0.08, color='blue', label='2σ SE')
    ax1.fill_between(times, se1_lo, se1_up, alpha=0.12, color='cyan', label='1σ SE')
    ax1.plot(times, mean_line, color='white', lw=1, ls='--', alpha=0.5, label='Mean')

    # DMI cross markers
    for i in cross_long:
        ax1.axvline(times[i], color='lime', lw=1, alpha=0.4)
        ax1.scatter(times[i], closes[i], marker='^', c='lime', s=100, zorder=5, edgecolors='black')
    for i in cross_short:
        ax1.axvline(times[i], color='red', lw=1, alpha=0.4)
        ax1.scatter(times[i], closes[i], marker='v', c='red', s=100, zorder=5, edgecolors='black')

    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'{args.date} ({args.tf}) — Price + {W}-bar SE Bands + DMI Crosses + Volume\n'
                  f'Green ^=LONG cross | Red v=SHORT cross | Bands=1σ/2σ/3σ SE',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.15)

    # Panel 2: DMI
    ax2 = axes[1]
    ax2.plot(times, dmi_p, color='lime', lw=1, label='DMI+')
    ax2.plot(times, dmi_m, color='red', lw=1, label='DMI-')
    ax2.plot(times, dmi_smooth, color='yellow', lw=1.5, ls='--', label='Smooth diff')
    ax2.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax2.axhline(30, color='gray', lw=0.5, ls=':', alpha=0.3)
    for i in cross_long:
        ax2.axvline(times[i], color='lime', lw=0.5, alpha=0.3)
    for i in cross_short:
        ax2.axvline(times[i], color='red', lw=0.5, alpha=0.3)
    ax2.set_ylabel('DMI', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.15)

    # Panel 3: Volume
    ax3 = axes[2]
    colors = ['green' if closes[i] >= (closes[i-1] if i > 0 else closes[i]) else 'red'
              for i in range(len(closes))]
    ax3.bar(times, volumes, width=np.timedelta64(50, 's') if args.tf == '1m' else np.timedelta64(10, 's'),
            color=colors, alpha=0.5)
    ax3.plot(times, vol_avg, color='yellow', lw=1, label='30-bar avg')
    for i in cross_long:
        ax3.axvline(times[i], color='lime', lw=0.5, alpha=0.3)
    for i in cross_short:
        ax3.axvline(times[i], color='red', lw=0.5, alpha=0.3)
    ax3.set_ylabel('Volume', fontsize=10)
    ax3.set_xlabel('Time (UTC)', fontsize=12)
    ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.15)

    plt.tight_layout()

    if args.save:
        out = f'tools/plots/dmi_se_overlay_{args.date}_{args.tf}.png'
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f'Saved: {out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
