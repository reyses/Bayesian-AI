"""
Chart a single day of RM-slope trades.

Visual language mirrors `charts/cord_length_YYYY_MM_DD_RXX.png`:
  - Gray line: 1m price
  - Blue line: regression mean (60-bar OLS)
  - Green segment: winning trade (entry → exit)
  - Red segment: losing trade
  - Up-triangle / down-triangle markers: LONG / SHORT entry
  - X marker: exit
  - PnL label at exit for each trade

Usage:
    python tools/chart_rm_trades.py --day 2025_06_09
    python tools/chart_rm_trades.py --day 2025_06_09 --pkl rm_oos.pkl

Output:
    charts/rm_slope_trades_<day>.png
"""
import os
import sys
import argparse
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
TRADES_DIR = 'training_RM_physics/output/trades'
CHARTS_DIR = 'charts'

REG_WINDOW = 60


def rolling_rm(closes, window=REG_WINDOW):
    n = len(closes)
    rm = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom < 1e-12:
        return rm
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        ym = y.mean()
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = ym - slope * xm
        rm[i] = intercept + slope * (window - 1)
    return rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--pkl', default='rm_is.pkl',
                    help='Trades pickle filename under training_RM_physics/output/trades/')
    args = ap.parse_args()

    pkl_path = os.path.join(TRADES_DIR, args.pkl)
    if not os.path.exists(pkl_path):
        # Fallback: check rm_oos.pkl
        alt = os.path.join(TRADES_DIR, 'rm_oos.pkl')
        if args.day.startswith('2026'):
            pkl_path = alt
    with open(pkl_path, 'rb') as f:
        trades = pickle.load(f)
    # Filter to this day
    day_trades = [t for t in trades if t.get('day') == args.day]
    if not day_trades:
        # Try the other pickle
        for alt in ['rm_is.pkl', 'rm_oos.pkl']:
            p = os.path.join(TRADES_DIR, alt)
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    trades = pickle.load(f)
                day_trades = [t for t in trades if t.get('day') == args.day]
                if day_trades:
                    print(f'Loaded from {alt}')
                    break

    if not day_trades:
        print(f'No trades found for day {args.day}')
        return

    # Load price
    price_path = os.path.join(ATLAS_1M_DIR, f'{args.day}.parquet')
    df = pd.read_parquet(price_path).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].values.astype(np.float64)
    ts = df['timestamp'].values.astype(np.int64)
    dts = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]

    rm = rolling_rm(closes, REG_WINDOW)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 11),
        gridspec_kw={'height_ratios': [3.0, 1.0]}, sharex=True)

    # Panel 1: price + RM + trades
    ax1.plot(dts, closes, color='gray', linewidth=0.9, alpha=0.85,
              label='Price (1m)')
    ax1.plot(dts, rm, color='tab:blue', linewidth=2.0, label='Regression mean (60-bar)')

    for t in day_trades:
        entry_ts = int(t.get('timestamp', 0))
        held_min = int(t.get('held', 0))
        exit_ts = entry_ts + held_min * 60
        entry_dt = datetime.fromtimestamp(entry_ts, tz=timezone.utc)
        exit_dt = datetime.fromtimestamp(exit_ts, tz=timezone.utc)
        pnl = float(t['pnl'])
        color = 'tab:green' if pnl > 0 else 'tab:red'
        alpha = 0.85 if pnl > 0 else 0.55
        # Trade segment
        ax1.plot([entry_dt, exit_dt], [t['entry_price'], t['exit_price']],
                  color=color, linewidth=1.8, alpha=alpha, zorder=3)
        # Entry marker
        marker = '^' if t['dir'] == 'long' else 'v'
        ax1.scatter([entry_dt], [t['entry_price']],
                    marker=marker, s=90, c=color,
                    edgecolors='black', linewidths=0.7,
                    alpha=0.95, zorder=5)
        # Exit marker
        ax1.scatter([exit_dt], [t['exit_price']],
                    marker='x', s=60, c=color, linewidths=1.5,
                    alpha=0.95, zorder=5)
        # PnL text
        mid_dt = datetime.fromtimestamp(entry_ts + held_min * 30, tz=timezone.utc)
        ax1.annotate(f'${pnl:+.0f}',
                     xy=(exit_dt, t['exit_price']),
                     xytext=(6, 6 if pnl > 0 else -12),
                     textcoords='offset points',
                     fontsize=7, color=color,
                     alpha=0.9)

    total = sum(t['pnl'] for t in day_trades)
    wins = sum(1 for t in day_trades if t['pnl'] > 0)
    losses = sum(1 for t in day_trades if t['pnl'] < 0)
    dayWR = wins / max(len(day_trades), 1) * 100
    longs = sum(1 for t in day_trades if t['dir'] == 'long')
    shorts = sum(1 for t in day_trades if t['dir'] == 'short')

    ax1.set_title(
        f'{args.day} — RM-slope engine — '
        f'{len(day_trades)} trades ({longs}L / {shorts}S) — '
        f'count-WR {wins}/{len(day_trades)} ({dayWR:.0f}%) — '
        f'total ${total:+,.0f}',
        fontsize=13)
    ax1.set_ylabel('MNQ price', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.25)

    # Panel 2: slope (β) per 1m bar
    # Recompute slope for chart
    slopes = np.full(len(closes), np.nan)
    x = np.arange(REG_WINDOW, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    for i in range(REG_WINDOW - 1, len(closes)):
        y = closes[i - REG_WINDOW + 1: i + 1]
        ym = y.mean()
        slopes[i] = float((dx * (y - ym)).sum() / denom)

    ax2.plot(dts, slopes, color='tab:brown', linewidth=1.1)
    ax2.axhline(0, color='black', linewidth=0.7)
    # Entry threshold reference lines (match engine default 0.05)
    ax2.axhline(0.05, color='tab:green', linestyle='--',
                linewidth=0.7, alpha=0.7, label='±entry_thresh (0.05)')
    ax2.axhline(-0.05, color='tab:green', linestyle='--',
                linewidth=0.7, alpha=0.7)
    ax2.fill_between(dts, 0, slopes, where=(slopes > 0),
                      color='tab:green', alpha=0.20)
    ax2.fill_between(dts, 0, slopes, where=(slopes < 0),
                      color='tab:red', alpha=0.20)
    ax2.set_ylabel('RM slope (pts/1m bar)', fontsize=11)
    ax2.set_xlabel('Time (UTC)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(alpha=0.25)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    os.makedirs(CHARTS_DIR, exist_ok=True)
    out_path = os.path.join(CHARTS_DIR, f'rm_slope_trades_{args.day}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {out_path}')
    print(f'{args.day}: {len(day_trades)} trades, {longs}L/{shorts}S, '
          f'count-WR {dayWR:.0f}%, total ${total:+,.0f}')


if __name__ == '__main__':
    main()
