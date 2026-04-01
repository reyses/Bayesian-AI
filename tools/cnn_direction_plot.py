"""
CNN Direction Visualizer — plots price + CNN P(long) trajectory overlay.

Shows what the CNN sees vs what price actually did. Helps diagnose:
  - Where CNN is right but system exits too early
  - Where CNN is wrong (LONG during a drop)
  - Where CNN is uncertain (P near 0.5)

Panels:
  1. Price (candlestick) + CNN direction color band (green=LONG, red=SHORT, yellow=uncertain)
  2. P(long) at 3 horizons (n+1, n+5, n+10) — the trajectory curve
  3. Oscillation metrics: z_se + amplitude envelope
  4. DMI diff (raw) — what's driving the direction

Usage:
  python -m tools.cnn_direction_plot --date 2026-03-30 --hours 4
  python -m tools.cnn_direction_plot --date 2026-03-30 --start 23:00 --end 03:00
  python -m tools.cnn_direction_plot --date 2026-03-30 --trades reports/live/session_20260331.txt
"""
import argparse
import gc
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.trade_cnn import StatePredictor
from training.train_trade_cnn import extract_features_13d, FEATURE_NAMES_7D, HORIZONS

ATLAS_ROOT = 'DATA/ATLAS'
CHECKPOINT = 'checkpoints/trade_cnn/best_model.pt'
TICK = 0.25
LOOKBACK = 10  # CNN lookback window
OUT_DIR = 'reports/findings'


def load_model(checkpoint_path, device='cuda'):
    """Load frozen StatePredictor."""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model_state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
    model = StatePredictor(n_features=13, latent_dim=64, n_labels=21).to(dev)
    model.load_state_dict(model_state)
    model.eval()
    return model, dev


def run_cnn_over_data(df, feats_13d, model, device):
    """Run CNN over all bars. Returns per-bar predictions."""
    n = len(df)
    # Output: P(long) proxy from predicted dmi_diff at each horizon
    predictions = np.full((n, 3), np.nan)  # 3 horizons: h0, h1, h2
    directions = np.full(n, np.nan)        # aggregate direction: >0 = LONG, <0 = SHORT

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(LOOKBACK, n), desc="CNN predict"):
            window = feats_13d[i - LOOKBACK:i]
            x = torch.FloatTensor(window).unsqueeze(0).to(device)
            pred = model(x)[0].cpu().numpy()

            # Predicted dmi_diff at each horizon (index 0 of each 7D block)
            h0 = pred[0]          # horizon[0] = t+1
            h1 = pred[7]          # horizon[1] = t+5
            h2 = pred[14]         # horizon[2] = t+10

            predictions[i, 0] = h0
            predictions[i, 1] = h1
            predictions[i, 2] = h2

            # Direction: average of all 3 horizons
            directions[i] = (h0 + h1 + h2) / 3.0

    return predictions, directions


def parse_trades_from_session(session_path):
    """Parse trade log from session report. Returns list of dicts."""
    trades = []
    in_log = False
    with open(session_path, 'r') as f:
        for line in f:
            if 'Time       Side' in line:
                in_log = True
                continue
            if in_log and '----' in line:
                if trades:  # second dashes = end of log
                    break
                continue
            if in_log and line.strip():
                # Parse: "   1  23:00:12   SHORT   23,310.00  23,307.50 $    +5.00 unknown            1"
                parts = line.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    try:
                        trades.append({
                            'time': parts[1],
                            'side': parts[2],
                            'entry': float(parts[3].replace(',', '')),
                            'exit': float(parts[4].replace(',', '')),
                            'pnl': float(parts[5].replace('$', '').replace(',', '').replace('+', '')),
                        })
                    except (ValueError, IndexError):
                        pass
    return trades


def plot_cnn_direction(df, predictions, directions, feats_13d, trades=None,
                       title='CNN Direction Overlay', out_path=None):
    """4-panel plot: price+direction, P(long) horizons, z_se+amplitude, DMI."""

    timestamps = pd.to_datetime(df['timestamp'].values, unit='s', utc=True)
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n = len(df)

    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1.5, 1, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Panel 1: Price + CNN direction color band ──────────────
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')

    # Draw candlesticks
    for i in range(n):
        o, c, h, l = opens[i], prices[i], highs[i], lows[i]
        color = '#26A69A' if c >= o else '#EF5350'
        ax1.plot([timestamps[i], timestamps[i]], [l, h], color='#666666', linewidth=0.5)
        body = max(abs(c - o), TICK)
        bottom = min(o, c)
        width = timedelta(seconds=40)
        ax1.bar(timestamps[i], body, bottom=bottom, width=width, color=color,
                edgecolor='none', alpha=0.8)

    # CNN direction as background color
    for i in range(LOOKBACK, n):
        d = directions[i]
        if np.isnan(d):
            continue
        if d > 2.0:
            color = '#26A69A'    # strong LONG
            alpha = min(0.4, abs(d) / 20)
        elif d < -2.0:
            color = '#EF5350'    # strong SHORT
            alpha = min(0.4, abs(d) / 20)
        else:
            color = '#FFD700'    # uncertain
            alpha = 0.1
        ax1.axvspan(timestamps[i] - timedelta(seconds=30),
                    timestamps[i] + timedelta(seconds=30),
                    color=color, alpha=alpha, linewidth=0)

    # Trade markers
    if trades:
        for t in trades:
            # Find closest timestamp
            t_time = t['time']
            for i in range(n):
                bar_time = timestamps[i].strftime('%H:%M')
                if bar_time >= t_time[:5]:
                    marker = '^' if t['side'] == 'LONG' else 'v'
                    color = '#26A69A' if t['pnl'] > 0 else '#EF5350'
                    ax1.scatter(timestamps[i], t['entry'], marker=marker,
                               color=color, s=60, zorder=5, edgecolors='white', linewidths=0.5)
                    break

    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.15)
    ax1.legend(['CNN: green=LONG, red=SHORT, yellow=uncertain'], loc='upper left',
               fontsize=8, framealpha=0.7)

    # ── Panel 2: P(long) at 3 horizons ────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    h_labels = [f't+{HORIZONS[i]}' for i in range(3)]
    colors_h = ['#00FFFF', '#FF00FF', '#FFD700']

    for hi in range(3):
        valid = ~np.isnan(predictions[:, hi])
        ax2.plot(timestamps[valid], predictions[valid, hi],
                 color=colors_h[hi], linewidth=1, alpha=0.8, label=h_labels[hi])

    ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax2.axhline(y=2.0, color='#26A69A', linewidth=0.5, linestyle='--', alpha=0.3, label='LONG threshold')
    ax2.axhline(y=-2.0, color='#EF5350', linewidth=0.5, linestyle='--', alpha=0.3, label='SHORT threshold')
    ax2.set_ylabel('Predicted DMI diff')
    ax2.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax2.grid(True, alpha=0.15)

    # ── Panel 3: z_se + amplitude envelope ─────────────────────
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a2e')
    z_se = feats_13d[:, 5]  # index 5 = z_se
    ax3.plot(timestamps, z_se, color='#00FFFF', linewidth=0.8, label='z_se')

    # Rolling amplitude: max(z_se) - min(z_se) over 30 bars
    if n > 30:
        z_series = pd.Series(z_se)
        z_max = z_series.rolling(30, min_periods=5).max().values
        z_min = z_series.rolling(30, min_periods=5).min().values
        ax3.fill_between(timestamps, z_min, z_max, color='#00FFFF', alpha=0.1, label='amplitude')
        ax3.plot(timestamps, z_max, color='#26A69A', linewidth=0.5, alpha=0.5)
        ax3.plot(timestamps, z_min, color='#EF5350', linewidth=0.5, alpha=0.5)

    ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax3.set_ylabel('z_se')
    ax3.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax3.grid(True, alpha=0.15)

    # ── Panel 4: Raw DMI diff ──────────────────────────────────
    ax4 = axes[3]
    ax4.set_facecolor('#1a1a2e')
    dmi_diff = feats_13d[:, 0]  # index 0 = dmi_diff
    dmi_colors = ['#26A69A' if d > 0 else '#EF5350' for d in dmi_diff]
    ax4.bar(timestamps, dmi_diff, color=dmi_colors, width=timedelta(seconds=40), alpha=0.7)
    ax4.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax4.set_ylabel('DMI diff')
    ax4.grid(True, alpha=0.15)

    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUT_DIR, 'cnn_direction_overlay.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='CNN Direction Visualizer')
    parser.add_argument('--date', default='2026-03-30', help='Date to plot (YYYY-MM-DD)')
    parser.add_argument('--hours', type=float, default=4, help='Hours to show')
    parser.add_argument('--start', default=None, help='Start time HH:MM (overrides --hours)')
    parser.add_argument('--end', default=None, help='End time HH:MM')
    parser.add_argument('--trades', default=None, help='Path to session report for trade markers')
    parser.add_argument('--tf', default='1m', help='Timeframe')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='CNN checkpoint path')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    print(f"\nCNN Direction Visualizer")
    print(f"  Date: {args.date} | TF: {args.tf} | Hours: {args.hours}")

    # Load data
    month_str = args.date[:7].replace('-', '_')
    path = os.path.join(ATLAS_ROOT, args.tf, f'{month_str}.parquet')
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return

    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)

    # Filter to date range
    date_ts = pd.Timestamp(args.date).timestamp()
    if args.start and args.end:
        sh, sm = map(int, args.start.split(':'))
        eh, em = map(int, args.end.split(':'))
        start_ts = date_ts + sh * 3600 + sm * 60
        end_ts = date_ts + eh * 3600 + em * 60
        if end_ts <= start_ts:
            end_ts += 86400  # crosses midnight
    else:
        # Default: show --hours from start of day
        start_ts = date_ts
        end_ts = start_ts + args.hours * 3600

    print(f"  Filter: {datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M')} "
          f"to {datetime.utcfromtimestamp(end_ts).strftime('%H:%M')} UTC")

    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)].reset_index(drop=True)
    if len(df) < LOOKBACK + 5:
        print(f"  ERROR: only {len(df)} bars in range")
        return

    print(f"  Bars: {len(df)} ({args.tf})")
    print(f"  Range: {datetime.utcfromtimestamp(df['timestamp'].iloc[0]).strftime('%Y-%m-%d %H:%M')} "
          f"to {datetime.utcfromtimestamp(df['timestamp'].iloc[-1]).strftime('%H:%M')}")

    # Compute features
    print(f"  Computing SFE states + 13D features...")
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    feats_13d = extract_features_13d(states, df)
    del states, sfe; gc.collect()

    # Load CNN
    print(f"  Loading CNN: {args.checkpoint}")
    model, device = load_model(args.checkpoint)

    # Run predictions
    predictions, directions = run_cnn_over_data(df, feats_13d, model, device)

    # Stats
    valid = ~np.isnan(directions)
    n_long = np.sum(directions[valid] > 2.0)
    n_short = np.sum(directions[valid] < -2.0)
    n_uncertain = np.sum((directions[valid] >= -2.0) & (directions[valid] <= 2.0))
    print(f"  CNN direction: LONG={n_long} SHORT={n_short} uncertain={n_uncertain}")

    # Load trades if provided
    trades = None
    if args.trades and os.path.exists(args.trades):
        trades = parse_trades_from_session(args.trades)
        print(f"  Trades loaded: {len(trades)} from {args.trades}")

    # Plot
    date_str = args.date.replace('-', '')
    out_path = os.path.join(OUT_DIR, f'cnn_direction_{date_str}.png')
    title = f'CNN Direction Overlay — {args.date} {args.tf} ({len(df)} bars)'
    plot_cnn_direction(df, predictions, directions, feats_13d, trades=trades,
                       title=title, out_path=out_path)


if __name__ == '__main__':
    main()
