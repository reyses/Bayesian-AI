"""
CNN Conviction Plot — simple P(direction) per bar relative to previous bar.

For each bar i, using the last 10 bars as input:
  - CNN predicts dmi_diff at n+1
  - P(long) = sigmoid(predicted_dmi_diff) → 0 to 1
  - >0.5 = expects UP, <0.5 = expects DOWN
  - Distance from 0.5 = conviction strength

Two panels:
  1. Price (line) with P(long) color gradient (green→yellow→red)
  2. P(long) as a line (0 to 1) with 0.5 centerline

Usage:
  python -m tools.cnn_conviction_plot --date 2026-03-26 --start 14:00 --end 18:00
"""
import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.trade_cnn import StatePredictor
from training.train_trade_cnn import extract_features_13d

ATLAS_ROOT = 'DATA/ATLAS'
CHECKPOINT = 'checkpoints/trade_cnn/best_model.pt'
TICK = 0.25
LOOKBACK = 10
OUT_DIR = 'reports/findings'


def main():
    parser = argparse.ArgumentParser(description='CNN Conviction Plot')
    parser.add_argument('--date', default='2026-03-26')
    parser.add_argument('--start', default='14:00')
    parser.add_argument('--end', default='18:00')
    parser.add_argument('--tf', default='1m')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    args = parser.parse_args()

    # Load data
    month_str = args.date[:7].replace('-', '_')
    path = os.path.join(ATLAS_ROOT, args.tf, f'{month_str}.parquet')
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)

    # Filter time range
    date_ts = pd.Timestamp(args.date).timestamp()
    sh, sm = map(int, args.start.split(':'))
    eh, em = map(int, args.end.split(':'))
    start_ts = date_ts + sh * 3600 + sm * 60
    end_ts = date_ts + eh * 3600 + em * 60
    if end_ts <= start_ts:
        end_ts += 86400

    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)].reset_index(drop=True)
    n = len(df)
    print(f"  {n} bars from {args.start} to {args.end}")

    if n < LOOKBACK + 5:
        print("  Not enough bars")
        return

    # SFE + features
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    feats = extract_features_13d(states, df)
    del states, sfe; gc.collect()

    # Load CNN
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    model_state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
    model = StatePredictor(n_features=13, latent_dim=64, n_labels=21).to(dev)
    model.load_state_dict(model_state)
    model.eval()

    # Run CNN: for each bar, predict n+1 dmi_diff → sigmoid → P(long)
    p_long = np.full(n, np.nan)

    with torch.no_grad():
        for i in range(LOOKBACK, n):
            window = feats[i - LOOKBACK:i]
            x = torch.FloatTensor(window).unsqueeze(0).to(dev)
            pred = model(x)[0].cpu().numpy()
            # pred[0] = predicted dmi_diff at n+1 (first horizon, first feature)
            dmi_pred = pred[0]
            # Convert to probability via sigmoid
            p_long[i] = 1.0 / (1.0 + np.exp(-dmi_pred / 5.0))  # scale: dmi/5 so +-10 maps to ~0.88/0.12

    # Timestamps for plotting
    timestamps = pd.to_datetime(df['timestamp'].values, unit='s', utc=True)
    prices = df['close'].values

    # ── Plot ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'CNN Conviction — {args.date} {args.tf} ({n} bars)', fontsize=13, fontweight='bold')

    # Panel 1: Price colored by P(long)
    ax1.set_facecolor('#1a1a2e')
    # Create colored line segments
    points = np.array([mdates.date2num(timestamps), prices]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color: green (P>0.5) → yellow (P=0.5) → red (P<0.5)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('conviction',
        ['#EF5350', '#EF5350', '#FFD700', '#26A69A', '#26A69A'])
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Use P(long) of the second point of each segment for coloring
    seg_colors = np.array([p_long[i+1] if not np.isnan(p_long[i+1]) else 0.5
                           for i in range(n-1)])

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2)
    lc.set_array(seg_colors)
    ax1.add_collection(lc)
    ax1.set_xlim(mdates.date2num(timestamps[0]), mdates.date2num(timestamps[-1]))
    ax1.set_ylim(prices.min() - 10, prices.max() + 10)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.15)

    # Colorbar
    cb = fig.colorbar(lc, ax=ax1, pad=0.01, aspect=30)
    cb.set_label('P(long)', fontsize=9)

    # Panel 2: P(long) line
    ax2.set_facecolor('#1a1a2e')
    valid = ~np.isnan(p_long)
    ax2.plot(timestamps[valid], p_long[valid], color='#00FFFF', linewidth=1.2)
    ax2.axhline(y=0.5, color='white', linewidth=1, alpha=0.5, linestyle='--')
    ax2.axhline(y=0.65, color='#26A69A', linewidth=0.5, alpha=0.3, linestyle=':')
    ax2.axhline(y=0.35, color='#EF5350', linewidth=0.5, alpha=0.3, linestyle=':')
    ax2.fill_between(timestamps[valid], 0.5, p_long[valid],
                      where=p_long[valid] > 0.5, color='#26A69A', alpha=0.3)
    ax2.fill_between(timestamps[valid], 0.5, p_long[valid],
                      where=p_long[valid] < 0.5, color='#EF5350', alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('P(long)')
    ax2.grid(True, alpha=0.15)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f'cnn_conviction_{args.date.replace("-","")}.png')
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Stats
    valid_p = p_long[valid]
    print(f"\n  P(long) stats:")
    print(f"    mean={np.mean(valid_p):.3f}  median={np.median(valid_p):.3f}")
    print(f"    >0.65 (LONG):  {np.sum(valid_p > 0.65)} bars ({np.sum(valid_p > 0.65)/len(valid_p)*100:.0f}%)")
    print(f"    <0.35 (SHORT): {np.sum(valid_p < 0.35)} bars ({np.sum(valid_p < 0.35)/len(valid_p)*100:.0f}%)")
    print(f"    0.35-0.65 (uncertain): {np.sum((valid_p >= 0.35) & (valid_p <= 0.65))} bars")


if __name__ == '__main__':
    main()
