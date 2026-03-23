"""Peak Verification Chart — overlay detected peaks on 1-day 1s price.

Visual verification of the peak prediction research. Shows:
  - Price action at 1s resolution
  - Detected peaks marked (green = correct reversion, red = wrong)
  - Velocity panel
  - Volume panel with collapse threshold

Uses ONLY grounded features. No SFE, no PhysicsEngine.

Usage:
    python tools/peak_verification_chart.py
    python tools/peak_verification_chart.py --day 2026-02-05
    python tools/peak_verification_chart.py --data DATA/ATLAS_1WEEK
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25

# Peak detection config (best from research: vel_w=10, mag_p75, vol<0.3)
VEL_WINDOW = 10
MAG_WINDOW = 20
MAG_PCTILE = 0.75
VOL_COLLAPSE = 0.3
VOL_AVG_WINDOW = 60
LOOKAHEAD = 10  # verify over next 10s


def load_day(data_dir: str, target_day: str = None):
    """Load 1s data, optionally filter to one day."""
    files = sorted(glob.glob(os.path.join(data_dir, '1s', '*.parquet')))
    if not files:
        print(f'ERROR: No 1s parquet in {data_dir}/1s/')
        sys.exit(1)

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if 'timestamp' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['date'] = df['dt'].dt.strftime('%Y-%m-%d')

        if target_day:
            df = df[df['date'] == target_day]
        else:
            # Pick first full trading day
            day_counts = df.groupby('date').size()
            full_days = day_counts[day_counts > 5000]
            if len(full_days) > 0:
                target_day = full_days.index[0]
                df = df[df['date'] == target_day]

    print(f'Day: {target_day or "all"} | {len(df):,} bars')
    return df, target_day


def compute_and_detect(closes, volumes):
    """Compute features and detect peaks."""
    n = len(closes)
    dp = np.diff(closes) / TICK
    dp = np.concatenate([[0], dp])

    # Velocity
    velocity = np.full(n, 0.0)
    for i in range(VEL_WINDOW, n):
        velocity[i] = np.mean(dp[i-VEL_WINDOW:i])

    vel_sign = np.sign(velocity)

    # Velocity flip
    flips = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if vel_sign[i] != 0 and vel_sign[i-1] != 0 and vel_sign[i] != vel_sign[i-1]:
            flips[i] = True

    # Magnitude
    magnitude = np.full(n, 0.0)
    for i in range(MAG_WINDOW, n):
        magnitude[i] = abs(closes[i] - closes[i-MAG_WINDOW]) / TICK

    # Magnitude percentile (causal, 1h lookback)
    pctile_w = 3600
    mag_pct = np.full(n, 0.0)
    for i in range(max(MAG_WINDOW, pctile_w), n):
        recent = magnitude[i-pctile_w:i]
        mag_pct[i] = np.sum(recent < magnitude[i]) / len(recent)

    # Volume average
    vol_avg = np.full(n, 0.0)
    for i in range(VOL_AVG_WINDOW, n):
        vol_avg[i] = np.mean(volumes[i-VOL_AVG_WINDOW:i])

    # Detect peaks
    peaks = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not flips[i]:
            continue
        if mag_pct[i] < MAG_PCTILE:
            continue
        if vol_avg[i] > 0 and volumes[i] / vol_avg[i] > VOL_COLLAPSE:
            continue
        peaks[i] = True

    # Verify: was the reversion prediction correct?
    correct = np.zeros(n, dtype=bool)
    wrong = np.zeros(n, dtype=bool)
    for i in range(n - LOOKAHEAD):
        if not peaks[i]:
            continue
        future_change = closes[i + LOOKAHEAD] - closes[i]
        pred_sign = vel_sign[i]  # new direction after flip = reversion
        actual_sign = 1 if future_change > 0 else (-1 if future_change < 0 else 0)
        if pred_sign == actual_sign:
            correct[i] = True
        elif actual_sign != 0:
            wrong[i] = True

    return {
        'dp': dp,
        'velocity': velocity,
        'magnitude': magnitude,
        'mag_pct': mag_pct,
        'vol_avg': vol_avg,
        'peaks': peaks,
        'correct': correct,
        'wrong': wrong,
        'vel_sign': vel_sign,
    }


def plot_day(df, features, day_label):
    """Plot one day with peaks overlaid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, MinuteLocator

    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    times = df['dt'].values if 'dt' in df.columns else np.arange(len(df))

    peaks = features['peaks']
    correct = features['correct']
    wrong = features['wrong']
    velocity = features['velocity']
    vol_avg = features['vol_avg']

    n_peaks = int(np.sum(peaks))
    n_correct = int(np.sum(correct))
    n_wrong = int(np.sum(wrong))
    acc = n_correct / max(1, n_correct + n_wrong) * 100

    fig, axes = plt.subplots(3, 1, figsize=(48, 20), sharex=True,
                             gridspec_kw={'height_ratios': [4, 1.5, 1]})

    # Panel 1: Price with peak markers
    ax1 = axes[0]
    ax1.plot(times, closes, color='#333333', lw=0.5, alpha=0.8)

    # Green diamonds: correct reversion
    correct_idx = np.where(correct)[0]
    if len(correct_idx) > 0:
        ax1.scatter(times[correct_idx], closes[correct_idx],
                    marker='D', c='lime', s=120, zorder=5, edgecolors='green',
                    lw=1.5, label=f'Correct ({n_correct})')

    # Red diamonds: wrong prediction
    wrong_idx = np.where(wrong)[0]
    if len(wrong_idx) > 0:
        ax1.scatter(times[wrong_idx], closes[wrong_idx],
                    marker='D', c='red', s=120, zorder=5, edgecolors='darkred',
                    lw=1.5, label=f'Wrong ({n_wrong})')

    # Draw arrows showing predicted direction at each peak
    for idx in np.where(peaks)[0]:
        direction = features['vel_sign'][idx]
        arrow_size = (closes.max() - closes.min()) * 0.02
        color = 'lime' if correct[idx] else 'red'
        ax1.annotate('', xy=(times[idx], closes[idx] + direction * arrow_size),
                     xytext=(times[idx], closes[idx]),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'{day_label} — Peak Detection Verification  |  '
                  f'{n_peaks} peaks  |  {acc:.0f}% correct  |  '
                  f'Config: vel_w={VEL_WINDOW} mag>p{MAG_PCTILE*100:.0f} vol<{VOL_COLLAPSE}',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Velocity
    ax2 = axes[1]
    vel_colors = np.where(velocity > 0, 'green', 'red')
    ax2.bar(times, velocity, width=np.timedelta64(800, 'ms') if hasattr(times[0], 'astype') else 0.8,
            color=vel_colors, alpha=0.6)
    ax2.axhline(0, color='black', lw=0.5)

    # Mark flips
    peak_idx = np.where(peaks)[0]
    if len(peak_idx) > 0:
        ax2.scatter(times[peak_idx], velocity[peak_idx],
                    marker='v', c='orange', s=60, zorder=5, label='Peak (vel flip)')

    ax2.set_ylabel(f'Velocity ({VEL_WINDOW}s)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Panel 3: Volume with collapse threshold
    ax3 = axes[2]
    ax3.bar(times, volumes, width=np.timedelta64(800, 'ms') if hasattr(times[0], 'astype') else 0.8,
            color='steelblue', alpha=0.5)
    ax3.plot(times, vol_avg * VOL_COLLAPSE, color='red', lw=1, ls='--',
             label=f'Collapse threshold ({VOL_COLLAPSE*100:.0f}% of avg)', alpha=0.7)
    ax3.set_ylabel('Volume', fontsize=10)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    if hasattr(times[0], 'astype'):
        ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    plt.tight_layout()
    out_dir = os.path.join('tools', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'peak_verification_{day_label}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Chart: {out_path}')
    print(f'  Peaks: {n_peaks} | Correct: {n_correct} | Wrong: {n_wrong} | Accuracy: {acc:.1f}%')


def main():
    parser = argparse.ArgumentParser(description='Peak verification chart')
    parser.add_argument('--data', default='DATA/ATLAS_OOS',
                        help='Data directory')
    parser.add_argument('--day', default=None,
                        help='Target day (YYYY-MM-DD)')
    args = parser.parse_args()

    df, day_label = load_day(args.data, args.day)
    if len(df) < 1000:
        print(f'Too few bars ({len(df)}). Need a full trading day.')
        sys.exit(1)

    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))

    print('Detecting peaks...')
    features = compute_and_detect(closes, volumes)

    plot_day(df, features, day_label or 'unknown')


if __name__ == '__main__':
    main()
