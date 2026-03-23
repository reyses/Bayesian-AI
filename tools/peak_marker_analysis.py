"""Peak Marker Analysis — analyze 1s data before human-marked peaks.

Loads human peaks from all TFs, overlays them, and examines the grounded
features in the 60s before each marked peak.

Questions answered:
  1. Which 5m peaks sit near 4h/1h walls? (within N ticks)
  2. What did velocity, volume, std look like before each peak?
  3. Is there a consistent grounded signature across TFs?
  4. How does the 1s data differ before REAL peaks vs random bars?

Usage:
    python tools/peak_marker_analysis.py --date 2026-02-05
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
SEEDS_DIR = 'DATA/regime_seeds'
LOOKBACK_SECS = 60  # analyze 60s before each peak
WALL_PROXIMITY_TICKS = 40  # peak within 40 ticks of higher-TF wall = "near wall"


def load_human_peaks(date_str):
    """Load all human peak files for a date."""
    peaks_by_tf = {}
    # Match both single-date and range files that contain this date
    for f in glob.glob(os.path.join(SEEDS_DIR, 'human_peaks_*.json')):
        with open(f) as fh:
            data = json.load(fh)
        tf = data.get('tf', '?')
        # Check if this file covers the requested date
        fname = os.path.basename(f)
        if date_str in fname or 'to' in fname:
            peaks_by_tf[tf] = data['peaks']
            print(f'  {tf}: {len(data["peaks"])} peaks from {fname}')
    return peaks_by_tf


def load_1s_data(data_dir, date_str):
    """Load 1s data for one day."""
    month = date_str[:4] + '_' + date_str[5:7]
    files = sorted(glob.glob(os.path.join(data_dir, '1s', f'{month}*.parquet')))
    if not files:
        files = sorted(glob.glob(os.path.join('DATA/ATLAS_OOS', '1s', f'{month}*.parquet')))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df[df['dt'].dt.strftime('%Y-%m-%d') == date_str].reset_index(drop=True)
    return df


def compute_1s_features(df):
    """Compute grounded features on 1s data."""
    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    timestamps = df['timestamp'].values
    n = len(closes)

    dp = np.diff(closes) / TICK
    dp = np.concatenate([[0], dp])

    # Velocity (5-bar)
    velocity = np.full(n, 0.0)
    for i in range(5, n):
        velocity[i] = np.mean(dp[i-5:i])

    # Volume average (60s)
    vol_avg = np.full(n, 0.0)
    for i in range(60, n):
        vol_avg[i] = np.mean(volumes[i-60:i])

    # Std of price changes (20s)
    std_price = np.full(n, 0.0)
    for i in range(20, n):
        std_price[i] = np.std(dp[i-20:i], ddof=1)

    # Acceleration (change in velocity)
    accel = np.diff(velocity)
    accel = np.concatenate([[0], accel])

    # DMI proxy (directional movement from highs/lows)
    highs = df['high'].values
    lows = df['low'].values
    up_move = np.concatenate([[0], np.diff(highs)]) / TICK
    down_move = np.concatenate([[0], -np.diff(lows)]) / TICK
    dmi_proxy = up_move - down_move

    # Wick ratio
    bar_range = highs - lows
    wick_upper = highs - np.maximum(df['open'].values, closes) if 'open' in df.columns else np.zeros(n)
    wick_lower = np.minimum(df['open'].values, closes) - lows if 'open' in df.columns else np.zeros(n)

    return {
        'dp': dp,
        'velocity': velocity,
        'acceleration': accel,
        'volumes': volumes,
        'vol_avg': vol_avg,
        'std_price': std_price,
        'dmi_proxy': dmi_proxy,
        'closes': closes,
        'timestamps': timestamps,
        'bar_range': bar_range,
        'wick_upper': wick_upper,
        'wick_lower': wick_lower,
    }


def find_nearest_1s_bar(timestamps, peak_ts):
    """Find the 1s bar index closest to a peak timestamp."""
    idx = np.argmin(np.abs(timestamps - peak_ts))
    return int(idx)


def analyze_peak_window(features, center_idx, lookback=LOOKBACK_SECS):
    """Extract grounded feature stats for the window before a peak."""
    n = len(features['closes'])
    start = max(0, center_idx - lookback)
    end = center_idx

    if end - start < 10:
        return None

    window = slice(start, end)
    dp = features['dp'][window]
    vel = features['velocity'][window]
    vol = features['volumes'][window]
    vol_avg = features['vol_avg'][window]
    std_p = features['std_price'][window]
    accel = features['acceleration'][window]
    dmi = features['dmi_proxy'][window]

    # Velocity trajectory: was it building or collapsing?
    vel_first_half = np.mean(vel[:len(vel)//2]) if len(vel) > 1 else 0
    vel_second_half = np.mean(vel[len(vel)//2:]) if len(vel) > 1 else 0
    vel_trend = vel_second_half - vel_first_half  # negative = decelerating

    # Volume trajectory
    vol_first = np.mean(vol[:len(vol)//2]) if len(vol) > 1 else 0
    vol_second = np.mean(vol[len(vol)//2:]) if len(vol) > 1 else 0
    vol_trend = vol_second - vol_first  # negative = drying up

    # Volume ratio at peak
    peak_vol = features['volumes'][center_idx]
    peak_vol_avg = features['vol_avg'][center_idx]
    vol_ratio = peak_vol / peak_vol_avg if peak_vol_avg > 0 else 1.0

    return {
        'vel_mean': float(np.mean(vel)),
        'vel_abs_mean': float(np.mean(np.abs(vel))),
        'vel_at_peak': float(features['velocity'][center_idx]),
        'vel_trend': float(vel_trend),
        'vel_max': float(np.max(np.abs(vel))),
        'accel_mean': float(np.mean(accel)),
        'accel_at_peak': float(features['acceleration'][center_idx]),
        'vol_mean': float(np.mean(vol)),
        'vol_trend': float(vol_trend),
        'vol_ratio_at_peak': float(vol_ratio),
        'std_price_mean': float(np.mean(std_p)),
        'std_price_at_peak': float(std_p[-1] if len(std_p) > 0 else 0),
        'dmi_mean': float(np.mean(dmi)),
        'dmi_abs_max': float(np.max(np.abs(dmi))),
        'magnitude': float(abs(features['closes'][center_idx] - features['closes'][start]) / TICK),
        'bar_range_mean': float(np.mean(features['bar_range'][window])),
    }


def analyze_random_windows(features, n_random=500, lookback=LOOKBACK_SECS):
    """Analyze random non-peak windows for comparison."""
    n = len(features['closes'])
    np.random.seed(42)
    indices = np.random.choice(range(lookback + 10, n - 10), size=n_random, replace=False)

    results = []
    for idx in indices:
        stats = analyze_peak_window(features, idx, lookback)
        if stats:
            results.append(stats)
    return results


def check_wall_proximity(peak_price, higher_tf_peaks, threshold_ticks=WALL_PROXIMITY_TICKS):
    """Check if a peak is near a higher-TF wall."""
    for hp in higher_tf_peaks:
        wall_price = hp['price']
        dist_ticks = abs(peak_price - wall_price) / TICK
        if dist_ticks <= threshold_ticks:
            return True, wall_price, dist_ticks
    return False, 0, 0


def main():
    parser = argparse.ArgumentParser(description='Analyze human-marked peaks')
    parser.add_argument('--date', default='2026-02-05')
    parser.add_argument('--data', default='DATA/ATLAS_OOS')
    args = parser.parse_args()

    print(f'Loading human peaks for {args.date}...')
    peaks_by_tf = load_human_peaks(args.date)
    if not peaks_by_tf:
        print('No human peaks found. Run peak_marker.py first.')
        sys.exit(1)

    print(f'\nLoading 1s data...')
    df_1s = load_1s_data(args.data, args.date)
    if df_1s is None or len(df_1s) == 0:
        print('No 1s data found.')
        sys.exit(1)
    print(f'  {len(df_1s):,} bars')

    print('Computing grounded features...')
    features = compute_1s_features(df_1s)

    # Build wall list from higher TFs (4h, 1h)
    walls = []
    for tf in ['4h', '1h']:
        if tf in peaks_by_tf:
            walls.extend(peaks_by_tf[tf])

    lines = []
    lines.append('=' * 90)
    lines.append(f'PEAK MARKER ANALYSIS — {args.date}')
    lines.append(f'1s data: {len(df_1s):,} bars')
    lines.append(f'TFs loaded: {", ".join(sorted(peaks_by_tf.keys()))}')
    lines.append(f'Walls (4h+1h): {len(walls)} levels')
    lines.append('=' * 90)

    # Analyze each TF
    all_peak_stats = []
    for tf in ['4h', '1h', '15m', '5m', '1m']:
        if tf not in peaks_by_tf:
            continue

        peaks = peaks_by_tf[tf]
        lines.append(f'\n{"="*60}')
        lines.append(f'TF: {tf} — {len(peaks)} peaks')
        lines.append(f'{"="*60}')

        tf_stats = []
        for i, peak in enumerate(peaks):
            ts = peak['timestamp']
            idx = find_nearest_1s_bar(features['timestamps'], ts)

            stats = analyze_peak_window(features, idx)
            if not stats:
                continue

            # Wall proximity (for 5m and 1m peaks)
            near_wall, wall_px, wall_dist = check_wall_proximity(peak['price'], walls)
            stats['near_wall'] = near_wall
            stats['wall_dist_ticks'] = float(wall_dist)
            stats['direction'] = peak.get('direction', '?')
            stats['price'] = peak['price']
            stats['time'] = peak.get('time_utc', '?')
            tf_stats.append(stats)
            all_peak_stats.append({**stats, 'tf': tf})

            wall_str = f' NEAR WALL ({wall_dist:.0f}t)' if near_wall else ''
            lines.append(f'  #{i+1} {peak.get("time_utc","?")} {peak.get("direction","?")} '
                         f'@ {peak["price"]:.2f}{wall_str}')
            lines.append(f'    vel: mean={stats["vel_mean"]:+.2f} trend={stats["vel_trend"]:+.2f} '
                         f'at_peak={stats["vel_at_peak"]:+.2f}')
            lines.append(f'    vol: ratio={stats["vol_ratio_at_peak"]:.2f} '
                         f'trend={stats["vol_trend"]:+.1f}')
            lines.append(f'    std: {stats["std_price_at_peak"]:.2f}  '
                         f'mag: {stats["magnitude"]:.0f}t  '
                         f'accel: {stats["accel_at_peak"]:+.3f}')

        if tf_stats:
            lines.append(f'\n  {tf} AVERAGES (n={len(tf_stats)}):')
            for key in ['vel_abs_mean', 'vel_trend', 'vol_ratio_at_peak', 'vol_trend',
                         'std_price_at_peak', 'magnitude', 'accel_mean']:
                vals = [s[key] for s in tf_stats]
                lines.append(f'    {key:<25} mean={np.mean(vals):+.3f}  std={np.std(vals):.3f}')

    # Random baseline comparison
    print('Computing random baseline...')
    random_stats = analyze_random_windows(features, n_random=500)

    lines.append(f'\n{"="*60}')
    lines.append('PEAK vs RANDOM COMPARISON')
    lines.append(f'{"="*60}')
    lines.append(f'  {"Feature":<25} {"Peak Mean":>12} {"Random Mean":>12} {"Delta":>10} {"Signal?":>8}')
    lines.append('  ' + '-' * 70)

    for key in ['vel_abs_mean', 'vel_trend', 'vol_ratio_at_peak', 'vol_trend',
                 'std_price_at_peak', 'magnitude', 'accel_mean', 'dmi_abs_max']:
        peak_vals = [s[key] for s in all_peak_stats if key in s]
        rand_vals = [s[key] for s in random_stats if key in s]
        if peak_vals and rand_vals:
            pm = np.mean(peak_vals)
            rm = np.mean(rand_vals)
            delta = pm - rm
            # Simple significance: delta > 0.5 * std of random
            sig = abs(delta) > 0.5 * np.std(rand_vals)
            lines.append(f'  {key:<25} {pm:>+12.3f} {rm:>+12.3f} {delta:>+10.3f} '
                         f'{"YES" if sig else "no":>8}')

    # Wall proximity analysis (for 5m peaks)
    if '5m' in peaks_by_tf:
        five_min_stats = [s for s in all_peak_stats if s.get('tf') == '5m']
        near = [s for s in five_min_stats if s.get('near_wall')]
        far = [s for s in five_min_stats if not s.get('near_wall')]
        lines.append(f'\n{"="*60}')
        lines.append(f'5m PEAKS: NEAR WALL vs FAR FROM WALL')
        lines.append(f'{"="*60}')
        lines.append(f'  Near wall (<{WALL_PROXIMITY_TICKS}t): {len(near)} peaks')
        lines.append(f'  Far from wall: {len(far)} peaks')
        if near and far:
            for key in ['vel_abs_mean', 'vol_ratio_at_peak', 'std_price_at_peak', 'magnitude']:
                nm = np.mean([s[key] for s in near])
                fm = np.mean([s[key] for s in far])
                lines.append(f'    {key:<25} near={nm:+.3f}  far={fm:+.3f}  delta={nm-fm:+.3f}')

    summary = '\n'.join(lines)

    out_path = os.path.join('reports', 'findings', f'peak_marker_analysis_{args.date}.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nSummary: {out_path}')
    print('\n' + summary)

    # Plot
    _plot(df_1s, features, peaks_by_tf, walls, args.date)


def _plot(df_1s, features, peaks_by_tf, walls, date_str):
    """Overlay all TF peaks on 1s price."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    times = df_1s['dt'].values
    closes = features['closes']

    fig, axes = plt.subplots(3, 1, figsize=(48, 20), sharex=True,
                             gridspec_kw={'height_ratios': [4, 1.5, 1]})

    # Price with multi-TF peaks
    ax1 = axes[0]
    ax1.plot(times, closes, color='#555', lw=0.3, alpha=0.6)

    # Draw wall levels as horizontal lines
    for wall in walls:
        ax1.axhline(wall['price'], color='orange', lw=1, ls='--', alpha=0.4)

    tf_colors = {'4h': 'purple', '1h': 'blue', '15m': 'cyan', '5m': 'green', '1m': 'yellow'}
    tf_sizes = {'4h': 300, '1h': 200, '15m': 150, '5m': 100, '1m': 60}

    for tf in ['4h', '1h', '15m', '5m', '1m']:
        if tf not in peaks_by_tf:
            continue
        for peak in peaks_by_tf[tf]:
            idx = np.argmin(np.abs(features['timestamps'] - peak['timestamp']))
            d = peak.get('direction', '?')
            marker = '^' if d == 'LONG' else ('v' if d == 'SHORT' else 'D')
            ax1.scatter(times[idx], closes[idx], marker=marker,
                        c=tf_colors.get(tf, 'gray'), s=tf_sizes.get(tf, 80),
                        zorder=10, edgecolors='black', lw=0.5, alpha=0.9)

    ax1.set_ylabel('Price')
    ax1.set_title(f'{date_str} — Multi-TF Peak Overlay (1s price)\n'
                  f'Purple=4h  Blue=1h  Cyan=15m  Green=5m  Yellow=1m  '
                  f'Orange lines=walls', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Velocity
    ax2 = axes[1]
    vel = features['velocity']
    ax2.plot(times, vel, color='steelblue', lw=0.3, alpha=0.7)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_ylabel('Velocity (5s)')
    ax2.grid(True, alpha=0.2)

    # Volume
    ax3 = axes[2]
    ax3.bar(times, features['volumes'], width=np.timedelta64(800, 'ms'),
            color='steelblue', alpha=0.4)
    ax3.set_ylabel('Volume')
    ax3.set_xlabel('Time (UTC)')
    ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    out = f'tools/plots/peak_marker_analysis_{date_str}.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f'Chart: {out}')


if __name__ == '__main__':
    main()
