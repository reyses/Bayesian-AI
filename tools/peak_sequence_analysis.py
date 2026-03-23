"""Peak Sequence Analysis — what ACTUALLY happens bar-by-bar around marked peaks?

No theory. Just measure the 1s data -30 to +30 bars around each human-marked
peak and report what velocity, volume, std, and dmi did at each bar offset.

Outputs:
  - Average feature value at each offset (-30 to +30) across all peaks
  - Heatmap showing the temporal signature

Usage:
    python tools/peak_sequence_analysis.py --date 2026-02-05
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
SEEDS_DIR = 'DATA/regime_seeds'
WINDOW = 30  # bars before and after peak


def load_human_peaks(date_str):
    """Load all human peak files for a date."""
    peaks_all = []
    for f in glob.glob(os.path.join(SEEDS_DIR, 'human_peaks_*.json')):
        with open(f) as fh:
            data = json.load(fh)
        fname = os.path.basename(f)
        if date_str in fname or 'to' in fname:
            tf = data.get('tf', '?')
            for p in data['peaks']:
                peaks_all.append({**p, 'source_tf': tf})
    return peaks_all


def load_1s_data(date_str):
    """Load 1s data for one day."""
    month = date_str[:4] + '_' + date_str[5:7]
    for data_dir in ['DATA/ATLAS_OOS', 'DATA/ATLAS']:
        files = sorted(glob.glob(os.path.join(data_dir, '1s', f'{month}*.parquet')))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df = df[df['dt'].dt.strftime('%Y-%m-%d') == date_str].reset_index(drop=True)
            if len(df) > 0:
                return df
    return None


def compute_features(df):
    """Compute grounded features on 1s data."""
    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    highs = df['high'].values
    lows = df['low'].values
    n = len(closes)

    dp = np.diff(closes) / TICK
    dp = np.concatenate([[0], dp])

    # Velocity (5-bar mean of price changes)
    velocity = np.full(n, 0.0)
    for i in range(5, n):
        velocity[i] = np.mean(dp[i-5:i])

    # Volume (raw)
    vol = volumes.copy()

    # Volume delta (signed — positive = uptick volume)
    # Proxy: volume * sign(price change)
    vol_delta = vol * np.sign(dp)

    # Rolling volume average (60s)
    vol_avg = np.full(n, 1.0)
    for i in range(60, n):
        vol_avg[i] = np.mean(vol[i-60:i])

    # Volume ratio
    vol_ratio = np.where(vol_avg > 0, vol / vol_avg, 1.0)

    # Std of price changes (20s)
    std_price = np.full(n, 0.0)
    for i in range(20, n):
        std_price[i] = np.std(dp[i-20:i], ddof=1)

    # Acceleration
    accel = np.diff(velocity)
    accel = np.concatenate([[0], accel])

    # Bar range
    bar_range = (highs - lows) / TICK

    # Wick ratio (how much of bar is wick vs body)
    if 'open' in df.columns:
        opens = df['open'].values
        body = np.abs(closes - opens) / TICK
        total_range = bar_range
        wick_ratio = np.where(total_range > 0, 1.0 - body / total_range, 0.0)
    else:
        wick_ratio = np.zeros(n)

    return {
        'dp': dp,
        'velocity': velocity,
        'acceleration': accel,
        'volume': vol,
        'vol_delta': vol_delta,
        'vol_ratio': vol_ratio,
        'std_price': std_price,
        'bar_range': bar_range,
        'wick_ratio': wick_ratio,
        'timestamps': df['timestamp'].values,
    }


def extract_windows(features, peaks, window=WINDOW):
    """Extract feature windows around each peak."""
    timestamps = features['timestamps']
    n = len(timestamps)

    feature_names = ['dp', 'velocity', 'acceleration', 'volume',
                     'vol_delta', 'vol_ratio', 'std_price', 'bar_range', 'wick_ratio']

    # Collect windows: shape (n_peaks, 2*window+1) per feature
    windows = {f: [] for f in feature_names}
    valid_peaks = 0

    for peak in peaks:
        idx = int(np.argmin(np.abs(timestamps - peak['timestamp'])))

        if idx < window or idx >= n - window:
            continue

        valid_peaks += 1
        for fname in feature_names:
            arr = features[fname]
            windows[fname].append(arr[idx - window:idx + window + 1])

    # Convert to arrays and compute mean/std across peaks
    results = {}
    for fname in feature_names:
        if windows[fname]:
            stacked = np.array(windows[fname])
            results[fname] = {
                'mean': np.mean(stacked, axis=0),
                'std': np.std(stacked, axis=0),
                'median': np.median(stacked, axis=0),
                'n': len(windows[fname]),
            }

    return results, valid_peaks


def main():
    parser = argparse.ArgumentParser(description='Peak sequence analysis')
    parser.add_argument('--date', default='2026-02-05')
    args = parser.parse_args()

    print(f'Loading peaks for {args.date}...')
    peaks = load_human_peaks(args.date)
    print(f'  {len(peaks)} peaks from all TFs')

    print('Loading 1s data...')
    df = load_1s_data(args.date)
    if df is None:
        print('No 1s data found')
        sys.exit(1)
    print(f'  {len(df):,} bars')

    print('Computing features...')
    features = compute_features(df)

    print('Extracting windows...')
    results, valid = extract_windows(features, peaks)
    print(f'  {valid} valid peaks (with full window)')

    # Format output
    offsets = np.arange(-WINDOW, WINDOW + 1)
    lines = []
    lines.append('=' * 90)
    lines.append(f'PEAK SEQUENCE ANALYSIS — {args.date}')
    lines.append(f'Peaks: {valid} (from {len(peaks)} marked across all TFs)')
    lines.append(f'Window: {WINDOW} bars before/after peak (1s resolution)')
    lines.append(f'Bar 0 = peak moment')
    lines.append('=' * 90)

    for fname in ['velocity', 'acceleration', 'volume', 'vol_delta',
                   'vol_ratio', 'std_price', 'bar_range', 'wick_ratio', 'dp']:
        if fname not in results:
            continue
        r = results[fname]
        lines.append(f'\n--- {fname.upper()} (n={r["n"]}) ---')
        lines.append(f'{"offset":>7}  {"mean":>10}  {"median":>10}  {"std":>10}')

        # Show every bar for compact features, every 5th for verbose
        step = 1 if WINDOW <= 30 else 5
        for i in range(0, len(offsets), step):
            off = offsets[i]
            m = r['mean'][i]
            med = r['median'][i]
            s = r['std'][i]
            marker = '  <-- PEAK' if off == 0 else ''
            lines.append(f'{off:>+7}  {m:>+10.3f}  {med:>+10.3f}  {s:>10.3f}{marker}')

    # Summary: what changes at the peak?
    lines.append(f'\n{"="*90}')
    lines.append('SUMMARY: BEFORE vs AT vs AFTER peak (mean values)')
    lines.append(f'{"="*90}')
    lines.append(f'{"Feature":<15} {"Before(-30:-10)":>15} {"Approach(-10:-1)":>17} '
                 f'{"AT PEAK(0)":>12} {"After(+1:+10)":>15} {"After(+10:+30)":>16}')
    lines.append('-' * 90)

    for fname in ['velocity', 'volume', 'vol_delta', 'vol_ratio',
                   'std_price', 'bar_range', 'dp']:
        if fname not in results:
            continue
        r = results[fname]['mean']
        before = np.mean(r[:WINDOW-10])         # -30 to -10
        approach = np.mean(r[WINDOW-10:WINDOW])  # -10 to -1
        at_peak = r[WINDOW]                       # 0
        after_early = np.mean(r[WINDOW+1:WINDOW+11])  # +1 to +10
        after_late = np.mean(r[WINDOW+11:])       # +10 to +30

        lines.append(f'{fname:<15} {before:>+15.3f} {approach:>+17.3f} '
                     f'{at_peak:>+12.3f} {after_early:>+15.3f} {after_late:>+16.3f}')

    summary = '\n'.join(lines)

    out_path = os.path.join('reports', 'findings', f'peak_sequence_{args.date}.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nSummary: {out_path}')
    print('\n' + summary[-2000:])  # print last 2K chars

    # Plot
    _plot(results, offsets, args.date, valid)


def _plot(results, offsets, date_str, n_peaks):
    """Plot feature evolution around peaks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    features_to_plot = ['velocity', 'volume', 'vol_delta', 'vol_ratio',
                         'std_price', 'bar_range', 'dp', 'acceleration']
    features_to_plot = [f for f in features_to_plot if f in results]

    n_plots = len(features_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots), sharex=True)
    fig.suptitle(f'Peak Sequence — {date_str} ({n_peaks} peaks)\n'
                 f'Bar 0 = peak moment. Shaded = ±1 std.',
                 fontsize=14, fontweight='bold')

    for i, fname in enumerate(features_to_plot):
        ax = axes[i] if n_plots > 1 else axes
        r = results[fname]
        mean = r['mean']
        std = r['std']

        ax.plot(offsets, mean, color='steelblue', lw=2)
        ax.fill_between(offsets, mean - std, mean + std, alpha=0.2, color='steelblue')
        ax.axvline(0, color='red', lw=2, ls='--', alpha=0.7)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel(fname, fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Offset from peak (1s bars)')
    plt.tight_layout()
    out = f'tools/plots/peak_sequence_{date_str}.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f'Plot: {out}')


if __name__ == '__main__':
    main()
