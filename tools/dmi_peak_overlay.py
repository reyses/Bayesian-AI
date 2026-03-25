"""
Overlay existing human-marked peaks on DMI + Volume chart.
Shows what DMI and volume looked like at each marked peak.

Usage:
    python -m tools.dmi_peak_overlay --date 2026-02-05 --data DATA/ATLAS_OOS
    python -m tools.dmi_peak_overlay --date 2026-02-05:2026-02-06 --data DATA/ATLAS_OOS
"""
import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from core.statistical_field_engine import StatisticalFieldEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='YYYY-MM-DD or range')
    parser.add_argument('--tf', default='1m')
    parser.add_argument('--data', default='DATA/ATLAS_OOS')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    # Load bar data
    tf_folder = args.tf
    pattern = os.path.join(args.data, tf_folder, '*.parquet')
    files = sorted(glob.glob(pattern))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    if ':' in args.date:
        d1, d2 = args.date.split(':')
        mask = (df['dt'].dt.strftime('%Y-%m-%d') >= d1) & (df['dt'].dt.strftime('%Y-%m-%d') <= d2)
    else:
        mask = df['dt'].dt.strftime('%Y-%m-%d') == args.date
    df = df[mask].reset_index(drop=True)
    print(f"Bars: {len(df)}")

    # Compute states
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)

    dmi_p = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_plus', 0) for s in states])
    dmi_m = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_minus', 0) for s in states])
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    prices = df['close'].values
    times = df['dt'].values
    ts_arr = df['timestamp'].values

    dmi_diff = dmi_p - dmi_m
    dmi_gap = np.abs(dmi_diff)
    vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values
    price_vel = np.abs(np.diff(prices, prepend=prices[0]))

    # Load human peaks — match by timestamp range of loaded data
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    peak_files = sorted(glob.glob('DATA/regime_seeds/human_peaks_*_1m.json'))
    all_peaks = []
    for f in peak_files:
        with open(f) as fh:
            d = json.load(fh)
        for p in d['peaks']:
            pts = p.get('timestamp', 0)
            if ts_min <= pts <= ts_max:
                all_peaks.append(p)
    print(f"Human peaks loaded: {len(all_peaks)} (from {len(peak_files)} files)")

    # Match peaks to bar indices
    peak_indices = []
    peak_dirs = []
    for p in all_peaks:
        idx = np.argmin(np.abs(ts_arr - p['timestamp']))
        if abs(ts_arr[idx] - p['timestamp']) < 120:
            peak_indices.append(idx)
            peak_dirs.append(p.get('direction', p.get('_direction_hint', 'UNKNOWN')))

    print(f"Matched: {len(peak_indices)}")

    # --- PLOT ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1, 1]})

    # Panel 1: DMI+ and DMI- with price overlay
    ax1.plot(times, dmi_p, color='#00CC00', lw=1.5, label='DMI+')
    ax1.plot(times, dmi_m, color='#CC0000', lw=1.5, label='DMI-')
    ax1.fill_between(times, dmi_p, dmi_m,
                     where=dmi_p > dmi_m, alpha=0.1, color='green')
    ax1.fill_between(times, dmi_p, dmi_m,
                     where=dmi_m > dmi_p, alpha=0.1, color='red')
    ax1.axhline(y=30, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax1b = ax1.twinx()
    ax1b.plot(times, prices, color='#4A9EFF', lw=0.8, alpha=0.4)
    ax1b.set_ylabel('Price', color='#4A9EFF', fontsize=8)

    # Panel 2: Volume
    ax2.bar(times, volumes, width=np.timedelta64(50, 's'),
            color='#CCAA00', alpha=0.5)
    ax2.plot(times, vol_avg, color='white', lw=1, ls='--', label='Avg(30)')
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)

    # Panel 3: |DMI gap| + price velocity
    ax3.plot(times, dmi_gap, color='cyan', lw=1, label='|DMI gap|')
    ax3b = ax3.twinx()
    ax3b.plot(times, price_vel, color='yellow', lw=0.8, alpha=0.5, label='|Price vel|')
    ax3.set_ylabel('|DMI gap|', color='cyan', fontsize=10)
    ax3b.set_ylabel('|Price vel|', color='yellow', fontsize=8)

    # Overlay peaks on ALL panels
    for i, (idx, direction) in enumerate(zip(peak_indices, peak_dirs)):
        color = '#00FF00' if direction == 'LONG' else '#FF4444'
        if direction == 'UNKNOWN':
            color = '#FFAA00'
        # Vertical lines across all panels
        for ax in (ax1, ax2, ax3):
            ax.axvline(times[idx], color=color, lw=0.8, ls='--', alpha=0.6)
        # Diamond on DMI panel
        marker = '^' if direction == 'LONG' else 'v'
        y_mark = dmi_m[idx] if direction == 'LONG' else dmi_p[idx]
        ax1.scatter(times[idx], y_mark, marker=marker, c=color, s=80,
                    zorder=10, edgecolors='black', lw=0.5)

    ax1.set_ylabel('DMI', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax2.grid(True, alpha=0.2)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlabel('Time (UTC)', fontsize=10)
    ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Auto-rescale volume and velocity on zoom
    def on_xlim_change(event_ax):
        xlim = ax1.get_xlim()
        visible = [(i, v) for i, v in enumerate(volumes)
                   if xlim[0] <= matplotlib.dates.date2num(pd.Timestamp(times[i]).to_pydatetime()) <= xlim[1]]
        if visible:
            vis_vols = [v for _, v in visible]
            ax2.set_ylim(0, max(vis_vols) * 1.1 if max(vis_vols) > 0 else 1)
            vis_vel = [price_vel[i] for i, _ in visible]
            if vis_vel and max(vis_vel) > 0:
                ax3b.set_ylim(0, max(vis_vel) * 1.1)
            fig.canvas.draw_idle()

    ax1.callbacks.connect('xlim_changed', on_xlim_change)

    fig.suptitle(f'{args.date} ({args.tf}) — Human peaks on DMI + Volume\n'
                 f'{len(peak_indices)} peaks | Green=LONG Red=SHORT',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if args.save:
        date_label = args.date.replace(':', '_to_')
        out = f'examples/dmi_peak_overlay_{date_label}_{args.tf}.png'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
