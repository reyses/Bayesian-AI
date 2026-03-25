"""
DMI + Volume peak marker — mark exhaustion points on DMI/Volume chart.

Shows DMI+/DMI- and Volume instead of price candlesticks.
Click to mark peaks, see the exhaustion pattern directly.

Usage:
    python -m tools.dmi_peak_marker --date 2026-02-05 --data DATA/ATLAS_OOS
    python -m tools.dmi_peak_marker --date 2026-02-05:2026-02-07 --data DATA/ATLAS_OOS
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


def load_data(data_root, tf, date_str):
    """Load ATLAS data for date range."""
    tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'}
    tf_folder = tf_map.get(tf, tf)
    pattern = os.path.join(data_root, tf_folder, '*.parquet')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files in {pattern}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    if ':' in date_str:
        d1, d2 = date_str.split(':')
        mask = (df['dt'].dt.strftime('%Y-%m-%d') >= d1) & (df['dt'].dt.strftime('%Y-%m-%d') <= d2)
    else:
        mask = df['dt'].dt.strftime('%Y-%m-%d') == date_str
    df = df[mask].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD')
    parser.add_argument('--tf', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--data', default='DATA/ATLAS_OOS', help='ATLAS root')
    args = parser.parse_args()

    df = load_data(args.data, args.tf, args.date)
    if df.empty:
        print(f"No data for {args.date} at {args.tf}")
        return
    print(f"Loaded {len(df)} bars for {args.date} ({args.tf})")

    # Compute SFE states for DMI
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    print(f"Computed {len(states)} states")

    # Extract DMI+, DMI-, volume
    dmi_p = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_plus', 0) for s in states])
    dmi_m = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_minus', 0) for s in states])
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    prices = df['close'].values
    times = df['dt'].values

    # Compute DMI diff and abs gap
    dmi_diff = dmi_p - dmi_m
    dmi_gap = np.abs(dmi_diff)

    # Rolling avg volume for reference line
    vol_window = 30
    vol_avg = pd.Series(volumes).rolling(vol_window, min_periods=1).mean().values

    # --- PLOT: 3 panels ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(48, 24), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1, 1]})

    # Panel 1: DMI+ (green) and DMI- (red) with shaded gap
    ax1.plot(times, dmi_p, color='#00CC00', lw=1.5, label='DMI+')
    ax1.plot(times, dmi_m, color='#CC0000', lw=1.5, label='DMI-')
    # Shade the gap
    ax1.fill_between(times, dmi_p, dmi_m,
                     where=dmi_p > dmi_m, alpha=0.15, color='green', label='Buyers')
    ax1.fill_between(times, dmi_p, dmi_m,
                     where=dmi_m > dmi_p, alpha=0.15, color='red', label='Sellers')
    # Price as thin blue overlay (second Y axis)
    ax1b = ax1.twinx()
    ax1b.plot(times, prices, color='#4A9EFF', lw=0.8, alpha=0.5)
    ax1b.set_ylabel('Price', color='#4A9EFF', fontsize=8)
    ax1b.tick_params(axis='y', colors='#4A9EFF', labelsize=7)
    ax1.set_ylabel('DMI', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.axhline(y=30, color='gray', ls='--', lw=0.5, alpha=0.5)

    # Panel 2: Volume bars with average line
    ax2.bar(times, volumes, width=np.timedelta64(int(50000), 'ms'),
            color='#CCAA00', alpha=0.6)
    ax2.plot(times, vol_avg, color='white', lw=1, ls='--', label=f'Avg({vol_window})')
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Panel 3: DMI gap magnitude + price velocity
    price_vel = np.abs(np.diff(prices, prepend=prices[0]))
    vel_avg = pd.Series(price_vel).rolling(vol_window, min_periods=1).mean().values
    ax3.plot(times, dmi_gap, color='cyan', lw=1, label='|DMI gap|')
    ax3b = ax3.twinx()
    ax3b.plot(times, price_vel, color='yellow', lw=0.8, alpha=0.6, label='|Price vel|')
    ax3b.plot(times, vel_avg, color='orange', lw=1, ls='--', alpha=0.8, label=f'Vel avg({vol_window})')
    ax3.set_ylabel('|DMI gap|', color='cyan', fontsize=10)
    ax3b.set_ylabel('|Price vel|', color='yellow', fontsize=8)
    ax3.legend(loc='upper left', fontsize=8)
    ax3b.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlabel('Time (UTC)', fontsize=10)
    ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    fig.suptitle(f'{args.date} ({args.tf}) — DMI + Volume + Price Velocity\n'
                 f'Click to mark exhaustion peaks | S=flip | D=delete | Q=save+quit',
                 fontsize=14, fontweight='bold')

    # --- Interactive peak marking ---
    peaks = []
    markers = []

    def on_click(event):
        if event.inaxes not in (ax1, ax2, ax3):
            return
        # Find nearest bar by x (time)
        click_x = matplotlib.dates.num2date(event.xdata).replace(tzinfo=None)
        import datetime
        dts = [pd.Timestamp(t).to_pydatetime().replace(tzinfo=None) for t in times]
        dists = [abs((dt - click_x).total_seconds()) for dt in dts]
        idx = int(np.argmin(dists))

        # Auto-detect direction from DMI at click point
        direction = 'LONG' if dmi_diff[idx] < 0 else 'SHORT'

        peak = {
            'bar_index': int(idx),
            'timestamp': float(df.iloc[idx]['timestamp']),
            'time_utc': str(df.iloc[idx]['dt']),
            'price': float(prices[idx]),
            'direction': direction,
            'dmi_plus': float(dmi_p[idx]),
            'dmi_minus': float(dmi_m[idx]),
            'dmi_gap': float(dmi_gap[idx]),
            'volume': float(volumes[idx]),
        }
        peaks.append(peak)

        # Draw marker on all panels
        color = '#00FF00' if direction == 'LONG' else '#FF4444'
        marker = '^' if direction == 'LONG' else 'v'
        m1 = ax1.axvline(times[idx], color=color, lw=1, ls='--', alpha=0.7)
        m2 = ax2.axvline(times[idx], color=color, lw=1, ls='--', alpha=0.7)
        m3 = ax3.axvline(times[idx], color=color, lw=1, ls='--', alpha=0.7)
        m4 = ax1.scatter(times[idx], dmi_p[idx] if direction == 'LONG' else dmi_m[idx],
                         marker=marker, c=color, s=100, zorder=10)
        label = ax1.annotate(f'#{len(peaks)} {direction}',
                             (times[idx], max(dmi_p[idx], dmi_m[idx]) + 2),
                             fontsize=7, color=color, fontweight='bold')
        markers.append((m1, m2, m3, m4, label))

        print(f"  Peak #{len(peaks)}: {peak['time_utc']} {direction} "
              f"DMI+={dmi_p[idx]:.1f} DMI-={dmi_m[idx]:.1f} gap={dmi_gap[idx]:.1f} vol={volumes[idx]:.0f}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 's' and peaks:
            # Flip last peak direction
            old = peaks[-1]['direction']
            peaks[-1]['direction'] = 'SHORT' if old == 'LONG' else 'LONG'
            print(f"  Flipped peak #{len(peaks)} -> {peaks[-1]['direction']}")
            # Redraw would be complex, just note it
        elif event.key == 'd' and peaks:
            p = peaks.pop()
            if markers:
                for m in markers.pop():
                    try:
                        m.remove()
                    except Exception:
                        pass
            print(f"  Deleted peak at {p['time_utc']}")
            fig.canvas.draw_idle()
        elif event.key == 'q':
            # Save
            date_label = args.date.replace(':', '_to_')
            out_path = f'DATA/regime_seeds/human_dmi_peaks_{date_label}_{args.tf}.json'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out = {
                'date': args.date,
                'tf': args.tf,
                'n_peaks': len(peaks),
                'peaks': peaks,
            }
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"\n  Saved {len(peaks)} DMI peaks to {out_path}")

            # Save screenshot
            ss_path = f'examples/dmi_peak_marker_{date_label}_{args.tf}.png'
            os.makedirs(os.path.dirname(ss_path), exist_ok=True)
            fig.savefig(ss_path, dpi=150, bbox_inches='tight')
            print(f"  Screenshot: {ss_path}")
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
