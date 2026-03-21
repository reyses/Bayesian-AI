"""
Full-day I-chart with human seed trades overlaid on 1m price.

Layout (3 panels, shared x-axis):
  Top:    Price (1m close) with seed trade spans (green=LONG, red=SHORT)
          Entry arrows, MFE markers, 10-bar lookback shading
  Middle: DMI (DI+ green, DI- red, ADX gray dashed)
  Bottom: Volume bars

Usage:
    python tools/seed_daily_chart.py [--date 2025-01-02] [--seed-file PATH]

Output:
    reports/findings/seed_daily_chart_YYYY-MM-DD.png
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine


def load_1m_data(data_root, year_month='2025_01'):
    """Load 1m OHLCV + compute MarketState for DMI."""
    tf_dir = os.path.join(data_root, '1m')
    fn = os.path.join(tf_dir, f'{year_month}.parquet')
    if not os.path.exists(fn):
        raise FileNotFoundError(f"No 1m parquet: {fn}")

    print(f"  Loading {fn}...")
    df = pd.read_parquet(fn)
    if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())

    print("  Computing MarketState (DMI, volume, etc.)...")
    engine = StatisticalFieldEngine()
    raw_states = engine.batch_compute_states(df)
    states = []
    for r in raw_states:
        states.append(r['state'] if isinstance(r, dict) and 'state' in r else r)

    return df, states


def filter_day(df, states, target_date):
    """Filter to a single trading day."""
    timestamps = df['timestamp'].values.astype(np.float64)

    # Convert target_date to timestamp range
    dt_start = datetime.strptime(target_date, '%Y-%m-%d').replace(
        hour=0, minute=0, second=0, tzinfo=timezone.utc)
    dt_end = dt_start.replace(hour=23, minute=59, second=59)
    ts_start = dt_start.timestamp()
    ts_end = dt_end.timestamp()

    mask = (timestamps >= ts_start) & (timestamps <= ts_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No bars for {target_date}")

    df_day = df.iloc[idx].reset_index(drop=True)
    states_day = [states[i] for i in idx]
    print(f"  {target_date}: {len(df_day)} bars "
          f"({df_day['timestamp'].iloc[0]:.0f} -> {df_day['timestamp'].iloc[-1]:.0f})")
    return df_day, states_day


def load_seeds(seed_file):
    """Load seeds from JSON."""
    with open(seed_file) as f:
        data = json.load(f)

    seeds = data.get('seeds', [])
    tf = data.get('timeframe', '1m')
    if isinstance(tf, list):
        tf = tf[0]
    tf_sec = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}.get(tf, 60)

    entries = []
    for s in seeds:
        lb = s.get('lookback_bars', 10)
        entries.append({
            'ts_lookback_start': s['ts_start'],
            'ts_entry': s['ts_start'] + lb * tf_sec,
            'ts_exit': s['ts_end'],
            'direction': s['direction'].upper(),
            'entry_price': s.get('entry_price', 0),
            'exit_price': s.get('exit_price', 0),
            'mfe_ticks': s.get('mfe_ticks', 0),
            'mae_ticks': s.get('mae_ticks', 0),
            'change_ticks': s.get('change_ticks', 0),
            'n_bars': s.get('n_bars', 0),
            'lookback_bars': lb,
            'time_to_mfe_mins': s.get('time_to_mfe_mins', 0),
        })
    print(f"  {len(entries)} seeds from {os.path.basename(seed_file)}")
    return entries


def plot_daily(df_day, states_day, seeds, target_date, output_path):
    """Plot full-day chart: price + seeds, DMI, volume."""
    timestamps = df_day['timestamp'].values.astype(np.float64)
    closes = df_day['close'].values
    highs = df_day['high'].values
    lows = df_day['low'].values
    volumes = df_day['volume'].values

    # Extract DMI from states
    dmi_plus = np.array([getattr(s, 'dmi_plus', 0) or 0 for s in states_day])
    dmi_minus = np.array([getattr(s, 'dmi_minus', 0) or 0 for s in states_day])
    adx = np.array([getattr(s, 'adx_strength', 0) or 0 for s in states_day])

    # Convert timestamps to bar indices for x-axis
    n_bars = len(timestamps)
    bar_idx = np.arange(n_bars)

    # Create time labels (HH:MM) for x-axis
    time_labels = []
    tick_positions = []
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.minute == 0:  # label every hour
            time_labels.append(dt.strftime('%H:%M'))
            tick_positions.append(i)

    # --- Figure ---
    fig = plt.figure(figsize=(53, 30))  # ~8K at 200 dpi
    gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 0.4, 1.5], hspace=0.06)

    # Helper: find bar index for a timestamp
    def ts_to_bar(ts):
        idx = int(np.searchsorted(timestamps, ts, side='right')) - 1
        return max(0, min(idx, n_bars - 1))

    # Pre-compute seed bar positions
    long_count = 0
    short_count = 0
    seed_bars = []
    for seed in seeds:
        entry_bar = ts_to_bar(seed['ts_entry'])
        exit_bar = ts_to_bar(seed['ts_exit'])
        is_long = seed['direction'] == 'LONG'
        if is_long:
            long_count += 1
        else:
            short_count += 1
        seed_bars.append((entry_bar, exit_bar, is_long))

    # Build trade density array (how many trades active per bar)
    long_density = np.zeros(n_bars)
    short_density = np.zeros(n_bars)
    for entry_bar, exit_bar, is_long in seed_bars:
        if is_long:
            long_density[entry_bar:exit_bar + 1] += 1
        else:
            short_density[entry_bar:exit_bar + 1] += 1

    # ========================
    # Panel 1: Price + Seeds
    # ========================
    ax1 = fig.add_subplot(gs[0])

    # Price line (clean, no high/low fill — too noisy with seeds)
    ax1.plot(bar_idx, closes, color='#1a1a2e', linewidth=1.0, alpha=0.9, zorder=2)

    # Seed entries/exits as thin vertical lines + small markers
    for entry_bar, exit_bar, is_long in seed_bars:
        color = '#2ecc71' if is_long else '#e74c3c'

        # Thin entry line
        ax1.axvline(entry_bar, color=color, linewidth=0.3, alpha=0.4, zorder=1)

        # Small entry marker on price
        entry_price = closes[entry_bar] if entry_bar < n_bars else 0
        marker = '^' if is_long else 'v'
        ax1.scatter(entry_bar, entry_price, marker=marker, color=color,
                    s=18, zorder=5, edgecolors='none', alpha=0.7)

    ax1.set_title(f'MNQ 1m — {target_date} — {len(seeds)} Human Seeds '
                  f'({long_count}L / {short_count}S)\n'
                  f'Triangles = entries (green ^LONG, red vSHORT)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(0, n_bars - 1)

    # ========================
    # Panel 2: Trade density strip
    # ========================
    ax_dens = fig.add_subplot(gs[1], sharex=ax1)

    ax_dens.bar(bar_idx, long_density, width=1.0, color='#2ecc71', alpha=0.7,
                label='LONG active')
    ax_dens.bar(bar_idx, -short_density, width=1.0, color='#e74c3c', alpha=0.7,
                label='SHORT active')
    ax_dens.axhline(0, color='gray', linewidth=0.5)
    ax_dens.set_ylabel('Trades', fontsize=9)
    ax_dens.set_yticks([])
    ax_dens.set_xticks(tick_positions)
    ax_dens.set_xticklabels([])
    max_dens = max(long_density.max(), short_density.max(), 1)
    ax_dens.set_ylim(-max_dens * 1.1, max_dens * 1.1)
    ax_dens.legend(loc='upper right', fontsize=7, ncol=2)

    # ========================
    # Panel 3: DMI + Volume overlay
    # ========================
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    # Volume as background bars on secondary y-axis
    vol_cap = np.percentile(volumes, 95)
    vol_clipped = np.minimum(volumes, vol_cap)
    ax2v = ax2.twinx()
    ax2v.bar(bar_idx, vol_clipped, width=0.8, color='#b0bec5', alpha=0.25,
             edgecolor='none', zorder=1, label='Volume')
    vol_ma = np.minimum(
        pd.Series(volumes).rolling(20, min_periods=1).mean().values, vol_cap)
    ax2v.plot(bar_idx, vol_ma, color='#78909c', linewidth=1.0, alpha=0.5,
              zorder=2, label='Vol MA(20)')
    ax2v.set_ylim(0, vol_cap * 3)  # compress volume to bottom third
    ax2v.set_ylabel('Volume', fontsize=11, color='#78909c')
    ax2v.tick_params(axis='y', labelcolor='#78909c')

    # DMI lines on top
    ax2.plot(bar_idx, dmi_plus, color='#2ecc71', linewidth=1.5, label='DI+', zorder=4)
    ax2.plot(bar_idx, dmi_minus, color='#e74c3c', linewidth=1.5, label='DI−', zorder=4)
    ax2.plot(bar_idx, adx, color='#7f8c8d', linewidth=1.0, linestyle='--',
             alpha=0.7, label='ADX', zorder=3)

    # Fill between DI+ and DI- to show dominance
    ax2.fill_between(bar_idx, dmi_plus, dmi_minus,
                     where=(dmi_plus > dmi_minus),
                     alpha=0.12, color='#2ecc71', interpolate=True, zorder=3)
    ax2.fill_between(bar_idx, dmi_plus, dmi_minus,
                     where=(dmi_plus < dmi_minus),
                     alpha=0.12, color='#e74c3c', interpolate=True, zorder=3)

    ax2.set_ylabel('DMI', fontsize=11)
    ax2.set_xlabel('Time (UTC)', fontsize=11)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(time_labels, rotation=45, fontsize=9)
    ax2.grid(True, alpha=0.15)
    ax2.set_zorder(ax2v.get_zorder() + 1)
    ax2.patch.set_visible(False)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2v.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=9, ncol=5)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Full-day seed chart')
    parser.add_argument('--date', default='2025-01-02',
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--seed-file', default=None,
                        help='Seed JSON file (auto-detected from date if omitted)')
    parser.add_argument('--data-root', default='DATA/ATLAS',
                        help='ATLAS data root')
    args = parser.parse_args()

    target_date = args.date
    year_month = target_date[:4] + '_' + target_date[5:7]

    # Auto-detect seed file
    if args.seed_file:
        seed_file = args.seed_file
    else:
        seed_dir = 'DATA/regime_seeds'
        candidates = [f for f in os.listdir(seed_dir)
                      if f.startswith(f'seeds_{target_date}') and f.endswith('.json')]
        if not candidates:
            print(f"No seed file found for {target_date} in {seed_dir}")
            sys.exit(1)
        seed_file = os.path.join(seed_dir, sorted(candidates)[0])
        print(f"  Auto-detected: {seed_file}")

    # Load data
    df, states = load_1m_data(args.data_root, year_month)
    df_day, states_day = filter_day(df, states, target_date)
    seeds = load_seeds(seed_file)

    # Plot
    output_path = f'reports/findings/seed_daily_chart_{target_date}.png'
    plot_daily(df_day, states_day, seeds, target_date, output_path)


if __name__ == '__main__':
    main()
