"""
Zone Analysis: measure oscillation, variance, and daily ranges within level zones.

For each month's levels, price lives in zones between adjacent levels.
This tool measures what happens inside each zone:
  - Time spent in zone (% of bars)
  - Daily range within zone (ticks)
  - Oscillation period (bars between reversals at zone boundaries)
  - Variance (how noisy is the oscillation)
  - Touch rate (how often does price reach the zone boundaries)

Usage:
  python -m tools.zone_analysis
  python -m tools.zone_analysis --month 2025-06
"""
import argparse
import gc
import glob
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.directory'] = os.path.abspath('examples')
import matplotlib.pyplot as plt

ATLAS = 'DATA/ATLAS'
TICK = 0.25


def load_levels(date_str):
    """Load levels for a given month."""
    path = os.path.join('DATA/levels', f'levels_{date_str}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)['levels']


def analyze_zone(df, level_low, level_high, tf='1m'):
    """Analyze price behavior within a zone between two levels."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n = len(closes)
    if n < 10:
        return None

    zone_range = level_high - level_low

    # Bars in zone (close between levels)
    in_zone = (closes >= level_low) & (closes <= level_high)
    time_in_zone = in_zone.mean()

    # Bars touching boundaries
    touch_high = np.sum(highs >= level_high - zone_range * 0.05)
    touch_low = np.sum(lows <= level_low + zone_range * 0.05)

    # Daily ranges within zone
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['timestamp'], unit='s').dt.date
    daily_ranges = []
    for date, grp in df_copy.groupby('date'):
        day_in = grp[(grp['close'] >= level_low) & (grp['close'] <= level_high)]
        if len(day_in) > 5:
            daily_ranges.append((day_in['high'].max() - day_in['low'].min()) / TICK)

    # Oscillation within zone: count reversals at boundaries
    # A reversal = price was heading toward one boundary and turned back
    reversals = 0
    direction = 0  # 1=heading up, -1=heading down
    for i in range(1, n):
        if not in_zone[i]:
            continue
        if closes[i] > closes[i-1]:
            new_dir = 1
        elif closes[i] < closes[i-1]:
            new_dir = -1
        else:
            continue
        if direction != 0 and new_dir != direction:
            reversals += 1
        direction = new_dir

    bars_in_zone = in_zone.sum()
    if bars_in_zone > 0 and reversals > 0:
        avg_half_cycle = bars_in_zone / reversals
    else:
        avg_half_cycle = 0

    # Variance of closes within zone
    zone_closes = closes[in_zone]
    variance = zone_closes.std() / TICK if len(zone_closes) > 1 else 0

    # Bar body sizes within zone (small bars = consolidation)
    zone_bodies = np.abs(closes[in_zone] - opens[in_zone]) / TICK
    avg_body = zone_bodies.mean() if len(zone_bodies) > 0 else 0

    return {
        'level_low': level_low,
        'level_high': level_high,
        'zone_range_ticks': zone_range / TICK,
        'time_in_zone_pct': time_in_zone * 100,
        'touch_high': int(touch_high),
        'touch_low': int(touch_low),
        'n_reversals': reversals,
        'avg_half_cycle': avg_half_cycle,
        'daily_range_mean': np.mean(daily_ranges) if daily_ranges else 0,
        'daily_range_median': np.median(daily_ranges) if daily_ranges else 0,
        'variance_ticks': variance,
        'avg_body_ticks': avg_body,
        'bars_in_zone': int(bars_in_zone),
        'n_daily_ranges': len(daily_ranges),
    }


def analyze_month(month_str):
    """Analyze all zones for a given month."""
    # Find the level file
    level_files = sorted(glob.glob('DATA/levels/levels_*.json'))
    levels = None
    for f in level_files:
        with open(f) as fh:
            d = json.load(fh)
        # Match month
        if d['date'][:7] == month_str or d['date'] <= f'{month_str}-31':
            if d['date'][:7] <= month_str:
                levels = d['levels']  # use latest levels at or before this month

    if levels is None or len(levels) < 2:
        return None, None

    prices = sorted([l['price'] for l in levels])

    # Load 1h data for the month
    files = sorted(glob.glob(os.path.join(ATLAS, '1h', '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    month_start = pd.Timestamp(f'{month_str}-01').timestamp()
    if int(month_str[5:7]) == 12:
        month_end = pd.Timestamp(f'{int(month_str[:4])+1}-01-01').timestamp()
    else:
        month_end = pd.Timestamp(f'{month_str[:4]}-{int(month_str[5:7])+1:02d}-01').timestamp()

    df_month = df[(df['timestamp'] >= month_start) & (df['timestamp'] < month_end)].reset_index(drop=True)

    if len(df_month) < 20:
        return None, None

    # Analyze each zone (between adjacent levels)
    zones = []
    for i in range(len(prices) - 1):
        result = analyze_zone(df_month, prices[i], prices[i+1])
        if result:
            zones.append(result)

    return zones, prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', default=None, help='Single month (YYYY-MM) or all')
    args = parser.parse_args()

    if args.month:
        months = [args.month]
    else:
        # All months with level data
        months = sorted(set(f'{y}-{m:02d}'
                            for y in range(2025, 2027)
                            for m in range(1, 13)
                            if glob.glob(f'DATA/levels/levels_{y}-*')))[:18]

    all_results = []

    for month in tqdm(months, desc="Months"):
        zones, prices = analyze_month(month)
        if zones is None:
            continue

        print(f"\n{'='*70}")
        print(f"{month} — {len(prices)} levels, {len(zones)} zones")
        print(f"{'='*70}")
        print(f"{'Zone':>20} {'Range(t)':>8} {'Time%':>6} {'HalfCyc':>8} {'DailyRng':>9} {'Var(t)':>7} {'Body':>6} {'TouchH':>6} {'TouchL':>6}")
        print('-' * 90)

        for z in zones:
            zone_label = f"{z['level_low']:.0f}-{z['level_high']:.0f}"
            print(f"{zone_label:>20} {z['zone_range_ticks']:>7.0f} {z['time_in_zone_pct']:>5.1f}% "
                  f"{z['avg_half_cycle']:>7.1f} {z['daily_range_mean']:>8.0f} "
                  f"{z['variance_ticks']:>6.0f} {z['avg_body_ticks']:>5.1f} "
                  f"{z['touch_high']:>6} {z['touch_low']:>6}")

            z['month'] = month
            all_results.append(z)

    if not all_results:
        print("No results")
        return

    # Save CSV
    df_results = pd.DataFrame(all_results)
    csv_path = 'reports/findings/zone_analysis.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY ({len(all_results)} zones across {len(months)} months)")
    print(f"{'='*70}")
    print(f"  Avg zone range:     {df_results['zone_range_ticks'].mean():.0f} ticks")
    print(f"  Avg time in zone:   {df_results['time_in_zone_pct'].mean():.1f}%")
    print(f"  Avg half-cycle:     {df_results['avg_half_cycle'].mean():.1f} bars")
    print(f"  Avg daily range:    {df_results['daily_range_mean'].mean():.0f} ticks")
    print(f"  Avg variance:       {df_results['variance_ticks'].mean():.0f} ticks")
    print(f"  Avg bar body:       {df_results['avg_body_ticks'].mean():.1f} ticks")

    # Chart: zone characteristics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Zone Analysis — Oscillation Within Level Zones', fontsize=14, fontweight='bold')

    # 1. Zone range vs daily range
    ax = axes[0, 0]
    ax.scatter(df_results['zone_range_ticks'], df_results['daily_range_mean'], s=20, alpha=0.5)
    ax.set_xlabel('Zone Range (ticks)')
    ax.set_ylabel('Avg Daily Range (ticks)')
    ax.set_title('Zone Size vs Daily Range')
    ax.grid(True, alpha=0.2)

    # 2. Half-cycle distribution
    ax = axes[0, 1]
    valid = df_results[df_results['avg_half_cycle'] > 0]['avg_half_cycle']
    ax.hist(valid, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=valid.median(), color='red', linestyle='--', label=f'median={valid.median():.0f}')
    ax.set_xlabel('Half-Cycle (bars)')
    ax.set_ylabel('Count')
    ax.set_title('Oscillation Half-Cycle Distribution')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 3. Time in zone vs zone range
    ax = axes[1, 0]
    ax.scatter(df_results['zone_range_ticks'], df_results['time_in_zone_pct'], s=20, alpha=0.5)
    ax.set_xlabel('Zone Range (ticks)')
    ax.set_ylabel('Time in Zone (%)')
    ax.set_title('Zone Size vs Time Spent')
    ax.grid(True, alpha=0.2)

    # 4. Variance vs zone range
    ax = axes[1, 1]
    ax.scatter(df_results['zone_range_ticks'], df_results['variance_ticks'], s=20, alpha=0.5)
    ax.set_xlabel('Zone Range (ticks)')
    ax.set_ylabel('Variance (ticks)')
    ax.set_title('Zone Size vs Variance')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    chart_path = 'examples/zone_analysis.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Chart: {chart_path}")


if __name__ == '__main__':
    main()
