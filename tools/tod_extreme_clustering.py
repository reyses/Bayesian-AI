"""Time-of-day clustering analysis: do model breakdowns cluster around the same UTC hour?

A 'breakdown' = an event whose actual max_z exceeds the cell's q90 prediction
significantly, or simply an event in the catastrophic tail (|max_z| >= 8).

Tests user hypothesis (2026-05-10): catastrophic outliers happen around the
same time of day (e.g. US afternoon session, post-lunch reversals).

USAGE
    python tools/tod_extreme_clustering.py
    python tools/tod_extreme_clustering.py --threshold 6.0  --split IS
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro-csv',
                     default='reports/findings/band_touch_aggregation/macro_events_1h_hl.csv')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/tod_extreme_clustering')
    ap.add_argument('--threshold', type=float, default=6.0,
                     help='|max_z| threshold for "extreme" (default 6.0)')
    ap.add_argument('--split', default='all',
                     help='IS, OOS, or all')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.macro_csv)
    if args.split != 'all':
        df = df[df['split'] == args.split]
    print(f'Loaded {len(df):,} macro events at 1h HL k>=3sigma  (split={args.split})')
    print(f'TOD hour column: tod_hour (UTC)')
    print(f'Mean max_abs_z: {df["max_abs_z"].mean():.2f}')
    print(f'q90 max_abs_z:  {df["max_abs_z"].quantile(0.90):.2f}')
    print(f'q95 max_abs_z:  {df["max_abs_z"].quantile(0.95):.2f}')
    print(f'q99 max_abs_z:  {df["max_abs_z"].quantile(0.99):.2f}')
    print(f'max max_abs_z:  {df["max_abs_z"].max():.2f}')

    # Mark extremes
    df['is_extreme'] = df['max_abs_z'] >= args.threshold
    n_extreme = int(df['is_extreme'].sum())
    print(f'\nExtreme events (|max_z| >= {args.threshold}): {n_extreme} ({100*n_extreme/len(df):.1f}%)')

    # TOD distribution: ALL events vs EXTREMES
    print(f'\n=== TOD HISTOGRAM (UTC hour) ===')
    by_hour_all = df.groupby('tod_hour').size()
    by_hour_extreme = df[df['is_extreme']].groupby('tod_hour').size()
    tab = pd.DataFrame({
        'all_events': by_hour_all,
        'extreme': by_hour_extreme,
    }).fillna(0).astype(int)
    tab['extreme_rate_pct'] = (100 * tab['extreme'] / tab['all_events']).round(1)
    tab['pct_of_all'] = (100 * tab['all_events'] / tab['all_events'].sum()).round(1)
    tab['pct_of_extremes'] = (100 * tab['extreme'] / tab['extreme'].sum()).round(1)
    print(tab.to_string())

    tab.to_csv(os.path.join(args.out_dir, f'tod_dist_t{args.threshold}.csv'))

    # Statistical test: is the TOD distribution of extremes uniform vs all-events distribution?
    # Use a chi-squared-like comparison (expected = all_events * total_extremes / total_events)
    expected = tab['all_events'] * tab['extreme'].sum() / tab['all_events'].sum()
    observed = tab['extreme']
    valid = expected >= 5  # chi-squared validity
    chi2 = float(((observed[valid] - expected[valid])**2 / expected[valid]).sum())
    dof = int(valid.sum() - 1)
    print(f'\nChi-squared TOD dependence test:')
    print(f'  chi2 = {chi2:.2f}  dof = {dof}')
    print(f'  Critical at p=0.05: 23.7 (df=14)  / p=0.01: 29.1')
    print(f'  -> {"DEPENDENT (significant)" if chi2 > 23.7 else "consistent with uniform"}')

    # Render bar chart: all events + extremes overlay + extreme-rate line
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    ax1, ax2 = axes
    hours = tab.index.values

    ax1.bar(hours, tab['all_events'], color='#1E88E5', alpha=0.6,
              label=f'all events (n={tab["all_events"].sum():,})')
    ax1.bar(hours, tab['extreme'], color='#E53935', alpha=0.85,
              label=f'extreme |max_z|>={args.threshold} (n={tab["extreme"].sum():,})')
    ax1.set_ylabel('event count')
    ax1.set_title(f'Macro event TOD distribution (UTC hour)\n'
                   f'Population: {len(df):,} events at 1h HL k>=3sigma  '
                   f'(split={args.split})  threshold |max_z|>={args.threshold}',
                   fontsize=11)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Extreme RATE per hour (extremes / all events in that hour)
    ax2.bar(hours, tab['extreme_rate_pct'], color='#FB8C00', alpha=0.85)
    ax2.axhline(100 * tab['extreme'].sum() / tab['all_events'].sum(),
                  color='black', ls='--', lw=0.8,
                  label=f'baseline rate {100*tab["extreme"].sum()/tab["all_events"].sum():.1f}%')
    ax2.set_xlabel('UTC hour'); ax2.set_ylabel('extreme rate (%)')
    ax2.set_title(f'P(event is extreme | hour) — chi2 = {chi2:.1f}, dof={dof}', fontsize=10)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(24))

    # Annotate ET equivalent on x-axis as second xlabel
    fig.text(0.5, 0.005, 'UTC hour (US ET = UTC - 4 in EDT, UTC - 5 in EST)',
              ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out_png = os.path.join(args.out_dir, f'tod_dist_t{args.threshold}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')

    # Top hours for extremes
    print(f'\n=== TOP-3 HOURS FOR EXTREMES ===')
    top = tab.sort_values('extreme', ascending=False).head(3)
    for hr, row in top.iterrows():
        et_edt = (hr - 4) % 24
        et_est = (hr - 5) % 24
        print(f'  UTC hour {hr:>2d}  (ET ~{et_est:>2d}-{et_edt:>2d}):  '
               f'{int(row["extreme"]):>3d} extremes / {int(row["all_events"]):>4d} all  '
               f'= {row["extreme_rate_pct"]:.1f}% extreme rate  '
               f'({row["pct_of_extremes"]:.1f}% of all extremes)')

    # Also: for each day, when does its WORST event happen?
    print(f'\n=== DAY-LEVEL ANALYSIS: when does each day\'s WORST extreme occur? ===')
    df_extreme = df[df['is_extreme']].copy()
    if not df_extreme.empty:
        worst_per_day = df_extreme.loc[df_extreme.groupby('day')['max_abs_z'].idxmax()]
        worst_hr_dist = worst_per_day['tod_hour'].value_counts().sort_index()
        print(f'  Days with extremes: {worst_per_day["day"].nunique()}')
        print(f'  Worst-event-hour distribution:')
        for hr, n in worst_hr_dist.items():
            et_edt = (hr - 4) % 24
            print(f'    UTC {hr:>2d}  (ET ~{et_edt}):  {n} days')


if __name__ == '__main__':
    main()
