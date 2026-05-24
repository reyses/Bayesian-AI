"""
Per-day PnL distribution for the pivot-physics chain simulator.

Measures what the user actually experiences: how often is a day positive,
what's the typical (mode/median) day, what are the tails.

Runs both IS (2025) and OOS (2026) across configurable chain counts.
Saves a distribution report + a histogram chart.

Usage:
    python tools/pivot_daily_distribution.py --max-chains 5
    python tools/pivot_daily_distribution.py --max-chains 1 3 5 10

Output:
    reports/findings/pivot_daily_distribution.md
    charts/pivot_daily_distribution.png
"""
import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pivot_physics_exit import load_day, DOLLAR_PER_POINT
from tools.pivot_physics_chains import simulate as simulate_chains

ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_daily_distribution.md'
OUT_CHART = 'charts/pivot_daily_distribution.png'

# Default bin width for mode — coarse enough to have duplicates but fine
# enough to be meaningful. $25 wide buckets.
MODE_BIN = 25.0


def collect_per_day(paths, max_chains, r_entry_pts, r_reg_pts,
                    min_res, sniper_sec, label):
    """Return {day: pnl_total} for all days with data."""
    per_day = {}
    for p in tqdm(paths, desc=f'{label} ch={max_chains}', unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        sec_path = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
        feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(sec_path) or not os.path.exists(feat_path):
            continue
        loaded = load_day(sec_path, p, feat_path)
        if loaded is None:
            continue
        sec, closes_1m, ts_1m, residuals_1s, residuals_1m = loaded
        trades = simulate_chains(sec, closes_1m, ts_1m, residuals_1s,
                                 residuals_1m, r_entry_pts, r_reg_pts,
                                 min_res, sniper_sec, max_chains)
        day_pnl = sum(t['pnl'] for t in trades)
        per_day[day] = day_pnl
    return per_day


def distribution_stats(per_day):
    """Return mode (bin center), median, mean, N, win-day%, percentiles."""
    if not per_day:
        return None
    pnls = np.array(list(per_day.values()))
    n = len(pnls)

    # Mode via binning ($25 buckets)
    bins = np.floor(pnls / MODE_BIN) * MODE_BIN
    unique, counts = np.unique(bins, return_counts=True)
    mode_bin_low = unique[np.argmax(counts)]
    mode_bin_center = mode_bin_low + MODE_BIN / 2
    mode_freq = counts.max()

    # Win/loss day counts
    win_days = int((pnls > 0).sum())
    loss_days = int((pnls < 0).sum())
    flat_days = int((pnls == 0).sum())

    return {
        'n_days': n,
        'total': float(pnls.sum()),
        'mean': float(pnls.mean()),
        'median': float(np.median(pnls)),
        'mode_bin_center': float(mode_bin_center),
        'mode_freq': int(mode_freq),
        'mode_pct': mode_freq / n * 100,
        'std': float(pnls.std()),
        'p05': float(np.percentile(pnls, 5)),
        'p25': float(np.percentile(pnls, 25)),
        'p75': float(np.percentile(pnls, 75)),
        'p95': float(np.percentile(pnls, 95)),
        'min': float(pnls.min()),
        'max': float(pnls.max()),
        'win_days': win_days,
        'loss_days': loss_days,
        'flat_days': flat_days,
        'win_day_pct': win_days / n * 100,
        'pnls': pnls,  # keep for chart
    }


def fmt_row(label, s):
    if s is None:
        return f'| {label} | no data |'
    return (f'| {label} | {s["n_days"]} | '
            f'${s["mean"]:+,.1f} | ${s["median"]:+,.1f} | '
            f'${s["mode_bin_center"]:+,.0f} ({s["mode_pct"]:.0f}%) | '
            f'${s["p05"]:+,.0f} | ${s["p25"]:+,.0f} | ${s["p75"]:+,.0f} | '
            f'${s["p95"]:+,.0f} | ${s["min"]:+,.0f} | ${s["max"]:+,.0f} | '
            f'{s["win_day_pct"]:.0f}% |')


def main():
    ap = argparse.ArgumentParser()
    # Default chains=[1] — we can only trade 1 contract until equity reaches
    # $600. Pass --max-chains 1 3 5 if you want to see the post-unlock curves
    # for planning purposes.
    ap.add_argument('--max-chains', type=int, nargs='+', default=[1])
    ap.add_argument('--r-entry', type=float, default=2.0)
    ap.add_argument('--r-reg', type=float, default=8.0)
    ap.add_argument('--min-res', type=float, default=0.5)
    ap.add_argument('--sniper-sec', type=int, default=30)
    ap.add_argument('--is-only', action='store_true')
    args = ap.parse_args()

    r_entry_pts = args.r_entry / DOLLAR_PER_POINT
    r_reg_pts = args.r_reg / DOLLAR_PER_POINT

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    if args.is_only:
        oos_paths = []

    results = {}  # {(chains, dataset): stats}
    per_day_all = {}  # {(chains, dataset): {day: pnl}}

    for mc in args.max_chains:
        is_pd = collect_per_day(is_paths, mc, r_entry_pts, r_reg_pts,
                                args.min_res, args.sniper_sec, 'IS')
        per_day_all[(mc, 'IS')] = is_pd
        results[(mc, 'IS')] = distribution_stats(is_pd)
        if oos_paths:
            oos_pd = collect_per_day(oos_paths, mc, r_entry_pts, r_reg_pts,
                                     args.min_res, args.sniper_sec, 'OOS')
            per_day_all[(mc, 'OOS')] = oos_pd
            results[(mc, 'OOS')] = distribution_stats(oos_pd)

    # Text output to stdout + markdown file
    header = ('| Config | Days | Mean | Median | Mode ($' + f'{MODE_BIN:.0f}' +
              ' bin) | p05 | p25 | p75 | p95 | Min | Max | Win-day% |')
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    print('\n' + header)
    print(sep)
    for (mc, ds), s in results.items():
        print(fmt_row(f'{ds} chains={mc}', s))

    # Markdown
    out = ['# Per-day PnL distribution — pivot_physics_chains', '']
    out.append(f'Config: r_entry=${args.r_entry}  r_reg=${args.r_reg}  '
               f'min_res={args.min_res}  sniper={args.sniper_sec}s  '
               f'mode_bin=${MODE_BIN}')
    out.append('')
    out.append(header)
    out.append(sep)
    for (mc, ds), s in results.items():
        out.append(fmt_row(f'{ds} chains={mc}', s))
    out.append('')
    out.append('## How to read')
    out.append('')
    out.append(f'- **Mode**: most common ${MODE_BIN:.0f}-bucket. Shows the '
               f'typical day-PnL a trader experiences.')
    out.append('- **Median** < **Mean**: right-tail skew (rare big wins lift '
               'the average).')
    out.append('- **Win-day%**: fraction of days with net > $0.')
    out.append('- **p05**: 5th percentile — your worst reasonably-likely day.')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')

    # Chart: histograms side-by-side
    n_cfg = len(results)
    n_rows = len(args.max_chains)
    n_cols = 2 if oos_paths else 1
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 3.2 * n_rows),
                              squeeze=False)
    for r_idx, mc in enumerate(args.max_chains):
        for c_idx, ds in enumerate(['IS', 'OOS'][:n_cols]):
            ax = axes[r_idx][c_idx]
            s = results.get((mc, ds))
            if not s:
                ax.set_title(f'{ds} chains={mc} — no data')
                continue
            ax.hist(s['pnls'], bins=50, color='tab:blue', alpha=0.7,
                    edgecolor='black', linewidth=0.3)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.axvline(s['mean'], color='tab:orange', linewidth=1.4,
                       linestyle='--', label=f'mean ${s["mean"]:+,.0f}')
            ax.axvline(s['median'], color='tab:green', linewidth=1.4,
                       linestyle='--', label=f'median ${s["median"]:+,.0f}')
            ax.axvline(s['mode_bin_center'], color='tab:red', linewidth=1.4,
                       linestyle='-', label=f'mode ${s["mode_bin_center"]:+,.0f}')
            ax.axvline(300, color='purple', linewidth=0.9,
                       linestyle=':', label='$300 target')
            ax.set_title(f'{ds} chains={mc}  n={s["n_days"]}  '
                         f'win-days={s["win_day_pct"]:.0f}%', fontsize=10)
            ax.set_xlabel('PnL ($)')
            ax.set_ylabel('Days')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(OUT_CHART), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {OUT_CHART}')


if __name__ == '__main__':
    main()
