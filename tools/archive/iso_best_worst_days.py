"""Visualize best and worst N days from iso pipeline output.

Aggregates trades from all tier pickles per day, ranks days by total
$, picks top-N best and bottom-N worst, and renders one chart per day:

    Top panel:    5s price + per-tier trade entries (triangles, color-coded)
                  + exits (circles=winners, X=losers) + entry-to-exit lines
    Bottom panel: cumulative day PnL over time, with per-tier breakdown
                  as a small horizontal bar at the right edge

Output:
    reports/findings/iso_best_worst_days/{best_NN_<day>.png, worst_NN_<day>.png}
    reports/findings/iso_best_worst_days/day_summary.csv

Usage:
    python tools/iso_best_worst_days.py
    python tools/iso_best_worst_days.py --prefix training_iso_v2/output/oos
    python tools/iso_best_worst_days.py --top-n 10 --bottom-n 10
    python tools/iso_best_worst_days.py --exclude-tiers NMP_FADE_RAW,NMP_RIDE_RAW
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.regret.regret import _load_5s_ohlcv


REGIME_LABELS_CSV = 'DATA/ATLAS/regime_labels_2d.csv'


def load_regime_labels() -> dict:
    if not os.path.exists(REGIME_LABELS_CSV):
        return {}
    df = pd.read_csv(REGIME_LABELS_CSV)
    out = {}
    for _, r in df.iterrows():
        day_iso = str(r.get('day') or r.get('date'))
        if not day_iso:
            continue
        # Normalize to YYYY_MM_DD
        day = day_iso.replace('-', '_')
        regime = r.get('regime_2d') or r.get('regime')
        if pd.isna(regime):
            continue
        out[day] = str(regime)
    return out


def load_all_trades(prefix: str, exclude_tiers: set) -> list:
    pkls = sorted(glob.glob(f'{prefix}_*.pkl'))
    pkls = [p for p in pkls
                  if 'regret' not in p and 'summary' not in p]
    all_trades = []
    for path in pkls:
        tier = os.path.basename(path).replace('.pkl', '').split('_', 1)[1]
        if tier in exclude_tiers:
            continue
        with open(path, 'rb') as f:
            ts = pickle.load(f)
        all_trades.extend(ts)
    return all_trades


def aggregate_by_day(trades: list) -> list:
    by_day = defaultdict(list)
    for t in trades:
        by_day[t.entry_day].append(t)
    rows = []
    for day, ts in by_day.items():
        per_tier_pnl = defaultdict(float)
        per_tier_n = defaultdict(int)
        for t in ts:
            per_tier_pnl[t.entry_tier] += t.pnl
            per_tier_n[t.entry_tier] += 1
        rows.append({
            'day': day,
            'total_pnl': sum(t.pnl for t in ts),
            'n_trades': len(ts),
            'n_winners': sum(1 for t in ts if t.pnl > 0),
            'per_tier_pnl': dict(per_tier_pnl),
            'per_tier_n': dict(per_tier_n),
            'trades': ts,
        })
    return rows


def plot_day(day_data: dict, regime: str, out_path: str, rank_label: str):
    trades = day_data['trades']
    ohlcv = _load_5s_ohlcv(day_data['day'])
    if ohlcv is None or len(ohlcv) == 0:
        print(f'  no OHLCV for {day_data["day"]}, skipping')
        return False

    ts_arr = ohlcv['timestamp'].values.astype(np.int64)
    close = ohlcv['close'].values
    dt_arr = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_arr]

    # Tier color map
    tiers = sorted(set(t.entry_tier for t in trades))
    cmap = plt.cm.tab20
    tier_color = {t: cmap(i / max(len(tiers) - 1, 1)) for i, t in enumerate(tiers)}

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.18)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── Top: price + trade markers ───────────────────────────────────
    ax1.plot(dt_arr, close, color='black', lw=0.7, alpha=0.7,
                  label='5s close')

    # Group trades by tier for legend cleanliness
    plotted_legend = set()
    for t in trades:
        c = tier_color[t.entry_tier]
        marker = '^' if t.direction == 'long' else 'v'
        entry_dt = datetime.fromtimestamp(int(t.entry_ts), tz=timezone.utc)
        exit_dt = datetime.fromtimestamp(int(t.exit_ts), tz=timezone.utc)
        win = t.pnl > 0

        # Entry marker
        label = t.entry_tier if t.entry_tier not in plotted_legend else None
        ax1.scatter([entry_dt], [t.entry_price], c=[c], marker=marker,
                        s=80, zorder=5, edgecolors='black', linewidths=0.6,
                        label=label, alpha=0.9)
        plotted_legend.add(t.entry_tier)

        # Exit marker
        ex_marker = 'o' if win else 'x'
        ex_edge = 'darkgreen' if win else 'red'
        ax1.scatter([exit_dt], [t.exit_price], c=[c], marker=ex_marker,
                        s=50, zorder=5, edgecolors=ex_edge, linewidths=1.0,
                        alpha=0.7)
        # Connector
        ax1.plot([entry_dt, exit_dt], [t.entry_price, t.exit_price],
                      color=c, lw=0.5, alpha=0.4, zorder=2)

    title = (f'{rank_label}  |  {day_data["day"]}  |  regime: {regime}\n'
                  f'TOTAL: ${day_data["total_pnl"]:+.2f}   '
                  f'n={day_data["n_trades"]} trades   '
                  f'wr={day_data["n_winners"]}/{day_data["n_trades"]}')
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel('price')
    ax1.legend(loc='upper left', fontsize=8, ncol=2,
                    framealpha=0.85)
    ax1.grid(True, alpha=0.3)

    # Per-tier table (annotation block on the right)
    tier_lines = []
    for tier, pnl in sorted(day_data['per_tier_pnl'].items(),
                                       key=lambda kv: kv[1], reverse=True):
        n = day_data['per_tier_n'].get(tier, 0)
        tier_lines.append(f'{tier:<14} ${pnl:>+7.2f}  ({n})')
    table_text = '\n'.join(tier_lines)
    ax1.text(1.005, 0.98, f'Per-tier:\n{table_text}',
                  transform=ax1.transAxes, fontsize=8,
                  verticalalignment='top', family='monospace',
                  bbox=dict(facecolor='whitesmoke', alpha=0.85, pad=4))

    # ── Bottom: cumulative day PnL ───────────────────────────────────
    sorted_trades = sorted(trades, key=lambda t: t.exit_ts)
    cum_pnl = np.cumsum([t.pnl for t in sorted_trades])
    exit_times = [datetime.fromtimestamp(int(t.exit_ts), tz=timezone.utc)
                          for t in sorted_trades]
    ax2.plot(exit_times, cum_pnl, color='steelblue', lw=1.5,
                  drawstyle='steps-post')
    ax2.fill_between(exit_times, 0, cum_pnl, alpha=0.25,
                              color='steelblue', step='post')
    ax2.axhline(0, color='gray', lw=0.6)
    ax2.set_ylabel('cum $')
    ax2.set_xlabel('time (UTC)')
    ax2.grid(True, alpha=0.3)

    # Format x-axis as time
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='training_iso_v2/output/is')
    ap.add_argument('--out-dir',
                          default='reports/findings/iso_best_worst_days')
    ap.add_argument('--top-n', type=int, default=10,
                          help='Number of BEST days to plot')
    ap.add_argument('--bottom-n', type=int, default=10,
                          help='Number of WORST days to plot')
    ap.add_argument('--exclude-tiers', type=str, default='',
                          help='Comma-sep tier names to skip (e.g., NMP_FADE_RAW)')
    args = ap.parse_args()

    excluded = {t.strip() for t in args.exclude_tiers.split(',') if t.strip()}
    if excluded:
        print(f'Excluding tiers: {sorted(excluded)}')

    print(f'Loading trades from {args.prefix}_*.pkl ...')
    trades = load_all_trades(args.prefix, excluded)
    print(f'Loaded {len(trades)} trades')
    if not trades:
        print('No trades; aborting')
        return

    days = aggregate_by_day(trades)
    days.sort(key=lambda r: r['total_pnl'])
    regimes = load_regime_labels()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Day summary CSV ────────────────────────────────────────────────
    rows = []
    for d in sorted(days, key=lambda r: r['total_pnl'], reverse=True):
        row = {
            'day': d['day'],
            'regime': regimes.get(d['day'], ''),
            'total_pnl': d['total_pnl'],
            'n_trades': d['n_trades'],
            'n_winners': d['n_winners'],
        }
        for tier, pnl in d['per_tier_pnl'].items():
            row[f'pnl__{tier}'] = pnl
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'day_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f'Day summary CSV -> {csv_path}')

    # ── Best days ──────────────────────────────────────────────────────
    print(f'\n{"=" * 90}')
    print(f'TOP {args.top_n} BEST DAYS')
    print(f'{"=" * 90}')
    print(f'{"rank":>4}  {"day":<12} {"regime":<14} {"total$":>10} {"n":>5} {"wr":>5}')
    best = sorted(days, key=lambda r: r['total_pnl'], reverse=True)[:args.top_n]
    for i, d in enumerate(best):
        regime = regimes.get(d['day'], 'UNKNOWN')
        wr = d['n_winners'] / d['n_trades'] if d['n_trades'] else 0
        print(f'{i+1:>4}.  {d["day"]:<12} {regime:<14} '
                  f'${d["total_pnl"]:>+8.2f} {d["n_trades"]:>5} {wr:>5.1%}')
        out_path = os.path.join(args.out_dir,
                                              f'best_{i+1:02d}_{d["day"]}.png')
        plot_day(d, regime, out_path, rank_label=f'BEST #{i+1}')

    # ── Worst days ─────────────────────────────────────────────────────
    print(f'\n{"=" * 90}')
    print(f'BOTTOM {args.bottom_n} WORST DAYS')
    print(f'{"=" * 90}')
    print(f'{"rank":>4}  {"day":<12} {"regime":<14} {"total$":>10} {"n":>5} {"wr":>5}')
    worst = sorted(days, key=lambda r: r['total_pnl'])[:args.bottom_n]
    for i, d in enumerate(worst):
        regime = regimes.get(d['day'], 'UNKNOWN')
        wr = d['n_winners'] / d['n_trades'] if d['n_trades'] else 0
        print(f'{i+1:>4}.  {d["day"]:<12} {regime:<14} '
                  f'${d["total_pnl"]:>+8.2f} {d["n_trades"]:>5} {wr:>5.1%}')
        out_path = os.path.join(args.out_dir,
                                              f'worst_{i+1:02d}_{d["day"]}.png')
        plot_day(d, regime, out_path, rank_label=f'WORST #{i+1}')

    print(f'\n{">" * 6}  PNGs saved -> {args.out_dir}')


if __name__ == '__main__':
    main()
