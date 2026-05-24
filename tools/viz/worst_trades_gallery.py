"""Worst-trades gallery — chart the largest losing legs to see what they
look like.

User (2026-05-21): instead of guessing conditions for the probability table,
look directly at the biggest bad trades and find their common signature
(blender-first EDA).

Pulls the N most-negative FLAT hardened legs (OOS by default) and charts each
in context — the day's 5s price around the leg, the surrounding zigzag legs
(faint; green win / red loss), and the bad leg itself (thick red, entry/exit
dots). A 3-wide grid: a rogues' gallery for eyeballing the shared setup.

Usage:  python -m tools.viz.worst_trades_gallery --n 9 --target oos
Output: reports/findings/oos_bad_days/worst_trades_gallery_<target>.png
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent.parent
LEG_CSV = {
    'oos': REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv',
    'is': REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv',
}
RAW_NT8 = REPO / 'DATA/ATLAS_NT8'
RAW_ATLAS = REPO / 'DATA/ATLAS'
OUT_DIR = REPO / 'reports/findings/oos_bad_days'
TZ = 'America/New_York'
GREEN, RED = '#1a9850', '#d73027'


def bars_path(day: str, tf: str) -> Path:
    nt8 = RAW_NT8 / tf / f'{day}.parquet'
    return nt8 if nt8.exists() else RAW_ATLAS / tf / f'{day}.parquet'


def et_arr(ts):
    return (pd.to_datetime(pd.Series(ts), unit='s', utc=True)
            .dt.tz_convert(TZ).dt.tz_localize(None))


def et1(ts):
    return (pd.Timestamp(float(ts), unit='s', tz='UTC')
            .tz_convert(TZ).tz_localize(None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=9)
    ap.add_argument('--target', default='oos', choices=['oos', 'is'])
    ap.add_argument('--lead-min', type=float, default=30.0)
    ap.add_argument('--trail-min', type=float, default=15.0)
    args = ap.parse_args()

    legs = pd.read_csv(LEG_CSV[args.target])
    worst = legs.nsmallest(args.n, 'pnl_usd').reset_index(drop=True)

    ncol = 3
    nrow = math.ceil(args.n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).flatten()
    bars_cache = {}

    for i, (_, lg) in enumerate(worst.iterrows()):
        ax = axes[i]
        day = str(lg['day'])
        if day not in bars_cache:
            bars_cache[day] = (pd.read_parquet(bars_path(day, '5s'))
                               .sort_values('timestamp')
                               .reset_index(drop=True))
        b5 = bars_cache[day]
        ts = b5['timestamp'].values.astype(np.int64)
        px = b5['close'].values.astype(float)
        w0 = lg['entry_ts'] - args.lead_min * 60
        w1 = lg['exit_ts'] + args.trail_min * 60
        m = (ts >= w0) & (ts <= w1)
        if m.sum() < 2:
            ax.set_visible(False)
            continue
        ax.plot(et_arr(ts[m]), px[m], color='0.6', lw=0.8, zorder=1)
        # surrounding legs in the window (faint, coloured by win/loss)
        ctx = legs[(legs['day'] == lg['day']) &
                   (legs['entry_ts'] >= w0) & (legs['entry_ts'] <= w1)]
        for _, cl in ctx.iterrows():
            c = GREEN if cl['pnl_usd'] > 0 else RED
            ax.plot([et1(cl['entry_ts']), et1(cl['exit_ts'])],
                    [cl['entry_price'], cl['exit_price']],
                    color=c, lw=1.0, alpha=0.35, zorder=2)
        # the bad leg itself
        ax.plot([et1(lg['entry_ts']), et1(lg['exit_ts'])],
                [lg['entry_price'], lg['exit_price']],
                color=RED, lw=2.6, zorder=4)
        ax.plot(et1(lg['entry_ts']), lg['entry_price'], 'o', color=GREEN,
                ms=6, zorder=5)
        ax.plot(et1(lg['exit_ts']), lg['exit_price'], 'o', color=RED,
                ms=6, zorder=5)
        dur = (lg['exit_ts'] - lg['entry_ts']) / 60.0
        amp = abs(lg['exit_price'] - lg['entry_price'])
        ax.set_title(
            f"#{i + 1}  {day}  {et1(lg['entry_ts']):%H:%M}  "
            f"${lg['pnl_usd']:+.0f}  {lg['leg_dir']}  {dur:.0f}min  "
            f"{amp:.0f}pt", fontsize=9)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    for j in range(len(worst), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{args.target.upper()} — {len(worst)} largest losing legs   '
                 f'(thick red = the bad leg; faint = surrounding legs, '
                 f'green win / red loss)', fontsize=11, y=0.999)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'worst_trades_gallery_{args.target}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f'Wrote {out}')
    print()
    print(f'{"#":>2}  {"day":>11}  {"ET":>5}  {"pnl$":>7}  {"dir":>5}  '
          f'{"min":>4}  {"amp pt":>6}')
    for i, (_, lg) in enumerate(worst.iterrows()):
        dur = (lg['exit_ts'] - lg['entry_ts']) / 60.0
        amp = abs(lg['exit_price'] - lg['entry_price'])
        print(f'{i + 1:>2}  {str(lg["day"]):>11}  '
              f'{et1(lg["entry_ts"]):%H:%M}  {lg["pnl_usd"]:>+7.0f}  '
              f'{lg["leg_dir"]:>5}  {dur:>4.0f}  {amp:>6.0f}')


if __name__ == '__main__':
    main()
