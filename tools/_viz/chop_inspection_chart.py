"""Chop inspection chart — visualize the production zigzag on one (choppy) day.

Built 2026-05-21 so the ATR / chop argument can be made with images. For a
given day it draws three stacked panels:
  1. 5s price + the production hardened-leg zigzag, each leg an entry->exit
     segment coloured green (winning leg) / red (losing leg);
  2. per-leg P&L;
  3. the intraday equity curve.
Chop episodes ("focal points") are deliberately left unmarked — for the
viewer to annotate when making the argument.

Usage:  python -m tools._viz.chop_inspection_chart --day 2026_05_05
Output: reports/findings/oos_bad_days/chop_inspection_<day>.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_LEGS = REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
OUT_DIR = REPO / 'reports/findings/oos_bad_days'
TZ = 'America/New_York'
GREEN, RED = '#1a9850', '#d73027'


def to_et(ts):
    return (pd.to_datetime(ts, unit='s', utc=True)
            .dt.tz_convert(TZ).dt.tz_localize(None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True, help='YYYY_MM_DD')
    ap.add_argument('--root', default='DATA/ATLAS_NT8',
                    help='ATLAS root holding 5s/<day>.parquet')
    ap.add_argument('--legs', default=str(DEFAULT_LEGS))
    args = ap.parse_args()
    day = args.day

    bars = (pd.read_parquet(REPO / args.root / '5s' / f'{day}.parquet')
            .sort_values('timestamp').reset_index(drop=True))
    legs = pd.read_csv(args.legs)
    legs = legs[legs['day'] == day].sort_values('entry_ts').reset_index(drop=True)
    if len(legs) == 0:
        raise SystemExit(f'No legs for {day} in {args.legs}')

    bars_dt = to_et(bars['timestamp'])
    close = bars['close'].astype(float).values
    legs['entry_dt'] = to_et(legs['entry_ts'])
    legs['exit_dt'] = to_et(legs['exit_ts'])

    pnl = legs['pnl_usd'].values
    total, n = pnl.sum(), len(legs)
    winners = int((pnl > 0).sum())
    atr = float(legs['atr_pts'].iloc[0])
    rprice = float(legs['r_price'].iloc[0])

    fig, ax = plt.subplots(3, 1, figsize=(17, 11), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1, 1]})

    # panel 1 — price + zigzag legs coloured by win/loss
    ax[0].plot(bars_dt, close, color='0.72', lw=0.7, zorder=1)
    for _, lg in legs.iterrows():
        c = GREEN if lg['pnl_usd'] > 0 else RED
        ax[0].plot([lg['entry_dt'], lg['exit_dt']],
                   [lg['entry_price'], lg['exit_price']],
                   color=c, lw=1.6, marker='o', ms=2.6, zorder=3)
    ax[0].set_ylabel('price')
    ax[0].set_title(
        f'{day}    FLAT P&L ${total:+,.0f}    {n} legs, {winners}/{n} winning '
        f'({winners / n:.0%})    ATR {atr:.2f}pt / R-trigger {rprice:.2f}pt    '
        f'green = winning leg, red = losing leg', fontsize=11)
    ax[0].grid(alpha=0.25)

    # panel 2 — per-leg P&L
    span = (bars_dt.iloc[-1] - bars_dt.iloc[0]).total_seconds() / 86400.0
    w = span / max(n, 1) * 0.7
    ax[1].bar(legs['entry_dt'], pnl, width=w,
              color=[GREEN if p > 0 else RED for p in pnl])
    ax[1].axhline(0, color='0.4', lw=0.8)
    ax[1].set_ylabel('leg P&L  $')
    ax[1].grid(alpha=0.25)

    # panel 3 — intraday equity curve
    cum = np.cumsum(pnl)
    ax[2].fill_between(legs['exit_dt'], cum, 0, step='post',
                       alpha=0.18, color='#3b3b9b')
    ax[2].step(legs['exit_dt'], cum, where='post', color='#3b3b9b', lw=1.4)
    ax[2].axhline(0, color='0.4', lw=0.8)
    ax[2].set_ylabel('cumulative  $')
    ax[2].set_xlabel('time (ET)')
    ax[2].grid(alpha=0.25)
    ax[2].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'chop_inspection_{day}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'Wrote {out}')
    print(f'{day}: {n} legs  FLAT ${total:+.0f}  {winners}/{n} winners '
          f'({winners / n:.0%})  ATR {atr:.2f}pt  R {rprice:.2f}pt')


if __name__ == '__main__':
    main()
