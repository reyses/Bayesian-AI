"""Zigzag range comparison — one day's price with the zigzag drawn at several
ATR multipliers, one panel each, with the per-leg give-up tax quantified.

Built 2026-05-21 for the "find the zigzag sweet spot" argument. The zigzag's
pivot threshold = ATR(14) x multiplier; the R-trigger give-up is ~2 x R per
round trip. A TIGHTER multiplier gives up less of each swing — but detects
smaller swings and flips more often (whipsaw). Each panel reports
`give-up = 2 x R-trigger / median swing` (>100% => the median OFFLINE leg is a
structural loser before friction) and the leg count -> implied daily friction.

IMPORTANT — this is the OFFLINE zigzag. It shows ONE of the two forces:
  force 1 (give-up tax)  — falls monotonically as the multiplier shrinks;
  force 2 (whipsaw)      — a forward pass engine flips on noise and takes whipsaw
                           losers; NOT visible here (offline never whipsaws).
The real sweet spot is where the two balance — that needs a forward pass sim, not
this chart. This chart quantifies force 1 only.

Usage:  python -m tools.viz.zigzag_range_compare --day 2026_05_05 --atr-mults 2,3,4,6,8
Output: reports/findings/oos_bad_days/zigzag_range_compare_<day>.png
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

from tools.viz.auto_swing_marker import detect_swings

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / 'reports/findings/oos_bad_days'
TZ = 'America/New_York'
TICK = 0.25
ATR_PERIOD = 14
FRICTION_PER_LEG = 6.0


def compute_atr(b1: pd.DataFrame) -> float:
    h, l, c = (b1[x].values.astype(float) for x in ('high', 'low', 'close'))
    if len(c) < 2:
        return 1.0
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    return (float(np.median(tr[-ATR_PERIOD * 3:])) if len(tr) >= ATR_PERIOD
            else float(tr.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True, help='YYYY_MM_DD')
    ap.add_argument('--root', default='DATA/ATLAS_NT8')
    ap.add_argument('--atr-mults', default='2,3,4,6,8')
    args = ap.parse_args()
    day = args.day
    mults = [float(x) for x in args.atr_mults.split(',')]

    root = REPO / args.root
    b5 = (pd.read_parquet(root / '5s' / f'{day}.parquet')
          .sort_values('timestamp').reset_index(drop=True))
    b1 = (pd.read_parquet(root / '1m' / f'{day}.parquet')
          .sort_values('timestamp').reset_index(drop=True))
    closes = b5['close'].values.astype(float)
    dt = (pd.to_datetime(b5['timestamp'], unit='s', utc=True)
          .dt.tz_convert(TZ).dt.tz_localize(None))
    atr_pts = compute_atr(b1)

    rows = []
    for m in mults:
        min_rev = max(4, int(round(atr_pts / TICK * m)))
        r_pt = min_rev * TICK
        piv = detect_swings(closes, min_reversal=min_rev, min_bars=36,
                            max_bars=0)
        amps = [abs(closes[piv[k + 1]] - closes[piv[k]])
                for k in range(len(piv) - 1)]
        med = float(np.median(amps)) if amps else float('nan')
        giveup = (2 * r_pt / med) if (med and np.isfinite(med) and med > 0) \
            else float('nan')
        rows.append(dict(m=m, r_pt=r_pt, piv=piv, n_legs=max(len(piv) - 1, 0),
                         med_swing=med, giveup=giveup))

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(17, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        ax.plot(dt, closes, color='0.78', lw=0.6, zorder=1)
        pv_dt = [dt.iloc[k] for k in r['piv']]
        pv_px = [closes[k] for k in r['piv']]
        ax.plot(pv_dt, pv_px, color='#1f3b9b', lw=1.3, marker='o', ms=3,
                zorder=3)
        flag = ('  <-- median swing is a structural loser'
                if (np.isfinite(r['giveup']) and r['giveup'] > 1) else '')
        ax.set_title(
            f"ATR x{r['m']:g}   R-trigger {r['r_pt']:.1f}pt   |   "
            f"{r['n_legs']} legs, friction ${r['n_legs'] * FRICTION_PER_LEG:,.0f}"
            f"   |   median swing {r['med_swing']:.1f}pt   |   "
            f"give-up 2R/swing = {r['giveup'] * 100:.0f}%{flag}", fontsize=10)
        ax.set_ylabel('price')
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('time (ET)')
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.suptitle(f'{day} — zigzag at different ATR multipliers   '
                 f'(ATR(14) = {atr_pts:.2f} pt)   '
                 f'give-up = force 1 only; whipsaw cost not shown',
                 fontsize=12, y=0.999)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'zigzag_range_compare_{day}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'Wrote {out}')
    for r in rows:
        print(f"  ATR x{r['m']:g}: R {r['r_pt']:.1f}pt  {r['n_legs']} legs  "
              f"med swing {r['med_swing']:.1f}pt  give-up {r['giveup'] * 100:.0f}%")


if __name__ == '__main__':
    main()
