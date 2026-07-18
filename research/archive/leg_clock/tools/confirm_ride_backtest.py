"""Confirm-then-ride backtest (Moises' entry=hold rule, 2026-07-07).

Rule: don't predict the leg's start. Let it CONFIRM (price moves C ticks in one
direction off the last swing), enter in that direction, then TRAIL — exit when
price pulls back R ticks from the running extreme. Pure momentum ride; the
leg-clock finding says even a late entry has most of the move ahead.

Fixed rule, no fitting -> 2025 is honest OOS. Costs charged per round trip.
Reports trades/day, hit%, mean $/trade, $/day for a small (C,R) grid.
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'research', 'level_hold', 'tools'))
from level_hold_study import atlas  # noqa: E402

TICK = 0.25
TICK_VALUE = 0.50   # MNQ $ per tick
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s); lines.append(s)


def backtest_day(c, C, R, cost_ticks):
    """One day of 1m closes. Returns list of trade PnLs in ticks (net of cost)."""
    trades = []
    pos = 0            # 0 flat, +1 long, -1 short
    entry = 0.0
    ext = c[0]         # running extreme since entry
    swing_lo = swing_hi = c[0]   # track confirmation swing while flat
    for p in c[1:]:
        if pos == 0:
            swing_hi = max(swing_hi, p)
            swing_lo = min(swing_lo, p)
            up_conf = (p - swing_lo) >= C * TICK      # confirmed up-leg
            dn_conf = (swing_hi - p) >= C * TICK      # confirmed down-leg
            if up_conf:
                pos, entry, ext = 1, p, p
            elif dn_conf:
                pos, entry, ext = -1, p, p
        elif pos == 1:
            ext = max(ext, p)
            if (ext - p) >= R * TICK:                 # trail hit
                trades.append((p - entry) / TICK - cost_ticks)
                pos, swing_hi = 0, p
                swing_lo = p
        else:  # pos == -1
            ext = min(ext, p)
            if (p - ext) >= R * TICK:
                trades.append((entry - p) / TICK - cost_ticks)
                pos, swing_lo = 0, p
                swing_hi = p
    if pos != 0:                                       # close at end
        last = c[-1]
        pnl = (last - entry) if pos == 1 else (entry - last)
        trades.append(pnl / TICK - cost_ticks)
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=str, default='2025')
    ap.add_argument('--cost-ticks', type=float, default=4.0)  # ~$2 round trip
    args = ap.parse_args()
    days = sorted(os.path.basename(f).replace('.parquet', '')
                  for f in glob.glob(os.path.join(_REPO, 'DATA', 'ATLAS', '1m',
                                                  f'{args.year}_*.parquet')))
    closes = []
    for d in days:
        try:
            closes.append(atlas(d, '1m')['close'].to_numpy())
        except Exception:
            pass
    log(f"confirm-then-ride | {args.year} | {len(closes)} days | cost={args.cost_ticks}t/round-trip")
    log(f"{'C(conf)':>8}{'R(trail)':>9}{'trades/d':>9}{'hit%':>7}{'$/trade':>9}{'$/day':>8}{'PF':>6}")
    for C in (50, 100, 150):
        for R in (40, 80):
            allt = []
            for c in closes:
                allt.extend(backtest_day(c, C, R, args.cost_ticks))
            a = np.array(allt)
            if len(a) == 0:
                continue
            dollars = a * TICK_VALUE
            wins = dollars[dollars > 0].sum()
            losses = -dollars[dollars < 0].sum()
            pf = wins / losses if losses > 0 else float('inf')
            log(f"{C:>8}{R:>9}{len(a)/len(closes):>9.1f}"
                f"{100*(a>0).mean():>7.1f}{dollars.mean():>9.2f}"
                f"{dollars.sum()/len(closes):>8.1f}{pf:>6.2f}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f'confirm_ride_{args.year}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
