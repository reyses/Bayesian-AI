#!/usr/bin/env python
"""MECHANICAL +/-N BRACKET BACKTEST (owner 2026-07-30, "it's testing the strategy").

The strategy under test, exactly as the owner defined it in the dojo:
  - stop loss at -N points  -> CLOSE and FLIP to the opposite direction
  - take profit at +N points -> CLOSE and REENTER the same direction
  - both re-armed at +/-N around each new entry price
  -> the position is ALWAYS in the market; every trigger is a round-trip
     (pays friction) and immediately opens the next position.

WHY THIS EXISTS: hand-chaining these round trips through the live dojo is
slow and produces an anecdote, not a measurement. The live session saw 4
consecutive stop-flips and no take-profit, which SUGGESTS the bracket is
narrower than bar-to-bar noise -- but n=4 proves nothing. This runs the
same mechanics over full days and sweeps the width so the "is +/-10 too
narrow" question gets a real answer.

RUNS ON 5s BARS, not 1m. Intrabar ordering matters here: with a +/-N
bracket both sides can be touched inside one 1m bar and the 1m aggregate
cannot say which came first -- that ambiguity would silently decide trades.
5s resolution mostly removes it (a residual same-5s-bar tie is counted and
reported, never silently resolved).

Does NOT touch pocket_dojo live state or its corpus. Pure read-only sim.

Usage:
  python bracket_test.py --day 2025_06_05
  python bracket_test.py --day 2025_06_05 --from-bar 1069 --dir short --entry 21770.75
  python bracket_test.py --days 40 --widths 5,10,20,30,50
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
D1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports')

FRICTION_PT = 0.89        # round-trip, points (project constant)
PT_USD = 2.0              # MNQ: 0.25 tick = $0.50 -> $2/pt
BOOT = 4000               # bootstrap resamples (project standard)
SEED = 11
WIDTHS_DEFAULT = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]


def _bars5(day):
    p = os.path.join(D5, f'{day}.parquet')
    if not os.path.exists(p):
        return None
    return (pd.read_parquet(p)[['timestamp', 'open', 'high', 'low', 'close']]
            .sort_values('timestamp').reset_index(drop=True))


def run_bracket(d5, width, start_i=0, direction='long', entry=None):
    """Run the always-in-market +/-width bracket from start_i to end of day.

    Returns (trades, ties) where trades is a list of dicts. `ties` counts
    bars where BOTH sides were inside the same 5s bar -- reported, not
    silently resolved (we take the adverse/stop side there, the
    conservative read, and count it so the number is auditable)."""
    hi = d5['high'].to_numpy()
    lo = d5['low'].to_numpy()
    cl = d5['close'].to_numpy()
    ts = d5['timestamp'].to_numpy()
    n = len(d5)

    if entry is None:
        entry = float(cl[start_i])
    d = 1 if direction == 'long' else -1
    entry_i = start_i
    trades, ties = [], 0

    for i in range(start_i + 1, n):
        stop = entry - d * width          # adverse side
        targ = entry + d * width          # favorable side
        hit_s = (lo[i] <= stop) if d > 0 else (hi[i] >= stop)
        hit_t = (hi[i] >= targ) if d > 0 else (lo[i] <= targ)
        if not (hit_s or hit_t):
            continue
        if hit_s and hit_t:
            ties += 1
            hit_t = False                 # conservative: assume adverse first
        px = float(stop if hit_s else targ)
        pts = (px - entry) * d - FRICTION_PT
        trades.append({'i': int(i), 'ts': int(ts[i]), 'kind': 'STOP' if hit_s else 'TARGET',
                       'dir': 'long' if d > 0 else 'short', 'entry': float(entry),
                       'exit': px, 'pts': round(pts, 2), 'bars_held': int(i - entry_i)})
        if hit_s:
            d = -d                        # stop -> FLIP
        # target -> reenter SAME direction (d unchanged)
        entry, entry_i = px, i

    return trades, ties


def pf_trade_wr(pts):
    """Project-canonical Trade WR = profit factor - 1 (NOT count-based)."""
    a = np.asarray(pts, float)
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    if l <= 0:
        return float('inf') if w > 0 else 0.0
    return w / l - 1.0


def boot_ci(x, stat=np.mean, n=BOOT, seed=SEED):
    a = np.asarray(x, float)
    if len(a) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    s = [stat(a[rng.integers(0, len(a), len(a))]) for _ in range(n)]
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def summarize(trades, ties, label):
    if not trades:
        return {'label': label, 'n': 0, 'note': 'no trades'}
    pts = [t['pts'] for t in trades]
    stops = sum(1 for t in trades if t['kind'] == 'STOP')
    targs = len(trades) - stops
    lo, hi = boot_ci(pts)
    held = [t['bars_held'] for t in trades]
    return {
        'label': label,
        'n': len(trades),
        'stops': stops,
        'targets': targs,
        'stop_frac': round(stops / len(trades), 3),
        'ties_ambiguous': ties,
        'net_pts': round(float(np.sum(pts)), 2),
        'net_usd': round(float(np.sum(pts)) * PT_USD, 2),
        'mean_pts_per_trade': round(float(np.mean(pts)), 3),
        'mean_usd_per_trade': round(float(np.mean(pts)) * PT_USD, 2),
        'ci95_usd_per_trade': [round(lo * PT_USD, 2), round(hi * PT_USD, 2)],
        'significant': not (lo * PT_USD <= 0 <= hi * PT_USD),
        'pf_trade_wr': round(pf_trade_wr(pts), 3),
        'median_bars_held_5s': int(np.median(held)),
        'median_sec_held': int(np.median(held) * 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day')
    ap.add_argument('--days', type=int, help='backtest the N most recent available days')
    ap.add_argument('--widths', default=None, help='comma-separated point widths')
    ap.add_argument('--from-bar', type=int, default=None, help='1m bar index to start from')
    ap.add_argument('--dir', default='long')
    ap.add_argument('--entry', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    widths = [float(x) for x in a.widths.split(',')] if a.widths else WIDTHS_DEFAULT
    days = ([a.day] if a.day else
            sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:])

    start_i, direction, entry = 0, a.dir, a.entry
    if a.from_bar is not None:
        # translate a 1m bar index into the matching 5s index (causal: the
        # first 5s bar at or after that 1m bar's timestamp)
        d1 = pd.read_parquet(os.path.join(D1M, f'{days[0]}.parquet'))
        t0 = int(d1['timestamp'].iloc[a.from_bar])
        d5 = _bars5(days[0])
        start_i = int(np.searchsorted(d5['timestamp'].to_numpy(), t0, side='left'))

    results, per_day_rows = [], []
    for w in widths:
        all_tr, all_ties = [], 0
        for day in days:
            d5 = _bars5(day)
            if d5 is None or len(d5) < 100:
                continue
            si = start_i if (a.from_bar is not None and day == days[0]) else 0
            tr, ties = run_bracket(d5, w, si, direction, entry if day == days[0] else None)
            for t in tr:
                t['day'] = day
            all_tr += tr
            all_ties += ties
            if len(days) > 1:
                per_day_rows.append({'width': w, 'day': day, 'n': len(tr),
                                     'net_pts': round(sum(t['pts'] for t in tr), 2)})
        results.append(summarize(all_tr, all_ties, f'+/-{w:g}pt'))

    out = a.out or os.path.join(OUT, 'bracket_test_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'days': days, 'n_days': len(days), 'widths': widths,
                   'friction_pt': FRICTION_PT, 'from_bar': a.from_bar,
                   'results': results, 'per_day': per_day_rows}, f, indent=1)

    print(f"\nMECHANICAL +/-N BRACKET  ({len(days)} day(s), 5s bars, friction {FRICTION_PT}pt/round-trip)")
    print(f"  stop -> flip | target -> reenter same dir | always in market\n")
    hdr = f"{'width':>8} {'n':>6} {'stop%':>6} {'net $':>10} {'$/trade':>9} {'95% CI':>18} {'sig':>4} {'tradeWR':>8} {'hold':>7}"
    print(hdr); print('-' * len(hdr))
    for r in results:
        if r['n'] == 0:
            print(f"{r['label']:>8}   (no trades)"); continue
        ci = f"[{r['ci95_usd_per_trade'][0]:+.2f},{r['ci95_usd_per_trade'][1]:+.2f}]"
        print(f"{r['label']:>8} {r['n']:>6} {r['stop_frac']*100:>5.1f}% {r['net_usd']:>10.0f} "
              f"{r['mean_usd_per_trade']:>9.2f} {ci:>18} {'YES' if r['significant'] else 'no':>4} "
              f"{r['pf_trade_wr']:>8.3f} {r['median_sec_held']:>6}s")
    ties = sum(r.get('ties_ambiguous', 0) for r in results)
    if ties:
        print(f"\n  ({ties} same-5s-bar ties across all widths -- resolved adverse-first, conservative)")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
