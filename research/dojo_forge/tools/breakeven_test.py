#!/usr/bin/env python
"""BREAKEVEN-PLUS STOP TEST (owner 2026-07-30: "+2 stop, this works to make
sure we dont lose money").

THE PROPOSAL: once a trade is far enough in profit, move the stop to
entry+2 (in the profit direction) so the worst case becomes a small WIN
(+2 minus friction) instead of a loss. Intuitively it "guarantees" no
losing trades.

THE THING THE INTUITION MISSES, and what this measures: the guarantee only
binds on trades that REACH the arming trigger. Trades that never get there
still eat the full initial stop. So the real question is not "does the
guarantee hold" (it does, by construction) but whether the trades it saves
outweigh (a) the trades it never protects and (b) the winners it truncates
at +2 that would otherwise have run.

A/B by construction: identical entries, identical initial stop, identical
horizon -- the ONLY difference is whether the breakeven move is armed. So
the delta is attributable, and it gets its own bootstrap CI.

Independent sampled entries (not always-in-market) because this is a
single-trade MANAGEMENT rule -- chaining would confound it with the
reentry mechanics already shown dead in BRACKET_STRATEGY_VERDICT.md.

Read-only. Never touches pocket_dojo live state or its corpus.

Usage:
  python breakeven_test.py --days 60
  python breakeven_test.py --days 60 --stop 10 --arms 5,10,20 --be 2
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports')

FRICTION_PT = 0.89
PT_USD = 2.0
BOOT = 4000
SEED = 11


def boot_ci(x, n=BOOT, seed=SEED):
    a = np.asarray(x, float)
    if len(a) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    s = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_ci_delta(x, y, n=BOOT, seed=SEED):
    """CI on mean(y) - mean(x); paired, since A/B shares entries."""
    d = np.asarray(y, float) - np.asarray(x, float)
    rng = np.random.default_rng(seed)
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def simulate(hi, lo, i0, d, stop_pt, horizon, arm_pt=None, be_pt=2.0, entry=None):
    """One trade. Returns (points_net, outcome_tag).

    arm_pt=None -> plain fixed stop (the control arm).
    Otherwise once profit touches +arm_pt, the stop moves to entry+be_pt.
    Conservative intrabar rule: within a bar, the STOP is checked before the
    arm-trigger, so a bar that both arms and stops out counts as a stop at
    the OLD level. Never lets the sim claim a fill it can't prove ordering for.
    """
    stop = entry - d * stop_pt
    armed = False
    n = len(hi)
    end = min(i0 + horizon, n - 1)
    for i in range(i0 + 1, end + 1):
        hit = (lo[i] <= stop) if d > 0 else (hi[i] >= stop)
        if hit:
            return (stop - entry) * d - FRICTION_PT, ('BE_STOP' if armed else 'STOP')
        if arm_pt is not None and not armed:
            reach = (hi[i] >= entry + arm_pt) if d > 0 else (lo[i] <= entry - arm_pt)
            if reach:
                armed = True
                stop = entry + d * be_pt
    px = (hi[end] + lo[end]) / 2.0          # horizon exit, mid of final bar
    return (px - entry) * d - FRICTION_PT, 'HORIZON'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--stop', type=float, default=10.0)
    ap.add_argument('--arms', default='5,10,20', help='profit levels that arm the BE move')
    ap.add_argument('--be', type=float, default=2.0, help='where the stop moves to (entry + this)')
    ap.add_argument('--every', type=int, default=120, help='sample an entry every N 5s bars')
    ap.add_argument('--horizon', type=int, default=360, help='max hold, 5s bars (360=30min)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    arms = [float(x) for x in a.arms.split(',')]
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:]

    ctrl, variants = [], {arm: [] for arm in arms}
    tags = {arm: [] for arm in arms}
    ctrl_tags = []

    for day in days:
        df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[['high', 'low', 'close']]
        if len(df) < a.horizon + 10:
            continue
        hi = df['high'].to_numpy(); lo = df['low'].to_numpy(); cl = df['close'].to_numpy()
        for i0 in range(a.every, len(df) - a.horizon - 1, a.every):
            for d in (1, -1):
                e = float(cl[i0])
                p, t = simulate(hi, lo, i0, d, a.stop, a.horizon, None, a.be, e)
                ctrl.append(p); ctrl_tags.append(t)
                for arm in arms:
                    p2, t2 = simulate(hi, lo, i0, d, a.stop, a.horizon, arm, a.be, e)
                    variants[arm].append(p2); tags[arm].append(t2)

    def row(pts, tg, label):
        pts = np.asarray(pts, float)
        lo_, hi_ = boot_ci(pts)
        losers = float((pts < 0).mean())
        usd = pts * PT_USD
        # RISK metrics -- the objective that actually matters for a safety
        # rule (owner 2026-07-30: "it is a safety need... im unsure that
        # this will play out arm +2 to protect current equity"). EV is the
        # PRICE of the protection; these are the PRODUCT.
        eq = np.cumsum(usd)
        dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
        return {'label': label, 'n': len(pts),
                'usd_per_trade': round(pts.mean() * PT_USD, 3),
                'ci95': [round(lo_ * PT_USD, 2), round(hi_ * PT_USD, 2)],
                'pct_losing_trades': round(losers * 100, 1),
                'std_usd': round(float(usd.std()), 2),
                'p05_usd': round(float(np.percentile(usd, 5)), 2),
                'p01_usd': round(float(np.percentile(usd, 1)), 2),
                'worst_usd': round(float(usd.min()), 2),
                'cvar05_usd': round(float(usd[usd <= np.percentile(usd, 5)].mean()), 2),
                'max_drawdown_usd': round(dd, 0),
                'outcome_mix': {k: round(100 * tg.count(k) / len(tg), 1)
                                for k in sorted(set(tg))}}

    res = [row(ctrl, ctrl_tags, f'CONTROL fixed -{a.stop:g} stop')]
    for arm in arms:
        r = row(variants[arm], tags[arm], f'BE +{a.be:g} armed at +{arm:g}')
        dlo, dhi = boot_ci_delta(ctrl, variants[arm])
        r['delta_usd_vs_control'] = round((np.mean(variants[arm]) - np.mean(ctrl)) * PT_USD, 3)
        r['delta_ci95'] = [round(dlo * PT_USD, 2), round(dhi * PT_USD, 2)]
        r['delta_significant'] = not (dlo * PT_USD <= 0 <= dhi * PT_USD)
        res.append(r)

    out = a.out or os.path.join(OUT, 'breakeven_test_results.json')
    with open(out, 'w') as f:
        json.dump({'days': days, 'n_days': len(days), 'stop': a.stop, 'be': a.be,
                   'arms': arms, 'horizon_bars5s': a.horizon, 'results': res}, f, indent=1)

    print(f"\nBREAKEVEN-PLUS STOP  ({len(days)} days, 5s bars, init stop -{a.stop:g}pt, "
          f"BE level +{a.be:g}pt, horizon {a.horizon*5/60:.0f}min, friction {FRICTION_PT}pt)\n")
    h = f"{'config':>26} {'n':>7} {'$/trade':>9} {'95% CI':>18} {'%losers':>8} {'delta $':>9} {'delta CI':>18} {'sig':>4}"
    print(h); print('-' * len(h))
    for r in res:
        d = f"{r['delta_usd_vs_control']:+.3f}" if 'delta_usd_vs_control' in r else '--'
        dci = (f"[{r['delta_ci95'][0]:+.2f},{r['delta_ci95'][1]:+.2f}]"
               if 'delta_ci95' in r else '--')
        sig = ('YES' if r.get('delta_significant') else 'no') if 'delta_ci95' in r else '--'
        print(f"{r['label']:>26} {r['n']:>7} {r['usd_per_trade']:>9.3f} "
              f"[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]".rjust(0).ljust(0).rjust(0) +
              f"{'':>1}{r['pct_losing_trades']:>7.1f}% {d:>9} {dci:>18} {sig:>4}")
    print("\nRISK PROFILE  (the objective for a safety rule -- EV above is the PRICE, this is the PRODUCT)")
    h2 = f"{'config':>26} {'%losers':>8} {'std $':>8} {'p05 $':>9} {'p01 $':>9} {'CVaR5 $':>9} {'worst $':>9} {'maxDD $':>9}"
    print(h2); print('-' * len(h2))
    for r in res:
        print(f"{r['label']:>26} {r['pct_losing_trades']:>7.1f}% {r['std_usd']:>8.2f} "
              f"{r['p05_usd']:>9.2f} {r['p01_usd']:>9.2f} {r['cvar05_usd']:>9.2f} "
              f"{r['worst_usd']:>9.2f} {r['max_drawdown_usd']:>9.0f}")
    print('\noutcome mix (%):')
    for r in res:
        print(f"  {r['label']:>26}  {r['outcome_mix']}")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
