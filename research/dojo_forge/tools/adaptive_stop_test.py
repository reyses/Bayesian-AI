#!/usr/bin/env python
"""ADAPTIVE STOP BY REGIME (owner 2026-07-30: "when we see that we are in a
chop we arm an adaptive stop loss").

THE HYPOTHESIS: the right stop width depends on REGIME. In chop a wide stop
just bleeds (you get whipsawed anyway, only slower and for more); in trend a
tight stop cuts you out of the move. So detect chop and adapt.

WHY THIS ONE IS DIFFERENT from every other variant tested today: it is
STRUCTURE-conditioned, not outcome-conditioned -- the regime is measured from
bars STRICTLY BEFORE the entry, so it is a pre-registered setup condition, not
a retroactive selection on how the move turned out. That is exactly the
distinction the owner drew when killing the wakeup timer.

CHOP DETECTOR: Kaufman Efficiency Ratio, computed causally.
    ER = |c[t] - c[t-W]| / sum(|c[i] - c[i-1]|, i in window)
  ER -> 1 : pure directional travel (trend)
  ER -> 0 : all motion, no progress (chop)

THE PREREQUISITE THIS ACTUALLY TESTS: an adaptive rule can only beat a fixed
one if the OPTIMAL STOP WIDTH GENUINELY DIFFERS BY REGIME. So rather than
assume that and tune a rule, this bins entries by ER and reports each stop
width within each bin. If the ranking of stop widths does not flip across
bins, adaptation has nothing to exploit and the idea dies here -- cheaply,
and before anyone builds it.

Read-only. Never touches pocket_dojo live state or its corpus.

Usage:
  python adaptive_stop_test.py --days 60
  python adaptive_stop_test.py --days 60 --er-window 60 --stops 2,5,10
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


def efficiency_ratio(c, W):
    """Causal Kaufman ER. er[i] uses ONLY c[i-W .. i] -- no lookahead."""
    d = np.abs(np.diff(c, prepend=c[0]))
    denom = pd.Series(d).rolling(W, min_periods=W).sum().to_numpy()
    num = np.abs(c - pd.Series(c).shift(W).to_numpy())
    with np.errstate(invalid='ignore', divide='ignore'):
        er = np.where(denom > 0, num / denom, np.nan)
    return er


def simulate(hi, lo, i0, d, stop_pt, horizon, arm_pt, be_pt, entry):
    """One trade; identical mechanics to breakeven_test.simulate."""
    stop = entry - d * stop_pt
    armed = False
    end = min(i0 + horizon, len(hi) - 1)
    for i in range(i0 + 1, end + 1):
        hit = (lo[i] <= stop) if d > 0 else (hi[i] >= stop)
        if hit:
            return (stop - entry) * d - FRICTION_PT
        if arm_pt is not None and not armed:
            reach = (hi[i] >= entry + arm_pt) if d > 0 else (lo[i] <= entry - arm_pt)
            if reach:
                armed = True
                stop = entry + d * be_pt
    px = (hi[end] + lo[end]) / 2.0
    return (px - entry) * d - FRICTION_PT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--stops', default='2,5,10')
    ap.add_argument('--er-window', type=int, default=60, help='5s bars (60 = 5min)')
    ap.add_argument('--arm', type=float, default=5.0)
    ap.add_argument('--be', type=float, default=2.0)
    ap.add_argument('--every', type=int, default=120)
    ap.add_argument('--horizon', type=int, default=360)
    ap.add_argument('--nbins', type=int, default=5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    stops = [float(x) for x in a.stops.split(',')]
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:]

    recs = []
    for day in days:
        df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[['high', 'low', 'close']]
        if len(df) < a.horizon + a.er_window + 10:
            continue
        hi, lo, cl = df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy()
        er = efficiency_ratio(cl, a.er_window)
        lo_i = max(a.every, a.er_window + 1)
        for i0 in range(lo_i, len(df) - a.horizon - 1, a.every):
            e_val = er[i0]
            if not np.isfinite(e_val):
                continue
            for d in (1, -1):
                r = {'er': float(e_val)}
                for S in stops:
                    r[S] = simulate(hi, lo, i0, d, S, a.horizon, a.arm, a.be, float(cl[i0]))
                recs.append(r)

    dfr = pd.DataFrame(recs)
    # equal-count ER bins: bin 0 = most CHOP (lowest ER), last = most TREND
    dfr['bin'] = pd.qcut(dfr['er'], a.nbins, labels=False, duplicates='drop')

    print(f"\nADAPTIVE STOP BY REGIME  ({len(days)} days, n={len(dfr)}, "
          f"ER window {a.er_window*5/60:.0f}min, BE +{a.be:g} armed at +{a.arm:g})")
    print("  bin 0 = deepest CHOP (low efficiency ratio)  ->  "
          f"bin {int(dfr['bin'].max())} = strongest TREND\n")

    hdr = f"{'ER bin':>7} {'ER range':>15} {'n':>6} " + ''.join(f"{'stop -'+format(S,'g'):>22}" for S in stops)
    print(hdr); print('-' * len(hdr))
    rows = []
    for b in sorted(dfr['bin'].dropna().unique()):
        g = dfr[dfr['bin'] == b]
        cells, best, bestv = '', None, -9e9
        rec = {'bin': int(b), 'er_lo': round(g['er'].min(), 3), 'er_hi': round(g['er'].max(), 3),
               'n': len(g), 'by_stop': {}}
        for S in stops:
            v = g[S].to_numpy() * PT_USD
            clo, chi = boot_ci(g[S].to_numpy())
            cells += f"{v.mean():>9.2f} [{clo*PT_USD:>+5.1f},{chi*PT_USD:>+5.1f}]"
            rec['by_stop'][str(S)] = {'usd': round(float(v.mean()), 3),
                                      'ci': [round(clo * PT_USD, 2), round(chi * PT_USD, 2)],
                                      'std': round(float(v.std()), 2),
                                      'worst': round(float(v.min()), 2)}
            if v.mean() > bestv:
                bestv, best = v.mean(), S
        rec['best_stop'] = best
        rows.append(rec)
        print(f"{int(b):>7} {g['er'].min():>6.3f}-{g['er'].max():<8.3f} {len(g):>6} {cells}   best:-{best:g}")

    bests = [r['best_stop'] for r in rows]
    print()
    if len(set(bests)) == 1:
        print(f"  VERDICT: the best stop is -{bests[0]:g} in EVERY regime bin.")
        print("  The optimal width does NOT depend on chop/trend -> an adaptive rule")
        print("  has nothing to exploit here. Hypothesis not supported.")
    else:
        print(f"  VERDICT: best stop CHANGES across regime bins -> {bests}")
        print("  The ranking flips, so adaptation has something real to exploit.")
        print("  Next step: build the switching rule and A/B it against the best fixed stop.")

    out = a.out or os.path.join(OUT, 'adaptive_stop_results.json')
    with open(out, 'w') as f:
        json.dump({'days': days, 'n_days': len(days), 'er_window': a.er_window,
                   'stops': stops, 'arm': a.arm, 'be': a.be, 'bins': rows}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
