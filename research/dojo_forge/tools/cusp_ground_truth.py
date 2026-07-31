#!/usr/bin/env python
"""GROUND-TRUTH CUSP ANALYSIS (owner 2026-07-31: "in the ground truth labels we
can measure it... assume we nailed the entry, from there measure what we need
to survive and exit accordingly and be able to survive the fakeouts").

True cusps = HINDSIGHT zigzag pivots (non-causal on purpose -- these are
labels, not signals). An LLM frame-review would only approximate what this
computes exactly; the subjective owner-style-entry curation is a separate,
later question.

For every leg (cusp -> opposite cusp), assume a PERFECT entry at the cusp
price and measure, bar by bar to the true top/bottom:
  - required breathing room: the deepest intra-leg retracement as a fraction
    of running MFE (dips only counted once MFE >= MINPK so early-noise
    fractions don't dominate). Every intra-leg dip is a FAKEOUT by
    construction -- with hindsight we know the leg continues.
  - fakeout count per leg (edge-triggered touches of the 80%-of-MFE line)
  - capture curve: for trailing room b (hybrid: room_pts = max(MINR, b*MFE)),
    what fraction of the true leg does the trail capture, and what fraction
    of legs survive to the actual top?

Outputs a JSON + printed tables. Read-only; touches nothing live.

Usage:
  python cusp_ground_truth.py --days 200 --R 30
  python cusp_ground_truth.py --days 200 --R 20,30,50 --rooms 0.1,0.2,0.3,0.5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports')

FRICTION_PT = 0.89
PT_USD = 2.0
MINPK = 10.0        # count dip-fractions only once MFE >= this (aligns with arm level)
MINR = 5.0          # absolute room floor (pts) in the hybrid trail
WARN = 0.80         # the 80%-of-MFE fakeout line (owner's breathing line)


def hindsight_zigzag(hi, lo, R):
    """Non-causal pivot labeling: alternating tops/bottoms with reversals >= R.
    Returns list of (bar_idx, price, kind) kind in {'T','B'}."""
    n = len(hi)
    piv = []
    d = 0                      # +1 tracking top, -1 tracking bottom
    mx, mxi, mn, mni = hi[0], 0, lo[0], 0
    for i in range(1, n):
        if hi[i] > mx: mx, mxi = hi[i], i
        if lo[i] < mn: mn, mni = lo[i], i
        if d >= 0 and mx - lo[i] >= R:          # top confirmed
            piv.append((mxi, mx, 'T')); d = -1
            # next-bottom tracking must start AT the pivot bar, not at the
            # confirmation bar -- bars in (mxi, i] are part of the new down-leg
            # and may already contain its true extreme (caught 2026-07-31 on a
            # frame where the emitted pivot sat bars late inside a crash).
            if mxi < i:
                j = mxi + 1 + int(np.argmin(lo[mxi + 1:i + 1]))
                mn, mni = lo[j], j
            else:                               # pivot bar IS the confirm bar
                mn, mni = lo[i], i
        elif d <= 0 and hi[i] - mn >= R:        # bottom confirmed
            piv.append((mni, mn, 'B')); d = 1
            if mni < i:
                j = mni + 1 + int(np.argmax(hi[mni + 1:i + 1]))
                mx, mxi = hi[j], j
            else:
                mx, mxi = hi[i], i
    return piv


def walk_leg(hi, lo, p0, p1, price0, d):
    """Perfect entry at price0/bar p0; walk to the true cusp at p1.
    Returns (final_MFE, required_room_frac, worst_dip_pts, n_fakeouts,
             dips list as fractions)."""
    peak = 0.0
    req, worst_pts = 0.0, 0.0
    trough = None
    fk, below = 0, False
    dips = []
    for i in range(p0 + 1, p1 + 1):
        fav = (hi[i] - price0) if d > 0 else (price0 - lo[i])
        adv = (lo[i] - price0) if d > 0 else (price0 - hi[i])
        if fav > peak:
            if trough is not None and peak >= MINPK:
                frac = (peak - trough) / peak
                dips.append(frac)
                req = max(req, frac)
                worst_pts = max(worst_pts, peak - trough)
            peak, trough, below = fav, adv, False
            continue
        trough = adv if trough is None else min(trough, adv)
        if peak >= MINPK:
            nb = adv <= peak * WARN        # retraced >= 20% of running MFE
            if nb and not below: fk += 1   # edge-triggered
            below = nb
    # terminal dip inside the leg (should be ~0 by construction; keep for honesty)
    if trough is not None and peak >= MINPK:
        frac = (peak - trough) / peak
        dips.append(frac); req = max(req, frac)
        worst_pts = max(worst_pts, peak - trough)
    return peak, req, worst_pts, fk, dips


def capture(hi, lo, p0, p1, price0, d, b, n_after=720):
    """Hybrid trail room = max(MINR, b*MFE). Walk through the leg AND past the
    true cusp (next n_after bars) so the exit after the top is priced too.
    Returns (captured_pts, survived_to_top)."""
    peak, stop_ex = 0.0, -1e9
    end = min(p1 + n_after, len(hi) - 1)
    survived = True
    for i in range(p0 + 1, end + 1):
        cur_adv = (lo[i] - price0) if d > 0 else (price0 - hi[i])
        if cur_adv <= stop_ex:
            return stop_ex, (i > p1)
        fav = (hi[i] - price0) if d > 0 else (price0 - lo[i])
        if fav > peak:
            peak = fav
        stop_ex = peak - max(MINR, b * peak)
    return max(stop_ex, 0.0), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=200)
    ap.add_argument('--R', default='30', help='comma list of reversal thresholds (pts)')
    ap.add_argument('--rooms', default='0.10,0.15,0.20,0.25,0.30,0.40,0.50')
    ap.add_argument('--min-leg', type=float, default=20.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    Rs = [float(x) for x in a.R.split(',')]
    rooms = [float(x) for x in a.rooms.split(',')]
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:]

    payload = {'days': len(days), 'minpk': MINPK, 'minr': MINR, 'results': {}}
    for R in Rs:
        legs = []
        for day in tqdm(days, desc=f'R={R:g}'):
            df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[['high', 'low']]
            if len(df) < 500: continue
            hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
            piv = hindsight_zigzag(hi, lo, R)
            for (i0, pr0, k0), (i1, pr1, k1) in zip(piv, piv[1:]):
                if i1 - i0 < 3: continue
                d = 1 if k0 == 'B' else -1
                L = (pr1 - pr0) * d
                if L < a.min_leg: continue
                mfe, req, wpts, fk, dips = walk_leg(hi, lo, i0, i1, pr0, d)
                caps = {b: capture(hi, lo, i0, i1, pr0, d, b) for b in rooms}
                legs.append({'day': day, 'dir': d, 'L': L, 'mfe': mfe, 'req': req,
                             'worst_pts': wpts, 'fk': fk,
                             'caps': {str(b): [c[0], bool(c[1])] for b, c in caps.items()}})
        if not legs: continue
        req = np.array([l['req'] for l in legs]); mfe = np.array([l['mfe'] for l in legs])
        fk = np.array([l['fk'] for l in legs])
        print(f"\n===== R={R:g}pt  |  {len(legs)} true legs (>= {a.min_leg:g}pt), "
              f"median MFE {np.median(mfe):.1f}pt =====")
        print("\nREQUIRED BREATHING ROOM to survive from perfect entry to the true cusp")
        print("  (deepest intra-leg dip, % of running MFE -- every one a fakeout by construction)")
        qs = [50, 75, 90, 95]
        print('   ' + '  '.join(f"p{q}={100*np.percentile(req,q):.0f}%" for q in qs))
        print(f"\nFAKEOUT COUNT per true leg (80% line): mean {fk.mean():.2f}  "
              f"0:{100*(fk==0).mean():.0f}%  1-2:{100*((fk>=1)&(fk<=2)).mean():.0f}%  "
              f"3+:{100*(fk>=3).mean():.0f}%")
        print(f"\nCAPTURE vs breathing room b (hybrid room = max({MINR:g}pt, b*MFE)):")
        print(f"   {'b':>5} {'survive-to-cusp':>16} {'mean capture':>13} {'mean $/leg':>11}")
        best = None
        for b in rooms:
            arr = np.array([l['caps'][str(b)][0] for l in legs])
            surv = np.array([l['caps'][str(b)][1] for l in legs])
            capfr = arr / mfe
            dollars = (arr.mean() - FRICTION_PT) * PT_USD
            if best is None or arr.mean() > best[1]: best = (b, arr.mean())
            print(f"   {b:>5.2f} {100*surv.mean():>15.1f}% {100*capfr.mean():>12.1f}% "
                  f"{dollars:>11.2f}")
        print(f"   -> best room b={best[0]:g} ({best[1]:.1f}pt avg captured)")
        payload['results'][str(R)] = {
            'n_legs': len(legs), 'median_mfe': float(np.median(mfe)),
            'req_room_pcts': {str(q): float(np.percentile(req, q)) for q in qs},
            'fakeouts_mean': float(fk.mean()),
            'capture': {str(b): {'mean_pts': float(np.mean([l['caps'][str(b)][0] for l in legs])),
                                 'survive': float(np.mean([l['caps'][str(b)][1] for l in legs]))}
                        for b in rooms}}
    out = a.out or os.path.join(OUT, 'cusp_ground_truth.json')
    with open(out, 'w') as f: json.dump(payload, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
