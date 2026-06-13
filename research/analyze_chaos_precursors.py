"""What precedes CHAOS?  (precursor-to-chaos diagnostic, 2026-06-12)

Chaos (PURE_CHAOS, 28% of segments, the un-fittable jagged blocks) is the regime the
fit-based vocabulary throws away -- and plausibly where the drawdown/spikes live. Question:
before a chaos block ignites, is there a readable tell in the segment(s) leading into it?

Segment-level / semi-causal first pass (the KT1-oracle flavour: IF you could identify the
prior block perfectly, could you call chaos?). Uses only the CONTIGUOUS chronological timeline
(same-day, physically adjacent) -- no overnight/gap spurious transitions. All probabilities as
LIFT over the chaos base rate, day-block bootstrap 95% CI.

vol_tier here = FIT-CLEANLINESS tier (v1-2 = PRISTINE, v3-8 = RECOVERED, v9 = CHAOS), NOT
market volatility -- so a 'ramp' v1->...->v9 = fittability deteriorating into chaos.

If nothing lifts above base -> chaos arrives unforecastable from the prior block (stop, or go
to a causal bar-level V2 test). If a worse-fit prior lifts chaos probability -> deterioration
is a readable early warning (then test causally on V2 features = KT2).
"""
import json
import numpy as np
from collections import defaultdict

N_BOOT = 4000
RNG = np.random.RandomState(42)

def main():
    segs = json.load(open('artifacts/stage2_year_segments.json'))
    recs = []
    for s in segs:
        st = s['status']
        if st == 'PURE_CHAOS':
            chaos, vt = 1, s.get('volatility_tier', 9)
        elif st in ('PRISTINE', 'RECOVERED'):
            chaos, vt = 0, s.get('volatility_tier', -1)
        else:
            continue
        si, ei = s.get('start_idx'), s.get('end_idx')
        if si is None or ei is None:
            continue
        recs.append((s['day'], int(si), int(ei), chaos, vt))
    recs.sort(key=lambda r: (r[0], r[1]))

    day_blocks = defaultdict(list)     # day -> [chaos flag per block]  (base rate)
    day_pairs = defaultdict(list)      # day -> [(cur_chaos, cur_vt, next_chaos)] contiguous
    same_day = contig = 0
    for k in range(len(recs)):
        day_blocks[recs[k][0]].append(recs[k][3])
        if k + 1 < len(recs):
            d0, _, e0, c0, vt0 = recs[k]
            d1, s1, _, c1, _ = recs[k + 1]
            if d0 == d1:
                same_day += 1
                if e0 == s1:
                    contig += 1
                    day_pairs[d0].append((c0, vt0, c1))
    days = sorted(day_blocks)
    allflags = np.array([c for d in days for c in day_blocks[d]])
    base = allflags.mean()

    print(f"blocks {len(recs):,} over {len(days)} days | contiguous pairs {contig:,}/{same_day:,} same-day")
    print(f"BASE RATE  P(block = CHAOS) = {100*base:.1f}%   <- every lift below is measured against this\n")

    def boot(cond, label):
        per_day = []
        for d in days:
            num = sum(1 for c0, vt0, c1 in day_pairs[d] if cond(c0, vt0) and c1 == 1)
            den = sum(1 for c0, vt0, c1 in day_pairs[d] if cond(c0, vt0))
            per_day.append((num, den))
        per_day = np.array(per_day, float)
        den_tot = per_day[:, 1].sum()
        if den_tot == 0:
            print(f"  {label:30s} no pairs"); return
        pt = per_day[:, 0].sum() / den_tot
        ps = []
        nd = len(days)
        for _ in range(N_BOOT):
            s = per_day[RNG.randint(0, nd, nd)]
            d = s[:, 1].sum()
            if d > 0:
                ps.append(s[:, 0].sum() / d)
        ci = np.percentile(ps, [2.5, 97.5])
        lift = pt / base
        lci = ci / base
        flag = ('LIFT >1' if lci[0] > 1 else 'below base' if lci[1] < 1 else 'n.s.')
        print(f"  P(next=CHAOS | {label:24s}) = {100*pt:5.1f}%  CI[{100*ci[0]:4.1f},{100*ci[1]:4.1f}]  "
              f"lift {lift:.2f}x [{lci[0]:.2f},{lci[1]:.2f}]  {flag:11s} n={int(den_tot):,}")

    print("--- H1: does chaos cluster? (prior block state) ---")
    boot(lambda c, vt: c == 1, "prior = CHAOS")
    boot(lambda c, vt: c == 0, "prior = fittable")

    print("\n--- H2/H3: does fit-cleanliness deteriorate INTO chaos? (prior fittable block's tier) ---")
    boot(lambda c, vt: c == 0 and vt in (1, 2), "prior PRISTINE  (v1-2)")
    boot(lambda c, vt: c == 0 and vt in (3, 4, 5), "prior RECOVERED (v3-5)")
    boot(lambda c, vt: c == 0 and vt in (6, 7, 8), "prior RECOVERED (v6-8)")
    print("\n(If v6-8 lifts above base while v1-2 sits below, fittability is degrading before chaos\n"
          " -> a readable early warning, worth a causal V2-feature test. If all ~1.0x, chaos is\n"
          " unforecastable from the prior block at this resolution.)")

if __name__ == '__main__':
    main()
