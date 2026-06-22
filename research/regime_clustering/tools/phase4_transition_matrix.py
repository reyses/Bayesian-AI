"""Regime transition matrix ("if this, then that") — CORRECTED (2026-06-12).

Fixes over the committed version (b12a06b8):
  1. CONTIGUOUS timeline only: a transition i->i+1 is counted only if the two segments are
     same-day AND physically adjacent (seg[i].end_idx == seg[i+1].start_idx). The old version
     treated every consecutive entry of the PRISTINE/RECOVERED-filtered list as a transition,
     so 51.8% of them jumped over removed segments + 341 spanned overnight gaps.
  2. PURE_CHAOS is an EXPLICIT state (not dropped). The old "NOISE" was only unbucketed-valid
     segments; the 31,572 chaos blocks were absent from the timeline entirely.
  3. Every transition reported as LIFT over the destination's BASE RATE, with a DAY-BLOCK
     bootstrap 95% CI. "P(R1->R1)=63%" is meaningless until compared to P(any seg = R1); and a
     lift CI that includes 1.0 = not distinguishable from chance.

States: 0 = NOISE (valid but unbucketed), 1..R = regimes, R+1 = CHAOS (PURE_CHAOS segments).
"""
import json
import numpy as np

N_BOOT = 4000
RNG = np.random.RandomState(42)

def main():
    with open('artifacts/regime_buckets.json') as f:
        buckets = json.load(f)
    with open('artifacts/stage2_year_segments.json') as f:
        segs = json.load(f)

    R = len(buckets)
    CHAOS = R + 1
    n_states = R + 2

    # map valid-list position -> regime id (buckets index into the PRISTINE/RECOVERED list)
    pos2regime = {}
    for bid, d in buckets.items():
        rid = int(bid)
        for t in ('members_tier_1', 'members_tier_2', 'members_tier_3', 'members_tier_4'):
            for m in d[t]:
                pos2regime[m] = rid

    # build the full chronological record list with state + adjacency keys
    recs = []
    vpos = -1
    for s in segs:
        st = s['status']
        if st in ('PRISTINE', 'RECOVERED'):
            vpos += 1
            state = pos2regime.get(vpos, 0)            # 0 = NOISE
        elif st == 'PURE_CHAOS':
            state = CHAOS
        else:
            continue
        si = s.get('start_idx'); ei = s.get('end_idx')
        if si is None or ei is None:
            continue
        recs.append((s['day'], int(si), int(ei), int(state)))
    recs.sort(key=lambda r: (r[0], r[1]))             # chronological within day

    # full (uncorrected) matrix for the record, + contiguous-only counts for the claims
    tc_all = np.zeros((n_states, n_states), dtype=np.int64)
    tc = np.zeros((n_states, n_states), dtype=np.int64)
    # per-day structures for day-block bootstrap
    from collections import defaultdict
    day_states = defaultdict(list)                    # day -> [state,...] (for base rate)
    day_pairs = defaultdict(list)                     # day -> [(cur,next),...] contiguous
    same_day = contig = 0
    for k in range(len(recs)):
        day_states[recs[k][0]].append(recs[k][3])
        if k + 1 < len(recs):
            d0, _, e0, c = recs[k]
            d1, s1, _, nx = recs[k + 1]
            tc_all[c, nx] += 1
            if d0 == d1:
                same_day += 1
                if e0 == s1:
                    contig += 1
                    tc[c, nx] += 1
                    day_pairs[d0].append((c, nx))

    np.save('artifacts/transition_matrix.npy', tc)
    days = sorted(day_states.keys())
    total_segs = len(recs)
    base = np.bincount([r[3] for r in recs], minlength=n_states).astype(float) / total_segs

    print(f"[Markov] segments {total_segs:,} ({R} regimes + NOISE + CHAOS) over {len(days)} days")
    print(f"[Markov] consecutive pairs: same-day {same_day:,} -> contiguous {contig:,} "
          f"({100*contig/max(same_day,1):.1f}% kept)")
    print(f"[Markov] base rates: NOISE {100*base[0]:.1f}%  R1 {100*base[1]:.1f}%  "
          f"CHAOS {100*base[CHAOS]:.1f}%")

    def boot_transition(cur, dst):
        """day-block bootstrap of P(next=dst | cur) and lift over base(dst)."""
        # precompute per-day: (#cur->dst, #cur, #dst_states, #states)
        per_day = []
        for d in days:
            pairs = day_pairs[d]
            nd = sum(1 for c, n in pairs if c == cur and n == dst)
            nc = sum(1 for c, n in pairs if c == cur)
            sd = sum(1 for st in day_states[d] if st == dst)
            sn = len(day_states[d])
            per_day.append((nd, nc, sd, sn))
        per_day = np.array(per_day, dtype=float)
        ps, lifts = [], []
        n_d = len(days)
        for _ in range(N_BOOT):
            samp = per_day[RNG.randint(0, n_d, n_d)]
            nc = samp[:, 1].sum()
            sn = samp[:, 3].sum()
            if nc == 0 or sn == 0:
                continue
            p = samp[:, 0].sum() / nc
            br = samp[:, 2].sum() / sn
            ps.append(p)
            if br > 0:
                lifts.append(p / br)
        p_pt = (per_day[:, 0].sum() / per_day[:, 1].sum()) if per_day[:, 1].sum() else float('nan')
        lift_pt = p_pt / base[dst] if base[dst] > 0 else float('nan')
        pci = np.percentile(ps, [2.5, 97.5]) if ps else [np.nan, np.nan]
        lci = np.percentile(lifts, [2.5, 97.5]) if lifts else [np.nan, np.nan]
        return p_pt, pci, lift_pt, lci, int(per_day[:, 1].sum())

    print("\n--- CLAIM CHECKS (contiguous timeline, lift over base rate, day-block 95% CI) ---")
    for cur, label in ((1, 'Regime 1 -> Regime 1  ("Law of Inertia")'),
                       (0, 'NOISE -> Regime 1     ("Chaos Resolution")'),
                       (CHAOS, 'CHAOS -> Regime 1')):
        p, pci, lift, lci, n = boot_transition(cur, 1)
        verdict = ('LIFT > 1 (real)' if lci[0] > 1.0
                   else 'NOT distinguishable from base rate' if (lci[0] <= 1.0 <= lci[1])
                   else 'BELOW base rate')
        print(f"\n{label}   (n={n:,} pairs)")
        print(f"  P(next=R1)= {100*p:.2f}%  CI[{100*pci[0]:.2f}, {100*pci[1]:.2f}]   "
              f"base(R1)= {100*base[1]:.2f}%")
        print(f"  LIFT= {lift:.2f}x  CI[{lci[0]:.2f}, {lci[1]:.2f}]  ->  {verdict}")

    # dominant transitions per top regime (point estimates, for the map)
    print("\n--- top destinations per dominant regime (point estimate) ---")
    for i in range(1, min(6, R + 1)):
        tot = tc[i].sum()
        if tot == 0:
            continue
        dest = np.argsort(tc[i])[::-1][:4]
        parts = []
        for dj in dest:
            nm = 'NOISE' if dj == 0 else ('CHAOS' if dj == CHAOS else f'R{dj}')
            parts.append(f"{nm} {100*tc[i,dj]/tot:.0f}%")
        print(f"  R{i} (n={tot:,}) -> " + ", ".join(parts))

    print("\n[Markov] saved artifacts/transition_matrix.npy")

if __name__ == '__main__':
    main()
