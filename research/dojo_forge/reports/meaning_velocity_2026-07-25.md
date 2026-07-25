# Meaning-Velocity Report — the night of 2026-07-24 → 25

**The owner's question**: can the LLM produce meaningful memories? **Answer:
yes — proven, gated, and running at scale by 01:30**, via 7 sprints, 2 design
gates, and 4 owner-steered design pivots from a phone.

## The number
Memo information-rate: **0.7% → 100%** (v1 baseline: 1 data-bearing memo in
151; final config: every admitted memo carries magnitudes + a tape-verified
causal warrant). Bank selectivity: ~100% emission → **11-22% admitted**
(reflection + backstops). All pre-registered bars, none tuned on outcomes.

## How it was won (chronological)
1. **Sprint 1** — education (KNOWLEDGE_PACK v1) + the teacher's own memo #9
   as exemplar: info-rate 0.7%→100% in ONE sprint. Content was never a
   capability problem; it was an instruction problem.
2. **Sprints 2-3** — selectivity by instruction: unreachable. Temp-0 greedy
   decoding pattern-completes the memo schema every frame; harder prompt
   pressure REGRESSED content to 0% (destructive retro). Lesson: don't argue
   with a sampler.
3. **Session stage** — selectivity by CONSTRUCTION: bank-side curation
   (Guard C: cap + dedup). Two of my infra bugs found and fixed en route
   (cross-run cap starvation; concurrent-loop collision).
4. **Sprint 7 — PASS** (4/4 data-bearing, 22% selectivity, cross-referencing
   narrative arcs). Full course launched.
5. **Owner pivots, all gated then deployed**:
   - "Reflection is the guard" → episode-end reflection replaces mechanical
     curation as the admission mind (mechanical demoted to safety+backstop).
   - "Part of the mechanism of curation is the knowledge" → the curator
     judges WITH the education+genome (spy-drill verified).
   - "Look back on the tape — this worked BECAUSE of this" → full-tape
     causal lookback; every KEEP carries a minute-referenced warrant.
   - "Memos are assumptions that become proven or disproven" → hypothesis
     framing; the writer knows the curator is coming. Gate: PASS (8/8,
     11%, warrants + genome cross-links). One truncation bug found
     (reflection think > 1400 tokens = the exam lesson re-learned; fixed
     loudly per exam-v3 rule).

## Sample of the final distillate
> reversion_prob_30=0.999 at 1m with 38% giveback resolved as continuation |
> BECAUSE: trend intact (band_pos_30=+0.716) through m42–m55, validating
> G1.3's "heavy right tail"

Hypothesis → tape-verified verdict → minute-referenced warrant → genome
cross-link. The scientific method as a memory loop.

## The unexpected chapter: ground-truth F-space (owner's 2am question)
- **The exit prize is open**: never-bail leaves mean **+58.8 pts/ep** vs the
  true peak (median 54, p90 110) ≈ **65× friction**. Doc 107 closed the CUT
  side; the ride-exit side still holds a fortune.
- **Gen-0 vs truth**: median 8 min early, +57 pts left — captured ~nothing.
- **Our labels differ by GAIN, not GEOMETRY**: paired frames show gen-1's
  p_exit ~85× elevated exactly where gen-0 fired (p90 0.17 vs 0.002) — same
  boundary, recalibrated trigger.
- **Stated vs effective causes diverge**: the teacher CITES reversion_prob
  (d=+0.07 at its exits — non-discriminating); its exits were actually
  DRIVEN by giveback (d=+0.56). First measured narrated-vs-real-cause gap.
- **The true top has a leading signature**: volume surges (+0.20z) at
  t−2..−1 while velocity fades — effort-without-result divergence BEFORE the
  peak; loudest confirmation at t+1 (+0.26z vol, −0.36z velocity). Weak
  per-instance; real in aggregate; the natural learned-feature candidate for
  the exit head under the asymmetric leash.
- Caveats: hindsight measures; exact-minute framing; day-block CIs required
  before claims. Reports: fspace_label_divergence.md, fspace_groundtruth.md,
  fspace_gt_volume.md.

## Speed verdicts (the denominator)
- anchor2p (E2 port): **×2.49** census speedup, in production.
- E3 server concurrency: 55.5 tok/s aggregate (+63%) for future batch work.
- E1c ngram: +20% on real text (pending E1c-real). E1b draft: died at model
  load — diagnosis backlog.

## Infra hardening receipts (same night)
Self-repair drill round-2 full autonomous PASS (15.5 min sabotage→recovery);
reflection truncation caught by loud-parse rule; classifier-fragility pattern:
long shell one-liners → files invoked by path.

## Standing owner items
Fable-dojo scope (A/B) · 3 policy ratifications (entry freeze, ε(direction,
duration) leash, teacher calibration) · genome fork defaults · docs 151/152.
