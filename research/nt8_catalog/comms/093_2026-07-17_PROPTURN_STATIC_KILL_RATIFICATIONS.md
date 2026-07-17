# PROP-TURN (static) — clean kill; deviations ratified; dynamic version launched
**Doc:** 093 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executor:** Opus worker (ladder trial #9). Design: Moises — proportional leg-turn
(stall + r×A giveback), stop-and-reverse, 2024-tuned.

## 1. Reviewer ratifications (the worker asked for two)
1. **Escape-clause deviation: RATIFIED.** The literal spec deadlocks (after a flip
   the new leg's A starts below A_min and the gate can freeze all day — 82/318
   test days zeroed). The escape (re-designate a sub-minimal leg on a full A_min
   counter-move, ~3% of fires) is the honest test of the CONCEPT. No clean
   literal-spec result exists; de-stuck numbers are the numbers.
2. **The fires/day ≤ 60 cap was a REVIEWER SPEC ERROR (mine).** It forced the
   2024 selection into degenerate stall cells: direction-correctness cliffs
   1.00 (S0) → 0.99 (S1) → 0.85 (S2) → 0.28 (S3) → 0.18 (S5), and every ≤60/day
   cell is S≥3. The usable regime (S≤2, dir-correct 0.85-1.00, dir-recall up to
   0.32 ≈ RENKO's ceiling) fires 94-705/day and was excluded by my cap.
   Exploration shows capture fails there too, so the CAPTURE kill stands
   regardless — but the TURN-DETECTION potential of S0/S1 was not fairly
   selected. Corrected in the dynamic spec (no hard cap; dir-correctness ≥ 0.8
   as the constraint; precision/recall traded in the objective).

## 2. Verified verdicts (frozen cell r=5%, S=3, A_min=15; test 2025+26)
- **Turn bar: FAIL.** dir-recall@2m 0.042 [0.039,0.046]; precision 0.102 vs
  0.43 chance. (Context: @±5m it reaches 0.31/0.39 — the geometry sees the
  turn REGION, not the ±2m moment.)
- **Capture (the 50-80% budget): FAIL decisively.** Stop-and-reverse at 59
  trades/day: captured mode −14.5 pts, median −3.25, net −0.80 [−1.21,−0.38]
  pts/trade after friction; capture-ratio median −0.05; only 2% of legs land in
  the 0.5-0.8 budget; NO grid regime reaches it (best exploration −0.009).
  The proportional confirm alone cannot tell a noise retrace from a real turn:
  it over-trades and whipsaws. 10% of a 21-37-pt leg ≈ 2-4 pts ≈ the noise floor.
- **The residual: league AUC 0.636** (base 0.57, terciles 0.46→0.71, N=16.5k
  test) — the fires carry real direction information as a combiner FEATURE.
  Same fate as most of the catalog: feeds the state, doesn't stand alone.

## 3. What this kill teaches (for the dynamic version, already specified by Moises)
The static threshold's failure mode is exactly the hypothesis for PROP-TURN-P:
a fixed r cannot separate "noise giveback" from "conviction giveback." The
dynamic version modulates r with the live turn-evidence bundle (opposite-fire
freshness, EXIT-KMDR recency, TURN-CLIMAX marks, stall, ER chop state, leg
size/age vs the label ledger, P_hold drain) — 2024-fitted, sealed, and required
to beat BOTH this static baseline and the standing turn bar. If conviction
modulation can't rescue the geometry, the concept family is closed and the
sequential lane proceeds alone.

## 4. Ladder trial #9: PASS (exemplary)
Found and proved a spec deadlock instead of shipping bug artifacts; separated
bug-driven from real numbers; asked for ratification instead of self-certifying.
Artifacts: reports/propturn.md + 5 raw files; pipeline generator PROP-TURN.
