---
name: zigzag-conditional-table-confounds
description: Confounds that fake signals in conditional-probability tables built on zigzag legs, and the discipline that catches them
metadata:
  type: feedback
---

When building empirical conditional-probability tables over zigzag-leg events
("if K low-range legs, then P(next leg range)"), the zigzag's own structure
injects confounds that manufacture a fake signal. Five were caught in the
2026-05-21 table arc ([[conditional-probability-table-2026-05-21]]). The
discipline:

1. **Never measure DIRECTION on leg-anchored windows.** Zigzag legs strictly
   alternate up/down/up/down, so any directional statistic over a window
   pinned to leg boundaries is eaten by alternation PARITY — it alternates by
   K, not by market behavior. `trend_continuation.py` v1's chop table read
   65/37/64/36/65 — pure parity. Fix: measure direction LEG-DECOUPLED — sign
   of the regression slope of raw closes over a LONG window (90 min) so no
   single leg dominates; the zigzag only marks the event.
2. **Measure the predictor over a FIXED window**, never one that co-grows
   with the outcome. `leg_chop_survival.py` v1 measured the chop ratio over
   the whole wide leg and correlated it with the wide leg's length —
   tautological. Fix: measure chop over the first K tight legs only.
3. **Read the FULL curve shape, not endpoints.** `leg_age_hazard.py`'s first
   verdict compared S(age 0.5min) vs S(age 18min), saw a drop, said
   "exhaustion" — but the curve is a HUMP (falls to a ~5-min trough, then
   recovers). An endpoint test skips the trough. Fix: find the trough/peak,
   test both arms.
4. **OOS-confirm every cell** — trust a cell only where IS and OOS agree
   (cf. [[quantile-selection-overfit]]).
5. **A FLAT oracle-zigzag forward pass is a hindsight partition** — its
   $/day is monotonic in zigzag subdivision, not valid cross-parameter
   ([[flat-pipeline-cross-param]]). The same offline-vs-causal trap recurs:
   `atr_consensus_measure.py`'s offline +$57/leg consensus signal INVERTED
   under a causal measurement (corr +0.34 -> -0.13).

**Why:** This is a real-money system; a confounded table that looks like a
signal would get a bad gate shipped. The user's standing rule on the zigzag
work: "if there's no parity it's back to the drawing board" — a directional
result that just tracks K-parity is no result.

**How to apply:** Before trusting any number in a new zigzag-leg
conditional-table entry, ask: is this a directional measure on a leg-anchored
window? does the predictor co-grow with the outcome? am I reading endpoints
of a non-monotone curve? does it replicate OOS? Build the validation gate
INTO the tool — `trend_continuation.py` prints PASS/FAIL on the parity check;
`leg_age_hazard.py` runs a shape-aware two-arm verdict, not an endpoint test.
