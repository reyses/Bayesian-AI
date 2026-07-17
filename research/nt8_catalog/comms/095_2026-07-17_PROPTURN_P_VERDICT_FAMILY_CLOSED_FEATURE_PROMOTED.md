# PROP-TURN-P verdict — family CLOSED as detector/strategy; PROMOTED as the
# program's strongest single feature
**Doc:** 095 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executor:** Opus worker (ladder trial #10). Reviewer reproduction: dir-recall@2m
0.302, precision 0.174, league AUC 0.689 — byte-exact.

## 1. The kill-rule ruling (letter vs spirit)
- **Letter: PASS.** Dynamic beats static on both metrics, non-overlapping CIs
  (dir-recall@2m 0.302 vs 0.042; precision 0.173 vs 0.101).
- **Spirit: FAIL — and the spirit governs.** The pre-registered rule existed to
  test whether CONVICTION-MODULATION rescues the geometry. The worker's own §4a
  shows it does not: the winning cell is a near-static sensitive tracker firing
  425/day (8× static); the modulation knobs move dir-recall by 0.0009 (inert —
  P_turn AUC 0.60 is too weak to concentrate fires, so max-recall selection
  routed around it); precision remains BELOW the 0.43 chance line; and capture
  is WORSE than static (net −0.88/trade, 0.00 of legs in the 50-80% budget).
- **RULING: the proportional-turn family (static + dynamic) is CLOSED as a turn
  detector and as a stop-and-reverse strategy.** Further tuning of a
  demonstrated-inert mechanism is the overfit path the graveyard exists to block.

## 2. The promotion (the design still produced something big)
**PROP-TURN-P is the strongest single standalone signal in the program:**
- League OOS AUC **0.689** on a perfectly balanced base (0.50), N=131,370 test
  fires, terciles **0.33 → 0.48 → 0.69** (36pp spread) — one stream matching the
  entire 40-stream pooled combiner's AUC, at the highest density yet.
- What actually carries it (P_turn coefs): **leg geometry** — leg_age −0.42,
  amplitude +0.39, ER +0.27, giveback +0.14. The fire-event marks (KMDR/CLIMAX/
  HA) contributed ~nothing. Moises' "flat and the giveback" intuition survives
  as STATE, not trigger: the leg-geometry conviction dial is a top-tier
  direction feature for the combiner and the sequential model's observation.
- Actions: fires stay in signal_rows_PROPTURNP.parquet (combiner pickup);
  the leg-geometry feature block (age, A, ER, g, stall) is REQUIRED input for
  the Mamba state vector (handoff spec).

## 3. Bookkeeping
- Worker discipline: exemplary again — argued against its own literal PASS with
  evidence; corrected a metric (turn-anchored dir-correctness) and validated
  the scorer against doc-093 exactly. Ladder: 10 trials, 10 passes.
- The turn problem's final tally: 46 event detectors + 409-dim snapshot +
  static and dynamic proportional geometry — ALL fail the ±2m bar. The turn is
  a sequential/path object. The night proceeds to the Mamba (anti-freeze reward
  + supervised warm-start per Moises' authorization).
