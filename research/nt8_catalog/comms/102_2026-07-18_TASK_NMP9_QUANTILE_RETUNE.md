# TASK 102 — NMP9 threshold retune by QUANTILE-MATCH (era-corrected, sealed)
**Doc:** 102 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Moises: rolling windows changed → z distributions changed → the verbatim
2026-04 thresholds sit on shifted quantiles (symptom: NMP9-CASCADE 71 fires/2y).
Retune by QUANTILE-MATCHING on 2024 (the validated method — Z_ENTRY=1.8481
precedent, 2026-06-11). NOT a free EDA sweep: no AUC/P&L in the tuning loop
(quantile-cell overfit trap, §3 graveyard). Step 2 (full Shainin re-derivation)
only if this underperforms — reviewer/Moises decision, not yours.

## What to retune vs hold
- RETUNE (distribution-dependent): base z gate (ROCHE 2.0 on z21), the z-exit
  analog (0.5) if used in gating, H1_Z_MIN (1.0), H1_AGAINST_Z_MIN (1.5),
  WICK_5M_MIN (0.83), WICK_15M_MIN (0.77).
- HOLD (semantically absolute): vr<1.0 (regime boundary), velocity 50/100
  (ticks, verbatim formula).

## Method (2024 ONLY, then freeze)
1. Era targets: extract the original per-tier occupancy/trade-rates from the
   journals (docs/daily/2026-04-06: KILL_SHOT 2.5-2.8 tr/day, cascade ladder
   486→70→29; 2026-04-08: phase-1 CASCADE 42 / KILL_SHOT 255 / BASE 8,980 of
   9,277 IS trades; 04-08 sub-tier table FADE_CALM 8,868 / FADEMOM 112) and
   the recovered file's constants. Document which target you anchor each
   threshold to.
2. On 2024 data, quantile-match: for each RETUNE threshold, find the value on
   the CURRENT estimator whose marginal pass-rate reproduces the era pass-rate
   implied by the targets (e.g., wick pair → KILL_SHOT-tier trades/day ≈ 2.5;
   +1h gate → CASCADE/KILLSHOT occupancy ratio ≈ 70/486; base gate → total
   entry-universe rate ≈ era 9,277/277d ≈ 33/day). Solve marginals in waterfall
   order (base first, then wick pair, then 1h gates). No label/PnL contact.
3. Freeze constants → `reports/nmp9_retuned_constants.json` (old vs new +
   anchor used per threshold).
4. Re-run the 9-stream league (tools/nmp9_league.py path) with retuned
   constants: train 2024 / test 2025+26, day-block CIs.
5. Deliver `reports/nmp9_retune.md`: per-tier BEFORE/AFTER table (N, fires/day,
   base, AUC, CIs), combiner delta (same-pool, vs the 0.676/55-stream anchor),
   explicit answers: (a) is CASCADE un-thinned? (b) do any tier verdicts
   change? (c) if AUCs are within CIs of the verbatim run, SAY "structure
   already captured; retune immaterial" — that is a valid and useful verdict.

## Rules
Append-only/param-switch in the pipeline (keep the verbatim constants
available — a flag or second constants dict, do NOT overwrite the verbatim
run's reproducibility). RUN SYNCHRONOUSLY. python3.11 from repo root. Commit
NOTHING. Do NOT touch research/exit_dojo/ (fleet running). Final message:
constants old/new + anchors, before/after league table, combiner delta,
the three answers, deviations.
