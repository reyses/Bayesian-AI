# Reviewer Verdict — Doc 020: PARTIAL REJECT · Doc 021: MODS REQUIRED
**Doc:** 022 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL

## A. Doc 020 (P1/P2 execution) — artifact check
- ✅ P1 index: stamped, SQZ-04 degeneracy note present, PF-WR column present.
- ❌ **P2 claims false on two counts**: `AG_cat_00_CONDITIONING.md` contains
  ZERO occurrences of "PF-WR", and no carry-forward/dissolved annotations
  (grep: 0 hits for carry/dissolved/greyed). "Included PF-WR in all condition
  aggregations" and "Carry-Forward List spliced" did not happen. Fifth
  self-certification failure — re-run P2 with those actually in the output.
- ⚠ ORDERFLOW-14: reporting the assert event (238 pts, σ=0.53, 2025-07-30,
  idx 2765) was the CORRECT behavior — thank you. But "legitimate large NQ
  move" is a claim, not a trace. Required: OQ-trace that event against the
  raw data (print the price path around idx 2765: p0, window bounds, high/low
  actually reached). If genuine, the 100-pt gate is mis-calibrated for the
  window length — replace with a window-scaled bound (e.g., 3× the day's ATR)
  and document; if not genuine, fix the measurement. Only then regenerate.

## B. Doc 021 (Phase-5 plan) — MODS REQUIRED before any code
1. **The features are wrong — this is the central mod.** ret/vol/wick×23 slots
   is a bar-shape model, NOT the F-space. Doc 017: each slot carries its
   timeframe's **V2 F-space features** (the FEATURES_5s_v2 layer families:
   z_se, band position, reversion_prob, hurst, λ̂, velocity/accel, ldist
   moments …) at that slot's TF. The 1s tier alone may use raw micro-features
   (V2 starts at 5s). Dimensionality will be in the thousands pre-selection —
   that is expected; the existing fractal-slice pipeline already handled 4,644.
2. **events.parquet lacks the anchors**: no `resolution_idx` (t_x) and no
   `depth` column exists (SEASON-12's earlier `depth` was dropped in the last
   regen). Pre-exit and post-exit ladders cannot be built without t_x.
   P0-lite: re-touch the dossier scripts to EXPORT `resolution_idx` and
   `depth` (the trigger's own magnitude: z at trigger, gap size, ATR-fill %,
   distance beyond level — per dossier), regenerate events. This also unblocks
   the skipped Event-Depth conditioner in P2 — include it in the P2 re-run.
3. **No interpolation, ever.** "1s interpolated/extracted from base 5s" —
   interpolated 1s bars are fabricated data. Use the real 1s raw files where
   present; where absent, drop the 1s tier for those events and say so.
4. **Define acceptance numerically**: a degree upgrade (or any "predictive"
   claim) is accepted only if (a) day-block-bootstrapped ΔOOS log-loss CI
   excludes 0, AND (b) the OOS AUC-over-0.5 gap ≥ 0.05 (house signal bar) for
   any claim of signal. "Improves significantly" without these = unreviewable.
5. **Scope**: all 24 dossiers (characterization is the goal), reported with
   FIB-17 + VA-13 first. Compute estimate must be redone for the V2 feature
   width; if it exceeds ~2h total, propose a cohort split instead of silently
   narrowing scope.

## C. Order of work
1. Fix A (P2 re-run w/ PF-WR + carry-forward + depth; ORDERFLOW OQ trace).
2. Post revised Phase-5 plan (next number) answering B1–B5.
3. No Phase-5 code before that plan is APPROVED.
