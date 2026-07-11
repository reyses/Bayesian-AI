# Phase-5 Directive — F-space Binary Logistic (telescoping ladders, degree sweep)
**Doc:** 017 · **Date:** 2026-07-11 · **Author:** Claude (reviewer), design by Moises · **Status:** FINAL
**BLOCKED-UNTIL:** the doc-016 redo is VERIFIED (trustworthy events.parquet first).

## Purpose
Characterize how the F-space evolves (a) approaching t(e), (b) leading up to the
exit t(x), and (c) AFTER the exit — "how the state converted". Targets per doc
013: registered-response binary (logistic) + RAW unclamped magnitude.
**No during-trade phase** (Moises, 2026-07-11): entry and exit ladders only,
plus the post-exit conversion ladder.

## The telescoping ladder (same tiers for all three anchors)
Fine near the anchor, compressed further out:
| Tier | Bars | Covers |
|---|---|---|
| 1s  | 5 | the anchor's 5s bar, decomposed |
| 5s  | 3 | completes the 15s |
| 15s | 4 | completes the 1m |
| 1m  | 4 | completes the 5m |
| 5m  | 3 | completes the 15m |
| 15m | 4 | the surrounding hour |
≈23 bar-slots × that tier's F-space features. Anchors:
- **ENTRY ladder**: slots end at t(e) (backward).
- **PRE-EXIT ladder**: slots end at t(x) (backward).
- **POST-EXIT ladder**: slots start at t(x) (FORWARD) — conversion measurement.
  Label-side/diagnostic ONLY; never a live-path feature.
Implementation: EXTEND the existing F-space extraction pipeline (the
`Ph1_1s_TminusN_...` fractal-slice machinery) — do not rebuild. Add the two new
anchors as `PhX_` prefixes (PhE entry, PhXit pre-exit, PhPost post-exit).

## Degree sweep: linear → quadratic → cubic (select-then-expand — MANDATORY)
1. Stepwise-select ~15 features at LINEAR degree (existing PyTorch pipeline).
2. Expand ONLY the selected set: add squared terms (quadratic model), then
   cubed terms + pairwise interactions capped at the selected set (cubic).
3. A higher degree is accepted ONLY if it improves DAY-DISJOINT out-of-sample
   performance: train 2024 → test 2025, day-block CV within train for tuning.
   In-sample improvement counts for nothing (June scar: 94.8%-quadratic
   selection = the overfit signature).
4. Report per degree: OOS log-loss, AUC, pseudo-R²; house signal bar applies
   (an AUC-over-0.5 gap < 0.05 is noise — say so, don't dress it).

## Discipline
- One model per anchor (entry / pre-exit / post-exit) per target — no pooled
  phase dummies; coefficients stay interpretable.
- Day-block everything (events cluster within days; effective-N rule).
- Magnitude model consumes RAW points (doc 013); σ columns display-only.
- Reports: per-dossier `FSPACE_<ID>.md` in the dossier folder + master
  `reports/AG_cat_00_FSPACE.md`; plan → approval → execution via comms docs.

## Sequence
1. Finish the doc-016 redo → my verification.
2. Post your Phase-5 implementation plan (next comms number) for approval
   BEFORE building — include the exact feature list per tier (1s tier is raw
   micro-features only) and compute estimate.
