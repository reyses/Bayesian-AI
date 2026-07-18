# TASK 121 — Footprint Imbalance spin-out through the sealed harness
**Doc:** 121 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** TASK (Opus drone)
Moises: "let's see what info can be extracted." Source: the sole positive-CI
survivor of leg_clock's 11-concept sweep (+$17.23/day CI[+12.27,+22.99], June
era, pre-dates current rigor). Now archived at
research/archive/leg_clock/reports/ (AG_cat_10_Footprint_Imbalance.md + the
sweep machinery in that folder).

## Phase 1 — EXTRACT (before any porting)
Read the archived report + its generating code. Deliver in your report:
the EXACT rule (formula, thresholds, data inputs, timeframe), what data it
consumed (must be existing ATLAS/derived stores — if it needs unavailable
tick/order-flow data, STOP and report that; the tick purchase is REJECTED per
the order_flow graveyard), and how the June $/day number was computed (its
population, costs, any rigor gaps vs current standards).

## Phase 2 — TWO sealed tests (2024 tune if any params / 2025-26 single-shot)
1. **League line** (standard dossier harness): port as generator
   'FOOTPRINT-IMB' in dossier_signal_pipeline (append-only, reuse _tf_state
   conventions); direction-agreement AUC, base, terciles, fires/day,
   day-block CIs. Run ONLY this stream (nmp9_league.py pattern).
2. **THE MAIN EVENT — entry-fail filter** (Moises' don't-enter-fails thesis):
   on the powered-frontier population (research/exit_dojo/tools machinery;
   23,378 test engagements with terminal economic labels, natural mix), test:
   P(terminal-good | filter passes) vs unconditional base, with the volume
   cost (share of engagements retained) and the day-block CI on the
   good-rate delta. Sweep NOTHING on test — if the rule has free thresholds,
   fix them on 2024 first and freeze. Report the filter frontier point:
   fails avoided vs goods sacrificed vs volume remaining.
## Pre-registered bar
Retained only if the filtered population's terminal-good rate improves over
base with CI excluding 0 AND retained volume ≥ 30% of engagements (a filter
that keeps 3% of trades is a quantile-cell trap). Else: back to the archive
with the updated verdict on its card.

## Rules
Files: pipeline append + tools/footprint_spinout.py +
reports/footprint_imbalance_spinout.md. RUN SYNCHRONOUSLY; python3.11;
commit NOTHING; don't touch dojo_forge/exit_dojo gate dirs. Final message:
the extracted rule, both test results with CIs, PASS/FAIL, deviations.
