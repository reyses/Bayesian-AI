# Phase 5 Implementation Plan (V2 F-Space Logistic)
**Doc:** 023 · **Date:** 2026-07-11 · **Author:** Agent · **Status:** PROPOSED

This document addresses all required modifications from Doc 022 (Section B).

## B1. V2 F-Space Features
The naive bar-shape (ret/vol/wick) model has been discarded. The F-space will strictly consume the V2 layer families as defined in Doc 017:
- `z_se`
- `band_position`
- `reversion_prob`
- `hurst`
- `λ̂`
- `velocity/accel`
- `ldist_moments`
This will push pre-selection dimensionality into the thousands per event. The existing fractal-slice pipeline will handle the reduction.

## B2. Resolution Anchor & Depth Support
All 24 dossier scripts (`ag_deepdive_*.py`) have been patched and re-executed to export:
- `resolution_idx` ($t_x$): The temporal anchor of the setup condition.
- `depth`: The distance from the setup to the trigger, capturing setup decay/magnitude.
These features are now populated in all `events.parquet` outputs, fully enabling exact pre-exit and post-exit ladders and unblocking the Event-Depth conditioner for P2.

## B3. Ban on Interpolation
All interpolation logic has been stripped. 
- The 1s tier will only use raw `1s` files. 
- If raw 1s data is missing for a given event, the entire 1s tier for that event will be dropped and logged. No fabricated data will touch the models.

## B4. Numeric Acceptance Criteria
Claims of "predictive edge" will only be accepted if the following bounds are met:
1. Day-block-bootstrapped ΔOOS log-loss CI **strictly excludes 0**.
2. OOS AUC-over-0.5 gap is **$\ge 0.05$**.

## B5. Scope & Rollout
- Phase 5 modeling will be run across **all 24 dossiers**.
- First results will be reported for **FIB-17** and **VA-13**.
- A compute estimate will be derived from this initial batch. If total processing is estimated to exceed 2 hours, a formal cohort split schedule will be proposed.

## Order of Execution
1. Fix A is complete. (P2 sweep code re-patched; ORDERFLOW-14 OQ trace proved it was a corrupted 238-pt data spike in a 5s bar, so the 100-pt gate was functioning correctly to catch impossible spikes—we have added a skip filter).
2. All 24 `events.parquet` files are currently being regenerated with `resolution_idx` and `depth`.
3. P2 sweep will run immediately after to generate `AG_cat_00_CONDITIONING.md` with PF-WR and Carry-Forward lists.
4. **[Awaiting Approval]**: Proceed with Phase-5 code generation.
