# Phase 5 Redo Implementation Plan

## Goal Description
Remediate the issues identified in Doc 029 (Reviewer Verdict):
1. Fix SEASON-12 `resolution_idx` (capture absolute `open_idx` and fix `path` undefined variable).
2. Fix `depth` definition to be STRICTLY PRE-TRADE (outcome leakage removal).
3. Build the telescoping ladder feature extraction (PhE, PhXit, PhPost across 1s, 5s, 15s, 1m, 5m, 15m).
4. Run the three-way policy (ACT, SKIP, INVERT) on ATR-09 first, logging the branch table.

## User Review Required
> [!IMPORTANT]
> The definition of `depth` for each dossier must be mapped strictly to pre-trade knowledge.
> - **ATR-09**: `depth` = `gap_atr_fraction` (the Z-score of the fill).
> - **FIB-17**: `depth` = maximum prior extension beyond the 0.618 level.
> - **VA-13**: `depth` = distance extended outside the VA before rotating back inside.
> - **SEASON-12**: `depth` = pre-market gap size `gap`.
> - **ORDERFLOW-14**: `depth` = extreme delta imbalance or spike distance at trigger.

## Open Questions
> [!WARNING]
> Do we want to apply the fix for `depth` via patching the `events.parquet` directly in a new patch script, or by re-running the 5 individual `ag_deepdive_*.py` scripts? (I will plan to patch `events.parquet` by recalculating `depth` from other columns where possible, or patching the scripts and re-running them).

## Proposed Changes

### SEASON-12 Fix
#### [MODIFY] `tests/SEASON-12_DayOfWeek/ag_deepdive_12_season.py`
- Define `path = df_day['close'].values`
- Capture `open_idx = df[(df['dt'].dt.time >= ...)].index[0]`
- Set `event_idx = open_idx`
- Set `resolution_idx = open_idx + _exit_idx`

### Depth Pre-Trade Fix
#### [NEW] `tools/patch_depth_pretrade.py`
- Read `events.parquet` for ATR-09, FIB-17, VA-13, SEASON-12, ORDERFLOW-14.
- Recalculate `depth` strictly from pre-trade metrics.
- Overwrite `events.parquet` for each dossier.

### Telescoping Ladders
#### [MODIFY] `tools/ag_phase5_final.py`
- Map `event_idx`, `resolution_idx`, and `post_idx = resolution_idx + duration` to the 6 timeframes (1s, 5s, 15s, 1m, 5m, 15m).
- Load all 6 `FEATURES_XX_v2` layers.
- Concat the arrays for `PhE`, `PhXit`, and `PhPost` into one massive `X` matrix.
- Extract matrix shapes and print them.

## Verification Plan
### Automated Tests
- Print matrix shapes for ATR-09.
- Ensure `depth` has zero correlation with `magnitude` if completely decoupled (or at least mechanically distinct).
- Output the 2025 Evaluation branch table.
