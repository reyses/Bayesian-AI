# Consolidation Refactor Exit Report

**Date**: 2026-05-24
**Status**: COMPLETE

## Phases Completed

### Phase 1: `core/` Demotion
- Successfully swapped `core/` with `core_v2/`.
- Updated global imports from `core.` to `core_v2.`.
- Merged `core.features.extract_features` stub into `core_v2/features.py` to maintain compatibility with `training/compute_features.py` and downstream dependents.

### Phase 2-4: `training/` Consolidation & Archiving
- Consolidated `training/strategies/`, `training/filters/`, and `training/wicks.py`.
- Replaced duplicates in `training/` with their more complete `training_iso_v2/` counterparts (using Diff & Merge option B).
- Removed `training_iso_v2/` after migrating contents.
- Moved `nightmare.py` and `forward_blended.py` to `archive/` (kept `nightmare_blended.py` intact to satisfy live engine imports for the smoke test).
- Purged lookahead references and unused `cnn/` legacy artifacts.

### Phase 5: `tools/` Consolidation
- Reorganized `tools/_*` directories into `tools/*` without underscores.
- Script usages globally updated to point to the new structures.
- Restored missing `parity` directory scripts (`parity_check.py` and `live_parity_check.py`).

### Phase 6-7: Data & Root Cleanup
- Lookahead models removed from `models/`.
- Temporary training dirs, legacy caching, and root-level lookahead scripts deleted.

### Phase 8: Smoke Tests
All required smoke tests passed, including:
1. `python -m training.sourcing.validate_nt8_vs_databento --help`
2. `python -m training.research.screening --help`
3. `python -m training.suites.trade_outcome_suite.run_all --help`
4. `python -m training.pipelines.v2_native --help`
5. `python -m tools.parity.parity_check --help`
6. `python -m live.launcher --help`

## Key Fixes & Interventions
- **Import Errors**: The V2 swap required manual `from core_v2` namespace patches in `training/utils/ticker.py`, `core_v2/strategy_engine.py`, `tests/`, and `core_v2/exits.py`.
- **Duplicate Appends**: The automated append of `core/features.py` into `core_v2/features.py` caused constant collisions (`FEATURE_NAMES`, `TF_ORDER`), breaking `nightmare_blended.py`. The `features.py` file was manually restored to the original `307fa567` state, appending only the `extract_features` stub to allow clean imports.
- **Tools Restoration**: The Phase 5 tools consolidation script missed restoring `parity_check.py` and `live_parity_check.py` to their correct locations in `tools/parity/` (they were mistakenly archived). They have been manually placed.

## Next Steps
- The legacy Pytest suite in `tests/` currently fails due to outdated module references (`core.ledger` and deprecated constants like `_1M_OFFSET` in `nightmare_blended.py`). These tests will need to be rewritten or migrated to be fully compatible with V2.
- Risk-aware entry filter in `nightmare_blended.py` requires validation via `tools/z_range_filter_backtest.py` before live deployment.
