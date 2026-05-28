# Nightmare Engine V2 Migration Exit Document

This document summarizes the completion of the V2 migration phases, including the deprecation of legacy V1 concepts, the execution of validations, and the verification of lookahead hardening.

## Phase 1: V1 Compatibility Deprecation

The legacy `core_v2/v1_compat.py` shim has been completely removed from the project.

**Key Changes:**
- **File Deletion**: `core_v2/v1_compat.py` is deleted.
- **Nightmare Blended Refactor**: `training/nightmare_blended.py` was updated to no longer rely on V1 mappings.
  - `p_center` derivation and threshold checks (`> 0.60`) were replaced with V2 native equivalent `abs(z) < 0.6`.
  - `variance_ratio` (`vr`) checks were stripped completely, as the engine no longer requires V1 variance conversions to trigger tiers.
  - `vol_rel` logic was removed entirely.
  - The `wick_ratio` calculation was brought inline natively.
- **Test Suite Updates**: 
  - V1 mocks were purged from `tests/test_engine_evaluate.py` and `tests/test_sim_executor.py`.
  - Tests explicitly asserting `p_center` behaviors were ported to assert on raw `abs(z)` physics. 
  - All 106 tests in the suite now **pass**.

## Feature Contamination and Lookahead Verification

We audited the V2 implementation for any potential feature contamination or lookahead bias (the primary failure mode from the previous refactor):

1. **Feature Construction**: `tests/test_core_v2_lookahead.py` asserts that modifying future bars has strictly 0.0 impact on historical feature values. The V2 `FEATURES_5s_v2` pipeline is comprehensively hardened.
2. **Forward Pass System (FPS)**: The V2 `forward_pass_system.py` uses an exact `search_ts = ts - bar_interval` pre-adjustment when calling `np.searchsorted` on DataBento (start-of-bar) and NT8 (end-of-bar) timestamps. This correctly forces the simulation to only consume closed bars relative to the evaluation clock, preventing execution contamination.

## Validation Runs

### 1. Risk-Aware Entry Filter Validation
The `z_range_filter_backtest.py` was successfully run against the V2 feature sets (`L3_1h_z_high_12`, `L3_1h_z_low_12`). The output confirms that the extreme range gating filters perform correctly over the 139D dataset.

### 2. Nightmare Blended Baseline Differential
> **Results**: The `training/run_baseline.py` simulation is complete.
> **In-Sample (IS)**: $-81/day (N=277 days, 45% WinDays)
> **Out-of-Sample (OOS)**: $+110/day (N=68 days, 53% WinDays)
> 
> *Note: This represents the new pure-V2 physics baseline without the CNN applied. Any future feature improvements must be measured against this baseline (requiring 95% CIs and significance testing).*

---

> [!NOTE]
> The `deliverables/` folder has been marked out of scope and deprecated as requested, with `training_*` directories consolidated directly into `training/`.

## Phase 2: VizEngine Architecture Migration

The standalone visualization tools in `tools/viz/` are being migrated to a unified plugin architecture powered by `VizEngine`.

**Key Architectural Advancements:**
1. **Dynamic Infinite Scrolling**: The engine dynamically loads, concatenates, and caches adjacent trading days as the user pans or zooms out.
2. **Unified Core Engine**: All UI logic, geometry persistence, Tkinter event bindings, crosshairs, and bug fixes (e.g. layout snapping via 20px resize) are centralized in `core/engine.py`.
3. **Plugin System**: Tools implement the `VizPlugin` interface. They are decoupled from data-loading and event loops, and they simply draw overlays via the `draw(ax, ax_ind)` hook.
4. **Secondary Indicator Panels**: `VizEngine` conditionally supports a secondary `ax_ind` panel for plugins that require complex feature/ML visualization (e.g., `feature_marker.py`, `classifier_inspector.py`).
5. **Mobile-Friendly UI**: Implemented an extensible on-screen ribbon connected to the Matplotlib toolbar. Plugins can inject custom buttons via `get_buttons()`.

**Status**:
- `trade_visualizer` and `swing_inspector` ported.
- `auto_seeds_generator` and `manual_peak_marker` ported (replacing old crash-prone standalone scripts).
- `feature_marker` and `classifier_inspector` currently in progress.
