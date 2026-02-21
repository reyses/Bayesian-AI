# Summary: Unified Fractal DNA Tree & Feature Enhancements

## Overview
This refactor unifies the Fractal DNA Tree, removing the artificial Long/Short split at the root level. The tree now represents pure market structure (geometry), while directionality is derived from the live Worker Belief Network (momentum/context). Additionally, the feature set for pattern recognition has been expanded to include Relative Volume, ADX, DMI components, and PID states.

## Key Changes

### 1. Core Physics & State
- **`ThreeBodyQuantumState`**: Added `rel_volume` field (relative volume vs 20-bar rolling mean).
- **`QuantumFieldEngine`**:
  - Implemented `rel_volume` calculation in both CPU (`_batch_compute_cpu`) and GPU-path (`batch_compute_states` host-side logic) pipelines.
  - Defined `REL_VOLUME_WINDOW = 20` constant.

### 2. Fractal Discovery Agent
- Updated `_build_parent_chain` to capture a richer 10D feature vector for every ancestor in the fractal chain:
  - `z_score`, `velocity`, `momentum`, `coherence`
  - `rel_volume`, `adx`, `hurst`
  - `dmi_plus`, `dmi_minus` (split from `dmi_diff`)
  - `pid`, `osc_coh`

### 3. Fractal DNA Tree (Unified)
- **Refactoring**:
  - Removed `long_root` and `short_root`. The tree now has a single `root`.
  - Removed `direction` parameter from `fit`, `match`, and tree building methods.
  - `PatternDNA` keys no longer have `L|` or `S|` prefixes (e.g., `1h:0|15m:3`).
- **Feature Vector**:
  - Expanded from 8D to 10D (added `rel_volume` and split DMI).
- **Node Statistics**:
  - `TreeNode` now tracks `mean_adx` and `mean_rel_volume` for energy decay traversal analysis.

### 4. Orchestrator (Gate 1 & Direction)
- **Gate 1**:
  - Now uses `dna_tree.match(p)` against the unified tree.
  - Logic updated to use `dna_node` statistics (win rate, confidence) as a Bayesian prior.
- **Direction Logic**:
  - Removed reliance on static `dir_coeff` or `long_bias` from clustering.
  - **Direction is now exclusively decided by the `TimeframeBeliefNetwork` (`_belief.direction`)**, ensuring trade direction aligns with live multi-timeframe momentum context rather than historical shape bias.
- **Cleanup**:
  - Removed legacy loading of split tree files (`pattern_library_long.pkl`, etc.).
  - Updated logging to reflect new DNA key format.

## Outcome
The system now uses a cleaner, purely structural DNA tree for pattern matching, while delegating directional decision-making to the dynamic belief network. This separation of concerns (Structure vs. Momentum) is expected to reduce False Positives caused by shape-bias in conflicting momentum regimes.
