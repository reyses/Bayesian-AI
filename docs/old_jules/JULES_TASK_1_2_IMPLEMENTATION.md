# Jules Task 1 & 2 Implementation Details

## Task 1: Shape-First Clustering

### Goal
Replace K-means-first clustering with a two-stage pipeline: Shape Taxonomy Grouping -> Adj-R² Recursive Splitting. Remove LONG/SHORT snowflake split.

### Implementation
- **File:** `training/fractal_clustering.py`
- **Shape Taxonomy:** Added `_shape_label(p)` to group patterns by `depth|pattern_type|lagrange_zone|hurst_category`.
- **Adj-R² Split:** Added `_compute_adj_r2` and replaced `_recursive_split` to use Adjusted R² gain as the stopping criterion (`R2_STOP_THRESHOLD = 0.15`).
- **Fission:** Updated `refine_clusters` to split templates only if weighted Adj-R² gain > `R2_FISSION_MIN_GAIN` (0.05).
- **Unified Pipeline:** Rewrote `create_templates` to process all patterns in a single pass (`_fit_branch(..., 'ALL')`), relying on shape labels to separate physics regimes (and effectively direction).

## Task 2: CST (Coherent Structure Tether)

### Goal
Implement a "tether" check that forces an exit if the trade's physics state drifts too far from the entry template's basin of attraction.

### Implementation

#### 1. Basin Geometry (`training/fractal_clustering.py`)
- **PatternTemplate:** Added `basin_mean` (float) and `basin_std` (float).
- **Computation:** Computed during `_fit_branch` and `refine_clusters` as the mean/std of Euclidean distances of member patterns to the centroid in the 16D scaled feature space.

#### 2. Vector Reconstruction (`core/quantum_field_engine.py`)
- **Utility:** Added `build_16d_vector(state, ancestry_context)` static method to `QuantumFieldEngine`.
- **Function:** Reconstructs the 16D feature vector from a live `ThreeBodyQuantumState` and static ancestry context (depth, timeframe, parent info), matching `FractalClusteringEngine.extract_features` logic.

#### 3. Discovery Tracking (`training/fractal_discovery_agent.py`)
- **Oracle Loop:** Updated `_consult_oracle` to accept `states_map`.
- **Integrity Tracking:** For each bar in the lookahead window, computes the distance from the Entry State.
- **Metadata:** Stores the sequence of distances in `pattern.oracle_meta['structural_integrity']`.

#### 4. Execution Check (`training/wave_rider.py`)
- **Position:** Added `cst_centroid`, `cst_basin_mean`, `cst_basin_std`, `cst_ancestry` to the `Position` dataclass.
- **Check Method:** Implemented `check_structural_integrity(current_state)`.
- **Logic:** Calculates distance of `current_state` to `cst_centroid`. Returns `False` (broken) if `distance > basin_mean + 3.0 * basin_std`. Falls back to 4.5 sigma units if basin is undefined.

#### 5. Orchestrator Wiring (`training/orchestrator.py`)
- **Initialization:** Updated `register_template_logic` to store basin stats in the pattern library.
- **Entry:** Updated `run_forward_pass` to pass CST properties and ancestry context to `wave_rider.open_position`.
- **Live Loop:** Added a check for `wave_rider.check_structural_integrity(current_state)` in the tick loop.
- **Abort:** If structure breaks (`cst_broken`), triggers an immediate exit with reason `structural_break`.
- **Logging:** Added `structural_integrity_at_exit`, `bar_of_structural_death`, `bleed_bars`, `bleed_cost` to `oracle_trade_records`.

## Verification
- **Tests:** `tests/test_clustering_integration.py` confirms clustering logic stability.
- **Scripts:** `scripts/verify_cst.py` confirms `build_16d_vector` and `check_structural_integrity` functionality.
