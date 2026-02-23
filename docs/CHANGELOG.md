# Changelog - Bayesian-AI

---

## [V5.1.0] - 2026-02-20 — EXIT QUALITY BY DEPTH — real-time hold per depth

### New
- `feat: EXIT QUALITY BY DEPTH — real-time hold per depth`

---

## [V5.0.0] - 2026-02-19 — PID Oscillation Shadow System

### New
- `docs/JULES_PID_OSCILLATION.md` spec for PID sub-minute shadow analyzer
- `term_pid` (PID control force) and `oscillation_coherence` added to engine feature space (16D cluster vectors)
- `PIDOscillationAnalyzer`: STABLE vs TENSION classification, shadow oracle logging to `checkpoints/pid_oracle_log.csv`
- FN accounting separation: PID-regime bars excluded from fractal FN oracle log
- Gate 0 data-quality override: high-quality templates (≥10 members, WR≥55%, σ≤10 ticks) bypass ALL structural rules
- Per-skip reason labels (`gate0_noise`, `gate0_r3_snap`, `gate0_r3_struct`, `gate0_r4_nightmare`, `gate0_r4_struct`) logged for oracle FN comparison

### Docs
- `AGENTS.md` Section 8: mandatory coding standards (no magic numbers, CUDA patterns, UTF-8, asset constants, frozen dataclasses, checkpoints, warning suppression)
- `docs/OLD/JULES_MONTE_CARLO.md`: archived (superseded by DOE + fractal clustering)

---

## [V4.0.0] - 2026-02-18 — Worker Belief Network & Per-Cluster Regression

### New
- `training/timeframe_belief_network.py`: 8-worker TF belief cascade (1h→15s)
  - PATH CONVICTION = weighted geometric mean of P(direction) across active levels
  - Wave maturity: `decision_wave_maturity` from 5m worker drives TP timing
  - Gate 3: conviction < 0.52 → skip
- Per-cluster regression models in each `PatternTemplate`:
  - OLS MFE model (`mfe_coeff`, `mfe_intercept`): per-bar TP from live features
  - Logistic direction model (`dir_coeff`, `dir_intercept`): P(LONG) per bar
  - `regression_sigma_ticks` = OLS residual std / tick — trailing stop breathing room
- 3-tier direction gate in orchestrator:
  1. Oracle bias ≥ 0.55 → lock direction
  2. Oracle bias sum ≥ 0.10 → max(long_bias, short_bias)
  3. NO_ORACLE → live DMI sign (`dmi_plus - dmi_minus`), else particle_velocity
- Tier filter (`--min-tier N`): pre-filters centroid index; Tier 4 excluded (-$52K drag)
- Per-depth consensus system (`checkpoints/depth_weights.json`): score bonus/penalty + filter_out gate
- Skip reason counters: 4-gate breakdown (`skip_headroom`, `skip_dist`, `skip_brain`, `skip_conviction`) + depth histogram in report
- `--sweep-params`: sweeps min_tier × direction × noise_filter on existing oracle_trade_log.csv

### Fixes
- Direction gate bug: old fallback used `abs(z_score)` centroid → always returned SHORT
- Windows CP1252 crash: 56 non-ASCII chars replaced in orchestrator.py

---

## [V3.0.0] - 2026-02-10 — Oracle Engine & Fractal Clustering

### New
- `config/oracle_config.py`: all oracle thresholds as named constants
- `training/fractal_discovery_agent.py`: `_consult_oracle()` — look-ahead judge per pattern
  - MARKER_MEGA_LONG/SHORT (±2), MARKER_SCALP_LONG/SHORT (±1), MARKER_NOISE (0)
  - Timeframe-specific lookahead windows (15s=60 bars, 1h=8 bars, etc.)
- `training/fractal_clustering.py`:
  - `_aggregate_oracle_intelligence()`: win_rate, expectancy, mega_rate, risk_score, long_bias, short_bias per template
  - `_build_transition_matrix()`: Markov inter-template transition probabilities
  - Blind clustering (physics only), intelligence aggregated post-cluster
- Oracle audit gate: TP/FP_NOISE/FP_WRONG/TN/FN classification per trade
- Oracle report in Phase 4/5 output: precision, recall, per-template alignment
- `training/cuda_kmeans.py`: CUDAKMeans with sklearn CPU fallback
- Dynamic binner (`core/dynamic_binner.py`): Freedman-Diaconis bins for state hashing
- 8-layer `MultiTimeframeContext` cascade (`core/multi_timeframe_context.py`)
- Confidence-weighted probability blending: `prob = prior*(1-conf) + learned*conf`
- `AdaptiveConfidenceManager`: EXPLORATION phase forces threshold=0.0

### Data
- 1s → 15s pre-aggregation cached as parquet (~5,300 bars/day vs ~80,000)
- `training/dbn_to_parquet.py`: Databento .dbn.zst → parquet with front-month stitching
- OHLCV-1s dataset: 13.2M bars, 403 trading days (2025-01-01 → 2026-02-09)

### Fixes
- LONG-only simulation: direction now from z_score sign, dir_sign applied to PnL
- Brain never learns: categorical bins replaced with dynamic FD bins
- P&L inflated: trading_cost_points subtracted from all PnL paths
- Sigma = std_err bug: residual std used in `_calculate_center_mass()`
- P&L in points not dollars: all 3 sim paths multiply by `self.asset.point_value` (MNQ=$2/pt)
- DOE threshold too high: `confidence_threshold` range changed from 0.70–0.90 to 0.30–0.70

---

## [V2.0.1] - Naming Consistency Verification
- Verified no "ProjectX" placeholder remains; codebase consistently uses "Bayesian-AI"

## [V2.0.0] - Unified System Realignment

### Refactoring & Architecture
- Global rename from legacy "Sniper" nomenclature to "Bayesian-AI"
- Formalized LOAD → TRANSFORM → ANALYZE → VISUALIZE pipeline

### Logic Shifts
- Replaced heuristic entry logic with `BayesianBrain` (hash-map O(1) state lookup)
- 9-layer temporal hierarchy: Static (L1–L4) vs Fluid (L5–L9) contexts
- Velocity Gate (L9): `detect_cascade` for sub-second volatility (≥10 pts in <0.5s)

### Dependencies
- NumPy pinned to `<2.0` (Numba compatibility)
- Numba integrated for JIT compilation
- Databento native `.dbn` ingestion via `DatabentoLoader`
