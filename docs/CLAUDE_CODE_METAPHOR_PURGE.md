# CLAUDE CODE SPEC: Physics Metaphor Purge (Phase 4b)

## CONFIDENTIAL — BayesianBridge Internal

---

## Context

Phase 4 (Terminology Refactor) renamed class-level identifiers:
- `QuantumFieldEngine` → `StatisticalFieldEngine`
- `quantum_sigma` → `regression_sigma`
- A handful of others

But the actual purge was superficial. The codebase still contains **hundreds**
of physics/quantum metaphors in: file names, class names, variable names,
field names on MarketState, function names, constants, docstrings, comments,
and string literals.

This matters because:
1. New readers (including Claude Code) can't distinguish real statistical
   concepts from metaphors dressed as math
2. The "wave function" is a 3-class softmax — calling it quantum mechanics
   obscures what it actually does
3. "Roche limit," "event horizon," "singularity," "tunneling" have no
   mathematical meaning here — they're just band thresholds
4. Code review and debugging are harder when every variable name requires
   translation from physics to statistics

---

## What Stays (Legitimate Statistical/Mathematical Terms)

These are NOT metaphors — they are real techniques used correctly:

| Term | Why it stays |
|------|-------------|
| `regression_center`, `regression_sigma` | OLS regression — real statistics |
| `z_score` | Standardized score — real statistics |
| `entropy`, `entropy_normalized` | Shannon entropy — real information theory |
| `hurst_exponent` | Rescaled range analysis — real fractal math |
| `adx_strength`, `dmi_plus`, `dmi_minus` | Wilder DMI — real indicators |
| `Monte Carlo` | Simulation method — real statistics |
| `mean_reversion` | Mean reversion — standard finance concept |
| `fractal` | Self-similarity across scales — real math |
| `Bayesian` | Probability updates — real statistics |
| `Ornstein-Uhlenbeck` | Stochastic process — real math |
| `erfi` | Imaginary error function — real math |
| `PID` | Proportional-integral-derivative control — real control theory |
| `Lyapunov` | Stability theory — real dynamical systems |
| `Kelly Criterion` | Position sizing — real finance math |
| `Markov` | Transition probabilities — real math |

---

## OPERATION ORDER

**CRITICAL: Rename order matters. Do NOT rename files until all internal
references are updated. Follow this exact sequence:**

```
Phase A: Variable/field renames (MarketState fields, kernel outputs)
Phase B: Function/method renames
Phase C: Class renames
Phase D: Constant renames
Phase E: String literal renames (band_zone values, etc.)
Phase F: Docstring and comment cleanup
Phase G: File renames (LAST — after all imports updated)
Phase H: Import path updates (match new file names)
Phase I: Verify — run tests, grep for stragglers
```

---

## Phase A: Variable & Field Renames

### MarketState fields (`core/three_body_state.py`)

| Old Field | New Field | Reason |
|-----------|-----------|--------|
| `amplitude_center` | `prob_weight_center` | sqrt(prob) — not a wave amplitude |
| `amplitude_upper` | `prob_weight_upper` | same |
| `amplitude_lower` | `prob_weight_lower` | same |
| `P_at_center` | `P_at_center` | **KEEP** — already statistical |
| `P_near_upper` | `P_near_upper` | **KEEP** |
| `P_near_lower` | `P_near_lower` | **KEEP** |
| `F_upper_repulsion` | `F_upper_band` | Not repulsion — band pressure |
| `F_lower_repulsion` | `F_lower_band` | same |
| `barrier_height` | `reversion_potential` | OU potential, not quantum barrier |
| `band_zone` | `band_zone` | **KEEP** — already clean |
| `reversion_probability` | `reversion_probability` | **KEEP** — already renamed |
| `breakout_probability` | `breakout_probability` | **KEEP** — already renamed |
| `pattern_maturity` | `pattern_maturity` | **KEEP** |
| `structure_confirmed` | `structure_confirmed` | **KEEP** |
| `cascade_detected` | `cascade_detected` | **KEEP** — momentum cascade is standard |
| `reversal_confirmed` | `reversal_confirmed` | **KEEP** |
| `stability_index` | `stability_index` | **KEEP** |
| `entropy_normalized` | `entropy_normalized` | **KEEP** — Shannon entropy |
| `resonance_type` | `alignment_type` | Not physical resonance |
| `cascade_probability` | `cascade_probability` | **KEEP** |
| `amplitude_multiplier` | `signal_multiplier` | Not a wave amplitude |
| `alignment_score` | `alignment_score` | **KEEP** |
| `fractal_confidence` | `fractal_confidence` | **KEEP** |
| `fractal_edge` | `fractal_edge` | **KEEP** |
| `multi_tf_alignment_count` | `multi_tf_alignment_count` | **KEEP** |

**Files affected:** Every file that reads/writes these MarketState fields.
Use project-wide find-and-replace for each rename.

### Kernel output variables (`core/cuda_physics.py`, `core/quantum_field_engine.py`)

| Old Variable | New Variable | Context |
|-------------|-------------|---------|
| `out_coherence` | `out_entropy_norm` | Already IS entropy_normalized |
| `roche_snap` | `band_snap` | Not Roche limit — band extreme + velocity |
| `structural_drive` | `trend_drive` | Momentum + low entropy = trending |
| `d_roche` | `d_band_snap` | CUDA device array |
| `d_drive` | `d_trend_drive` | CUDA device array |

### Physics engine internal variables (`core/quantum_field_engine.py`)

| Old Variable | New Variable |
|-------------|-------------|
| `tunnel_prob` | `reversion_prob` |
| `escape_prob` | `breakout_prob` |
| `barrier_height_arr` | `reversion_potential_arr` |
| `lz_arr` | `band_zone_arr` |
| `a0_arr`, `a1_arr`, `a2_arr` | `pw_center_arr`, `pw_upper_arr`, `pw_lower_arr` |

### Risk engine (`core/risk_engine.py`)

| Old Variable | New Variable |
|-------------|-------------|
| `p_tunnel` | `p_reversion` |
| `p_escape` | `p_breakout` |
| `target_level` | `reversion_target` |
| `stop_level` | `breakout_level` |

### Exit engine (`core/exit_engine.py`)

| Old | New | Notes |
|-----|-----|-------|
| `cst_centroid` | `cluster_centroid` | CST = Cluster State Tracking |
| `cst_basin_mean` | `cluster_basin_mean` | |
| `cst_basin_std` | `cluster_basin_std` | |
| `cst_ancestry` | `cluster_ancestry` | |

---

## Phase B: Function & Method Renames

### `core/three_body_state.py`

| Old | New |
|-----|-----|
| `get_trade_directive()` | `get_trade_directive()` | **KEEP** — name is clean |

### `core/quantum_field_engine.py`

| Old | New |
|-----|-----|
| `calculate_three_body_state()` | `calculate_market_state()` |
| `batch_compute_states()` | `batch_compute_states()` | **KEEP** |
| `_detect_patterns_unified()` | `_detect_patterns_unified()` | **KEEP** |
| `_compute_hurst_numpy()` | `_compute_hurst_numpy()` | **KEEP** |
| `_batch_compute_cpu()` | `_batch_compute_cpu()` | **KEEP** |

### `core/cuda_physics.py`

| Old | New |
|-----|-----|
| `compute_physics_kernel` | `compute_regression_kernel` |
| `detect_archetype_kernel` | `detect_pattern_flags_kernel` |
| `compute_hurst_kernel` | `compute_hurst_kernel` | **KEEP** |
| `compute_dm_tr_kernel` | `compute_dm_tr_kernel` | **KEEP** |

### `core/bayesian_brain.py`

| Old | New |
|-----|-----|
| `get_quantum_probability()` | `get_state_probability()` |
| `should_fire_quantum()` | Delete entirely — dead code. Only `should_fire()` is used. |

### `core/risk_engine.py`

| Old | New |
|-----|-----|
| `calculate_probabilities()` | `calculate_reversion_probabilities()` |

---

## Phase C: Class Renames

| Old Class | New Class | File |
|-----------|-----------|------|
| `QuantumBayesianBrain` | `MarketBayesianBrain` | `core/bayesian_brain.py` |
| `MonteCarloRiskEngine` | `MonteCarloRiskEngine` | **KEEP** — Monte Carlo is real statistics |
| `StatisticalFieldEngine` | `StatisticalFieldEngine` | **KEEP** — already renamed in Phase 4 |
| `MarketState` | `MarketState` | **KEEP** — already renamed |

**Note:** `QuantumBayesianBrain` extends `BayesianBrain` and adds methods that
check `band_zone in ['UPPER_EXTREME', 'LOWER_EXTREME']` and
`structure_confirmed and cascade_detected`. These are just additional filter
conditions — there's nothing "quantum" about them. The class name implies
something it isn't.

**Evaluate for deletion:** `MarketBayesianBrain` (formerly `QuantumBayesianBrain`)
only adds `get_quantum_probability()` (= `get_probability()`) and
`should_fire_quantum()` (= `should_fire()` with extra conditions).
Consider merging into `BayesianBrain` directly and deleting the subclass.
If kept, rename to `MarketBayesianBrain`.

---

## Phase D: Constant Renames

### `core/cuda_physics.py`

| Old | New |
|-----|-----|
| `SIGMA_ROCHE` | `SIGMA_EXTREME` |
| `SIGMA_EVENT` | `SIGMA_BREAKOUT` |
| `GRAVITY_THETA` | `REVERSION_THETA` |
| `REPULSION_EPSILON` | `BAND_PRESSURE_EPSILON` |
| `REPULSION_FORCE_CAP` | `BAND_PRESSURE_CAP` |
| `VELOCITY_THRESHOLD` | `VELOCITY_THRESHOLD` | **KEEP** |
| `MOMENTUM_THRESHOLD` | `MOMENTUM_THRESHOLD` | **KEEP** |
| `COHERENCE_THRESHOLD` | `ENTROPY_THRESHOLD` |

### `core/quantum_field_engine.py`

| Old | New |
|-----|-----|
| `self.SIGMA_ROCHE_MULTIPLIER` | `self.SIGMA_EXTREME_MULTIPLIER` |
| `self.SIGMA_EVENT_MULTIPLIER` | `self.SIGMA_BREAKOUT_MULTIPLIER` |
| `self.TIDAL_FORCE_EXPONENT` | Delete — unused in current code |
| `self.GRAVITY_THETA` | `self.REVERSION_THETA` |
| `self.REPULSION_EPSILON` | `self.BAND_PRESSURE_EPSILON` |
| `self.REPULSION_FORCE_CAP` | `self.BAND_PRESSURE_CAP` |
| `self.COHERENCE_THRESHOLD` | `self.ENTROPY_THRESHOLD` |
| `VELOCITY_CASCADE_THRESHOLD` | `VELOCITY_SPIKE_THRESHOLD` |
| `RANGE_CASCADE_THRESHOLD` | `RANGE_SPIKE_THRESHOLD` |
| `RISK_THETA` | `REVERSION_THETA_MC` |
| `RISK_HORIZON_SECONDS` | `MC_HORIZON_SECONDS` |
| `DEFAULT_GRAVITY_THETA` | `DEFAULT_REVERSION_THETA` |

### `core/cuda_pattern_detector.py`

| Old | New |
|-----|-----|
| All `K_GEO_*` and `K_CDL_*` constants | **KEEP** — these are pattern detection, not physics |

---

## Phase E: String Literal Renames

### Band zone values (used in MarketState.band_zone and hash/eq)

| Old | New | Reason |
|-----|-----|--------|
| `'UPPER_EXTREME'` | `'UPPER_EXTREME'` | **KEEP** — descriptive, not metaphorical |
| `'LOWER_EXTREME'` | `'LOWER_EXTREME'` | **KEEP** |
| `'INNER'` | `'INNER'` | **KEEP** |
| `'CHAOS'` | `'TRANSITION'` | "Chaos" implies physics chaos theory |

### Archetype names (used in pattern classification)

| Old | New | Where |
|-----|-----|-------|
| `'BAND_REVERSAL'` | `'BAND_REVERSAL'` | **KEEP** — standard trading term |
| `'MOMENTUM_BREAK'` | `'MOMENTUM_BREAK'` | **KEEP** |

### Resonance type values

| Old | New |
|-----|-----|
| `'NONE\|PARTIAL\|FULL\|CRITICAL'` | `'NONE\|PARTIAL\|FULL\|STRONG'` |

### Market regime values

| Old | New |
|-----|-----|
| `'STABLE'` | `'STABLE'` | **KEEP** |
| `'CHAOTIC'` | `'VOLATILE'` |
| `'UNKNOWN'` | `'UNKNOWN'` | **KEEP** |

### Fractal confidence values

| Old | New |
|-----|-----|
| `'LOW\|MEDIUM\|HIGH\|EXTREME'` | **KEEP** — these are fine |

### Trade directive strings (`get_trade_directive()`)

| Old | New |
|-----|-----|
| `'Fractal Filter: H=...'` | **KEEP** — Hurst is real |
| `'Not at Roche. Zone:'` | `'Not at band extreme. Zone:'` |
| `'Momentum too strong (breakout likely)'` | **KEEP** |
| `'Wave function not collapsed'` | `'Confirmation signals incomplete'` |
| `'Regime Filter: Cannot short BULL expansion'` | **KEEP** |
| `'Tunnel prob too low'` | `'Reversion prob too low'` |

---

## Phase F: Docstring & Comment Cleanup

### `core/three_body_state.py` — Complete rewrite of class docstring

**Old:**
```
Three-Body Quantum State Vector
Unified field theory for market microstructure
Multi-timeframe cascade with 8 layers (1D -> 1S)
```
```
PHYSICS MODEL:
- Body 1 (Center Star): Fair value regression - ATTRACTIVE
- Body 2 (Upper Singularity): +2 sigma resistance - REPULSIVE
- Body 3 (Lower Singularity): -2 sigma support - REPULSIVE
- Particle: Price - exists in SUPERPOSITION until measured

QUANTUM MECHANICS:
- Wave function psi = a0*psi_center + a1*psi_upper + a2*psi_lower
- Measurement (L8-L9) causes collapse to definite state
- Tunneling probability determines mean reversion likelihood
```

**New:**
```
Market State Vector
Statistical representation of market microstructure.
Multi-timeframe cascade with 8 layers (1D -> 1S).
```
```
REGRESSION MODEL:
- Center: Fair value (OLS regression center)
- Upper band: +2 sigma resistance
- Lower band: -2 sigma support
- Price: Current position relative to bands

PROBABILITY MODEL:
- 3-class softmax: P(near center), P(near upper), P(near lower)
- Confirmation signals (structure + cascade) validate setups
- Reversion probability from OU first-passage analysis
```

### Section header renames in MarketState

| Old | New |
|-----|-----|
| `# ═══ THREE ATTRACTORS ═══` | `# ═══ REGRESSION BANDS ═══` |
| `# ═══ PARTICLE STATE ═══` | `# ═══ PRICE STATE ═══` |
| `# ═══ FORCE FIELDS ═══` | `# ═══ STATISTICAL FORCES ═══` |
| `# ═══ QUANTUM WAVE FUNCTION ═══` | `# ═══ PROBABILITY DISTRIBUTION ═══` |
| `# ═══ DECOHERENCE ═══` | `# ═══ SIGNAL QUALITY ═══` |
| `# ═══ MEASUREMENT OPERATORS ═══` | `# ═══ CONFIRMATION SIGNALS ═══` |
| `# ═══ LAGRANGE CLASSIFICATION ═══` | `# ═══ BAND CLASSIFICATION ═══` |
| `# ═══ QUANTUM TUNNELING ═══` | `# ═══ REVERSION STATISTICS ═══` |
| `# ═══ NIGHTMARE FIELD EQUATION COMPONENTS ═══` | `# ═══ VOLATILITY MODEL COMPONENTS ═══` |
| `# ═══ RESONANCE (PHASE 3 EXTENSION) ═══` | `# ═══ ALIGNMENT (PHASE 3 EXTENSION) ═══` |

### `core/quantum_field_engine.py` — Module docstring

**Old:**
```
Quantum Field Calculator
Computes three-body gravitational fields + quantum wave function
Integrates Nightmare Protocol gravity calculations
```

**New:**
```
Statistical Field Engine
Computes regression bands, z-scores, probability distributions,
and mean-reversion/breakout statistics from price data.
GPU-accelerated via Numba CUDA kernels.
```

### `core/cuda_physics.py` — Module docstring

**Old:**
```
CUDA-Accelerated Physics Engine
Implements fused kernels for StatisticalFieldEngine.
Replacing CPU-bound physics calculations with parallel GPU compute.
```

**New:**
```
CUDA-Accelerated Statistical Engine
Implements fused kernels for StatisticalFieldEngine.
Regression, z-score, and probability computations on GPU.
```

### `core/cuda_physics.py` — Kernel docstrings

**`compute_physics_kernel` → `compute_regression_kernel`:**

Old:
```
Fused Physics Kernel:
1. Rolling Linear Regression (Center, Sigma, Slope)
2. Z-Score & Volatility
3. Tidal Forces (Gravity, Momentum)
4. Wave Function (Probabilities, Entropy, Coherence)
```

New:
```
Fused Statistical Kernel:
1. Rolling Linear Regression (Center, Sigma, Slope)
2. Z-Score & Volatility
3. Mean Reversion + Band Pressure Forces
4. Probability Distribution (3-class softmax, Entropy)
```

**`detect_archetype_kernel` → `detect_pattern_flags_kernel`:**

Old:
```
Detects Physics Archetypes based on computed fields.
```

New:
```
Detects statistical pattern flags (band snap, trend drive).
```

### Inline comments to update (all files)

Search and replace these comment patterns:

| Old Pattern | New Pattern |
|-------------|-------------|
| `# Roche Snap` | `# Band Snap (extreme + velocity)` |
| `# Structural Drive` | `# Trend Drive (momentum + low entropy)` |
| `# Forces (Gravity)` | `# Mean Reversion Force` |
| `# Repulsion` | `# Band Pressure` |
| `# Wave Function` | `# Probability Distribution` |
| `# quantum state` | `# market state` |
| `# quantum mechanics` | (delete comment) |
| `# Nightmare Protocol` | `# Volatility model` |
| `# Three-body` | `# Band regression` |
| `# Lagrange` | `# Band zone` |
| `# superposition` | `# probability distribution` |
| `# tunneling` | `# reversion` |
| `# event horizon` | `# 3-sigma breakout level` |
| `# singularity` | `# band extreme` |
| `# Roche limit` | `# 2-sigma band extreme` |
| `# particle` | `# price` |
| `# wave function collapsed` | `# signals confirmed` |
| `# tidal force` | `# reversion force` |
| `# gravitational` | `# reversion` |

### `core/risk_engine.py` — Module docstring

**Old:**
```
Quantum Risk Engine
Powered by Numpy Vectorization (formerly QuantLib)
Performs Monte Carlo simulations for event horizon probability
```

**New:**
```
Reversion Probability Engine
Monte Carlo simulation for mean-reversion vs breakout probability.
Models price as Ornstein-Uhlenbeck process between regression bands.
```

### `core/bayesian_brain.py` — Module docstring is fine, but class docstring:

**Old `QuantumBayesianBrain` docstring:**
```
Extends BayesianBrain for MarketState
```

**New `MarketBayesianBrain` docstring:**
```
Extends BayesianBrain with MarketState-specific filters.
Checks band zone, confirmation signals, and reversion probability
before firing.
```

### `core/fractal_clustering.py` — `generate_semantic_name()`

| Old | New |
|-----|-----|
| `"Singularity"` | `"Extreme"` |
| `"Roche"` | `"BandSnap"` |
| `"MeanRev"` | `"MeanRev"` | **KEEP** |
| `"Chop"` | `"Chop"` | **KEEP** |
| `"Shock"` | `"Shock"` | **KEEP** — market shock is standard |
| `"Drive"` | `"Drive"` | **KEEP** |
| `"Flow"` | `"Flow"` | **KEEP** |
| `"Grind"` | `"Grind"` | **KEEP** |
| `"Trend"` | `"Trend"` | **KEEP** |
| `"MR"` | `"MR"` | **KEEP** — mean reverting |
| `"Persist"` | `"Persist"` | **KEEP** |
| `"Range"` | `"Range"` | **KEEP** |

---

## Phase G: File Renames

**Do this LAST — after all internal references updated.**

| Old Path | New Path |
|----------|----------|
| `core/quantum_field_engine.py` | `core/statistical_field_engine.py` |
| `core/three_body_state.py` | `core/market_state.py` |
| `core/cuda_physics.py` | `core/cuda_statistics.py` |

**Note:** `core/risk_engine.py` keeps its name — Monte Carlo risk engine is
legitimate terminology.

---

## Phase H: Import Path Updates

After file renames, update ALL imports across the project:

```python
# Old → New
from core.quantum_field_engine import StatisticalFieldEngine
→ from core.statistical_field_engine import StatisticalFieldEngine

from core.three_body_state import MarketState
→ from core.market_state import MarketState

from core.cuda_physics import (compute_physics_kernel, detect_archetype_kernel, ...)
→ from core.cuda_statistics import (compute_regression_kernel, detect_pattern_flags_kernel, ...)

from core.bayesian_brain import QuantumBayesianBrain
→ from core.bayesian_brain import MarketBayesianBrain
```

### Files that import from renamed modules (exhaustive list from codebase):

| Importing File | Old Import | New Import |
|---------------|-----------|-----------|
| `core/quantum_field_engine.py` → `core/statistical_field_engine.py` | `from core.three_body_state import MarketState` | `from core.market_state import MarketState` |
| `core/quantum_field_engine.py` → `core/statistical_field_engine.py` | `from core.cuda_physics import (compute_physics_kernel, detect_archetype_kernel, compute_dm_tr_kernel, compute_hurst_kernel)` | `from core.cuda_statistics import (compute_regression_kernel, detect_pattern_flags_kernel, compute_dm_tr_kernel, compute_hurst_kernel)` |
| `core/execution_engine.py` | `from core.exit_engine import ExitEngine, ExitAction, PositionState` | **KEEP** — no rename |
| `live/live_engine.py` | `from core.quantum_field_engine import StatisticalFieldEngine` | `from core.statistical_field_engine import StatisticalFieldEngine` |
| `live/live_engine.py` | `from core.bayesian_brain import QuantumBayesianBrain` | `from core.bayesian_brain import MarketBayesianBrain` |
| `live/bar_aggregator.py` | `from core.quantum_field_engine import StatisticalFieldEngine` | `from core.statistical_field_engine import StatisticalFieldEngine` |
| `training/trainer.py` (not in context but exists) | All of the above patterns | Update accordingly |
| `training/orchestrator.py` (not in context but exists) | All of the above patterns | Update accordingly |
| Any test files | All of the above patterns | Update accordingly |

**Grep command to find ALL importers:**
```bash
grep -rn "from core.quantum_field_engine\|from core.three_body_state\|from core.cuda_physics\|QuantumBayesianBrain" --include="*.py" | grep -v __pycache__
```

---

## Phase I: Verification

### 1. Grep for stragglers

```bash
# Should return ZERO results after purge:
grep -rni "quantum\|three.body\|singularity\|roche\|event.horizon\|superposition\|decoherence\|wave.function\|tidal.force\|nightmare\|particle.*price\|collapsed.*state" --include="*.py" | grep -v __pycache__ | grep -v "# KEEP:" | grep -v test_

# Allowed exceptions (grep for these to confirm they're the only hits):
# - "Monte Carlo" (real statistics)
# - "Ornstein-Uhlenbeck" (real stochastic process)
# - "erfi" (real math function)
# - "Hurst" (real fractal analysis)
# - "entropy" (Shannon entropy)
# - "fractal" (real math)
```

### 2. Run existing tests

```bash
python -m pytest test_phase1.py -v
# Plus any other test files in the project
```

### 3. Import check

```bash
python -c "from core.market_state import MarketState; print('OK')"
python -c "from core.statistical_field_engine import StatisticalFieldEngine; print('OK')"
python -c "from core.cuda_statistics import compute_regression_kernel; print('OK')"
python -c "from core.bayesian_brain import MarketBayesianBrain; print('OK')"
```

### 4. Checkpoint compatibility

**WARNING:** Pickled checkpoints (brain, pattern_library) contain old class names.
After renaming `QuantumBayesianBrain` → `MarketBayesianBrain`, loading old
pickle files will fail with `AttributeError: Can't get attribute 'QuantumBayesianBrain'`.

**Fix:** Add backward-compat alias at module level:

```python
# core/bayesian_brain.py — bottom of file
# Backward compatibility for pickled checkpoints
QuantumBayesianBrain = MarketBayesianBrain
```

Do the same for any renamed classes that appear in pickle files.

Similarly, `MarketState` was already renamed from something in Phase 4 —
verify the alias still exists in the new `core/market_state.py`.

### 5. MarketState hash stability

**CRITICAL:** Renaming `band_zone` values (e.g., `'CHAOS'` → `'TRANSITION'`)
changes the hash of MarketState objects. This invalidates the entire
BayesianBrain probability table — every state key becomes a miss.

**Mitigation options:**
- **Option A (recommended):** Do NOT rename band_zone string values. Keep
  `UPPER_EXTREME`, `LOWER_EXTREME`, `INNER`, `CHAOS` as-is. They're internal
  enum-like values that don't appear in user-facing output. Document that
  `'CHAOS'` means "transition zone between inner and extreme" in a comment.
- **Option B:** Rename values AND retrain from scratch (resets brain).
- **Option C:** Write a migration script that re-keys the brain table.

**Decision: Use Option A.** Mark `CHAOS` with an inline comment:
```python
# 'CHAOS' = transition zone (1σ < |z| < 2σ) — name is historical, do not rename
# (renaming would invalidate all BayesianBrain hash keys)
```

Same logic applies to `market_regime = 'CHAOTIC'` → keep as `'CHAOTIC'` or
accept brain invalidation.

---

## Dead Code to Delete

While doing the purge, remove these confirmed dead items:

| Item | File | Reason |
|------|------|--------|
| `QuantumBayesianBrain.get_quantum_probability()` | `core/bayesian_brain.py` | Wrapper for `get_probability()` — adds nothing |
| `QuantumBayesianBrain.should_fire_quantum()` | `core/bayesian_brain.py` | Hardcoded physics checks that duplicate gate logic in ExecutionEngine |
| `self.TIDAL_FORCE_EXPONENT` | `core/quantum_field_engine.py` | Set to 2.0, never read |
| `VELOCITY_CASCADE_THRESHOLD` | `core/quantum_field_engine.py` | Never read in current code |
| `RANGE_CASCADE_THRESHOLD` | `core/quantum_field_engine.py` | Never read in current code |
| `self.risk_engine` (MonteCarloRiskEngine instance) | `core/quantum_field_engine.py` | Instantiated in `__init__` but never called — OU analytical solution replaced it |
| `_compute_rs_numba` (commented out non-parallel version) | `core/quantum_field_engine.py` | Dead commented code above the active `@numba.njit(parallel=True)` version |
| `MarketState.get_trade_directive()` | `core/three_body_state.py` | Returns trade signals based on hardcoded z-score thresholds — superseded by ExecutionEngine gate cascade. **Evaluate:** Is this called anywhere? If not, delete. |

---

## Summary Stats

| Category | Count |
|----------|-------|
| Fields renamed | 7 |
| Functions renamed | 5 |
| Classes renamed | 1 |
| Constants renamed | 12 |
| String literals renamed | 2 (resonance_type, semantic_name) |
| String literals KEPT (hash stability) | 3 (CHAOS, CHAOTIC, CRITICAL) |
| Files renamed | 3 |
| Dead code deleted | 7+ items |
| Docstrings rewritten | 6 major blocks |
| Comments updated | ~30 inline comments |

**Estimated effort:** 1-2 hours for Claude Code with this spec (mechanical find-and-replace, no logic changes).

**Risk:** LOW — all renames are cosmetic. No algorithmic logic changes.
The only risk is pickle compatibility (mitigated by backward-compat aliases).
