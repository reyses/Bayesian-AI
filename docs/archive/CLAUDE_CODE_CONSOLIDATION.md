# CLAUDE CODE SPEC: Code Consolidation & Dead Code Removal

## CONFIDENTIAL — BayesianBridge Internal

---

## Scope

Eliminate duplicate logic, dead code, and unnecessary abstractions across the
codebase. Every item below is either: (A) logic that exists in 2+ places and
must stay in sync manually, or (B) code that is never called in production.

**Rule:** If removing/merging something requires retraining, it gets flagged
and deferred. The goal is zero behavioral changes — same outputs, less code.

---

## PRIORITY 1: Delete Dead Code (Zero Risk)

These items are never called in the production pipeline. Deleting them reduces
surface area with zero behavioral impact.

### 1.1 — `MonteCarloRiskEngine` instantiation

**File:** `core/quantum_field_engine.py` (→ `core/statistical_field_engine.py` after metaphor purge)

**What:** `StatisticalFieldEngine.__init__()` creates `self.risk_engine = MonteCarloRiskEngine(...)`.
This instance is never called — the OU analytical solution (`erfi`) replaced
Monte Carlo simulation for reversion/breakout probabilities.

**Action:** Delete `self.risk_engine` instantiation AND the import of
`MonteCarloRiskEngine`. Keep `core/risk_engine.py` as a file (it may be used
by `training/integrated_statistical_system.py` or future validation).

```python
# DELETE from __init__():
from core.risk_engine import MonteCarloRiskEngine  # ← delete import
self.risk_engine = MonteCarloRiskEngine(            # ← delete block
    theta=RISK_THETA,
    horizon_seconds=RISK_HORIZON_SECONDS
)
```

Also delete the now-orphaned constants:
```python
RISK_THETA = 0.1              # ← delete
RISK_HORIZON_SECONDS = 600    # ← delete
```

**Lines removed:** ~8

---

### 1.2 — `QuantumBayesianBrain` subclass methods

**File:** `core/bayesian_brain.py`

**What:** `QuantumBayesianBrain` adds two methods:
- `get_quantum_probability()` — literally calls `self.get_probability(state)`. One-line wrapper.
- `should_fire_quantum()` — hardcodes physics checks (`band_zone`, `structure_confirmed`,
  `cascade_detected`, `F_momentum > mean_reversion_force * 1.5`) then calls base `should_fire()`.
  These checks are ALL handled by `ExecutionEngine` gates 0-4. This method is not called
  by any production code path.

**Action:** Delete both methods. Keep the class (renamed to `MarketBayesianBrain`
per metaphor purge) as an empty subclass for type compatibility, OR merge into
`BayesianBrain` directly.

**Preferred:** Delete the subclass entirely. `LiveEngine` creates it as
`self._brain = QuantumBayesianBrain()` but only ever calls base class methods
(`update`, `save`, `load`, `should_fire`, `get_probability`, `direction_learn`,
`get_dir_bias`, `get_dir_probability`). Change to `self._brain = BayesianBrain()`.

```python
# DELETE entire class:
class QuantumBayesianBrain(BayesianBrain):
    ...  # ~40 lines
```

**Files to update:**
- `live/live_engine.py`: `QuantumBayesianBrain()` → `BayesianBrain()`
- Any training files that instantiate `QuantumBayesianBrain`

**Lines removed:** ~40

---

### 1.3 — `MarketState.get_trade_directive()`

**File:** `core/three_body_state.py` (→ `core/market_state.py`)

**What:** Returns trade signals based on hardcoded z-score thresholds, Hurst filter,
band_zone checks, and regime filters. This was the original Phase 1 decision logic.
`ExecutionEngine` completely supersedes it with a configurable gate cascade.

**Action:** Grep for callers first:
```bash
grep -rn "get_trade_directive" --include="*.py" | grep -v __pycache__
```
If zero callers → delete. If callers exist → trace and eliminate them.

**Lines removed:** ~40

---

### 1.4 — Scalar pattern detection functions

**File:** `core/pattern_utils.py`

**What:** Two single-bar functions that process only the LAST bar:
- `detect_geometric_pattern(highs, lows)` → returns single string
- `detect_candlestick_pattern(opens, highs, lows, closes)` → returns single string

The production pipeline uses ONLY the vectorized versions
(`detect_geometric_patterns_vectorized`, `detect_candlestick_patterns_vectorized`)
or the CUDA kernel (`detect_patterns_cuda`). The scalar versions exist from
Phase 1 prototyping.

**Action:** Grep for callers:
```bash
grep -rn "detect_geometric_pattern\b\|detect_candlestick_pattern\b" --include="*.py" | grep -v vectorized | grep -v cuda | grep -v __pycache__
```
If only imported by `quantum_field_engine.py` and that import is unused → delete.

Check: `quantum_field_engine.py` imports both but only calls `_detect_patterns_unified()`
which dispatches to CUDA or vectorized. The scalar functions are imported but never called.

**Lines removed:** ~60

---

### 1.5 — `detect_geometric_patterns_cuda()` legacy wrapper

**File:** `core/cuda_pattern_detector.py`

**What:** Passes dummy open/close arrays and calls the unified kernel, returning
only geometric patterns. Comment says "Legacy wrapper for backward compatibility."

**Action:** Grep for callers. If zero → delete.

**Lines removed:** ~8

---

### 1.6 — Commented-out `_compute_rs_numba` (non-parallel version)

**File:** `core/quantum_field_engine.py`

**What:** ~30 lines of commented-out code above the active `@numba.njit(parallel=True)`
version. The comment says the parallel version is a 3.7x speedup.

**Action:** Delete the commented block entirely.

**Lines removed:** ~30

---

### 1.7 — Dead imports and unused variables

**File:** `core/quantum_field_engine.py`

```python
# These are imported but never used in the current code:
try:
    import matplotlib              # ← never used in this module
    matplotlib.use('Agg')
    import matplotlib.pyplot       # ← never used
except ImportError:
    pass

try:
    import pandas_ta as ta         # ← PANDAS_TA_AVAILABLE set but never checked
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    from hurst import compute_Hc   # ← HURST_AVAILABLE set but never checked
    HURST_AVAILABLE = True          #    (custom _compute_hurst_numpy used instead)
except ImportError:
    HURST_AVAILABLE = False

# Torch import — only used for self.device which is set but never read in batch path
try:
    import torch
    import torch.nn.functional as F    # ← F never used
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

**Action:** Delete all four import blocks. Delete `self.device` from `__init__`.
Delete `PANDAS_TA_AVAILABLE`, `HURST_AVAILABLE`, `TORCH_AVAILABLE` flags
and any conditionals that check them (verify none exist first).

**Lines removed:** ~25

---

### 1.8 — `StateVector` class

**File:** `core/state_vector.py`

**What:** Phase 1 prototype. 9-layer immutable state with discrete categorical layers
(`L1_bias: str = 'bull'`, `L7_pattern: str = 'flag'`, etc.). `MarketState` replaced
it entirely with continuous physics fields + dynamic binning.

`BayesianBrain` imports `StateVector` in its type hints:
```python
from core.state_vector import StateVector
```
`TradeOutcome.state` type is `Union[StateVector, MarketState, str, int]`.

But in production, `TradeOutcome.state` is ALWAYS either a `template_id` (int/str)
or a `MarketState`. No code creates `StateVector` objects anymore.

**Action:**
1. Remove `StateVector` from `TradeOutcome.state` union type
2. Remove `from core.state_vector import StateVector` from bayesian_brain.py
3. Keep `core/state_vector.py` file for now (test_phase1.py references it) but
   mark it as `# DEPRECATED — Phase 1 prototype, not used in production`

**Lines changed:** ~3 (import + type hint)

---

### 1.9 — Unused constants in `quantum_field_engine.py`

```python
VELOCITY_CASCADE_THRESHOLD = 1.0  # Never read
RANGE_CASCADE_THRESHOLD = 10.0    # Never read
ADX_LENGTH = ADX_PERIOD           # Alias for ADX_PERIOD — never used (ADX_PERIOD used directly)
```

**Action:** Delete all three.

**Lines removed:** 3

---

**Total Priority 1 deletions: ~260 lines of dead code**

---

## PRIORITY 2: Merge Duplicate Logic (Low Risk)

These are places where the same computation exists in 2+ places and must be
manually kept in sync. Merging them into a single source of truth prevents
drift bugs.

### 2.1 — Feature extraction: 2 implementations → 1

**Files:**
- `core/fractal_clustering.py` → `FractalClusteringEngine.extract_features()` (static method)
- `core/timeframe_belief_network.py` → `TimeframeBeliefNetwork.state_to_features()` (static method)

**What:** Both produce a 16D feature vector. TBN's version says in its docstring:
"Same order as FractalClusteringEngine.extract_features()."

But they're NOT identical:
- `extract_features(p)` takes a PatternEvent with `.z_score`, `.velocity`, `.momentum`,
  `.entropy_normalized`, `.parent_chain`, `.state.adx_strength`, etc.
- `state_to_features(state, tf_secs, depth)` takes a MarketState + tf_secs + depth,
  and sets ancestry features to 0.0 (no parent chain in live).

The difference is the INPUT type, not the computation. Both produce the same
16D vector when given the same values.

**Action:** Create a single `extract_features_from_values()` function that takes
raw values (not objects). Both methods become thin wrappers that unpack their
respective input types and call the shared function.

```python
# core/feature_extraction.py (NEW FILE — ~40 lines)

import numpy as np
from training.fractal_discovery_agent import TIMEFRAME_SECONDS

def extract_feature_vector(
    z_score: float, velocity: float, momentum: float,
    entropy_normalized: float, tf_seconds: int, depth: float,
    parent_is_roche: float,
    adx: float, hurst: float, dmi_diff: float,
    parent_z: float, parent_dmi_diff: float,
    root_is_roche: float, tf_alignment: float,
    pid: float, osc_coherence: float,
) -> list[float]:
    """Canonical 16D feature vector. Single source of truth.
    
    Both FractalClusteringEngine.extract_features() and
    TimeframeBeliefNetwork.state_to_features() delegate here.
    """
    v_feat = np.log1p(abs(velocity))
    m_feat = np.log1p(abs(momentum))
    tf_scale = np.log2(max(1, tf_seconds))
    
    return [
        abs(z_score), v_feat, m_feat, entropy_normalized,
        tf_scale, depth, parent_is_roche,
        adx, hurst, dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
        pid, osc_coherence,
    ]
```

```python
# core/fractal_clustering.py — refactored extract_features()
from core.feature_extraction import extract_feature_vector

@staticmethod
def extract_features(p) -> list[float]:
    """Extract 16D features from a PatternEvent."""
    state = getattr(p, 'state', None)
    chain = getattr(p, 'parent_chain', None) or []
    
    # Unpack ancestry
    if chain:
        parent_z = abs(chain[0].get('z', 0.0))
        parent_dmi = (chain[0].get('dmi_plus', 0.0) - chain[0].get('dmi_minus', 0.0)) / 100.0
        root = chain[-1]
        root_is_roche = 1.0 if root.get('type') == 'BAND_REVERSAL' else 0.0
        root_dmi = (root.get('dmi_plus', 0.0) - root.get('dmi_minus', 0.0)) / 100.0
        self_dmi = (getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0
        self_dir = 1.0 if self_dmi > 0 else -1.0
        root_dir = 1.0 if root_dmi > 0 else -1.0
        tf_alignment = self_dir * root_dir
    else:
        parent_z = parent_dmi = root_is_roche = tf_alignment = 0.0
    
    tf = getattr(p, 'timeframe', '15s')
    tf_secs = TIMEFRAME_SECONDS.get(tf, 15)
    
    return extract_feature_vector(
        z_score=getattr(p, 'z_score', 0.0),
        velocity=getattr(p, 'velocity', 0.0),
        momentum=getattr(p, 'momentum', 0.0),
        entropy_normalized=getattr(p, 'entropy_normalized', 0.0),
        tf_seconds=tf_secs,
        depth=float(getattr(p, 'depth', 0)),
        parent_is_roche=1.0 if getattr(p, 'parent_type', '') == 'BAND_REVERSAL' else 0.0,
        adx=getattr(state, 'adx_strength', 0.0) / 100.0 if state else 0.0,
        hurst=getattr(state, 'hurst_exponent', 0.5) if state else 0.5,
        dmi_diff=(getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0 if state else 0.0,
        parent_z=parent_z,
        parent_dmi_diff=parent_dmi if chain else 0.0,
        root_is_roche=root_is_roche,
        tf_alignment=tf_alignment,
        pid=getattr(state, 'term_pid', 0.0) if state else 0.0,
        osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0) if state else 0.0,
    )
```

```python
# core/timeframe_belief_network.py — refactored state_to_features()
from core.feature_extraction import extract_feature_vector

@staticmethod
def state_to_features(state, tf_secs: int, depth: int = 0) -> list:
    """Extract 16D features from a MarketState (no ancestry)."""
    return extract_feature_vector(
        z_score=getattr(state, 'z_score', 0.0),
        velocity=getattr(state, 'velocity', 0.0),
        momentum=getattr(state, 'momentum_strength', 0.0),
        entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
        tf_seconds=tf_secs,
        depth=float(depth),
        parent_is_roche=0.0,    # no parent chain in live
        adx=getattr(state, 'adx_strength', 0.0) / 100.0,
        hurst=getattr(state, 'hurst_exponent', 0.5),
        dmi_diff=(getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0,
        parent_z=0.0,
        parent_dmi_diff=0.0,
        root_is_roche=0.0,
        tf_alignment=0.0,
        pid=getattr(state, 'term_pid', 0.0),
        osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0),
    )
```

**Bug this catches:** TBN reads `state.momentum_strength` while clustering reads
`p.momentum`. If MarketState.momentum_strength != PatternEvent.momentum (which
depends on how events are constructed), live features won't match training
features. The shared function makes this explicit.

**Lines:** ~80 lines in new file, ~60 lines removed from the two callers. Net: ~+20 but
eliminates a class of silent drift bugs.

---

### 2.2 — Exit sizing: 2 implementations → 1

**Files:**
- `core/execution_engine.py` → `_compute_sizing()` — uses template library stats
- `core/exit_engine.py` → `open_position()` — ALSO computes SL/TP from the same library stats

**What:** Both read `p25_mae`, `mean_mae`, `regression_sigma`, `p75_mfe` from
`lib_entry` and compute SL/TP ticks with the same cascade logic. If you change
the multiplier in one, you must remember to change the other.

The flow is: `ExecutionEngine._compute_sizing()` computes SL/TP → passes to
`exec_engine.position_opened(sl_ticks=...)` → which calls
`ExitEngine.open_position()` → which RECOMPUTES SL/TP from the same lib_entry
and then gets OVERWRITTEN by the caller's values. The ExitEngine computation
is wasted work that could silently diverge.

**Action:** Remove SL/TP/trail computation from `ExitEngine.open_position()`.
Make it accept pre-computed values only. Single source = `ExecutionEngine._compute_sizing()`.

```python
# core/exit_engine.py — simplified open_position()
def open_position(self, side, entry_price, entry_bar_index, template_id,
                  sl_ticks: float, tp_ticks: float,
                  trail_ticks: float = 0.0, trail_activation_ticks: float = 0.0,
                  max_hold_bars: int = 120,
                  network_tp: float = None) -> PositionState:
    """Initialize position with pre-computed exit parameters.
    
    Sizing is computed by ExecutionEngine._compute_sizing() — this method
    does NOT recompute. It only sets up position state and absolute price levels.
    """
    pos = PositionState(
        side=side, entry_price=entry_price,
        entry_bar_index=entry_bar_index,
        template_id=template_id,
        tick_size=self.tick_size, tick_value=self.tick_value,
        sl_ticks=sl_ticks, tp_ticks=tp_ticks,
        trailing_stop_ticks=trail_ticks,
        trail_activation_ticks=trail_activation_ticks,
        max_hold_bars=max_hold_bars,
    )
    
    # Compute absolute price levels from tick values
    if side == 'long':
        pos.current_trail = entry_price - (sl_ticks * self.tick_size)
        pos.peak_favorable = entry_price
        pos.stop_loss = entry_price - (sl_ticks * self.tick_size)
        pos.profit_target = entry_price + (tp_ticks * self.tick_size) if tp_ticks > 0 else 0.0
    else:
        pos.current_trail = entry_price + (sl_ticks * self.tick_size)
        pos.peak_favorable = entry_price
        pos.stop_loss = entry_price + (sl_ticks * self.tick_size)
        pos.profit_target = entry_price - (tp_ticks * self.tick_size) if tp_ticks > 0 else 0.0
    
    # Envelope initialization
    pos.envelope_active = True
    pos.envelope_level = tp_ticks * self.tick_size
    pos.envelope_T0 = float(sl_ticks)
    pos.high_water_mark = entry_price
    pos.original_trail_ticks = trail_ticks
    
    import time as _time
    pos.entry_time = _time.time()
    
    return pos
```

**Lines removed:** ~50 (the entire SL/TP cascade in `open_position()` that gets
overwritten anyway).

---

### 2.3 — `make_position()` free function → delete

**File:** `core/exit_engine.py`

**What:** `make_position()` is a free function that creates `PositionState` objects.
It's used ONLY by `LiveEngine` for manual trades and ping-pong flips. It duplicates
`ExitEngine.open_position()` with a slightly different interface.

After 2.2 simplifies `open_position()`, `make_position()` becomes fully redundant.

**Action:** Delete `make_position()`. Change all callers to use
`exit_engine.open_position()` with pre-computed sizing.

**Callers to update:**
- `live/live_engine.py` → `_flip_position()`, `_enter_ping_pong()`,
  `_execute_manual_entry()`, `_auto_tp_reentry()`, `_check_entry()` (all call `make_position`)

**Lines removed:** ~30

---

### 2.4 — Physics constants: 2 copies → 1

**Files:**
- `core/cuda_physics.py`: module-level constants `GRAVITY_THETA = 0.5`, etc.
- `core/quantum_field_engine.py`: instance attributes `self.GRAVITY_THETA = 0.5`, etc.

**What:** The CUDA kernel reads module-level constants (baked in at compile time).
The CPU fallback reads `self.*` instance attributes. Both must match.

**Action:** Delete instance attributes from `StatisticalFieldEngine.__init__()`.
Import constants from `cuda_physics.py` (→ `cuda_statistics.py`) and use them
in the CPU path too.

```python
# core/statistical_field_engine.py (after metaphor purge rename)
from core.cuda_statistics import (
    REVERSION_THETA, BAND_PRESSURE_EPSILON, BAND_PRESSURE_CAP,
    VELOCITY_THRESHOLD, MOMENTUM_THRESHOLD, ENTROPY_THRESHOLD,
    SIGMA_EXTREME, SIGMA_BREAKOUT,
)

class StatisticalFieldEngine:
    def __init__(self, ...):
        # DELETE all self.GRAVITY_THETA, self.REPULSION_*, etc.
        # Use module constants from cuda_statistics directly
        self.SIGMA_EXTREME_MULTIPLIER = SIGMA_EXTREME  # just reference, not redefine
        ...
```

**Lines removed:** ~15

---

### 2.5 — CPU/CUDA path sync: document, don't merge

**Files:**
- `core/cuda_physics.py` → `compute_physics_kernel` (CUDA)
- `core/quantum_field_engine.py` → `_batch_compute_cpu()` (NumPy)

**What:** Both compute the same physics: regression, z-score, forces, probabilities.
CUDA uses a per-element kernel. CPU uses vectorized convolution + numpy ops.
They MUST produce identical results.

**Action:** Do NOT merge (different compute paradigms). Instead:
1. Add a header comment to both linking them:
   ```python
   # SYNC NOTICE: This computation must match _batch_compute_cpu() in
   # statistical_field_engine.py exactly. Any change here must be mirrored there.
   ```
2. Add a test that runs both paths on the same input and asserts outputs match
   within floating-point tolerance:
   ```python
   # tests/test_cpu_cuda_parity.py
   def test_physics_parity():
       """CUDA and CPU paths must produce identical results."""
       df = load_test_data()
       engine = StatisticalFieldEngine()
       cuda_results = engine.batch_compute_states(df, use_cuda=True)
       cpu_results = engine.batch_compute_states(df, use_cuda=False)
       for i in range(len(cuda_results)):
           assert_state_equal(cuda_results[i]['state'], cpu_results[i]['state'], atol=1e-4)
   ```

**Lines added:** ~30 (test + comments). No code removed (intentional — separate implementations
serve different hardware).

---

### 2.6 — Hurst: 2 implementations but CUDA is primary

**Files:**
- `core/quantum_field_engine.py` → `_compute_hurst_numpy()` (CPU fallback)
- `core/cuda_physics.py` → `compute_hurst_kernel` (CUDA primary)

**What:** Different algorithms. CUDA uses 4 sub-windows with log(R/S) regression.
NumPy uses sliding_window_view with pinv regression. Both produce Hurst [0,1] but
may give slightly different values.

**Action:** Same as 2.5 — add sync notice + parity test. Don't merge.

---

## PRIORITY 3: Simplify Abstractions (Medium Risk)

### 3.1 — `PositionState` has too many fields

**File:** `core/exit_engine.py`

**What:** `PositionState` has 30+ fields including backward-compat aliases:
- `bars_held` AND `bars_in_trade` (sync'd every bar)
- `peak_favorable` AND `high_water_mark` (sync'd every bar)
- `current_trail` AND `stop_loss` (separate but confused — trail moves, SL is fixed)
- `entry_time` (set by `time.time()` — wall clock, not bar time)
- `entry_layer_state` (set to None or MarketState — never read by ExitEngine)
- `entry_dmi_inverse` (never set to True by any code path)
- `last_adjustment_reason` (never read)
- `breakeven_level` (never set, never read)
- CST fields: `cst_centroid`, `cst_basin_mean`, `cst_basin_std`, `cst_ancestry` — only set by `make_position()` (being deleted in 2.3). Never read by ExitEngine.

**Action:** Delete unused fields. Consolidate aliases.

```python
# Fields to DELETE from PositionState:
entry_layer_state: object = None        # never read by ExitEngine
entry_dmi_inverse: bool = False         # never set to True
last_adjustment_reason: str = ''        # never read
breakeven_level: float = 0.0            # never set, never read
cst_centroid: object = None             # only set by make_position (being deleted)
cst_basin_mean: float = 0.0            # same
cst_basin_std: float = 0.0             # same
cst_ancestry: object = None             # same

# Fields to MERGE (keep one, delete alias):
bars_in_trade → DELETE (keep bars_held, set by ExitEngine.evaluate())
high_water_mark → DELETE (keep peak_favorable, set by ExitEngine.evaluate())
```

**Lines removed:** ~15 field definitions + any code that reads/writes deleted fields.

---

### 3.2 — `BayesianBrain.should_fire_validated()` — evaluate for deletion

**File:** `core/bayesian_brain.py`

**What:** ~60-line method that wraps `should_fire()` with `BayesianStateValidator`
and `MonteCarloRiskAnalyzer` from `training/integrated_statistical_system.py`.
Uses optional imports with try/except.

**Question:** Is this called anywhere in the production pipeline?

```bash
grep -rn "should_fire_validated" --include="*.py" | grep -v __pycache__
```

If called only by training validation scripts → keep but move to a validation
utility. If never called → delete.

**Lines removed if deleted:** ~60

---

### 3.3 — `BayesianBrain.get_all_states_above_threshold()` — evaluate

**File:** `core/bayesian_brain.py`

**What:** Analysis helper that finds all high-probability states. Used in
reports or interactive analysis.

**Action:** Keep if used in training reports. Move to a separate analysis
utility if desired. Low priority.

---

## PRIORITY 4: Structural Improvements (Implements with Other Specs)

These overlap with the Compressed Replay + Thin Wrapper spec and should be
done in that pass:

### 4.1 — Direction cascade unification (covered in CLAUDE_CODE_COMPRESSED_REPLAY.md)
- Delete `LiveEngine._determine_direction()`
- Add `live_momentum` + `live_bias` to `ExecutionEngine._direction_cascade()`
- LiveEngine delegates to ExecutionEngine

### 4.2 — Gate cascade unification (covered in CLAUDE_CODE_COMPRESSED_REPLAY.md)
- Delete `LiveEngine._check_entry()` inline gates
- LiveEngine delegates to `ExecutionEngine.on_bar()`

### 4.3 — Exit param computation (covered above in 2.2 + 2.3)
- Delete `LiveEngine._compute_exit_params()`
- Delete `make_position()`
- Simplify `ExitEngine.open_position()`

---

## Files Created

| File | Purpose |
|------|---------|
| `core/feature_extraction.py` | Single source of truth for 16D feature vector |
| `tests/test_cpu_cuda_parity.py` | Validates CUDA and CPU paths produce identical outputs |

## Files Modified

| File | Changes |
|------|---------|
| `core/bayesian_brain.py` | Delete `QuantumBayesianBrain` class (or merge), delete `should_fire_validated` if unused, remove `StateVector` import |
| `core/exit_engine.py` | Simplify `open_position()` (accept pre-computed sizing), delete `make_position()`, trim `PositionState` fields |
| `core/quantum_field_engine.py` | Delete: MC risk engine, dead imports, commented code, unused constants. Import physics constants from cuda module. |
| `core/pattern_utils.py` | Delete scalar `detect_geometric_pattern()` and `detect_candlestick_pattern()` if unused |
| `core/cuda_pattern_detector.py` | Delete `detect_geometric_patterns_cuda()` legacy wrapper if unused |
| `core/fractal_clustering.py` | Refactor `extract_features()` to delegate to `core/feature_extraction.py` |
| `core/timeframe_belief_network.py` | Refactor `state_to_features()` to delegate to `core/feature_extraction.py` |
| `live/live_engine.py` | `QuantumBayesianBrain` → `BayesianBrain`, delete `_compute_exit_params()`, replace `make_position()` calls |

---

## Implementation Order

**Phase 1 — Dead code (zero risk, do first):**
1. Delete `MonteCarloRiskEngine` instantiation + orphaned constants
2. Delete `QuantumBayesianBrain` subclass (update all instantiation sites)
3. Delete `get_trade_directive()` if unused
4. Delete scalar pattern functions if unused  
5. Delete legacy CUDA wrapper if unused
6. Delete commented-out numba code
7. Delete dead imports (matplotlib, pandas_ta, hurst, torch)
8. Delete unused constants
9. Remove `StateVector` from `TradeOutcome` type hint

**Phase 2 — Merge duplicates (low risk):**
10. Create `core/feature_extraction.py`, refactor both callers
11. Simplify `ExitEngine.open_position()` to accept pre-computed sizing
12. Delete `make_position()`, update all callers
13. Consolidate physics constants (single source in cuda module)
14. Trim `PositionState` unused fields

**Phase 3 — Sync guards (no code risk, adds safety):**
15. Add sync-notice comments to CUDA/CPU dual implementations
16. Write `test_cpu_cuda_parity.py`

**Phase 4 — Done in other specs (reference only):**
17. Direction cascade unification → CLAUDE_CODE_COMPRESSED_REPLAY.md
18. Gate cascade unification → CLAUDE_CODE_COMPRESSED_REPLAY.md
19. LiveEngine._compute_exit_params() deletion → follows from 11-12

---

## Verification

### Before:
```bash
find . -name "*.py" -not -path "./__pycache__/*" | xargs wc -l | tail -1
# Record total line count
```

### After:
```bash
# Same command — expect ~400-500 fewer lines
# Grep for deleted symbols:
grep -rn "QuantumBayesianBrain\|get_trade_directive\|make_position\|detect_geometric_pattern[^s]\|RISK_THETA\|VELOCITY_CASCADE" --include="*.py" | grep -v __pycache__ | grep -v "# DEPRECATED"
# Should return ZERO results (except backward-compat aliases)
```

### Behavioral verification:
```bash
# Run training pipeline — outputs must be identical
python -m training.orchestrator --phase 4 --days 1
# Compare: WR, PnL, trade count, gate skip counts must match pre-refactor
```

---

## Summary

| Category | Items | Lines Removed |
|----------|-------|---------------|
| Dead code deletion | 9 items | ~260 |
| Duplicate merge | 6 items | ~95 |
| Field cleanup | 10 fields | ~15 |
| Sync guards added | 2 items | +30 |
| Feature extraction module | 1 new file | +40 |
| **Net** | | **~-300 lines** |

Combined with the Thin Wrapper spec (-250 lines from LiveEngine), total
consolidation across both specs: **~-550 lines** and significantly fewer
places where logic can silently diverge.
