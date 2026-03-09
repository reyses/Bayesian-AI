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

## PRIORITY 5: LiveEngine Decomposition (Implementable Now)

### 5.1 — Split LiveEngine into focused modules

**File:** `live/live_engine.py` — currently 1,400+ lines, 50+ instance variables.

After the thin wrapper (CLAUDE_CODE_COMPRESSED_REPLAY.md) removes ~250 lines
of inline gates, ~1,100 remain. Still a monolith. One bug in any subsystem
crashes the entire live loop. No subsystem is independently testable.

**Action:** Extract four focused modules. LiveEngine becomes a ~350 line
orchestrator that wires them together.

#### Module A: `live/session_tracker.py` (~200 lines extracted)

**Owns:** Session statistics, trade log, exit capture buckets, drawdown
tracking, equity curve, and session report generation.

```python
# live/session_tracker.py

from dataclasses import dataclass, field
from collections import defaultdict
import os, time

@dataclass
class SessionStats:
    pnl: float = 0.0
    wins: int = 0
    trades: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    consec_losses: int = 0
    consec_loss_dollars: float = 0.0
    max_consec_losses: int = 0
    max_consec_loss_dollars: float = 0.0
    consec_wins: int = 0
    peak_equity: float = 0.0
    session_drawdown: float = 0.0
    max_session_drawdown: float = 0.0
    exit_buckets: dict = field(default_factory=lambda: {
        'optimal': 0, 'partial': 0, 'early': 0, 'reversed': 0})

class SessionTracker:
    """Tracks session PnL, drawdowns, trade log, and writes reports."""
    
    def __init__(self, config):
        self.stats = SessionStats()
        self.trade_log: list[dict] = []
        self._cfg = config
        self._start_time = time.time()
    
    def record_trade(self, pnl: float, trade_info: dict):
        """Record a completed trade. Updates all stats atomically."""
        s = self.stats
        s.pnl += pnl
        s.trades += 1
        
        if pnl > 0:
            s.wins += 1
            s.gross_win += pnl
            s.consec_wins += 1
            s.consec_losses = 0
            s.consec_loss_dollars = 0.0
        else:
            s.gross_loss += pnl
            s.consec_losses += 1
            s.consec_loss_dollars += abs(pnl)
            s.consec_wins = 0
            s.max_consec_losses = max(s.max_consec_losses, s.consec_losses)
            s.max_consec_loss_dollars = max(s.max_consec_loss_dollars, s.consec_loss_dollars)
        
        if s.pnl > s.peak_equity:
            s.peak_equity = s.pnl
        s.session_drawdown = s.peak_equity - s.pnl
        s.max_session_drawdown = max(s.max_session_drawdown, s.session_drawdown)
        
        # Capture bucket (MFE-based exit quality)
        mfe_ticks = trade_info.get('mfe_ticks', 0)
        pnl_ticks = trade_info.get('pnl_ticks', 0)
        if mfe_ticks > 0:
            capture = pnl_ticks / mfe_ticks * 100
        else:
            capture = 0.0 if pnl <= 0 else 100.0
        if capture >= 80:   s.exit_buckets['optimal'] += 1
        elif capture >= 20: s.exit_buckets['partial'] += 1
        elif capture > 0:   s.exit_buckets['early'] += 1
        else:               s.exit_buckets['reversed'] += 1
        
        trade_info['capture'] = capture
        self.trade_log.append(trade_info)
    
    @property
    def win_rate(self) -> float:
        return self.stats.wins / self.stats.trades if self.stats.trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        return self.stats.gross_win / abs(self.stats.gross_loss) if self.stats.gross_loss != 0 else 0.0
    
    def write_report(self, gate_stats: dict, brain_dir_bias: dict,
                     account_snapshot: dict, pp_flip_count: int = 0) -> str:
        """Write session report to reports/live/. Returns path."""
        # Entire _write_session_report() body moves here unchanged.
        # Accepts gate_stats from ExecutionEngine.get_skip_counts(),
        # brain_dir_bias from brain.dir_bias, account from NT8.
        ...
```

**Moves from LiveEngine:**
- `self._session_pnl`, `self._session_wins`, `self._session_trades` → `SessionStats`
- `self._gross_win`, `self._gross_loss` → `SessionStats`
- `self._consec_losses`, `self._consec_loss_dollars`, `self._max_consec_losses`, etc. → `SessionStats`
- `self._peak_equity`, `self._session_drawdown`, `self._max_session_drawdown` → `SessionStats`
- `self._exit_buckets` → `SessionStats`
- `self._trade_log` → `SessionTracker.trade_log`
- `self._session_start` → `SessionTracker._start_time`
- `_write_session_report()` → `SessionTracker.write_report()`
- `_brain_learn()` stat-tracking portion → `SessionTracker.record_trade()`

**LiveEngine usage:**
```python
self._session = SessionTracker(self._cfg)

# On trade close:
self._session.record_trade(pnl, trade_info)

# On shutdown:
self._session.write_report(
    gate_stats=self._exec_engine.get_skip_counts(),
    brain_dir_bias=self._brain.dir_bias,
    account_snapshot={...},
)
```

---

#### Module B: `live/gui_bridge.py` (~120 lines extracted)

**Owns:** All GUI queue interactions, belief bar computation, stat formatting.

```python
# live/gui_bridge.py

import time
from typing import Optional
import queue

class GUIBridge:
    """Non-blocking bridge between LiveEngine and the Tk popup.
    
    Encapsulates all GUI message formatting and throttling.
    LiveEngine never touches gui_queue directly.
    """
    
    def __init__(self, gui_queue: Optional[queue.Queue] = None):
        self._q = gui_queue
        self._last_push = 0.0
    
    def push(self, msg: dict):
        """Non-blocking push. Drops if full or no GUI."""
        if self._q is None:
            return
        try:
            self._q.put_nowait(msg)
        except Exception:
            pass
    
    def push_tick(self, price: float, bars: int):
        self.push({'type': 'TICK_UPDATE', 'price': price, 'bars': bars})
    
    def push_trade_marker(self, action: str, side: str, price: float, pnl: float = 0.0):
        msg = {'type': 'TRADE_MARKER', 'action': action, 'side': side, 'price': price}
        if pnl:
            msg['pnl'] = pnl
        self.push(msg)
    
    def push_stats(self, session_stats, exec_gate_stats: dict,
                   belief_pct: float, in_position: bool,
                   daily_pnl: float):
        """Throttled stats push — max 1/second."""
        now = time.time()
        if now - self._last_push < 1.0:
            return
        self._last_push = now
        
        s = session_stats
        eb = s.exit_buckets
        wr = s.wins / s.trades * 100 if s.trades > 0 else 0.0
        pf = s.gross_win / abs(s.gross_loss) if s.gross_loss != 0 else 0.0
        
        _bar_label = f'life {belief_pct:.0f}%' if in_position else f'belief {belief_pct:.0f}%'
        
        self.push({
            'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
            'step': _bar_label, 'pct': belief_pct,
            'pnl': daily_pnl, 'wr': wr, 'trades': s.trades, 'pf': pf,
            'exit_optimal': eb['optimal'], 'exit_partial': eb['partial'],
            'exit_early': eb['early'], 'exit_reversed': eb['reversed'],
            'gross_w': s.gross_win, 'gross_l': abs(s.gross_loss),
        })
    
    def push_account(self, cash: float, realized: float,
                     unrealized: float, net_liq: float):
        self.push({'type': 'ACCOUNT_UPDATE', 'cash_value': cash,
                   'realized_pnl': realized, 'unrealized_pnl': unrealized,
                   'net_liquidation': net_liq})
    
    def push_shutdown_ready(self, status: str):
        self.push({'type': 'SHUTDOWN_READY', 'status': status})
    
    def push_phase(self, step: str, pct: float):
        self.push({'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
                   'step': step, 'pct': pct})
```

**Moves from LiveEngine:**
- `self._gui_queue` → `GUIBridge._q`
- `_gui_push()` → `GUIBridge.push()`
- `_gui_push_stats()` → `GUIBridge.push_stats()`
- `_compute_life_pct()` → keep in LiveEngine (reads position state) but result
  passed to `gui.push_stats()`
- `self._last_gui_push` → `GUIBridge._last_push`
- `self._entry_belief_pct` / `self._exit_belief_pct` → stay in LiveEngine (computed there)

All 30+ `self._gui_push({...})` calls throughout LiveEngine → `self._gui.push_*(...)`.

---

#### Module C: `live/ping_pong.py` (~250 lines extracted)

**Owns:** All ping-pong flip logic, direction refinement, continuation detection.

```python
# live/ping_pong.py

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class FlipRequest:
    """Deferred flip that fires when NT8 confirms flat."""
    exited_side: str
    price: float
    ts: float

class PingPongManager:
    """Manages continuous wave-riding with direction refinement.
    
    Responsibilities:
    - Decide flip vs continuation (same side = no orders, zero latency)
    - Build 2-contract flip orders (close + open in one shot)
    - Track flip count and deferred flip state
    - Delegate direction to ExecutionEngine cascade
    
    Does NOT own: order submission, position state, brain learning.
    LiveEngine owns those and calls back.
    """
    
    def __init__(self, config, brain, belief_network, tuning: dict):
        self._cfg = config
        self._brain = brain
        self._belief_network = belief_network
        self._tuning = tuning
        
        self.flip_count = 0
        self.pending_flip: Optional[FlipRequest] = None
        self.last_exit_side = ''
        self.last_exit_params = None
    
    @property
    def enabled(self) -> bool:
        return self._cfg.ping_pong
    
    def determine_flip(self, exited_side: str, price: float, ts: float,
                       state, exec_engine, side_lock: str = None) -> dict:
        """Decide what to do after an exit in PP mode.
        
        Returns dict with:
          'action': 'flip' | 'continuation' | 'stop'
          'side': target direction
          'sl_ticks', 'tp_ticks', 'trail_ticks', 'trail_act': sizing
          'dir_source': which cascade level decided
        """
        # Body from _flip_position direction logic + sizing
        ...
    
    def schedule_flip(self, exited_side: str, price: float, ts: float):
        """Deferred flip — fires when NT8 confirms flat."""
        self.pending_flip = FlipRequest(exited_side, price, ts)
    
    def consume_pending(self) -> Optional[FlipRequest]:
        """Pop and return pending flip (None if nothing pending)."""
        flip = self.pending_flip
        self.pending_flip = None
        return flip
```

**Moves from LiveEngine:**
- `_flip_position()` → `PingPongManager.determine_flip()` + LiveEngine calls
  `_execute_entry()` with the result
- `_enter_ping_pong()` → `PingPongManager.determine_flip()`
- `_schedule_ping_pong_flip()` → `PingPongManager.schedule_flip()`
- `_auto_tp_reentry()` → stays in LiveEngine (depends on exit engine + orders)
  but uses `PingPongManager.determine_flip()` for direction
- `self._pp_flip_count` → `PingPongManager.flip_count`
- `self._pp_pending_flip` → `PingPongManager.pending_flip`
- `self._pp_min_conviction`, `self._pp_agree_veto`, etc. → `PingPongManager` config
- `self._flip_in_progress` → stays in LiveEngine (guards NT8 order state)
- `self._last_exit_side` → `PingPongManager.last_exit_side`
- `self._pp_last_exit_params` → `PingPongManager.last_exit_params`

---

#### Module D: `live/exit_watcher.py` (~80 lines extracted)

**Owns:** Post-exit counterfactual tracking.

```python
# live/exit_watcher.py

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ExitWatcher:
    """Tracks price after exit to measure exit quality (regret analysis)."""
    
    def __init__(self, tick_size: float, point_value: float):
        self._tick = tick_size
        self._pv = point_value
        self._watchers: list[dict] = []
    
    def add(self, tid, side: str, entry_px: float, exit_px: float,
            exit_pnl: float, reason: str):
        """Start watching price movement after this exit."""
        self._watchers.append({
            'tid': tid, 'side': side,
            'entry_px': entry_px, 'exit_px': exit_px,
            'exit_pnl': exit_pnl, 'exit_time': __import__('time').time(),
            'peak_favorable': exit_px, 'peak_adverse': exit_px,
            'bars_watched': 0, 'reason': reason,
        })
    
    def tick(self, price: float):
        """Called every 15s bar. Delivers verdicts after 60 bars."""
        # _tick_exit_watchers body moves here unchanged
        ...
```

**Moves from LiveEngine:**
- `_tick_exit_watchers()` → `ExitWatcher.tick()`
- `self._exit_watchers` → `ExitWatcher._watchers`

---

#### Resulting LiveEngine structure after decomposition

```python
class LiveEngine:
    """Orchestrator — wires components, runs main loop.
    
    After decomposition: ~350 lines (down from 1,400+).
    
    Owns: main loop, bar processing, NT8 message dispatch,
    position lifecycle, checkpoint loading.
    
    Delegates to:
    - ExecutionEngine: entry gates + direction (via thin wrapper)
    - ExitEngine: exit evaluation
    - SessionTracker: stats, trade log, reports
    - GUIBridge: popup communication
    - PingPongManager: flip logic (if enabled)
    - ExitWatcher: post-exit counterfactual tracking
    - OrderManager: NT8 order lifecycle
    - BarAggregator: bar accumulation + state compute
    - TimeframeBeliefNetwork: multi-TF conviction
    """
    
    def __init__(self, config, ...):
        # Core components (from checkpoints)
        self._brain = BayesianBrain()
        self._exec_engine = ExecutionEngine(...)
        self._exit_engine = ExitEngine(...)
        self._belief_network = TimeframeBeliefNetwork(...)
        
        # Live subsystems
        self._session = SessionTracker(config)
        self._gui = GUIBridge(gui_queue)
        self._pp = PingPongManager(config, self._brain, self._belief_network, self._tuning)
        self._exit_watcher = ExitWatcher(config.tick_size, config.point_value)
        self._orders = OrderManager(config)
        self._aggregator = LiveBarAggregator(...)
        self._trade_logger = TradeLogger()
        
        # Position state (minimal — EE owns gate state, EE owns exit state)
        self._position_open = False
        self._entry_price = 0.0
        self._active_side = ''
        self._active_tid = None
        ...  # ~15 variables instead of 50+
```

---

### 5.2 — Delete disabled trail stop logic in ExitEngine

**File:** `core/exit_engine.py`

**What:** `ExitEngine.evaluate()` has two disabled paths:

```python
# -- 4. MAX HOLD -- DISABLED: envelope_decay handles time-based exits
#    better ($46/trade vs $24/trade). Max hold cut winners short.

# -- 7. TRAIL STOP -- DISABLED: trail exits avg $3-4/trade with 84%
#    "too early" rate.
```

But `_update_trail()` is still called every bar at step 7 to update the
trailing stop level. And `_check_breakeven()` runs every bar at step 8.
The trail update runs a full band-context-aware tightening calculation:

```python
def _update_trail(self, pos, best_price, band_context=None):
    trail_mult = 1.0
    if band_context is not None:
        _sup = band_context.get('support_score', 0.0)
        _res = band_context.get('resistance_score', 0.0)
        if pos.side == 'long' and _res > 0.5:
            trail_mult = 0.6
        ...
    progress_ratio = favorable_move / max(1, pos.trail_activation_ticks)
    tightening = max(0.4, 1.0 - (progress_ratio - 1.0) * 0.15)
    trail_dist_ticks = pos.sl_ticks * tightening * trail_mult
    ...
```

This computation runs but its output is NEVER used for an exit decision.
The only consumer is `_check_breakeven()` which reads `pos.current_trail`
to lock breakeven. But breakeven lock only needs `peak_favorable` and
`trail_activation_ticks` — not the full trail calculation.

**Action:**

1. Delete `_update_trail()` entirely (~35 lines)
2. Delete `_is_trail_hit()` (~5 lines)
3. Simplify `_check_breakeven()` to use `peak_favorable` directly:

```python
def _check_breakeven(self, pos: PositionState):
    """Lock stop to breakeven once profit exceeds activation threshold."""
    if pos.breakeven_locked:
        return
    
    if pos.side == 'long':
        favorable = (pos.peak_favorable - pos.entry_price) / self.tick_size
    else:
        favorable = (pos.entry_price - pos.peak_favorable) / self.tick_size
    
    if favorable >= pos.trail_activation_ticks * 0.6:
        if pos.side == 'long':
            be_level = pos.entry_price + (2 * self.tick_size)
            # Use stop_loss field (fixed SL) as the floor, not current_trail
            pos.stop_loss = max(pos.stop_loss, be_level)
        else:
            be_level = pos.entry_price - (2 * self.tick_size)
            pos.stop_loss = min(pos.stop_loss, be_level)
        pos.breakeven_locked = True
```

4. Remove `pos.current_trail` references from `evaluate()` step 7.
5. Remove `trail_level` from `ExitResult` (set to 0.0 always, or delete field).
6. Delete `PositionState.current_trail` field (no longer needed).
7. Delete `PositionState.trailing_stop_ticks` field (trail distance — unused).
8. Delete `PositionState.original_trail_ticks` field (never read).
9. Delete `trail_activation_ticks` only if breakeven can be computed from
   `sl_ticks * 0.6` instead. **Evaluate:** Currently `trail_activation_ticks`
   is set by `ExecutionEngine._compute_sizing()` to `p25_mae * 0.3` or
   `atr * 0.6`. If it's always a fraction of SL, we can derive it. If not,
   keep it on PositionState.

**Lines removed:** ~55

**Risk:** LOW — trail exits are already disabled. The only behavioral change
is that `_check_breakeven` uses `stop_loss` instead of `current_trail`. Since
`current_trail` was initialized to `stop_loss` and only moved UP (tighter),
using `stop_loss` directly is equivalent to a breakeven lock that doesn't
inherit trail tightening — which is correct (breakeven = entry + 2 ticks,
regardless of where the never-executed trail would have been).

---

### 5.3 — Delete `_LiveCandidate` class

**File:** `live/live_engine.py`

**What:** 30-line dataclass that duplicates `Candidate` from
`core/execution_engine.py` with extra fields for feature extraction:

```python
@dataclass
class _LiveCandidate:
    pattern_type: str
    z_score: float
    velocity: float
    momentum: float
    entropy_normalized: float
    state: object
    depth: int = 10
    timeframe: str = '15s'
    parent_type: str = 'MOMENTUM_BREAK'
    parent_chain: list = None
    timestamp: float = 0.0
    price: float = 0.0
    idx: int = 0
    file_source: str = 'live'
    window_data: object = None
    oracle_marker: int = 0
    oracle_meta: dict = None
```

After the thin wrapper, `LiveEngine` builds `Candidate` objects for
`ExecutionEngine.on_bar()`. Feature extraction reads attributes off
`raw_event` — which can be ANY object with the right attribute names.

**Action:** Delete `_LiveCandidate`. Use `Candidate` directly with `.state`
set to the MarketState. For feature extraction, `extract_features()` reads
from `.state` attributes (z_score, velocity, etc.) — MarketState already
has all of these.

The one field that matters is `parent_chain` (needed by
`extract_features()` for ancestry). In live, parent_chain is always `[]`.
`extract_features()` already handles `chain = []` → ancestry features = 0.0.

So `Candidate.raw_event` can be set to a minimal namespace or the Candidate
itself:

```python
# In LiveEngine thin wrapper:
from core.execution_engine import Candidate

cand = Candidate(
    state=state,
    depth=self._anchor_depth,
    timeframe=self._anchor_tf,
    timestamp=ts,
    pattern_type=state.pattern_type,
    z_score=state.z_score,
    raw_event=None,  # extract_features_from_values() used directly
)
```

Better yet, with the unified `core/feature_extraction.py` from item 2.1,
the `feature_extractor` lambda in `ExecutionEngine.__init__` can call
`extract_feature_vector()` directly from MarketState attributes, bypassing
the need for a PatternEvent-shaped object entirely:

```python
# In ExecutionEngine init:
feature_extractor=lambda cand: extract_feature_vector(
    z_score=cand.state.z_score,
    velocity=cand.state.velocity,
    momentum=cand.state.momentum_strength,
    entropy_normalized=cand.state.entropy_normalized,
    tf_seconds=TIMEFRAME_SECONDS.get(str(cand.timeframe), 15),
    depth=float(cand.depth),
    parent_is_roche=0.0,
    adx=cand.state.adx_strength / 100.0,
    hurst=cand.state.hurst_exponent,
    dmi_diff=(cand.state.dmi_plus - cand.state.dmi_minus) / 100.0,
    parent_z=0.0, parent_dmi_diff=0.0,
    root_is_roche=0.0, tf_alignment=0.0,
    pid=cand.state.term_pid,
    osc_coherence=cand.state.oscillation_entropy_normalized,
)
```

This eliminates the need for `_LiveCandidate`, the `_build_candidate()` method,
and the PatternEvent duck-typing contract entirely in live mode.

**Lines removed:** ~45 (`_LiveCandidate` class + `_build_candidate()` method)

---

### 5.4 — Collapse `_on_bar()` two-tier processing

**File:** `live/live_engine.py`

**What:** `_on_bar()` is ~120 lines with two processing tiers:
- "1-SECOND PROCESSING" — push tick, check exits
- "15-SECOND PROCESSING" — state recompute, TBN tick, entry check

But both tiers are interleaved with warmup checks, history mode checks,
staleness checks, GUI pushes, and TBN re-preparation. The method does 8
different things.

**Action:** Extract the 1s and 15s processing into named methods:

```python
async def _on_bar(self, msg: dict):
    """Route bar to 1s or 15s processing."""
    bar_period = int(msg.get('bar_period_s', 1))
    
    # Capture multi-TF bars for TBN (5s, 4h)
    if bar_period in self._tf_bars:
        self._tf_bars[bar_period].append(self._parse_bar(msg))
    
    # Only primary chart bars feed the aggregator
    if bar_period != self._primary_period and bar_period != self._anchor_period:
        return
    
    price = float(msg['close'])
    ts = float(msg['timestamp'])
    self._last_price = price
    self._last_ts = ts
    self._last_bar_high = float(msg.get('high', price))
    self._last_bar_low = float(msg.get('low', price))
    
    # Aggregate (may trigger state recompute)
    loop = asyncio.get_event_loop()
    states = await loop.run_in_executor(None, self._aggregator.add_bar, msg)
    
    if not self._aggregator.is_warmed_up:
        return
    
    # Every bar: exits + GUI tick
    await self._process_1s(price, ts)
    
    # 15s bars only: entries + TBN
    if states is not None:
        self._last_states = states
        self._bar_i += 1
        await self._process_15s(price, ts, states)

async def _process_1s(self, price: float, ts: float):
    """Sub-second processing: exit checks, GUI tick, staleness."""
    self._gui.push_tick(price, self._bar_i)
    
    if time.time() - ts > 120:
        return  # stale
    
    if self._position_open:
        await self._check_exit(price, ts)

async def _process_15s(self, price: float, ts: float, states: list):
    """Per-anchor-bar processing: TBN, entries, ATR, tuning reload."""
    self._update_live_atr()
    
    if self._bar_i % 20 == 0:
        self._load_tuning()
    
    if self._bar_i % 240 == 1:
        self._refresh_tbn(states)
    
    self._belief_network.tick_all(self._bar_i)
    
    cooldown_ok = (time.time() - self._last_exit_time) > float(self._anchor_period)
    if not self._position_open and not self._orders.loss_limit_hit and cooldown_ok:
        await self._check_entry(price, ts, states)
    
    self._exit_watcher.tick(price)
```

**Lines:** Net ~0 (restructure, not removal) but each method is 15-25 lines
instead of one 120-line method. Independently testable.

---

## Files Created (Updated)

| File | Purpose | Lines |
|------|---------|-------|
| `core/feature_extraction.py` | Single source of truth for 16D feature vector | ~40 |
| `live/session_tracker.py` | Session stats, trade log, drawdown, reports | ~200 |
| `live/gui_bridge.py` | GUI queue abstraction | ~120 |
| `live/ping_pong.py` | Flip logic, direction refinement, deferred flips | ~250 |
| `live/exit_watcher.py` | Post-exit counterfactual tracking | ~80 |
| `tests/test_cpu_cuda_parity.py` | CUDA/CPU output validation | ~30 |

## Files Modified (Updated)

| File | Changes |
|------|---------|
| `core/bayesian_brain.py` | Delete `QuantumBayesianBrain` class, delete `should_fire_validated` if unused, remove `StateVector` import |
| `core/exit_engine.py` | Simplify `open_position()`, delete `make_position()`, trim `PositionState` fields, **delete `_update_trail()`, `_is_trail_hit()`, simplify `_check_breakeven()`** |
| `core/quantum_field_engine.py` | Delete: MC risk engine, dead imports, commented code, unused constants. Import physics constants from cuda module. |
| `core/pattern_utils.py` | Delete scalar `detect_geometric_pattern()` and `detect_candlestick_pattern()` if unused |
| `core/cuda_pattern_detector.py` | Delete `detect_geometric_patterns_cuda()` legacy wrapper if unused |
| `core/fractal_clustering.py` | Refactor `extract_features()` to delegate to `core/feature_extraction.py` |
| `core/timeframe_belief_network.py` | Refactor `state_to_features()` to delegate to `core/feature_extraction.py` |
| `live/live_engine.py` | **Delete `_LiveCandidate`, `_build_candidate()`, `_compute_exit_params()`, `_determine_direction()` (thin wrapper covers), `_write_session_report()`, `_gui_push()`, `_gui_push_stats()`, `_compute_life_pct()` GUI parts, `_tick_exit_watchers()`, PP methods. Extract into modules. Collapse `_on_bar()` into `_process_1s()`/`_process_15s()`.** |

---

## Implementation Order (Updated)

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
10. Delete disabled trail stop logic (`_update_trail`, `_is_trail_hit`, simplify `_check_breakeven`)
11. Delete `_LiveCandidate` class and `_build_candidate()` method

**Phase 2 — Merge duplicates (low risk):**
12. Create `core/feature_extraction.py`, refactor both callers
13. Simplify `ExitEngine.open_position()` to accept pre-computed sizing
14. Delete `make_position()`, update all callers
15. Consolidate physics constants (single source in cuda module)
16. Trim `PositionState` unused fields (including `current_trail`, `trailing_stop_ticks`, `original_trail_ticks`)

**Phase 3 — LiveEngine decomposition (medium risk — pure refactor):**
17. Create `live/session_tracker.py`, move stats + trade log + report
18. Create `live/gui_bridge.py`, move all GUI queue interactions
19. Create `live/ping_pong.py`, move flip logic
20. Create `live/exit_watcher.py`, move counterfactual tracking
21. Collapse `_on_bar()` into `_process_1s()`/`_process_15s()`
22. Reduce LiveEngine `__init__` from 50+ to ~15 instance variables

**Phase 4 — Sync guards (no code risk, adds safety):**
23. Add sync-notice comments to CUDA/CPU dual implementations
24. Write `test_cpu_cuda_parity.py`

**Phase 5 — Done in other specs (reference only):**
25. Direction cascade unification → CLAUDE_CODE_COMPRESSED_REPLAY.md
26. Gate cascade unification → CLAUDE_CODE_COMPRESSED_REPLAY.md
27. LiveEngine._compute_exit_params() deletion → follows from 13-14

---

## Verification (Updated)

### Before:
```bash
find . -name "*.py" -not -path "./__pycache__/*" | xargs wc -l | tail -1
# Record total line count
```

### After:
```bash
# Same command — expect ~600-700 fewer net lines
# (more extracted into modules, but dead code and duplication gone)

# Grep for deleted symbols:
grep -rn "QuantumBayesianBrain\|get_trade_directive\|make_position\|detect_geometric_pattern[^s]\|RISK_THETA\|VELOCITY_CASCADE\|_LiveCandidate\|_update_trail\|_is_trail_hit" --include="*.py" | grep -v __pycache__ | grep -v "# DEPRECATED"
# Should return ZERO results (except backward-compat aliases)

# Verify LiveEngine is ≤ 400 lines:
wc -l live/live_engine.py
# Should be ~300-400 (down from 1,400+)

# Verify no subsystem imports from live_engine (no circular deps):
grep -rn "from live.live_engine import\|import live.live_engine" live/session_tracker.py live/gui_bridge.py live/ping_pong.py live/exit_watcher.py
# Should return ZERO results
```

### Behavioral verification:
```bash
# Run training pipeline — outputs must be identical
python -m training.orchestrator --phase 4 --days 1
# Compare: WR, PnL, trade count, gate skip counts must match pre-refactor

# Run live engine in dry-run — verify all modules wire correctly:
python -m live.launcher --dry-run --skip-replay --warmup-bars 20
```

---

## Summary (Updated)

| Category | Items | Lines Removed | Lines Added |
|----------|-------|---------------|-------------|
| Dead code deletion | 11 items | ~370 | 0 |
| Duplicate merge | 6 items | ~95 | +40 |
| Field cleanup | 13 fields | ~20 | 0 |
| Trail stop deletion | 3 methods | ~55 | 0 |
| `_LiveCandidate` deletion | 1 class + 1 method | ~45 | 0 |
| LiveEngine decomposition | 4 new modules | ~0 (moved) | ~650 (new files) |
| LiveEngine shrinkage | methods moved out | ~720 | 0 |
| `_on_bar()` restructure | 1 method split | ~0 | ~0 |
| Sync guards | 2 items | 0 | +30 |
| Feature extraction module | 1 new file | 0 | +40 |
| **Net lines** | | **~-1,305 removed** | **~+760 in new files** |
| **True net** | | | **~-545 lines** |

LiveEngine goes from **~1,400 lines** to **~350 lines**.
Total instance variables go from **50+** to **~15**.
Four new independently testable modules replace the monolith.

Combined with Thin Wrapper (-250 from gate cascade) and Metaphor Purge (rename-only),
the codebase shrinks by **~800 lines** with zero behavioral changes and
significantly fewer places where logic can silently diverge.

---

## DEFERRED: Needs Replay Data First

These items require the Compressed Replay `direction_source_dist` and
per-worker influence data before a decision can be made:

| Item | What's Needed | Action If Data Confirms |
|------|---------------|------------------------|
| Remove low-weight TBN workers (3m, 30s, 1s) | Per-worker contribution to direction decisions | Delete workers, drop from 11 to 6-7 |
| Prune direction cascade levels | `direction_source_dist` showing which levels never fire | Delete levels that fire <1% of the time |
| Validate ping-pong in OOS | Replay with PP enabled vs disabled | If PP is net-negative, quarantine the module |
| Time-of-day filter | Trade PnL by hour from replay | Add `good_hours_utc` gate if clear dead zones exist |
