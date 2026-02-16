# Jules Task: Fix Pipeline Bugs & Refactor Worker Architecture

## Context
The fractal training pipeline (`python training/orchestrator.py --fresh --no-dashboard`) runs 3 phases:
- **Phase 2**: Top-down fractal discovery (1D → 4h → 1h → 15m → ... → 1s)
- **Phase 2.5**: K-Means clustering into templates
- **Phase 3**: Per-template parameter optimization with fission

Current state: Phase 2 completes (71 patterns across 3 levels), but Phase 2.5 hangs and the drill-down dies too early at 15m. There are 3 bugs and 1 refactor needed.

---

## Bug 1: MiniBatchKMeans hangs when batch_size > n_samples

**File**: `training/fractal_clustering.py`

**Problem**: The `MiniBatchKMeans` is initialized with `batch_size=256` (line 36), but when only 71 patterns exist, `batch_size > n_samples` causes MiniBatchKMeans to hang or behave unpredictably.

**Fix**: In `create_templates()`, after computing `target_k`, create a fresh KMeans instance sized appropriately instead of mutating `self.model.n_clusters`. For small datasets (< 500 samples), use regular `KMeans` instead of `MiniBatchKMeans`:

```python
# In create_templates(), replace lines ~87-93:
# Old:
#   self.model.n_clusters = target_k
#   labels = self.model.fit_predict(X_scaled)
# New:
if len(valid_patterns) < 500:
    model = KMeans(n_clusters=target_k, random_state=42, n_init=10)
else:
    model = MiniBatchKMeans(n_clusters=target_k, batch_size=min(256, len(valid_patterns)), random_state=42)
labels = model.fit_predict(X_scaled)
```

Also remove `self.model` from `__init__` since we create the model dynamically in `create_templates()`.

---

## Bug 2: Drill-down dies at 15m because regression_period > available bars

**File**: `training/fractal_discovery_agent.py`

**Problem**: `QuantumFieldEngine.batch_compute_states()` requires `n >= regression_period` (default 21). When drilling down from 1h → 15m, only 1 merged window of 2h exists. At 15m resolution, 2h = only 8 bars. Since 8 < 21, `batch_compute_states()` returns an empty list → 0 patterns → drill-down stops.

The core issue: drill-down windows are too narrow. Each parent pattern creates a window of exactly `tf_secs` (one parent bar duration). But the engine needs at least 21 bars of context to compute meaningful states.

**Fix**: In `scan_atlas_topdown()`, when building drill-down windows, expand each window to guarantee enough bars for the CHILD timeframe's regression period. Change lines ~193-198:

```python
# Old:
drilldown_secs = tf_secs  # One bar's worth of time

# New — ensure enough bars for child's regression period
child_tf_idx = TIMEFRAME_HIERARCHY.index(tf) + 1
if child_tf_idx < len(timeframes):
    child_tf = timeframes[child_tf_idx]
    child_tf_secs = TIMEFRAME_SECONDS.get(child_tf, 15)
    min_bars_needed = 30  # regression_period(21) + some margin
    min_window_secs = child_tf_secs * min_bars_needed
    drilldown_secs = max(tf_secs, min_window_secs)
else:
    drilldown_secs = tf_secs
```

This ensures that when drilling from 1h to 15m, the window is at least `15m * 30 = 7.5h` instead of just `1h`, giving the engine enough context bars.

---

## Bug 3 (Refactor): Move standalone functions out of orchestrator.py into orchestrator_worker.py

**Files**: `training/orchestrator.py`, `training/orchestrator_worker.py`

**Problem**: `orchestrator_worker.py` line 3 imports functions FROM `orchestrator.py`:
```python
from training.orchestrator import _optimize_pattern_task, _optimize_template_task, simulate_trade_standalone
```

While `orchestrator.py` line 556 lazy-imports from `orchestrator_worker.py`:
```python
from training.orchestrator_worker import _process_template_job
```

This is a circular dependency. It doesn't crash because the orchestrator.py import is lazy (inside `train()`), but it causes **every multiprocessing worker to load the entire orchestrator.py** (with all its heavy imports: torch, numba, sklearn, etc.) just to get 3 small functions. On Windows (spawn mode), this is slow and wasteful.

**Fix — move these functions from `orchestrator.py` to `orchestrator_worker.py`**:

1. Move these 3 functions from `orchestrator.py` into `orchestrator_worker.py`:
   - `simulate_trade_standalone()` (orchestrator.py lines 103-169)
   - `_optimize_pattern_task()` (orchestrator.py lines 171-216)
   - `_optimize_template_task()` (orchestrator.py lines 218-276)

2. Move these 2 constants from `orchestrator.py` to `orchestrator_worker.py`:
   - `DEFAULT_BASE_SLIPPAGE = 0.25`
   - `DEFAULT_VELOCITY_SLIPPAGE_FACTOR = 0.1`
   - `REPRESENTATIVE_SUBSET_SIZE = 20`
   - `FISSION_SUBSET_SIZE = 50`
   - `INDIVIDUAL_OPTIMIZATION_ITERATIONS = 20`

3. Update `orchestrator_worker.py` imports:
   - Remove: `from training.orchestrator import _optimize_pattern_task, _optimize_template_task, simulate_trade_standalone`
   - Remove: `from training.orchestrator import FISSION_SUBSET_SIZE, INDIVIDUAL_OPTIMIZATION_ITERATIONS`
   - Add: `from core.bayesian_brain import TradeOutcome` (needed by simulate_trade_standalone)
   - Add: `from training.doe_parameter_generator import DOEParameterGenerator` (if needed)
   - Add: `import numpy as np`

4. Update `orchestrator.py`:
   - Add import at top: `from training.orchestrator_worker import simulate_trade_standalone, _optimize_pattern_task, _optimize_template_task`
   - Add import at top: `from training.orchestrator_worker import FISSION_SUBSET_SIZE, INDIVIDUAL_OPTIMIZATION_ITERATIONS, DEFAULT_BASE_SLIPPAGE, DEFAULT_VELOCITY_SLIPPAGE_FACTOR`
   - Remove the moved function definitions and constants from orchestrator.py
   - Remove the lazy import on line 556 (`from training.orchestrator_worker import _process_template_job`) and put it at the top with the other imports
   - Keep the wrapper methods `_optimize_pattern_task(self, args)` and `_optimize_template_batch(self, subset)` on the class — they just call the imported standalone functions

5. Verify `orchestrator.py` still references `simulate_trade_standalone` in:
   - `validate_template_group()` method
   - `validate_pattern()` method

---

## Verification

After making all changes, run:
```bash
python training/orchestrator.py --fresh --no-dashboard --iterations 50
```

Expected behavior:
1. Phase 2 discovery should cascade deeper than 15m (the expanded windows should give enough bars)
2. Phase 2.5 clustering should complete instantly (not hang)
3. Phase 3 optimization should process templates and checkpoint after each batch
4. No circular import errors or warnings

Also run a quick import check:
```bash
python -c "from training.orchestrator_worker import _process_template_job; print('OK')"
python -c "from training.orchestrator import BayesianTrainingOrchestrator; print('OK')"
```

Both should print "OK" without errors.

---

## File Summary

| File | Action |
|------|--------|
| `training/orchestrator_worker.py` | Receives moved functions + constants, becomes self-contained |
| `training/orchestrator.py` | Loses standalone functions, imports them from worker |
| `training/fractal_clustering.py` | Fix KMeans to handle small datasets |
| `training/fractal_discovery_agent.py` | Expand drill-down windows for child TF regression period |
