# Code Cleanup: Tasks 2–6
**For: Claude Code / Jules | Priority: Do in order | Risk: Low**
**Repo state: commit 678c9bc (2026-03-09)**

Do these sequentially. Each task takes 5-15 minutes. Run syntax check after each.

```bash
python -c "import ast; ast.parse(open('FILE.py').read())"
```

---

## Task 2: Delete `make_position()` → use `open_position()`

### Problem
`make_position()` is a standalone function at `exit_engine.py:91` that duplicates
`ExitEngine.open_position()` at line 213. Both create PositionState with absolute
price levels. The standalone version is older, doesn't use lib_entry overrides,
and sets deprecated fields.

### What to do

**Step 1:** In `live/live_engine.py`, replace all 5 `make_position()` calls with
`self._exit_engine.open_position()`. The signatures differ:

```python
# OLD (make_position):
make_position(entry_price=price, side=side, tick_size=..., tick_value=...,
              stop_distance_ticks=sl_ticks, profit_target_ticks=tp_ticks,
              trailing_stop_ticks=trail_ticks, trail_activation_ticks=trail_act,
              template_id=tid, state=state)

# NEW (open_position):
self._exit_engine.open_position(
    entry_price=price, side=side,
    sl_ticks=sl_ticks, tp_ticks=tp_ticks,
    trail_ticks=trail_ticks, trail_activation_ticks=trail_act,
    template_id=tid, entry_bar_index=self._bar_count)
```

Key differences:
- `stop_distance_ticks` → `sl_ticks`
- `profit_target_ticks` → `tp_ticks`
- `trailing_stop_ticks` → `trail_ticks`
- No `tick_size`/`tick_value` (ExitEngine has them as instance attrs)
- No `state` param (open_position doesn't use it)
- Add `entry_bar_index` (use current bar count)

Call sites at lines: 874, 1013, 1120, 1190, 1482.

**Step 2:** In `training/trainer.py`, replace 2 calls (lines 1091, 1225) same way.
Trainer has `self.exit_engine` (no underscore). Use `self._bar_index` or equivalent
for entry_bar_index.

**Step 3:** Delete the import `from core.exit_engine import make_position` from both files.

**Step 4:** Delete `make_position()` function from `exit_engine.py` (lines 91-118).

**Step 5:** Update `docs/memory/MEMORY.md` — change the Architecture bullet:
```
- **Position factory**: `make_position()` in `core/exit_engine.py`
```
to:
```
- **Position factory**: `ExitEngine.open_position()` — creates PositionState with absolute levels
```

### Verify
```bash
python -c "import ast; ast.parse(open('core/exit_engine.py').read())"
python -c "import ast; ast.parse(open('live/live_engine.py').read())"
python -c "import ast; ast.parse(open('training/trainer.py').read())"
grep -rn "make_position" core/ live/ training/  # should return 0 hits
```

---

## Task 3: Delete disabled trail stop logic

### Problem
Trail stops are DISABLED (decision 2026-03-06, envelope handles exits better).
But the fields and partial logic remain: `current_trail`, `original_trail_ticks`,
`trailing_stop_ticks` in PositionState, and `_check_breakeven()` references
`current_trail` for its level calculation.

### What to do

**Step 1:** In `PositionState` (exit_engine.py ~line 50), delete these fields:
- `current_trail: float = 0.0` (line 70)
- `original_trail_ticks: float = 0.0` (line 82)
- `trailing_stop_ticks: float = 0.0` (line 66)

**Step 2:** In `open_position()` (~line 213), remove all references to deleted fields:
- Delete line setting `trailing_stop_ticks` (line ~251)
- Delete lines setting `current_trail` (lines ~260, ~264)
- Delete line setting `original_trail_ticks` (line ~273)

**Step 3:** In `evaluate()` or wherever `current_trail` appears in exit checks,
replace `pos.current_trail` with `pos.stop_loss` — they're set to the same value
since trail is disabled. Grep to find all: lines ~334, ~346, ~383, ~392.

**Step 4:** Simplify `_check_breakeven()` (line 683). It currently uses
`trail_activation_ticks` — keep that field (it's used for breakeven activation
threshold). The method itself is fine, just remove any `current_trail` refs.

**Step 5:** In `make_position()` — skip if you already deleted it in Task 2.

### Verify
```bash
python -c "import ast; ast.parse(open('core/exit_engine.py').read())"
grep -n "current_trail\|original_trail_ticks\|trailing_stop_ticks" core/exit_engine.py
# Should return 0 hits (or only the trail_activation_ticks which stays)
```

---

## Task 4: Flatten empty `MarketBayesianBrain`

### Problem
`MarketBayesianBrain` at `bayesian_brain.py:292` is:
```python
class MarketBayesianBrain(BayesianBrain):
    """Extends BayesianBrain with MarketState-specific filters..."""
    pass
```
Empty class. Exists only for pickle backward compat — old checkpoints may have
`MarketBayesianBrain` or `QuantumBayesianBrain` in their pickle headers.

### What to do

**Step 1:** Replace the class definition with aliases:
```python
# Backward compat: old checkpoints may pickle these class names
MarketBayesianBrain = BayesianBrain
QuantumBayesianBrain = BayesianBrain
```

**Step 2:** Check if `QuantumBayesianBrain` is referenced anywhere:
```bash
grep -rn "QuantumBayesianBrain" core/ live/ training/
```
If yes, add the alias. If no, only add `MarketBayesianBrain`.

### Verify
```bash
python -c "import ast; ast.parse(open('core/bayesian_brain.py').read())"
python -c "from core.bayesian_brain import MarketBayesianBrain; print(MarketBayesianBrain.__name__)"
# Should print "BayesianBrain"
```

---

## Task 5: Trim dead PositionState fields

### Depends on: Tasks 2 + 3 complete

### Problem
After Tasks 2-3, these fields should have zero references:
- `high_water_mark` (alias for `peak_favorable` — live_engine reads it)
- `last_adjustment_reason` (set but never read)
- `trailing_stop_ticks` (deleted in Task 3, confirm no refs)

### What to do

**Step 1:** Grep to confirm each field has zero meaningful reads:
```bash
grep -rn "high_water_mark" core/ live/ training/ | grep -v "peak_favorable"
grep -rn "last_adjustment_reason" core/ live/ training/
grep -rn "trailing_stop_ticks" core/ live/ training/
```

**Step 2:** For `high_water_mark`: if live_engine.py still reads it, replace those
reads with `peak_favorable` (same value). Then delete the field and all assignments.

**Step 3:** Delete `last_adjustment_reason` field + any assignments.

**Step 4:** Delete `trailing_stop_ticks` if not already gone from Task 3.

### Verify
```bash
python -c "import ast; ast.parse(open('core/exit_engine.py').read())"
grep -rn "high_water_mark\|last_adjustment_reason" core/ live/ training/  # 0 hits
```

---

## Task 6: Purge 8 metaphor remnants

### Problem
8 physics metaphors remain in 5 files. Each is a string/comment replacement.

### Exact changes

**1. `core/market_state.py:75`**
```python
# BEFORE:
entropy_normalized: float                # 1.0=superposition, 0.0=collapsed
# AFTER:
entropy_normalized: float                # 1.0=uncertain/mixed, 0.0=decisive/aligned
```

**2-3. `core/statistical_field_engine.py:383,386`**
```python
# BEFORE (line 383):
# boundary 0 (center) before boundary B (event horizon) starting at z:
# AFTER:
# boundary 0 (center) before boundary B (3σ breakout boundary) starting at z:

# BEFORE (line 386):
_B = 3.0  # event horizon in z-score units
# AFTER:
_B = 3.0  # breakout boundary in z-score units
```

**4. `training/orchestrator_worker.py:164`**
```python
# BEFORE:
if hasattr(state, 'cascade_detected') and state.cascade_detected: # Roche Snap
# AFTER:
if hasattr(state, 'cascade_detected') and state.cascade_detected: # Band Snap
```

**5-6. `training/pid_oscillation_analyzer.py:30,36`**
```python
# BEFORE (line 30):
#   1. z_score near outer Roche (>= 1.5σ) — PID fighting possible breakout
# AFTER:
#   1. z_score near outer 2σ band (>= 1.5σ) — PID fighting possible breakout

# BEFORE (line 36):
PID_TENSION_Z_MIN        = 1.5    # z >= this → approaching outer Roche → TENSION
# AFTER:
PID_TENSION_Z_MIN        = 1.5    # z >= this → approaching 2σ band → TENSION
```

**7. `training/pid_oscillation_analyzer.py:133`**
```python
# BEFORE:
tension_reason = 'outer_roche'      # approaching outer Roche limit
# AFTER:
tension_reason = 'outer_band'       # approaching outer 2σ band
```

**8. `training/trainer.py:3528`**
```python
# BEFORE:
rpt.append(f"  Roche-backed: {roche_roots}")
# AFTER:
rpt.append(f"  Band-backed: {band_roots}")
```

For line 3528, also find the variable `roche_roots` and rename to `band_roots`
(check ~5-10 lines above for where it's computed).

### Verify
```bash
grep -rn "Roche\|event.horizon\|superposition.*collapsed\|collapsed.*superposition" \
  core/ training/ | grep -v __pycache__
# Should return 0 hits
```

---

## After All Tasks Complete

1. Run full syntax check:
```bash
for f in core/*.py live/*.py training/*.py; do
  python -c "import ast; ast.parse(open('$f').read())" 2>&1 | grep -v "^$" && echo "FAIL: $f"
done
echo "All syntax checks passed"
```

2. Update `docs/daily/YYYY-MM-DD.md` with what was done.

3. Commit:
```bash
git add -A
git commit -m "refactor: delete make_position, trail remnants, dead fields, flatten brain alias, purge metaphors"
```
