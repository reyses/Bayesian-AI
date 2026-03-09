# LiveEngine Decomposition: Task 7
**For: Claude Code / Jules | Priority: After Tasks 2-6 | Risk: Medium**
**Repo state: commit 678c9bc (2026-03-09)**

## Current State

LiveEngine: **1,785 lines, 28 methods**. Four modules already extracted and wired:

| Module | Lines | Status |
|--------|-------|--------|
| `ping_pong.py` | 117 | Working. Owns direction + sizing. LiveEngine uses `self._pp`. |
| `session_tracker.py` | 289 | Working. Owns PnL/drawdown/log. LiveEngine uses `self._session`. |
| `gui_bridge.py` | 79 | Working. Owns GUI pushes. LiveEngine uses `self._gui`. |
| `exit_watcher.py` | 86 | Working. Owns post-exit counterfactual. LiveEngine uses. |

**Problem:** LiveEngine still has 1,785 lines because the flip/PP orchestration,
brain learning, and lifecycle computation haven't been extracted — only their
data structures moved to the modules.

## Top 5 Methods by Size (extraction candidates)

```
263 lines  _main_loop        — core dispatch, stays
116 lines  _load_checkpoints — init, stays
103 lines  _flip_position    — PP flip orchestration → move to ping_pong.py
 97 lines  _execute_entry    — order submit, stays (needs NT8 client)
 93 lines  _brain_learn      — brain update → move to brain_learner module
```

---

## Phase A: Extract `_brain_learn` → `live/brain_learner.py`

### Why
93 lines of pure logic: direction memory updates, depth weight adjustments,
per-template statistics. No NT8 dependency. Currently `_brain_learn(self, pnl)`
reads `self._brain`, `self._direction_memory`, `self._depth_weights` and updates
them. This is a clean extraction.

### What to do

**1. Create `live/brain_learner.py`:**

```python
"""Brain learning — direction memory, depth weights, per-template stats."""

import logging
logger = logging.getLogger(__name__)


class BrainLearner:
    """Manages brain updates after trade completion.

    Owns: direction_memory updates, depth_weight adjustments,
    per-template win/loss tracking, direction_memory.json persistence.
    """

    def __init__(self, brain, direction_memory: dict, depth_weights: dict):
        self._brain = brain
        self._dir_mem = direction_memory
        self._depth_weights = depth_weights

    def learn(self, pnl: float, template_id, side: str, depth: int,
              state_hash=None, p_long: float = 0.5):
        # Move body of LiveEngine._brain_learn() here
        ...

    def save_direction_memory(self, path: str):
        ...
```

**2. Move the body of `_brain_learn()` (lines 1297-1389) into `BrainLearner.learn()`.**

**3. In LiveEngine.__init__, create `self._brain_learner = BrainLearner(...)`.
Replace `self._brain_learn(pnl)` calls with `self._brain_learner.learn(pnl, tid, side, depth, ...)`.**

**4. Delete `_brain_learn()` from live_engine.py.**

**Expected reduction: ~90 lines.**

---

## Phase B: Move flip orchestration into `PingPongManager`

### Why
`_flip_position()` (103 lines) and `_enter_ping_pong()` (68 lines) are PP-specific
orchestration. PingPongManager already owns direction + sizing. The missing piece
is the position creation + order flow coordination.

### What to do

The tricky part: these methods call `make_position()` (or `open_position()` after
Task 2) and submit orders via `self._client`. They can't fully move because they
need NT8 client access.

**Split approach:**

**1. Move decision logic into PingPongManager:**
- `_flip_position` has ~30 lines of direction/sizing logic → already in `determine_flip()`
- ~40 lines of state management (resetting counters, updating tracking vars) → move to
  `PingPongManager.record_flip(flip_decision, position_created)` method
- ~30 lines of position creation + order submission → stays in LiveEngine as a thin caller

**2. Move `_enter_ping_pong` sizing into PingPongManager:**
- Sizing logic already in `compute_sizing()` ✓
- Direction already in `determine_flip()` ✓
- State management (setting `_pp_last_exit_params`, logging) → move to PingPongManager

**3. Move `_auto_tp_reentry` (63 lines) decision logic into PingPongManager:**
- Decision: "should we re-enter?" → `PingPongManager.should_reenter(belief, conviction)`
- Execution: position creation + order → stays in LiveEngine

**4. In LiveEngine, collapse these three methods into thin wrappers:**

```python
async def _flip_position(self, reason, exited_side, price, ts):
    decision = self._pp.determine_flip(exited_side, state, self._exec_engine, ...)
    if decision is None:
        return
    self._pp.record_flip(decision)
    # 5-6 lines: create position + submit order
    pos = self._exit_engine.open_position(...)
    await self._submit_order(...)
```

**Expected reduction: ~80-100 lines** (from 234 combined → ~130-150 remaining).

---

## Phase C: Extract `_compute_life_pct` → `exit_watcher.py`

### Why
54 lines computing "life percentage" of current trade — how much of expected
hold time has elapsed, what's the profit trajectory. Pure calculation on
PositionState. Natural fit for ExitWatcher (already tracks post-exit, this
adds pre-exit monitoring).

### What to do

**1. Add method to `ExitWatcher`:**
```python
def compute_life_pct(self, pos, bars_held, current_price, belief_pct) -> float:
    # Move body of LiveEngine._compute_life_pct() here
    ...
```

**2. Replace calls in LiveEngine with `self._exit_watcher.compute_life_pct(...)`.**

**Expected reduction: ~50 lines.**

---

## Phase D: Collapse `_main_loop` dispatch

### Why
263 lines — the single largest method. It handles message routing from NT8:
bar updates, order fills, position updates, account updates, manual commands.
Most of the body is `if msg_type == 'X':` dispatch blocks.

### What to do (conservative)

**Don't try to decompose the event loop itself.** It's orchestration —
LiveEngine's core job. Instead, extract the handler bodies:

**1. `_on_account_update()` already exists (10 lines) — fine.**

**2. Extract order-fill handling** (~40 lines inside _main_loop that process
order fill confirmations) into `_on_order_fill(msg)` method.

**3. Extract position-update handling** (~30 lines) into `_on_position_update(msg)`.

This doesn't reduce total LiveEngine lines but makes `_main_loop` scannable
(~180 lines of dispatch, each calling a named handler).

**Expected reduction: 0 net lines, but ~70 lines move from _main_loop to named handlers.**

---

## Realistic Target

| Phase | Lines extracted | From → To |
|-------|----------------|-----------|
| A (brain_learner) | ~90 | live_engine → brain_learner.py |
| B (PP orchestration) | ~80-100 | live_engine → ping_pong.py |
| C (life_pct) | ~50 | live_engine → exit_watcher.py |
| D (dispatch) | 0 net | _main_loop → named handlers |

**After all phases: ~1,785 → ~1,550-1,570 lines.**

The original "~500 lines" target assumed moving ALL orchestration out. That's not
realistic — LiveEngine IS the orchestrator. It needs NT8 client, shared state,
and the event loop. A realistic lean target is **~1,200 lines** if we also:
- Merge `_execute_manual_entry` into `_execute_entry` (81 → shared code)
- Inline `_sync_position_state` (20 lines, called once)
- Merge `_init_belief_network` + `_init_exec_engine` into `__init__`

But those are diminishing returns. **Phases A-C are the high-value moves.**

---

## Verification After Each Phase

```bash
# Syntax
python -c "import ast; ast.parse(open('live/live_engine.py').read())"
python -c "import ast; ast.parse(open('live/ping_pong.py').read())"
python -c "import ast; ast.parse(open('live/exit_watcher.py').read())"

# Import check
python -c "from live.live_engine import LiveEngine; print('OK')"

# Line count
wc -l live/live_engine.py  # should decrease
wc -l live/*.py            # total should stay ~same (code moved, not deleted)
```

---

## Commit
```bash
git add -A
git commit -m "refactor: LiveEngine decomposition — extract brain_learner, expand PP manager, move life_pct"
```
