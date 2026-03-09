# Jules Spec: Compressed History Replay Engine

**Source:** `docs/CLAUDE_CODE_COMPRESSED_REPLAY.md` (full design spec)
**Priority:** Before next live session
**Scope:** 2 new files, 4 modified files, ~250 net line change

---

## Overview

Replace NT8 history dump (3-5 min TCP) with local ATLAS parquet replay (~2 min).
Produces fully warmed brain/TBN/exit engine, validates against OOS, then hands off
to live engine for delta-only NT8 sync.

**CRITICAL:** Live engine currently reimplements gate cascade inline (~310 lines).
This refactor deletes that duplication and delegates to `ExecutionEngine` from
`core/execution_engine.py` — one source of truth for gates + direction.

---

## Implementation Order

### Part 1: `live/atlas_loader.py` (NEW ~80 lines)

Create ATLAS parquet loader:

```python
def load_atlas_range(atlas_root: str, tf: str, n_days: int, end_date=None) -> pd.DataFrame:
    """Load N trading days of data from ATLAS parquet files."""
    # 1. Find YYYY_MM.parquet files in atlas_root/tf/
    # 2. Read + concat
    # 3. Filter to last n_days trading days (exclude weekends, maintenance 4-5pm CT)
    # 4. Return df with: timestamp, open, high, low, close, volume

def load_multi_tf(atlas_root: str, n_days: int = 5) -> dict:
    """Returns {tf_label: pd.DataFrame} for 15s (always), 5s, 1s, 4h (if available)."""

def split_trading_days(df: pd.DataFrame) -> list:
    """Split into trading days (5PM CT → 4PM CT next day)."""
    # Detect by gaps > 30 min in timestamps
```

Testable standalone: `python -m live.atlas_loader --days 5`

### Part 2: `core/execution_engine.py` modifications

**A. Add `mode` parameter:**
```python
def __init__(self, ..., mode: str = 'is', ...):
    self.mode = mode  # 'is', 'oos', 'live', 'replay'
```

**B. Add `live_momentum` to `_direction_cascade()` (Priority 0.3, after pp_override):**
- Only fires when `self.mode in ('live', 'replay')`
- Uses `state.velocity + 0.5 * state.net_force`
- Threshold: abs(momentum) > 0.5

**C. Add `live_bias` to `_direction_cascade()` (Priority -0.5):**
- Only fires when `self.mode in ('live', 'replay')`
- Uses `brain.get_dir_bias(tid)` — long/short win rates from PP learning
- Requires min 5 trades before firing

**D. Add `set_live_atr()` + ATR floor in `_compute_sizing()`:**
- In live/replay mode, enforce ATR-based minimum for SL/TP
- `_atr_sl = max(4, int(round(atr * 3.0)))`, `_atr_tp = max(4, int(round(atr * 5.0)))`

**Updated direction priority order:**
```
-1   : pp_override (ping-pong)
-0.5 : live_bias (brain dir_bias) — live/replay only
 0.3 : live_momentum (velocity+accel) — live/replay only
 0.5 : signed_mfe
 1   : logistic
 1.5 : brain_dir
 2   : template_bias
 3   : band_confluence
 4   : dmi
 5   : velocity (fallback)
```

### Part 3: `live/history_replay.py` (NEW ~200 lines)

```python
@dataclass
class ValidationReport:
    replay_trades: int
    replay_wr: float
    replay_pnl: float
    replay_avg_trade: float
    oos_trades: int
    oos_wr: float
    oos_pnl: float
    oos_avg_trade: float
    parity_score: float      # 0.0-1.0
    gate_stats: dict
    direction_source_dist: dict
    warnings: list
    passed: bool             # True if parity >= 0.80

@dataclass
class ReplayResult:
    brain: QuantumBayesianBrain
    belief_network: TimeframeBeliefNetwork
    exit_engine: ExitEngine
    execution_engine: ExecutionEngine
    last_timestamp: float
    validation: ValidationReport
    states_micro: list
    df_micro: pd.DataFrame

class HistoryReplayEngine:
    def __init__(self, config, checkpoint_dir, n_days=5, atlas_root='DATA/ATLAS'):
        ...

    def run(self) -> ReplayResult:
        # 1. Load checkpoints (same as LiveEngine._load_checkpoints)
        # 2. Load ATLAS multi-TF data
        # 3. Split into trading days
        # 4. Init: StatisticalFieldEngine, Brain, ExitEngine, ExecutionEngine, TBN
        # 5. Per-day forward pass: batch_compute_states → TBN.prepare_day → per-bar loop
        # 6. Build validation report (compare to OOS checkpoint)
        # 7. Return ReplayResult with all warmed state

    def _replay_day(self, day_df, tf_data, engine, exec_engine, tbn):
        # Mirrors trainer Phase 4:
        # 1. batch_compute_states (bulk GPU)
        # 2. TBN prepare_day (resample TFs)
        # 3. Per-bar: TBN tick + exec_engine.on_bar()
        # 4. Track entries/exits, brain learning, exit self-tune

    def _build_validation(self, trades) -> ValidationReport:
        # Compare replay WR/PnL/freq to OOS reference
        # Parity scoring: WR within 5pp, avg trade within 50%, freq within 30%
```

### Part 4: `live/live_engine.py` refactor (LARGEST CHANGE)

**DELETE (~310 lines):**
- `_check_entry()` — ~200 lines of reimplemented gates
- `_determine_direction()` — ~80 lines of reimplemented direction cascade
- `_compute_exit_params()` — ~30 lines
- `_gate_stats` dict (use `exec_engine.gate_stats` instead)
- Module constants: `_ADX_TREND_CONFIRMATION`, `_HURST_TREND_CONFIRMATION`,
  `_GATE1_DIST_THRESHOLD`, `_WORKER_BYPASS_CONV`

**ADD (~60 lines):**
- `__init__`: instantiate `ExecutionEngine(mode='live', ...)`
- New thin `_check_entry()`: build candidates → `exec_engine.on_bar()` → if ENTER, call `_execute_entry()`
- `_execute_entry(action, price, ts)`: position creation, NT8 order, TBN tracking, GUI push
- `_gate_label_to_pct(gate_label)`: map EE rejection labels to GUI belief bar %
- Session report: pull from `exec_engine.get_skip_counts()` instead of `_gate_stats`

**KEEP (don't touch):**
- `_check_exit()` — already uses ExitEngine correctly, just add `exec_engine.position_closed()` on exit
- `_flip_position()` — PP flips bypass gates intentionally, but call `exec_engine.position_opened/closed()`
- `_handle_manual_order()` — manual orders bypass gates, sync with EE position state
- All NT8 communication, GUI pushes, trade logging

**Aggression scaling:** Wrapper sets `exec_engine.gate1_dist` before each `on_bar()` call.
YOLO mode: `gate1_dist = inf`. Normal: `base + agg * 10.0`.

### Part 5: Integration wiring

**`live/live_engine.py` `run()` flow:**
1. Run `HistoryReplayEngine.run()` (blocking, ~2 min)
2. Validation gate: if `parity < 0.80`, refuse to trade
3. Transfer warmed state: brain, TBN, exit_engine, last_timestamp
4. Connect NT8 with `RESUME_FROM` (delta only)
5. Normal live loop (starts warm)

**`live/bar_aggregator.py`:** Add `seed_from_replay(df, states)` to accept pre-computed state.

**`live/launcher.py`:** Add `--skip-replay` flag (bypass validation, load from NT8 as before).

**`live/config.py`:** Add `atlas_root`, `replay_days`, `replay_validate` fields.

### Part 6: Session report update

Session report currently reads from `self._gate_stats`. Change to read from
`self._exec_engine.get_skip_counts()`. Use the semantic names already in place:
Pattern Quality, Depth Filter, Template Match, Brain Reject, Low Conviction,
Momentum Misalign, Physics Quality.

---

## Key Risks & Mitigations

1. **`Candidate.raw_event` with `parent_chain`**: Live has no fractal tree.
   `extract_features()` already returns 0.0 for ancestry features when empty. Verify.

2. **Exit timing**: Live checks exits on every 1s bar, entries on 15s bars.
   Wrapper calls `_check_exit()` on every bar, `exec_engine.on_bar()` on 15s only.

3. **PP flip / manual orders**: Bypass EE gates intentionally but MUST call
   `exec_engine.position_opened()` / `position_closed()` to keep state synced.

4. **Aggression scaling**: Set EE thresholds dynamically before each `on_bar()`.

---

## Acceptance Criteria

1. Cold start to live-ready in < 120 seconds (5 days of 15s data)
2. Validation report prints before first live trade
3. If parity < 0.80, engine refuses to trade
4. Brain dir_bias populated for all templates seen in replay
5. TBN workers have current beliefs (not None) for active TFs
6. Exit engine self-tune calibrated (not defaults)
7. NT8 receives RESUME_FROM only (no REQUEST_HISTORY on warm start)
8. `--skip-replay` flag exists for debugging
9. `_check_entry()` is <= 80 lines (wrapper only)
10. `_determine_direction()` does not exist (deleted)
11. Direction cascade includes `live_momentum` and `live_bias`
12. Session report gate_stats from `ExecutionEngine.get_skip_counts()`

---

## Files Summary

| File | Action | Lines |
|------|--------|-------|
| `live/atlas_loader.py` | NEW | ~80 |
| `live/history_replay.py` | NEW | ~200 |
| `core/execution_engine.py` | MODIFY | +~50 (mode, live_momentum, live_bias, ATR) |
| `live/live_engine.py` | REFACTOR | -310, +60 (delete gates, add thin wrapper) |
| `live/bar_aggregator.py` | MODIFY | +10 (seed_from_replay) |
| `live/launcher.py` | MODIFY | +5 (--skip-replay flag) |
| `live/config.py` | MODIFY | +5 (atlas_root, replay_days fields) |
