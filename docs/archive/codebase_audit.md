# Codebase Duplication Audit
> Generated: 2024-05-18

## Summary
- 13 duplications found
- 2 dead code blocks found
- 5 files affected

## Duplications Found

### [D1] Direct Trade Recording in Live Engine
- **File**: `live/live_engine.py:1519`
- **Duplicates**: `BarProcessor.force_close()` / `_handle_exit()` trade recording logic
- **Severity**: HIGH
- **Recommendation**: Live engine should not import `record_trade` from `core.bayesian_brain` or call it directly. It should use `BarProcessor.force_close()` or `BarProcessor.process_bar()` which handles trade recording and brain updates internally.

### [D2] Direct Exit Engine Execution in Live Engine
- **File**: `live/live_engine.py:1199`
- **Duplicates**: `BarProcessor` handles exit engine outcomes
- **Severity**: HIGH
- **Recommendation**: `self._exit_engine.record_trade_outcome` should not be called directly.

### [D3] Manual Position State Tracking in Live Engine
- **File**: `live/live_engine.py:1103`, `1110`, `1200`, `1240`, `1345`, `1413`, `1645`, `1748`
- **Duplicates**: `BarProcessor`'s internal position tracking
- **Severity**: HIGH
- **Recommendation**: Remove `self._position` assignments. `BarProcessor` uses `exec_engine.in_position` internally.

### [D4] Manual Position Open/Close Calls in Live Engine
- **File**: `live/live_engine.py:1203`, `1353`, `1655`
- **Duplicates**: `BarProcessor` manages entry/exit triggers
- **Severity**: HIGH
- **Recommendation**: Do not call `self._exec_engine.position_opened()` or `self._exec_engine.position_closed()` directly in `live_engine.py`.

### [D5] Manual Belief Network Ticking in Live Engine
- **File**: `live/live_engine.py:675`
- **Duplicates**: `BarProcessor.process_bar()` already calls `belief_network.tick_all()`
- **Severity**: HIGH
- **Recommendation**: Remove manual `tick_all()` in the main bar loop.

### [D6] Manual Belief Network Trade Tracking in Live Engine
- **File**: `live/live_engine.py:1127`, `1257`, `1369`, `1431`, `1691` (starts) and `397`, `444`, `885`, `907`, `954`, `1282`, `1294` (stops)
- **Duplicates**: `BarProcessor` automatically starts/stops tracking upon position changes
- **Severity**: HIGH
- **Recommendation**: Remove `start_trade_tracking` and `stop_trade_tracking` calls from `live_engine.py`.

### [D7] Manual Belief Network Ticking in History Replay
- **File**: `live/history_replay.py:183`
- **Duplicates**: `BarProcessor.process_bar()` ticks the belief network
- **Severity**: HIGH
- **Recommendation**: Remove manual `tick_all()` call.

### [D8] Manual Position State Tracking in Trainer
- **File**: `training/trainer.py:158`, `3793`
- **Duplicates**: Legacy inline loop variable
- **Severity**: MEDIUM
- **Recommendation**: Remove `self._position = None`.

### [D9] Manual Position Open/Close Calls in Trainer
- **File**: `training/trainer.py:1669`
- **Duplicates**: `BarProcessor` manages entry/exit triggers
- **Severity**: HIGH
- **Recommendation**: Do not call `_exec_engine.position_opened()` outside of `BarProcessor` (unless it's part of the ping-pong flip logic).

### [D10] Manual Belief Network Ticking in Trainer
- **File**: `training/trainer.py:1480`, `1497`, `1506`
- **Duplicates**: `BarProcessor` manages ticking
- **Severity**: MEDIUM
- **Recommendation**: Manual `tick_all()` should only be for maintenance/DD-stop bars where `process_bar` is explicitly skipped. Verify these occurrences.

### [D11] Manual Belief Network Trade Tracking in Trainer
- **File**: `training/trainer.py:1691`
- **Duplicates**: `BarProcessor` manages tracking
- **Severity**: MEDIUM
- **Recommendation**: Check if this is outside `BarProcessor` and not part of the ping-pong flip.

### [D12] Candidate Construction outside BarProcessor
- **File**: `live/ping_pong.py:88`
- **Duplicates**: Inline `Candidate()` construction
- **Severity**: MEDIUM
- **Recommendation**: Should ideally be part of `bar_processor._build_candidates()`.

### [D13] Inline Report Breakdowns in Trainer
- **File**: `training/trainer.py:2305` (exit reason loop), `2706` (exit reason loop), `2404` (direction loop), etc.
- **Duplicates**: `core/report_engine.py` functions
- **Severity**: MEDIUM
- **Recommendation**: `_write_forward_pass_reports` contains inline breakdowns that duplicate logic handled by `report_engine.py`'s `compute_stats()`.

## Dead Code Found

### [DC1] Unused Feature Extraction Import
- **File**: `training/trainer.py:53`
- **Reason**: `extract_feature_vector` imported but not used directly (it's used in `bar_processor.py` now).
- **Safe to delete**: yes

### [DC2] Unused Trade Tracking Variables
- **File**: `training/trainer.py` (assumed based on `self._position`)
- **Reason**: Need a deeper sweep to clean up unused position tracking variables `current_position_open`, `active_entry_price`.
- **Safe to delete**: yes

## Clean (No Issues)
- `live/session_tracker.py` delegates to `format_scorecard` correctly.
- No direct calls to `exec_engine.on_bar()` or `exit_engine.evaluate()` outside their designated engines found.
- No `_detect_peak_reversal` calls outside of `bar_processor.py` found.
- No references to `_live_val`, `OOS3`, `_lv_processor`, `_write_live_validation_report` found.
