# Jules Spec: Codebase Duplication Audit

## Objective
READ-ONLY audit. No code changes. Produce a report identifying all duplicated logic across the codebase. Focus on code paths that SHOULD be shared but aren't.

## Context
We just completed a major refactor:
- IS forward pass (1,029 lines inline) replaced with `BarProcessor.process_bar()` calls
- OOS3 parity replay deleted (was a second BarProcessor instance comparing against inline)
- Shared `core/report_engine.py` replaces duplicate scorecard logic in trainer + live
- Live maintenance flatten added (was missing)

The goal was ONE engine, ONE report module, ONE code path for IS/OOS/live. This audit verifies we succeeded.

## What to Audit

### 1. Bar Processing Paths
Search for ANY code that makes entry/exit decisions outside of `BarProcessor.process_bar()`.

Files to check:
- `training/trainer.py` — should only call `_bp.process_bar()` and `_bp.force_close()`
- `live/live_engine.py` — should only call `self._processor.process_bar()` and `force_close()`
- `live/history_replay.py` — should only call `process_bar()` and `force_close()`

Red flags:
- Direct calls to `exec_engine.on_bar()` outside of bar_processor.py
- Direct calls to `exit_engine.evaluate()` outside of execution_engine.py
- Direct calls to `record_trade()` outside of bar_processor.py
- Inline candidate building (feature extraction + Candidate construction) outside bar_processor.py
- Inline peak detection (`_detect_peak_reversal`) outside bar_processor.py

### 2. Report Generation
Search for ANY report formatting that duplicates `core/report_engine.py`.

Files to check:
- `training/trainer.py` — `_write_actionable_scorecard` should delegate to report_engine
- `training/trainer.py` — `_write_forward_pass_reports` may still have inline breakdowns
- `live/session_tracker.py` — `write_report` should use `format_scorecard`
- `live/history_replay.py` — any inline report generation

Red flags:
- Exit reason breakdown computed inline (looping oracle_trade_records by exit_reason)
- Direction breakdown computed inline (looping by direction/side)
- Hold duration buckets computed inline
- PF/WR/avg calculations that could use `compute_stats()`

### 3. Trade Recording
Search for ANY trade recording outside of BarProcessor.

Files to check:
- `training/trainer.py` — should NOT call `record_trade()` directly
- `live/live_engine.py` — should NOT call `record_trade()` directly

Red flags:
- `from core.bayesian_brain import record_trade` in trainer.py or live_engine.py
- Direct `self.brain.update()` calls outside bayesian_brain.py

### 4. Position Management
Search for ANY position open/close outside of BarProcessor.

Files to check:
- `training/trainer.py` — should NOT call `_exit_eng.open_position()` or `_exec_engine.position_opened/closed()` directly (except ping-pong flip)
- `live/live_engine.py` — position management goes through NT8 orders, but BarProcessor tracks internal state

Red flags:
- `self._position = ` assignments in trainer.py (legacy from inline loop)
- `_exec_engine.position_closed()` calls in trainer.py outside of _bp
- `_exec_engine.position_opened()` calls in trainer.py outside of _bp (except PP flip)

### 5. TBN (Belief Network) Ticking
Search for ANY TBN tick calls outside of BarProcessor.

Files to check:
- `training/trainer.py` — manual `belief_network.tick_all()` calls should ONLY be for maintenance/DD-stop bars where process_bar is skipped
- `live/live_engine.py` — TBN ticking should be inside process_bar

Red flags:
- `belief_network.tick_all()` in the main bar loop (process_bar already does this)
- `belief_network.start_trade_tracking()` outside bar_processor.py (except PP flip)
- `belief_network.stop_trade_tracking()` outside bar_processor.py

### 6. Feature Extraction
Search for ANY feature vector construction outside of `core/feature_extraction.py` and `bar_processor._build_candidates()`.

Red flags:
- `extract_feature_vector()` calls in trainer.py (should be inside _build_candidates)
- Manual numpy feature array construction in trainer.py or live_engine.py
- Candidate() construction in trainer.py (should be inside _build_candidates)

### 7. Dead Code
Search for unreachable or unused code left over from the refactor.

Red flags:
- Variables assigned but never read (e.g., `current_position_open`, `active_entry_price` if no longer needed)
- Functions/methods that are never called
- Import statements for modules no longer used
- `if False:` blocks or commented-out code blocks > 5 lines
- References to `_live_val`, `OOS3`, `_lv_processor`, `_write_live_validation_report`

## Output Format

Write findings to `reports/findings/codebase_audit.md` with this structure:

```markdown
# Codebase Duplication Audit
> Generated: YYYY-MM-DD

## Summary
- X duplications found
- Y dead code blocks found
- Z files affected

## Duplications Found
### [D1] Description
- **File**: path:line
- **Duplicates**: what it duplicates
- **Severity**: HIGH/MEDIUM/LOW
- **Recommendation**: what should change

## Dead Code Found
### [DC1] Description
- **File**: path:line
- **Reason**: why it's dead
- **Safe to delete**: yes/no

## Clean (No Issues)
List of audited areas that passed clean.
```

## Rules
- DO NOT modify any files
- DO NOT run any commands that modify state
- Read files, grep patterns, report findings
- Be thorough — check every file mentioned above
- If uncertain whether something is a duplication, flag it as MEDIUM severity with explanation
