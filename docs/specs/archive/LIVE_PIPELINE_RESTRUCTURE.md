# Live Pipeline Restructure Spec

**Date**: 2026-03-16
**Status**: Draft
**Goal**: Live Engine becomes a thin shell feeding bars to the shared BarProcessor. One codepath for OOS, replay, and live.

---

## 1. Current Architecture (Problems)

**live_engine.py** = 1,954 lines / ~94KB. Three concerns tangled:
1. NT8 plumbing (connect, heartbeat, reconnect, message dispatch)
2. Bar aggregation + state management (warmup, TBN feeding, regression)
3. Trading logic (_check_entry, _check_exit, candidate building, exit evaluation)

**Three codepaths exist today for the same per-bar decision:**
| Codepath | File | Used By |
|----------|------|---------|
| OOS compressed | `training/trainer.py` via `BarProcessor` | Phase 4b |
| History replay | `live/history_replay.py` via `BarProcessor` | Phase 7 |
| Live trading | `live/live_engine.py` (manual inline logic) | Production |

`BarProcessor` (`core/bar_processor.py`) already unifies OOS and replay. Live partially uses it (imports exist, `_create_processor()` at line 1929) but still has ~800 lines of duplicate `_check_entry`/`_check_exit` logic wrapping around it. This is the root of live/OOS parity failures.

**Specific duplication in live_engine.py:**
- `_check_entry()` (line 1626, ~300 lines) — rebuilds candidates, feature extraction, EE gates
- `_check_exit()` (line 732, ~200 lines) — exit cascade evaluation, trade recording
- `_process_15s()` (line 658) — bar-level orchestration that BarProcessor.process_bar() already does
- Ping-pong direction logic interleaved with entry/exit

---

## 2. Target Architecture

```
LiveEngine (thin async shell, ~300 lines)
  ├── NT8Client           (connection, heartbeat, reconnect)
  ├── LiveBarAggregator   (tick/1s -> multi-TF bars, regression state)
  ├── OrderManager        (fill tracking, position shadow, NT8 order dispatch)
  ├── Watchdog            (new: maintenance windows, daily loss limit, shutdown)
  └── on_bar() -> BarProcessor.process_bar()   ← SAME instance as OOS
        ├── TBN workers (belief network update)
        ├── Pattern matching + scoring (EE gate cascade)
        ├── Exit engine evaluation
        └── Returns BarResult -> LiveEngine routes to OrderManager -> NT8
```

**Key invariant**: `BarProcessor.process_bar()` is the ONLY code that makes entry/exit decisions. LiveEngine never inspects features, runs gates, or evaluates exits directly.

**BarProcessorHooks (already exist) handle live-specific side effects:**
- `on_entry` -> OrderManager.submit_entry() + TradeLogger.log_entry()
- `on_exit` -> OrderManager.submit_exit() + SessionTracker.record() + ExitWatcher.start()
- `on_bar` -> GUIBridge.push_state()

---

## 3. File Disposition

### KEEP (10) — no structural changes
| File | Role | Lines |
|------|------|-------|
| `nt8_client.py` | TCP bridge to NT8 | ~200 |
| `order_manager.py` | Fill tracking, position shadow | ~250 |
| `bar_aggregator.py` | 1s -> multi-TF aggregation + regression | ~300 |
| `launcher.py` | CLI entry point, mode routing | ~150 |
| `config.py` | LiveConfig dataclass | ~45 |
| `protocol.py` | Wire format encode/decode | ~200 |
| `session_tracker.py` | PnL, drawdown, session reports | ~200 |
| `trade_logger.py` | Per-trade diagnostic CSV | ~100 |
| `exit_watcher.py` | Post-exit counterfactual tracking | ~150 |
| `gui_bridge.py` | Non-blocking queue for Tk dashboard | ~80 |

### MERGE (2) — extract reusable logic, then gut
| File | Extract To | What Remains |
|------|-----------|--------------|
| `live_engine.py` (1,954 lines) | Trading logic deleted (BarProcessor owns it). NT8 event loop + hook wiring stays. Target: ~300 lines. | Thin async shell: connect, warm up, feed bars, route signals. |
| `ping_pong.py` | Direction-flip logic moves into `BarProcessorHooks.on_exit` callback. ATR sizing stays in PingPongManager but called from hook, not inline. | Stateless helper, no direct bar access. |

### DELETE (3) — replaced by shared BarProcessor path
| File | Reason |
|------|--------|
| `history_replay.py` | Phase 7 replay moves to trainer.py using BarProcessor directly (same as OOS). No separate replay engine needed. |
| `replay_bridge.py` | Scaffolding for history_replay. Unnecessary when replay is just another BarProcessor consumer. |
| `atlas_loader.py` | Used only by history_replay. Trainer already has its own ATLAS loading. |

---

## 4. Warmup Protocol

**Problem**: BarProcessor needs warm TBN + regression state before the first signal. OOS prepends IS tail bars. Live needs equivalent.

**Solution**:
1. On startup, NT8Client sends `REQUEST_HISTORY` for 10,000 bars (configurable via `LiveConfig.warmup_bars`)
2. NT8 bridge responds with historical OHLCV batch
3. LiveBarAggregator feeds history through StatisticalFieldEngine (builds regression bands)
4. Each bar also fed to `BarProcessor.process_bar()` with `warmup=True` flag
   - TBN workers update beliefs (state accumulates)
   - Entry/exit evaluation skipped (no signals during warmup)
5. After warmup completes, LiveEngine transitions to live signal mode
6. First live bar index = warmup_count + 1 (continuous indexing)

**Parity guarantee**: Same warmup logic as IS-tail-prepend in OOS compressed path. The BarProcessor sees the same sequence of `process_bar()` calls regardless of caller.

---

## 5. Migration Phases

### Phase A: Verify BarProcessor Coverage
**Goal**: Confirm BarProcessor.process_bar() handles every decision live_engine makes today.
- Audit `_check_entry()` and `_check_exit()` line by line against BarProcessor
- Document any live-only logic that BarProcessor lacks (likely: ping-pong direction flip, session-aware gating)
- Add missing capabilities to BarProcessor via hooks (NOT by adding live-specific code to BarProcessor core)
- **Gate**: Diff report showing 1:1 coverage

### Phase B: Wire LiveEngine to BarProcessor
**Goal**: LiveEngine calls `process_bar()` instead of inline logic.
- Replace `_process_15s()` -> `self._processor.process_bar()`
- Replace `_check_entry()` -> hook-driven (BarProcessor calls `on_entry` hook)
- Replace `_check_exit()` -> hook-driven (BarProcessor calls `on_exit` hook)
- Keep old methods commented out (not deleted) for one release cycle
- **Gate**: Replay parity test — run Phase 7 replay via BarProcessor, diff results against current history_replay output. Zero divergence.

### Phase C: Delete Duplicate Logic
**Goal**: LiveEngine < 400 lines.
- Delete `_check_entry()`, `_check_exit()`, `_process_15s()`, all inline candidate/feature code
- Move ping-pong flip logic into hook callback
- Extract Watchdog (daily loss limit, maintenance window) into `live/watchdog.py` (~50 lines)
- **Gate**: Live dry-run produces identical signals to Phase B output

### Phase D: Delete Replay Stack
**Goal**: One fewer codepath.
- Delete `history_replay.py`, `replay_bridge.py`, `atlas_loader.py`
- Move Phase 7 into trainer.py: load ATLAS data, feed through BarProcessor with live hooks disabled
- Update launcher.py `--replay-only` to call trainer's Phase 7 directly
- **Gate**: `--replay-only` output matches pre-migration replay output

---

## 6. Risk Assessment

### What breaks if BarProcessor has a bug?
**Blast radius**: ALL paths (OOS, replay, live) break simultaneously. This is both the risk and the benefit — a bug is caught in training before it reaches live.

**Mitigations**:
- BarProcessor is already battle-tested in OOS and replay. The risk is in wiring, not in the processor itself.
- Hooks isolate live-specific behavior. A hook bug only affects live, not training.
- `--dry-run` mode: no orders sent to NT8. Full signal generation with zero capital risk.
- NT8 sim account: even with orders, sim account catches sizing/direction errors.

### Rollback plan
- Phase B keeps old methods commented out. Rollback = uncomment + revert hook wiring (~10 min).
- Git branch per phase. Each phase merges only after gate passes.
- If Phase C or D breaks: `git revert` to Phase B (BarProcessor wired but old code still present).

### Testing strategy
| Layer | Method |
|-------|--------|
| Unit | BarProcessor.process_bar() with synthetic MarketState sequences |
| Integration | Feed ATLAS_1DAY through LiveEngine in dry-run, compare signals to OOS output |
| Parity | Phase 7 replay before/after migration — zero diff on trade log |
| Smoke | 30-min live dry-run against NT8 sim, verify signal timing + order submission |
| Regression | Full `--fresh` run, compare IS/OOS metrics to current baseline |

### Residual risks
| Risk | Severity | Mitigation |
|------|----------|------------|
| Warmup state drift (live vs OOS tail-prepend) | Medium | Log first 10 live signals + manual spot-check against OOS |
| Async timing (live is event-driven, OOS is sequential) | Medium | BarProcessor is synchronous — async boundary stays in LiveEngine shell |
| Hook ordering (entry hook fails but BarProcessor already updated state) | Low | Hooks are post-decision notifications, not pre-decision gates. State is always consistent. |
| Ping-pong regression (direction logic moves) | Medium | Dedicated ping-pong integration test with known flip scenarios |

---

## 7. Success Criteria

1. `live_engine.py` < 400 lines (from 1,954)
2. Zero trading-logic imports in live_engine.py (no direct use of ExitEngine, ExecutionEngine, feature_extraction)
3. `--replay-only` and OOS produce identical trade logs for same data
4. Live dry-run signal timing within 1 bar of OOS (accounting for warmup)
5. `history_replay.py`, `replay_bridge.py`, `atlas_loader.py` deleted
6. All existing tests pass without modification
