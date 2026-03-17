# Live Pipeline Phase A: Coverage Audit

**Date**: 2026-03-16
**Status**: COMPLETE — BarProcessor covers all trading logic

## BarProcessor Coverage Verification

| live_engine.py function | BarProcessor equivalent | Status |
|------------------------|------------------------|--------|
| `_check_entry()` (L1626, ~300 lines) | `process_bar()` → `_build_candidates()` → `exec_engine.on_bar()` | COVERED |
| `_check_exit()` (L732, ~200 lines) | `_process_exit()` → `get_exit_signal()` → `exec_engine.on_bar()` | COVERED |
| `_process_15s()` (L658) | `process_bar()` (top-level orchestrator) | COVERED |
| Feature extraction (inline) | `_build_features()` → `extract_feature_vector()` | COVERED |
| Trade recording | `_handle_exit()` → `record_trade()` | COVERED |
| TBN tick | `process_bar()` line 190: `tick_all(bar_index)` | COVERED |
| Exit signal gathering | `_process_exit()` lines 229-236 | COVERED |
| Self-tuning exits | `_handle_exit()` lines 364-372 | COVERED |
| Ping-pong direction | `pp_dir_override` param in `process_bar()` | COVERED (param) |
| Trade tracking (TBN) | `_handle_entry()` lines 304-314 | COVERED |

## Live-Only Logic NOT in BarProcessor (stays in LiveEngine shell)

| Function | What it does | Where it stays |
|----------|-------------|----------------|
| NT8 connect/reconnect | TCP lifecycle | `nt8_client.py` |
| Bar aggregation (tick→15s) | Multi-TF buffer management | `bar_aggregator.py` |
| Order submission | Market/limit/OCO to NT8 | `order_manager.py` |
| Position shadow sync | Reconcile local vs NT8 state | `order_manager.py` |
| Session PnL tracking | Daily stats, drawdown | `session_tracker.py` |
| Warmup (10K bars) | Historical bar request | `live_engine.py` (keep) |
| Maintenance flatten | CME 16:15 cutoff | `watchdog.py` (new) |
| Daily loss limit | Stop trading if DD > limit | `watchdog.py` (new) |
| GUI updates | State → dashboard queue | `gui_bridge.py` via hooks |
| Ping-pong flip submission | Deferred re-entry after exit | hooks + `ping_pong.py` |

## Missing from BarProcessor (needs hook wiring)

1. **discovery_tf_seconds in exit_signal** — BarProcessor's `_process_exit()` calls
   `get_exit_signal(side, entry_price)` but doesn't pass `discovery_tf_seconds`.
   Fix: read from `exec_engine.pos_state.discovery_tf_seconds`.

2. **Dashboard TICK_UPDATE / TRADE_MARKER** — BarProcessor doesn't push to GUI.
   Fix: Add to hooks (`on_bar` callback pushes price, `on_entry`/`on_exit` push markers).

3. **Slippage injection** — `modify_pnl` hook exists but needs calibration data.
   Fix: Wire tunnel_probability → slippage model (future).

4. **Macro observation columns** — trainer adds `_macro_obs()` to trade log.
   Fix: Add to `on_entry` hook or extend BarResult.

## Gate: Phase A PASSED

BarProcessor.process_bar() handles every decision the live engine makes.
Proceed to Phase B: Wire LiveEngine to call BarProcessor.
