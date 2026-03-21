# IS Inline Path -> BarProcessor Refactor

## What to do
Replace lines 1139-2167 in trainer.py (1,029 lines) with process_bar() calls.
The OOS path (lines 2567-2613) already does this in ~20 lines.

## The IS loop has two interleaved concerns:

### A. Entry/Exit decisions (DELETE — BarProcessor handles)
- Peak detection: _bp._detect_peak_reversal + _1m_confirms_peak (~30 lines)
- Compressed signal check (~10 lines)
- Candidate building + gate cascade: _exec_engine.on_bar() (~50 lines)
- Exit evaluation: _exec_engine.on_bar() for exits (~80 lines)
- Maintenance window flatten (~50 lines)
- Ping-pong flip after exit (~30 lines)

### B. Bookkeeping (KEEP — move to hooks or loop wrapper)
- Per-day progress update + daily ledger flush (~50 lines)
- Dashboard push (~30 lines)
- Oracle tracking: pending_oracle dict, capture rate (~80 lines)
- Trade recording: record_trade + oracle_trade_records append (~100 lines)
- Equity tracking: running_equity, drawdown, ruin (~30 lines)
- Streaming trade log: _stream_trade() (~10 lines)
- Trade replay capture (~20 lines)
- Cumulative PnL + drawdown tracking (~20 lines)
- Daily DD stop circuit breaker (~10 lines)
- TBN tick_all (~3 lines)
- PID analyzer tick (~10 lines)

## Refactored loop structure (~100 lines):

```python
for row in df_15s.itertuples():
    total_bars_processed += 1
    ts_raw = row.timestamp
    price = row.close
    _bar_i += 1

    # A. Day boundary bookkeeping (same as before)
    _row_day = int(ts_raw) // 86400
    if _row_day != _current_day:
        # ... daily ledger flush (keep as-is, ~30 lines)

    # B. TBN tick + dashboard (same as before, ~5 lines)
    belief_network.tick_all(_bar_i + _warmup_offset)

    # C. Daily DD stop check (~5 lines)
    if _daily_dd_stopped:
        continue

    # D. THE REFACTORED PART: process_bar() replaces 300+ lines
    _bar_state = _states_map.get(_bar_i)
    if _bar_state is None:
        continue

    result = _bp.process_bar(
        bar_index=_bar_i,
        price=price,
        bar_high=row.high,
        bar_low=row.low,
        timestamp=ts_raw,
        state=_bar_state,
    )

    # E. Handle result (replaces inline entry/exit logic)
    if result.trade_completed:
        trade = result.trade_completed
        pnl = trade['pnl']

        # Slippage
        if _slip_rng:
            pnl += _slip_rng.uniform(-_slip_ticks, _slip_ticks) * _tick_val

        # Record to brain
        outcome = record_trade(self.brain, ...)
        day_trades.append(outcome)
        _cal_day_trades.append(outcome)

        # Oracle tracking
        oracle_trade_records.append({...})
        _stream_trade(oracle_trade_records[-1])

        # Equity tracking
        _day_running_pnl += pnl
        _cumul_pnl += pnl
        # ... drawdown updates
```

## BarProcessorHooks needed:
- pre_exit_eval: provide sub_bar_highs, sub_bar_lows from 1s data
- modify_pnl: slippage injection
- on_exit: already exists, wire trade recording

## Verification:
1. Run python training/trainer.py --data DATA/ATLAS_1WEEK (5 min)
2. Compare trade count + PnL against last full run
3. If within 5%: refactor is clean
4. Run full IS+OOS and compare scorecards
