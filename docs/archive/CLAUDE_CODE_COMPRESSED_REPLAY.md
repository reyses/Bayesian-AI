# CLAUDE CODE SPEC: Compressed History Replay Engine

## CONFIDENTIAL — BayesianBridge Internal

---

## Problem Statement

Live engine cold-starts are slow and fragile:
- NT8 pumps bars one-by-one over TCP (~3-5 min for 10K bars)
- Disconnect mid-history = full restart
- Brain direction bias is cold (no learned preferences)
- No validation that live logic would reproduce OOS results
- TBN workers start empty — first 5-10 minutes of signals have no conviction

**Goal:** Replace NT8 history dump with a local compressed forward pass that:
1. Loads ATLAS parquet directly from disk (seconds, not minutes)
2. Runs full gate cascade + direction + entry/exit (validates against OOS)
3. Warms brain.dir_bias, brain.dir_table, TBN workers, exit engine self-tune
4. Produces a validation report (WR, PnL, trade count vs checkpoint OOS)
5. Hands off fully warmed state to live engine
6. NT8 only provides delta bars from last replay timestamp forward

---

## Architecture

```
ATLAS Parquet (disk)
    │
    ▼
┌─────────────────────┐
│  HistoryReplayEngine │  (new file: live/history_replay.py)
│                      │
│  1. Load N days of   │
│     multi-TF data    │
│  2. batch_compute    │
│     _states() per TF │
│  3. TBN.prepare_day  │
│  4. Forward pass:    │
│     ExecutionEngine  │◄── REUSE, not reimplement
│     .on_bar()        │
│  5. Validation report│
│  6. Return warm state│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    LiveEngine        │
│                      │
│  Receives:           │
│  - warmed brain      │
│  - warmed TBN        │
│  - warmed exit_engine│
│  - last_timestamp    │
│  - validation_report │
│                      │
│  Requests RESUME_FROM│
│  last_timestamp only │
└─────────────────────┘
```

---

## CRITICAL DESIGN DECISION: Reuse ExecutionEngine

**The live engine currently reimplements gates 0-4 inline in `_check_entry()`.
This replay MUST use `ExecutionEngine` from `core/execution_engine.py` — the same
module used in training/OOS. This forces direction cascade parity.**

If the replay produces different results than OOS, the divergence is in the
live engine's reimplemented gates — which must then be fixed before going live.

After this spec is complete, `LiveEngine._check_entry()` should be refactored
to delegate to `ExecutionEngine` as well (separate task, not this spec).

---

## File: `live/history_replay.py`

### Class: `HistoryReplayEngine`

```python
class HistoryReplayEngine:
    """
    Compressed forward pass over historical ATLAS data.
    Produces fully warmed state for live handoff.
    
    Usage:
        replay = HistoryReplayEngine(config, checkpoints)
        result = replay.run(n_days=5)
        # result.brain, result.tbn, result.exit_engine, result.last_timestamp
        # result.validation_report  (WR, PnL, trade count)
    """
```

### Constructor Parameters

| Param | Type | Source |
|-------|------|--------|
| `config` | `LiveConfig` | Same config as LiveEngine |
| `checkpoint_dir` | `str` | Path to checkpoints/ |
| `n_days` | `int` | Days of history to replay (default: 5) |
| `atlas_root` | `str` | Path to `DATA/ATLAS/` |
| `anchor_tf` | `str` | Primary signal TF (default: '15s') |
| `validate_against_oos` | `bool` | Compare results to OOS checkpoint (default: True) |

### Return: `ReplayResult` dataclass

```python
@dataclass
class ReplayResult:
    brain: QuantumBayesianBrain       # warmed with direction bias
    belief_network: TimeframeBeliefNetwork  # all workers populated
    exit_engine: ExitEngine           # self-tune calibrated
    execution_engine: ExecutionEngine # gate stats populated
    last_timestamp: float             # for RESUME_FROM
    validation: ValidationReport      # comparison to OOS
    states_micro: list                # latest day's states (for TBN)
    df_micro: pd.DataFrame            # latest day's bars
```

```python
@dataclass
class ValidationReport:
    replay_trades: int
    replay_wr: float
    replay_pnl: float
    replay_avg_trade: float
    oos_trades: int          # from checkpoint (0 if not available)
    oos_wr: float
    oos_pnl: float
    oos_avg_trade: float
    parity_score: float      # 0.0-1.0, how close replay matches OOS
    gate_stats: dict          # from ExecutionEngine.get_skip_counts()
    direction_source_dist: dict  # {source: count} for dir cascade
    warnings: list[str]       # divergence flags
    passed: bool              # True if parity_score > 0.80
```

---

## Implementation Steps

### Step 1: ATLAS Data Loader

**File location:** `live/atlas_loader.py`

**Function:** `load_atlas_range(atlas_root, tf, n_days, end_date=None) -> pd.DataFrame`

Logic:
1. Determine which `YYYY_MM.parquet` files cover the requested date range
2. Read and concatenate (parquet read is ~100ms for a month of 15s data)
3. Filter to last N trading days (exclude weekends, CME maintenance windows)
4. Return DataFrame with columns: `timestamp, open, high, low, close, volume`

**Multi-TF loading:** Load parquet for each TF that TBN workers need:
- `15s` (primary) — always
- `5s` (sub-resolution worker) — if available
- `1s` (sub-resolution worker) — if available  
- `30s, 1m, 3m, 5m, 15m, 30m, 1h` — TBN resamples from 15s, no separate load needed
- `4h` — load separately (supra-resolution, can't resample from 15s in one day)

```python
def load_multi_tf(atlas_root: str, n_days: int = 5) -> dict:
    """Returns {tf_label: pd.DataFrame} for all needed TFs."""
    result = {}
    result['15s'] = load_atlas_range(atlas_root, '15s', n_days)
    
    # Optional sub-resolution
    for tf in ['5s', '1s']:
        try:
            result[tf] = load_atlas_range(atlas_root, tf, n_days)
        except FileNotFoundError:
            pass  # worker stays inactive
    
    # Supra-resolution (4h needs more history for meaningful states)
    try:
        result['4h'] = load_atlas_range(atlas_root, '4h', n_days * 4)
    except FileNotFoundError:
        pass
    
    return result
```

### Step 2: Day-by-Day Forward Pass

**Why day-by-day:** TBN `prepare_day()` resamples micro bars to each TF level
and computes states once. This matches how the training orchestrator works.
Running all 5 days at once would produce incorrect TBN state (bars from day 1
shouldn't bleed into day 5's TF aggregation).

```python
def run(self) -> ReplayResult:
    # 1. Load checkpoints (same as LiveEngine._load_checkpoints)
    self._load_checkpoints()
    
    # 2. Load ATLAS data
    tf_data = load_multi_tf(self.atlas_root, self.n_days)
    df_15s = tf_data['15s']
    
    # 3. Split into trading days (6PM-5PM ET boundaries)
    days = split_trading_days(df_15s)
    
    # 4. Initialize components
    engine = StatisticalFieldEngine()
    brain = QuantumBayesianBrain()
    brain.load(...)  # from checkpoint
    exit_engine = ExitEngine(mode='training', ...)
    
    exec_engine = ExecutionEngine(
        brain=brain,
        belief_network=tbn,
        exit_engine=exit_engine,
        pattern_library=self.pattern_library,
        scaler=self.scaler,
        centroids_scaled=self.centroids_scaled,
        valid_tids=self.valid_tids,
        # ... same params as training orchestrator
    )
    
    # 5. Forward pass per day
    all_trades = []
    for day_df in days:
        day_trades = self._replay_day(
            day_df, tf_data, engine, exec_engine, tbn)
        all_trades.extend(day_trades)
    
    # 6. Build validation report
    report = self._build_validation(all_trades)
    
    # 7. Package result
    return ReplayResult(
        brain=brain,
        belief_network=tbn,
        exit_engine=exit_engine,
        execution_engine=exec_engine,
        last_timestamp=df_15s.iloc[-1]['timestamp'],
        validation=report,
        states_micro=last_day_states,
        df_micro=days[-1],
    )
```

### Step 3: Single Day Replay

```python
def _replay_day(self, day_df, tf_data, engine, exec_engine, tbn):
    """Compressed forward pass for one trading day.
    
    Mirrors orchestrator Phase 4 logic exactly:
      1. batch_compute_states (all bars at once — no per-bar loop for physics)
      2. TBN prepare_day (resample + compute TF states)
      3. Per-bar loop: TBN tick + ExecutionEngine.on_bar()
      4. Record trades for validation
    """
    # 1. Compute states (bulk — ~0.5s per day on GPU)
    states = engine.batch_compute_states(day_df, use_cuda=True)
    
    # 2. Prepare TBN workers
    df_5s = _slice_day(tf_data.get('5s'), day_df)
    df_1s = _slice_day(tf_data.get('1s'), day_df)
    df_4h = _slice_day(tf_data.get('4h'), day_df)
    tbn.prepare_day(day_df, states_micro=states,
                    df_5s=df_5s, df_1s=df_1s, df_4h=df_4h)
    
    # 3. Per-bar loop (fast — no I/O, no sleep, pure compute)
    trades = []
    for bar_i, result in enumerate(states):
        state = result['state']
        price = result['price']
        bar_high = day_df.iloc[bar_i + engine.regression_period]['high']
        bar_low = day_df.iloc[bar_i + engine.regression_period]['low']
        
        # Tick TBN
        tbn.tick_all(bar_i)
        
        # Build candidates (same logic as live _check_entry)
        candidates = []
        if state.cascade_detected or state.structure_confirmed:
            candidates.append(Candidate(
                state=state,
                depth=self.anchor_depth,
                timeframe=self.anchor_tf,
                timestamp=state.timestamp,
                pattern_type=state.pattern_type,
                z_score=state.z_score,
            ))
        
        # ExecutionEngine handles everything: gates, direction, exit
        action = exec_engine.on_bar(
            price=price,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_index=bar_i,
            candidates=candidates if not exec_engine.in_position else None,
        )
        
        if action.type == ActionType.ENTER:
            exec_engine.position_opened(
                side=action.side,
                price=action.price,
                bar_index=bar_i,
                template_id=action.template_id,
                lib_entry=action.lib_entry,
                sl_ticks=action.sl_ticks,
                tp_ticks=action.tp_ticks,
                max_hold_bars=action.max_hold_bars,
            )
            # Record entry for trade log
            _current_entry = {
                'side': action.side, 'entry_price': price,
                'entry_bar': bar_i, 'tid': action.template_id,
                'dir_source': action.dir_source,
            }
        
        elif action.type == ActionType.EXIT:
            # Record completed trade
            pnl_dollars = action.pnl_dollars
            trades.append({
                **_current_entry,
                'exit_price': action.price,
                'exit_bar': bar_i,
                'pnl': pnl_dollars,
                'exit_reason': action.exit_reason,
                'bars_held': action.bars_held,
            })
            
            # Brain learning (direction + outcome)
            brain.direction_learn(
                _current_entry['tid'],
                _current_entry['side'],
                pnl_dollars,
            )
            
            # Exit engine self-tune
            # (compute capture rate from MFE tracking)
            exit_engine.record_trade_outcome(...)
            
            exec_engine.position_closed()
    
    return trades
```

### Step 4: Validation Report

```python
def _build_validation(self, trades: list) -> ValidationReport:
    """Compare replay results to OOS checkpoint numbers."""
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    wr = wins / n if n > 0 else 0
    avg = total_pnl / n if n > 0 else 0
    
    # Load OOS reference (from training report or checkpoint)
    oos = self._load_oos_reference()
    
    # Direction source distribution
    dir_dist = {}
    for t in trades:
        src = t.get('dir_source', 'unknown')
        dir_dist[src] = dir_dist.get(src, 0) + 1
    
    # Parity scoring
    warnings = []
    parity = 1.0
    
    if oos and oos['trades'] > 0:
        # WR within 5 percentage points
        wr_delta = abs(wr - oos['wr'])
        if wr_delta > 0.05:
            warnings.append(f"WR diverged: replay={wr:.1%} vs OOS={oos['wr']:.1%}")
            parity -= min(0.3, wr_delta * 3)
        
        # Avg trade within 50%
        if oos['avg_trade'] > 0:
            avg_ratio = avg / oos['avg_trade']
            if avg_ratio < 0.5 or avg_ratio > 2.0:
                warnings.append(f"Avg trade diverged: ${avg:.2f} vs OOS ${oos['avg_trade']:.2f}")
                parity -= 0.2
        
        # Trade frequency within 30%
        # Scale replay trades to OOS day count for fair comparison
        replay_days = self.n_days
        oos_days = oos.get('days', 1)
        replay_rate = n / max(1, replay_days)
        oos_rate = oos['trades'] / max(1, oos_days)
        rate_ratio = replay_rate / max(0.01, oos_rate)
        if rate_ratio < 0.7 or rate_ratio > 1.3:
            warnings.append(f"Trade frequency diverged: {replay_rate:.1f}/day vs OOS {oos_rate:.1f}/day")
            parity -= 0.15
    else:
        warnings.append("No OOS reference available — cannot validate")
        parity = 0.5  # unknown
    
    parity = max(0.0, min(1.0, parity))
    
    return ValidationReport(
        replay_trades=n,
        replay_wr=wr,
        replay_pnl=total_pnl,
        replay_avg_trade=avg,
        oos_trades=oos.get('trades', 0) if oos else 0,
        oos_wr=oos.get('wr', 0) if oos else 0,
        oos_pnl=oos.get('pnl', 0) if oos else 0,
        oos_avg_trade=oos.get('avg_trade', 0) if oos else 0,
        parity_score=parity,
        gate_stats=self.exec_engine.get_skip_counts(),
        direction_source_dist=dir_dist,
        warnings=warnings,
        passed=parity >= 0.80,
    )
```

---

## Integration with LiveEngine

### Modified `LiveEngine.run()` flow:

```python
async def run(self):
    # BEFORE: _load_checkpoints() then wait for NT8 history dump
    # AFTER:  compressed replay FIRST, then NT8 delta only
    
    # 1. Run compressed replay (blocking, ~2 min)
    replay = HistoryReplayEngine(
        config=self._cfg,
        checkpoint_dir=self._cfg.checkpoint_dir,
        n_days=5,
        atlas_root='DATA/ATLAS',
    )
    result = replay.run()
    
    # 2. Validation gate — refuse to go live if parity is bad
    if not result.validation.passed:
        logger.error("REPLAY VALIDATION FAILED — refusing to go live")
        for w in result.validation.warnings:
            logger.error(f"  WARNING: {w}")
        logger.error(f"  Parity score: {result.validation.parity_score:.2f}")
        return  # DO NOT PROCEED
    
    # 3. Transfer warmed state
    self._brain = result.brain
    self._belief_network = result.belief_network
    self._exit_engine = result.exit_engine
    self._last_states = result.states_micro
    
    # 4. Log validation results
    logger.info(f"REPLAY VALIDATED: {result.validation.replay_trades} trades, "
                f"WR={result.validation.replay_wr:.1%}, "
                f"PnL=${result.validation.replay_pnl:+,.2f}, "
                f"Parity={result.validation.parity_score:.2f}")
    
    # 5. Connect to NT8 — delta sync only
    self._client.set_resume_timestamp(result.last_timestamp)
    connected = await self._client.connect()
    
    # 6. Normal live loop (starts with warm state, not cold)
    await self._main_loop()
```

### Modified aggregator behavior:

The aggregator no longer needs `_history_mode`. It starts pre-populated:

```python
# After replay, seed aggregator with latest day's bars
self._aggregator._rows = result.df_micro.to_dict('records')
self._aggregator._states = result.states_micro
self._aggregator._warmed_up = True
# Delta bars from NT8 append normally
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `live/atlas_loader.py` | Load ATLAS parquet for date ranges, multi-TF |
| `live/history_replay.py` | `HistoryReplayEngine` + `ReplayResult` + `ValidationReport` |

## Files to Modify

| File | Change |
|------|--------|
| `live/live_engine.py` | (1) Insert replay before connect; transfer warm state. (2) **DELETE `_check_entry()`, `_determine_direction()`, and all inline gate logic. Replace with thin wrapper around `ExecutionEngine.on_bar()`.** (3) Delete `_gate_stats` dict (use `exec_engine.gate_stats` instead). |
| `live/bar_aggregator.py` | Add `seed_from_replay(df, states)` method to accept pre-computed state |
| `live/launcher.py` | Add `--skip-replay` flag for debugging (bypass validation) |
| `live/config.py` | Add `atlas_root`, `replay_days`, `replay_validate` fields |
| `core/execution_engine.py` | Add `live_momentum` priority level to `_direction_cascade()` (see below) |

---

## PART 2: Thin Wrapper — LiveEngine Delegates to ExecutionEngine

### Problem

`LiveEngine` currently reimplements the full gate cascade (gates 0-4), direction
determination, and exit sizing inline in `_check_entry()` (~200 lines) and
`_determine_direction()` (~80 lines). This duplicated logic has **already diverged**
from `ExecutionEngine`:

| Aspect | ExecutionEngine (training/OOS) | LiveEngine (live) |
|--------|-------------------------------|-------------------|
| Direction priorities | 7 levels: pp_override → signed_mfe → logistic → brain_dir → template_bias → band_confluence → dmi → velocity | Different order: live_bias → brain_dir → **live_momentum** (doesn't exist in EE) → signed_mfe → balanced_dir → template_bias → band → dmi → velocity |
| Gate 0 physics | Uses `self.hurst_min`, `self.tunnel_prob_min`, `self.momentum_override_ratio` from `gate_thresholds.json` | Hardcoded `_ADX_TREND_CONFIRMATION=25`, `_HURST_TREND_CONFIRMATION=0.6` from module constants, plus tuning overrides |
| Gate 1 threshold | Fixed `self.gate1_dist = 4.5` | Dynamic: `_g1_base + agg * 10.0` (aggression scaling) |
| Gate 2 brain | `min_prob=0.05, min_conf=0.0` | `min_prob=0.05*(1-agg), min_conf=0.0` (aggression scaling) |
| Gate 3 conviction | `belief.is_confident` check | Same but skipped at `agg > 0.75` |
| Gate 4 | `gate4_momentum_align` (F_momentum vs trade direction) | `_dir_conf < _g4_dir` (direction confidence threshold) — **completely different gate** |
| Exit sizing | Template library stats: p25_mae, mean_mae, reg_sigma, p75_mfe | **ATR-based**: `atr * multiplier` from live_tuning.json — **different source** |
| Depth filter | `depth_blacklist`, `depth_filter_out`, `depth_only` | `depth_filter_out` only (no blacklist) |
| Score competition | tier_adj + depth_adj across all candidates | Same structure but fewer candidates (live has 1-2 per bar, training has many) |
| Feature extraction | `cand.features` or `feature_extractor(raw_event)` | `FractalClusteringEngine.extract_features(p)` inline |

**Any WR measured in OOS does not apply to live.** The direction cascade alone
determines which side every trade takes. Different priorities = different trades.

### Solution: Delete and Delegate

**Delete from `live_engine.py`:**
- `_check_entry()` — ~200 lines of reimplemented gates
- `_determine_direction()` — ~80 lines of reimplemented direction cascade
- `_compute_exit_params()` — ~30 lines (EE handles sizing)
- `_gate_stats` dict — use `self._exec_engine.gate_stats` instead
- All gate constants at module level: `_ADX_TREND_CONFIRMATION`, `_HURST_TREND_CONFIRMATION`, `_GATE1_DIST_THRESHOLD`, `_WORKER_BYPASS_CONV`

**Replace with thin wrapper** (~60 lines total):

```python
# In LiveEngine.__init__(), after _load_checkpoints():
self._exec_engine = ExecutionEngine(
    brain=self._brain,
    belief_network=self._belief_network,
    exit_engine=self._exit_engine,
    pattern_library=self._pattern_library,
    scaler=self._scaler,
    centroids_scaled=self._centroids_scaled,
    valid_tids=self._valid_tids,
    tick_size=self._asset.tick_size,
    point_value=self._asset.point_value,
    mode='live',
    tier_score_adj=self._tier_score_adj,
    depth_score_adj=self._depth_score_adj,
    template_tier_map=self._template_tier_map,
    exception_tids=self._exception_tids,
    depth_filter_out=self._depth_filter_out,
    feature_extractor=lambda p: FractalClusteringEngine.extract_features(p),
)
```

```python
async def _check_entry(self, price: float, ts: float, states: list):
    """Thin wrapper: build candidates, delegate to ExecutionEngine."""
    if not states or self._instrument_mismatch:
        self._entry_belief_pct = 0
        return
    
    latest = states[-1]
    state = latest['state']
    
    # Build candidates (same as before — this is live-specific because
    # live has no fractal tree, just the current state)
    candidates = []
    if state.cascade_detected or state.structure_confirmed:
        cand = self._build_candidate(state, price, ts)
        candidates.append(Candidate(
            state=state,
            depth=self._anchor_depth,
            timeframe=self._anchor_tf,
            timestamp=ts,
            pattern_type=state.pattern_type,
            z_score=state.z_score,
            raw_event=cand,   # _LiveCandidate for feature extraction
        ))
    
    # YOLO: force candidate if nothing triggered
    agg = self._shared_state.get('aggression', 0.5)
    if not candidates and agg >= 0.99:
        cand = self._build_candidate(state, price, ts)
        candidates.append(Candidate(
            state=state, depth=self._anchor_depth,
            timeframe=self._anchor_tf, timestamp=ts,
            pattern_type='MOMENTUM_BREAK', z_score=state.z_score,
            raw_event=cand,
        ))
    
    if not candidates:
        self._entry_belief_pct = 0
        return
    
    # Apply aggression scaling to EE thresholds (hot-tunable)
    self._exec_engine.gate1_dist = (
        float('inf') if agg >= 0.99
        else self._tuning.get('gate1_dist', 4.5) + agg * 10.0
    )
    
    # Side lock override
    pp_dir = None
    _side_lock = self._shared_state.get('side_lock')
    if _side_lock:
        pp_dir = _side_lock
    
    # === THE DELEGATION — one call replaces 200 lines ===
    action = self._exec_engine.on_bar(
        price=price,
        bar_high=getattr(self, '_last_bar_high', price),
        bar_low=getattr(self, '_last_bar_low', price),
        bar_index=self._bar_i,
        candidates=candidates,
        band_context=self._belief_network.get_band_confluence(),
        pp_dir_override=pp_dir,
    )
    
    if action.type == ActionType.ENTER:
        # ExecutionEngine decided to enter — execute it
        await self._execute_entry(action, price, ts)
    else:
        # Compute belief % from gate progress for GUI
        self._entry_belief_pct = self._gate_label_to_pct(action.gate_label)
```

```python
async def _execute_entry(self, action: TradeAction, price: float, ts: float):
    """Execute an ENTER action from ExecutionEngine.
    
    Handles: position creation, exit state init, NT8 order,
    TBN tracking, GUI push, trade logger. All the live-specific
    plumbing that ExecutionEngine correctly doesn't own.
    """
    side = action.side
    tid = action.template_id
    
    # Apply tuning offsets to EE-computed sizing
    _floor = max(4, self._tuning.get('min_tick_floor', 4))
    sl_ticks = max(_floor, action.sl_ticks + self._tuning.get('sl_offset', 0))
    tp_ticks = max(_floor, action.tp_ticks + self._tuning.get('tp_offset', 0))
    trail_ticks = max(_floor, action.trail_ticks + self._tuning.get('trail_offset', 0))
    trail_act = max(_floor, action.trail_activation_ticks + self._tuning.get('trail_act_offset', 0))
    
    logger.info(f"ENTRY: {side.upper()} @ {price:.2f}  "
                f"tid={tid}  dist={action.dist:.2f}  "
                f"dir_src={action.dir_source}  "
                f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")
    
    self._position = make_position(
        entry_price=price, side=side, state=action.belief_state,
        tick_size=self._asset.tick_size, tick_value=self._asset.tick_value,
        stop_distance_ticks=sl_ticks, profit_target_ticks=tp_ticks,
        trailing_stop_ticks=trail_ticks, trail_activation_ticks=trail_act,
        template_id=tid,
    )
    self._init_exit_state(side, price, sl_ticks, tp_ticks, tid, action.lib_entry)
    
    # Tell ExecutionEngine about the opened position
    self._exec_engine.position_opened(
        side=side, price=price, bar_index=self._bar_i,
        template_id=tid, lib_entry=action.lib_entry,
        sl_ticks=sl_ticks, tp_ticks=tp_ticks,
        network_tp=action.network_tp,
        max_hold_bars=action.max_hold_bars,
    )
    
    self._position_open = True
    self._entry_price = price
    self._entry_time = ts
    self._entry_bar = self._bar_i
    self._active_side = side
    self._active_tid = tid
    self._entry_depth = action.depth
    self._max_hold_bars = action.max_hold_bars
    self._predicted_mfe_ticks = round(action.conviction, 2)
    
    self._trade_logger.start_trade(
        self._session_trades + 1, side, price, ts)
    self._belief_network.start_trade_tracking(
        side=side, entry_bar=self._bar_i,
        pattern_horizon_bars=action.max_hold_bars)
    
    self._gui_push({'type': 'TRADE_MARKER', 'action': 'entry',
                    'side': side, 'price': price})
    
    if self._dry_run:
        logger.info("[DRY RUN] Entry logged but no order sent")
        return
    
    order_msg = self._orders.build_entry_order(
        'BUY' if side == 'long' else 'SELL')
    if order_msg:
        self._order_send_ts = time.perf_counter()
        await self._client.send(order_msg)


def _gate_label_to_pct(self, gate_label: str) -> int:
    """Map ExecutionEngine gate rejection labels to belief bar %."""
    _MAP = {
        'gate0_no_features': 5,
        'gate0': 10, 'gate0_noise': 10,
        'gate0_r3_struct': 15, 'gate0_r3_snap': 15,
        'gate0_r4_nightmare': 15, 'gate0_r4_struct': 15,
        'gate0_hurst': 18, 'gate0_momentum': 18, 'gate0_tunnel': 18,
        'gate0_5': 20,
        'gate1': 35,
        'gate2': 55,
        'score_loser': 60,
        'no_direction': 65,
        'gate3': 75,
        'gate4_momentum_align': 85,
    }
    return _MAP.get(gate_label or '', 0)
```

### Exit Path: Also Delegate

The exit path already uses `ExitEngine.evaluate()` correctly in live. However,
`_check_exit()` should also sync with `ExecutionEngine.pos_state` so the EE
knows when to accept new candidates:

```python
async def _check_exit(self, price: float, ts: float):
    """Exit check — delegates to ExitEngine (already correct).
    
    Only change: on exit, also notify ExecutionEngine so it
    knows the position is closed and can accept new entries.
    """
    # ... existing ExitEngine.evaluate() logic stays the same ...
    
    if _exit_result.action != ExitAction.HOLD:
        # Existing close/flip logic
        ...
        # NEW: sync ExecutionEngine position state
        self._exec_engine.position_closed()
```

### Changes to ExecutionEngine._direction_cascade()

Live engine has one direction source that training doesn't: **live_momentum**
(velocity + acceleration from current physics state). This is actually a good
signal — it uses real-time market direction instead of stale regression
coefficients. But it needs to be IN the ExecutionEngine, not bolted on outside.

**Add as Priority 0.3** (after pp_override, before signed_mfe):

```python
# In ExecutionEngine._direction_cascade(), after pp_override check:

# ── Priority 0.3: Live momentum (velocity + net_force) ─────
# Real-time market direction from physics engine. Only fires in
# live/replay mode where state has fresh velocity data.
# Does not fire in IS mode (where oracle marker handles direction).
if self.mode in ('live', 'replay'):
    _vel = float(getattr(state, 'velocity', 0.0))
    _acc = float(getattr(state, 'net_force', 0.0))
    _mom = _vel + 0.5 * _acc
    _mom_thresh = 0.5  # tunable
    if abs(_mom) > _mom_thresh:
        side = 'long' if _mom > 0 else 'short'
        _p = 0.5 + min(abs(_mom) / 10.0, 0.45) * (1 if _mom > 0 else -1)
        return side, _p, 'live_momentum'
```

Also add Priority -0.5 for **live direction bias** (brain.dir_bias from PP learning):

```python
# ── Priority -0.5: Live direction bias (PP learning) ─────
# H0/H1 counterfactual learning from live trades. Only fires when
# brain has sufficient observations for this template.
if self.mode in ('live', 'replay'):
    bias = self.brain.get_dir_bias(tid)
    if bias:
        lw, ll = bias.get('long_w', 0), bias.get('long_l', 0)
        sw, sl = bias.get('short_w', 0), bias.get('short_l', 0)
        lt, st = lw + ll, sw + sl
        _min_t = 5  # min trades before bias fires
        if lt >= _min_t or st >= _min_t:
            l_wr = lw / lt if lt > 0 else 0.5
            s_wr = sw / st if st > 0 else 0.5
            if l_wr > 0.60 and (st < 3 or s_wr < 0.40):
                return 'long', 0.5 + l_wr * 0.4, 'live_bias'
            if s_wr > 0.60 and (lt < 3 or l_wr < 0.40):
                return 'short', 0.5 - s_wr * 0.4, 'live_bias'
```

**Updated full priority order after changes:**

```
Priority -1   : pp_override (ping-pong caller override)
Priority -0.5 : live_bias (brain.dir_bias from PP learning) — live/replay only
Priority 0.3  : live_momentum (velocity + acceleration) — live/replay only
Priority 0.5  : signed_mfe (learned signed MFE regression)
Priority 1    : logistic (per-cluster logistic regression)
Priority 1.5  : brain_dir (brain direction-specific win rate)
Priority 2    : template_bias (aggregate long/short bias)
Priority 3    : band_confluence (multi-TF SE bands)
Priority 4    : dmi (trend-following)
Priority 5    : velocity (fallback)
```

### Changes to ExecutionEngine Constructor

Add `mode` parameter awareness for live-specific behavior:

```python
class ExecutionEngine:
    def __init__(self, ..., mode: str = 'is', ...):
        self.mode = mode  # 'is', 'oos', 'live', 'replay'
        # mode affects:
        # 1. Which direction cascade priorities are active
        # 2. Whether aggression scaling is applied (future)
        # 3. Whether oracle_marker_fn is expected (IS only)
```

### Exit Sizing: Live ATR Override

ExecutionEngine._compute_sizing() uses template library stats (p25_mae, mean_mae,
reg_sigma, p75_mfe). In live mode, we also want ATR-based sizing as a fallback/
override when template stats are stale or missing.

**Add to ExecutionEngine:**

```python
def set_live_atr(self, atr_ticks: float):
    """Provide current ATR for live exit sizing fallback."""
    self._live_atr_ticks = atr_ticks

def _compute_sizing(self, lib_entry, network_tp, live_scaled=None):
    # ... existing template-based logic ...
    
    # Live ATR floor: if template stats produce tiny stops,
    # use ATR as minimum (live markets are noisier than historical)
    if self.mode in ('live', 'replay') and self._live_atr_ticks > 0:
        _atr_sl = max(4, int(round(self._live_atr_ticks * 3.0)))
        _atr_tp = max(4, int(round(self._live_atr_ticks * 5.0)))
        sl_ticks = max(sl_ticks, _atr_sl)
        tp_ticks = max(tp_ticks, _atr_tp)
    
    return sl_ticks, tp_ticks, trail_ticks, trail_act_ticks
```

LiveEngine calls `self._exec_engine.set_live_atr(self._live_atr_ticks)` after
each ATR update.

---

## What Gets Deleted from LiveEngine

**Methods deleted entirely** (replaced by ExecutionEngine delegation):

| Method | Lines | Replacement |
|--------|-------|-------------|
| `_check_entry()` | ~200 | Thin wrapper → `exec_engine.on_bar()` |
| `_determine_direction()` | ~80 | `ExecutionEngine._direction_cascade()` |
| `_compute_exit_params()` | ~30 | `ExecutionEngine._compute_sizing()` |

**Module-level constants deleted:**

| Constant | Replacement |
|----------|-------------|
| `_ADX_TREND_CONFIRMATION = 25.0` | `gate_thresholds.json` via EE |
| `_HURST_TREND_CONFIRMATION = 0.6` | `gate_thresholds.json` via EE |
| `_GATE1_DIST_THRESHOLD = 4.5` | `ExecutionEngine.gate1_dist` |
| `_WORKER_BYPASS_CONV = 0.65` | `ExecutionEngine.worker_bypass_conviction` |

**Instance variables deleted:**

| Variable | Replacement |
|----------|-------------|
| `self._gate_stats` | `self._exec_engine.gate_stats` |
| `self._entry_belief_pct` calculations in `_check_entry` | `_gate_label_to_pct()` |

**Total code removed:** ~310 lines of duplicated logic.
**Total code added:** ~60 lines (wrapper + `_execute_entry` + `_gate_label_to_pct`).
**Net:** -250 lines and one source of truth for the gate cascade.

---

## Trading Day Boundaries

CME MNQ futures session: Sunday 5PM CT → Friday 4PM CT.
Daily maintenance: 4:00 PM - 5:00 PM CT (no trading).

```python
def split_trading_days(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split continuous 15s bars into trading days.
    
    A trading day starts at 17:00 CT (23:00 UTC) and ends at 
    16:00 CT (22:00 UTC) next day. Maintenance gap = 16:00-17:00 CT.
    
    Returns list of DataFrames, one per day, sorted chronologically.
    """
    # Convert timestamps to CT (UTC-6 or UTC-5 depending on DST)
    # Detect day boundaries by gaps > 30 minutes
    # Each segment between gaps = one trading day
```

---

## Acceptance Criteria

1. Cold start to live-ready in < 120 seconds (5 days of 15s data)
2. Validation report prints before first live trade
3. If parity_score < 0.80, engine refuses to trade (hard gate)
4. Brain dir_bias table has entries for all templates seen in replay
5. TBN workers have current beliefs (not None) for all active TFs
6. Exit engine self-tune parameters are calibrated (not defaults)
7. NT8 receives only RESUME_FROM (no REQUEST_HISTORY on warm start)
8. `--skip-replay` flag exists for debugging (bypasses validation, loads from NT8 as before)
9. **`LiveEngine._check_entry()` is ≤ 80 lines** (wrapper only, no gate logic)
10. **`LiveEngine._determine_direction()` does not exist** (deleted)
11. **`ExecutionEngine._direction_cascade()` includes `live_momentum` and `live_bias`** priorities
12. **Session report `gate_stats` come from `ExecutionEngine.get_skip_counts()`**, not a separate dict

---

## Implementation Order

1. `atlas_loader.py` — standalone, testable with `python -m live.atlas_loader --days 5`
2. **`core/execution_engine.py` modifications** — add `mode` param, `live_momentum` + `live_bias` to direction cascade, `set_live_atr()`, ATR floor in `_compute_sizing()`
3. `history_replay.py` — uses ExecutionEngine directly, produces ReplayResult
4. **`live/live_engine.py` refactor** — delete `_check_entry`, `_determine_direction`, `_compute_exit_params`; add thin wrapper + `_execute_entry` + instantiate `ExecutionEngine`
5. Integration: wire replay before connect; transfer warm state
6. `bar_aggregator.py` — add `seed_from_replay()` 
7. Validation comparison — load OOS metrics from training report
8. CLI flags — `--skip-replay`, `--replay-days N`
9. Update session report to pull gate stats from `exec_engine.get_skip_counts()`

---

## Risk: What Could Go Wrong

**1. ExecutionEngine expects `Candidate.raw_event` with `parent_chain` for ancestry features.**
In live, there's no fractal tree — candidates have empty parent chains. This is already
handled: `extract_features()` returns 0.0 for ancestry features when parent_chain is empty.
Verify that the replay produces the same feature vectors as live.

**2. ExecutionEngine.on_bar() calls exit evaluation when in_position.**
Live currently checks exits on every 1s bar (not just 15s bars). The wrapper must
continue calling `_check_exit()` on every bar, and only call `exec_engine.on_bar()`
for entry evaluation on 15s bars. This split is correct — exits need sub-second
responsiveness, entries don't.

**3. Ping-pong flip bypasses ExecutionEngine.**
`_flip_position()` creates positions directly without going through EE gates.
This is intentional — PP flips are direction-only decisions, not gate cascade
decisions. The wrapper doesn't change this. However, `exec_engine.position_opened()`
and `position_closed()` must still be called so EE tracks position state correctly.

**4. Manual orders bypass ExecutionEngine.**
`_handle_manual_order()` creates positions without gates. Same as PP — intentional.
Still need to sync `exec_engine.position_opened()` / `position_closed()`.

**5. Aggression scaling currently modifies gate thresholds dynamically.**
The wrapper sets `exec_engine.gate1_dist` before each `on_bar()` call. This is
the correct approach — EE doesn't need to know about aggression, it just uses
whatever threshold is set. The tuning hot-reload also updates EE thresholds
every 20 bars.
