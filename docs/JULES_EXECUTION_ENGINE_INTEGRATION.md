# Jules Task: Integrate ExecutionEngine into Orchestrator + Live Engine

## Goal

Replace the duplicated entry/exit/direction logic in `training/orchestrator.py`
(lines ~1050-1947) and `live/live_engine.py` with calls to the unified
`core/execution_engine.py`. After this, both IS/OOS and live share one
decision-making engine. The orchestrator keeps oracle/audit bookkeeping,
the live engine keeps socket I/O and NT8 orders.

## Architecture Principle

```
ExecutionEngine (core/execution_engine.py)
  Owns: gate cascade, direction cascade, exit evaluation, position lifecycle, stop/TP sizing
  Does NOT own: oracle labeling, audit tracking, signal logging, order submission, equity tracking

Orchestrator (training/orchestrator.py)
  Owns: data loading, day loop, oracle record assembly, decision_matrix_records,
        audit TP/FP/FN tracking, signal_log, equity simulation, ping-pong flip logic,
        belief_network.start/stop_trade_tracking, brain.update

LiveEngine (live/live_engine.py)
  Owns: socket I/O, NT8 order submission, reconnection, session management,
        trade_logger, tuning.json hot-reload
```

## Part 1: Update ExecutionEngine to match orchestrator's full logic

The current `core/execution_engine.py` has a simplified gate cascade. It needs
to be updated to match the orchestrator's full logic:

### 1a. Gate 0 — Full physics safety (Rule 5)

Current ExecutionEngine Gate 0 only checks pattern/z_score/data_override.
Add Rule 5 checks from orchestrator lines 1286-1299:

```python
# After existing gate0 checks, before gate0.5:
if not should_skip and not _yolo:
    _st = cand.state
    # 5a. Hurst < 0.5
    if _st.hurst_exponent < 0.5:
        self.gate_stats['gate0_skip'] += 1
        return reject('gate0_hurst')
    # 5b. Momentum override
    elif abs(_st.F_momentum) > abs(_st.mean_reversion_force) * 1.5 and abs(_st.mean_reversion_force) > 0:
        self.gate_stats['gate0_skip'] += 1
        return reject('gate0_momentum')
    # 5c. Tunnel probability < 40%
    elif _st.reversion_probability < 0.40:
        self.gate_stats['gate0_skip'] += 1
        return reject('gate0_tunnel')
```

### 1b. Gate 1 — Score competition

Current ExecutionEngine uses simple `dist < gate1_dist`. Add the orchestrator's
scoring system (orch lines 1353-1367):

```python
# In _evaluate_candidate, after gate1 distance check:
p_depth = cand.depth
tier_adj = self.tier_score_adj.get(self.template_tier_map.get(tid, 3), 0.0)
depth_adj = self.depth_score_adj.get(p_depth, 0.0)
score = p_depth + dist + tier_adj + depth_adj
```

Add `tier_score_adj`, `depth_score_adj`, `template_tier_map` as constructor params.

### 1c. Worker-bypass path

After all candidates fail Gate 1, the orchestrator has a bypass path
(lines 1396-1910) where high-conviction belief alone can fire a trade.
Add to ExecutionEngine:

```python
def _check_worker_bypass(self, bypass_candidate, bypass_dist, price, bar_index) -> TradeAction:
    """Gate 1 override: fire on belief conviction alone (no template match)."""
    belief = self.belief_network.get_belief()
    if belief is None or belief.conviction < self.worker_bypass_conviction:
        return TradeAction(type=ActionType.HOLD)
    # Physics quality gate
    depth = getattr(bypass_candidate, 'depth', 5)
    z = getattr(bypass_candidate, 'z_score', 0.0)
    if depth > 3 or z >= 0:
        return TradeAction(type=ActionType.HOLD, gate_label='physics_qg')
    side = belief.direction
    # Simplified sizing for bypass trades
    ...
    return TradeAction(type=ActionType.ENTER, side=side, ...)
```

### 1d. Exit sizing — full MFE regression

Current ExecutionEngine exit sizing is simplified. Match orchestrator's full
logic (lines 1556-1615):

- Phase 1 hard stop: p25_mae * 3.0, fallback mean_mae * 2.0, fallback DOE param
- Phase 2 trailing: reg_sigma * 1.1, fallback mean_mae * 1.1, fallback DOE param
- Trail activation: p25_mae * 0.3
- TP: network_tp > per-bar OLS regression > template p75 > DOE param
- MFE OLS prediction: `np.dot(live_scaled, mfe_coeff) + mfe_intercept`

### 1e. Direction cascade — match orchestrator priorities

Current ExecutionEngine direction cascade is close but has ordering differences.
Match orchestrator's exact priority order (lines 1416-1521):

```
Priority -1: ping-pong live bias (live only, caller provides override)
Priority  0: oracle marker (IS only)
Priority 0.5: signed MFE regression
Priority  1: per-cluster logistic regression (NOT brain_dir — orch has these swapped)
Priority 1.5: brain direction-specific win rate
Priority  2: template aggregate bias (includes long+short >= 0.10 rule)
Priority  3: multi-TF band confluence
Priority  4: DMI (with _DMI_THRESH)
Priority  5: velocity fallback
```

**Note**: In the orchestrator, logistic regression (Priority 1) comes BEFORE
brain direction (Priority 1.5). The current ExecutionEngine has these reversed.
Fix the order to match orchestrator.

### 1f. Return additional info for caller bookkeeping

The orchestrator needs several intermediate values from the gate cascade for
oracle records and decision_matrix. Add these to TradeAction:

```python
@dataclass
class TradeAction:
    # ... existing fields ...
    # Additional diagnostic fields for caller bookkeeping
    conviction: float = 0.0
    belief_state: Any = None       # BeliefState snapshot
    bypass_candidate: Any = None   # if worker-bypass fired
    live_features: Optional[np.ndarray] = None  # scaled feature vector
    long_bias: float = 0.0
    short_bias: float = 0.0
    parent_tf: str = ''
    max_hold_bars: int = 0
    trail_activation_ticks: float = 0.0
    # Per-candidate gate tracking (for decision matrix)
    candidate_gates: dict = field(default_factory=dict)  # id(p) -> gate_label
```

### 1g. New constructor parameters

```python
class ExecutionEngine:
    def __init__(self, ...,
        # Tier/depth scoring
        tier_score_adj: dict = None,     # {tier: score_adj}
        depth_score_adj: dict = None,    # {depth: score_adj}
        template_tier_map: dict = None,  # {tid: tier}
        exception_tids: set = None,      # high-quality TIDs for data override
        # Thresholds
        bias_threshold: float = 0.55,
        dmi_threshold: float = 0.0,
        worker_bypass_conviction: float = 0.65,
        # Equity tracking (caller-managed)
        equity_enabled: bool = False,
        # Asset info
        asset_tick_size: float = 0.25,
        asset_point_value: float = 2.0,
    ):
```

## Part 2: Refactor orchestrator's per-bar loop

### 2a. Instantiate ExecutionEngine before day loop

At ~line 580 (after scaler/centroids are loaded), create the engine:

```python
from core.execution_engine import ExecutionEngine, ActionType, TradeAction, Candidate

engine = ExecutionEngine(
    brain=self.brain,
    belief_network=belief_network,
    exit_engine=_exit_engine,
    pattern_library=self.pattern_library,
    scaler=self.scaler,
    centroids_scaled=centroids_scaled,
    valid_tids=valid_template_ids,
    tick_size=self.asset.tick_size,
    point_value=self.asset.point_value,
    mode='is' if not getattr(self, '_oos_mode', False) else 'oos',
    tier_score_adj=_TIER_SCORE_ADJ,
    depth_score_adj=_DEPTH_SCORE_ADJ,
    template_tier_map=template_tier_map,
    exception_tids=_exception_tids,
    depth_filter_out=_DEPTH_FILTER_OUT,
    bias_threshold=bias_threshold,
    dmi_threshold=dmi_threshold,
)
```

### 2b. Convert pattern_map candidates to Candidate objects

Before the candidate loop, wrap PatternEvent objects as Candidate:

```python
if not current_position_open and ts in pattern_map:
    raw_candidates = pattern_map[ts]
    candidates = []
    for p in raw_candidates:
        features = np.array([FractalClusteringEngine.extract_features(p)])
        candidates.append(Candidate(
            state=p.state,
            depth=getattr(p, 'depth', 6),
            timeframe=getattr(p, 'timeframe', '15s'),
            timestamp=ts,
            pattern_type=p.pattern_type,
            z_score=p.z_score,
            features=features,
            # Preserve raw event for oracle/audit
            _raw_event=p,
        ))
```

**Important**: Add `_raw_event: Any = None` to the Candidate dataclass so the
orchestrator can access the original PatternEvent for oracle_marker, oracle_meta,
parent_chain, etc.

### 2c. Replace gate/direction/sizing with engine.on_bar()

The current ~700 lines of gate cascade (1215-1924) becomes:

```python
action = engine.on_bar(
    price=price, bar_high=bar_high, bar_low=bar_low,
    bar_index=_bar_i, candidates=candidates,
    net_force=_net_force,
    sub_bar_highs=sub_bar_highs, sub_bar_lows=sub_bar_lows,
    oracle_marker=_effective_oracle(best_raw_event),
)
```

### 2d. Handle ENTER action (orchestrator bookkeeping stays)

```python
if action.type == ActionType.ENTER:
    side = action.side
    best_tid = action.template_id
    lib_entry = action.lib_entry
    _network_tp = action.network_tp
    _belief = action.belief_state

    # Equity ruin check (stays in orchestrator)
    if _equity_enabled:
        _max_loss_usd = action.sl_ticks * self.asset.tick_size * self.asset.point_value
        if _max_loss_usd > running_equity * 0.50:
            skipped_ruin += 1
            continue

    # Open position via WaveRider (stays in orchestrator for now)
    self.wave_rider.open_position(
        entry_price=price, side=side, state=raw_candidate.state,
        stop_distance_ticks=action.sl_ticks,
        profit_target_ticks=action.tp_ticks,
        trailing_stop_ticks=action.trail_ticks,
        trail_activation_ticks=action.trail_activation_ticks,
        template_id=best_tid,
    )

    # Notify engine
    engine.position_opened(side, price, _bar_i, best_tid, lib_entry,
                           sl_ticks=action.sl_ticks, tp_ticks=action.tp_ticks,
                           network_tp=_network_tp)

    # All oracle/audit bookkeeping stays exactly as-is
    current_position_open = True
    active_entry_price = price
    ...
    pending_oracle = { ... }  # same as current
    # decision_matrix record: same as current
    # audit TP/FP: same as current
```

### 2e. Handle EXIT (use engine result, keep oracle bookkeeping)

The exit section (~lines 1086-1175) becomes:

```python
if current_position_open:
    # Get exit signal for belief network
    _exit_sig = belief_network.get_exit_signal(self.wave_rider.position.side)

    # WaveRider trail update (stays — ExitEngine is separate from WaveRider)
    res = self.wave_rider.update_trail(price, None, ts_raw, exit_signal=_exit_sig)

    # Sub-bar wick check stays
    if not res['should_exit'] and _has_1s:
        # ... same 1s inner loop ...

    if res['should_exit']:
        # Build TradeOutcome, brain.update, oracle record — all stays
        outcome = TradeOutcome(...)
        self.brain.update(outcome)
        engine.position_closed()
        ...
```

**Key insight**: The EXIT path changes very little because WaveRider manages
trail/stops and ExitEngine is already separate. The engine just needs to be
notified via `engine.position_closed()`.

### 2f. Keep ping-pong logic in orchestrator

Ping-pong (flip after exit) is orchestrator-specific simulation logic.
Keep it as-is, just notify engine of the new position:

```python
if _should_flip:
    engine.position_opened(_flip_side, price, _bar_i, _pp_tid, ...)
```

## Part 3: Refactor live_engine.py

### 3a. `_check_entry()` (~line 1718, ~300 lines)

Replace the duplicate gate cascade and direction logic with:

```python
async def _check_entry(self, price, ts, states):
    candidates = [Candidate(
        state=s.state, depth=s.depth, timeframe=s.timeframe,
        timestamp=ts, pattern_type=s.pattern_type,
        z_score=s.z_score, features=s.features,
    ) for s in states]

    action = self.engine.on_bar(
        price=price, bar_high=price, bar_low=price,
        bar_index=self._bar_count, candidates=candidates,
    )

    if action.type == ActionType.ENTER:
        await self._submit_entry_order(action)
        self.engine.position_opened(action.side, price, self._bar_count,
                                     action.template_id, action.lib_entry)
```

### 3b. `_check_exit()` (~line 769, ~100 lines)

```python
async def _check_exit(self, price, ts):
    action = self.engine.on_bar(
        price=price, bar_high=self._bar_high, bar_low=self._bar_low,
        bar_index=self._bar_count,
    )
    if action.type == ActionType.EXIT:
        await self._submit_exit_order(action)
        self.engine.learn_direction(action.template_id, action.side, action.pnl_ticks)
        self.engine.position_closed()
```

### 3c. `_determine_direction()` (~line 2023)

DELETE this method entirely. Direction comes from `engine._direction_cascade()`.

### 3d. Instantiation

In `LiveEngine.__init__()` or `start()`:

```python
from core.execution_engine import ExecutionEngine
self.engine = ExecutionEngine(
    brain=self.brain, belief_network=self.belief_network,
    exit_engine=self.exit_engine, pattern_library=self.pattern_library,
    scaler=self.scaler, centroids_scaled=self.centroids_scaled,
    valid_tids=self.valid_tids,
    tick_size=self.asset.tick_size,
    point_value=self.asset.point_value,
    mode='live',
)
```

## Part 4: Testing

1. Run `python training/orchestrator.py --forward-pass --data DATA/ATLAS_1DAY`
   - Should produce same trade count / WR / PnL as before (within rounding)
   - If significantly different, the gate logic translation has a bug

2. Check `reports/is/oracle_trade_log.csv` columns are all populated

3. Verify `python training/orchestrator.py --forward-pass` (full ATLAS) runs
   without errors

## Files Modified

| File | Changes |
|------|---------|
| `core/execution_engine.py` | Add: Rule 5 physics, score competition, worker bypass, full MFE sizing, fix direction order, diagnostic fields |
| `training/orchestrator.py` | Replace ~700 lines of gate/direction/sizing with engine calls. Keep oracle/audit/signal bookkeeping |
| `live/live_engine.py` | Replace `_check_entry`, `_check_exit`, delete `_determine_direction`, add engine instantiation |

## Critical Constraints

1. **Do NOT change oracle/audit logic** — the orchestrator's oracle_trade_records,
   fn_oracle_records, decision_matrix_records, audit TP/FP/FN tracking must
   produce IDENTICAL output
2. **Do NOT change WaveRider** — position management stays as-is
3. **Do NOT change ExitEngine** — exit evaluation stays as-is
4. **Keep belief_network.start/stop_trade_tracking** in orchestrator — these
   are simulation lifecycle events, not engine decisions
5. **Keep ping-pong logic** in orchestrator — it's IS/OOS simulation-specific
6. **ExecutionEngine must be stateless w.r.t. oracle** — it never sees
   oracle_marker in OOS or live mode
7. **Direction cascade order must match orchestrator exactly** — verify with
   the priority numbers in orchestrator comments
8. **All gate skip counters must still be correct** — the report section
   (lines ~2940-3110) reads these counters
