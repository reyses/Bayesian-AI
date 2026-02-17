# Jules Task: Numba JIT Trade Simulation (orchestrator_worker.py)

## Problem
`simulate_trade_standalone()` iterates over DataFrame rows in Python — `data.iloc[i]` per bar is extremely slow. In Phase 3, this function is called **thousands of times** (50 iterations x 50 patterns per template x N templates). The DataFrame `.iloc` access dominates runtime.

## Solution
1. Add a `@numba.jit(nopython=True)` inner loop that operates on pre-extracted numpy arrays
2. Extract price/timestamp arrays from DataFrame **once** in the callers (`_optimize_pattern_task`, `_optimize_template_task`, `_process_template_job`), then pass raw arrays to every simulation call
3. The JIT function returns numeric codes; the wrapper maps them back to `TradeOutcome` objects

## File: `training/orchestrator_worker.py`

### Step 1: Add the JIT-compiled inner loop

Add this at the top of the file, after the existing imports:

```python
from numba import njit

@njit(cache=True)
def _fast_sim_loop(prices, timestamps, entry_price, entry_time, dir_sign,
                   stop_loss, take_profit, max_hold, trading_cost, total_slippage,
                   point_value):
    """
    Numba-compiled trade simulation loop.
    Returns: (exit_idx, exit_price, pnl, duration, outcome_code)
    outcome_code: 0=no_trade, 1=TP_WIN, 2=SL_LOSS, 3=TIME_WIN, 4=TIME_LOSS
    """
    n = len(prices)
    for i in range(1, n):
        price = prices[i]
        pnl = (price - entry_price) * dir_sign - trading_cost - total_slippage
        exit_time = timestamps[i]
        duration = exit_time - entry_time

        if pnl >= take_profit:
            return i, price, take_profit * point_value, duration, 1  # TP WIN
        elif pnl <= -stop_loss:
            return i, price, -stop_loss * point_value, duration, 2  # SL LOSS
        elif duration >= max_hold:
            final_pnl = pnl * point_value
            code = 3 if pnl > 0 else 4  # TIME WIN or TIME LOSS
            return i, price, final_pnl, duration, code

    return -1, 0.0, 0.0, 0.0, 0  # No trade
```

### Step 2: Add a helper to extract arrays from DataFrame

```python
def _extract_arrays(data):
    """Extract price and timestamp numpy arrays from DataFrame (done once per pattern)."""
    if 'price' in data.columns:
        prices = data['price'].values.astype(np.float64)
    else:
        prices = data['close'].values.astype(np.float64)

    ts_col = data['timestamp']
    if hasattr(ts_col.iloc[0], 'timestamp'):
        # datetime objects -> float seconds
        timestamps = np.array([t.timestamp() for t in ts_col], dtype=np.float64)
    else:
        timestamps = ts_col.values.astype(np.float64)

    return prices, timestamps
```

### Step 3: Update `simulate_trade_standalone` to use the JIT path

The function should accept EITHER a DataFrame OR a pre-extracted `(prices, timestamps)` tuple. When callers pre-extract arrays, they pass the tuple to skip repeated DataFrame access.

```python
def simulate_trade_standalone(entry_price: float, data: Any, state: Any,
                              params: Dict[str, Any], point_value: float) -> Optional[TradeOutcome]:
    """
    Simulate single trade with lookahead — direction-aware.
    `data` can be a DataFrame OR a (prices_array, timestamps_array) tuple.
    """
    stop_loss = params.get('stop_loss_ticks', 15) * 0.25
    take_profit = params.get('take_profit_ticks', 40) * 0.25
    max_hold = float(params.get('max_hold_seconds', 600))
    trading_cost = float(params.get('trading_cost_points', 0.50))

    # Dynamic Slippage
    velocity = state.particle_velocity
    slippage = DEFAULT_BASE_SLIPPAGE + DEFAULT_VELOCITY_SLIPPAGE_FACTOR * abs(velocity)
    total_slippage = slippage * 2.0

    # Direction from Archetype/State
    if hasattr(state, 'cascade_detected') and state.cascade_detected:
        direction = 'SHORT' if state.z_score > 0 else 'LONG'
    elif hasattr(state, 'structure_confirmed') and state.structure_confirmed:
        direction = 'LONG' if state.momentum_strength > 0 else 'SHORT'
    else:
        direction = 'LONG'

    dir_sign = -1.0 if direction == 'SHORT' else 1.0
    entry_time = float(state.timestamp)

    # Accept pre-extracted arrays OR DataFrame
    if isinstance(data, tuple):
        prices, timestamps = data
    else:
        prices, timestamps = _extract_arrays(data)

    if len(prices) < 2:
        return None

    # Call JIT-compiled loop
    exit_idx, exit_price, pnl, duration, code = _fast_sim_loop(
        prices, timestamps, float(entry_price), entry_time, dir_sign,
        float(stop_loss), float(take_profit), float(max_hold),
        float(trading_cost), float(total_slippage), float(point_value)
    )

    if code == 0:
        return None

    # Map numeric codes back to strings
    RESULT_MAP = {1: 'WIN', 2: 'LOSS', 3: 'WIN', 4: 'LOSS'}
    REASON_MAP = {1: 'TP', 2: 'SL', 3: 'TIME', 4: 'TIME'}

    exit_time = timestamps[exit_idx]

    return TradeOutcome(
        state=state, entry_price=entry_price, exit_price=exit_price,
        pnl=pnl, result=RESULT_MAP[code], timestamp=exit_time,
        exit_reason=REASON_MAP[code], entry_time=entry_time, exit_time=exit_time,
        duration=duration, direction=direction
    )
```

### Step 4: Update `_optimize_pattern_task` to pre-extract arrays

The key optimization: extract arrays ONCE, reuse across all 50+ parameter iterations.

```python
def _optimize_pattern_task(args):
    pattern, iterations, generator, point_value = args

    window = pattern.window_data
    if window is None or window.empty:
        return {}, {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}

    # PRE-EXTRACT arrays once (avoids DataFrame.iloc in every iteration)
    arrays = _extract_arrays(window)

    all_param_sets = []
    for i in range(iterations):
        ps = generator.generate_parameter_set(iteration=i, day=pattern.idx, context='PATTERN')
        all_param_sets.append(ps.parameters)

    best_pnl = -float('inf')
    best_params = {}
    best_result_dict = {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}

    entry_price = pattern.price
    state = pattern.state

    for params in all_param_sets:
        outcome = simulate_trade_standalone(
            entry_price=entry_price,
            data=arrays,           # <-- pass tuple, not DataFrame
            state=state,
            params=params,
            point_value=point_value
        )

        if outcome:
            if outcome.pnl > best_pnl:
                best_pnl = outcome.pnl
                best_params = params
                best_result_dict = {
                    'trades': [outcome],
                    'sharpe': 1.0 if outcome.pnl > 0 else -1.0,
                    'win_rate': 1.0 if outcome.result == 'WIN' else 0.0,
                    'pnl': outcome.pnl
                }

    return best_params, best_result_dict
```

### Step 5: Update `_optimize_template_task` similarly

```python
def _optimize_template_task(args):
    template, subset, iterations, generator, point_value = args

    ref_pattern = subset[0]
    param_sets = []
    for i in range(iterations):
        ps = generator.generate_parameter_set(iteration=i, day=ref_pattern.idx, context='TEMPLATE')
        param_sets.append(ps.parameters)

    # PRE-EXTRACT arrays for ALL patterns in subset (done once)
    pattern_arrays = []
    for pattern in subset:
        window = pattern.window_data
        if window is not None and not window.empty:
            pattern_arrays.append((pattern, _extract_arrays(window)))
        else:
            pattern_arrays.append((pattern, None))

    best_sharpe = -float('inf')
    best_params = {}

    for params in param_sets:
        pnls = []

        for pattern, arrays in pattern_arrays:
            if arrays is None:
                continue

            outcome = simulate_trade_standalone(
                entry_price=pattern.price,
                data=arrays,           # <-- pass tuple, not DataFrame
                state=pattern.state,
                params=params,
                point_value=point_value
            )

            if outcome:
                pnls.append(outcome.pnl)
            else:
                pnls.append(0.0)

        if not pnls:
            continue

        pnl_array = np.array(pnls)
        if len(pnl_array) > 1 and np.std(pnl_array) > 1e-9:
            sharpe = np.mean(pnl_array) / np.std(pnl_array)
        else:
            sharpe = 0.0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

    return best_params, best_sharpe
```

### Step 6: Update `_process_template_job` validation section

In `_process_template_job`, the validation loop (step 5) also calls `simulate_trade_standalone`. Pre-extract there too:

```python
    # 5. Validation (in _process_template_job)
    t4 = time.perf_counter()
    val_pnl = 0.0
    val_count = 0
    validation_subset = template.patterns[FISSION_SUBSET_SIZE:]
    if validation_subset:
        for p in validation_subset:
            if p.window_data is None or p.window_data.empty:
                continue
            arrays = _extract_arrays(p.window_data)
            outcome = simulate_trade_standalone(
                entry_price=p.price,
                data=arrays,
                state=p.state,
                params=best_params,
                point_value=point_value
            )
            if outcome:
                val_pnl += outcome.pnl
                val_count += 1
    t_validation = time.perf_counter() - t4
```

---

## Performance Impact

| Operation | Before (Python loop) | After (Numba JIT) |
|-----------|---------------------|-------------------|
| 1 sim call (1000 bars) | ~2-5ms | ~0.01-0.05ms |
| 50 iterations x 1 pattern | ~100-250ms | ~0.5-2.5ms |
| 50 iters x 50 patterns (template) | ~5-12s | ~25-125ms |
| Phase 3 total (26+ templates) | minutes | seconds |

The `_fast_sim_loop` compiles on first call (~0.5s warmup), then runs at native speed. The `cache=True` flag persists the compiled version to disk so subsequent runs skip compilation.

---

## Verification

```bash
python training/orchestrator.py --fresh --no-dashboard --iterations 50
```

Expected: Phase 3 should be dramatically faster. Look for per-template timing in the output — individual optimization should drop from seconds to milliseconds.

Also verify the first call triggers JIT compilation (you'll see a brief pause), then subsequent calls are instant.

---

## File Summary

| File | Action |
|------|--------|
| `training/orchestrator_worker.py` | Add `_fast_sim_loop` JIT kernel, `_extract_arrays` helper, update all 4 functions |

## Key Points
- `@njit(cache=True)` — nopython mode, cached to disk
- Pre-extract arrays ONCE per pattern, reuse across all parameter iterations
- `simulate_trade_standalone` accepts both DataFrame and (prices, timestamps) tuple for backward compat
- Numeric outcome codes inside JIT, mapped to strings outside
- All `float()` casts before JIT call ensure type stability
- No new dependencies (numba already in requirements.txt)
