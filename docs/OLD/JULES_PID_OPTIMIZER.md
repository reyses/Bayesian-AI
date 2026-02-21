# JULES TASK: Replace DOE with Analytical Exits + Bayesian PID Optimizer

## Background

The DOE currently runs 1000 iterations per template during Phase 3, searching over 6 parameters.
After analysis, only 3 of those 6 params need search — the PID triplet `(pid_kp, pid_ki, pid_kd)`
that feeds `quantum_field_engine.batch_compute_states()`.

The other 3 (`stop_loss_ticks`, `take_profit_ticks`, `trailing_stop_ticks`) are analytically
derived directly from each template's oracle MFE/MAE distribution — no search needed. The current
code already does this override in `_optimize_template_task()` but inconsistently (only when
`mean_mfe > 5.0`, otherwise falls back to DOE-searched values).

**Goal:** Make the analytical exit computation the authoritative path for ALL templates, and
replace the 5-phase DOE with Optuna TPE (Tree-structured Parzen Estimator) for PID-only search.
With 3 continuous parameters, Optuna TPE converges in ~50-100 trials vs 1000 random samples.

---

## What Changes

### Problem 1: TP/SL/Trail still in DOE
`_define_parameter_ranges()` still includes `stop_loss_ticks`, `take_profit_ticks`,
`trailing_stop_ticks`. Remove them — they will be computed analytically per template.

### Problem 2: Oracle exit anchoring is partial
In `_optimize_template_task()` (orchestrator_worker.py ~line 261), the override only fires
when `mean_mfe > 5.0`, leaving small/new templates to use random DOE TP/SL. Fix: always
compute analytical TP/SL/trail from oracle data when available; use hardcoded defaults otherwise.

### Problem 3: 5-phase DOE wastes budget on 3 params
Latin Hypercube for 3 parameters needs ~20 points for full coverage, not 500. Replace the
entire `DOEParameterGenerator` class internals with Optuna TPE, keeping the same external
interface so orchestrator.py and orchestrator_worker.py need minimal changes.

---

## File 1: `training/doe_parameter_generator.py` — Full Rewrite of Class Internals

Replace the `DOEParameterGenerator` class with an Optuna-based optimizer. Keep the same
public interface:
- `__init__(self, context_detector)` — same signature
- `generate_parameter_set(iteration, day, context)` → `ParameterSet` — **remove this method**
  (it is no longer called; see File 2 for how optimization changes)

Add a new public method that replaces the entire for-loop pattern in `_optimize_template_task`:

```python
def optimize_pid(
    self,
    objective_fn,          # callable(pid_kp, pid_ki, pid_kd) -> float (Sharpe)
    n_trials: int = 200,   # number of Optuna trials
    seed: int = 42,
) -> dict:
    """
    Run Optuna TPE to find pid_kp, pid_ki, pid_kd that maximize Sharpe
    across all cluster members.

    Returns: {'pid_kp': float, 'pid_ki': float, 'pid_kd': float}
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial output

    def _optuna_objective(trial):
        pid_kp = trial.suggest_float('pid_kp', 0.1, 1.0)
        pid_ki = trial.suggest_float('pid_ki', 0.01, 0.2)
        pid_kd = trial.suggest_float('pid_kd', 0.1, 0.5)
        return objective_fn(pid_kp, pid_ki, pid_kd)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(_optuna_objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    return {
        'pid_kp': best['pid_kp'],
        'pid_ki': best['pid_ki'],
        'pid_kd': best['pid_kd'],
    }
```

Keep `_define_parameter_ranges()` with only the 3 PID params (it's used by `_log_parameter_configuration`).
Keep `_log_parameter_configuration()` as-is.
Remove: `generate_baseline_set`, `generate_latin_hypercube_set`, `generate_mutation_set`,
`generate_response_surface_set`, `generate_crossover_set`, `generate_random_set`,
`generate_parameter_set`, `update_best_params`, `get_exploitation_ratio`.
Keep: `__init__`, `optimize_pid`, `_define_parameter_ranges`, `_log_parameter_configuration`.

---

## File 2: `training/orchestrator_worker.py` — Rewrite `_optimize_template_task`

Replace the current two-step pattern (generate all param sets upfront → evaluate all) with:

**Step 1: Compute analytical TP/SL/Trail from oracle data**

```python
def _analytical_exits(template) -> dict:
    """
    Derive TP, SL, trail from the template's oracle MFE/MAE distribution.
    Falls back to hardcoded defaults when oracle data is insufficient.
    """
    FALLBACK_TP    = 40   # ticks
    FALLBACK_SL    = 15   # ticks
    FALLBACK_TRAIL = 8    # ticks

    mean_mfe = getattr(template, 'mean_mfe_ticks', 0.0) if template else 0.0
    p75_mfe  = getattr(template, 'p75_mfe_ticks',  0.0) if template else 0.0
    mean_mae = getattr(template, 'mean_mae_ticks',  0.0) if template else 0.0
    p25_mae  = getattr(template, 'p25_mae_ticks',   0.0) if template else 0.0

    if mean_mfe > 2.0:
        tp    = max(5,  int(round(p75_mfe  * 0.85))) if p75_mfe  > 2.0 else max(5, int(round(mean_mfe)))
        sl    = max(3,  int(round(mean_mae * 2.00))) if mean_mae  > 1.0 else FALLBACK_SL
        trail = max(2,  int(round(mean_mae * 1.10))) if mean_mae  > 1.0 else FALLBACK_TRAIL
    else:
        tp, sl, trail = FALLBACK_TP, FALLBACK_SL, FALLBACK_TRAIL

    return {
        'take_profit_ticks':  tp,
        'stop_loss_ticks':    sl,
        'trailing_stop_ticks': trail,
        # keep max_hold_seconds and trading_cost_points fixed
        'max_hold_seconds':   600,
        'trading_cost_points': 0.50,
    }
```

**Step 2: Optuna objective wrapping `simulate_trade_standalone`**

Replace the entire for-loop in `_optimize_template_task` with:

```python
# 1. Analytical exits — fixed for all trials
fixed_exits = _analytical_exits(template)

# 2. Pre-process subset
processed_subset = [
    (p, _extract_arrays_from_df(p.window_data))
    for p in subset
    if p.window_data is not None and not p.window_data.empty
]
processed_subset = [(p, d) for p, d in processed_subset if d is not None]

if not processed_subset:
    return {}, -float('inf')

# 3. Optuna objective: vary only PID, exits are fixed
def _sharpe_objective(pid_kp, pid_ki, pid_kd):
    params = {**fixed_exits, 'pid_kp': pid_kp, 'pid_ki': pid_ki, 'pid_kd': pid_kd}
    pnls = []
    for pattern, sim_data in processed_subset:
        outcome = simulate_trade_standalone(
            entry_price=pattern.price,
            data=sim_data,
            state=pattern.state,
            params=params,
            point_value=point_value,
            template=template,
            template_library=pattern_library,
        )
        pnls.append(outcome.pnl if outcome else 0.0)

    pnl_array = np.array(pnls)
    if len(pnl_array) > 1 and np.std(pnl_array) > 1e-9:
        return float(np.mean(pnl_array) / np.std(pnl_array))
    return 0.0

# 4. Run Optuna TPE
n_trials = max(50, min(iterations, 200))  # 50 min, 200 max — TPE converges fast
seed = getattr(template, 'template_id', 42) if template else 42
best_pid = generator.optimize_pid(_sharpe_objective, n_trials=n_trials, seed=seed)

# 5. Final best params = analytical exits + best PID
best_params = {**fixed_exits, **best_pid}
best_sharpe = _sharpe_objective(**best_pid)

return best_params, best_sharpe
```

**Also update `simulate_trade_standalone`** (~line 110): it currently reads `stop_loss_ticks`,
`take_profit_ticks`, `max_hold_seconds`, `trading_cost_points` from params. Add
`trailing_stop_ticks` as an additional param read (currently it doesn't use it at all — the
numba loop uses fixed TP/SL with no trailing). If a trailing stop simulation is not yet
implemented in `_fast_sim_loop`, that's fine — just ensure `trailing_stop_ticks` is stored in
`best_params` so the forward pass can read it from `pattern_library`.

**Remove** the oracle exit anchoring block (~lines 261–277) entirely — it is replaced by
`_analytical_exits()`.

---

## File 3: `requirements.txt`

Add:
```
optuna>=3.5.0
```

---

## File 4: `training/doe_parameter_generator.py` — Update `_define_parameter_ranges`

Already done (cleanup earlier this session): only `pid_kp`, `pid_ki`, `pid_kd` remain.
Verify and leave as-is.

---

## What Does NOT Change

- `training/orchestrator.py` — forward pass reads `pattern_library[best_tid]['params']` which
  will now contain `{take_profit_ticks, stop_loss_ticks, trailing_stop_ticks, pid_kp, pid_ki,
  pid_kd, max_hold_seconds, trading_cost_points}`. Same keys, same reads. No change needed.
- `core/quantum_field_engine.py` — still reads `pid_kp/ki/kd` from params dict. No change.
- `training/wave_rider.py` — unchanged.
- Any other file — unchanged.

---

## Expected Outcome

- Phase 3 DOE runs **50-200 Optuna trials** per template instead of 1000
- TP/SL/trail values are now **deterministic and analytically grounded** (no more DOE lottery)
- PID optimization is **sample-efficient** — TPE builds a probabilistic model of pid → Sharpe
  and targets high-probability regions, not random fills
- Cluster members with very few trades use sensible hardcoded defaults, not random DOE params
- Overall Phase 3 runtime should **decrease by ~60-80%** while producing better PID fits

---

## Test Plan

1. Run `--fresh` with new code — check that `pattern_library` pkl entries contain
   `take_profit_ticks`, `stop_loss_ticks`, `trailing_stop_ticks`, `pid_kp/ki/kd`
2. Spot-check a template with `mean_mfe=12` ticks: verify `take_profit_ticks ≈ p75_mfe * 0.85`
3. Spot-check a small template (2-3 members, no oracle data): verify fallback values
   `tp=40, sl=15, trail=8`
4. Run forward pass — should produce same or better PnL vs current; Phase 3 should be faster
5. Check DOE config print at startup shows only PID params (3 lines)
