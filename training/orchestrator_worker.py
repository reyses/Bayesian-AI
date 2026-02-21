
import time
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from numba import jit
from core.bayesian_brain import TradeOutcome
from training.doe_parameter_generator import DOEParameterGenerator
from config.settings import DEFAULT_BASE_SLIPPAGE, DEFAULT_VELOCITY_SLIPPAGE_FACTOR
from config.oracle_config import MARKER_NOISE

# Constants moved from orchestrator.py
REPRESENTATIVE_SUBSET_SIZE = 20
FISSION_SUBSET_SIZE = 50
INDIVIDUAL_OPTIMIZATION_ITERATIONS = 20

# ── Worker process globals (set once via pool initializer, never pickled per-task) ──
_WORKER_CLUSTERING_ENGINE = None
_WORKER_PATTERN_LIBRARY   = {}

def _worker_pool_init(clustering_engine, pattern_library):
    """Pool initializer — called once per worker process on startup.
    Stores large shared objects as globals so they don't get pickled into every task,
    avoiding the Windows WinError 1450 pipe buffer overflow.
    """
    global _WORKER_CLUSTERING_ENGINE, _WORKER_PATTERN_LIBRARY
    _WORKER_CLUSTERING_ENGINE = clustering_engine
    _WORKER_PATTERN_LIBRARY   = pattern_library

# --- Standalone Helpers for Multiprocessing ---

@jit(nopython=True)
def _fast_sim_loop(entry_price, prices, timestamps, dir_sign,
                   take_profit, stop_loss, max_hold, trading_cost, total_slippage,
                   entry_time):
    """
    Numba-optimized simulation loop.
    Iterates through prices/timestamps arrays to find exit condition.
    Returns tuple: (exit_price, raw_pnl, result_code, exit_time, exit_reason_code, duration)
    result_code: 0=None, 1=WIN, -1=LOSS
    exit_reason_code: 1=TP, 2=SL, 3=TIME
    """
    n = len(prices)
    # Start from index 1 (next bar) as index 0 is entry bar
    for i in range(1, n):
        price = prices[i]
        curr_time = timestamps[i]

        # Calculate PnL (points)
        raw_diff = (price - entry_price) * dir_sign
        pnl = raw_diff - trading_cost - total_slippage

        duration = curr_time - entry_time

        if pnl >= take_profit:
            # WIN (TP)
            return price, take_profit, 1, curr_time, 1, duration
        elif pnl <= -stop_loss:
            # LOSS (SL) - assumes fill at stop
            return price, -stop_loss, -1, curr_time, 2, duration
        elif duration >= max_hold:
            # TIME EXIT
            return price, pnl, (1 if pnl > 0 else -1), curr_time, 3, duration

    return 0.0, 0.0, 0, 0.0, 0, 0.0

def _extract_arrays_from_df(df: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Helper to extract prices and timestamps from DataFrame."""
    prices = None
    timestamps = None

    if 'price' in df.columns:
        prices = df['price'].values
    elif 'close' in df.columns:
        prices = df['close'].values
    else:
        return None

    if 'timestamp' in df.columns:
        ts_data = df['timestamp'].values
        # Handle datetime64[ns] conversion to float seconds
        if ts_data.dtype.type == np.datetime64:
             timestamps = ts_data.astype('int64') / 1e9
        else:
             timestamps = ts_data.astype(np.float64)
    else:
        return None

    return prices, timestamps

def simulate_trade_standalone(entry_price: float, data: Any, state: Any,
                              params: Dict[str, Any], point_value: float,
                              template: Any = None,
                              template_library: Dict = None) -> Optional[TradeOutcome]:
    """
    Simulate single trade with lookahead — direction-aware
    Uses params for stop loss and take profit.
    Optimized with Numba.

    Includes Opportunity Cost logic if template/library provided.
    """
    # 0. OPPORTUNITY COST CHECK (The "Marshmallow Test")
    if template and template_library and getattr(template, 'transition_probs', None):
        current_ev = getattr(template, 'expected_value', 0.0)

        # Check connected clusters
        for next_id, prob in template.transition_probs.items():
            if prob < 0.2: continue # Ignore low probability paths

            next_entry = template_library.get(next_id)
            if not next_entry: continue

            # Retrieve EV from library entry
            next_ev = next_entry.get('expected_value', 0.0)

            # Discounted Future Value
            future_ev = next_ev * prob

            # If future is significantly better (e.g. 50% better)
            if future_ev > (current_ev * 1.5) and future_ev > 10.0: # Ensure absolute value is meaningful
                 # WAITING FOR BETTER OPPORTUNITY
                 return None

    stop_loss = params.get('stop_loss_ticks', 15) * 0.25
    take_profit = params.get('take_profit_ticks', 40) * 0.25
    max_hold = params.get('max_hold_seconds', 600)
    trading_cost = params.get('trading_cost_points', 0.50)

    # Dynamic Slippage
    base_slippage = DEFAULT_BASE_SLIPPAGE
    velocity_factor = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

    velocity = state.particle_velocity
    slippage = base_slippage + velocity_factor * abs(velocity)
    total_slippage = slippage * 2.0

    # Direction from Archetype/State
    if hasattr(state, 'cascade_detected') and state.cascade_detected: # Roche Snap
        direction = 'SHORT' if state.z_score > 0 else 'LONG'
    elif hasattr(state, 'structure_confirmed') and state.structure_confirmed: # Structural Drive
        direction = 'LONG' if state.momentum_strength > 0 else 'SHORT'
    else:
        direction = 'LONG' # Fallback

    dir_sign = -1.0 if direction == 'SHORT' else 1.0

    entry_time = state.timestamp

    # Prepare data arrays
    prices = None
    timestamps = None

    if isinstance(data, tuple) and len(data) == 2:
        # Optimized path: Pass tuple (prices, timestamps) directly
        prices, timestamps = data
    elif hasattr(data, 'values'): # DataFrame or Series
        # Slow path: Extract numpy arrays using helper
        arrays = _extract_arrays_from_df(data)
        if arrays is None:
            return None
        prices, timestamps = arrays
    else:
        return None

    # Call Numba function
    exit_price, final_pnl, res_code, exit_time, exit_reason_code, duration = _fast_sim_loop(
        float(entry_price), prices, timestamps, float(dir_sign),
        float(take_profit), float(stop_loss), float(max_hold), float(trading_cost), float(total_slippage),
        float(entry_time)
    )

    if res_code == 0:
        return None

    result_str = 'WIN' if res_code == 1 else 'LOSS'
    reason_map = {1: 'TP', 2: 'SL', 3: 'TIME'}
    exit_reason = reason_map.get(exit_reason_code, 'UNKNOWN')

    return TradeOutcome(
        state=state, entry_price=entry_price, exit_price=exit_price,
        pnl=final_pnl * point_value, result=result_str, timestamp=exit_time,
        exit_reason=exit_reason, entry_time=entry_time, exit_time=exit_time,
        duration=duration, direction=direction
    )

# Analytical Exit Constants
TP_P75_MFE_FACTOR = 0.85
SL_MEAN_MAE_FACTOR = 2.00
TRAIL_MEAN_MAE_FACTOR = 1.10
MIN_MFE_FOR_ANALYTICAL_EXIT = 2.0
FALLBACK_TP_TICKS = 40
FALLBACK_SL_TICKS = 15
FALLBACK_TRAIL_TICKS = 8

def _analytical_exits(template) -> dict:
    """
    Derive TP, SL, trail from the template's oracle MFE/MAE distribution.
    Falls back to hardcoded defaults when oracle data is insufficient.
    """
    mean_mfe = getattr(template, 'mean_mfe_ticks', 0.0) if template else 0.0
    p75_mfe  = getattr(template, 'p75_mfe_ticks',  0.0) if template else 0.0
    mean_mae = getattr(template, 'mean_mae_ticks',  0.0) if template else 0.0
    p25_mae  = getattr(template, 'p25_mae_ticks',   0.0) if template else 0.0

    if mean_mfe > MIN_MFE_FOR_ANALYTICAL_EXIT:
        tp    = max(5,  int(round(p75_mfe  * TP_P75_MFE_FACTOR))) if p75_mfe  > 2.0 else max(5, int(round(mean_mfe)))
        sl    = max(3,  int(round(mean_mae * SL_MEAN_MAE_FACTOR))) if mean_mae  > 1.0 else FALLBACK_SL_TICKS
        trail = max(2,  int(round(mean_mae * TRAIL_MEAN_MAE_FACTOR))) if mean_mae  > 1.0 else FALLBACK_TRAIL_TICKS
    else:
        tp, sl, trail = FALLBACK_TP_TICKS, FALLBACK_SL_TICKS, FALLBACK_TRAIL_TICKS

    return {
        'take_profit_ticks':  tp,
        'stop_loss_ticks':    sl,
        'trailing_stop_ticks': trail,
        # keep max_hold_seconds and trading_cost_points fixed
        'max_hold_seconds':   600,
        'trading_cost_points': 0.50,
    }

def _optimize_pattern_task(args):
    """
    Task function for multiprocessing.
    NOTE: This is the legacy "per-pattern" optimization which used to feed the
    fission logic. In the new Optuna-based system, fission is done by passing
    the subset to _optimize_template_task (which now uses Optuna).

    However, if _optimize_pattern_task is still called by Fission logic (step 2),
    we need to support it. The new DOE generator does NOT support generate_parameter_set.

    Since Fission is also being replaced/updated implicitly by the instruction to
    "Remove legacy optimization logic", we should probably update _process_template_job
    to NOT call this anymore, OR update this to use Optuna for a single pattern.

    BUT the instruction only provided replacement for _optimize_template_task.
    I will assume _process_template_job might need updates if it calls this.
    Checking _process_template_job: it calls _optimize_pattern_task.

    I will implement a simple random search or Optuna for this single pattern if needed,
    or just use the analytical exits + default PID for individual patterns?
    Fission uses individual optimal params to cluster.
    If I change this to return analytical exits + optimized PID for ONE pattern,
    it works.
    """
    if len(args) == 6:
        pattern, iterations, generator, point_value, template, pattern_library = args
    else:
        pattern, iterations, generator, point_value = args
        template = None
        pattern_library = None

    # 1. Analytical exits
    fixed_exits = _analytical_exits(template)

    window = pattern.window_data
    if window is None or window.empty:
        return {}, {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}

    arrays = _extract_arrays_from_df(window)
    if arrays is None:
         return {}, {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}
    sim_data = arrays

    # 2. Optuna for one pattern?
    # Optuna overhead might be high for thousands of patterns in fission.
    # But we only do 20 iterations?
    # Generator.optimize_pid handles Optuna.

    # Single-pattern objective: capture ratio (pnl / oracle_target), capped at 1.0
    # R² is undefined for n=1, so use fraction-of-oracle-captured instead
    _oracle_target = float((getattr(pattern, 'oracle_meta', None) or {}).get('mfe', 0.0)) * point_value
    def _capture_objective(pid_kp, pid_ki, pid_kd):
        params = {**fixed_exits, 'pid_kp': pid_kp, 'pid_ki': pid_ki, 'pid_kd': pid_kd}
        outcome = simulate_trade_standalone(
            entry_price=pattern.price,
            data=sim_data,
            state=pattern.state,
            params=params,
            point_value=point_value,
            template=template,
            template_library=pattern_library
        )
        pnl = float(outcome.pnl) if outcome else 0.0
        if _oracle_target > 0:
            return min(pnl / _oracle_target, 1.0)  # capture ratio [<=1.0]
        return pnl  # fallback for noise patterns

    # Use fewer trials for single pattern
    best_pid = generator.optimize_pid(_capture_objective, n_trials=max(10, iterations), seed=pattern.idx)
    best_params = {**fixed_exits, **best_pid}

    # Re-run to get result dict
    outcome = simulate_trade_standalone(
        entry_price=pattern.price,
        data=sim_data,
        state=pattern.state,
        params=best_params,
        point_value=point_value,
        template=template,
        template_library=pattern_library
    )

    best_result_dict = {
        'trades': [outcome] if outcome else [],
        'sharpe': 1.0 if (outcome and outcome.pnl > 0) else -1.0,
        'win_rate': 1.0 if (outcome and outcome.result == 'WIN') else 0.0,
        'pnl': outcome.pnl if outcome else 0.0
    }

    return best_params, best_result_dict

def _optimize_template_task(args):
    """
    Optimizes a Template Group using Optuna TPE.
    args: (template, subset, iterations, generator, point_value, pattern_library)
    Returns: (best_params, best_sharpe)
    """
    if isinstance(args, dict):
        template = args['template']
        subset = args['subset']
        iterations = args['iterations']
        generator = args['generator']
        point_value = args['point_value']
        pattern_library = args.get('pattern_library')
    elif len(args) == 6:
        template, subset, iterations, generator, point_value, pattern_library = args
    else:
        template, subset, iterations, generator, point_value = args
        pattern_library = None

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

    # 3. Optuna objective: adjusted R² of captured PnL vs oracle MFE
    # y_true[i] = oracle_mfe * point_value  (what was available)
    # y_hat[i]  = outcome.pnl              (what these params captured)
    # Adj R² = 1 - (1 - R²) * (n-1)/(n-k-1)  where k=3 (kp, ki, kd)
    def _adj_r2_objective(pid_kp, pid_ki, pid_kd):
        params = {**fixed_exits, 'pid_kp': pid_kp, 'pid_ki': pid_ki, 'pid_kd': pid_kd}
        y_true = []
        y_hat  = []
        for pattern, sim_data in processed_subset:
            oracle_mfe = float((getattr(pattern, 'oracle_meta', None) or {}).get('mfe', 0.0))
            target = oracle_mfe * point_value
            if target <= 0.0:
                continue  # skip noise patterns — no oracle signal to capture
            outcome = simulate_trade_standalone(
                entry_price=pattern.price,
                data=sim_data,
                state=pattern.state,
                params=params,
                point_value=point_value,
                template=template,
                template_library=pattern_library,
            )
            y_true.append(target)
            y_hat.append(outcome.pnl if outcome else 0.0)

        n = len(y_true)
        k = 3  # PID params: kp, ki, kd
        if n < k + 2:
            return -1.0  # not enough observations to compute adj R²
        y_true = np.array(y_true)
        y_hat  = np.array(y_hat)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-9:
            return -1.0  # all targets identical — degenerate cluster
        ss_res = np.sum((y_true - y_hat) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
        return float(adj_r2)

    # 4. Run Optuna TPE
    MIN_OPTUNA_TRIALS = 50
    MAX_OPTUNA_TRIALS = 200
    n_trials = max(MIN_OPTUNA_TRIALS, min(iterations, MAX_OPTUNA_TRIALS))  # TPE converges fast
    seed = getattr(template, 'template_id', 42) if template else 42
    best_pid = generator.optimize_pid(_adj_r2_objective, n_trials=n_trials, seed=seed)

    # 5. Final best params = analytical exits + best PID
    best_params = {**fixed_exits, **best_pid}
    best_adj_r2 = _adj_r2_objective(**best_pid)

    return best_params, best_adj_r2

def _process_template_job(args):
    """
    Multiprocessing Worker Function
    Executes the Fission/Optimization logic for a single template.
    Returns a result dict with timing breakdown.
    """
    # Unpack — support both dict and tuple formats
    if isinstance(args, dict):
        template = args['template']
        # Fall back to worker-process globals to avoid pickling large objects per-task
        clustering_engine = args.get('clustering_engine') or _WORKER_CLUSTERING_ENGINE
        iterations = args['iterations']
        generator = args['generator']
        point_value = args['point_value']
        pattern_library = args.get('pattern_library') or _WORKER_PATTERN_LIBRARY
    elif len(args) == 6:
        template, clustering_engine, iterations, generator, point_value, pattern_library = args
    else:
        template, clustering_engine, iterations, generator, point_value = args
        pattern_library = _WORKER_PATTERN_LIBRARY or {}

    t0 = time.perf_counter()

    # Scale DOE iterations proportionally to member count.
    # Tiny templates (1-5 members) never benefit from 1000 iterations — cap at n*10.
    n_members = len(template.patterns)
    effective_iterations = max(50, min(iterations, n_members * 10))

    # Fast-path: trivial templates with <= 2 members cannot be meaningfully optimized.
    # Skip individual + fission, go straight to a minimal consensus pass.
    if n_members <= 2:
        subset = template.patterns[:FISSION_SUBSET_SIZE]
        t_individual = 0.0
        t_fission    = 0.0
        consensus_args = {
            'template': template,
            'subset': subset,
            'iterations': effective_iterations,
            'generator': generator,
            'point_value': point_value,
            'pattern_library': pattern_library
        }
        t3 = time.perf_counter()
        best_params, _ = _optimize_template_task(consensus_args)
        t_consensus = time.perf_counter() - t3
        # Jump straight to validation block below
        validation_subset = template.patterns[FISSION_SUBSET_SIZE:]
        val_pnl = val_count = val_wins = 0
        pnls = []
        for p in validation_subset:
            outcome = simulate_trade_standalone(
                entry_price=p.price, data=p.window_data, state=p.state,
                params=best_params, point_value=point_value,
                template=template, template_library=pattern_library)
            if outcome:
                val_pnl += outcome.pnl; val_count += 1; pnls.append(outcome.pnl)
                if outcome.pnl > 0: val_wins += 1
        outcome_variance = float(np.std(pnls)) if pnls else 0.0
        avg_drawdown     = float(abs(np.mean([p for p in pnls if p < 0]))) if any(p < 0 for p in pnls) else 0.0
        win_rate  = val_wins / val_count if val_count > 0 else 0.0
        risk_score = (1.0 - win_rate) * 0.5  # tiny template: no adj R² possible
        t_validation = time.perf_counter() - t3 - t_consensus
        elapsed = time.perf_counter() - t0
        return {
            'status': 'DONE', 'template_id': template.template_id, 'template': template,
            'best_params': best_params, 'val_pnl': val_pnl, 'val_adj_r2': 0.0,
            'val_count': val_count, 'val_wins': val_wins,
            'outcome_variance': outcome_variance,
            'avg_drawdown': avg_drawdown, 'risk_score': risk_score,
            'member_count': template.member_count,
            'timing': (f'individual={t_individual:.1f}s consensus={t_consensus:.1f}s '
                       f'validation={t_validation:.1f}s ({val_count} trades) total={elapsed:.1f}s')
        }

    # 1. Select Training Subset
    subset = template.patterns[:FISSION_SUBSET_SIZE]

    # 2. Run Individual Optimization (for Fission Check)
    t1 = time.perf_counter()
    member_optimals = []
    for pattern in subset:
        best_p, _ = _optimize_pattern_task((pattern, INDIVIDUAL_OPTIMIZATION_ITERATIONS, generator, point_value, template, pattern_library))
        member_optimals.append(best_p)
    t_individual = time.perf_counter() - t1

    # 3. Check for Behavioral Fission (Regret-Based)
    t2 = time.perf_counter()
    new_sub_templates = clustering_engine.refine_clusters(template.template_id, member_optimals, subset)
    t_fission = time.perf_counter() - t2

    if new_sub_templates:
        # FISSION DETECTED
        elapsed = time.perf_counter() - t0
        return {
            'status': 'SPLIT',
            'template_id': template.template_id,
            'new_templates': new_sub_templates,
            'timing': f'individual={t_individual:.1f}s fission={t_fission:.1f}s total={elapsed:.1f}s'
        }

    # 4. Consensus Optimization (No Fission) — use scaled iteration count
    t3 = time.perf_counter()
    consensus_args = {
        'template': template,
        'subset': subset,
        'iterations': effective_iterations,
        'generator': generator,
        'point_value': point_value,
        'pattern_library': pattern_library
    }
    best_params, _ = _optimize_template_task(consensus_args)
    t_consensus = time.perf_counter() - t3

    # 5. Validation & Risk Calculation
    t4 = time.perf_counter()
    val_pnl = 0.0
    val_count = 0
    val_wins = 0
    pnls = []
    val_y_true = []  # oracle MFE targets for adj R² computation
    val_y_hat  = []  # captured PnL per pattern

    validation_subset = template.patterns[FISSION_SUBSET_SIZE:]
    if validation_subset:
        for p in validation_subset:
             outcome = simulate_trade_standalone(
                entry_price=p.price,
                data=p.window_data,
                state=p.state,
                params=best_params,
                point_value=point_value,
                template=template,
                template_library=pattern_library
            )
             if outcome:
                 val_pnl += outcome.pnl
                 val_count += 1
                 pnls.append(outcome.pnl)
                 if outcome.pnl > 0:
                     val_wins += 1
             # Collect oracle target for adj R²
             oracle_mfe = float((getattr(p, 'oracle_meta', None) or {}).get('mfe', 0.0))
             target = oracle_mfe * point_value
             if target > 0:
                 val_y_true.append(target)
                 val_y_hat.append(outcome.pnl if outcome else 0.0)

    # Adjusted R²: how well do best_params capture oracle MFE on validation set?
    _k = 3  # PID params
    _n = len(val_y_true)
    if _n >= _k + 2:
        _yt = np.array(val_y_true)
        _yh = np.array(val_y_hat)
        _ss_tot = np.sum((_yt - np.mean(_yt)) ** 2)
        if _ss_tot > 1e-9:
            _ss_res = np.sum((_yt - _yh) ** 2)
            _r2 = 1.0 - _ss_res / _ss_tot
            val_adj_r2 = float(1.0 - (1.0 - _r2) * (_n - 1) / (_n - _k - 1))
        else:
            val_adj_r2 = 0.0
    else:
        val_adj_r2 = 0.0

    # Calculate Risk Metrics
    if pnls:
        outcome_variance = float(np.std(pnls))
        # Drawdown approximation (using average loss as proxy)
        avg_drawdown = float(abs(np.mean([p for p in pnls if p < 0]))) if any(p < 0 for p in pnls) else 0.0
    else:
        outcome_variance = 0.0
        avg_drawdown = 0.0

    # Risk Score: complement of adj R² (higher R² = lower risk), blended with WR
    win_rate = val_wins / val_count if val_count > 0 else 0.0
    risk_score = (1.0 - max(val_adj_r2, 0.0)) * 0.5 + (1.0 - win_rate) * 0.5

    t_validation = time.perf_counter() - t4

    elapsed = time.perf_counter() - t0

    return {
        'status': 'DONE',
        'template_id': template.template_id,
        'template': template,
        'best_params': best_params,
        'val_pnl': val_pnl,
        'val_adj_r2': val_adj_r2,
        'val_count': val_count,
        'val_wins': val_wins,
        'outcome_variance': outcome_variance,
        'avg_drawdown': avg_drawdown,
        'risk_score': risk_score,
        'member_count': template.member_count,
        'timing': (
            f'individual={t_individual:.1f}s consensus={t_consensus:.1f}s '
            f'validation={t_validation:.1f}s ({val_count} trades) total={elapsed:.1f}s'
        )
    }

def _audit_trade(outcome, pattern):
    """
    Compare strategy decision against oracle ground truth.

    Returns dict with audit metrics:
        - oracle_match: bool (did strategy agree with oracle?)
        - oracle_marker: int (what the oracle said)
        - classification: str (TP, FP, TN, FN)
    """
    oracle_says = getattr(pattern, 'oracle_marker', MARKER_NOISE)

    # True Positive: Agent traded in oracle's direction and oracle was right
    # False Positive: Agent traded but oracle said noise or opposite
    # True Negative: Agent skipped and oracle said noise
    # False Negative: Agent skipped but oracle said profitable

    if outcome is not None:
        # Agent traded
        agent_long = outcome.direction == 'LONG'
        oracle_long = oracle_says > 0
        oracle_short = oracle_says < 0

        if agent_long and oracle_long:
            classification = 'TP'
        elif not agent_long and oracle_short:
            classification = 'TP'
        elif oracle_says == MARKER_NOISE:
            classification = 'FP_NOISE'   # Traded noise
        else:
            classification = 'FP_WRONG'   # Traded wrong direction
    else:
        # Agent skipped
        if oracle_says == MARKER_NOISE:
            classification = 'TN'
        else:
            classification = 'FN'  # Missed a real move

    return {
        'oracle_match': classification == 'TP',
        'oracle_marker': oracle_says,
        'classification': classification
    }
