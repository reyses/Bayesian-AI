
import time
import numpy as np
from typing import Any, Dict, Optional, List
from core.bayesian_brain import TradeOutcome
from training.doe_parameter_generator import DOEParameterGenerator

# Constants moved from orchestrator.py
DEFAULT_BASE_SLIPPAGE = 0.25
DEFAULT_VELOCITY_SLIPPAGE_FACTOR = 0.1
REPRESENTATIVE_SUBSET_SIZE = 20
FISSION_SUBSET_SIZE = 50
INDIVIDUAL_OPTIMIZATION_ITERATIONS = 20

# --- Standalone Helpers for Multiprocessing ---

def simulate_trade_standalone(entry_price: float, data: Any, state: Any,
                              params: Dict[str, Any], point_value: float) -> Optional[TradeOutcome]:
    """
    Simulate single trade with lookahead — direction-aware
    Uses params for stop loss and take profit
    """
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

    for i in range(1, len(data)):
        row = data.iloc[i]
        price = row['price'] if 'price' in row else row['close']
        pnl = (price - entry_price) * dir_sign - trading_cost - total_slippage

        exit_time = row['timestamp']
        # Handle timestamp types safely
        if hasattr(exit_time, 'timestamp'):
            exit_time = exit_time.timestamp()

        duration = exit_time - entry_time

        # Check TP/SL
        if pnl >= take_profit:
            return TradeOutcome(
                state=state, entry_price=entry_price, exit_price=price,
                pnl=take_profit * point_value, result='WIN', timestamp=exit_time,
                exit_reason='TP', entry_time=entry_time, exit_time=exit_time,
                duration=duration, direction=direction
            )
        elif pnl <= -stop_loss:
            return TradeOutcome(
                state=state, entry_price=entry_price, exit_price=price,
                pnl=-stop_loss * point_value, result='LOSS', timestamp=exit_time,
                exit_reason='SL', entry_time=entry_time, exit_time=exit_time,
                duration=duration, direction=direction
            )
        elif duration >= max_hold:
            return TradeOutcome(
                state=state, entry_price=entry_price, exit_price=price,
                pnl=pnl * point_value, result='WIN' if pnl > 0 else 'LOSS', timestamp=exit_time,
                exit_reason='TIME', entry_time=entry_time, exit_time=exit_time,
                duration=duration, direction=direction
            )

    return None

def _optimize_pattern_task(args):
    """
    Task function for multiprocessing.
    args: (pattern, iterations, param_generator, point_value)
    Returns: (best_params, best_result_dict)
    """
    pattern, iterations, generator, point_value = args

    window = pattern.window_data
    if window is None or window.empty:
        return {}, {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}

    # Generate parameters
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
            data=window,
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

def _optimize_template_task(args):
    """
    Optimizes a Template Group.
    args: (template, subset, iterations, generator, point_value)
    Returns: (best_params, best_sharpe)
    """
    template, subset, iterations, generator, point_value = args

    # 1. Generate Parameter Sets (DOE)
    # We use the first pattern in subset to drive the generator context,
    # but params are applicable to all.
    ref_pattern = subset[0]
    param_sets = []
    for i in range(iterations):
        ps = generator.generate_parameter_set(iteration=i, day=ref_pattern.idx, context='TEMPLATE')
        param_sets.append(ps.parameters)

    best_sharpe = -float('inf')
    best_params = {}

    # 2. Iterate through Parameter Sets
    for params in param_sets:
        pnls = []

        # 3. Test Params on ALL members of the subset
        for pattern in subset:
            window = pattern.window_data
            if window is None or window.empty:
                continue

            outcome = simulate_trade_standalone(
                entry_price=pattern.price,
                data=window,
                state=pattern.state,
                params=params,
                point_value=point_value
            )

            if outcome:
                pnls.append(outcome.pnl)
            else:
                pnls.append(0.0) # No trade = 0 PnL

        if not pnls:
            continue

        # 4. Calculate Combined Metric (Sharpe)
        # combined_pnl = sum(pnls)
        pnl_array = np.array(pnls)
        if len(pnl_array) > 1 and np.std(pnl_array) > 1e-9:
            sharpe = np.mean(pnl_array) / np.std(pnl_array)
        else:
            sharpe = 0.0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

    return best_params, best_sharpe

def _process_template_job(args):
    """
    Multiprocessing Worker Function
    Executes the Fission/Optimization logic for a single template.
    Returns a result dict with timing breakdown.
    """
    template, clustering_engine, iterations, generator, point_value = args
    t0 = time.perf_counter()

    # 1. Select Training Subset
    subset = template.patterns[:FISSION_SUBSET_SIZE]

    # 2. Run Individual Optimization (for Fission Check)
    t1 = time.perf_counter()
    member_optimals = []
    for pattern in subset:
        best_p, _ = _optimize_pattern_task((pattern, INDIVIDUAL_OPTIMIZATION_ITERATIONS, generator, point_value))
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

    # 4. Consensus Optimization (No Fission)
    t3 = time.perf_counter()
    best_params, _ = _optimize_template_task((template, subset, iterations, generator, point_value))
    t_consensus = time.perf_counter() - t3

    # 5. Validation
    t4 = time.perf_counter()
    val_pnl = 0.0
    val_count = 0
    validation_subset = template.patterns[FISSION_SUBSET_SIZE:]
    if validation_subset:
        for p in validation_subset:
             outcome = simulate_trade_standalone(
                entry_price=p.price,
                data=p.window_data,
                state=p.state,
                params=best_params,
                point_value=point_value
            )
             if outcome:
                 val_pnl += outcome.pnl
                 val_count += 1
    t_validation = time.perf_counter() - t4

    elapsed = time.perf_counter() - t0

    return {
        'status': 'DONE',
        'template_id': template.template_id,
        'template': template,
        'best_params': best_params,
        'val_pnl': val_pnl,
        'member_count': template.member_count,
        'timing': (
            f'individual={t_individual:.1f}s consensus={t_consensus:.1f}s '
            f'validation={t_validation:.1f}s ({val_count} trades) total={elapsed:.1f}s'
        )
    }
