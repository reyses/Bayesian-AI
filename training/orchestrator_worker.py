
import time
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from numba import jit
from core.bayesian_brain import TradeOutcome
from config.settings import DEFAULT_BASE_SLIPPAGE, DEFAULT_VELOCITY_SLIPPAGE_FACTOR
from config.oracle_config import MARKER_NOISE
from core.physics_utils import extract_dominant_cycle, calculate_kinetic_damping

# Constants (legacy names kept for import compat)
REPRESENTATIVE_SUBSET_SIZE = 20
FISSION_SUBSET_SIZE = 50
INDIVIDUAL_OPTIMIZATION_ITERATIONS = 20

# ── Pool-initializer globals (loaded once per worker process) ──────────
_W_CLUSTERING_ENGINE = None
_W_GENERATOR = None
_W_POINT_VALUE = None
_W_PATTERN_LIBRARY = None

def _init_pool_worker(clustering_engine, generator, point_value, pattern_library):
    """Called once per worker process — stores heavy objects in process globals."""
    global _W_CLUSTERING_ENGINE, _W_GENERATOR, _W_POINT_VALUE, _W_PATTERN_LIBRARY
    _W_CLUSTERING_ENGINE = clustering_engine
    _W_GENERATOR = generator
    _W_POINT_VALUE = point_value
    _W_PATTERN_LIBRARY = pattern_library

# Spectral Exit Constants
Z_SCORE_CYCLE_WINDOW = 60
VELOCITY_DAMPING_WINDOW = 20
KINETIC_DAMPING_EXIT_THRESHOLD = 0.8

# --- Standalone Helpers for Multiprocessing ---

@jit(nopython=True)
def _fast_sim_loop(entry_price, prices, timestamps, periods, dampings, dir_sign,
                   take_profit, stop_loss, max_hold, trading_cost, total_slippage,
                   entry_time):
    """
    Numba-optimized simulation loop.
    Iterates through prices/timestamps arrays to find exit condition.
    Returns tuple: (exit_price, raw_pnl, result_code, exit_time, exit_reason_code, duration)
    result_code: 0=None, 1=WIN, -1=LOSS
    exit_reason_code: 1=TP, 2=SL, 3=TIME, 4=KINETIC_EXHAUSTION
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

        # Stop loss is absolute overriding physics
        if pnl <= -stop_loss:
            # LOSS (SL) - assumes fill at stop
            return price, -stop_loss, -1, curr_time, 2, duration

        # Spectral Evaluation
        min_hold_seconds = periods[i] / 2.0

        # FOURIER GATE: Prohibit TP/Exit before half-cycle completes (unless SL hits)
        if duration < min_hold_seconds and pnl > -(stop_loss / 2.0):
            continue

        # LAPLACE GATE: Exit if kinetic energy is critically damped
        if pnl > 0 and dampings[i] > KINETIC_DAMPING_EXIT_THRESHOLD:
            return price, pnl, 1, curr_time, 4, duration # 4 = KINETIC_EXHAUSTION

        if pnl >= take_profit:
            # WIN (TP)
            return price, take_profit, 1, curr_time, 1, duration
        elif duration >= max_hold:
            # TIME EXIT
            return price, pnl, (1 if pnl > 0 else -1), curr_time, 3, duration

    return 0.0, 0.0, 0, 0.0, 0, 0.0

def _extract_arrays_from_df(df: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Helper to extract prices, timestamps, periods, and dampings from DataFrame."""
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

    n = len(prices)
    periods = np.zeros(n)
    dampings = np.zeros(n)

    if 'z_score' in df.columns and 'velocity' in df.columns:
        z_scores = df['z_score'].values
        velocities = df['velocity'].values

        # Calculate average dt from timestamps (for correct cycle period in seconds)
        dt = 1.0
        if timestamps is not None and len(timestamps) > 1:
            diffs = np.diff(timestamps)
            dt = float(np.median(diffs))
            if dt <= 0: dt = 1.0

        for i in range(10, n):
            w_z = z_scores[max(0, i - Z_SCORE_CYCLE_WINDOW):i]
            w_v = velocities[max(0, i - VELOCITY_DAMPING_WINDOW):i]
            periods[i] = extract_dominant_cycle(w_z, dt=dt)
            dampings[i] = calculate_kinetic_damping(w_v)

    return prices, timestamps, periods, dampings

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
    periods = None
    dampings = None

    if isinstance(data, tuple) and len(data) == 4:
        # Optimized path: Pass tuple (prices, timestamps, periods, dampings) directly
        prices, timestamps, periods, dampings = data
    elif hasattr(data, 'values'): # DataFrame or Series
        # Slow path: Extract numpy arrays using helper
        arrays = _extract_arrays_from_df(data)
        if arrays is None:
            return None
        prices, timestamps, periods, dampings = arrays
    else:
        return None

    # Call Numba function
    exit_price, final_pnl, res_code, exit_time, exit_reason_code, duration = _fast_sim_loop(
        float(entry_price), prices, timestamps, periods, dampings, float(dir_sign),
        float(take_profit), float(stop_loss), float(max_hold), float(trading_cost), float(total_slippage),
        float(entry_time)
    )

    if res_code == 0:
        return None

    result_str = 'WIN' if res_code == 1 else 'LOSS'
    reason_map = {1: 'TP', 2: 'SL', 3: 'TIME', 4: 'KINETIC_EXHAUSTION'}
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

REFINE_MAX_PASSES = 10      # Max trim iterations per template
REFINE_TRIM_PCT = 0.10      # Remove worst 10% per pass
REFINE_MIN_PATTERNS = 30    # Stop trimming if template gets too small
REFINE_MFE_DELTA = 0.30     # Temporal-half MFE stability threshold
REFINE_MFE_CV = 3.0         # MFE spread threshold

def _mfe_consistency(mfes):
    """Compute MFE consistency metrics. Returns (mfe_delta, mfe_cv, avg_mfe)."""
    mid = len(mfes) // 2
    mean_first = float(np.mean(mfes[:mid]))
    mean_second = float(np.mean(mfes[mid:]))
    avg_mfe = float(np.mean(mfes))
    mfe_delta = abs(mean_first - mean_second) / max(abs(avg_mfe), 0.01)
    mfe_cv = float(np.std(mfes)) / max(abs(avg_mfe), 0.01)
    return mfe_delta, mfe_cv, avg_mfe

def _validate_template_consistency(template, patterns, point_value):
    """Stepwise refinement: trim MFE outliers until template is consistent.

    Each pass removes the worst REFINE_TRIM_PCT of patterns (by distance
    from median MFE), then re-checks consistency. Stops when:
      - Template passes (mfe_delta < 0.30, mfe_cv < 3.0), OR
      - Template too small (< REFINE_MIN_PATTERNS), OR
      - Max passes reached

    Returns: (is_valid, consistency_score, diagnostics)
    Side effect: template.patterns is trimmed in-place to the refined set.
    """
    original_n = len(patterns)

    # Build working list with oracle MFE attached
    work = [(p, getattr(p, 'oracle_meta', {}).get('mfe', 0.0)) for p in patterns]
    work = [(p, mfe) for p, mfe in work if mfe != 0.0]

    if len(work) < 5:
        return True, 0.0, {'reason': 'too_few_patterns', 'n_patterns': len(work),
                            'trimmed': 0}

    trimmed_total = original_n - len(work)  # already lost from zero-MFE

    for pass_i in range(REFINE_MAX_PASSES):
        mfes = np.array([mfe for _, mfe in work])
        mfe_delta, mfe_cv, avg_mfe = _mfe_consistency(mfes)

        # Check if we pass
        if mfe_delta < REFINE_MFE_DELTA and mfe_cv < REFINE_MFE_CV:
            break

        # Stop if too small to trim further
        if len(work) <= REFINE_MIN_PATTERNS:
            break

        # Remove worst REFINE_TRIM_PCT by distance from median MFE
        median_mfe = float(np.median(mfes))
        dists = np.abs(mfes - median_mfe)
        n_trim = max(1, int(len(work) * REFINE_TRIM_PCT))
        # Keep indices sorted by distance (ascending) — drop the tail
        keep_idx = np.argsort(dists)[:len(work) - n_trim]
        work = [work[i] for i in keep_idx]
        trimmed_total += n_trim

    # Final metrics
    mfes = np.array([mfe for _, mfe in work])
    maes = np.array([getattr(p, 'oracle_meta', {}).get('mae', 0.0) for p, _ in work])
    mfe_delta, mfe_cv, avg_mfe = _mfe_consistency(mfes)
    is_valid = mfe_delta < REFINE_MFE_DELTA and mfe_cv < REFINE_MFE_CV
    consistency_score = 1.0 - (mfe_delta * 0.4 + min(mfe_cv / 6.0, 0.6))

    # Update template.patterns in-place to the refined set
    template.patterns = [p for p, _ in work]

    return is_valid, consistency_score, {
        'mfe_delta': mfe_delta,
        'mfe_cv': mfe_cv,
        'mfe_mean': avg_mfe,
        'mae_mean': float(np.mean(maes)) if len(maes) else 0.0,
        'n_patterns': len(work),
        'trimmed': trimmed_total,
        'passes': pass_i + 1 if 'pass_i' in dir() else 0,
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
