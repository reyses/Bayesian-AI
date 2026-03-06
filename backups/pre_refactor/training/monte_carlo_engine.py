"""
Monte Carlo Multi-Timeframe Optimizer Engine
Brute-force simulator: sweep templates × timeframes × parameters.
"""

import os
import sys
import glob
import pickle
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

# Project imports
from core.quantum_field_engine import QuantumFieldEngine
from core.three_body_state import ThreeBodyQuantumState
from core.bayesian_brain import BayesianBrain, TradeOutcome
from training.doe_parameter_generator import DOEParameterGenerator
from training.fractal_discovery_agent import TIMEFRAME_SECONDS

@dataclass
class TradeResult:
    """Lightweight trade result for MC simulation"""
    pnl: float
    side: str
    bars_held: int
    exit_reason: str
    template_id: int
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float

    def to_outcome(self, state_key) -> TradeOutcome:
        return TradeOutcome(
            state=state_key,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            pnl=self.pnl,
            result='WIN' if self.pnl > 0 else 'LOSS',
            timestamp=self.exit_time,
            exit_reason=self.exit_reason,
            entry_time=self.entry_time,
            exit_time=self.exit_time,
            duration=self.exit_time - self.entry_time,
            direction=self.side.upper(),
            template_id=self.template_id
        )

@dataclass
class IterationSummary:
    """Lightweight summary — no individual trades stored"""
    params: Dict[str, Any]
    total_pnl: float
    num_trades: int
    win_rate: float
    wins: int
    losses: int
    sharpe: float
    max_drawdown: float

@dataclass
class ComboResult:
    template_id: int
    timeframe: str
    best_params: Optional[Dict[str, Any]]
    best_pnl: float
    best_win_rate: float
    best_sharpe: float
    total_iterations: int
    total_wins: int
    total_losses: int
    top_iterations: List[IterationSummary]  # Only top 10 by PnL

def extract_features_from_state(state: ThreeBodyQuantumState, timeframe: str) -> np.ndarray:
    """
    Extracts 14D feature vector from a ThreeBodyQuantumState.
    Fills hierarchy features with defaults for flat scanning.
    """
    tf_secs = TIMEFRAME_SECONDS.get(timeframe, 15)
    tf_scale = np.log2(max(1, tf_secs))

    # Defaults for hierarchy (flat scan assumption)
    depth = 1.0
    parent_ctx = 0.0
    parent_z = 0.0
    parent_dmi_diff = 0.0
    root_is_roche = 0.0
    tf_alignment = 0.0

    # From state
    z = getattr(state, 'z_score', 0.0)
    v = getattr(state, 'velocity', 0.0)
    m = getattr(state, 'momentum', 0.0)
    c = getattr(state, 'coherence', 0.0)

    # Self regime
    self_adx = getattr(state, 'adx_strength', 0.0) / 100.0
    self_hurst = getattr(state, 'hurst_exponent', 0.5)
    self_dmi_diff = (getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0

    return np.array([
        abs(z), abs(v), abs(m), c,
        tf_scale, depth, parent_ctx,
        self_adx, self_hurst, self_dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment
    ])

    # TopStep rule: all positions flat by 3:00 PM EST
TOPSTEP_FLAT_HOUR = 15       # 3 PM EST (15:00)
TOPSTEP_FLAT_MINUTE = 0
TOPSTEP_NO_ENTRY_MINUTES_BEFORE_FLAT = 15  # No new entries within 15 min of flat deadline


def _ts_to_est_datetime(ts_value):
    """Convert a timestamp value to EST datetime, handling numpy.int64 and nanoseconds."""
    from datetime import datetime, timezone, timedelta
    import numpy as np
    EST = timezone(timedelta(hours=-5))
    if hasattr(ts_value, 'hour'):
        # Already a datetime-like object
        return ts_value
    ts = int(ts_value)
    # Detect nanoseconds: timestamps > 1e15 are nanoseconds (year ~33658 in seconds)
    if ts > 1_000_000_000_000_000:
        ts = ts // 1_000_000_000
    elif ts > 1_000_000_000_000:
        # Milliseconds
        ts = ts // 1_000
    return datetime.fromtimestamp(ts, tz=EST)


def _is_past_flat_deadline(ts_value):
    """Check if timestamp is at or past 3:00 PM EST."""
    dt = _ts_to_est_datetime(ts_value)
    return dt.hour > TOPSTEP_FLAT_HOUR or (dt.hour == TOPSTEP_FLAT_HOUR and dt.minute >= TOPSTEP_FLAT_MINUTE)


def _is_near_flat_deadline(ts_value):
    """Check if timestamp is within TOPSTEP_NO_ENTRY_MINUTES_BEFORE_FLAT of 3:00 PM EST."""
    dt = _ts_to_est_datetime(ts_value)
    flat_minutes = TOPSTEP_FLAT_HOUR * 60 + TOPSTEP_FLAT_MINUTE
    cur_minutes = dt.hour * 60 + dt.minute
    return cur_minutes >= (flat_minutes - TOPSTEP_NO_ENTRY_MINUTES_BEFORE_FLAT)


def simulate_month(data: pd.DataFrame, match_indices: np.ndarray, params: Dict[str, Any],
                   asset: Any, template_id: int, z_scores: np.ndarray) -> List[TradeResult]:
    """
    Simulate trades for matched bars in one month.
    Enforces TopStep flat-by-3PM-EST rule.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_side = 'long'
    entry_time = 0.0
    high_water = 0.0

    stop_ticks = params.get('stop_loss_ticks', 15)
    tp_ticks = params.get('take_profit_ticks', 40)
    max_hold = params.get('max_hold_bars', 50)
    trail_tight = params.get('trail_distance_tight', 10)
    trail_wide = params.get('trail_distance_wide', 30)
    cost_points = params.get('trading_cost_points', 0.5)

    prices = data['close'].values
    timestamps = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(prices))

    # Convert match_indices to a set for O(1) lookup
    match_set = set(match_indices)

    tick_size = asset.tick_size
    point_value = asset.point_value

    for idx in range(len(prices)):
        price = prices[idx]
        ts = timestamps[idx]

        # TopStep: force close at 3 PM EST
        if in_position and _is_past_flat_deadline(ts):
            bars_held = idx - entry_idx
            if entry_side == 'long':
                pnl_points = price - entry_price
            else:
                pnl_points = entry_price - price
            pnl_usd = (pnl_points - cost_points) * point_value
            trades.append(TradeResult(
                pnl=pnl_usd, side=entry_side, bars_held=bars_held, exit_reason='flat_deadline',
                template_id=template_id, entry_price=entry_price, exit_price=price,
                entry_time=entry_time, exit_time=ts
            ))
            in_position = False
            continue

        if in_position:
            # Check exits
            bars_held = idx - entry_idx

            if entry_side == 'long':
                pnl_points = price - entry_price
                high_water = max(high_water, pnl_points)
                exit_res = None

                # Stop loss
                if pnl_points <= -stop_ticks * tick_size:
                    exit_res = 'stop'
                # Take profit
                elif pnl_points >= tp_ticks * tick_size:
                    exit_res = 'target'
                # Trailing stop
                elif high_water > 0:
                    trail = trail_tight * tick_size if high_water < 2.0 else trail_wide * tick_size
                    if pnl_points < high_water - trail:
                        exit_res = 'trail'
                # Max hold
                elif bars_held >= max_hold:
                    exit_res = 'timeout'

                if exit_res:
                    pnl_usd = (pnl_points - cost_points) * point_value
                    trades.append(TradeResult(
                        pnl=pnl_usd, side='long', bars_held=bars_held, exit_reason=exit_res,
                        template_id=template_id, entry_price=entry_price, exit_price=price,
                        entry_time=entry_time, exit_time=ts
                    ))
                    in_position = False

            else:  # short
                pnl_points = entry_price - price
                high_water = max(high_water, pnl_points)
                exit_res = None

                # Stop loss
                if pnl_points <= -stop_ticks * tick_size:
                    exit_res = 'stop'
                # Take profit
                elif pnl_points >= tp_ticks * tick_size:
                    exit_res = 'target'
                # Trailing stop
                elif high_water > 0:
                    trail = trail_tight * tick_size if high_water < 2.0 else trail_wide * tick_size
                    if pnl_points < high_water - trail:
                        exit_res = 'trail'
                # Max hold
                elif bars_held >= max_hold:
                    exit_res = 'timeout'

                if exit_res:
                    pnl_usd = (pnl_points - cost_points) * point_value
                    trades.append(TradeResult(
                        pnl=pnl_usd, side='short', bars_held=bars_held, exit_reason=exit_res,
                        template_id=template_id, entry_price=entry_price, exit_price=price,
                        entry_time=entry_time, exit_time=ts
                    ))
                    in_position = False

        else:
            # Check for new entry (block if near flat deadline)
            if idx in match_set and not _is_near_flat_deadline(ts):
                entry_price = price
                entry_idx = idx
                entry_time = ts
                high_water = 0.0
                # Direction: use z_score from the state (mean reversion)
                # z > 0 → overbought → short; z < 0 → oversold → long
                # Note: This logic assumes we trade AGAINST the Z-score (Reversion).
                # If we want Momentum, we'd trade WITH it.
                # Doc says: "z > 0 → overbought → short; z < 0 → oversold → long"
                entry_side = 'short' if z_scores[idx] > 0 else 'long'
                in_position = True

    return trades

def simulate_template_tf_combo(template_id: int, timeframe: str, n_iterations: int,
                               data_root: str, template_info: Dict, asset: Any,
                               original_scaler: Optional[StandardScaler] = None,
                               mutation_base: Optional[Dict] = None,
                               mutation_scale: float = 0.1,
                               month_filter: Optional[List[str]] = None) -> ComboResult:
    """
    Standalone worker: For one (template, timeframe) pair,
    run n_iterations random parameter samples.
    """
    try:
        centroid = template_info['centroid']

        # Load data
        tf_dir = os.path.join(data_root, timeframe)
        if not os.path.exists(tf_dir):
            return ComboResult(template_id, timeframe, None, 0, 0, 0, n_iterations, 0, 0, [])

        if month_filter:
            monthly_files = month_filter
        else:
            monthly_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))

        all_data = []
        for f in monthly_files:
            try:
                df = pd.read_parquet(f)
                # Ensure timestamp is float/numeric if needed, or handle in simulate_month
                if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
                     df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not all_data:
             return ComboResult(template_id, timeframe, None, 0, 0, 0, n_iterations, 0, 0, [])

        # Pre-compute physics states
        engine = QuantumFieldEngine(use_gpu=False)
        all_states = []
        all_z_scores = []

        for month_data in all_data:
            raw_results = engine.batch_compute_states(month_data)
            # batch_compute_states returns [{'bar_idx', 'state', 'price', 'structure_ok'}]
            states = [r['state'] for r in raw_results]
            all_states.append(states)
            z_s = np.array([s.z_score for s in states])
            all_z_scores.append(z_s)

        # Extract features
        all_features = []
        for states in all_states:
            features = np.array([extract_features_from_state(s, timeframe) for s in states])
            all_features.append(features)

        # Local Scaling & Matching
        # We need to scale features to match the "sigma" space of the centroid.
        # 1. Fit local scaler on this timeframe's data
        combined_features = np.vstack(all_features) if all_features else np.array([])

        if combined_features.size == 0:
             return ComboResult(template_id, timeframe, None, 0, 0, 0, n_iterations, 0, 0, [])

        local_scaler = StandardScaler()
        local_scaler.fit(combined_features)

        # 2. Transform features
        all_features_scaled = [local_scaler.transform(f) for f in all_features]

        # 3. Transform centroid using ORIGINAL scaler (if provided) to get its sigma representation
        # If original_scaler is None (should not happen if passed correctly), fallback to raw comparison (bad)
        if original_scaler:
            centroid_scaled = original_scaler.transform([centroid])[0]
        else:
            # Fallback: assume centroid is already scaled or we compare raw (likely to fail)
            centroid_scaled = centroid

        # Match indices
        match_indices_per_month = []
        for features_scaled in all_features_scaled:
            if len(features_scaled) == 0:
                match_indices_per_month.append([])
                continue

            # Distance in sigma space
            distances = np.linalg.norm(features_scaled - centroid_scaled, axis=1)
            matches = np.where(distances < 3.0)[0]
            match_indices_per_month.append(matches)

        # Parameter Generator
        # We need a dummy context detector for init
        class DummyContextDetector:
            pass

        param_generator = DOEParameterGenerator(DummyContextDetector())

        # Helper for mutation
        def mutate(base, scale):
            p = base.copy()
            ranges = param_generator._define_parameter_ranges()
            for name, val in base.items():
                if name in ranges:
                    lo, hi, dtype = ranges[name]
                    spread = (hi - lo) * scale if isinstance(hi, (int, float)) else 0
                    if dtype == 'int':
                         # spread is float, val is int
                        change = np.random.uniform(-spread, spread)
                        new_val = int(val + change)
                        p[name] = np.clip(new_val, lo, hi)
                    elif dtype == 'float':
                        change = np.random.uniform(-spread, spread)
                        p[name] = np.clip(val + change, lo, hi)
            return p

        # Keep only top 10 summaries by PnL (no individual trades in memory)
        import heapq
        top_heap = []  # min-heap of (pnl, IterationSummary)
        best_pnl = -float('inf')
        best_params = None
        best_win_rate = 0.0
        best_sharpe = -float('inf')
        grand_wins = 0
        grand_losses = 0

        for i in range(n_iterations):
            if mutation_base:
                params = mutate(mutation_base, mutation_scale)
            else:
                params = param_generator.generate_random_set(i)

            total_pnl = 0.0
            trade_pnls = []

            for month_idx, (month_data, matches) in enumerate(zip(all_data, match_indices_per_month)):
                if len(matches) == 0:
                    continue

                month_trades = simulate_month(
                    month_data, matches, params, asset,
                    template_id=template_id,
                    z_scores=all_z_scores[month_idx]
                )
                for t in month_trades:
                    trade_pnls.append(t.pnl)
                    total_pnl += t.pnl

            wins = sum(1 for p in trade_pnls if p > 0)
            losses = len(trade_pnls) - wins
            win_rate = wins / len(trade_pnls) if trade_pnls else 0.0
            grand_wins += wins
            grand_losses += losses

            # Compute sharpe
            if len(trade_pnls) > 1:
                arr = np.array(trade_pnls)
                sharpe = float(np.mean(arr) / np.std(arr)) if np.std(arr) > 0 else 0.0
            else:
                sharpe = 0.0

            # Max drawdown
            max_dd = 0.0
            if trade_pnls:
                cumsum = np.cumsum(trade_pnls)
                peak = np.maximum.accumulate(cumsum)
                max_dd = float(np.max(peak - cumsum))

            summary = IterationSummary(
                params=params, total_pnl=total_pnl, num_trades=len(trade_pnls),
                win_rate=win_rate, wins=wins, losses=losses,
                sharpe=sharpe, max_drawdown=max_dd
            )

            # Track best
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_params = params
                best_win_rate = win_rate
                best_sharpe = sharpe

            # Keep top 10 (i as tie-breaker so heapq never compares dataclasses)
            if len(top_heap) < 10:
                heapq.heappush(top_heap, (total_pnl, i, summary))
            elif total_pnl > top_heap[0][0]:
                heapq.heapreplace(top_heap, (total_pnl, i, summary))

        top_iterations = [s for _, _, s in sorted(top_heap, key=lambda x: -x[0])]

        return ComboResult(
            template_id=template_id,
            timeframe=timeframe,
            best_params=best_params,
            best_pnl=best_pnl if best_pnl > -float('inf') else 0.0,
            best_win_rate=best_win_rate,
            best_sharpe=best_sharpe if best_sharpe > -float('inf') else 0.0,
            total_iterations=n_iterations,
            total_wins=grand_wins,
            total_losses=grand_losses,
            top_iterations=top_iterations
        )

    except Exception as e:
        print(f"Worker Exception {template_id} {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        return ComboResult(template_id, timeframe, None, 0, 0, 0, n_iterations, 0, 0, [])


class MonteCarloEngine:
    """
    Brute-force simulator: sweep templates × timeframes × parameters.
    """
    def __init__(self, checkpoint_dir, asset, pattern_library, brain, num_workers=None):
        self.checkpoint_dir = checkpoint_dir
        self.asset = asset
        self.pattern_library = pattern_library
        self.brain = brain
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 2)

        self.results_db = {} # (tid, tf) -> ComboResult
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']

        # Load scaler
        scaler_path = os.path.join(checkpoint_dir, 'clustering_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"[MC] Loaded clustering scaler from {scaler_path}")
        else:
            print(f"[MC] WARNING: clustering_scaler.pkl not found in {checkpoint_dir}. Phase 2.5 might be incomplete.")
            self.scaler = None

    def _load_checkpoint(self) -> Set[Tuple[int, str]]:
        ckpt_path = os.path.join(self.checkpoint_dir, 'mc_sweep_state.pkl')
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, 'rb') as f:
                    data = pickle.load(f)
                    self.results_db = data.get('results_db', {})
                    return data.get('completed', set())
            except Exception as e:
                print(f"[MC] Error loading checkpoint: {e}")
        return set()

    def _save_checkpoint(self, completed):
        ckpt_path = os.path.join(self.checkpoint_dir, 'mc_sweep_state.pkl')
        # Save results_db, but maybe too big?
        # The plan says "mc_combo_results/ Individual combo results for memory efficiency".
        # Let's implement that.

        # Save main state
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'completed': completed,
                # We don't save full results_db here if we use separate files,
                # but we need to know what we have.
                # For now, let's keep results_db in memory and save it.
                # Optimization: clear results_db iterations after saving to individual files?
                # The ANOVA analyzer needs results_db.
                # Let's just save the summary info in results_db (drop trades list) if memory issue.
                'results_db': self.results_db
            }, f)

    def run_sweep(self, data_root='DATA/ATLAS', iterations_per_combo=2000):
        combos = list(itertools.product(
            self.pattern_library.keys(),
            self.timeframes
        ))

        completed = self._load_checkpoint()
        remaining = [(tid, tf) for tid, tf in combos if (tid, tf) not in completed]

        total = len(combos)
        done = len(completed)
        print(f"Monte Carlo Sweep: {total} combos, {len(remaining)} remaining, {self.num_workers} workers")
        print(f"  Iterations per combo: {iterations_per_combo}")

        if not remaining:
            print("All combos completed.")
            return

        import time as _time
        t_start = _time.perf_counter()
        bar_width = 40

        def _print_bar(done, total, elapsed, best_pnl, best_combo):
            pct = done / total if total else 0
            filled = int(bar_width * pct)
            bar = '█' * filled + '░' * (bar_width - filled)
            # ETA
            if done > 0 and elapsed > 0:
                eta_s = elapsed / done * (total - done)
                eta_m, eta_sec = divmod(int(eta_s), 60)
                eta_str = f"{eta_m}m{eta_sec:02d}s"
            else:
                eta_str = "???"
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            print(f"\r  [{bar}] {done}/{total} ({pct*100:.1f}%) | "
                  f"{elapsed_m}m{elapsed_s:02d}s elapsed | ETA {eta_str} | "
                  f"Best: ${best_pnl:+.0f} ({best_combo})", end='', flush=True)

        best_global_pnl = 0.0
        best_global_combo = "---"

        batch_size = self.num_workers
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            jobs = []
            for tid, tf in batch:
                jobs.append((
                    tid, tf, iterations_per_combo, data_root,
                    self.pattern_library[tid], self.asset, self.scaler
                ))

            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.starmap(simulate_template_tf_combo, jobs)

            for (tid, tf), result in zip(batch, results):
                self.results_db[(tid, tf)] = result

                # Feed aggregate stats to brain
                key = f"{tid}_{tf}"
                self.brain.table[key]['wins'] += result.total_wins
                self.brain.table[key]['losses'] += result.total_losses
                self.brain.table[key]['total'] += result.total_wins + result.total_losses

                # Track global best
                if result.best_pnl > best_global_pnl:
                    best_global_pnl = result.best_pnl
                    best_global_combo = f"T-{tid}@{tf}"

                completed.add((tid, tf))

            done = len(completed)
            elapsed = _time.perf_counter() - t_start
            self._save_checkpoint(completed)
            _print_bar(done, total, elapsed, best_global_pnl, best_global_combo)

        # Final newline after progress bar
        print(f"\n  Sweep complete! Best: ${best_global_pnl:+.2f} ({best_global_combo})")
