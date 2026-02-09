"""
BAYESIAN-AI TRAINING ORCHESTRATOR
Single entry point for all training operations

Integrates:
- Walk-forward training (day-by-day DOE)
- Live dashboard (real-time visualization)
- Pattern analysis (strongest states)
- Progress reporting (terminal output)
- Batch regret analysis (end-of-day evaluation)
- Checkpoint management (resume capability)
"""
import os
import sys
import pickle
import argparse
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import time

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Core components
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.quantum_field_engine import QuantumFieldEngine
from core.context_detector import ContextDetector
from core.adaptive_confidence import AdaptiveConfidenceManager

# Training components
from training.doe_parameter_generator import DOEParameterGenerator
from training.pattern_analyzer import PatternAnalyzer
from training.progress_reporter import ProgressReporter, DayMetrics
from training.databento_loader import DatabentoLoader

# Execution components
from execution.integrated_statistical_system import IntegratedStatisticalEngine
from execution.batch_regret_analyzer import BatchRegretAnalyzer

# Visualization
try:
    from visualization.live_training_dashboard import LiveDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("WARNING: Live dashboard not available")

# Configuration
from config.symbols import MNQ


@dataclass
class DayResults:
    """Results from one day of training"""
    day_number: int
    date: str
    total_iterations: int
    best_iteration: int
    best_params: Dict[str, Any]
    best_sharpe: float
    best_win_rate: float
    best_pnl: float
    total_trades: int
    states_learned: int
    high_confidence_states: int
    execution_time_seconds: float
    avg_duration: float


class BayesianTrainingOrchestrator:
    """
    UNIFIED TRAINING ORCHESTRATOR

    Runs complete walk-forward training with:
    - Day-by-day DOE parameter optimization
    - Live visualization dashboard
    - Real-time terminal progress
    - Pattern analysis and reporting
    - Batch regret analysis
    - Automatic checkpointing
    """

    def __init__(self, config):
        self.config = config
        self.asset = MNQ  # Default asset
        self.checkpoint_dir = config.checkpoint_dir

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize core components
        self.brain = QuantumBayesianBrain()
        self.engine = QuantumFieldEngine()
        self.context_detector = ContextDetector()
        self.param_generator = DOEParameterGenerator(self.context_detector)
        self.confidence_manager = AdaptiveConfidenceManager(self.brain)
        self.stat_validator = IntegratedStatisticalEngine(self.asset)

        # Analysis components
        self.pattern_analyzer = PatternAnalyzer()
        self.progress_reporter = ProgressReporter()
        self.regret_analyzer = BatchRegretAnalyzer()

        # Training state
        self.day_results: List[DayResults] = []
        self.todays_trades: List[TradeOutcome] = []
        self.dashboard = None
        self.dashboard_thread = None

    def train(self, data: pd.DataFrame):
        """
        Master training loop

        Args:
            data: Full dataset with timestamps
        """
        print("\n" + "="*80)
        print("BAYESIAN-AI TRAINING ORCHESTRATOR")
        print("="*80)

        # CUDA status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                print(f"CUDA: ENABLED  |  GPU: {gpu_name}  |  VRAM: {gpu_mem:.1f} GB")
            else:
                print("CUDA: NOT AVAILABLE  |  Running on CPU")
        except ImportError:
            print("CUDA: NOT AVAILABLE (PyTorch not installed)  |  Running on CPU")

        print(f"Asset: {self.asset.ticker}")
        print(f"Checkpoint Dir: {self.checkpoint_dir}")
        print(f"Iterations per Day: {self.config.iterations}")

        # Launch dashboard if available
        if DASHBOARD_AVAILABLE and not self.config.no_dashboard:
            self.launch_dashboard()

        # Split into trading days
        days = self.split_into_trading_days(data)
        total_days = len(days)

        if self.config.max_days:
            days = days[:self.config.max_days]
            total_days = len(days)
            print(f"Limiting to first {total_days} days")

        print(f"\nTraining on {total_days} trading days...")
        print(f"Date range: {days[0][0]} to {days[-1][0]}")
        print("="*80 + "\n")

        # Train day by day
        for day_idx, (date, day_data) in enumerate(days):
            day_number = day_idx + 1

            # Print day header
            self.progress_reporter.print_day_header(
                day_number, date, total_days, len(day_data)
            )

            # Optimize this day (DOE)
            day_result = self.optimize_day(day_number, date, day_data)

            # Batch regret analysis (end of day)
            if self.todays_trades:
                regret_analysis = self.regret_analyzer.batch_analyze_day(
                    self.todays_trades,
                    day_data
                )
                self.regret_analyzer.print_analysis(regret_analysis)
            else:
                regret_analysis = None

            # Update reports
            day_metrics = DayMetrics(
                day_number=day_number,
                date=date,
                total_trades=day_result.total_trades,
                win_rate=day_result.best_win_rate,
                sharpe=day_result.best_sharpe,
                pnl=day_result.best_pnl,
                states_learned=day_result.states_learned,
                high_conf_states=day_result.high_confidence_states,
                avg_duration=day_result.avg_duration,
                execution_time=day_result.execution_time_seconds
            )

            self.progress_reporter.print_day_summary(day_metrics)

            # Get top patterns
            top_patterns = self.pattern_analyzer.get_strongest_patterns(self.brain, top_n=5)
            self.progress_reporter.print_cumulative_summary(top_patterns)

            # Update dashboard if available
            if self.dashboard:
                try:
                    self.dashboard.update(day_metrics, regret_analysis)
                except:
                    pass  # Dashboard update is non-critical

            # Save checkpoint
            self.save_checkpoint(day_number, date, day_result)

            self.day_results.append(day_result)

        # Final summary
        self.print_final_summary()

        return self.day_results

    def _precompute_day_states(self, day_data: pd.DataFrame):
        """
        Pre-compute all states, probabilities, and confidences for a day ONCE.

        Uses vectorized batch computation (numpy arrays + optional CUDA)
        instead of per-bar loop. ~10-50x faster than original.

        Returns:
            List of dicts with bar_idx, state, price, prob, conf, structure_ok
        """
        if len(day_data) < 21:
            return []

        # Ensure 'close' column exists for batch computation
        if 'price' in day_data.columns and 'close' not in day_data.columns:
            day_data = day_data.copy()
            day_data['close'] = day_data['price']

        # Pre-extract arrays for fast trade simulation
        prices = day_data['price'].values if 'price' in day_data.columns else day_data['close'].values
        timestamps = day_data['timestamp'].values if 'timestamp' in day_data.columns else np.zeros(len(day_data))

        print(f"  Pre-computing states for {len(day_data) - 21:,} bars (vectorized)...", end='', flush=True)
        precompute_start = time.time()

        # === VECTORIZED BATCH COMPUTATION ===
        batch_results = self.engine.batch_compute_states(day_data, use_cuda=True)

        # Add prob/conf from brain (these are dict lookups, fast)
        for bar in batch_results:
            state = bar['state']
            prob = self.brain.get_probability(state)
            conf = self.brain.get_confidence(state)
            bar['prob'] = prob
            bar['conf'] = conf
            # Refine structure_ok with confidence check
            bar['structure_ok'] = bar['structure_ok'] and conf >= 0.30

        elapsed = time.time() - precompute_start
        print(f" done ({elapsed:.1f}s)")

        # Store prices/timestamps as arrays for fast trade sim
        self._day_prices = prices
        self._day_timestamps = timestamps

        return batch_results

    def _simulate_from_precomputed(self, precomputed: list, day_data: pd.DataFrame,
                                    params: Dict[str, Any], pbar=None, best_sharpe: float = -999.0) -> List[TradeOutcome]:
        """
        Fast simulation using pre-computed states. Only applies param thresholds + trade sim.
        """
        trades = []
        min_prob = params.get('confidence_threshold', 0.80)
        stop_loss = params.get('stop_loss_ticks', 15) * 0.25
        take_profit = params.get('take_profit_ticks', 40) * 0.25
        max_hold = params.get('max_hold_seconds', 600)

        prices = self._day_prices
        timestamps = self._day_timestamps
        max_lookahead = 200
        n_bars = len(prices)

        trade_wins = 0
        trade_pnl = 0.0

        for bar in precomputed:
            # Only param-dependent check: probability threshold
            if not bar['structure_ok'] or bar['prob'] < min_prob:
                continue

            # --- Inline fast trade simulation (avoid DataFrame overhead) ---
            entry_idx = bar['bar_idx']
            entry_price = bar['price']
            entry_time = timestamps[entry_idx]
            if isinstance(entry_time, pd.Timestamp):
                entry_time = entry_time.timestamp()
            elif hasattr(entry_time, 'item'):
                entry_time = pd.Timestamp(entry_time).timestamp()

            end_idx = min(n_bars, entry_idx + max_lookahead)
            outcome = None

            for j in range(entry_idx + 1, end_idx):
                price = prices[j]
                pnl = price - entry_price

                exit_time = timestamps[j]
                if isinstance(exit_time, pd.Timestamp):
                    exit_time = exit_time.timestamp()
                elif hasattr(exit_time, 'item'):
                    exit_time = pd.Timestamp(exit_time).timestamp()

                duration = exit_time - entry_time

                if pnl >= take_profit:
                    outcome = TradeOutcome(
                        state=bar['state'], entry_price=entry_price, exit_price=price,
                        pnl=take_profit, result='WIN', timestamp=exit_time,
                        exit_reason='TP', entry_time=entry_time, duration=duration
                    )
                    break
                elif pnl <= -stop_loss:
                    outcome = TradeOutcome(
                        state=bar['state'], entry_price=entry_price, exit_price=price,
                        pnl=-stop_loss, result='LOSS', timestamp=exit_time,
                        exit_reason='SL', entry_time=entry_time, duration=duration
                    )
                    break
                elif duration >= max_hold:
                    outcome = TradeOutcome(
                        state=bar['state'], entry_price=entry_price, exit_price=price,
                        pnl=pnl, result='WIN' if pnl > 0 else 'LOSS', timestamp=exit_time,
                        exit_reason='TIME', entry_time=entry_time, duration=duration
                    )
                    break

            # End of data fallback
            if outcome is None and entry_idx + 1 < n_bars:
                last_price = prices[end_idx - 1]
                pnl = last_price - entry_price
                last_time = timestamps[end_idx - 1]
                if isinstance(last_time, pd.Timestamp):
                    last_time = last_time.timestamp()
                elif hasattr(last_time, 'item'):
                    last_time = pd.Timestamp(last_time).timestamp()
                outcome = TradeOutcome(
                    state=bar['state'], entry_price=entry_price, exit_price=last_price,
                    pnl=pnl, result='WIN' if pnl > 0 else 'LOSS', timestamp=last_time,
                    exit_reason='EOD', entry_time=entry_time, duration=last_time - entry_time
                )

            if outcome:
                trades.append(outcome)
                trade_pnl += outcome.pnl
                if outcome.result == 'WIN':
                    trade_wins += 1
                # Live per-trade update
                if pbar:
                    n = len(trades)
                    pbar.set_postfix({
                        'Trades': n,
                        'WR': f'{trade_wins/n:.0%}',
                        'P&L': f'${trade_pnl:.0f}',
                        'Best': f'{best_sharpe:.2f}'
                    })

        return trades

    def optimize_day(self, day_number: int, date: str, day_data: pd.DataFrame) -> DayResults:
        """
        Run DOE optimization for single day.
        Routes to GPU-parallel path when CUDA available, CPU sequential otherwise.
        """
        start_time = time.time()
        self.todays_trades = []

        # === PRE-COMPUTE ALL STATES ONCE ===
        precomputed = self._precompute_day_states(day_data)

        if not precomputed:
            return DayResults(
                day_number=day_number, date=date, total_iterations=self.config.iterations,
                best_iteration=0, best_params={}, best_sharpe=0.0, best_win_rate=0.0,
                best_pnl=0.0, total_trades=0, states_learned=len(self.brain.table),
                high_confidence_states=len(self.brain.get_all_states_above_threshold()),
                execution_time_seconds=time.time() - start_time, avg_duration=0.0
            )

        # Generate ALL param sets upfront
        all_param_sets = []
        for i in range(self.config.iterations):
            ps = self.param_generator.generate_parameter_set(iteration=i, day=day_number, context='CORE')
            all_param_sets.append(ps.parameters)

        # Route to GPU or CPU path
        try:
            import torch
            if torch.cuda.is_available():
                best_idx, all_results = self._optimize_gpu_parallel(
                    precomputed, day_data, all_param_sets, day_number
                )
            else:
                best_idx, all_results = self._optimize_cpu_sequential(
                    precomputed, day_data, all_param_sets, day_number
                )
        except ImportError:
            best_idx, all_results = self._optimize_cpu_sequential(
                precomputed, day_data, all_param_sets, day_number
            )

        # Unpack best result
        best_sharpe = all_results[best_idx]['sharpe']
        best_params = all_param_sets[best_idx]
        best_trades = all_results[best_idx]['trades']

        # Collect all trades for regret analysis
        for r in all_results:
            self.todays_trades.extend(r['trades'])

        # Update brain with best iteration's trades
        for trade in best_trades:
            self.brain.update(trade)
            self.confidence_manager.record_trade(trade)

        if best_params:
            self.param_generator.update_best_params(best_params)

        execution_time = time.time() - start_time

        if best_trades:
            pnls = [t.pnl for t in best_trades]
            durations = [t.duration for t in best_trades]
            wins = sum(1 for t in best_trades if t.result == 'WIN')
            best_win_rate = wins / len(best_trades)
            best_pnl = sum(pnls)
            avg_duration = np.mean(durations)
        else:
            best_win_rate = 0.0
            best_pnl = 0.0
            avg_duration = 0.0

        return DayResults(
            day_number=day_number, date=date,
            total_iterations=self.config.iterations,
            best_iteration=best_idx,
            best_params=best_params or {},
            best_sharpe=best_sharpe,
            best_win_rate=best_win_rate,
            best_pnl=best_pnl,
            total_trades=len(best_trades),
            states_learned=len(self.brain.table),
            high_confidence_states=len(self.brain.get_all_states_above_threshold()),
            execution_time_seconds=execution_time,
            avg_duration=avg_duration
        )

    def _optimize_gpu_parallel(self, precomputed, day_data, all_param_sets, day_number):
        """
        GPU-PARALLEL: Run ALL iterations simultaneously on CUDA.
        Each iteration = different param thresholds applied to same precomputed states.
        All trade simulations run in parallel across iterations.
        """
        import torch
        device = torch.device('cuda')
        n_iters = len(all_param_sets)

        # Extract param arrays → GPU tensors [n_iters]
        thresholds = torch.tensor([p.get('confidence_threshold', 0.80) for p in all_param_sets],
                                  device=device, dtype=torch.float64)
        tp_points = torch.tensor([p.get('take_profit_ticks', 40) * 0.25 for p in all_param_sets],
                                 device=device, dtype=torch.float64)
        sl_points = torch.tensor([p.get('stop_loss_ticks', 15) * 0.25 for p in all_param_sets],
                                 device=device, dtype=torch.float64)
        max_holds = torch.tensor([p.get('max_hold_seconds', 600) for p in all_param_sets],
                                 device=device, dtype=torch.float64)

        # Price/time arrays → GPU
        prices_gpu = torch.tensor(self._day_prices, device=device, dtype=torch.float64)
        # Convert timestamps to float seconds
        ts_raw = self._day_timestamps
        if hasattr(ts_raw[0], 'timestamp'):
            ts_float = np.array([t.timestamp() for t in ts_raw], dtype=np.float64)
        elif hasattr(ts_raw[0], 'item'):
            ts_float = np.array([pd.Timestamp(t).timestamp() for t in ts_raw], dtype=np.float64)
        else:
            ts_float = ts_raw.astype(np.float64)
        times_gpu = torch.tensor(ts_float, device=device, dtype=torch.float64)
        n_bars_total = len(self._day_prices)

        # Filter bars that pass structure_ok
        candidate_bars = [b for b in precomputed if b['structure_ok']]
        n_candidates = len(candidate_bars)

        print(f"  GPU parallel: {n_iters} iterations × {n_candidates} candidate bars on CUDA")

        if n_candidates == 0:
            empty = [{'trades': [], 'sharpe': -999.0, 'win_rate': 0.0, 'pnl': 0.0} for _ in range(n_iters)]
            return 0, empty

        # Extract candidate probs → GPU [n_candidates]
        cand_probs = torch.tensor([b['prob'] for b in candidate_bars], device=device, dtype=torch.float64)
        cand_indices = [b['bar_idx'] for b in candidate_bars]
        cand_prices_entry = torch.tensor([b['price'] for b in candidate_bars], device=device, dtype=torch.float64)

        max_lookahead = 200

        # === CORE GPU KERNEL: simulate all trades in parallel ===
        # For each candidate × each iteration: compute trade result
        # Result tensors: [n_candidates, n_iters]
        trade_pnl = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.float64)
        trade_won = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.bool)
        trade_fired = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.bool)
        trade_duration = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.float64)

        for c_idx in range(n_candidates):
            bar_idx = cand_indices[c_idx]
            prob = cand_probs[c_idx]
            entry_price = cand_prices_entry[c_idx]
            entry_time = times_gpu[bar_idx]

            # Which iterations fire? (prob >= threshold)
            fires = prob >= thresholds  # [n_iters] bool
            if not fires.any():
                continue

            # Look ahead window
            end_idx = min(n_bars_total, bar_idx + max_lookahead)
            if bar_idx + 1 >= end_idx:
                continue

            future_prices = prices_gpu[bar_idx + 1:end_idx]  # [window]
            future_times = times_gpu[bar_idx + 1:end_idx]    # [window]
            window_len = len(future_prices)

            # P&L for each future bar: [window]
            pnl_curve = future_prices - entry_price
            durations = future_times - entry_time  # [window]

            # For each firing iteration, find exit bar
            # Expand for broadcasting: pnl_curve [window,1] vs tp/sl [1,n_iters]
            pnl_expanded = pnl_curve.unsqueeze(1)       # [window, 1]
            dur_expanded = durations.unsqueeze(1)        # [window, 1]
            tp_expanded = tp_points.unsqueeze(0)         # [1, n_iters]
            sl_expanded = sl_points.unsqueeze(0)         # [1, n_iters]
            mh_expanded = max_holds.unsqueeze(0)         # [1, n_iters]

            # Exit conditions: [window, n_iters]
            hit_tp = pnl_expanded >= tp_expanded
            hit_sl = pnl_expanded <= -sl_expanded
            hit_time = dur_expanded >= mh_expanded
            exit_mask = hit_tp | hit_sl | hit_time  # [window, n_iters]

            # Find first exit bar for each iteration
            # Use argmax on exit_mask (returns first True index along dim=0)
            any_exit = exit_mask.any(dim=0)  # [n_iters]

            # For iterations that have an exit
            firing_and_exiting = fires & any_exit
            if not firing_and_exiting.any():
                # All firing iterations reach EOD
                fire_indices = fires.nonzero(as_tuple=True)[0]
                last_pnl = pnl_curve[-1]
                last_dur = durations[-1]
                trade_fired[c_idx, fire_indices] = True
                trade_pnl[c_idx, fire_indices] = last_pnl
                trade_won[c_idx, fire_indices] = last_pnl > 0
                trade_duration[c_idx, fire_indices] = last_dur
                continue

            # Get first exit index per iteration: [n_iters]
            # Set non-exit positions to window_len so argmax ignores them
            exit_indices = exit_mask.float().argmax(dim=0)  # [n_iters]

            for iter_idx in firing_and_exiting.nonzero(as_tuple=True)[0]:
                ii = iter_idx.item()
                exit_bar = exit_indices[ii].item()
                pnl_val = pnl_curve[exit_bar].item()
                dur_val = durations[exit_bar].item()

                if hit_tp[exit_bar, ii]:
                    final_pnl = tp_points[ii].item()
                    won = True
                elif hit_sl[exit_bar, ii]:
                    final_pnl = -sl_points[ii].item()
                    won = False
                else:  # time exit
                    final_pnl = pnl_val
                    won = pnl_val > 0

                trade_fired[c_idx, ii] = True
                trade_pnl[c_idx, ii] = final_pnl
                trade_won[c_idx, ii] = won
                trade_duration[c_idx, ii] = dur_val

            # Handle firing but no-exit iterations (EOD)
            fire_no_exit = fires & ~any_exit
            if fire_no_exit.any():
                fe_indices = fire_no_exit.nonzero(as_tuple=True)[0]
                last_pnl = pnl_curve[-1]
                last_dur = durations[-1]
                trade_fired[c_idx, fe_indices] = True
                trade_pnl[c_idx, fe_indices] = last_pnl
                trade_won[c_idx, fe_indices] = last_pnl > 0
                trade_duration[c_idx, fe_indices] = last_dur

        # === AGGREGATE RESULTS PER ITERATION (on GPU) ===
        # trade_fired: [n_candidates, n_iters]
        n_trades = trade_fired.sum(dim=0)               # [n_iters]
        n_wins = (trade_fired & trade_won).sum(dim=0)   # [n_iters]
        total_pnl = (trade_pnl * trade_fired).sum(dim=0)  # [n_iters]

        # Sharpe per iteration (on GPU)
        # Need mean and std of PnL per iteration
        sharpes = torch.full((n_iters,), -999.0, device=device, dtype=torch.float64)
        for ii in range(n_iters):
            mask = trade_fired[:, ii]
            count = mask.sum().item()
            if count >= 5:
                pnls_iter = trade_pnl[mask, ii]
                mean_pnl = pnls_iter.mean()
                std_pnl = pnls_iter.std() + 1e-6
                sharpes[ii] = mean_pnl / std_pnl

        # Find best iteration
        best_idx = sharpes.argmax().item()
        best_sharpe = sharpes[best_idx].item()

        # Move results to CPU
        n_trades_cpu = n_trades.cpu().numpy()
        n_wins_cpu = n_wins.cpu().numpy()
        total_pnl_cpu = total_pnl.cpu().numpy()
        sharpes_cpu = sharpes.cpu().numpy()

        # Print best
        if best_sharpe > -999.0:
            best_n = int(n_trades_cpu[best_idx])
            best_wr = int(n_wins_cpu[best_idx]) / best_n if best_n > 0 else 0
            print(f"  GPU result: Best iter {best_idx} | Sharpe: {best_sharpe:.2f} | "
                  f"WR: {best_wr:.1%} | Trades: {best_n} | P&L: ${total_pnl_cpu[best_idx]:.2f}")

        # Build result dicts — reconstruct TradeOutcome objects only for best iteration
        all_results = []
        for ii in range(n_iters):
            n = int(n_trades_cpu[ii])
            wr = int(n_wins_cpu[ii]) / n if n > 0 else 0.0
            all_results.append({
                'trades': [],  # Populated only for best below
                'sharpe': float(sharpes_cpu[ii]),
                'win_rate': wr,
                'pnl': float(total_pnl_cpu[ii]),
            })

        # Reconstruct TradeOutcome objects for the BEST iteration only
        best_trades = []
        best_mask = trade_fired[:, best_idx]
        for c_idx in range(n_candidates):
            if not best_mask[c_idx].item():
                continue
            bar = candidate_bars[c_idx]
            pnl_val = trade_pnl[c_idx, best_idx].item()
            won = trade_won[c_idx, best_idx].item()
            dur = trade_duration[c_idx, best_idx].item()
            entry_time_val = ts_float[bar['bar_idx']]

            best_trades.append(TradeOutcome(
                state=bar['state'],
                entry_price=bar['price'],
                exit_price=bar['price'] + pnl_val,
                pnl=pnl_val,
                result='WIN' if won else 'LOSS',
                timestamp=entry_time_val + dur,
                exit_reason='GPU',
                entry_time=entry_time_val,
                duration=dur,
            ))

        all_results[best_idx]['trades'] = best_trades

        return best_idx, all_results

    def _optimize_cpu_sequential(self, precomputed, day_data, all_param_sets, day_number):
        """CPU fallback: run iterations sequentially (still uses precomputed states)."""
        n_iters = len(all_param_sets)
        best_sharpe = -999.0
        best_idx = 0
        all_results = []

        pbar = tqdm(range(n_iters), desc=f"Optimizing Day {day_number} (CPU)", ncols=100)

        for iteration in pbar:
            trades = self._simulate_from_precomputed(
                precomputed, day_data, all_param_sets[iteration], pbar, best_sharpe
            )

            if trades:
                pnls = [t.pnl for t in trades]
                wins = sum(1 for t in trades if t.result == 'WIN')
                win_rate = wins / len(trades)
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6)
            else:
                win_rate = 0.0
                sharpe = 0.0

            all_results.append({
                'trades': trades,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'pnl': sum(t.pnl for t in trades),
            })

            if sharpe > best_sharpe and len(trades) >= 5:
                best_sharpe = sharpe
                best_idx = iteration
                best_pnl_so_far = sum(t.pnl for t in trades)
                tqdm.write(f"  [Iter {iteration:3d}] New best! Sharpe: {best_sharpe:.2f} | WR: {win_rate:.1%} | Trades: {len(trades)} | P&L: ${best_pnl_so_far:.2f}")

            pbar.set_postfix({
                'Trades': len(trades),
                'WR': f'{win_rate:.1%}',
                'Sharpe': f'{sharpe:.2f}',
                'Best': f'{best_sharpe:.2f}'
            })

        pbar.close()
        return best_idx, all_results

    def simulate_trading_day(self, day_data: pd.DataFrame, params: Dict[str, Any], on_trade=None) -> List[TradeOutcome]:
        """
        Fast simulation of trading day with given parameters

        No regret analysis overhead - just execute trades

        Args:
            day_data: OHLCV data for single day
            params: Parameter set to test
            on_trade: Optional callback called with each TradeOutcome as it fires

        Returns:
            List of TradeOutcome objects
        """
        trades = []

        # Need at least 21 bars for regression
        if len(day_data) < 21:
            return trades

        # Simulate bar-by-bar
        for i in range(21, len(day_data)):
            current_row = day_data.iloc[i]

            # Get macro context
            df_macro = day_data.iloc[i-21:i+1].copy()
            if 'price' in df_macro.columns and 'close' not in df_macro.columns:
                df_macro['close'] = df_macro['price']

            # Calculate state
            current_price = current_row['price'] if 'price' in current_row else current_row['close']
            current_volume = current_row.get('volume', 0)
            if pd.isna(current_volume):
                current_volume = 0.0

            state = self.engine.calculate_three_body_state(
                df_macro=df_macro,
                df_micro=df_macro,
                current_price=current_price,
                current_volume=float(current_volume),
                tick_velocity=0.0
            )

            # Decision (using parameters)
            min_prob = params.get('confidence_threshold', 0.80)
            min_conf = 0.30

            prob = self.brain.get_probability(state)
            conf = self.brain.get_confidence(state)

            # Check if should fire
            should_fire = (
                state.lagrange_zone in ['L2_ROCHE', 'L3_ROCHE'] and
                state.structure_confirmed and
                state.cascade_detected and
                prob >= min_prob and
                conf >= min_conf
            )

            if should_fire:
                # Simulate trade
                outcome = self._simulate_trade(
                    current_idx=i,
                    entry_price=current_price,
                    data=day_data,
                    state=state,
                    params=params
                )

                if outcome:
                    trades.append(outcome)
                    if on_trade:
                        on_trade(outcome)

        return trades

    def _simulate_trade(self, current_idx: int, entry_price: float,
                       data: pd.DataFrame, state: Any, params: Dict[str, Any]) -> Optional[TradeOutcome]:
        """
        Simulate single trade with lookahead

        Uses params for stop loss and take profit
        """
        stop_loss = params.get('stop_loss_ticks', 15) * 0.25  # Convert ticks to points
        take_profit = params.get('take_profit_ticks', 40) * 0.25
        max_hold = params.get('max_hold_seconds', 600)

        # Look ahead
        max_lookahead = 200
        end_idx = min(len(data), current_idx + max_lookahead)
        future_data = data.iloc[current_idx+1:end_idx]

        if future_data.empty:
            return None

        entry_time = data.iloc[current_idx].get('timestamp', 0)
        if isinstance(entry_time, pd.Timestamp):
            entry_time = entry_time.timestamp()

        for idx, row in future_data.iterrows():
            price = row['price'] if 'price' in row else row['close']
            pnl = price - entry_price  # Long only

            exit_time = row.get('timestamp', 0)
            if isinstance(exit_time, pd.Timestamp):
                exit_time = exit_time.timestamp()

            duration = exit_time - entry_time

            # Check TP/SL
            if pnl >= take_profit:
                return TradeOutcome(
                    state=state,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl=take_profit,
                    result='WIN',
                    timestamp=exit_time,
                    exit_reason='TP',
                    entry_time=entry_time,
                    duration=duration
                )
            elif pnl <= -stop_loss:
                return TradeOutcome(
                    state=state,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl=-stop_loss,
                    result='LOSS',
                    timestamp=exit_time,
                    exit_reason='SL',
                    entry_time=entry_time,
                    duration=duration
                )
            elif duration >= max_hold:
                return TradeOutcome(
                    state=state,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl=pnl,
                    result='WIN' if pnl > 0 else 'LOSS',
                    timestamp=exit_time,
                    exit_reason='TIME',
                    entry_time=entry_time,
                    duration=duration
                )

        # Reached end of data
        last_price = future_data.iloc[-1]['price'] if 'price' in future_data.iloc[-1] else future_data.iloc[-1]['close']
        pnl = last_price - entry_price

        last_time = future_data.iloc[-1].get('timestamp', 0)
        if isinstance(last_time, pd.Timestamp):
            last_time = last_time.timestamp()

        return TradeOutcome(
            state=state,
            entry_price=entry_price,
            exit_price=last_price,
            pnl=pnl,
            result='WIN' if pnl > 0 else 'LOSS',
            timestamp=last_time,
            exit_reason='EOD',
            entry_time=entry_time,
            duration=last_time - entry_time
        )

    def split_into_trading_days(self, data: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Split data into trading days"""
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

        # Extract date
        data['date'] = data['timestamp'].dt.date

        # Group by date
        days = []
        for date, day_df in data.groupby('date'):
            date_str = str(date)
            days.append((date_str, day_df.reset_index(drop=True)))

        return days

    def launch_dashboard(self):
        """Launch dashboard in background thread"""
        def run_dashboard():
            try:
                self.dashboard = LiveDashboard()
                self.dashboard.launch()
            except Exception as e:
                print(f"WARNING: Dashboard failed to launch: {e}")

        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        print("Dashboard launching in background...")
        time.sleep(2)  # Give it time to initialize

    def save_checkpoint(self, day_number: int, date: str, day_result: DayResults):
        """Save brain and parameters to checkpoint"""
        # Save brain
        brain_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_brain.pkl")
        self.brain.save(brain_path)

        # Save best params
        if day_result.best_params:
            params_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_params.pkl")
            with open(params_path, 'wb') as f:
                pickle.dump(day_result.best_params, f)

        # Save day results
        results_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(day_result, f)

    def print_final_summary(self):
        """Print comprehensive final summary"""
        self.progress_reporter.print_final_summary()

        # Pattern analysis report
        pattern_report = self.pattern_analyzer.generate_pattern_report(
            self.brain,
            self.day_results
        )
        print(pattern_report)

        # Save progress log
        log_path = os.path.join(self.checkpoint_dir, "training_log.json")
        self.progress_reporter.save_progress_log(log_path)


def check_and_install_requirements():
    """Auto-install requirements.txt if missing packages detected"""
    import subprocess
    requirements_path = os.path.join(PROJECT_ROOT, 'requirements.txt')
    if not os.path.exists(requirements_path):
        return

    print("Checking dependencies...")
    try:
        # pip install --quiet skips already-installed packages fast
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"WARNING: Some dependencies failed to install:\n{result.stderr[:500]}")
        else:
            # Check CUDA status after install
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"Dependencies OK | CUDA ready: {torch.cuda.get_device_name(0)}")
                else:
                    print("Dependencies OK | WARNING: CUDA not available — GPU acceleration disabled")
            except ImportError:
                print("Dependencies OK | WARNING: PyTorch not installed — install manually:")
                print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except subprocess.TimeoutExpired:
        print("WARNING: pip install timed out, continuing anyway...")
    except Exception as e:
        print(f"WARNING: Could not check dependencies: {e}")


def main():
    """Single entry point - command line interface"""
    # Auto-install dependencies
    check_and_install_requirements()

    parser = argparse.ArgumentParser(
        description="Bayesian-AI Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', required=True, help="Path to parquet data file")
    parser.add_argument('--iterations', type=int, default=1000, help="Iterations per day (default: 1000)")
    parser.add_argument('--max-days', type=int, default=None, help="Limit number of days")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--no-dashboard', action='store_true', help="Disable live dashboard")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        if args.data.endswith('.parquet'):
            data = pd.read_parquet(args.data)
        else:
            # Try databento loader
            data = DatabentoLoader.load_data(args.data)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return 1

    print(f"Loaded {len(data):,} rows")

    # Create orchestrator
    orchestrator = BayesianTrainingOrchestrator(args)

    # Run training
    try:
        results = orchestrator.train(data)
        print("\n=== Training Complete ===")
        return 0
    except KeyboardInterrupt:
        print("\n\nWARNING: Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
