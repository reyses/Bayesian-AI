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

    def optimize_day(self, day_number: int, date: str, day_data: pd.DataFrame) -> DayResults:
        """
        Run DOE optimization for single day

        Tests N parameter combinations and finds best

        Args:
            day_number: Day index in sequence
            date: Date string (YYYY-MM-DD)
            day_data: OHLCV data for this day

        Returns:
            DayResults with best parameters and metrics
        """
        start_time = time.time()
        self.todays_trades = []  # Reset

        iteration_results = []
        best_sharpe = -999.0
        best_params = None
        best_trades = []

        # Progress bar
        pbar = tqdm(
            range(self.config.iterations),
            desc=f"Optimizing Day {day_number}",
            ncols=100
        )

        for iteration in pbar:
            # Generate parameters
            param_set = self.param_generator.generate_parameter_set(
                iteration=iteration,
                day=day_number,
                context='CORE'
            )

            # Simulate trading day (FAST - no regret overhead)
            trades = self.simulate_trading_day(day_data, param_set.parameters)

            # Calculate metrics
            if trades:
                pnls = [t.pnl for t in trades]
                wins = sum(1 for t in trades if t.result == 'WIN')
                win_rate = wins / len(trades)
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6)
            else:
                win_rate = 0.0
                sharpe = 0.0

            # Track best
            if sharpe > best_sharpe and len(trades) >= 5:
                best_sharpe = sharpe
                best_params = param_set.parameters
                best_trades = trades

                # Update progress bar
                pbar.set_postfix({
                    'Trades': len(trades),
                    'WR': f'{win_rate:.1%}',
                    'Sharpe': f'{sharpe:.2f}'
                })

            # Collect all trades for regret analysis
            self.todays_trades.extend(trades)

        pbar.close()

        # Update brain with best iteration's trades
        for trade in best_trades:
            self.brain.update(trade)
            self.confidence_manager.record_trade(trade)

        # Record best parameters
        if best_params:
            self.param_generator.update_best_params(best_params)

        # Calculate metrics
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
            day_number=day_number,
            date=date,
            total_iterations=self.config.iterations,
            best_iteration=0,  # TODO: Track which iteration was best
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

    def simulate_trading_day(self, day_data: pd.DataFrame, params: Dict[str, Any]) -> List[TradeOutcome]:
        """
        Fast simulation of trading day with given parameters

        No regret analysis overhead - just execute trades

        Args:
            day_data: OHLCV data for single day
            params: Parameter set to test

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


def main():
    """Single entry point - command line interface"""
    parser = argparse.ArgumentParser(
        description="Bayesian-AI Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', required=True, help="Path to parquet data file")
    parser.add_argument('--iterations', type=int, default=1000, help="Iterations per day (default: 1000)")
    parser.add_argument('--max-days', type=int, default=None, help="Limit number of days")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--no-dashboard', action='store_true', help="Disable live dashboard")

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
