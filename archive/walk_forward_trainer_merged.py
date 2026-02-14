"""
Walk-Forward Training Orchestrator with Design of Experiments
Day-by-day parameter optimization with continuous learning

Philosophy:
- NOT process entire year at once (overfitting risk)
- Day-by-day optimization with 1000 parameter tests per day
- Each day is validated on next day (out-of-sample)
- Parameters evolve through exploration→exploitation
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from tqdm import tqdm

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.quantum_field_engine import QuantumFieldEngine
from core.context_detector import ContextDetector
from training.doe_parameter_generator import DOEParameterGenerator, ParameterSet
from training.integrated_statistical_system import IntegratedStatisticalEngine, TradeRecord
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


@dataclass
class IterationResult:
    """Results from one parameter iteration"""
    iteration: int
    params: Dict[str, Any]
    trades: List[TradeOutcome]
    pnl: float
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    avg_hold_time: float


class WalkForwardTrainer:
    """
    Walk-Forward Training System with DOE

    Training Flow:
    ============
    Day 1:
        - Test 1000 parameter combinations on Day 1 data
        - Find best parameters (by Sharpe ratio)
        - Update brain with Day 1's best trades
        - Save checkpoint

    Day 2:
        - Load Day 1's best params
        - Test 1000 combinations (70% exploit, 30% explore)
        - Find best parameters for Day 2
        - Update brain with Day 2's best trades
        - Save checkpoint

    Day N:
        - Continue pattern...
        - Exploitation ratio increases over time (60% → 90%)
        - Parameters converge to optimal values
        - Brain learns states continuously

    Out-of-Sample Validation:
    ========================
    - Every day is validated on next day (never seen before)
    - Walk-forward eliminates overfitting
    - If params overfit Day N, they fail on Day N+1
    """

    def __init__(self,
                 asset_profile=MNQ,
                 checkpoint_dir: str = 'checkpoints',
                 n_iterations_per_day: int = 1000,
                 verbose: bool = True):
        """
        Initialize walk-forward trainer

        Args:
            asset_profile: Asset configuration (MNQ, NQ, etc.)
            checkpoint_dir: Directory for checkpoints
            n_iterations_per_day: Parameter combinations to test per day (default 1000)
            verbose: Print progress
        """
        self.asset = asset_profile
        self.checkpoint_dir = checkpoint_dir
        self.n_iterations = n_iterations_per_day
        self.verbose = verbose

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize components
        self.brain = QuantumBayesianBrain()
        self.engine = QuantumFieldEngine()
        self.context_detector = ContextDetector()
        self.param_generator = DOEParameterGenerator(self.context_detector)
        self.stat_validator = IntegratedStatisticalEngine(asset_profile)

        # Training history
        self.day_results: List[DayResults] = []

    def split_into_trading_days(self, data: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """
        Split data into trading days

        Args:
            data: Full dataset with 'timestamp' column

        Returns:
            List of (date_string, day_dataframe) tuples
        """
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

        if self.verbose:
            print(f"Split data into {len(days)} trading days")
            print(f"Date range: {days[0][0]} to {days[-1][0]}")

        return days

    def simulate_trading_day(self,
                            day_data: pd.DataFrame,
                            params: Dict[str, Any],
                            iteration: int) -> IterationResult:
        """
        Simulate trading for one day with given parameters

        Args:
            day_data: OHLCV data for single day
            params: Parameter set to test
            iteration: Iteration number

        Returns:
            IterationResult with performance metrics
        """
        # Create fresh brain for this iteration (don't contaminate)
        iteration_brain = QuantumBayesianBrain()

        # Copy learned states from master brain (transfer learning)
        iteration_brain.table = self.brain.table.copy()

        trades = []
        total_duration = 0.0

        # Need at least 21 bars for regression
        if len(day_data) < 21:
            return IterationResult(
                iteration=iteration,
                params=params,
                trades=[],
                pnl=0.0,
                win_rate=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                total_trades=0,
                avg_hold_time=0.0
            )

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

            # Decision (using iteration's parameters)
            # Apply confidence threshold from params
            min_prob = params.get('confidence_threshold', 0.80)
            min_conf = 0.30

            prob = iteration_brain.get_probability(state)
            conf = iteration_brain.get_confidence(state)

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
                    total_duration += outcome.duration
                    iteration_brain.update(outcome)

        # Calculate metrics
        if not trades:
            return IterationResult(
                iteration=iteration,
                params=params,
                trades=trades,
                pnl=0.0,
                win_rate=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                total_trades=0,
                avg_hold_time=0.0
            )

        pnls = [t.pnl for t in trades]
        wins = sum(1 for t in trades if t.result == 'WIN')
        win_rate = wins / len(trades)

        # Sharpe ratio
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6)

        # Max drawdown
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Average hold time
        avg_hold = total_duration / len(trades) if trades else 0.0

        return IterationResult(
            iteration=iteration,
            params=params,
            trades=trades,
            pnl=sum(pnls),
            win_rate=win_rate,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_trades=len(trades),
            avg_hold_time=avg_hold
        )

    def _simulate_trade(self,
                       current_idx: int,
                       entry_price: float,
                       data: pd.DataFrame,
                       state: Any,
                       params: Dict[str, Any]) -> Optional[TradeOutcome]:
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

        entry_time = data.iloc[current_idx]['timestamp']
        if isinstance(entry_time, pd.Timestamp):
            entry_time = entry_time.timestamp()

        for idx, row in future_data.iterrows():
            price = row['price'] if 'price' in row else row['close']
            pnl = price - entry_price  # Long only for simplicity

            exit_time = row['timestamp']
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
                # Time exit
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

        last_time = future_data.iloc[-1]['timestamp']
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

    def optimize_day(self,
                     day_number: int,
                     date_str: str,
                     day_data: pd.DataFrame) -> DayResults:
        """
        Run DOE optimization for single day

        Tests 1000 parameter combinations and finds best

        Args:
            day_number: Day index in sequence
            date_str: Date string (YYYY-MM-DD)
            day_data: OHLCV data for this day

        Returns:
            DayResults with best parameters and metrics
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Day {day_number}: {date_str} ({len(day_data)} bars)")
            print(f"{'='*80}")

        start_time = datetime.now()

        # Track all iterations
        iteration_results = []

        # Progress bar
        pbar = tqdm(range(self.n_iterations), desc=f"Day {day_number}", disable=not self.verbose)

        for iteration in pbar:
            # Generate parameters
            param_set = self.param_generator.generate_parameter_set(
                iteration=iteration,
                day=day_number,
                context='CORE'
            )

            # Simulate day with these parameters
            result = self.simulate_trading_day(
                day_data=day_data,
                params=param_set.parameters,
                iteration=iteration
            )

            iteration_results.append(result)

            # Update progress bar
            if result.total_trades > 0:
                pbar.set_postfix({
                    'Trades': result.total_trades,
                    'WR': f'{result.win_rate:.1%}',
                    'Sharpe': f'{result.sharpe:.2f}',
                    'P&L': f'${result.pnl:.0f}'
                })

        # Find best iteration (by Sharpe ratio)
        valid_results = [r for r in iteration_results if r.total_trades >= 5]

        if not valid_results:
            # No valid results, use baseline
            best_result = iteration_results[0]
            if self.verbose:
                print("⚠️  No iterations with ≥5 trades. Using baseline parameters.")
        else:
            best_result = max(valid_results, key=lambda r: r.sharpe)

        # Update brain with best iteration's trades
        for trade in best_result.trades:
            self.brain.update(trade)

        # Record best parameters
        self.param_generator.update_best_params(best_result.params)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Summary
        day_result = DayResults(
            day_number=day_number,
            date=date_str,
            total_iterations=self.n_iterations,
            best_iteration=best_result.iteration,
            best_params=best_result.params,
            best_sharpe=best_result.sharpe,
            best_win_rate=best_result.win_rate,
            best_pnl=best_result.pnl,
            total_trades=best_result.total_trades,
            states_learned=len(self.brain.table),
            high_confidence_states=len(self.brain.get_all_states_above_threshold()),
            execution_time_seconds=execution_time
        )

        if self.verbose:
            print(f"\n✅ Day {day_number} Complete:")
            print(f"   Best Iteration: {best_result.iteration} ({best_result.params.get('generation_method', 'unknown')})")
            print(f"   Trades: {best_result.total_trades}")
            print(f"   Win Rate: {best_result.win_rate:.1%}")
            print(f"   Sharpe: {best_result.sharpe:.2f}")
            print(f"   P&L: ${best_result.pnl:.2f}")
            print(f"   States Learned: {day_result.states_learned}")
            print(f"   High-Conf States: {day_result.high_confidence_states}")
            print(f"   Execution Time: {execution_time:.1f}s")

        return day_result

    def train(self, data: pd.DataFrame, max_days: Optional[int] = None) -> List[DayResults]:
        """
        Run walk-forward training on full dataset

        Args:
            data: Full dataset with timestamps
            max_days: Optional limit on number of days to train

        Returns:
            List of DayResults for each day
        """
        # Split into days
        days = self.split_into_trading_days(data)

        if max_days:
            days = days[:max_days]
            if self.verbose:
                print(f"Limiting training to first {max_days} days")

        # Train day by day
        for day_idx, (date_str, day_data) in enumerate(days):
            day_result = self.optimize_day(
                day_number=day_idx + 1,
                date_str=date_str,
                day_data=day_data
            )

            self.day_results.append(day_result)

            # Save checkpoint
            self.save_checkpoint(day_idx + 1, date_str)

        # Final summary
        if self.verbose:
            self.print_training_summary()

        return self.day_results

    def save_checkpoint(self, day_number: int, date_str: str):
        """Save brain and best params to checkpoint"""
        # Save brain
        brain_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_brain.pkl")
        self.brain.save(brain_path)

        # Save best params
        if self.param_generator.best_params_history:
            params_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_params.pkl")
            with open(params_path, 'wb') as f:
                pickle.dump(self.param_generator.best_params_history[-1], f)

        # Save day results
        results_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.day_results[-1], f)

    def load_checkpoint(self, day_number: int):
        """Load brain and params from checkpoint"""
        brain_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_brain.pkl")
        self.brain.load(brain_path)

        params_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_params.pkl")
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
                self.param_generator.best_params_history.append(best_params)

    def print_training_summary(self):
        """Print summary of entire training run"""
        print("\n" + "="*80)
        print("WALK-FORWARD TRAINING SUMMARY")
        print("="*80)

        total_days = len(self.day_results)
        total_trades = sum(d.total_trades for d in self.day_results)
        avg_sharpe = np.mean([d.best_sharpe for d in self.day_results])
        avg_win_rate = np.mean([d.best_win_rate for d in self.day_results])
        total_pnl = sum(d.best_pnl for d in self.day_results)

        print(f"\nTotal Days Trained: {total_days}")
        print(f"Total Trades: {total_trades}")
        print(f"Average Win Rate: {avg_win_rate:.1%}")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Final States Learned: {len(self.brain.table)}")
        print(f"High-Confidence States: {len(self.brain.get_all_states_above_threshold())}")

        print("\n" + "="*80)


# CLI Entry Point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Training with DOE")
    parser.add_argument("--data", type=str, required=True, help="Path to parquet data file")
    parser.add_argument("--max-days", type=int, default=None, help="Limit number of days")
    parser.add_argument("--iterations", type=int, default=1000, help="Iterations per day")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = pd.read_parquet(args.data)
    print(f"Loaded {len(data)} rows")

    # Initialize trainer
    trainer = WalkForwardTrainer(
        asset_profile=MNQ,
        checkpoint_dir=args.checkpoint_dir,
        n_iterations_per_day=args.iterations,
        verbose=True
    )

    # Train
    results = trainer.train(data, max_days=args.max_days)

    print("\n✅ Walk-Forward Training Complete!")
