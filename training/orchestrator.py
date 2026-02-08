"""
Adaptive Learning Training Orchestrator - ENHANCED WITH PROGRESS TRACKING
Integrates all components for end-to-end learning with real-time progress display
"""
import os
import sys
import glob
import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import pickle

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.quantum_field_engine import QuantumFieldEngine
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.adaptive_confidence import AdaptiveConfidenceManager
from core.three_body_state import ThreeBodyQuantumState
from training.databento_loader import DatabentoLoader

# Progress Display Helper
class TrainingProgressBar:
    """Manages nested progress bars for phases and iterations"""
    
    def __init__(self, total_iterations: int, phase_name: str = "Training"):
        self.total_iterations = total_iterations
        self.phase_name = phase_name
        self.start_time = time.time()
        
        # Main progress bar
        self.pbar_main = tqdm(
            total=total_iterations,
            desc=f"[{phase_name}]",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Metrics display (position 1, below main bar)
        self.pbar_metrics = tqdm(
            total=0,
            bar_format='{desc}',
            position=1,
            leave=False
        )
        
        self.current_metrics = {
            'trades': 0,
            'states': 0,
            'pnl': 0.0,
            'win_rate': 0.0,
            'avg_duration': 0.0
        }
    
    def update(self, n=1, **metrics):
        """Update progress and metrics"""
        self.pbar_main.update(n)
        
        # Update stored metrics
        for key, value in metrics.items():
            self.current_metrics[key] = value
        
        # Format metrics display
        trades = self.current_metrics.get('trades', 0)
        states = self.current_metrics.get('states', 0)
        pnl = self.current_metrics.get('pnl', 0.0)
        wr = self.current_metrics.get('win_rate', 0.0)
        avg_dur = self.current_metrics.get('avg_duration', 0.0)
        
        metrics_str = f"Trades: {trades:>6} | States: {states:>5} | P&L: ${pnl:>8,.2f} | WR: {wr:>5.1%} | AvgDur: {avg_dur:.1f}s"
        self.pbar_metrics.set_description_str(metrics_str)
    
    def set_phase(self, phase_name: str, day: int = None, total_days: int = None):
        """Update phase information"""
        if day and total_days:
            desc = f"[{phase_name}] Day {day}/{total_days}"
        else:
            desc = f"[{phase_name}]"
        self.pbar_main.set_description(desc)
    
    def close(self):
        """Close all progress bars"""
        self.pbar_main.close()
        self.pbar_metrics.close()


def get_data_source(filepath: str) -> pd.DataFrame:
    """
    Loads data from .dbn (via DatabentoLoader) or .parquet files.
    """
    if filepath.endswith('.dbn') or filepath.endswith('.dbn.zst'):
        return DatabentoLoader.load_data(filepath)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def load_data_from_directory(data_dir: str) -> List[str]:
    """
    Finds .dbn and .parquet files in directory.
    """
    files = glob.glob(os.path.join(data_dir, "**", "*.dbn"), recursive=True)
    files.extend(glob.glob(os.path.join(data_dir, "**", "*.dbn.zst"), recursive=True))
    files.extend(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    return sorted(files)

class TrainingOrchestrator:
    """
    Orchestrates the training process using the Quantum System logic.
    Supports both legacy API (for tests) and new CLI usage.
    """
    
    def __init__(self, asset_ticker: str = "MNQ", data: pd.DataFrame = None,
                 use_gpu: bool = False, output_dir: str = '.',
                 verbose: bool = False, debug_file: str = None, 
                 mode: str = "QUANTUM"):
        self.asset_ticker = asset_ticker
        self.data = data
        self.use_gpu = use_gpu # Legacy flag, currently unused in new logic
        self.output_dir = output_dir
        self.verbose = verbose
        self.mode = mode
        self.debug_file = debug_file

        # Initialize Core Components
        self.brain = QuantumBayesianBrain()
        self.manager = AdaptiveConfidenceManager(self.brain)
        self.engine = QuantumFieldEngine() # Default regression period 21

        # Internal State
        self.trades: List[TradeOutcome] = []
        self.daily_pnl = 0.0
        self.start_time = None

        # For legacy compatibility
        self.kill_zones = [21500, 21550] # Example kill zones
        self.asset = type('Asset', (), {'ticker': asset_ticker})
        self.raw_data = data # Legacy attribute

    def _get_win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.result == 'WIN')
        return wins / len(self.trades)

    def _get_avg_confidence(self) -> float:
        """Calculate average confidence of all learned states"""
        if not self.brain.table:
            return 0.0
        total_conf = sum(self.brain.get_confidence(s) for s in self.brain.table)
        return total_conf / len(self.brain.table)

    def run_training(self, iterations: int = 1000, params: Dict[str, Any] = None,
                    on_progress=None):
        """
        Main training loop.
        Iterates through the data bar-by-bar, computing state and simulating trades.
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in self.data.columns:
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)

        self.start_time = time.time()

        # Initialize progress tracking
        progress = TrainingProgressBar(
            total_iterations=iterations,
            phase_name=f"{self.mode} MODE"
        )
        
        total_pnl = 0.0
        
        # Dashboard JSON path
        json_path = os.path.join(os.path.dirname(__file__), 'training_progress.json')

        try:
            # We need enough history for regression (21 bars)
            start_idx = 21
            # Limit iterations if specified, else run through data
            max_idx = min(len(self.data) - 1, start_idx + iterations)

            # Simple simulation loop
            for i in range(start_idx, max_idx):
                current_row = self.data.iloc[i]

                # Update macro view (simple sliding window for demo/test)
                # In real system, this would be proper resampling
                df_macro = self.data.iloc[i-21:i+1].copy()
                df_macro['close'] = df_macro['price'] if 'price' in df_macro.columns else df_macro['close']

                # Calculate State
                state = self.engine.calculate_three_body_state(
                    df_macro=df_macro,
                    df_micro=df_macro, # Using same data for micro for simplicity in this loop
                    current_price=current_row['price'] if 'price' in current_row else current_row['close'],
                    current_volume=current_row['volume'] if 'volume' in current_row else 0,
                    tick_velocity=0.0 # Velocity calc requires prev tick
                )

                # Decision
                decision = self.manager.should_fire(state)

                if decision['should_fire']:
                    # Simulate Trade
                    entry_price = current_row['price'] if 'price' in current_row else current_row['close']
                    outcome = self._simulate_trade_with_state(i, entry_price, state)

                    if outcome:
                        self.trades.append(outcome)
                        self.brain.update(outcome)
                        self.manager.record_trade(outcome)
                        total_pnl += outcome.pnl

                # Update progress
                progress_val = i - start_idx
                
                avg_duration = 0.0
                if self.trades:
                    avg_duration = sum(t.duration for t in self.trades) / len(self.trades)

                progress.update(
                    n=1,
                    trades=len(self.trades),
                    states=len(self.brain.table),
                    pnl=total_pnl,
                    win_rate=self._get_win_rate(),
                    avg_duration=avg_duration
                )
                
                # Periodic updates
                if progress_val % 10 == 0:
                    self._update_dashboard_json(json_path, progress_val, iterations, total_pnl, i)

                    if on_progress:
                        metrics = {
                            'iteration': progress_val,
                            'total_iterations': iterations,
                            'pnl': total_pnl,
                            'win_rate': self._get_win_rate(),
                            'average_confidence': self._get_avg_confidence(),
                            'avg_duration': avg_duration
                        }
                        on_progress(metrics)
        
        finally:
            progress.close()
            # Save final model
            self.save_model(os.path.join(self.output_dir, "quantum_probability_table.pkl"))
        
        return {
            'total_trades': len(self.trades),
            'pnl': total_pnl,
            'win_rate': self._get_win_rate(),
            'unique_states': len(self.brain.table)
        }

    def _simulate_trade(self, current_idx: int, entry_price: float) -> Optional[TradeOutcome]:
        """
        Simulate a trade by looking ahead in data.
        Simple logic: 20 ticks TP, 10 ticks SL.
        """
        # Look ahead up to 100 ticks
        max_lookahead = 100
        end_idx = min(len(self.data), current_idx + max_lookahead)

        future_data = self.data.iloc[current_idx+1:end_idx]
        if future_data.empty:
            return None

        # Get Entry Time
        entry_time = self.data.iloc[current_idx]['timestamp'] if 'timestamp' in self.data.columns else 0

        # Simple random-ish or trend logic for test
        # Let's use a fixed TP/SL for now
        tp = 20.0
        sl = 10.0

        for idx, row in future_data.iterrows():
            price = row['price'] if 'price' in row else row['close']
            pnl = price - entry_price # Long only for simplicity

            exit_time = row['timestamp'] if 'timestamp' in row else 0
            duration = exit_time - entry_time

            if pnl >= tp:
                return TradeOutcome(
                    state=ThreeBodyQuantumState.null_state(),
                    entry_price=entry_price,
                    exit_price=price,
                    pnl=tp,
                    result='WIN',
                    timestamp=exit_time,
                    exit_reason='TP',
                    entry_time=entry_time,
                    duration=duration
                )
            elif pnl <= -sl:
                return TradeOutcome(
                    state=ThreeBodyQuantumState.null_state(),
                    entry_price=entry_price,
                    exit_price=price,
                    pnl=-sl,
                    result='LOSS',
                    timestamp=exit_time,
                    exit_reason='SL',
                    entry_time=entry_time,
                    duration=duration
                )

        # Time exit
        last_price = future_data.iloc[-1]['price'] if 'price' in future_data.iloc[-1] else future_data.iloc[-1]['close']
        pnl = last_price - entry_price

        exit_time = future_data.iloc[-1]['timestamp'] if 'timestamp' in future_data.iloc[-1] else 0
        duration = exit_time - entry_time

        return TradeOutcome(
            state=ThreeBodyQuantumState.null_state(),
            entry_price=entry_price,
            exit_price=last_price,
            pnl=pnl,
            result='WIN' if pnl > 0 else 'LOSS',
            timestamp=exit_time,
            exit_reason='TIME',
            entry_time=entry_time,
            duration=duration
        )

    def _simulate_trade_with_state(self, current_idx: int, entry_price: float, state: ThreeBodyQuantumState) -> Optional[TradeOutcome]:
        outcome = self._simulate_trade(current_idx, entry_price)
        if outcome:
            outcome.state = state
        return outcome

    def _update_dashboard_json(self, json_path: str, iteration: int, total_iterations: int, pnl: float, current_data_idx: int = 0):
        """Write progress to JSON for dashboard"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Get recent candles for chart
        recent_candles = []
        if self.data is not None and current_data_idx > 0:
            start = max(0, current_data_idx - 50)
            subset = self.data.iloc[start:current_data_idx+1]
            # Convert to list of dicts
            for _, row in subset.iterrows():
                candle = {
                    'timestamp': str(row['timestamp']) if 'timestamp' in row else '', # Convert to string for JSON
                    'close': float(row['price']) if 'price' in row else float(row['close']),
                    # Add open/high/low/volume if available, else simulate
                    'open': float(row['open']) if 'open' in row else float(row['price']),
                    'high': float(row['high']) if 'high' in row else float(row['price']),
                    'low': float(row['low']) if 'low' in row else float(row['price']),
                    'volume': float(row['volume']) if 'volume' in row else 0
                }
                recent_candles.append(candle)

        data = {
            'iteration': iteration,
            'total_iterations': total_iterations,
            'elapsed_seconds': elapsed,
            'states_learned': len(self.brain.table),
            'high_confidence_states': len(self.brain.get_all_states_above_threshold()),
            'trades': [
                {'pnl': t.pnl, 'result': t.result, 'duration': t.duration} for t in self.trades[-50:] # Last 50 trades
            ],
            'recent_candles': recent_candles
        }
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass # Ignore write errors (race conditions)

    def _run_single_iteration(self, iteration_num: int):
        """
        Legacy adapter for tests/test_phase2.py.
        Runs logic for a single 'tick' or step.
        """
        if self.data is None:
             return {'total_trades': 0, 'pnl': 0.0, 'unique_states': 0}

        # Determine index
        idx = iteration_num + 21 # Offset for regression
        if idx >= len(self.data):
            idx = len(self.data) - 1

        current_row = self.data.iloc[idx]
        df_macro = self.data.iloc[max(0, idx-21):idx+1].copy()
        if 'price' in df_macro.columns and 'close' not in df_macro.columns:
            df_macro['close'] = df_macro['price']

        state = self.engine.calculate_three_body_state(
            df_macro=df_macro,
            df_micro=df_macro,
            current_price=current_row['price'] if 'price' in current_row else current_row['close'],
            current_volume=current_row['volume'] if 'volume' in current_row else 0,
            tick_velocity=0.0
        )

        decision = self.manager.should_fire(state)
        if decision['should_fire']:
            entry_price = current_row['price'] if 'price' in current_row else current_row['close']
            outcome = self._simulate_trade_with_state(idx, entry_price, state)
            if outcome:
                self.trades.append(outcome)
                self.brain.update(outcome)
                self.manager.record_trade(outcome)

        return {
            'total_trades': len(self.trades),
            'pnl': sum(t.pnl for t in self.trades),
            'unique_states': len(self.brain.table)
        }

    def save_model(self, filepath: str):
        # Create directory if not exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.brain.save(filepath)

# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian-AI Training Orchestrator")
    parser.add_argument("--data-file", type=str, required=True, help="Path to .dbn or .parquet file")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--output", type=str, default="models/", help="Output directory")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU (Legacy)")

    args = parser.parse_args()

    # Load Data
    try:
        print(f"Loading data from {args.data_file}...")
        df = get_data_source(args.data_file)
        print(f"Loaded {len(df)} rows.")

        # Run Training
        orchestrator = TrainingOrchestrator(
            asset_ticker="MNQ",
            data=df,
            output_dir=args.output,
            mode="QUANTUM"
        )

        orchestrator.run_training(iterations=args.iterations)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
