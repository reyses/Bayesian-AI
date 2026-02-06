"""
Bayesian-AI - Training Orchestrator
File: bayesian_ai/training/orchestrator.py
"""
import pandas as pd
import os
import sys
import glob
import argparse
import itertools
import copy
import random
import json
import time
import datetime
import numpy as np
from typing import Dict, List, Any

# Add project root to sys.path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_core import BayesianEngine
from config.symbols import SYMBOL_MAP, MNQ
from config.settings import OPERATIONAL_MODE, RAW_DATA_PATH
from training.databento_loader import DatabentoLoader

def get_data_source(data_path: str) -> pd.DataFrame:
    """Loads data from a file path, supporting .dbn and .parquet files."""
    if data_path.endswith('.dbn') or data_path.endswith('.dbn.zst'):
        return DatabentoLoader.load_data(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        if 'price' not in df.columns and 'close' in df.columns:
             df['price'] = df['close']
        return df
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")

class TrainingOrchestrator:
    """Runs iterations on historical data to build the Bayesian prior"""
    def __init__(self, asset_ticker: str, data: pd.DataFrame = None, use_gpu: bool = True, output_dir: str = '.', verbose: bool = False, debug_file: str = None):
        self.data = data
        # Default to MNQ if ticker not found, or use provided ticker
        self.asset = SYMBOL_MAP.get(asset_ticker, MNQ)
        self.use_gpu = use_gpu
        self.output_dir = output_dir
        self.model_path = os.path.join(self.output_dir, 'probability_table.pkl')

        self.verbose = verbose
        self.debug_file = debug_file

        self.engine = BayesianEngine(self.asset, use_gpu=use_gpu, verbose=verbose, log_path=debug_file)

        # Helper variables for test introspection
        self.kill_zones = [21500, 21600, 21700] # Default
        self.raw_data = self.data # Alias for test

        # Create output directory if it doesn't exist
        if self.output_dir and self.output_dir != '.':
            os.makedirs(self.output_dir, exist_ok=True)

    def load_historical_data(self, data: pd.DataFrame):
        """Load data if not loaded in init (helper for tests)"""
        self.data = data
        self.raw_data = self.data

    def reset_engine(self):
        """Resets the engine to a fresh state."""
        self.engine = BayesianEngine(self.asset, use_gpu=self.use_gpu, verbose=self.verbose, log_path=self.debug_file)

    def run_training(self, iterations=1000, params: Dict[str, Any] = None, on_progress=None):
        if self.data is None:
            raise ValueError("Data not loaded")

        # Apply parameters if provided
        if params:
            if 'min_prob' in params:
                self.engine.MIN_PROB = params['min_prob']
            if 'min_conf' in params:
                self.engine.MIN_CONF = params['min_conf']
            if 'max_daily_loss' in params:
                self.engine.MAX_DAILY_LOSS = params['max_daily_loss']
            if 'kill_zones' in params:
                self.kill_zones = params['kill_zones']

        print(f"[TRAINING] Data: {len(self.data)} ticks. Target: {iterations} iterations.")

        # Prepare data for static context (needs DatetimeIndex)
        static_data = self.data.copy()
        if 'timestamp' in static_data.columns:
            static_data['timestamp'] = pd.to_datetime(static_data['timestamp'])
            static_data.set_index('timestamp', inplace=True)

        # Ensure OHLC data for LayerEngine
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(static_data.columns) and 'price' in static_data.columns:
            print("[TRAINING] Resampling tick data to 1s OHLC for Static Context...")
            ohlc = static_data['price'].resample('1s').ohlc()
            if 'volume' in static_data.columns:
                ohlc['volume'] = static_data['volume'].resample('1s').sum()
            else:
                raise ValueError("'volume' column is required for OHLC resampling but was not found.")
            static_data = ohlc.dropna()

        # Build Static Context (L1-L4)
        self.engine.initialize_session(static_data, self.kill_zones)

        # Load Existing Table for Progressive Learning
        # Note: If params are provided (e.g. for grid search), we typically want fresh learning,
        # but unless reset_engine() was called before, we might load.
        # For simplicity, if we are tuning, we assume the caller manages reset.
        if os.path.exists(self.model_path) and not params:
            self.engine.prob_table.load(self.model_path)
            print("[TRAINING] Resuming from existing probability table.")

        # Pre-compute tick data for performance (TRANSFORM Layer)
        # Ensure 'price' column exists (use 'close' if available for OHLC data)
        if 'price' not in self.data.columns and 'close' in self.data.columns:
            self.data['price'] = self.data['close']

        # Converting DataFrame to list of dicts once avoids repetitive overhead in the loop
        cols_to_use = ['timestamp', 'price', 'volume']

        # Ensure 'type' field exists and is string
        if 'type' in self.data.columns:
            self.data = self.data.copy()
            self.data['type'] = self.data['type'].fillna('trade').astype(str)
        else:
            self.data = self.data.copy()
            self.data['type'] = 'trade'

        cols_to_use.append('type')

        tick_records = self.data[cols_to_use].to_dict('records')

        # Pre-calculate 15m candles for dashboard (Optimization)
        self._prepare_dashboard_data()

        start_time = time.time()

        for iteration in range(iterations):
            self.engine.daily_pnl = 0.0
            self.engine.trades = []

            # Simulated Tick Stream
            for tick_dict in tick_records:
                self.engine.on_tick(tick_dict)

            # Metrics Logging (ANALYZE Layer)
            self._log_iteration(iteration)
            self._save_progress_json(iteration + 1, iterations, start_time)

            if on_progress:
                # Calculate metrics for callback
                avg_conf = 0.0
                high_conf = 0
                if hasattr(self.engine.prob_table, 'table'):
                    if len(self.engine.prob_table.table) > 0:
                        total_conf = sum(self.engine.prob_table.get_confidence(s) for s in self.engine.prob_table.table)
                        avg_conf = total_conf / len(self.engine.prob_table.table)

                    if hasattr(self.engine.prob_table, 'get_all_states_above_threshold'):
                         high_conf = len(self.engine.prob_table.get_all_states_above_threshold(0.8))

                metrics = {
                    'iteration': iteration + 1,
                    'total_iterations': iterations,
                    'pnl': self.engine.daily_pnl,
                    'win_rate': self._get_win_rate(),
                    'total_trades': len(self.engine.trades),
                    'average_confidence': avg_conf,
                    'high_confidence_states': high_conf
                }
                on_progress(metrics)

        # Persist Learned States (L1-L9 Fingerprints)
        # Only save if not in a temporary tuning run (params present often implies tuning)
        if not params:
            self.engine.prob_table.save(self.model_path)
            print(f"\n[TRAINING] Complete. Table saved to {self.model_path}")

        return {
            'total_trades': len(self.engine.trades), # Last iteration stats
            'pnl': self.engine.daily_pnl,
            'win_rate': self._get_win_rate()
        }

    def _get_win_rate(self):
        total = len(self.engine.trades)
        wins = sum(1 for t in self.engine.trades if t.result == 'WIN')
        return (wins / total) if total > 0 else 0.0

    def _log_iteration(self, idx):
        total = len(self.engine.trades)
        wins = sum(1 for t in self.engine.trades if t.result == 'WIN')
        wr = (wins / total) if total > 0 else 0
        print(f"Iter {idx+1}: {total} Trades | {wr:.1%} WR | PnL: ${self.engine.daily_pnl:.2f}")

    def _prepare_dashboard_data(self):
        """Pre-calculates 15m candles for the dashboard."""
        try:
            df = self.data.copy()
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            
            # Resample to 15m
            ohlc_15m = df['price'].resample('15min').ohlc()
            # Take last 100 bars
            self.candles_15m = ohlc_15m.tail(100).reset_index().to_dict('records')
            # Convert timestamps to string/iso
            for c in self.candles_15m:
                c['timestamp'] = c['timestamp'].isoformat()
        except Exception as e:
            print(f"[ORCHESTRATOR] Dashboard data prep failed: {e}")
            self.candles_15m = []

    def _save_progress_json(self, iteration, total_iterations, start_time):
        """Saves progress to training/training_progress.json"""
        try:
            # Summary stats
            summary = self.engine.prob_table.get_summary() if hasattr(self.engine.prob_table, 'get_summary') else {}
            states_learned = len(self.engine.prob_table.table) if hasattr(self.engine.prob_table, 'table') else 0
            
            # Count high confidence
            high_conf = 0
            if hasattr(self.engine.prob_table, 'get_all_states_above_threshold'):
                high_conf = len(self.engine.prob_table.get_all_states_above_threshold(0.8))

            # Format trades (last 100)
            trades_json = []
            for t in self.engine.trades[-100:]:
                trades_json.append({
                    "timestamp": datetime.datetime.fromtimestamp(t.timestamp).isoformat() if hasattr(t, 'timestamp') else "",
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "result": t.result,
                    "state_hash": str(hash(t.state)) if t.state else ""
                })

            # Current bar (last one from data)
            current_bar = {}
            if self.candles_15m:
                current_bar = self.candles_15m[-1]

            progress = {
                "iteration": iteration,
                "total_iterations": total_iterations,
                "elapsed_seconds": time.time() - start_time,
                "states_learned": states_learned,
                "high_confidence_states": high_conf,
                "trades": trades_json,
                "current_bar": current_bar,
                "recent_candles": self.candles_15m
            }
            
            output_path = os.path.join(self.output_dir, 'training_progress.json')
            # If output_dir is default or relative, make sure it maps to expected location for dashboard
            # Dashboard looks in 'training/training_progress.json'.
            # We should write to 'training/training_progress.json' explicitly OR rely on output_dir.
            # Requirement says "Read from: training/training_progress.json".
            # So I should write to 'training/training_progress.json'.
            
            progress_file = os.path.join(os.path.dirname(__file__), 'training_progress.json')
            
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                
        except Exception as e:
            # Don't crash training if dashboard update fails
            pass

    # Helper for test_phase2
    def _run_single_iteration(self, iteration_idx):
        if self.data is None:
             raise ValueError("Data not loaded")

        # Initialize session if not done
        if self.engine.fluid_engine.static_context is None:
             self.engine.initialize_session(self.data, self.kill_zones)

        self.engine.daily_pnl = 0.0
        self.engine.trades = []

        for tick in self.data.itertuples():
            tick_dict = {
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'type': getattr(tick, 'type', 'trade')
            }
            self.engine.on_tick(tick_dict)

        unique_states = len(self.engine.prob_table.table) if hasattr(self.engine.prob_table, 'table') else 0

        return {
            'total_trades': len(self.engine.trades),
            'pnl': self.engine.daily_pnl,
            'unique_states': unique_states
        }

    def run_grid_search(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Runs grid search over specified parameters.
        Args:
            param_grid: Dictionary where keys are parameter names and values are lists of values to try.
                        Supported keys: 'min_prob', 'min_conf', 'max_daily_loss', 'kill_zones'
        Returns:
            Dictionary with best parameters and results.
        """
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"[GRID SEARCH] Testing {len(combinations)} combinations...")

        results = []
        for i, params in enumerate(combinations):
            print(f"  [{i+1}/{len(combinations)}] Testing params: {params}")
            self.reset_engine()
            # Run for 1 iteration for grid search usually, or small number
            # We don't save model during grid search
            res = self.run_training(iterations=1, params=params)
            results.append({
                'params': params,
                'metrics': res
            })

        # Find best by PnL
        if not results:
            return {}

        best_result = max(results, key=lambda x: x['metrics']['pnl'])
        print("\n[GRID SEARCH] Best Result:")
        print(f"  Params: {best_result['params']}")
        print(f"  Metrics: {best_result['metrics']}")

        return best_result

    def run_walk_forward(self, train_window: int, test_window: int, step: int) -> List[Dict]:
        """
        Runs walk-forward validation.
        Args:
            train_window: Number of ticks for training.
            test_window: Number of ticks for testing.
            step: Number of ticks to move forward.
        """
        if self.data is None:
            raise ValueError("Data not loaded")

        total_ticks = len(self.data)
        if total_ticks < train_window + test_window:
            raise ValueError("Not enough data for walk-forward analysis")

        results = []
        start_idx = 0

        print(f"[WALK FORWARD] Starting analysis. Data length: {total_ticks}")

        # Save original data reference
        original_data = self.data

        try:
            while start_idx + train_window + test_window <= total_ticks:
                train_end = start_idx + train_window
                test_end = train_end + test_window

                # Slice data
                train_data = original_data.iloc[start_idx:train_end].copy()
                test_data = original_data.iloc[train_end:test_end].copy()

                print(f"  Window {len(results)+1}: Train [{start_idx}:{train_end}] | Test [{train_end}:{test_end}]")

                # Train
                self.reset_engine()
                self.data = train_data
                # Use params to indicate this is a temporary run (suppress saving)
                self.run_training(iterations=1, params={})

                # Test
                self.data = test_data
                test_res = self.run_training(iterations=1, params={'mode': 'walk_forward_test'})

                results.append({
                    'window_index': len(results),
                    'train_range': (start_idx, train_end),
                    'test_range': (train_end, test_end),
                    'test_metrics': test_res
                })

                start_idx += step

        finally:
            # Always restore data
            self.data = original_data

        return results

    def run_monte_carlo(self, iterations: int = 100, sample_fraction: float = 0.8) -> Dict[str, Any]:
        """
        Runs Monte Carlo simulation via random contiguous window sampling.
        Args:
            iterations: Number of simulations.
            sample_fraction: Fraction of data to sample (contiguous block).
        """
        if self.data is None:
            raise ValueError("Data not loaded")

        print(f"[MONTE CARLO] Running {iterations} simulations with sample fraction {sample_fraction}")

        results = []
        original_data = self.data

        try:
            n_samples = int(len(original_data) * sample_fraction)
            if n_samples < 100: # Minimum reasonable size
                 raise ValueError("Sample size too small for Monte Carlo")

            for i in range(iterations):
                # Random contiguous block to preserve temporal structure (L9 requirements)
                start_idx = random.randint(0, len(original_data) - n_samples)
                sample_data = original_data.iloc[start_idx : start_idx + n_samples].copy()

                # Train/Run
                self.reset_engine()
                self.data = sample_data
                res = self.run_training(iterations=1, params={'mode': 'monte_carlo'})
                results.append(res['pnl'])

                if (i+1) % 10 == 0:
                     print(f"  Sim {i+1}/{iterations}: PnL=${res['pnl']:.2f}")

        finally:
            self.data = original_data

        # Stats
        if not results:
            return {}

        pnls = np.array(results)
        stats = {
            'mean': np.mean(pnls),
            'std': np.std(pnls),
            'min': np.min(pnls),
            'max': np.max(pnls),
            'var_95': np.percentile(pnls, 5) # 5th percentile as simple VaR
        }

        print("\n[MONTE CARLO] Results:")
        print(f"  Mean PnL: ${stats['mean']:.2f}")
        print(f"  Std Dev: ${stats['std']:.2f}")
        print(f"  VaR (95%): ${stats['var_95']:.2f}")

        return stats

def load_data_from_directory(data_dir: str) -> pd.DataFrame:
    """Loads all supported files from a directory and concatenates them."""
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    files = []
    # Find .dbn.zst, .dbn, and .parquet files
    extensions = ['*.dbn.zst', '*.dbn', '*.parquet']
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not files:
        raise ValueError(f"No supported data files found in {data_dir}")

    print(f"Found {len(files)} data files. Loading...")
    dfs = []
    for f in sorted(files): # Sort to ensure chronological order if named appropriately
        print(f"  - Loading {os.path.basename(f)}...")
        try:
            df = get_data_source(f)
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Failed to load {f}: {e}")

    if not dfs:
        raise ValueError("Failed to load any data files.")

    print("Concatenating data...")
    full_df = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp to ensure correct playback
    if 'timestamp' in full_df.columns:
        full_df = full_df.sort_values('timestamp').reset_index(drop=True)

    return full_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian-AI Training Orchestrator")
    parser.add_argument("--data-dir", type=str, help="Directory containing .dbn.zst or .parquet files")
    parser.add_argument("--data-file", type=str, help="Single data file path (optional alternative to --data-dir)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--output", type=str, default="models/", help="Output directory for the model")
    parser.add_argument("--ticker", type=str, default="MNQ", help="Asset ticker symbol (default: MNQ)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) for data filtering")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) for data filtering")

    # New logging arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug-file", type=str, help="Path to debug log file")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")

    args = parser.parse_args()

    # Enforce OPERATIONAL_MODE
    if OPERATIONAL_MODE == "LEARNING" and not args.data_file:
        print(f"[ORCHESTRATOR] OPERATIONAL_MODE is '{OPERATIONAL_MODE}'. Enforcing ingestion from {RAW_DATA_PATH}.")
        args.data_dir = RAW_DATA_PATH

    try:
        data = None
        if args.data_dir:
            data = load_data_from_directory(args.data_dir)
        elif args.data_file:
            data = get_data_source(args.data_file)
        else:
            # Fallback for dev/testing if no args provided, or print help
            if len(sys.argv) == 1:
                parser.print_help()
                sys.exit(0)
            else:
                 raise ValueError("Must provide --data-dir or --data-file")

        # Filter data by date if requested
        if args.start_date or args.end_date:
            if 'timestamp' in data.columns:
                # Ensure datetime
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                     # Try to infer format or assume float seconds
                     try:
                         data['dt_temp'] = pd.to_datetime(data['timestamp'])
                     except:
                         data['dt_temp'] = pd.to_datetime(data['timestamp'], unit='s')
                else:
                    data['dt_temp'] = data['timestamp']

                if args.start_date:
                    print(f"[ORCHESTRATOR] Filtering start date: {args.start_date}")
                    data = data[data['dt_temp'] >= pd.to_datetime(args.start_date)]

                if args.end_date:
                    print(f"[ORCHESTRATOR] Filtering end date: {args.end_date}")
                    # Make end date inclusive for the day
                    end_dt = pd.to_datetime(args.end_date) + pd.Timedelta(days=1)
                    data = data[data['dt_temp'] < end_dt]

                # Cleanup
                data = data.drop(columns=['dt_temp'])
                print(f"[ORCHESTRATOR] Data rows after filtering: {len(data)}")

        # Auto-detect GPU capability for safe default
        if args.no_gpu:
            use_gpu = False
        else:
            try:
                from numba import cuda
                use_gpu = cuda.is_available()
            except:
                use_gpu = False

        if not use_gpu:
            print("[ORCHESTRATOR] CUDA not available. Running in CPU mode.")

        orchestrator = TrainingOrchestrator(
            asset_ticker=args.ticker,
            data=data,
            output_dir=args.output,
            use_gpu=use_gpu,
            verbose=args.verbose,
            debug_file=args.debug_file
        )
        orchestrator.run_training(iterations=args.iterations)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
