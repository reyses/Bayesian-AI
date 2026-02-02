"""
Bayesian-AI - Training Orchestrator
File: bayesian_ai/training/orchestrator.py
"""
import pandas as pd
import os
import sys
import glob
import argparse

# Add project root to sys.path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_core import BayesianEngine
from config.symbols import SYMBOL_MAP, MNQ
from training.databento_loader import DatabentoLoader

def get_data_source(data_path: str) -> pd.DataFrame:
    """Loads data from a file path, supporting .dbn and .parquet files."""
    if data_path.endswith('.dbn') or data_path.endswith('.dbn.zst'):
        return DatabentoLoader.load_data(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")

class TrainingOrchestrator:
    """Runs iterations on historical data to build the Bayesian prior"""
    def __init__(self, asset_ticker: str, data: pd.DataFrame = None, use_gpu: bool = True, output_dir: str = '.'):
        self.data = data
        # Default to MNQ if ticker not found, or use provided ticker
        self.asset = SYMBOL_MAP.get(asset_ticker, MNQ)
        self.engine = BayesianEngine(self.asset, use_gpu=use_gpu)
        self.output_dir = output_dir
        self.model_path = os.path.join(self.output_dir, 'probability_table.pkl')

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

    def run_training(self, iterations=1000):
        if self.data is None:
            raise ValueError("Data not loaded")

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
        if os.path.exists(self.model_path):
            self.engine.prob_table.load(self.model_path)
            print("[TRAINING] Resuming from existing probability table.")

        for iteration in range(iterations):
            self.engine.daily_pnl = 0.0
            self.engine.trades = []

            # Simulated Tick Stream (TRANSFORM Layer)
            for tick in self.data.itertuples():
                tick_dict = {
                    'timestamp': tick.timestamp,
                    'price': tick.price,
                    'volume': tick.volume,
                    'type': getattr(tick, 'type', 'trade')
                }
                self.engine.on_tick(tick_dict)

            # Metrics Logging (ANALYZE Layer)
            self._log_iteration(iteration)

        # Persist Learned States (L1-L9 Fingerprints)
        self.engine.prob_table.save(self.model_path)
        print(f"\n[TRAINING] Complete. Table saved to {self.model_path}")

    def _log_iteration(self, idx):
        total = len(self.engine.trades)
        wins = sum(1 for t in self.engine.trades if t.result == 'WIN')
        wr = (wins / total) if total > 0 else 0
        print(f"Iter {idx+1}: {total} Trades | {wr:.1%} WR | PnL: ${self.engine.daily_pnl:.2f}")

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

    args = parser.parse_args()

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

        orchestrator = TrainingOrchestrator(
            asset_ticker=args.ticker,
            data=data,
            output_dir=args.output
        )
        orchestrator.run_training(iterations=args.iterations)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
