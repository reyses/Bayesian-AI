"""
Bayesian-AI - Training Orchestrator
File: bayesian_ai/training/orchestrator.py
"""
import pandas as pd
import os
import sys

# Add project root to sys.path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_core import BayesianEngine
from config.symbols import SYMBOL_MAP
from training.databento_loader import DatabentoLoader

def get_data_source(data_path: str) -> pd.DataFrame:
    """Loads data from a file path, supporting .dbn and .parquet files."""
    if data_path.endswith('.dbn'):
        return DatabentoLoader.load_data(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")

class TrainingOrchestrator:
    """Runs 1000 iterations on historical data to build the Bayesian prior"""
    def __init__(self, asset_ticker: str, data: pd.DataFrame = None, use_gpu: bool = True):
        self.data = data
        self.asset = SYMBOL_MAP[asset_ticker]
        self.engine = BayesianEngine(self.asset, use_gpu=use_gpu)
        self.model_path = 'probability_table.pkl'

        # Helper variables for test introspection
        self.kill_zones = [21500, 21600, 21700] # Default
        self.raw_data = self.data # Alias for test

    def load_historical_data(self, data: pd.DataFrame):
        """Load data if not loaded in init (helper for tests)"""
        self.data = data
        self.raw_data = self.data

    def run_training(self, iterations=1000):
        if self.data is None:
            raise ValueError("Data not loaded")

        print(f"[TRAINING] Data: {len(self.data)} ticks. Target: {iterations} iterations.")

        # Build Static Context (L1-L4)
        self.engine.initialize_session(self.data, self.kill_zones)

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

if __name__ == "__main__":
    # Example usage
    try:
        data = get_data_source("./data/nq_2025_full_year.parquet")
        orchestrator = TrainingOrchestrator(
            data=data,
            asset_ticker="MNQ" # Changed to MNQ as NQ might not be in SYMBOL_MAP or requires funds
        )
        orchestrator.run_training(iterations=1000)
    except Exception as e:
        print(f"Skipping execution: {e}")
