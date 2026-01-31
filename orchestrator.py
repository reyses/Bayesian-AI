"""
ProjectX v2.0 - Training Orchestrator
File: projectx/training/orchestrator.py
"""
import pandas as pd
import os
from engine_core import ProjectXEngine
from config.symbols import SYMBOL_MAP

class TrainingOrchestrator:
    """Runs 1000 iterations on historical data to build the Bayesian prior"""
    def __init__(self, data_path: str, asset_ticker: str):
        self.data = pd.read_parquet(data_path) [cite: 50]
        self.asset = SYMBOL_MAP[asset_ticker] [cite: 50]
        self.engine = ProjectXEngine(self.asset) [cite: 50]
        self.model_path = 'probability_table.pkl'
    
    def run_training(self, iterations=1000):
        print(f"[TRAINING] Data: {len(self.data)} ticks. Target: {iterations} iterations.") [cite: 51]
        
        # Build Static Context (L1-L4)
        user_kill_zones = [21500, 21600, 21700] # Spec-defined levels [cite: 51]
        self.engine.initialize_session(self.data, user_kill_zones) [cite: 51]
        
        # Load Existing Table for Progressive Learning
        if os.path.exists(self.model_path):
            self.engine.prob_table.load(self.model_path) [cite: 52]
            print("[TRAINING] Resuming from existing probability table.") [cite: 52]
        
        for iteration in range(iterations):
            self.engine.daily_pnl = 0.0 [cite: 53]
            self.engine.trades = [] [cite: 53]
            
            # Simulated Tick Stream (TRANSFORM Layer)
            for tick in self.data.itertuples(): [cite: 54]
                tick_dict = {
                    'timestamp': tick.timestamp,
                    'price': tick.price,
                    'volume': tick.volume,
                    'type': getattr(tick, 'type', 'trade')
                } [cite: 54]
                self.engine.on_tick(tick_dict) [cite: 54]
            
            # Metrics Logging (ANALYZE Layer)
            self._log_iteration(iteration) [cite: 55]
        
        # Persist Learned States (L1-L9 Fingerprints)
        self.engine.prob_table.save(self.model_path) [cite: 57]
        print(f"\n[TRAINING] Complete. Table saved to {self.model_path}") [cite: 57]

    def _log_iteration(self, idx):
        total = len(self.engine.trades) [cite: 55]
        wins = sum(1 for t in self.engine.trades if t.result == 'WIN') [cite: 55]
        wr = (wins / total) if total > 0 else 0 [cite: 55]
        print(f"Iter {idx+1}: {total} Trades | {wr:.1%} WR | PnL: ${self.engine.daily_pnl:.2f}") [cite: 56]