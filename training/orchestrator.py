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
from typing import Dict, List, Any
from tqdm import tqdm

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
            'win_rate': 0.0
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
        
        metrics_str = f"Trades: {trades:>6} | States: {states:>5} | P&L: ${pnl:>8,.2f} | WR: {wr:>5.1%}"
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


def train_with_progress(historical_data_path: str, max_iterations: int = 1000, output_dir: str = 'models/'):
    """
    Training with real-time progress display
    
    PHASE 0: Unconstrained exploration (500 trades)
    PHASE 1-4: Adaptive confidence (progressive tightening)
    """
    
    print("\n" + "="*80)
    print("BAYESIAN-AI TRAINING ENGINE")
    print("="*80)
    print(f"Data: {historical_data_path}")
    print(f"Iterations: {max_iterations}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Initialize progress tracking
    progress = TrainingProgressBar(total_iterations=max_iterations, phase_name="PHASE 0: EXPLORATION")
    
    # Load data (simplified for example)
    print("Loading data...")
    # Your data loading logic here
    
    # Example training loop with progress
    exploration_trades = []
    
    for iteration in range(max_iterations):
        # Simulate training iteration
        time.sleep(0.01)  # Remove this - just for demo
        
        # Simulate metrics update
        trades_count = iteration * 5
        states_count = iteration * 2
        pnl = np.random.randn() * 100
        win_rate = 0.5 + (iteration / max_iterations) * 0.3
        
        # Update progress with metrics
        progress.update(
            n=1,
            trades=trades_count,
            states=states_count,
            pnl=pnl,
            win_rate=win_rate
        )
        
        # Phase transition at 500 iterations
        if iteration == 500:
            progress.set_phase("PHASE 1: ADAPTIVE LEARNING")
            print("\n\n" + "="*80)
            print("PHASE TRANSITION: Starting Adaptive Confidence Learning")
            print("="*80 + "\n")
    
    progress.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Iterations: {max_iterations}")
    print(f"Final Metrics:")
    print(f"  Trades: {trades_count}")
    print(f"  States: {states_count}")
    print(f"  P&L: ${pnl:,.2f}")
    print(f"  Win Rate: {win_rate:.1%}")
    print("="*80 + "\n")


# Integration example for your TrainingOrchestrator class
class EnhancedTrainingOrchestrator:
    """
    Drop-in replacement for your existing TrainingOrchestrator
    with progress tracking
    """
    
    def __init__(self, asset_ticker: str, data: pd.DataFrame = None, 
                 use_gpu: bool = True, output_dir: str = '.', 
                 verbose: bool = False, debug_file: str = None, 
                 mode: str = "LEGACY"):
        self.data = data
        self.output_dir = output_dir
        self.verbose = verbose
        self.mode = mode
        # ... rest of your init code
    
    def run_training(self, iterations=1000, params: Dict[str, Any] = None, 
                    on_progress=None):
        """
        Enhanced run_training with progress bars
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Initialize progress tracking
        progress = TrainingProgressBar(
            total_iterations=iterations,
            phase_name=f"{self.mode} MODE"
        )
        
        total_trades = 0
        total_pnl = 0.0
        
        try:
            for iteration in range(iterations):
                # Your existing training logic here
                # ...
                
                # Update progress every iteration
                progress.update(
                    n=1,
                    trades=total_trades,
                    states=len(self.engine.prob_table.table) if hasattr(self, 'engine') else 0,
                    pnl=total_pnl,
                    win_rate=self._get_win_rate() if hasattr(self, '_get_win_rate') else 0.0
                )
                
                # Call custom progress callback if provided
                if on_progress and iteration % 10 == 0:
                    metrics = {
                        'iteration': iteration,
                        'total_iterations': iterations,
                        'total_trades': total_trades,
                        'pnl': total_pnl,
                        'win_rate': self._get_win_rate() if hasattr(self, '_get_win_rate') else 0.0,
                        'average_confidence': 0.5  # Your actual confidence calc
                    }
                    on_progress(metrics)
        
        finally:
            progress.close()
        
        return {
            'total_trades': total_trades,
            'pnl': total_pnl,
            'win_rate': self._get_win_rate() if hasattr(self, '_get_win_rate') else 0.0,
            'unique_states': 0  # Your actual count
        }


# Example usage
if __name__ == "__main__":
    # Simple test
    print("Testing progress display...\n")
    train_with_progress(
        historical_data_path="test.parquet",
        max_iterations=100,
        output_dir="models/"
    )
