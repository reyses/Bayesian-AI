"""
Test Script - Progress Display Demo
Shows how the enhanced progress bars look during training
"""
import time
import numpy as np
from tqdm import tqdm

def demo_training_progress():
    """
    Simulates a training run to show progress display
    """
    
    print("\n" + "="*80)
    print("BAYESIAN-AI TRAINING DEMO")
    print("="*80)
    print("Simulating 100 iterations with progress tracking...")
    print("="*80 + "\n")
    
    # Phase 0: Exploration
    iterations_phase0 = 50
    pbar_phase0 = tqdm(
        total=iterations_phase0,
        desc="[PHASE 0: EXPLORATION]",
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    pbar_metrics = tqdm(
        total=0,
        bar_format='{desc}',
        position=1,
        leave=False
    )
    
    cumulative_pnl = 0.0
    total_trades = 0
    unique_states = 0
    
    for i in range(iterations_phase0):
        time.sleep(0.05)  # Simulate work
        
        # Simulate metrics
        trades_this_iter = np.random.randint(3, 8)
        total_trades += trades_this_iter
        unique_states += np.random.randint(0, 3)
        pnl_this_iter = np.random.randn() * 50
        cumulative_pnl += pnl_this_iter
        
        wins = int(total_trades * (0.45 + i/iterations_phase0 * 0.15))
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        # Update displays
        pbar_phase0.update(1)
        metrics_str = f"Trades: {total_trades:>4} | States: {unique_states:>3} | P&L: ${cumulative_pnl:>8,.2f} | WR: {win_rate:>5.1%}"
        pbar_metrics.set_description_str(metrics_str)
    
    pbar_phase0.close()
    pbar_metrics.close()
    
    print("\n" + "="*80)
    print("PHASE TRANSITION: EXPLORATION → ADAPTIVE LEARNING")
    print("="*80 + "\n")
    
    # Phase 1: Adaptive Learning
    iterations_phase1 = 50
    pbar_phase1 = tqdm(
        total=iterations_phase1,
        desc="[PHASE 1: ADAPTIVE]",
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    pbar_metrics2 = tqdm(
        total=0,
        bar_format='{desc}',
        position=1,
        leave=False
    )
    
    for i in range(iterations_phase1):
        time.sleep(0.05)
        
        trades_this_iter = np.random.randint(2, 5)  # Fewer trades (more selective)
        total_trades += trades_this_iter
        unique_states += np.random.randint(0, 2)
        pnl_this_iter = np.random.randn() * 70
        cumulative_pnl += pnl_this_iter
        
        wins = int(total_trades * (0.55 + i/iterations_phase1 * 0.15))
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        pbar_phase1.update(1)
        metrics_str = f"Trades: {total_trades:>4} | States: {unique_states:>3} | P&L: ${cumulative_pnl:>8,.2f} | WR: {win_rate:>5.1%}"
        pbar_metrics2.set_description_str(metrics_str)
    
    pbar_phase1.close()
    pbar_metrics2.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Iterations: 100")
    print(f"Total Trades: {total_trades}")
    print(f"Unique States: {unique_states}")
    print(f"Final P&L: ${cumulative_pnl:,.2f}")
    print(f"Win Rate: {win_rate:.1%}")
    print("="*80 + "\n")


def demo_nested_progress():
    """
    Shows nested progress bars (days within iterations)
    """
    print("\n" + "="*80)
    print("NESTED PROGRESS DEMO - Multi-Day Training")
    print("="*80 + "\n")
    
    days = 5
    iterations_per_day = 20
    
    pbar_days = tqdm(total=days, desc="Days", position=0, leave=True)
    
    for day in range(1, days + 1):
        pbar_days.set_description(f"Day {day}/{days}")
        
        pbar_iter = tqdm(
            total=iterations_per_day,
            desc=f"  Iterations",
            position=1,
            leave=False
        )
        
        for iteration in range(iterations_per_day):
            time.sleep(0.02)
            pbar_iter.update(1)
        
        pbar_iter.close()
        pbar_days.update(1)
    
    pbar_days.close()
    print("\n" + "="*80)
    print("Multi-day training complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🚀 PROGRESS DISPLAY DEMO\n")
    
    # Demo 1: Single phase with metrics
    demo_training_progress()
    
    time.sleep(1)
    
    # Demo 2: Nested progress
    demo_nested_progress()
    
    print("✅ Demo complete! This is how your training will look.\n")
