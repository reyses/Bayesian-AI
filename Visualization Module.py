"""
ProjectX v2.0 - Visualization Module
Plots equity curves and high-probability Bayesian states
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from core.bayesian_brain import BayesianBrain

def plot_training_results(model_path='probability_table.pkl'):
    # 1. LOAD
    brain = BayesianBrain()
    brain.load(model_path)
    
    if not brain.trade_history:
        print("Error: No trade history found for visualization.")
        return

    # 2. TRANSFORM (Equity Curve)
    pnl_data = [t.pnl for t in brain.trade_history]
    equity_curve = np.cumsum(pnl_data)
    
    # 3. TRANSFORM (High-Prob States)
    top_states = brain.get_all_states_above_threshold(min_prob=0.80)
    state_df = pd.DataFrame(top_states)
    
    # 4. VISUALIZE
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Equity Curve
    ax1.plot(equity_curve, color='#2ecc71', linewidth=2)
    ax1.fill_between(range(len(equity_curve)), equity_curve, alpha=0.2, color='#2ecc71')
    ax1.set_title('Cumulative P&L (Equity Curve)', fontsize=14)
    ax1.set_ylabel('USD ($)')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: State Win Probabilities
    if not state_df.empty:
        state_labels = [f"S_{i}" for i in range(len(state_df))]
        sns.barplot(x=state_labels, y='probability', data=state_df, ax=ax2, palette='viridis')
        ax2.axhline(0.80, color='red', linestyle='--', label='Alpha Threshold (80%)')
        ax2.set_title('Top 10 High-Probability States', fontsize=14)
        ax2.set_ylabel('Win Probability')
        ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_results()