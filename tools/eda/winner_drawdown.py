import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add trade_outcome_suite to path to load excursions
repo_root = Path(__file__).resolve().parents[2]
suite_path = repo_root / 'tools' / 'suites' / 'trade_outcome_suite' / 'trade_outcome_suite'
sys.path.insert(0, str(suite_path))

import excursions as ex

def main():
    print("Loading data...")
    IS = ex.load('IS', source='causal_flat')
    
    winners = IS[IS['pnl_usd'] > 0]
    
    print("\n" + "="*50)
    print("DRAWDOWN (MAE) ANALYSIS OF GOOD TRADES (IS Data)")
    print("="*50)
    
    print(f"\nSample: {len(winners)} Winning Trades")
    
    print("\n--- 1. Do winners ever draw down? (MAE of Winners) ---")
    print(f"Winners median MAE (drawdown): -${winners['mae_usd'].median():.0f}")
    
    w_mae = winners['mae_usd']
    print("\nPercentage of Winning Trades that suffered at least -$X drawdown before closing in profit:")
    for x in [10, 25, 50, 75, 100, 150]:
        pct = (w_mae >= x).mean() * 100
        print(f"  Suffered -${x}: {pct:.1f}%")
        
    print("\n--- 2. When does the drawdown happen? ---")
    print(f"Winners hit their absolute worst point (MAE bottom) at minute: {winners['t_to_bottom_min'].median():.1f} on average.")
    print(f"Because winners live for {winners['dur_min'].median():.1f} minutes, the bottom happens at ~{winners['frac_to_bottom'].median()*100:.0f}% into their life.")
    
    print("\nConclusion: Do winners experience heat?")

if __name__ == '__main__':
    main()
