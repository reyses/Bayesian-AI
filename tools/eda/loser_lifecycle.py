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
    losers = IS[IS['pnl_usd'] <= 0]
    
    print("\n" + "="*50)
    print("LIFECYCLE OF BAD TRADES vs GOOD TRADES (IS Data)")
    print("="*50)
    
    print(f"\nSample: {len(winners)} Winners vs {len(losers)} Losers")
    
    print("\n--- 1. How long do they live? (Duration) ---")
    print(f"Winners median life: {winners['dur_min'].median():.1f} minutes")
    print(f"Losers median life:  {losers['dur_min'].median():.1f} minutes")
    
    print("\n--- 2. How fast do losers go bad? ---")
    print(f"Losers reach their absolute worst point (MAE bottom) at minute: {losers['t_to_bottom_min'].median():.1f} on average.")
    print(f"Because losers live for {losers['dur_min'].median():.1f} minutes, they spend ~{losers['frac_to_bottom'].median()*100:.0f}% of their life bleeding out before hitting rock bottom.")
    
    print("\n--- 3. Do losers ever see profit before they die? (MFE of Losers) ---")
    print(f"Winners median peak (MFE): +${winners['mfe_usd'].median():.0f}")
    print(f"Losers median peak (MFE):  +${losers['mfe_usd'].median():.0f}")
    
    l_mfe = losers['mfe_usd']
    print("\nPercentage of Losing Trades that reached at least +$X in profit before dying:")
    for x in [10, 25, 50, 100, 150]:
        pct = (l_mfe >= x).mean() * 100
        print(f"  Reached +${x}: {pct:.1f}%")
        
    print("\nConclusion: The data clearly shows if bad trades are 'good entries that went bad' or 'bad right out of the gate'.")

if __name__ == '__main__':
    main()
