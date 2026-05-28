import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[3]

def main():
    csv_path = REPO / 'reports/findings/exit_simulation_oos.csv'
    if not csv_path.exists():
        print(f"Simulation file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    print("=========================================================")
    print("EXIT MODEL SIMULATION: OUT-OF-SAMPLE RESULTS")
    print("=========================================================\n")
    
    # Summary Table
    total_orig = df['orig_pnl'].sum()
    
    print("1. SUMMARY PERFORMANCE")
    print("-" * 65)
    print(f"{'Metric':<20} | {'Original':<10} | {'Thr=0.20':<10} | {'Thr=0.30':<10}")
    print("-" * 65)
    
    for metric, orig_col, col_20, col_30 in [
        ("Total PnL", "orig_pnl", "new_pnl_20", "new_pnl_30"),
    ]:
        print(f"{metric:<20} | ${total_orig:>9.2f} | ${df[col_20].sum():>9.2f} | ${df[col_30].sum():>9.2f}")
        
    print(f"{'Trades Cut Short':<20} | {'0':>10} | {df['cut_20'].sum():>10.0f} | {df['cut_30'].sum():>10.0f}")
    
    avg_saved_20 = (df[df['cut_20']==1]['orig_duration'] - df[df['cut_20']==1]['dur_20']).mean()
    avg_saved_30 = (df[df['cut_30']==1]['orig_duration'] - df[df['cut_30']==1]['dur_30']).mean()
    print(f"{'Avg Bars Saved':<20} | {'0.0':>10} | {avg_saved_20:>10.1f} | {avg_saved_30:>10.1f}")
    print("-" * 65)
    
    print("\n2. HOUR-BY-HOUR PnL COMPARISON (hrxhr)")
    print("-" * 50)
    print(f"{'Hour':<4} | {'Original':<12} | {'Thr=0.20':<12} | {'Thr=0.30':<12}")
    print("-" * 50)
    
    hr_grouped = df.groupby('hour').sum(numeric_only=True)
    for hr in sorted(hr_grouped.index):
        row = hr_grouped.loc[hr]
        print(f"{int(hr):02d}:00 | ${row['orig_pnl']:>10.2f} | ${row['new_pnl_20']:>10.2f} | ${row['new_pnl_30']:>10.2f}")
    print("-" * 50)
    
    print("\n3. SAMPLE OF TRADES CUT SHORT (Falling Knives Caught by Thr=0.30)")
    print("-" * 80)
    print(f"{'TradeID':<8} | {'Day':<12} | {'Orig_PnL':<10} | {'New_PnL':<10} | {'MAE_At_Cut':<10} | {'Bars_Saved':<10}")
    print("-" * 80)
    
    cut_df = df[df['cut_30'] == 1].copy()
    # Sort by the most money saved (Original PnL - New PnL)
    cut_df['money_saved'] = cut_df['new_pnl_30'] - cut_df['orig_pnl']
    cut_df = cut_df.sort_values('money_saved', ascending=False)
    
    sample = cut_df.head(15)
    for _, row in sample.iterrows():
        bars_saved = row['orig_duration'] - row['dur_30']
        print(f"{int(row['trade_id']):<8} | {row['day']:<12} | ${row['orig_pnl']:>9.2f} | ${row['new_pnl_30']:>9.2f} | ${row['mae_at_cut_30']:>9.2f} | {int(bars_saved):>10}")
        
    print("-" * 80)
    
    total_saved = cut_df['money_saved'].sum()
    print(f"\nTotal money saved across all {len(cut_df)} early exits: ${total_saved:.2f}")

if __name__ == '__main__':
    main()
