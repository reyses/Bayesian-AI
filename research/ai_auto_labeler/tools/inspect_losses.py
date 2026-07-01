import os
import glob
import json
import pandas as pd
from datetime import datetime

OUTPUT_DIR = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ai_cusp_picks'
multi_files = glob.glob(os.path.join(OUTPUT_DIR, "*_multi.json"))

losses = []
total_trades = 0

for mf in multi_files:
    with open(mf, 'r') as f:
        data = json.load(f)
        trades = data.get('trades', [])
        total_trades += len(trades)
        
        for t in trades:
            if t['pnl_dollars'] < 20.0:  # Less than 40 ticks
                t['date'] = os.path.basename(mf).replace('ai_picks_', '').replace('_multi.json', '')
                losses.append(t)

print(f"Total trades: {total_trades}")
print(f"Total losses (< 40 ticks MFE): {len(losses)}")

if losses:
    df = pd.DataFrame(losses)
    df['entry_time'] = pd.to_datetime(df['entry_ts'], unit='s')
    df['exit_time'] = pd.to_datetime(df['exit_ts'], unit='s')
    df['duration_mins'] = (df['exit_ts'] - df['entry_ts']) / 60.0
    
    # Check what time of day these losses occur
    df['hour'] = df['entry_time'].dt.hour
    
    print("\nLosses by Hour (UTC):")
    print(df['hour'].value_counts().sort_index())
    
    print("\nAverage Duration to MFE (mins) for losses:")
    print(df['duration_mins'].mean())
    
    print("\nAverage MFE (ticks) for losses:")
    print((df['pnl_dollars'] / 0.5).mean())
    
    print("\nSample of 5 losses:")
    for _, row in df.head(5).iterrows():
        print(f"{row['date']} {row['direction']} Entry: {row['entry_time'].strftime('%H:%M:%S')} Exit: {row['exit_time'].strftime('%H:%M:%S')} MFE Ticks: {row['pnl_dollars']/0.5:.1f} Duration: {row['duration_mins']:.1f}m")
