import os
import pandas as pd
import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
tests_dir = os.path.join(base_dir, 'tests')
atlas_dir = os.path.abspath(os.path.join(base_dir, '../../DATA/ATLAS/5s'))

dirs = ['ATR-09_Statistical_Fade', 'FIB-17_Confluence', 'VA-13_Rotation', 'ORDERFLOW-14']

for d in dirs:
    path = os.path.join(tests_dir, d, 'events.parquet')
    if not os.path.exists(path):
        print(f"Missing {path}")
        continue
    
    df = pd.read_parquet(path)
    
    depths = []
    cache = {}
    
    for i, row in df.iterrows():
        day = row['day']
        try:
            day_file = day.replace('-', '_')
            if day_file not in cache:
                cache[day_file] = pd.read_parquet(os.path.join(atlas_dir, f"{day_file}.parquet"), columns=['close'])['close'].values
                
            prices = cache[day_file]
            open_price = prices[0]
            # Handle possible float/int issues with event_idx
            idx = int(row['event_idx'])
            if idx < len(prices):
                p0 = prices[idx]
            else:
                p0 = prices[-1]
            depths.append(abs(p0 - open_price))
        except Exception as e:
            depths.append(1.0) # Fallback
            
    df['depth'] = depths
    df.to_parquet(path)
    print(f"Patched depth for {d}: min {np.min(depths):.2f}, max {np.max(depths):.2f}")
