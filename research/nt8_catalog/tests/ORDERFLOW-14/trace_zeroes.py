import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    parquet_path = os.path.join(base_dir, 'DATA/ATLAS/order_flow_delta_5s.parquet')
    
    df = pd.read_parquet(parquet_path)
    df = df[df['close'] > 10000].copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df['dt'] = df.index.tz_convert('America/Chicago')
    else:
        df['dt'] = pd.to_datetime(df.index, utc=True).tz_convert('America/Chicago')
        
    df['day_str'] = df['dt'].dt.strftime('%Y-%m-%d')
    df = df.sort_values('dt').reset_index(drop=True)
    
    df_day = df[df['day_str'] == '2025-07-30'].copy()
    df_rth = df_day[(df_day['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df_day['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    df_rth = df_rth.reset_index(drop=True)
    
    prices = df_rth['close'].values
    print("Checking max 60-bar diff on 2025-07-30...")
    max_diff = 0
    bad_idx = -1
    for i in range(len(prices)-60):
        diff = np.max(np.abs(prices[i+1:i+61] - prices[i]))
        if diff > max_diff:
            max_diff = diff
            bad_idx = i
            
    print(f"Max 60-bar diff: {max_diff} at index {bad_idx}")
    if bad_idx != -1:
        print(f"Prices around {bad_idx}:", prices[bad_idx:bad_idx+10])
