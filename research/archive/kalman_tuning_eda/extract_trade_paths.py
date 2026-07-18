import os
import pandas as pd
import numpy as np
import time

def main():
    trades_csv = 'C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/reports/findings/kalman_full_trades.csv'
    if not os.path.exists(trades_csv):
        print(f"Cannot find {trades_csv}")
        return
        
    df_trades = pd.read_csv(trades_csv)
    atlas_dir = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
    
    print(f"Extracting paths for {len(df_trades)} trades...")
    
    # Group trades by day to avoid re-loading parquet files
    grouped = df_trades.groupby('day')
    
    extracted = []
    
    t0 = time.time()
    
    for day, group in grouped:
        pq_path = os.path.join(atlas_dir, f"{day}.parquet")
        if not os.path.exists(pq_path):
            continue
            
        df_day = pd.read_parquet(pq_path, columns=['timestamp', 'close']).sort_values('timestamp').reset_index(drop=True)
        ts_arr = df_day['timestamp'].values
        px_arr = df_day['close'].values
        
        for _, trade in group.iterrows():
            ets = trade['entry_ts']
            xts = trade['exit_ts']
            
            # Find indices
            # Since ts_arr is sorted, we can use np.searchsorted
            start_idx = np.searchsorted(ts_arr, ets)
            end_idx = np.searchsorted(ts_arr, xts, side='right')
            
            if start_idx >= len(ts_arr) or start_idx >= end_idx:
                path = np.array([], dtype=np.float32)
            else:
                raw_path = px_arr[start_idx:end_idx]
                # Normalize relative to entry price and direction
                eprice = trade['entry_price']
                if trade['dir'] == 'LONG':
                    path = (raw_path - eprice).astype(np.float32)
                else:
                    path = (eprice - raw_path).astype(np.float32)
                    
            extracted.append({
                'day': day,
                'split': trade['split'],
                'entry_ts': ets,
                'dir': trade['dir'],
                'net_usd': trade['net_usd'],
                'path': path
            })
            
    df_paths = pd.DataFrame(extracted)
    out_path = 'C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/reports/findings/trade_paths.parquet'
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_paths.to_parquet(out_path)
    
    print(f"Saved {len(df_paths)} trade paths to {out_path} in {time.time()-t0:.2f} seconds.")
    print(f"Parquet file size: {os.path.getsize(out_path) / 1e6:.2f} MB")

if __name__ == '__main__':
    main()
