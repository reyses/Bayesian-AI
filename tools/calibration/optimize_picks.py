import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

PICKS_DIR = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/cusp_picks'
ATLAS_1S_DIR = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
OUTPUT_FILE = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/optimal_trades.json'

TICK_SIZE = 0.25
SEARCH_WINDOW_SEC = 30
FWD_MINS = 60

def get_1s_cache(cache, dt_key):
    if dt_key not in cache:
        path = os.path.join(ATLAS_1S_DIR, f"{dt_key.replace('-', '_')}.parquet")
        if os.path.exists(path):
            print(f"Loading 1s data for {dt_key}...")
            cache[dt_key] = pd.read_parquet(path)
        else:
            cache[dt_key] = None
    return cache[dt_key]

def optimize_picks():
    multi_files = glob.glob(os.path.join(PICKS_DIR, '*_multi.json'))
    
    optimal_trades = []
    _1s_cache = {}
    
    for mf in multi_files:
        date_key = os.path.basename(mf).split('_')[1]  # e.g., 2024-01-02
        
        with open(mf) as f:
            data = json.load(f)
            
        picks = data.get('picks', [])
        if not picks: continue
        
        # Load primary 1s file for this date
        df_1s = get_1s_cache(_1s_cache, date_key)
        
        # We might need adjacent files if the pick is near midnight UTC.
        # We'll just load the day before and day after into one mega DF for this file processing
        try:
            dt = datetime.strptime(date_key, '%Y-%m-%d')
            prev_key = (dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            next_key = (dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        except:
            continue
            
        df_prev = get_1s_cache(_1s_cache, prev_key)
        df_next = get_1s_cache(_1s_cache, next_key)
        
        dfs = []
        if df_prev is not None: dfs.append(df_prev)
        if df_1s is not None: dfs.append(df_1s)
        if df_next is not None: dfs.append(df_next)
        
        if not dfs:
            print(f"No 1s data for {date_key}, skipping...")
            continue
            
        df_combined = pd.concat(dfs).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        ts_array = df_combined['timestamp'].values.astype(float)
        
        print(f"Processing {len(picks)} picks from {date_key}...")
        
        for p in picks:
            orig_ts = float(p.get('timestamp', 0))
            direction = p.get('direction', 'LONG')
            
            # 1. OPTIMIZE ENTRY: Search +/- 30s
            ts_start_search = orig_ts - SEARCH_WINDOW_SEC
            ts_end_search = orig_ts + SEARCH_WINDOW_SEC
            
            mask_search = (ts_array >= ts_start_search) & (ts_array <= ts_end_search)
            if not np.any(mask_search):
                continue
                
            df_search = df_combined.iloc[mask_search]
            
            # Find extreme
            if direction == 'SHORT':
                # Best entry for SHORT is the highest High
                best_idx = df_search['high'].idxmax()
                entry_price = float(df_search.loc[best_idx, 'high'])
                entry_ts = float(df_search.loc[best_idx, 'timestamp'])
            else:
                # Best entry for LONG is the lowest Low
                best_idx = df_search['low'].idxmin()
                entry_price = float(df_search.loc[best_idx, 'low'])
                entry_ts = float(df_search.loc[best_idx, 'timestamp'])
                
            # 2. OPTIMIZE EXIT: Scan forward FWD_MINS
            ts_fwd_end = entry_ts + FWD_MINS * 60.0
            mask_fwd = (ts_array >= entry_ts) & (ts_array <= ts_fwd_end)
            if not np.any(mask_fwd):
                continue
                
            df_fwd = df_combined.iloc[mask_fwd]
            
            # Calculate MFE trajectory
            fwd_p = df_fwd['close'].values.astype(float)
            fwd_ts = df_fwd['timestamp'].values.astype(float)
            
            if direction == 'LONG':
                fav = (fwd_p - entry_price) / TICK_SIZE
                adv = (entry_price - fwd_p) / TICK_SIZE
            else:
                fav = (entry_price - fwd_p) / TICK_SIZE
                adv = (fwd_p - entry_price) / TICK_SIZE
                
            mfe_idx = int(np.argmax(fav))
            mfe_ticks = float(fav[mfe_idx])
            mae_ticks = float(np.max(adv[:mfe_idx + 1])) if mfe_idx > 0 else 0.0
            
            exit_ts = float(fwd_ts[mfe_idx])
            exit_price = float(fwd_p[mfe_idx])
            
            optimal_trades.append({
                'entry_ts': entry_ts,
                'exit_ts': exit_ts,
                'direction': direction,
                'side': 'Buy' if direction == 'LONG' else 'Sell',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_dollars': mfe_ticks * 0.50,  # TICK_VALUE = 0.50 for MNQ
                'mae_dollars': mae_ticks * 0.50,
                'original_pick_id': p.get('pick_id'),
                'original_timestamp': orig_ts
            })
            
    # Save optimized dataset matching --load-trades schema
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({'trades': optimal_trades}, f, indent=2)
        
    print(f"\nGenerated {len(optimal_trades)} optimal trades.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    optimize_picks()
