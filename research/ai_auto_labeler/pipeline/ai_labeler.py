import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from datetime import datetime

ATLAS_1M_DIR = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1m'
ATLAS_1S_DIR = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
OUTPUT_DIR = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ai_cusp_picks'
TICK_SIZE = 0.25

def get_1s_cache(cache, dt_key):
    if dt_key not in cache:
        path = os.path.join(ATLAS_1S_DIR, f"{dt_key}.parquet")
        if os.path.exists(path):
            cache[dt_key] = pd.read_parquet(path)
        else:
            cache[dt_key] = None
    return cache[dt_key]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', type=str, required=True, help='YYYY_MM')
    parser.add_argument('--prominence', type=float, default=7.0, help='Minimum follow-through points')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all 1m files for this month
    files_1m = glob.glob(os.path.join(ATLAS_1M_DIR, f"{args.month}_*.parquet"))
    files_1m.sort()
    
    _1s_cache = {}
    all_trades = []
    
    print(f"Running AI Labeler for {args.month} with Prominence >= {args.prominence}")
    
    for f1m in files_1m:
        date_key = os.path.basename(f1m).replace('.parquet', '')
        date_str = date_key.replace('_', '-')
        
        try:
            df_1m = pd.read_parquet(f1m)
        except:
            continue
            
        close_1m = df_1m['close'].values.astype(float)
        ts_1m = df_1m['timestamp'].values.astype(float)
        
        if len(close_1m) < 100: continue
        
        # Calculate Cubic Spline
        x = np.arange(len(close_1m))
        spline = UnivariateSpline(x, close_1m, s=len(close_1m)*30)
        cubic_curve = spline(x)
        
        # Find candidates
        tops, _ = find_peaks(cubic_curve)
        bottoms, _ = find_peaks(-cubic_curve)
        
        candidates = []
        for t in tops:
            candidates.append({'index': t, 'type': 'top', 'direction': 'SHORT'})
        for b in bottoms:
            candidates.append({'index': b, 'type': 'bottom', 'direction': 'LONG'})
            
        candidates.sort(key=lambda c: c['index'])
        
        # Filter by prominence
        valid_cands = []
        for cand in candidates:
            idx = cand['index']
            direction = cand['direction']
            
            fwd_curve = cubic_curve[idx : idx+60]
            if len(fwd_curve) < 2: continue
            
            if direction == 'LONG':
                prom = np.max(fwd_curve) - cubic_curve[idx]
            else:
                prom = cubic_curve[idx] - np.min(fwd_curve)
                
            if prom >= args.prominence:
                valid_cands.append(cand)
                
        if not valid_cands:
            continue
            
        # Load 1s data for optimization
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        prev_key = (dt - pd.Timedelta(days=1)).strftime('%Y_%m_%d')
        next_key = (dt + pd.Timedelta(days=1)).strftime('%Y_%m_%d')
        
        df_prev = get_1s_cache(_1s_cache, prev_key)
        df_curr = get_1s_cache(_1s_cache, date_key)
        df_next = get_1s_cache(_1s_cache, next_key)
        
        dfs = []
        if df_prev is not None: dfs.append(df_prev)
        if df_curr is not None: dfs.append(df_curr)
        if df_next is not None: dfs.append(df_next)
        
        if not dfs: continue
        df_combined = pd.concat(dfs).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        ts_array_1s = df_combined['timestamp'].values.astype(float)
        
        day_trades = []
        for cand in valid_cands:
            orig_ts = ts_1m[cand['index']]
            direction = cand['direction']
            
            # 1. OPTIMIZE ENTRY: Search +/- 30s
            ts_start_search = orig_ts - 30
            ts_end_search = orig_ts + 30
            mask_search = (ts_array_1s >= ts_start_search) & (ts_array_1s <= ts_end_search)
            if not np.any(mask_search): continue
            
            df_search = df_combined.iloc[mask_search]
            if direction == 'SHORT':
                best_idx = df_search['high'].idxmax()
                entry_price = float(df_search.loc[best_idx, 'high'])
                entry_ts = float(df_search.loc[best_idx, 'timestamp'])
            else:
                best_idx = df_search['low'].idxmin()
                entry_price = float(df_search.loc[best_idx, 'low'])
                entry_ts = float(df_search.loc[best_idx, 'timestamp'])
                
            # 2. OPTIMIZE EXIT: Scan forward 60 mins
            ts_fwd_end = entry_ts + 3600.0
            mask_fwd = (ts_array_1s >= entry_ts) & (ts_array_1s <= ts_fwd_end)
            if not np.any(mask_fwd): continue
            
            df_fwd = df_combined.iloc[mask_fwd]
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
            
            pnl_dollars = mfe_ticks * 0.50
            mae_dollars = mae_ticks * 0.50
            
            t = {
                'entry_ts': entry_ts,
                'exit_ts': exit_ts,
                'direction': direction,
                'side': 'Buy' if direction == 'LONG' else 'Sell',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_dollars': pnl_dollars,
                'mae_dollars': mae_dollars,
                'original_timestamp': orig_ts
            }
            day_trades.append(t)
            all_trades.append(t)
            
        if day_trades:
            # Save multi file for this day
            out_file = os.path.join(OUTPUT_DIR, f"ai_picks_{date_str}_multi.json")
            with open(out_file, 'w') as f:
                json.dump({'trades': day_trades}, f, indent=2)
                
    # Summary stats
    if all_trades:
        avg_mfe = np.mean([t['pnl_dollars'] for t in all_trades]) / 0.50
        avg_mae = np.mean([t['mae_dollars'] for t in all_trades]) / 0.50
        wins = len([t for t in all_trades if t['pnl_dollars'] >= 20.0]) # At least 40 ticks
        win_rate = wins / len(all_trades) * 100
        
        print(f"Generated {len(all_trades)} trades for {args.month}")
        print(f"Avg MFE Ticks: {avg_mfe:.1f}")
        print(f"Avg MAE Ticks: {avg_mae:.1f}")
        print(f"Win Rate (>=40 ticks): {win_rate:.1f}%")
    else:
        print(f"No valid trades generated for {args.month}")

if __name__ == '__main__':
    main()
