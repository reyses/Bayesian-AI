import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from core_v2.features import load_features
from core_v2.statistical_field_engine import N_BASE

REPO = Path(__file__).resolve().parents[1]
DOLLAR_PER_POINT = 2.0

def analyze_zscores():
    legs_csv = REPO / 'reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv'
    bars_dir = REPO / 'DATA/ATLAS_NT8/5s'
    
    legs = pd.read_csv(legs_csv)
    days = sorted(legs['day'].unique())
    
    tfs = ['1m', '5m', '15m', '1h', '4h', '1D']
    z_cols = [f'L3_{tf}_z_se_{N_BASE[tf]}' for tf in tfs]
    
    results = []
    
    for day in tqdm(days, desc='Extracting Z-scores and Trajectories'):
        day_legs = legs[legs['day'] == day]
        
        # Load 5s price bars for trajectories
        bp = bars_dir / f'{day}.parquet'
        if not bp.exists():
            continue
        b = pd.read_parquet(bp).sort_values('timestamp').reset_index(drop=True)
        ts_bars = b['timestamp'].values.astype(np.int64)
        hi = b['high'].values.astype(np.float64)
        lo = b['low'].values.astype(np.float64)
        
        # Load V2 features for Z-scores
        feats = load_features(days=[day], root='DATA/ATLAS_NT8/FEATURES_5s_v2', require_all=False)
        if feats.empty:
            continue
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
        ts_feats = feats['timestamp'].values.astype(np.int64)
        
        for idx, leg in day_legs.iterrows():
            entry_ts, exit_ts = int(leg['entry_ts']), int(leg['exit_ts'])
            ep = float(leg['entry_price'])
            d = str(leg['leg_dir'])
            direction = 1 if d == 'LONG' else -1
            
            # --- Trajectory Analysis ---
            ei_bar = int(np.searchsorted(ts_bars, entry_ts, side='left'))
            if ei_bar >= len(ts_bars) or ts_bars[ei_bar] != entry_ts:
                ei_bar = int(np.searchsorted(ts_bars, entry_ts, side='right') - 1)
            
            xi_bar = int(np.searchsorted(ts_bars, exit_ts, side='right') - 1)
            xi_bar = max(xi_bar, ei_bar)
            
            sh, sl = hi[ei_bar:xi_bar + 1], lo[ei_bar:xi_bar + 1]
            if len(sh) == 0:
                continue
                
            if d == 'LONG':
                min_pnl = (sl - ep) * DOLLAR_PER_POINT
                max_pnl = (sh - ep) * DOLLAR_PER_POINT
            else:
                min_pnl = (ep - sh) * DOLLAR_PER_POINT
                max_pnl = (ep - sl) * DOLLAR_PER_POINT
                
            plunged_60 = False
            bounced_20 = False
            for i in range(len(min_pnl)):
                if not plunged_60 and min_pnl[i] <= -60:
                    plunged_60 = True
                if plunged_60 and max_pnl[i] >= -20:
                    bounced_20 = True
                    break
                    
            # --- Z-Score Analysis ---
            ei_feat_arr = np.where(ts_feats == entry_ts)[0]
            if len(ei_feat_arr) == 0:
                continue
            ei_feat = ei_feat_arr[0]
            
            row = {
                'leg_id': idx, 
                'pnl_usd': leg['pnl_usd'],
                'is_falling_knife': bounced_20
            }
            
            for col in z_cols:
                if col in feats.columns:
                    z = feats.at[ei_feat, col]
                    row[f'{col}_against'] = z * direction
            results.append(row)
            
    df = pd.DataFrame(results)
    
    print("\n=========================================================")
    print("ENTRY Z-SCORE ANALYSIS: STRUCTURALLY AGAINST THE MEAN?")
    print("=========================================================")
    print("Positive value = Entered AGAINST the mean (e.g., LONG at top of channel).")
    print("Negative value = Entered WITH the mean (e.g., LONG at bottom of channel).")
    print("A value of +1.0 means we entered 1 standard error above the regression line.")
    print("=========================================================\n")
    
    winners = df[df['pnl_usd'] > 0]
    losers = df[df['pnl_usd'] < 0]
    knives = df[df['is_falling_knife'] == True]
    
    print(f"Total Winners: {len(winners)}")
    print(f"Total Losers:  {len(losers)}")
    print(f"Total Knives (-$60 -> -$20 bounce): {len(knives)}\n")
    
    print(f"{'Timeframe (N)':<15} | {'Winners':>10} | {'Losers':>10} | {'Knives':>10}")
    print("-" * 55)
    
    for col in z_cols:
        col_ag = f'{col}_against'
        if col_ag not in df.columns:
            continue
            
        w_mean = winners[col_ag].mean()
        l_mean = losers[col_ag].mean()
        k_mean = knives[col_ag].mean()
        
        tf_label = col.replace('L3_', '').replace('_z_se', '')
        print(f"{tf_label:<15} | {w_mean:>10.2f} | {l_mean:>10.2f} | {k_mean:>10.2f}")
        
    print("-" * 55)
    
    print("\n--- High-Risk Entry Threshold (> 1.5 SE against mean) ---")
    for col in z_cols:
        col_ag = f'{col}_against'
        if col_ag not in df.columns:
            continue
        pct_k = (knives[col_ag] > 1.5).mean() * 100
        pct_w = (winners[col_ag] > 1.5).mean() * 100
        tf_label = col.replace('L3_', '').replace('_z_se', '')
        print(f"{tf_label:<15} | Knives: {pct_k:>5.1f}% | Winners: {pct_w:>5.1f}%")

if __name__ == '__main__':
    analyze_zscores()
