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

def analyze_in_trade_zscores():
    legs_csv = REPO / 'reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv'
    bars_dir = REPO / 'DATA/ATLAS_NT8/5s'
    
    legs = pd.read_csv(legs_csv)
    days = sorted(legs['day'].unique())
    
    tfs = ['15s', '1m', '5m', '15m', '1h', '4h', '1D']
    z_cols = [f'L3_{tf}_z_se_{N_BASE[tf]}' for tf in tfs]
    
    results = []
    
    for day in tqdm(days, desc='Extracting In-Trade Z-scores'):
        day_legs = legs[legs['day'] == day]
        
        # Load 5s price bars for trajectories
        bp = bars_dir / f'{day}.parquet'
        if not bp.exists():
            continue
        b = pd.read_parquet(bp).sort_values('timestamp').reset_index(drop=True)
        ts_bars = b['timestamp'].values.astype(np.int64)
        hi = b['high'].values.astype(np.float64)
        lo = b['low'].values.astype(np.float64)
        
        # Load V2 features
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
            
            # --- Trajectory (for knife detection) ---
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
                    
            # --- Z-Score Analysis During Trade ---
            ei_feat = int(np.searchsorted(ts_feats, entry_ts, side='left'))
            if ei_feat >= len(ts_feats):
                continue
            # If the exact timestamp isn't found, searchsorted gives the insertion point.
            # We want to start from the bar just before or at entry_ts
            if ts_feats[ei_feat] > entry_ts and ei_feat > 0:
                ei_feat -= 1
                
            xi_feat = int(np.searchsorted(ts_feats, exit_ts, side='right') - 1)
            xi_feat = max(xi_feat, ei_feat)
            
            row = {
                'leg_id': idx, 
                'pnl_usd': leg['pnl_usd'],
                'is_falling_knife': bounced_20,
                'duration_bars': xi_feat - ei_feat + 1
            }
            
            for col in z_cols:
                if col in feats.columns:
                    z_vals = feats.iloc[ei_feat:xi_feat+1][col].values
                    z_vals = z_vals[~np.isnan(z_vals)]
                    if len(z_vals) > 0:
                        # Z_trend: >0 means price is on the PROFITABLE side of the regression mean
                        # (e.g. above the mean for a LONG, below the mean for a SHORT)
                        z_trend = z_vals * direction
                        row[f'{col}_mean_trend'] = np.mean(z_trend)
                        row[f'{col}_min_trend'] = np.min(z_trend) # The worst violation of the trend line
            results.append(row)
            
    df = pd.DataFrame(results)
    
    print("\n=========================================================")
    print("IN-TRADE Z-SCORE ANALYSIS: DO THEY FOLLOW THE TREND LINE?")
    print("=========================================================")
    print("Metric: Z_trend = Z-Score * Trade_Direction")
    print("  > 0 : Price is on the PROFITABLE side of the regression mean (follows trend line).")
    print("  < 0 : Price broke AGAINST the regression mean.")
    print("=========================================================\n")
    
    winners = df[df['pnl_usd'] > 0]
    losers = df[df['pnl_usd'] < 0]
    knives = df[df['is_falling_knife'] == True]
    
    print(f"{'Timeframe (N)':<15} | {'Winners':>10} | {'Losers':>10} | {'Knives':>10}")
    print("-" * 55)
    
    print("1. AVERAGE Z-SCORE DURING ENTIRE TRADE")
    for col in z_cols:
        col_m = f'{col}_mean_trend'
        if col_m not in df.columns:
            continue
        w_mean = winners[col_m].mean()
        l_mean = losers[col_m].mean()
        k_mean = knives[col_m].mean()
        tf_label = col.replace('L3_', '').replace('_z_se', '')
        print(f"{tf_label:<15} | {w_mean:>10.2f} | {l_mean:>10.2f} | {k_mean:>10.2f}")
        
    print("\n2. WORST Z-SCORE DURING TRADE (Maximum Break of Trend Line)")
    for col in z_cols:
        col_m = f'{col}_min_trend'
        if col_m not in df.columns:
            continue
        w_min = winners[col_m].mean()
        l_min = losers[col_m].mean()
        k_min = knives[col_m].mean()
        tf_label = col.replace('L3_', '').replace('_z_se', '')
        print(f"{tf_label:<15} | {w_min:>10.2f} | {l_min:>10.2f} | {k_min:>10.2f}")
        
    print("-" * 55)

if __name__ == '__main__':
    analyze_in_trade_zscores()
