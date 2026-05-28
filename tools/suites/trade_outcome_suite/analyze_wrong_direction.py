import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from core_v2.features import load_features, FEATURE_NAMES

def main():
    csv_path = REPO / 'reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv'
    trades = pd.read_csv(csv_path)
    days = trades['day'].unique()
    
    feat_idx = FEATURE_NAMES.index('L3_1h_z_se_12')
    
    results = []
    
    for day in tqdm(days, desc="Checking Macro Direction"):
        day_trades = trades[trades['day'] == day]
        feats = load_features(days=[day], root=str(REPO / 'DATA/ATLAS_NT8/FEATURES_5s_v2'), require_all=False)
        if feats.empty:
            continue
            
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
            
        ts_feats = feats['timestamp'].values.astype(np.int64)
        macro_z = feats['L3_1h_z_se_12'].values.astype(np.float32)
        
        for _, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            leg_dir = 1 if trade['leg_dir'] == 'LONG' else -1
            
            ei = int(np.searchsorted(ts_feats, entry_ts, side='left'))
            if ei >= len(ts_feats):
                continue
                
            macro_val = macro_z[ei]
            # If macro_val > 0, the 1H trend is UP. If < 0, 1H trend is DOWN.
            macro_dir = 1 if macro_val > 0 else -1
            
            is_wrong_dir = 1 if leg_dir != macro_dir else 0
            
            results.append({
                'trade_id': len(results),
                'pnl_usd': trade['pnl_usd'],
                'leg_dir': leg_dir,
                'macro_val': macro_val,
                'is_wrong_dir': is_wrong_dir
            })
            
    df = pd.DataFrame(results)
    
    total = len(df)
    wrong = df['is_wrong_dir'].sum()
    print(f"\nTotal OOS Trades: {total}")
    print(f"Trades taken in WRONG DIRECTION (Counter to 1H Trend): {wrong} ({(wrong/total)*100:.1f}%)")
    
    wrong_df = df[df['is_wrong_dir'] == 1]
    right_df = df[df['is_wrong_dir'] == 0]
    
    print("\nPnL of Wrong Direction Trades:")
    print(f"Total PnL: ${wrong_df['pnl_usd'].sum():.2f}")
    print(f"Win Rate: {(wrong_df['pnl_usd'] > 0).mean()*100:.1f}%")
    
    print("\nPnL of Right Direction Trades:")
    print(f"Total PnL: ${right_df['pnl_usd'].sum():.2f}")
    print(f"Win Rate: {(right_df['pnl_usd'] > 0).mean()*100:.1f}%")

if __name__ == '__main__':
    main()
