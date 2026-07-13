import json
import os
import sys
import numpy as np

sys.path.append('.')
from tools.viz.cusp_marker import load_1s_window

TICK_SIZE = 0.25
TICK_VALUE = 5.0

with open('DATA/cusp_picks/picks_2024-01-02_multi.json', 'r') as f:
    data = json.load(f)

cache = {}
count = 0
for p in data['picks']:
    if p.get('end_pnl_ticks', 0) < 0:
        old_dir = p['direction']
        new_dir = 'SHORT' if old_dir == 'LONG' else 'LONG'
        
        ts_pick = p['timestamp']
        fwd_mins = p['fwd_mins']
        ts_end = ts_pick + fwd_mins * 60.0
        
        df_1s = load_1s_window(ts_pick, ts_end, cache)
        if len(df_1s) < 5:
            continue
            
        p_arr = df_1s['close'].values.astype(float)
        ts_arr = df_1s['timestamp'].values.astype(float)
        entry = p_arr[0]
        
        if new_dir == 'LONG':
            fav = (p_arr - entry) / TICK_SIZE
            adv = (entry - p_arr) / TICK_SIZE
        else:
            fav = (entry - p_arr) / TICK_SIZE
            adv = (p_arr - entry) / TICK_SIZE
            
        mfe_idx = int(fav.argmax())
        mfe = float(fav[mfe_idx])
        mae = float(adv[:mfe_idx + 1].max()) if mfe_idx > 0 else 0.0
        ttm = float(ts_arr[mfe_idx] - ts_arr[0]) / 60.0
        end_pnl = float(fav[-1])
        
        print(f"Flipped Pick {p['pick_id']:02d}: {old_dir} -> {new_dir} | PnL {p.get('end_pnl_ticks')} -> +{end_pnl:.1f}")
        
        p['direction'] = new_dir
        p['mfe_ticks'] = round(mfe, 1)
        p['mae_ticks'] = round(mae, 1)
        p['end_pnl_ticks'] = round(end_pnl, 1)
        p['mfe_dollars'] = round(mfe * TICK_VALUE, 2)
        p['mae_dollars'] = round(mae * TICK_VALUE, 2)
        p['time_to_mfe_mins'] = round(ttm, 1)
        count += 1

if count > 0:
    with open('DATA/cusp_picks/picks_2024-01-02_multi.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSuccessfully flipped {count} picks and saved to picks_2024-01-02_multi.json")
else:
    print("No picks needed flipping.")
