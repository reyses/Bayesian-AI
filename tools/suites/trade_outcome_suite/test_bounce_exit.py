import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
DOLLAR_PER_POINT = 2.0
FRICTION_USD = 6.0

def test_bounces():
    legs_csv = REPO / 'reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv'
    bars_dir = REPO / 'DATA/ATLAS_NT8/5s'
    
    if not legs_csv.exists():
        print(f"Error: {legs_csv} not found. Run export script first.")
        return
        
    legs = pd.read_csv(legs_csv)
    days = sorted(legs['day'].unique())
    
    # Store trajectories
    trajectories = []
    skipped = 0
    
    for day in tqdm(days, desc='Extracting trajectories'):
        day_legs = legs[legs['day'] == day]
        bp = bars_dir / f'{day}.parquet'
        if not bp.exists():
            skipped += len(day_legs)
            continue
            
        b = pd.read_parquet(bp).sort_values('timestamp').reset_index(drop=True)
        ts = b['timestamp'].values.astype(np.int64)
        hi = b['high'].values.astype(np.float64)
        lo = b['low'].values.astype(np.float64)
        
        for _, leg in day_legs.iterrows():
            entry_ts, exit_ts = int(leg['entry_ts']), int(leg['exit_ts'])
            ep = float(leg['entry_price'])
            d = str(leg['leg_dir'])
            
            ei = int(np.searchsorted(ts, entry_ts, side='left'))
            if ei >= len(ts) or ts[ei] != entry_ts:
                ei = int(np.searchsorted(ts, entry_ts, side='right') - 1)
            if ei < 0:
                skipped += 1
                continue
                
            xi = int(np.searchsorted(ts, exit_ts, side='right') - 1)
            xi = max(xi, ei)
            
            sh, sl = hi[ei:xi + 1], lo[ei:xi + 1]
            if len(sh) == 0:
                skipped += 1
                continue
                
            if d == 'LONG':
                min_pnl = (sl - ep) * DOLLAR_PER_POINT
                max_pnl = (sh - ep) * DOLLAR_PER_POINT
            else:
                min_pnl = (ep - sh) * DOLLAR_PER_POINT
                max_pnl = (ep - sl) * DOLLAR_PER_POINT
                
            trajectories.append({
                'leg_id': leg.name,
                'pnl_usd': float(leg['pnl_usd']),
                'min_pnl': min_pnl,
                'max_pnl': max_pnl
            })
            
    print(f"\nExtracted {len(trajectories)} trajectories. Skipped {skipped}.")
    
    print("\n=========================================================")
    print("FALLING KNIFE (DEAD CAT BOUNCE) EV SWEEP")
    print("=========================================================")
    print("Condition: Trade plunges to <= -$D, then bounces to >= -$B")
    print("hold-bail > 0 means HOLDING to structural exit is better.")
    print("hold-bail < 0 means BAILING on the bounce saves money.")
    print("Bail PnL includes $6 friction (-B - 6).")
    print("=========================================================")
    print(f"{'D (Plunge)':>10} | {'B (Bounce)':>10} | {'Count':>7} | {'P(Recover >0)':>13} | {'Avg Close':>10} | {'Bail Net':>9} | {'hold-bail':>10}")
    print("-" * 86)
    
    for D in [40, 60, 80, 100]:
        for B in [30, 20, 10, 0]:
            if B >= D:
                continue
                
            count = 0
            recovers = 0
            total_close = 0.0
            
            for t in trajectories:
                plunged = False
                bounced = False
                for i in range(len(t['min_pnl'])):
                    if not plunged and t['min_pnl'][i] <= -D:
                        plunged = True
                    if plunged and t['max_pnl'][i] >= -B:
                        bounced = True
                        break
                
                if bounced:
                    count += 1
                    c = t['pnl_usd']
                    if c > 0:
                        recovers += 1
                    total_close += c
                    
            if count > 0:
                p_rec = recovers / count
                avg_close = total_close / count
                bail_net = -B - 6.0
                hold_bail = avg_close - bail_net
                print(f"-${D:<9} | -${B:<9} | {count:>7} | {p_rec*100:>12.1f}% | ${avg_close:>9.2f} | ${bail_net:>8.2f} | ${hold_bail:>9.2f}")
        print("-" * 86)

if __name__ == '__main__':
    test_bounces()
