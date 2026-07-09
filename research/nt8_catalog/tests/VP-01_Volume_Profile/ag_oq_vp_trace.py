import os
import sys
import numpy as np
import pandas as pd

# Load the daily profile logic from the deep dive script
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_deepdive_01_vol_profile import compute_daily_profile

def run_oq_trace(target_day, yesterday):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    
    yest_path = os.path.join(l0_dir, f"{yesterday}.parquet")
    today_path = os.path.join(l0_dir, f"{target_day}.parquet")
    
    if not os.path.exists(yest_path) or not os.path.exists(today_path):
        print(f"Data not found for {yesterday} or {target_day}")
        return
        
    print(f"--- OQ Trace: {target_day} ---")
    print(f"Loading Yesterday ({yesterday})...")
    
    yest_profile = compute_daily_profile(yest_path)
    print(f"[IQ/OQ] Yesterday's Computed Profile:")
    print(f"  Total Volume: {yest_profile['total_vol']}")
    print(f"  High:  {yest_profile['high']:.2f}")
    print(f"  VAH:   {yest_profile['vah']:.2f}")
    print(f"  POC:   {yest_profile['poc']:.2f}")
    print(f"  VAL:   {yest_profile['val']:.2f}")
    print(f"  Low:   {yest_profile['low']:.2f}")
    
    df = pd.read_parquet(today_path, columns=['close', 'timestamp'])
    prices = df['close'].values
    times = df['timestamp'].values
    
    if len(prices) == 0:
        print("No prices found for today.")
        return
        
    open_price = prices[0]
    print(f"\n[IQ/OQ] Today's Open Price: {open_price:.2f}")
    
    setup = 0
    vh = yest_profile['vah']
    vl = yest_profile['val']
    ph = yest_profile['high']
    pl = yest_profile['low']
    poc = yest_profile['poc']
    
    if vh < open_price < ph:
        print("[IQ/OQ] Categorization: SETUP 1 (Bullish Signal) - Open > VAH and Open < High")
        setup = 1
    elif pl < open_price < vl:
        print("[IQ/OQ] Categorization: SETUP 2 (Bearish Signal) - Open < VAL and Open > Low")
        setup = 2
    elif open_price > ph:
        print("[IQ/OQ] Categorization: SETUP 3 (Bullish Runner) - Open > High")
        setup = 3
    elif open_price < pl:
        print("[IQ/OQ] Categorization: SETUP 3 (Bearish Runner) - Open < Low")
        setup = 3
    else:
        print("[IQ/OQ] Categorization: NONE - Open is inside the Value Area")
        return
        
    if setup in [1, 2]:
        print("\n[OQ] Scanning for POC Retracement Trigger...")
        event_idx = -1
        for i, p in enumerate(prices):
            if setup == 1 and p <= poc:
                event_idx = i
                print(f"[OQ] TRIGGER HIT: Price retraced down to POC ({poc:.2f}) at index {i} (Time: {times[i]})")
                break
            elif setup == 2 and p >= poc:
                event_idx = i
                print(f"[OQ] TRIGGER HIT: Price retraced up to POC ({poc:.2f}) at index {i} (Time: {times[i]})")
                break
                
        if event_idx == -1:
            print("[OQ] No trigger hit. Price never touched the POC today.")
            
    print("-" * 40 + "\n")

if __name__ == '__main__':
    # OQ Traces for specific days in Jan 2024
    run_oq_trace('2024_01_03', '2024_01_02')
    run_oq_trace('2024_01_04', '2024_01_03')
    run_oq_trace('2024_01_05', '2024_01_04')
