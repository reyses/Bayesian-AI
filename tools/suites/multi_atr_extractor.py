import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import numba

REPO = Path(__file__).resolve().parents[2]


def load_raw_5s_closes(day, root):
    path = Path(root) / '5s' / f"{day}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=['timestamp', 'close'])

@numba.njit
def calc_zigzag_state(closes, ts, r_pts, min_bars=36):
    """
    Compute the ZigZag state for the entire day.
    Returns:
      leg_dir: array of +1 (LONG), -1 (SHORT), 0 (undecided)
      extreme_price: array of the current extreme price
      pivot_ts: array of the timestamp of the last confirmed pivot
    """
    n = len(closes)
    leg_dir = np.zeros(n, dtype=np.int32)
    extreme_price = np.zeros(n, dtype=np.float32)
    pivot_ts = np.zeros(n, dtype=np.int64)
    
    if n == 0:
        return leg_dir, extreme_price, pivot_ts
        
    cur_dir = 0
    cur_ext = closes[0]
    cur_ext_idx = 0
    cur_pivot_ts = ts[0]
    
    for i in range(1, n):
        c = closes[i]
        
        if cur_dir == 0:
            if c >= cur_ext + r_pts:
                cur_dir = 1
                cur_pivot_ts = ts[0]
                cur_ext = c
                cur_ext_idx = i
            elif c <= cur_ext - r_pts:
                cur_dir = -1
                cur_pivot_ts = ts[0]
                cur_ext = c
                cur_ext_idx = i
        elif cur_dir == 1:
            if c > cur_ext:
                cur_ext = c
                cur_ext_idx = i
            elif c <= cur_ext - r_pts and (i - cur_ext_idx) >= min_bars:
                cur_dir = -1
                cur_pivot_ts = ts[cur_ext_idx]
                cur_ext = c
                cur_ext_idx = i
        elif cur_dir == -1:
            if c < cur_ext:
                cur_ext = c
                cur_ext_idx = i
            elif c >= cur_ext + r_pts and (i - cur_ext_idx) >= min_bars:
                cur_dir = 1
                cur_pivot_ts = ts[cur_ext_idx]
                cur_ext = c
                cur_ext_idx = i
                
        leg_dir[i] = cur_dir
        extreme_price[i] = cur_ext
        pivot_ts[i] = cur_pivot_ts
        
    return leg_dir, extreme_price, pivot_ts


def process_dataset(csv_path, out_path, is_oos=False):
    trades = pd.read_csv(csv_path)
    days = trades['day'].unique()
    
    mults = [1.0, 2.0, 4.0, 8.0, 10.0]
    results = []
    
    # Pre-allocate new columns
    trades['true_pivot_ts'] = 0
    for m in mults:
        trades[f'dir_x{int(m)}'] = 0
        trades[f'dist_x{int(m)}'] = 0.0
        
    for day in tqdm(days, desc=f"Extracting {'OOS' if is_oos else 'IS'} Multi-ATR"):
        day_trades = trades[trades['day'] == day]
        if day_trades.empty:
            continue
            
        root = str(REPO / 'DATA/ATLAS_NT8' if is_oos else REPO / 'DATA/ATLAS')
        df_5s = load_raw_5s_closes(day, root)
        if df_5s.empty:
            continue
            
        ts = df_5s['timestamp'].values.astype(np.int64)
        closes = df_5s['close'].values.astype(np.float32)
        
        # Calculate ZigZag for all multipliers
        # Using ATR=5.75 points as a baseline if atr_pts isn't in trades (it is in V2)
        # We will use the atr_pts from the first trade of the day (it's daily ATR)
        atr_pts = day_trades.iloc[0]['atr_pts']
        
        states = {}
        for m in mults:
            r_pts = atr_pts * m
            # Min bars: standard is 36 for ATRx4. Let's scale it slightly, or keep 36.
            # 36 bars is 3 mins. For x1, maybe less? The strategy uses 36 for x4. 
            # I'll keep 36 for all to be consistent with the inspector, which uses same minimums usually.
            ldir, ext_p, piv_ts = calc_zigzag_state(closes, ts, r_pts, min_bars=36)
            states[m] = (ldir, ext_p, piv_ts)
            
        for idx, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            # Find closest 5s bar
            pos = np.searchsorted(ts, entry_ts, side='left')
            if pos >= len(ts):
                pos = len(ts) - 1
                
            c_price = closes[pos]
            
            # The true pivot anchor for the ATRx4 grid
            true_pivot_ts = states[4.0][2][pos]
            trades.at[idx, 'true_pivot_ts'] = true_pivot_ts
            
            for m in mults:
                ldir, ext_p, _ = states[m]
                d = ldir[pos]
                e = ext_p[pos]
                
                # Distance to extreme normalized by ATR
                # If LONG (+1), extreme is the High. Price is below it. Distance is (c_price - e) / atr_pts. (will be negative)
                # If SHORT (-1), extreme is the Low. Price is above it. Distance is (c_price - e) / atr_pts. (will be positive)
                # To make it symmetric: if LONG, distance from High; if SHORT, distance from Low.
                dist = (c_price - e) / atr_pts
                
                trades.at[idx, f'dir_x{int(m)}'] = d
                trades.at[idx, f'dist_x{int(m)}'] = dist

    trades.to_csv(out_path, index=False)
    print(f"Saved {len(trades)} trades to {out_path}")

def main():
    os.makedirs(str(REPO / 'reports/findings/multi_atr'), exist_ok=True)
    
    is_path = REPO / 'reports/findings/strategy_runs/zigzag_is_atr2.csv'
    is_out = REPO / 'reports/findings/multi_atr/multi_atr_is_atr2.csv'
    process_dataset(is_path, is_out, is_oos=False)
    
    oos_path = REPO / 'reports/findings/strategy_runs/zigzag_oos_atr2.csv'
    oos_out = REPO / 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    process_dataset(oos_path, oos_out, is_oos=True)

if __name__ == '__main__':
    main()
