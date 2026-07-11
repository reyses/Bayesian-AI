import os, glob
import pandas as pd
import numpy as np

def rolling_ols_bands(close, W):
    n = len(close)
    if n < W: return np.full(n, 1.0)
    x = np.linspace(-1.0, 1.0, W)
    X = np.stack([np.ones(W), x], axis=1)
    P = np.linalg.pinv(X)
    sw = np.lib.stride_tricks.sliding_window_view(close, W)
    C = sw @ P.T
    fit = C @ X.T
    sig = np.sqrt(((sw - fit) ** 2).mean(axis=1))
    pad = np.full(W - 1, np.nan)
    return np.r_[pad, sig]

def process_dossier(events_path):
    df = pd.read_parquet(events_path)
    print(f"Processing {events_path}...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    l0_dir = os.path.join(base_dir, '..', 'DATA', 'ATLAS', '5s')
    
    all_mfe = []
    all_mae = []
    all_mag_sig = []
    all_mfe_sig = []
    all_mae_sig = []
    
    day_cache = {}
    has_mfe = False
    
    for idx, row in df.iterrows():
        day = row['day']
        if day not in day_cache:
            try:
                day_df = pd.read_parquet(os.path.join(l0_dir, f"{day}.parquet"), columns=['close', 'timestamp'])
                day_df['dt'] = pd.to_datetime(day_df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
                rth_5s = day_df[(day_df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (day_df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
                rth = rth_5s.resample('1min', on='dt').agg({'close': 'last'}).reset_index()
                
                rth['sigma'] = rolling_ols_bands(rth['close'].values, W=12)
                rth['sigma'] = rth['sigma'].bfill().fillna(1.0)
                
                day_cache[day] = {
                    'prices': rth['close'].values,
                    'sigmas': rth['sigma'].values,
                }
            except Exception as e:
                print(f"Error loading {day}: {e}")
                day_cache[day] = None
                
        cache = day_cache[day]
        if cache is None:
            all_mfe.append(np.nan)
            all_mae.append(np.nan)
            all_mag_sig.append(np.nan)
            all_mfe_sig.append(np.nan)
            all_mae_sig.append(np.nan)
            continue
            
        e_idx = int(row['event_idx'])
        mag = row['magnitude']
        mode = row['mode'].lower()
        
        if 'bull' in mode or 'long' in mode:
            direction = 1
        elif 'bear' in mode or 'short' in mode or 'breakdown' in mode:
            direction = -1
        else:
            direction = 1
            
        prices = cache['prices']
        sigmas = cache['sigmas']
        
        if e_idx >= len(prices):
            all_mfe.append(np.nan)
            all_mae.append(np.nan)
            all_mag_sig.append(np.nan)
            all_mfe_sig.append(np.nan)
            all_mae_sig.append(np.nan)
            continue
            
        p0 = prices[e_idx]
        sigma = sigmas[e_idx]
        if np.isnan(sigma) or sigma <= 0: sigma = 0.25
        
        if not has_mfe:
            p_exit = p0 + (direction * mag)
            path = prices[e_idx+1:]
            exit_offset = -1
            for i, p in enumerate(path):
                if np.isclose(p, p_exit, atol=1e-5):
                    exit_offset = i
                    break
                    
            if exit_offset == -1:
                exit_offset = min(720, len(path)-1)
                
            segment = path[:exit_offset+1]
            if len(segment) == 0:
                mfe = 0.0
                mae = 0.0
            else:
                if direction == 1:
                    mfe = np.max(segment) - p0
                    mae = p0 - np.min(segment)
                else:
                    mfe = p0 - np.min(segment)
                    mae = np.max(segment) - p0
                    
            mfe = max(0.0, mfe)
            mae = max(0.0, mae)
        else:
            mfe = row['mfe']
            mae = row['mae']
            
        all_mfe.append(mfe)
        all_mae.append(mae)
        all_mag_sig.append(mag / sigma)
        all_mfe_sig.append(mfe / sigma)
        all_mae_sig.append(mae / sigma)
        
    if not has_mfe:
        df['mfe'] = all_mfe
        df['mae'] = all_mae
    df['magnitude_sigma'] = all_mag_sig
    df['mfe_sigma'] = all_mfe_sig
    df['mae_sigma'] = all_mae_sig
    
    df.to_parquet(events_path)
    print(f"Updated {events_path} with MFE/MAE and sigma columns.")

if __name__ == '__main__':
    for f in glob.glob('tests/*/*/events.parquet') + glob.glob('tests/*/events.parquet'):
        if 'archive' in f: continue
        process_dossier(f)
