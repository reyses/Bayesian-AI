import os
import numpy as np
import pandas as pd

def rolling_ols_bands(close, W):
    n = len(close)
    if n < W:
        return np.full(n, 1.0)
    x = np.linspace(-1.0, 1.0, W)
    X = np.stack([np.ones(W), x], axis=1)
    P = np.linalg.pinv(X)
    sw = np.lib.stride_tricks.sliding_window_view(close, W)
    C = sw @ P.T
    fit = C @ X.T
    sig = np.sqrt(((sw - fit) ** 2).mean(axis=1))
    pad = np.full(W - 1, np.nan)
    return np.r_[pad, sig]

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    parquet_path = os.path.join(base_dir, 'DATA/ATLAS/order_flow_delta_5s.parquet')
    
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if isinstance(df.index, pd.DatetimeIndex):
        df['dt'] = df.index.tz_convert('America/Chicago')
    else:
        df['dt'] = pd.to_datetime(df.index, utc=True).tz_convert('America/Chicago')
        
    df['day_str'] = df['dt'].dt.strftime('%Y-%m-%d')
    
    days = sorted(df['day_str'].unique())
    p10 = df['divergence'].dropna().quantile(0.10)
    p90 = df['divergence'].dropna().quantile(0.90)
    
    for test_day in days:
        df_day = df[df['day_str'] == test_day].copy()
        
        df_day['sigma'] = rolling_ols_bands(df_day['close'].values, W=12)
        df_day['sigma'] = df_day['sigma'].bfill().fillna(1.0)
        
        df_rth = df_day[(df_day['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df_day['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
        
        if len(df_rth) < 100:
            continue
            
        print(f"Testing on day {test_day} with {len(df_rth)} RTH rows")
        
        print("Sample of first 10 rows of df_rth:")
        print(df_rth[['dt', 'close', 'sigma']].head(10))
        print(f"Max sigma: {df_rth['sigma'].max()}")
        
        prices = df_rth['close'].values
        sigmas = df_rth['sigma'].values
        divergences = df_rth['divergence'].values
        deltas = df_rth['delta'].values
        
        df_rth = df_rth.reset_index(drop=True)
        is_peak = (df_rth['high'] == df_rth['high'].rolling(21, center=True).max()).values
        is_trough = (df_rth['low'] == df_rth['low'].rolling(21, center=True).min()).values
        
        cooldown = 0
        events = []
        
        for i in range(10, len(prices) - 60):
            if cooldown > 0:
                cooldown -= 1
                
            check_idx = i - 10
            setup = 0
            mode = 'none'
            
            if is_peak[check_idx]:
                d = deltas[check_idx]
                div = divergences[check_idx]
                if d > 0:
                    setup = 2; mode = 'bearish_runner'
                elif div < p10:
                    setup = 1; mode = 'bearish_bounce'
            elif is_trough[check_idx]:
                d = deltas[check_idx]
                div = divergences[check_idx]
                if d < 0:
                    setup = 2; mode = 'bullish_runner'
                elif div > p90:
                    setup = 1; mode = 'bullish_bounce'
                    
            if setup != 0 and cooldown <= 0:
                p0 = prices[i]
                path = prices[i+1 : i+61]
                std_path = sigmas[i+1 : i+61]
                
                if test_day == '2025-07-31' and i == 2278:
                    print("FOUND BAD EVENT!")
                    print("p0:", p0)
                    print("path:", path)
                    
                magnitude = 0.0
                if 'bearish' in mode:
                    for p, std in zip(path, std_path):
                        if p <= p0 - 3.0 * std:
                            magnitude = p0 - p
                            break
                        elif p >= p0 + 3.0 * std:
                            magnitude = p0 - p
                            break
                    if magnitude == 0.0:
                        magnitude = p0 - path[-1]
                elif 'bullish' in mode:
                    for p, std in zip(path, std_path):
                        if p >= p0 + 3.0 * std:
                            magnitude = p - p0
                            break
                        elif p <= p0 - 3.0 * std:
                            magnitude = p - p0
                            break
                    if magnitude == 0.0:
                        magnitude = path[-1] - p0
                events.append({
                    'idx': i,
                    'p0': p0,
                    'mode': mode,
                    'magnitude': magnitude,
                    'max_std': np.max(std_path)
                })
                if len(events) >= 3:
                    break
                cooldown = 60
                
        if len(events) > 0:
            print("\n--- TRACE BEFORE FIX ---")
            for e in events:
                print(f"Event at index {e['idx']} (Mode: {e['mode']}): p0 = {e['p0']:.2f}, magnitude = {e['magnitude']:.2f}, max_sigma in path = {e['max_std']:.2f}")
            break
