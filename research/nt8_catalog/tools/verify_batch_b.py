import os
import sys
out_file = open("verifier_output.txt", "w", encoding="utf-8")
sys.stdout = out_file
import sys
import glob
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from core_v2.FPS.forward_pass_system import ForwardPassSystem
from batch_b_detectors import (
    ADX08Detector, ATR09Detector, CROSS11Detector, DOW19Detector, FIB17Detector
)

def rth_ts(day_fmt):
    """RTH 5s bar timestamps for a day to map event_idx back to timestamps."""
    p = os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{day_fmt}.parquet')
    if not os.path.exists(p): return None
    df = pd.read_parquet(p, columns=['timestamp'])
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    m = (dt.dt.time >= pd.Timestamp('08:30').time()) & (dt.dt.time <= pd.Timestamp('15:15').time())
    return df['timestamp'].values[m.values].astype(np.int64)

def compute_adx(highs, lows, closes, n=14):
    if len(highs) < 2: return 0.0
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    df['upMove'] = df['high'] - df['high'].shift(1)
    df['downMove'] = df['low'].shift(1) - df['low']
    df['+DM'] = np.where((df['upMove'] > df['downMove']) & (df['upMove'] > 0), df['upMove'], 0.0)
    df['-DM'] = np.where((df['downMove'] > df['upMove']) & (df['downMove'] > 0), df['downMove'], 0.0)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/n, min_periods=n, adjust=False).mean() / (df['TR'].ewm(alpha=1/n, min_periods=n, adjust=False).mean() + 1e-10))
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/n, min_periods=n, adjust=False).mean() / (df['TR'].ewm(alpha=1/n, min_periods=n, adjust=False).mean() + 1e-10))
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10))
    adx = df['DX'].ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return adx.iloc[-1]

def build_daily_context():
    l0_dir = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    daily_data = {}
    valid_days = []
    
    print("Building daily context from parquets...")
    for f in files:
        if '2024' not in f and '2025' not in f: continue
        day = os.path.basename(f).replace('.parquet', '')
        try:
            df = pd.read_parquet(f, columns=['timestamp', 'close', 'high', 'low'])
            df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
            df_rth = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())]
            if len(df_rth) < 100: continue
            
            daily_data[day] = {
                'high': float(df_rth['high'].max()),
                'low': float(df_rth['low'].min()),
                'close': float(df_rth['close'].iloc[-1])
            }
            valid_days.append(day)
        except Exception:
            pass
    return daily_data, valid_days

def verify_day(day, daily_data, valid_days):
    print(f"\n--- Verifying {day} ---")
    ts_map = rth_ts(day)
    if ts_map is None:
        print("No 5s data for day.")
        return
        
    try:
        idx = valid_days.index(day)
    except ValueError:
        print("Day not in valid_days.")
        return
        
    if idx < 15:
        print("Not enough history for this day.")
        return
        
    # --- ATR-09 Context ---
    window_14_atr = valid_days[idx-15:idx] # matching script logic
    trs = []
    for j in range(1, 15):
        curr = daily_data[window_14_atr[j]]
        prev = daily_data[window_14_atr[j-1]]
        tr1 = curr['high'] - curr['low']
        tr2 = abs(curr['high'] - prev['close'])
        tr3 = abs(curr['low'] - prev['close'])
        trs.append(max(tr1, tr2, tr3))
    daily_atr = np.mean(trs)
    
    # --- CROSS-11 Context ---
    prior_day = valid_days[idx-1]
    p_prior = os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{prior_day}.parquet')
    df_yest = pd.read_parquet(p_prior, columns=['close'])
    prefill_closes = df_yest['close'].values.tolist()
    
    p_today = os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{day}.parquet')
    df_today_full = pd.read_parquet(p_today, columns=['timestamp', 'close'])
    df_today_full['dt'] = pd.to_datetime(df_today_full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_today_eth = df_today_full[df_today_full['dt'].dt.time < pd.Timestamp('08:30').time()]
    prefill_closes.extend(df_today_eth['close'].values.tolist())
    
    # --- FIB-17 Context ---
    window_14_fib = valid_days[idx-14:idx]
    highs = [daily_data[d]['high'] for d in window_14_fib]
    lows = [daily_data[d]['low'] for d in window_14_fib]
    closes = [daily_data[d]['close'] for d in window_14_fib]
    adx_val = compute_adx(highs, lows, closes, n=7)
    
    window_10 = valid_days[idx-10:idx]
    swing_high = max([daily_data[d]['high'] for d in window_10])
    swing_low = min([daily_data[d]['low'] for d in window_10])
    sma_10 = np.mean([daily_data[d]['close'] for d in window_10])
    last_close = daily_data[valid_days[idx-1]]['close']
    trend = 'UP' if last_close > sma_10 else 'DOWN'
    
    if trend == 'UP':
        range_val = swing_high - swing_low
        fib_50 = swing_high - (range_val * 0.50)
        fib_618 = swing_high - (range_val * 0.618)
    else:
        range_val = swing_high - swing_low
        fib_50 = swing_low + (range_val * 0.50)
        fib_618 = swing_low + (range_val * 0.618)
        
    detectors = {
        'ADX-08': ADX08Detector(),
        'ATR-09': ATR09Detector(daily_atr=daily_atr),
        'CROSS-11': CROSS11Detector(prefill_closes=prefill_closes),
        'DOW-19': DOW19Detector(),
        'FIB-17': FIB17Detector(fib_50=fib_50, fib_618=fib_618, adx_val=adx_val, trend=trend)
    }
    
    dossier_names = {
        'ADX-08': 'ADX-08_Trend_Gate',
        'ATR-09': 'ATR-09_Statistical_Fade',
        'CROSS-11': 'CROSS-11_Golden_Cross',
        'DOW-19': 'DOW-19_Price_Volume_Divergence',
        'FIB-17': 'FIB-17_Confluence'
    }
    
    triggers = {k: [] for k in detectors}
    
    print("Running FPS...")
    fps = ForwardPassSystem(day=day, atlas_root=os.path.join(ROOT, 'DATA', 'ATLAS'),
                            features_root=os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_5s_v2'),
                            labels_csv=os.path.join(ROOT, 'DATA', 'ATLAS', 'regime_labels_2d.csv'),
                            tfs=['5s'], layers=['L1'],
                            build_v2_dict=False, use_5s_price=True)
                            
    for st in fps:
        for name, det in detectors.items():
            setup, mode = det.on_bar(st)
            if setup != 0:
                triggers[name].append({
                    'timestamp': int(st.ohlcv_5s['timestamp']),
                    'setup': setup,
                    'mode': mode
                })
                
    for name in detectors:
        legacy_path = os.path.join(HERE, '..', 'tests', dossier_names[name], 'events.parquet')
        if not os.path.exists(legacy_path):
            print(f"{name}: No legacy events file found.")
            continue
            
        legacy_df = pd.read_parquet(legacy_path)
        legacy_day = legacy_df[legacy_df['day'] == day]
        
        legacy_events = []
        for _, row in legacy_day.iterrows():
            idx = int(row['event_idx'])
            if name == 'CROSS-11':
                # the cross script concatenated prior day data and THEN filtered RTH.
                # we just use event_idx mapping, it maps directly to RTH index of the *day*
                if idx < len(ts_map):
                    ts = int(ts_map[idx])
                else:
                    ts = 0
            elif idx < len(ts_map):
                ts = int(ts_map[idx])
            else:
                ts = 0
                
            legacy_events.append({
                'timestamp': ts,
                'setup': row['setup'],
                'mode': row['mode']
            })
                
        print(f"{name}:")
        print(f"  Native triggers: {len(triggers[name])}")
        print(f"  Legacy triggers: {len(legacy_events)}")
        
        if len(triggers[name]) > 0:
            print(f"  First native: {triggers[name][0]}")
        if len(legacy_events) > 0:
            print(f"  First legacy: {legacy_events[0]}")

if __name__ == '__main__':
    daily_data, valid_days = build_daily_context()
    days_to_test = valid_days[1:]
    for d in days_to_test:
        verify_day(d, daily_data, valid_days)
