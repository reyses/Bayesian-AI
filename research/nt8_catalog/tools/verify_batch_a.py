import os
import sys
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from core_v2.FPS.forward_pass_system import ForwardPassSystem
from batch_a_detectors import (
    ORB02Detector, SEASON12Detector, RENKO24Detector, 
    VWAP03Detector, OHLC01Detector, PIVOT16Detector, ROUND05Detector
)

def rth_ts(day_fmt):
    """RTH 5s bar timestamps for a day to map event_idx back to timestamps."""
    p = os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{day_fmt}.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['timestamp'])
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    m = (dt.dt.time >= pd.Timestamp('08:30').time()) & (dt.dt.time <= pd.Timestamp('15:15').time())
    return df['timestamp'].values[m.values].astype(np.int64)

def load_prior_ohlc(prior_day):
    p = os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{prior_day}.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['close', 'timestamp'])
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    m = (dt.dt.time >= pd.Timestamp('08:30').time()) & (dt.dt.time <= pd.Timestamp('15:15').time())
    df_rth = df[m]
    return {
        'high': float(df['close'].max()),
        'low': float(df['close'].min()),
        'close': float(df['close'].iloc[-1]),
        'rth_close': float(df_rth['close'].iloc[-1]) if len(df_rth) > 0 else float(df['close'].iloc[-1])
    }

def verify_day(day, prior_day):
    print(f"\n--- Verifying {day} ---")
    ts_map = rth_ts(day)
    if ts_map is None:
        print("No 5s data for day.")
        return
        
    yest_ohlc = load_prior_ohlc(prior_day)
    if yest_ohlc is None:
        print("No prior day ohlc.")
        return
        
    pdh, pdl, pdc = yest_ohlc['high'], yest_ohlc['low'], yest_ohlc['close']
    pdc_rth = yest_ohlc['rth_close']
    
    detectors = {
        'ORB-02': ORB02Detector(),
        'SEASON-12': SEASON12Detector(pdc_rth),
        'RENKO-24': RENKO24Detector(),
        'VWAP-03': VWAP03Detector(),
        'OHLC-01': OHLC01Detector(pdh, pdl, pdc),
        'PIVOT-16': PIVOT16Detector(pdh, pdl, pdc),
        'ROUND-05': ROUND05Detector()
    }
    
    dossier_names = {
        'ORB-02': 'ORB-02_Opening_Range',
        'SEASON-12': 'SEASON-12_DayOfWeek',
        'RENKO-24': 'RENKO-24_Time_Filtering',
        'VWAP-03': 'VWAP-03_Session_VWAP',
        'OHLC-01': 'OHLC-01_Prior_Day',
        'PIVOT-16': 'PIVOT-16_Floor_Levels',
        'ROUND-05': 'ROUND-05_Psych_Numbers'
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
            if name == 'ORB-02':
                idx += 360 # offset 08:30 to 09:00
            
            if name == 'RENKO-24':
                ts = 0 # Brick indices are unmappable
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
    verify_day('2024_03_04', '2024_03_01')
    verify_day('2024_03_05', '2024_03_04')
    verify_day('2024_03_06', '2024_03_05')
