import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def run_logistic_model(dossier_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    l0_dir = os.path.join(base_dir, '..', '..', 'DATA', 'ATLAS', '5s')
    
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    day_features = {}
    print("Loading 5s data for standard features...")
    for f in all_files:
        day = os.path.basename(f).replace('.parquet', '')
        try:
            df = pd.read_parquet(f, columns=['close', 'timestamp'])
        except Exception:
            continue
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        df_rth = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
        
        if len(df_rth) < 720:
            continue
            
        s = df_rth['close']
        net_change = s.diff(720).abs()
        sum_change = s.diff().abs().rolling(720).sum()
        er = net_change / sum_change
        vol = s.diff().abs().rolling(720).mean()
        
        day_features[day] = {
            'er': er.values,
            'vol': vol.values,
        }
        
    events_files = glob.glob(os.path.join(base_dir, 'tests', '**', 'events.parquet'), recursive=True)
    if dossier_path:
        events_files = [f for f in events_files if dossier_path in f]
        
    for ef in events_files:
        dossier = os.path.basename(os.path.dirname(ef))
        try:
            ev_df = pd.read_parquet(ef)
        except:
            continue
            
        if len(ev_df) == 0: continue
        
        er_list, vol_list, hour_list = [], [], []
        
        for _, row in ev_df.iterrows():
            day = row['day']
            idx = int(row['event_idx'])
            h = 8 + (30 + idx) // 60
            hour_list.append(h)
            
            idx_5s = idx * 12
            if day in day_features and idx_5s < len(day_features[day]['er']):
                e = day_features[day]['er'][idx_5s]
                v = day_features[day]['vol'][idx_5s]
            else:
                e = np.nan
                v = np.nan
            er_list.append(e)
            vol_list.append(v)
            
        ev_df['hour'] = hour_list
        ev_df['er'] = er_list
        ev_df['vol'] = vol_list
        ev_df = ev_df.dropna(subset=['er', 'vol', 'hour'])
        
        if len(ev_df) < 50:
            continue
            
        # Standardize features for logistic regression
        features = ['er', 'vol', 'hour']
        if 'depth' in ev_df.columns:
            ev_df['depth'] = ev_df['depth'].replace([np.inf, -np.inf], np.nan)
            ev_df = ev_df.dropna(subset=['depth'])
            if len(ev_df) > 50:
                features.append('depth')
                
        for f in features:
            ev_df[f] = (ev_df[f] - ev_df[f].mean()) / (ev_df[f].std() + 1e-9)
                
        X = ev_df[features]
        X = sm.add_constant(X)
        y = ev_df['hit']
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            print(f"\n--- LR Results for {dossier} ---")
            print(model.summary2().tables[1])
            
            out_pq = ef.replace('events.parquet', 'events_fspace.parquet')
            ev_df.to_parquet(out_pq)
            print(f"Saved F-space conditional data to {out_pq}")
        except Exception as e:
            print(f"LR Failed for {dossier}: {e}")

if __name__ == '__main__':
    run_logistic_model()
