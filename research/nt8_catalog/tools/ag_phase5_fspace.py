import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

def build_fspace():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    
    def get_tiers(is_1s):
        feat_dir = 'FEATURES_1s_v2' if is_1s else 'FEATURES_5s_v2'
        root = os.path.join(base_dir, 'DATA', 'ATLAS', feat_dir)
        return {
            'L0': os.path.join(root, 'L0'),
            'L1': os.path.join(root, 'L1_5s'),
            'L2': os.path.join(root, 'L2_5s'),
            'L3': os.path.join(root, 'L3_5s'),
            'L4': os.path.join(root, 'L4_5s'),
            'L5': os.path.join(root, 'L5_5s')
        }

    # We use a custom load function to just get the dataframe
    def load_tier(day, root):
        path = os.path.join(root, f"{day}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    target_dossiers = ['ATR-09_Statistical_Fade', 'FIB-17_Confluence', 'VA-13_Rotation', 'ORDERFLOW-14']
    
    for dossier in target_dossiers:
        nt8_tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
        ev_path = os.path.join(nt8_tests_dir, dossier, 'events.parquet')
        if not os.path.exists(ev_path):
            print(f"Skipping {dossier}: No events.parquet")
            continue
            
        ev_df = pd.read_parquet(ev_path)
        days = sorted(ev_df['day'].unique())
        
        # We will collect the F-space matrices
        X_Phe = []
        X_PhXit = []
        X_PhPost = []
        Y = []
        Mags = []
        Years = []
        
        feature_cols_cache = {}
        
        is_1s = 'ORDERFLOW' in dossier
        tiers = get_tiers(is_1s)
        
        for day in days:
            day_fmt = day.replace('-', '_')
            
            # Load all tiers
            dfs = {}
            valid_day = True
            for t, root in tiers.items():
                dfs[t] = load_tier(day_fmt, root)
                if dfs[t] is None:
                    print(f"Day {day_fmt} missing tier {t} at {root}")
                    valid_day = False
                    break
            
            if not valid_day:
                continue
                
            day_events = ev_df[ev_df['day'] == day]
            
            for _, row in day_events.iterrows():
                e_idx = int(row['event_idx'])
                r_idx = int(row['resolution_idx'])
                dur = int(row['duration_bars'])
                
                idx_base_e = e_idx
                idx_base_r = r_idx
                idx_base_p = r_idx + dur
                    
                # 3 Anchors
                anchors = {
                    'PhE': idx_base_e,
                    'PhXit': idx_base_r,
                    'PhPost': idx_base_p
                }
                
                # Check bounds
                if anchors['PhPost'] >= len(dfs['L0']):
                    continue
                    
                valid_event = True
                anchor_vecs = {'PhE': [], 'PhXit': [], 'PhPost': []}
                
                for anchor_name, base_idx in anchors.items():
                    # Extract from all 6 tiers
                    tier_vec = []
                    for t in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']:
                        ti = base_idx
                        
                        df_t = dfs[t]
                        if ti >= len(df_t):
                            valid_event = False
                            break
                            
                        # Get numeric columns if not cached
                        if t not in feature_cols_cache:
                            # Skip 'timestamp', 'open', 'high', 'low', 'close', 'volume'
                            cols = [c for c in df_t.columns if c.startswith('L')]
                            if len(cols) == 0:
                                cols = [c for c in df_t.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                            feature_cols_cache[t] = cols
                            
                        vals = df_t.iloc[ti][feature_cols_cache[t]].values
                        # Fill NaNs with 0
                        vals = np.nan_to_num(vals, nan=0.0)
                        tier_vec.extend(vals)
                        
                    if not valid_event: break
                    anchor_vecs[anchor_name] = tier_vec
                    
                if valid_event:
                    X_Phe.append(anchor_vecs['PhE'])
                    X_PhXit.append(anchor_vecs['PhXit'])
                    X_PhPost.append(anchor_vecs['PhPost'])
                    Y.append(row['hit'])
                    Mags.append(row['magnitude'])
                    Years.append(day[:4])
                    
        print("="*40)
        print(f"Extraction for {dossier}:")
        if len(X_Phe) > 0:
            X_Phe_np = np.array(X_Phe)
            X_PhXit_np = np.array(X_PhXit)
            X_PhPost_np = np.array(X_PhPost)
            print(f"PhE Shape: {X_Phe_np.shape}")
            print(f"PhXit Shape: {X_PhXit_np.shape}")
            print(f"PhPost Shape: {X_PhPost_np.shape}")
            
            # Combine into massive flat vector for this demonstration?
            # Or just save them for later?
            out_dir = os.path.join(nt8_tests_dir, dossier)
            np.save(os.path.join(out_dir, 'X_Phe.npy'), np.array(X_Phe))
            np.save(os.path.join(out_dir, 'X_PhXit.npy'), np.array(X_PhXit))
            np.save(os.path.join(out_dir, 'X_PhPost.npy'), np.array(X_PhPost))
            np.save(os.path.join(out_dir, 'Y.npy'), np.array(Y))
            np.save(os.path.join(out_dir, 'Mags.npy'), np.array(Mags))
            np.save(os.path.join(out_dir, 'Years.npy'), np.array(Years))
            
            print(f"Saved {len(Y)} samples for {dossier}")
        else:
            print("No valid events extracted.")

if __name__ == '__main__':
    build_fspace()
