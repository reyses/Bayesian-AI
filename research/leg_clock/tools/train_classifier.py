"""Phase 4: Final Classifier Assembly
Trains a Logistic Regression and MLP on the Top SFE features extracted from 2024,
and evaluates Out-Of-Sample on 2025.
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

sys.path.append(os.path.join(_REPO))
sys.path.append(os.path.join(_REPO, 'core_v2'))
from statistical_field_engine import StatisticalFieldEngine

def load_atlas(day, tf='1m'):
    path = os.path.join(_REPO, 'DATA', 'ATLAS', tf, f"{day}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def extract_dataset(year='2024'):
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', f'ai_picks_{year}-*_multi.json')))
    sfe = StatisticalFieldEngine()
    features = []
    
    # We use the top 10 features discovered in feat_006
    target_cols = [
        'L3_1m_z_high_30',
        'L3_1m_z_low_30',
        'L3_1m_z_se_30',
        'L3_1m_band_pos_30',
        'L3_1m_z_close_vs_high_30',
        'L3_1m_z_close_vs_low_30',
        'L2_1m_price_velocity_30',
        'L2_1m_vol_velocity_30',
        'L1_1m_vol_velocity_1b',
        'L1_1m_price_accel_1b'
    ]
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        try:
            d = json.load(open(f))
        except Exception:
            continue
            
        trades = d.get('trades', [])
        if not trades: continue
        
        df_L0 = sfe.compute_L0(df)
        df_L1 = sfe.compute_L1(df, '1m')
        df_L2 = sfe.compute_L2(df, '1m', N=30)
        df_L3 = sfe.compute_L3(df, '1m', N=30)
        df_sfe = pd.concat([df_L0, df_L1, df_L2, df_L3], axis=1)
        
        # Only keep target columns to save memory
        available_cols = [c for c in target_cols if c in df_sfe.columns]
        df_sfe = df_sfe[available_cols]
        
        ts = df['timestamp'].values
        hours = np.array([datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).hour for t in trades])
        df_hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts])
        
        for t, h in zip(trades, hours):
            entry_ts = t['entry_ts']
            is_long = str(t.get('direction', '')).upper().startswith('L')
            
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            if idx < 60: continue
            
            row_data = df_sfe.iloc[idx].to_dict()
            row_data['is_label'] = 1
            row_data['is_long'] = int(is_long)
            features.append(row_data)
            
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts) & (np.arange(len(ts)) >= 60))[0]
            null_candidates = [c for c in null_candidates if abs(c - idx) > 5]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                null_row = df_sfe.iloc[null_idx].to_dict()
                null_row['is_label'] = 0
                null_row['is_long'] = int(is_long)
                features.append(null_row)

    return pd.DataFrame(features)

def main():
    print("Extracting 2024 TRAIN dataset...")
    df_train = extract_dataset('2024')
    print(f"TRAIN Set: {len(df_train)} samples")
    
    print("Extracting 2025 TEST dataset...")
    df_test = extract_dataset('2025')
    print(f"TEST Set: {len(df_test)} samples")
    
    if len(df_train) == 0 or len(df_test) == 0:
        print("Missing data.")
        return

    # Prepare features
    feat_cols = [c for c in df_train.columns if c not in ['is_label', 'is_long', 'timestamp']]
    
    # Auto-align features on the train set
    alignments = {}
    for c in feat_cols:
        v_raw = df_train[c].values
        v_align = v_raw * np.where(df_train['is_long'] == 1, 1.0, -1.0)
        
        valid = ~np.isnan(v_raw)
        if valid.sum() == 0: continue
        
        try:
            auc_raw = roc_auc_score(df_train['is_label'][valid], v_raw[valid])
            if auc_raw < 0.5: auc_raw = 1 - auc_raw
            auc_align = roc_auc_score(df_train['is_label'][valid], v_align[valid])
            if auc_align < 0.5: auc_align = 1 - auc_align
        except:
            continue
            
        alignments[c] = auc_align > auc_raw

    # Build X, y matrices
    def build_xy(df):
        df_clean = df.dropna(subset=feat_cols)
        X = []
        for c in feat_cols:
            vals = df_clean[c].values
            if alignments.get(c, False):
                vals = vals * np.where(df_clean['is_long'] == 1, 1.0, -1.0)
            X.append(vals)
        return np.column_stack(X), df_clean['is_label'].values
        
    X_train, y_train = build_xy(df_train)
    X_test, y_test = build_xy(df_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n=== Model 1: Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    
    train_probs_lr = lr.predict_proba(X_train)[:, 1]
    test_probs_lr = lr.predict_proba(X_test)[:, 1]
    
    print(f"Train AUC: {roc_auc_score(y_train, train_probs_lr):.4f}")
    print(f"Test AUC:  {roc_auc_score(y_test, test_probs_lr):.4f}")
    
    print("\nLR Coefficients:")
    coefs = list(zip(feat_cols, lr.coef_[0]))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    for c, v in coefs:
        align_str = "(Aligned)" if alignments.get(c) else "(Raw)"
        print(f"  {v:>7.3f} | {c} {align_str}")
        
    print("\n=== Model 2: MLP (Hidden 16x8) ===")
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    
    train_probs_mlp = mlp.predict_proba(X_train)[:, 1]
    test_probs_mlp = mlp.predict_proba(X_test)[:, 1]
    
    print(f"Train AUC: {roc_auc_score(y_train, train_probs_mlp):.4f}")
    print(f"Test AUC:  {roc_auc_score(y_test, test_probs_mlp):.4f}")
    
    print("\nFinal Phase Complete. OOS validated.")

if __name__ == '__main__':
    main()
