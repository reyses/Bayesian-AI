import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath('.'))
from core_v2.statistical_field_engine import StatisticalFieldEngine
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def orthogonalize_day(day: str, wins: list):
    print(f"\nProcessing {day}...")
    fpath = f'DATA/ATLAS/1s/{day}.parquet'
    if not os.path.exists(fpath):
        print(f"Skipping {day}, file not found.")
        return
    
    b = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
    ts = b['timestamp'].values
    sfe = StatisticalFieldEngine()
    
    def ix(df):
        df = df.reset_index(drop=True).copy()
        df.index = ts[:len(df)]
        return df

    # We drop vwap to avoid weird volume artifacts in the kinematics
    def clean(df):
        return df[[c for c in df.columns if 'vwap' not in c]]

    print("  Extracting L1...")
    l1 = clean(ix(sfe.compute_L1(b, tf='1s'))).replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    # Store scale blocks
    scale_blocks = {}
    scale_blocks['L1'] = l1
    
    for N in wins:
        print(f"  Extracting L2/L3 for N={N}...")
        l2 = clean(ix(sfe.compute_L2(b, tf='1s', N=N)))
        l3 = clean(ix(sfe.compute_L3(b, tf='1s', N=N)))
        fn = pd.concat([l2, l3], axis=1).replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        scale_blocks[N] = fn

    # Align all rows (drop any remaining NaNs)
    all_df = pd.concat(scale_blocks.values(), axis=1).dropna()
    ts_aligned = all_df.index
    
    # Gram-Schmidt Orthogonalization (Method A)
    print("  Gram-Schmidt Orthogonalization...")
    ortho_blocks = {}
    
    # Base L1
    X_L1 = scale_blocks['L1'].loc[ts_aligned].values
    X_L1_std = StandardScaler().fit_transform(X_L1)
    
    cols_L1 = scale_blocks['L1'].columns
    ortho_blocks['L1'] = pd.DataFrame(X_L1_std, columns=cols_L1, index=ts_aligned)
    
    B = X_L1_std # Current basis matrix
    
    cond_raw_list = [np.linalg.cond(X_L1_std)]
    
    raw_accum = [X_L1_std]
    
    for N in wins:
        F_N = scale_blocks[N].loc[ts_aligned].values
        F_N_std = StandardScaler().fit_transform(F_N)
        
        raw_accum.append(F_N_std)
        curr_raw = np.concatenate(raw_accum, axis=1)
        
        # Residualize F_N against B
        lr = LinearRegression(fit_intercept=False).fit(B, F_N_std)
        F_N_ortho = F_N_std - lr.predict(B)
        
        # Standardize the residuals to prevent numeric underflow in variance
        F_N_ortho_std = StandardScaler().fit_transform(F_N_ortho)
        
        cols_N = scale_blocks[N].columns
        ortho_blocks[N] = pd.DataFrame(F_N_ortho_std, columns=cols_N, index=ts_aligned)
        
        # Append to basis
        B = np.concatenate([B, F_N_ortho_std], axis=1)
        
    cond_raw = np.linalg.cond(np.concatenate(raw_accum, axis=1))
    cond_ortho = np.linalg.cond(B)
    
    print(f"  Cond# RAW:   {cond_raw:,.0f}")
    print(f"  Cond# ORTHO: {cond_ortho:,.0f}")
    
    # PCA Whitening check (Method B - Sanity Check)
    pca = PCA(whiten=True)
    pca_whitened = pca.fit_transform(np.concatenate(raw_accum, axis=1))
    cond_pca = np.linalg.cond(pca_whitened)
    print(f"  Cond# PCA-W: {cond_pca:,.0f}")
    
    # Save Gram-Schmidt Orthogonalized features
    os.makedirs('DATA/ATLAS/FEATURES_1s_GS', exist_ok=True)
    
    # Format columns to ensure they are unique and scale-labeled
    final_dfs = []
    for scale, df in ortho_blocks.items():
        if scale != 'L1':
            df.columns = [f"{c}_gs" for c in df.columns]
        final_dfs.append(df)
        
    final_out = pd.concat(final_dfs, axis=1)
    final_out.reset_index(names='timestamp').to_parquet(f'DATA/ATLAS/FEATURES_1s_GS/{day}.parquet')
    print(f"  Saved {final_out.shape[1]} orthogonal features to DATA/ATLAS/FEATURES_1s_GS/{day}.parquet")

if __name__ == '__main__':
    wins = [60, 120, 240, 480, 960, 1920, 3840]
    
    # Pilot run days
    days = ['2024_02_20', '2024_02_21', '2024_02_22']
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'is_oos':
            days = [
                # IS: 10 days H1-2024
                '2024_02_20', '2024_02_21', '2024_02_22', '2024_02_23', '2024_02_26',
                '2024_02_27', '2024_02_28', '2024_02_29', '2024_03_01', '2024_03_04',
                # OOS 1: 10 days H1-2025
                '2025_02_20', '2025_02_21', '2025_02_24', '2025_02_25', '2025_02_26',
                '2025_02_27', '2025_02_28', '2025_03_03', '2025_03_04', '2025_03_05',
                # OOS 2: 10 days H2-2025
                '2025_08_20', '2025_08_21', '2025_08_22', '2025_08_25', '2025_08_26',
                '2025_08_27', '2025_08_28', '2025_08_29', '2025_09_02', '2025_09_03',
                # OOS 3: 10 days 2026
                '2026_02_20', '2026_02_23', '2026_02_24', '2026_02_25', '2026_02_26',
                '2026_02_27', '2026_03_02', '2026_03_03', '2026_03_04', '2026_03_05'
            ]
        else:
            days = sys.argv[1].split(',')
            
    for d in days:
        orthogonalize_day(d, wins)
