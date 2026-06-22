import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
from core_v2.statistical_field_engine import StatisticalFieldEngine

def evaluate_day(day: str, wins: list):
    fpath = f'DATA/ATLAS/1s/{day}.parquet'
    if not os.path.exists(fpath):
        return None
        
    b = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
    ts = b['timestamp'].values
    sfe = StatisticalFieldEngine()
    px = b['close'].values
    
    def ix(df):
        df = df.reset_index(drop=True).copy()
        df.index = ts[:len(df)]
        return df

    def clean(df):
        return df[[c for c in df.columns if 'vwap' not in c]]

    res = []
    
    for N in wins:
        # Extract RAW features for scale N (No orthogonalization needed for marginal surface)
        l2 = clean(ix(sfe.compute_L2(b, tf='1s', N=N)))
        l3 = clean(ix(sfe.compute_L3(b, tf='1s', N=N)))
        fn = pd.concat([l2, l3], axis=1).replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # Target A: Contemporaneous price-delta (Diagnostic)
        yA = pd.Series(px - pd.Series(px).shift(N).values, index=ts)
        
        # Target B: Causal forward price-delta (Tradeable)
        yB = pd.Series(pd.Series(px).shift(-N).values - px, index=ts)
        
        # Align
        df_A = fn.join(yA.rename('yA')).dropna()
        df_B = fn.join(yB.rename('yB')).dropna()
        
        if len(df_A) < 100 or len(df_B) < 100:
            continue
            
        Xs_A = StandardScaler().fit_transform(df_A.drop(columns='yA').values)
        yv_A = df_A['yA'].values
        
        Xs_B = StandardScaler().fit_transform(df_B.drop(columns='yB').values)
        yv_B = df_B['yB'].values
        
        r2_A = cross_val_score(RidgeCV(alphas=[0.1, 1, 10, 100]), Xs_A, yv_A, cv=5, scoring='r2').mean()
        r2_B = cross_val_score(RidgeCV(alphas=[0.1, 1, 10, 100]), Xs_B, yv_B, cv=5, scoring='r2').mean()
        
        res.append({'day': day, 'N': N, 'R2_Contemp': r2_A, 'R2_Forward': r2_B})

    return pd.DataFrame(res)

if __name__ == '__main__':
    wins = [60, 120, 240, 480, 960, 1920, 3840]
    
    if len(sys.argv) > 1 and sys.argv[1] == 'is_oos':
        blocks = {
            'IS_H1_2024': ['2024_02_20', '2024_02_21', '2024_02_22', '2024_02_23', '2024_02_26',
                           '2024_02_27', '2024_02_28', '2024_02_29', '2024_03_01', '2024_03_04'],
            'OOS1_H1_2025': ['2025_02_20', '2025_02_21', '2025_02_24', '2025_02_25', '2025_02_26',
                             '2025_02_27', '2025_02_28', '2025_03_03', '2025_03_04', '2025_03_05'],
            'OOS2_H2_2025': ['2025_08_20', '2025_08_21', '2025_08_22', '2025_08_25', '2025_08_26',
                             '2025_08_27', '2025_08_28', '2025_08_29', '2025_09_02', '2025_09_03'],
            'OOS3_2026': ['2026_02_20', '2026_02_23', '2026_02_24', '2026_02_25', '2026_02_26',
                          '2026_02_27', '2026_03_02', '2026_03_03', '2026_03_04', '2026_03_05']
        }
    else:
        blocks = {'PILOT': ['2024_02_20', '2024_02_21', '2024_02_22']}
        
    all_res = []
    
    for block_name, days in blocks.items():
        print(f"\n=== Running Block: {block_name} ===")
        for d in tqdm(days):
            res_df = evaluate_day(d, wins)
            if res_df is not None:
                res_df['block'] = block_name
                all_res.append(res_df)
                
    if not all_res:
        print("No valid days found.")
        sys.exit(1)
        
    final_df = pd.concat(all_res, ignore_index=True)
    os.makedirs('reports/findings', exist_ok=True)
    final_df.to_csv('reports/findings/1s_response_surface_raw.csv', index=False)
    
    # Generate overlaid surface plots
    # 1. Surface A (Contemporaneous)
    plt.figure(figsize=(10, 6))
    for b in final_df['block'].unique():
        b_df = final_df[final_df['block'] == b].groupby('N')['R2_Contemp'].mean().reset_index()
        ls = '-' if 'IS' in b or 'PILOT' in b else '--'
        plt.plot(b_df['N'], b_df['R2_Contemp'], marker='o', linestyle=ls, label=b)
    plt.xscale('log')
    plt.title('Surface A: Contemporaneous Fit (Diagnostic) R2 vs Horizon')
    plt.xlabel('Horizon (seconds)')
    plt.ylabel('R2 (Ridge 5-CV)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/findings/1s_surfaceA_contemp.png')
    
    # 2. Surface B (Forward)
    plt.figure(figsize=(10, 6))
    for b in final_df['block'].unique():
        b_df = final_df[final_df['block'] == b].groupby('N')['R2_Forward'].mean().reset_index()
        ls = '-' if 'IS' in b or 'PILOT' in b else '--'
        plt.plot(b_df['N'], b_df['R2_Forward'], marker='o', linestyle=ls, label=b)
    plt.xscale('log')
    plt.title('Surface B: Causal Forward Fit (Tradeable) R2 vs Horizon')
    plt.xlabel('Horizon (seconds)')
    plt.ylabel('R2 (Ridge 5-CV)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/findings/1s_surfaceB_forward.png')
    
    print("\nSurfaces saved to reports/findings/1s_surface*.png")
