import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs('reports/eda', exist_ok=True)
    parquet_path = 'C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/reports/findings/trade_paths.parquet'
    
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} trades.")
    
    # 1. Feature Engineering from raw paths
    print("Extracting physical features from paths...")
    durations = []
    mfes = []
    maes = []
    time_to_mfe = []
    time_to_mae = []
    
    for path in df['path']:
        if len(path) == 0:
            durations.append(0)
            mfes.append(0)
            maes.append(0)
            time_to_mfe.append(0)
            time_to_mae.append(0)
            continue
            
        durations.append(len(path))
        mfe_idx = np.argmax(path)
        mae_idx = np.argmin(path)
        mfes.append(path[mfe_idx])
        maes.append(path[mae_idx])
        time_to_mfe.append(mfe_idx)
        time_to_mae.append(mae_idx)
        
    df['duration_sec'] = durations
    df['mfe_pts'] = mfes
    df['mae_pts'] = maes
    df['time_to_mfe'] = time_to_mfe
    df['time_to_mae'] = time_to_mae
    
    # Drop empty paths
    df = df[df['duration_sec'] > 0].copy()
    
    # Categorize Trades
    def categorize(row):
        if row['net_usd'] > 0:
            if row['mfe_pts'] >= 40:
                return 'Big Winner'
            else:
                return 'Small Winner'
        else:
            if row['mae_pts'] <= -45:
                return 'Stopped Out'
            else:
                return 'Chopped Out'
                
    df['category'] = df.apply(categorize, axis=1)
    
    cat_counts = df['category'].value_counts()
    print("Trade Categories:")
    print(cat_counts)
    
    # Plot 1: Scatter of Duration vs MFE colored by Category
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='duration_sec', y='mfe_pts', hue='category', alpha=0.6, s=15, 
                    palette={'Big Winner': 'green', 'Small Winner': 'lightgreen', 'Stopped Out': 'red', 'Chopped Out': 'orange'})
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Trade Duration vs MFE (Points)")
    plt.xlabel("Duration (Seconds)")
    plt.ylabel("Maximum Favorable Excursion (Points)")
    plt.tight_layout()
    plt.savefig('reports/eda/scatter_duration_mfe.png', dpi=150)
    plt.close()
    
    # Plot 2: Time to MFE Density
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['category'] == 'Big Winner'], x='time_to_mfe', color='green', label='Big Winner', fill=True, alpha=0.3)
    sns.kdeplot(data=df[df['category'] == 'Stopped Out'], x='time_to_mae', color='red', label='Stopped Out (Time to MAE)', fill=True, alpha=0.3)
    plt.title("Density: Time to Peak (MFE for Winners, MAE for Losers)")
    plt.xlabel("Seconds from Entry")
    plt.xlim(0, 5000)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/eda/density_time_to_peak.png', dpi=150)
    plt.close()
    
    # Plot 3: Average Path Trajectories
    # We will interpolate all paths to a standard length (e.g. 100 steps) to find the "average shape"
    print("Computing average trajectory shapes...")
    steps = 100
    
    def interpolate_path(path):
        if len(path) < 2: return np.zeros(steps)
        x = np.linspace(0, 1, len(path))
        x_new = np.linspace(0, 1, steps)
        return np.interp(x_new, x, path)
        
    df['interp_path'] = df['path'].apply(interpolate_path)
    
    plt.figure(figsize=(12, 7))
    for cat, color in [('Big Winner', 'green'), ('Chopped Out', 'orange'), ('Stopped Out', 'red')]:
        cat_df = df[df['category'] == cat]
        if len(cat_df) == 0: continue
        
        # Matrix of paths
        mat = np.vstack(cat_df['interp_path'].values)
        mean_path = np.mean(mat, axis=0)
        p25 = np.percentile(mat, 25, axis=0)
        p75 = np.percentile(mat, 75, axis=0)
        
        x_pct = np.linspace(0, 100, steps)
        plt.plot(x_pct, mean_path, color=color, label=f'{cat} (Mean)', linewidth=3)
        plt.fill_between(x_pct, p25, p75, color=color, alpha=0.15)
        
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Normalized Trade Lifecycle (Average Path shapes)")
    plt.xlabel("% of Trade Duration")
    plt.ylabel("Points relative to Entry")
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/eda/avg_lifecycle_paths.png', dpi=150)
    plt.close()

    # Plot 4: The 71-Point MFE Archetype
    # Let's align all "Big Winners" by their MFE point instead of stretching them.
    # We will take [-1000, +1000] seconds around the MFE.
    print("Aligning Big Winners by MFE...")
    aligned_paths = []
    window = 600 # 10 mins before and after MFE
    
    for _, row in df[df['category'] == 'Big Winner'].iterrows():
        path = row['path']
        mfe_idx = row['time_to_mfe']
        
        # Extract [-600, +600]
        start = mfe_idx - window
        end = mfe_idx + window
        
        seg = np.full(window*2, np.nan)
        
        p_start = max(0, start)
        p_end = min(len(path), end)
        
        s_start = max(0, -start)
        s_end = window*2 - max(0, end - len(path))
        
        if s_end > s_start and p_end > p_start:
            seg[s_start:s_end] = path[p_start:p_end]
            # Normalize so MFE is at 0 (or we just leave it in relative entry points)
            aligned_paths.append(seg)
            
    if len(aligned_paths) > 0:
        mat_mfe = np.vstack(aligned_paths)
        mean_mfe = np.nanmean(mat_mfe, axis=0)
        p25_mfe = np.nanpercentile(mat_mfe, 25, axis=0)
        p75_mfe = np.nanpercentile(mat_mfe, 75, axis=0)
        
        plt.figure(figsize=(12, 7))
        x_sec = np.arange(-window, window)
        plt.plot(x_sec, mean_mfe, color='darkgreen', linewidth=3, label='Mean Path around MFE')
        plt.fill_between(x_sec, p25_mfe, p75_mfe, color='green', alpha=0.2)
        plt.axvline(0, color='black', linestyle='--', label='Peak MFE Moment')
        plt.title("Anatomy of a Big Winner (Aligned at Peak MFE)")
        plt.xlabel("Seconds relative to Peak MFE")
        plt.ylabel("Points relative to Entry")
        plt.legend()
        plt.tight_layout()
        plt.savefig('reports/eda/anatomy_of_mfe.png', dpi=150)
        plt.close()
        
    # Write summary stats
    summary = df.groupby('category')[['duration_sec', 'mfe_pts', 'mae_pts', 'time_to_mfe', 'time_to_mae']].median().round(2)
    summary.to_csv('reports/eda/category_medians.csv')
    print("EDA Generation Complete!")

if __name__ == '__main__':
    main()
