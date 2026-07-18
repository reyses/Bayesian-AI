import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats

# We need the exact _ols_fit_kernel from core_v2 to compute the trailing z-scores
from core_v2.statistical_field_engine import _ols_fit_kernel

def process_day(file_path):
    # Load 1s data
    df_1s = pd.read_parquet(file_path)
    if len(df_1s) == 0:
        return None
        
    df_1s = df_1s.sort_values('timestamp')
    df_1s['bar_ts'] = (df_1s['timestamp'] // 60) * 60
    
    # Fast OLS computation per 1m group using numpy
    # We want OLS intercept at the end of the bar (t = bar_ts + 59)
    # x = timestamp - (bar_ts + 59)
    
    # Calculate group properties
    def calc_ldist(group):
        if len(group) < 2:
            return pd.Series({'close': group['close'].iloc[-1], 'ldist_level': group['close'].iloc[-1]})
            
        y = group['close'].values
        x = group['timestamp'].values - (group['bar_ts'].iloc[0] + 59)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_var = np.sum((x - x_mean)**2)
        
        if x_var < 1e-12:
            ldist_level = y[-1]
        else:
            slope = np.sum((x - x_mean)*(y - y_mean)) / x_var
            ldist_level = y_mean - slope * x_mean
            
        return pd.Series({'close': y[-1], 'ldist_level': ldist_level})
        
    df_1m = df_1s.groupby('bar_ts').apply(calc_ldist).reset_index()
    df_1m = df_1m.sort_values('bar_ts')
    
    # Compute z-scores using the exact core_v2 logic (N=15 for 1m bars)
    N = 15
    close_vals = df_1m['close'].values.astype(np.float64)
    ldist_vals = df_1m['ldist_level'].values.astype(np.float64)
    
    rm_close, se_close = _ols_fit_kernel(close_vals, N)
    rm_level, se_level = _ols_fit_kernel(ldist_vals, N)
    
    z_close = np.full(len(df_1m), np.nan)
    mask_c = se_close > 1e-10
    z_close[mask_c] = (close_vals[mask_c] - rm_close[mask_c]) / se_close[mask_c]
    
    z_level = np.full(len(df_1m), np.nan)
    mask_l = se_level > 1e-10
    z_level[mask_l] = (ldist_vals[mask_l] - rm_level[mask_l]) / se_level[mask_l]
    
    df_1m['z_close'] = z_close
    df_1m['z_level'] = z_level
    
    # Forward 5m return
    df_1m['forward_5m'] = df_1m['close'].shift(-5) - df_1m['close']
    
    # Tag day for block-bootstrapping later
    day_str = os.path.basename(file_path).replace('.parquet', '')
    df_1m['day'] = day_str
    
    return df_1m

def main():
    print("Finding March 2024 1s data...")
    files = sorted(glob.glob('DATA/ATLAS/1s/2024_03_*.parquet'))
    print(f"Found {len(files)} trading days.")
    if len(files) == 0:
        print("No files found. Run ingestion first.")
        return
        
    all_days = []
    for f in files:
        print(f"Processing {os.path.basename(f)}...")
        res = process_day(f)
        if res is not None:
            all_days.append(res)
            
    df = pd.concat(all_days, ignore_index=True)
    
    # Filter out NaNs
    df = df.dropna(subset=['z_close', 'z_level', 'forward_5m'])
    
    # Daily separation analysis
    daily_stats = []
    
    for day, group in df.groupby('day'):
        if len(group) < 50: # Skip very short days
            continue
            
        y = group['forward_5m'].values
        
        # We look at |z| separating forward volatility/magnitude, or
        # signed z predicting signed forward return. The prompt asked for:
        # "see which |z| better predicts the next K=5 bars' move (Spearman + directional sign)"
        # Actually, |z| predicting |forward_move| is volatility prediction.
        # But |z| predicting forward move? That doesn't make sense (magnitudes don't predict signs).
        # It's likely `abs(z)` predicting `abs(forward_move)` OR `z` predicting `forward_move`.
        # I will calculate both:
        # 1. Spearman(z, forward_5m)
        # 2. Sign Accuracy: sign(z) == sign(forward_5m)
        
        z_c = group['z_close'].values
        z_l = group['z_level'].values
        
        rho_c, _ = spearmanr(z_c, y)
        rho_l, _ = spearmanr(z_l, y)
        
        # Directional sign accuracy (ignoring strict zeros)
        valid_c = (z_c != 0) & (y != 0)
        acc_c = np.mean(np.sign(z_c[valid_c]) == np.sign(y[valid_c])) if np.sum(valid_c) > 0 else 0
        
        valid_l = (z_l != 0) & (y != 0)
        acc_l = np.mean(np.sign(z_l[valid_l]) == np.sign(y[valid_l])) if np.sum(valid_l) > 0 else 0
        
        daily_stats.append({
            'day': day,
            'n_bars': len(group),
            'rho_close': rho_c,
            'rho_level': rho_l,
            'acc_close': acc_c,
            'acc_level': acc_l,
            'delta_rho': rho_l - rho_c,
            'delta_acc': acc_l - acc_c
        })
        
    df_stats = pd.DataFrame(daily_stats)
    
    # 95% CI on Delta
    # We use Student's t-test over the N independent daily blocks
    n_days = len(df_stats)
    t_crit = stats.t.ppf(0.975, df=n_days-1)
    
    mean_delta_rho = df_stats['delta_rho'].mean()
    se_delta_rho = df_stats['delta_rho'].std() / np.sqrt(n_days)
    ci_rho = (mean_delta_rho - t_crit * se_delta_rho, mean_delta_rho + t_crit * se_delta_rho)
    
    mean_delta_acc = df_stats['delta_acc'].mean()
    se_delta_acc = df_stats['delta_acc'].std() / np.sqrt(n_days)
    ci_acc = (mean_delta_acc - t_crit * se_delta_acc, mean_delta_acc + t_crit * se_delta_acc)
    
    print("\n" + "="*50)
    print("WEDGE TEST RESULTS: z_level vs z_close")
    print("="*50)
    print(f"Trading Days: {n_days}")
    print(f"Total Bars:   {df_stats['n_bars'].sum()}")
    print("-" * 50)
    print("1. Spearman Correlation (z vs forward_5m)")
    print(f"   z_close mean rho: {df_stats['rho_close'].mean():.4f}")
    print(f"   z_level mean rho: {df_stats['rho_level'].mean():.4f}")
    print(f"   DELTA:            {mean_delta_rho:.4f}")
    print(f"   95% CI on DELTA:  [{ci_rho[0]:.4f}, {ci_rho[1]:.4f}]")
    print("-" * 50)
    print("2. Directional Accuracy (sign(z) == sign(forward_5m))")
    print(f"   z_close mean acc: {df_stats['acc_close'].mean()*100:.2f}%")
    print(f"   z_level mean acc: {df_stats['acc_level'].mean()*100:.2f}%")
    print(f"   DELTA:            {mean_delta_acc*100:.2f}%")
    print(f"   95% CI on DELTA:  [{ci_acc[0]*100:.2f}%, {ci_acc[1]*100:.2f}%]")
    print("="*50)
    
    # Conclusion
    if ci_rho[0] > 0 and ci_acc[0] > 0:
        print("VERDICT: GO. ldist_level cleanly separates better than raw close.")
    elif ci_rho[1] < 0 and ci_acc[1] < 0:
        print("VERDICT: NO-GO. ldist_level is statistically worse than raw close.")
    else:
        print("VERDICT: INCONCLUSIVE. The CI for the delta includes zero.")

if __name__ == "__main__":
    main()
