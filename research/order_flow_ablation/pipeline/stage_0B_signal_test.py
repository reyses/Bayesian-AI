import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_stage_0B():
    print("Loading datasets...")
    delta_file = "DATA/ATLAS/order_flow_delta_5s.parquet"
    baseline_file = "DATA/ATLAS/baseline_features_416D.parquet"
    labels_file = "DATA/ATLAS/regime_labels_2d.csv"
    
    if not os.path.exists(delta_file) or not os.path.exists(baseline_file):
        print("Missing required datasets.")
        return
        
    delta_df = pd.read_parquet(delta_file)
    baseline_df = pd.read_parquet(baseline_file)
    
    # Timezone alignment
    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')
        
    print("Merging datasets...")
    merged = baseline_df.join(delta_df[['delta', 'volume', 'open']], how='inner', rsuffix='_trade')
    
    # Calculate components
    print("Calculating signal proxies...")
    # Using close from baseline and open from delta_df
    merged['price_delta'] = merged['close'] - merged['open']
    
    # Handle NaNs
    merged.dropna(subset=['price_delta', 'delta', 'volume'], inplace=True)
    
    merged['facsimile'] = np.sign(merged['price_delta']) * merged['volume']
    merged['true_delta'] = merged['delta']
    
    # 1. & 2. Correlations
    corr_true = merged['true_delta'].corr(merged['price_delta'])
    corr_facsimile = merged['facsimile'].corr(merged['price_delta'])
    facsimile_gap = corr_facsimile - corr_true
    
    # 3. Disagreement Rate
    sign_true = np.sign(merged['true_delta'])
    sign_price = np.sign(merged['price_delta'])
    disagreement_mask = (sign_true != 0) & (sign_price != 0) & (sign_true != sign_price)
    disagreement_rate = disagreement_mask.sum() / len(merged)
    
    print(f"Corr(true_delta, price_delta): {corr_true:.4f}")
    print(f"Corr(facsimile, price_delta): {corr_facsimile:.4f}")
    print(f"Facsimile Gap: {facsimile_gap:.4f}")
    print(f"Disagreement Rate: {disagreement_rate:.2%}")
    
    # 4. Range-conditioned absorption scatter
    print("Computing dynamic inflections (V2 methodology)...")
    # Identify local extrema that result in a reversal > R_k
    # For a 5m horizon (k=60)
    k = 60
    ret = merged['close'].diff()
    vol_5s = ret.rolling(window=120, min_periods=30).std()
    
    R_k = 1.5 * vol_5s * np.sqrt(k)
    fwd_min = merged['close'].shift(-k).rolling(window=k, min_periods=k).min()
    fwd_max = merged['close'].shift(-k).rolling(window=k, min_periods=k).max()
    prior_move = merged['close'] - merged['close'].shift(k)
    
    exh_mask = (prior_move.abs() > R_k)
    prior_up = prior_move > 0
    prior_dn = prior_move < 0
    
    reversal_down = fwd_min <= (merged['close'] - R_k)
    reversal_up = fwd_max >= (merged['close'] + R_k)
    
    is_inflection = (exh_mask & prior_up & reversal_down) | (exh_mask & prior_dn & reversal_up)
    
    # Find distance to nearest inflection
    print("Computing proximity to inflections...")
    inflection_indices = np.where(is_inflection)[0]
    
    # Fast distance calculation
    if len(inflection_indices) > 0:
        bar_indices = np.arange(len(merged))
        # Find index of closest inflection (searchsorted finds right insertion point)
        idx = np.searchsorted(inflection_indices, bar_indices)
        idx = np.clip(idx, 0, len(inflection_indices)-1)
        
        dist_right = np.abs(inflection_indices[idx] - bar_indices)
        idx_left = np.clip(idx - 1, 0, len(inflection_indices)-1)
        dist_left = np.abs(inflection_indices[idx_left] - bar_indices)
        
        min_dist = np.minimum(dist_left, dist_right)
        merged['bars_to_inflection'] = min_dist
    else:
        merged['bars_to_inflection'] = np.nan
        
    print("Generating scatter plot...")
    # Filter to a random sample for plotting to avoid massive overlap
    sample_df = merged.sample(n=min(50000, len(merged)), random_state=42).copy()
    
    # Absorption: high true_delta, low price_delta
    plt.figure(figsize=(10, 8))
    
    # Color map: red if close to inflection (<= 12 bars = 1 min), gray otherwise
    sample_df['is_inflection_zone'] = sample_df['bars_to_inflection'] <= 12
    
    colors = np.where(sample_df['is_inflection_zone'], 'red', 'lightgray')
    sizes = np.where(sample_df['is_inflection_zone'], 20, 5)
    alphas = np.where(sample_df['is_inflection_zone'], 0.8, 0.2)
    
    plt.scatter(sample_df['price_delta'], sample_df['true_delta'], 
                c=colors, s=sizes, alpha=alphas, edgecolors='none')
                
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    
    plt.xlabel('Price Delta (close - prev_close)')
    plt.ylabel('True Signed Volume (Delta)')
    plt.title('Absorption Scatter (Red = Near Oracle Inflection)')
    
    out_dir = "research/order_flow_ablation/reports"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "absorption_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, "stage_0B_signal_test.md"), "w") as f:
        f.write("# Stage 0B: Flow-vs-Price Signal Test\n\n")
        f.write(f"- **Corr(true_delta, price_delta):** {corr_true:.4f}\n")
        f.write(f"- **Corr(facsimile, price_delta):** {corr_facsimile:.4f}\n")
        f.write(f"- **Facsimile Gap (Delta's unique info):** {facsimile_gap:.4f}\n")
        f.write(f"- **Disagreement Rate:** {disagreement_rate:.2%} of bars have opposite sign.\n\n")
        f.write("## Absorption Scatter\n")
        f.write("![Absorption Scatter](/C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/order_flow_ablation/reports/absorption_scatter.png)\n")
        
    print("Stage 0B complete.")

if __name__ == '__main__':
    run_stage_0B()
