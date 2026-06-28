import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks

# --- Config ---
RAW_DATA_PATH = "DATA/ATLAS/order_flow_delta_5s.parquet"
OUTPUT_DIR = Path("research/wick_absorption_signal/reports/stage_1_profile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TFS = {
    '5s': '5s',
    '15s': '15s',
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',
    '1h': '1h'
}

print(f"Loading raw 5s data from {RAW_DATA_PATH}...")
df_raw = pd.read_parquet(RAW_DATA_PATH)
df_raw = df_raw.sort_index()

# Ensure we have OHLCV
required_cols = ['open', 'high', 'low', 'close', 'volume']
if not all(col in df_raw.columns for col in required_cols):
    raise ValueError(f"Missing required columns in raw data. Expected {required_cols}, got {df_raw.columns}")

def resample_tf(df, rule):
    print(f"Resampling to {rule}...")
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled

def compute_profile_a(df, tf_name):
    print(f"Computing Profile A for {tf_name}...")
    
    # Raw metrics
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df[['open', 'close']].max(axis=1).rsub(df['high'])
    df['lower_wick'] = df[['open', 'close']].min(axis=1).sub(df['low'])
    df['return'] = df['close'].pct_change()
    
    # Normalized metrics (avoid div by zero)
    safe_range = df['range'].replace(0, np.nan)
    df['norm_body'] = df['body'] / safe_range
    df['norm_upper_wick'] = df['upper_wick'] / safe_range
    df['norm_lower_wick'] = df['lower_wick'] / safe_range
    
    return df.dropna()

def compute_profile_b(df, tf_name):
    print(f"Computing Profile B (Oscillation Map) for {tf_name}...")
    
    # Use savgol_filter (local cubic regression) to smooth the close price
    # Window length must be odd. We'll use a small fixed bar window to find the natural turn frequency.
    window = min(11, len(df) - (len(df) % 2 == 0))
    if window < 3:
        return {"tf": tf_name, "half_cycle_bars": np.nan, "half_cycle_minutes": np.nan, "turns": 0}
        
    smoothed = savgol_filter(df['close'].values, window_length=window, polyorder=3)
    
    # Find peaks and troughs
    peaks, _ = find_peaks(smoothed)
    troughs, _ = find_peaks(-smoothed)
    
    all_turns = np.sort(np.concatenate([peaks, troughs]))
    
    if len(all_turns) > 1:
        # Distance between alternating peaks/troughs is the half cycle
        half_cycle_bars = np.diff(all_turns).mean()
        # Convert to minutes
        bar_seconds = (df.index[1] - df.index[0]).total_seconds()
        half_cycle_minutes = (half_cycle_bars * bar_seconds) / 60.0
    else:
        half_cycle_bars = np.nan
        half_cycle_minutes = np.nan
        
    return {
        "tf": tf_name,
        "half_cycle_bars": half_cycle_bars,
        "half_cycle_minutes": half_cycle_minutes,
        "turns": len(all_turns)
    }

# --- Execution ---

all_stats = []
oscillation_map = []
autocorrs = {}

for tf_name, rule in TFS.items():
    if rule == '5s':
        df_tf = df_raw.copy()
    else:
        df_tf = resample_tf(df_raw, rule)
        
    df_tf = compute_profile_a(df_tf, tf_name)
    
    # 1. Summary Stats
    cols = ['upper_wick', 'lower_wick', 'body', 'range', 'volume', 'return', 
            'norm_upper_wick', 'norm_lower_wick', 'norm_body']
    desc = df_tf[cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc['skew'] = df_tf[cols].skew()
    desc['kurtosis'] = df_tf[cols].kurtosis()
    desc.to_csv(OUTPUT_DIR / f"{tf_name}_summary_stats.csv")
    
    # 2. Histograms
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Univariate Distributions ({tf_name})')
    for i, col in enumerate(cols):
        ax = axes[i // 3, i % 3]
        sns.histplot(df_tf[col], bins=50, kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{tf_name}_histograms.png")
    plt.close()
    
    # 3. Relationships (Correlation Matrix)
    corr = df_tf[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix ({tf_name})')
    plt.savefig(OUTPUT_DIR / f"{tf_name}_correlations.png")
    plt.close()
    
    # 4. Joint Plots
    # Upper Wick vs Lower Wick
    sns.jointplot(data=df_tf, x='norm_upper_wick', y='norm_lower_wick', kind='hist', cmap='viridis')
    plt.savefig(OUTPUT_DIR / f"{tf_name}_joint_wick_vs_wick.png")
    plt.close()
    
    # Range vs Volume
    sns.jointplot(data=df_tf, x='range', y='volume', kind='hex', cmap='inferno')
    plt.savefig(OUTPUT_DIR / f"{tf_name}_joint_range_vs_volume.png")
    plt.close()
    
    # 5. Temporal (Autocorrelation)
    acfs = {
        'return_acf': [df_tf['return'].autocorr(lag=i) for i in range(1, 21)],
        'volume_acf': [df_tf['volume'].autocorr(lag=i) for i in range(1, 21)],
        'upper_wick_acf': [df_tf['upper_wick'].autocorr(lag=i) for i in range(1, 21)]
    }
    autocorrs[tf_name] = acfs
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 21), acfs['return_acf'], label='Returns', marker='o')
    plt.plot(range(1, 21), acfs['volume_acf'], label='Volume', marker='s')
    plt.plot(range(1, 21), acfs['upper_wick_acf'], label='Upper Wick', marker='^')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Autocorrelation up to 20 lags ({tf_name})')
    plt.xlabel('Lag (bars)')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"{tf_name}_autocorr.png")
    plt.close()
    
    # 6. Profile B
    osc_metrics = compute_profile_b(df_tf, tf_name)
    oscillation_map.append(osc_metrics)
    
# Save Oscillation Map
osc_df = pd.DataFrame(oscillation_map)
osc_df.to_csv(OUTPUT_DIR / "oscillation_map.csv", index=False)
print("Oscillation map saved.")

# Generate final summary markdown
def df_to_md_table(df):
    headers = " | ".join(df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(x) for x in row.values))
    return f"| {headers} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows)

summary_md = f"""# Wick Absorption Signal - Stage 1 Profiling

**Date/Time:** Generated automatically.

## Execution Complete
The script `stage_1_profiling.py` has successfully mapped the candle anatomies and cubic oscillations across all timeframes: {list(TFS.keys())}.

## Deliverables Generated in `reports/stage_1_profile/`:
- **Univariate Statistics:** `<tf>_summary_stats.csv` (includes skew, kurtosis, and tails)
- **Histograms:** `<tf>_histograms.png`
- **Correlations:** `<tf>_correlations.png`
- **Joint Distributions:** `<tf>_joint_wick_vs_wick.png` and `<tf>_joint_range_vs_volume.png`
- **Temporal Clustering (ACF):** `<tf>_autocorr.png`
- **Oscillation Map:** `oscillation_map.csv` (cubic regression turning points per TF)

### Oscillation Map Overview:
{df_to_md_table(osc_df)}

The profiles have been fully built without crossing the layers or testing hypotheses. Ready for collaborative review to determine where the structural signals live.
"""

with open(OUTPUT_DIR / "profiling_summary.md", "w") as f:
    f.write(summary_md)

print("Profiling complete. Summary written to profiling_summary.md")
