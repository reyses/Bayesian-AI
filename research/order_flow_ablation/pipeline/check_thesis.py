import pandas as pd
import numpy as np

print("Loading datasets...")
baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

if baseline_df.index.tz is None and delta_df.index.tz is not None:
    baseline_df.index = baseline_df.index.tz_localize('UTC')
elif baseline_df.index.tz is not None and delta_df.index.tz is None:
    delta_df.index = delta_df.index.tz_localize('UTC')

merged = baseline_df.join(delta_df[['delta', 'volume', 'open']], how='inner', rsuffix='_trade')

print("Calculating...")
merged['price_delta'] = merged['close'] - merged['open']
merged['facsimile'] = np.sign(merged['price_delta']) * merged['volume']

# Check how often True Delta sign == Facsimile sign
merged.dropna(subset=['delta', 'facsimile'], inplace=True)

# Disagreement
disagree = np.sign(merged['delta']) != np.sign(merged['facsimile'])
print(f"Disagreement Rate: {disagree.mean() * 100:.2f}%")

# Correlation between True Delta and Facsimile
corr = merged['delta'].corr(merged['facsimile'])
print(f"Correlation (True Delta, Facsimile): {corr:.4f}")

# Correlation of True Delta to next bar return
merged['fwd_ret_1'] = merged['close'].shift(-1) - merged['close']
print(f"Corr(True Delta, Fwd 1): {merged['delta'].corr(merged['fwd_ret_1']):.4f}")
print(f"Corr(Facsimile, Fwd 1): {merged['facsimile'].corr(merged['fwd_ret_1']):.4f}")
