import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_wick_facsimile():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open', 'high', 'low', 'close']], how='inner', rsuffix='_trade')

    print("Calculating Facsimiles...")
    # 1. Basic Facsimile (Direction * Volume)
    merged['price_delta'] = merged['close_trade'] - merged['open']
    merged['basic_facsimile'] = np.sign(merged['price_delta']) * merged['volume']
    
    # 2. Wick-Adjusted Facsimile (Close Location Value * Volume)
    # CLV = ((close - low) - (high - close)) / (high - low) 
    #     = (2 * close - high - low) / (high - low)
    # This precisely accounts for both the upper and lower wicks!
    epsilon = 1e-8
    merged['total_range'] = merged['high'] - merged['low']
    merged['clv'] = (2 * merged['close_trade'] - merged['high'] - merged['low']) / (merged['total_range'] + epsilon)
    merged['wick_facsimile'] = merged['clv'] * merged['volume']

    # Drop NaNs
    cols_to_correlate = ['delta', 'basic_facsimile', 'wick_facsimile', 'volume']
    df_clean = merged.dropna(subset=cols_to_correlate).copy()
    df_clean = df_clean[df_clean['volume'] > 0]

    # Calculate Correlations
    corr_matrix = df_clean[cols_to_correlate].corr()
    
    print("\n--- Correlation with True Delta (Market Trade Data) ---")
    print(corr_matrix[['delta']].to_string())
    
    # Compare R-squared (variance explained)
    r_basic = corr_matrix.loc['basic_facsimile', 'delta']
    r_wick = corr_matrix.loc['wick_facsimile', 'delta']
    print(f"\nBasic Facsimile R-squared: {r_basic**2:.4f}")
    print(f"Wick Facsimile R-squared:  {r_wick**2:.4f}")
    print(f"Improvement in explained variance: {((r_wick**2) - (r_basic**2)) / (r_basic**2) * 100:.2f}%")

if __name__ == "__main__":
    test_wick_facsimile()
