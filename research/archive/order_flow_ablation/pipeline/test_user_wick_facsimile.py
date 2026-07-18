import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def test_user_wick_facsimile():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open', 'high', 'low', 'close']], how='inner', rsuffix='_trade')

    print("Calculating User Features...")
    # 1. Open vs Close (Price Delta)
    merged['price_delta'] = merged['close_trade'] - merged['open']
    
    # 2. Wick distance from Close
    # If close > open (bullish bar), the wick from close is the upper wick (high - close)
    # If close < open (bearish bar), the wick from close is the lower wick (close - low)
    merged['wick_from_close'] = np.where(
        merged['price_delta'] > 0,
        merged['high'] - merged['close_trade'],
        merged['close_trade'] - merged['low']
    )
    
    # 3. Wick distance from Open
    # If close > open, wick from open is the lower wick (open - low)
    # If close < open, wick from open is the upper wick (high - open)
    merged['wick_from_open'] = np.where(
        merged['price_delta'] > 0,
        merged['open'] - merged['low'],
        merged['high'] - merged['open']
    )

    # Let's also volume-weight them as a true "facsimile" of the volume
    epsilon = 1e-8
    merged['total_range'] = merged['high'] - merged['low']
    merged['body_vol'] = (merged['price_delta'] / (merged['total_range'] + epsilon)) * merged['volume']
    merged['wick_close_vol'] = (merged['wick_from_close'] / (merged['total_range'] + epsilon)) * merged['volume']

    cols = ['delta', 'price_delta', 'wick_from_close', 'wick_from_open', 'volume', 'body_vol', 'wick_close_vol']
    df_clean = merged.dropna(subset=cols).copy()
    df_clean = df_clean[df_clean['volume'] > 0]

    corr_matrix = df_clean[cols].corr()
    
    print("\n--- Correlation with True Delta (Market Trade Data) ---")
    print(corr_matrix[['delta']].to_string())

if __name__ == "__main__":
    test_user_wick_facsimile()
