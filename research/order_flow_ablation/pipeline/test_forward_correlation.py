import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_forward_correlation():
    print("Loading 5s Order Flow Delta dataset...")
    df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    timeframes = ['5s', '15s', '1min', '5min', '15min']
    results = []

    print("Calculating Forward Return Correlations...")
    for tf in timeframes:
        if tf == '5s':
            resampled = df.copy()
        else:
            resampled = df.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'delta': 'sum'
            }).dropna()

        # Calculate features (Same bar)
        resampled['price_delta'] = resampled['close'] - resampled['open']
        
        resampled['wick_from_close'] = np.where(
            resampled['price_delta'] > 0,
            resampled['high'] - resampled['close'],
            resampled['close'] - resampled['low']
        )
        
        epsilon = 1e-8
        resampled['total_range'] = resampled['high'] - resampled['low']
        resampled['body_vol'] = (resampled['price_delta'] / (resampled['total_range'] + epsilon)) * resampled['volume']
        resampled['wick_close_vol'] = (resampled['wick_from_close'] / (resampled['total_range'] + epsilon)) * resampled['volume']

        # Calculate Forward Target: The return of the NEXT bar (1 period forward in the current TF)
        # Shift(-1) brings the next bar's close to the current row.
        resampled['fwd_ret_1_bar'] = (resampled['close'].shift(-1) - resampled['close'])
        # Also calculate a fixed 5m forward return for everything just to see
        # We'll stick to 1-bar forward for simplicity to measure the immediate next reaction
        
        cols = ['delta', 'body_vol', 'wick_close_vol', 'fwd_ret_1_bar', 'volume']
        df_clean = resampled.dropna(subset=cols).copy()
        df_clean = df_clean[df_clean['volume'] > 0]
        
        # Remove massive target outliers that distort correlation
        p_low = df_clean['fwd_ret_1_bar'].quantile(0.001)
        p_high = df_clean['fwd_ret_1_bar'].quantile(0.999)
        df_clean = df_clean[(df_clean['fwd_ret_1_bar'] >= p_low) & (df_clean['fwd_ret_1_bar'] <= p_high)]

        if len(df_clean) > 100:
            corr_matrix = df_clean[cols].corr()
            
            results.append({
                'Timeframe': tf,
                'True Delta -> Fwd Ret': corr_matrix.loc['delta', 'fwd_ret_1_bar'],
                'Body Volume -> Fwd Ret': corr_matrix.loc['body_vol', 'fwd_ret_1_bar'],
                'Wick Close Vol -> Fwd Ret': corr_matrix.loc['wick_close_vol', 'fwd_ret_1_bar']
            })

    results_df = pd.DataFrame(results).set_index('Timeframe')
    
    print("\n--- Correlation with FORWARD 1-Bar Return ---")
    print(results_df.to_string())

if __name__ == "__main__":
    run_forward_correlation()
