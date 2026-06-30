import pandas as pd
import numpy as np

def run_p0_decomposition():
    print("Loading datasets for P0 Reconciliation Decomposition...")
    # Load data
    df_bars = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet", columns=['close']) # Just need index for dates
    df_trades = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet", columns=['volume'])

    # Ensure UTC
    if df_bars.index.tz is None:
        df_bars.index = df_bars.index.tz_localize('UTC')
    if df_trades.index.tz is None:
        df_trades.index = df_trades.index.tz_localize('UTC')

    # Reconstruct volume from OHLCV (needs the raw data if baseline doesn't have it)
    # Ah, baseline_features_416D doesn't have raw volume, we need to load the raw OHLCV
    print("Loading raw OHLCV volume...")
    import glob
    try:
        files = glob.glob("DATA/ATLAS/5s/*.parquet")
        df_ohlcv = pd.concat([pd.read_parquet(f, columns=['timestamp', 'volume']) for f in files])
        df_ohlcv.set_index('timestamp', inplace=True)
        # convert timestamp to datetime
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index, unit='s')
        if df_ohlcv.index.tz is None:
            df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
    except Exception as e:
        print(f"Could not load 5s files: {e}")
        return

    # Extract Dates
    bars_dates = df_ohlcv.index.date
    trades_dates = df_trades.index.date

    # Group by Date
    ohlcv_vol_by_date = df_ohlcv.groupby(bars_dates)['volume'].sum()
    trade_vol_by_date = df_trades.groupby(trades_dates)['volume'].sum()

    # Combine into a single daily DataFrame
    daily_vol = pd.DataFrame({
        'ohlcv_vol': ohlcv_vol_by_date,
        'trade_vol': trade_vol_by_date
    }).fillna(0)

    # Classify Dates
    daily_vol['is_matched'] = (daily_vol['ohlcv_vol'] > 0) & (daily_vol['trade_vol'] > 0)
    daily_vol['is_trade_only'] = (daily_vol['ohlcv_vol'] == 0) & (daily_vol['trade_vol'] > 0)
    daily_vol['is_ohlcv_only'] = (daily_vol['ohlcv_vol'] > 0) & (daily_vol['trade_vol'] == 0)

    matched_dates_df = daily_vol[daily_vol['is_matched']]
    unmatched_trade_dates_df = daily_vol[daily_vol['is_trade_only']]

    # Calculations
    total_ohlcv_vol = daily_vol['ohlcv_vol'].sum()
    total_trade_vol = daily_vol['trade_vol'].sum()
    overall_ratio = total_trade_vol / total_ohlcv_vol if total_ohlcv_vol > 0 else 0

    matched_ohlcv_vol = matched_dates_df['ohlcv_vol'].sum()
    matched_trade_vol = matched_dates_df['trade_vol'].sum()
    matched_ratio = matched_trade_vol / matched_ohlcv_vol if matched_ohlcv_vol > 0 else 0

    unmatched_trade_vol = unmatched_trade_dates_df['trade_vol'].sum()
    unmatched_pct_of_total = (unmatched_trade_vol / total_trade_vol) * 100 if total_trade_vol > 0 else 0

    print("\n" + "="*50)
    print("P0: RECONCILIATION DECOMPOSITION")
    print("="*50)
    print(f"Overall Headline Ratio: {overall_ratio:.4f} (Trade: {total_trade_vol:,.0f} | OHLCV: {total_ohlcv_vol:,.0f})")
    print(f"Matched-Bar Ratio:      {matched_ratio:.4f} (Trade: {matched_trade_vol:,.0f} | OHLCV: {matched_ohlcv_vol:,.0f})")
    print(f"Unmatched Trade Volume: {unmatched_trade_vol:,.0f} ({unmatched_pct_of_total:.1f}% of all trades)")
    
    print("\n--- Unmatched Dates (Trades present, Bars missing) ---")
    if not unmatched_trade_dates_df.empty:
        for date, row in unmatched_trade_dates_df.iterrows():
            print(f"  {date}: {row['trade_vol']:,.0f} trades")
    else:
        print("  None.")

    print("\n--- Verdict ---")
    if 0.99 <= matched_ratio <= 1.01:
        print("[BENIGN] Matched-bar ratio is ~1.0. The excess volume is entirely from gap dates.")
        print("ACTION: Restrict Stage 1 Ablation to MATCHED DATES ONLY. Layer 3 is valid.")
    else:
        print(f"[FATAL] Matched-bar ratio is {matched_ratio:.4f}. There is a genuine per-bar mismatch.")
        print("ACTION: STOP. Diagnose contract/venue definition before testing Layer 3.")
    print("="*50)

if __name__ == "__main__":
    run_p0_decomposition()
