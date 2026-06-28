import os
import glob
import pandas as pd
import numpy as np
import pytz

def run_stage_0A():
    print("Loading True Delta Volume...")
    delta_df = pd.read_parquet('DATA/ATLAS/order_flow_delta_5s.parquet')
    
    if delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')
        
    delta_df['date'] = delta_df.index.tz_convert('America/New_York').date
    # RTH is 09:30 to 16:00 ET
    ny_time = delta_df.index.tz_convert('America/New_York')
    is_rth = (ny_time.time >= pd.to_datetime('09:30').time()) & (ny_time.time < pd.to_datetime('16:00').time())
    delta_df['session'] = np.where(is_rth, 'RTH', 'ETH')
    
    delta_agg = delta_df.groupby(['date', 'session'])['volume'].sum().reset_index()
    delta_agg.rename(columns={'volume': 'trade_volume'}, inplace=True)
    
    print("Loading OHLCV Volume...")
    unique_dates = delta_df['date'].unique()
    ohlcv_files = sorted(glob.glob('DATA/ATLAS/5s/*.parquet'))
    
    ohlcv_dfs = []
    for f in ohlcv_files:
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df.set_index('datetime', inplace=True)
        elif not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, utc=True)
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
            
        df['date'] = df.index.tz_convert('America/New_York').date
        
        # Intersection
        df_sub = df[df['date'].isin(unique_dates)].copy()
        if not df_sub.empty:
            ny_time_sub = df_sub.index.tz_convert('America/New_York')
            is_rth_sub = (ny_time_sub.time >= pd.to_datetime('09:30').time()) & (ny_time_sub.time < pd.to_datetime('16:00').time())
            df_sub['session'] = np.where(is_rth_sub, 'RTH', 'ETH')
            ohlcv_dfs.append(df_sub)
            
    ohlcv_df = pd.concat(ohlcv_dfs) if ohlcv_dfs else pd.DataFrame()
    
    ohlcv_agg = ohlcv_df.groupby(['date', 'session'])['volume'].sum().reset_index()
    ohlcv_agg.rename(columns={'volume': 'ohlcv_volume'}, inplace=True)
    
    # Merge
    merged = pd.merge(delta_agg, ohlcv_agg, on=['date', 'session'], how='outer').fillna(0)
    merged['ratio'] = np.where(merged['ohlcv_volume'] > 0, merged['trade_volume'] / merged['ohlcv_volume'], np.nan)
    
    out_dir = "research/order_flow_ablation/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage_0A_reconciliation.csv")
    merged.to_csv(out_path, index=False)
    
    total_trade = merged['trade_volume'].sum()
    total_ohlcv = merged['ohlcv_volume'].sum()
    print(f"Overall Ratio: {total_trade / total_ohlcv:.4f} (Trade: {total_trade}, OHLCV: {total_ohlcv})")
    print(f"Saved {len(merged)} session records to {out_path}")
    
if __name__ == '__main__':
    run_stage_0A()
