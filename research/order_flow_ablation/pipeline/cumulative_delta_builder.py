import os
import glob
import databento as db
import pandas as pd
import numpy as np

def build_delta_features():
    raw_dir = "DATA/RAW/GLBX-20260131-LBJYPKPMWM"
    files = sorted(glob.glob(os.path.join(raw_dir, "*.trades.*.dbn.zst")))
    print(f"Found {len(files)} trade files.")
    
    all_bars = []
    
    for f in files:
        print(f"Processing {f}...")
        store = db.DBNStore.from_file(f)
        df = store.to_df()
        
        if len(df) == 0:
            continue
            
        # Databento trades have 'ts_recv' as index (or 'ts_event')
        # We need to compute delta. Side: 'B' (Bid/Buy) -> Ask hit (aggressor buy) -> Positive Delta
        # 'A' (Ask/Sell) -> Bid hit (aggressor sell) -> Negative Delta
        
        # Let's map side to sign
        if 'side' in df.columns:
            conditions = [
                df['side'] == 'B', # Buy
                df['side'] == 'A'  # Sell
            ]
            choices = [1, -1]
            df['sign'] = np.select(conditions, choices, default=0)
        else:
            print("Warning: No 'side' column found.")
            df['sign'] = 0
            
        df['signed_volume'] = df['size'] * df['sign']
        
        # Resample to 5s bars
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
            
        # Group by 5s
        resampled = df.resample('5s').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('size', 'sum'),
            delta=('signed_volume', 'sum')
        )
        
        resampled.dropna(subset=['close'], inplace=True)
        all_bars.append(resampled)
        
    if not all_bars:
        print("No data extracted.")
        return
        
    final_df = pd.concat(all_bars).sort_index()
    
    # Compute derived features
    final_df['cum_delta'] = final_df['delta'].cumsum()
    final_df['price_change'] = final_df['close'] - final_df['open']
    final_df['divergence'] = final_df['price_change'] - (final_df['delta'] * 0.01)
    
    out_dir = "DATA/ATLAS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "order_flow_delta_5s.parquet")
    final_df.to_parquet(out_path)
    print(f"Saved {len(final_df)} bars to {out_path}")

if __name__ == '__main__':
    build_delta_features()
