import os
import pandas as pd

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    parquet_path = os.path.join(base_dir, 'DATA/ATLAS/order_flow_delta_5s.parquet')
    
    df = pd.read_parquet(parquet_path)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
    if 'symbol' in df.columns:
        print("Symbols:", df['symbol'].unique())
