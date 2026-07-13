import pandas as pd
import glob
import os

detectors = ['ADX-08', 'ATR-09', 'CROSS-11', 'DOW-19', 'FIB-17']
tests_dir = r'c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests'

for det in detectors:
    dirs = [d for d in os.listdir(tests_dir) if d.startswith(det)]
    if dirs:
        d = dirs[0]
        pq_file = os.path.join(tests_dir, d, 'events.parquet')
        if os.path.exists(pq_file):
            df = pd.read_parquet(pq_file)
            print(f"[{det}] events.parquet: modes={df['mode'].unique().tolist()}, max_idx={df['event_idx'].max()}, years={df.get('year', pd.Series([2024, 2025])).unique().tolist()}, len={len(df)}")
        else:
            print(f"[{det}] NO events.parquet")
