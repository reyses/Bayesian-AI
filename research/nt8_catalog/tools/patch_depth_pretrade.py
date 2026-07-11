import os
import glob
import pandas as pd

def patch_dossier(dossier, depth_col, default_val=0.0):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parquet_path = os.path.join(base_dir, 'tests', dossier, 'events.parquet')
    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return
        
    df = pd.read_parquet(parquet_path)
    if depth_col in df.columns:
        df['depth'] = df[depth_col].abs()
        print(f"[{dossier}] Patched depth using {depth_col}. Sample: {df['depth'].iloc[0]}")
    else:
        df['depth'] = default_val
        print(f"[{dossier}] Column {depth_col} not found. Patched depth using default {default_val}.")
        
    df.to_parquet(parquet_path)

if __name__ == '__main__':
    # ATR-09: gap_atr_fraction represents the normalized size of the fade target
    # Wait, 'gap_atr_fraction' was an old column. Let's look at ATR-09 events directly.
    # We will just print columns first to see what we can use.
    pass
