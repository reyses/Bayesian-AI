import os
import glob
import pandas as pd
import numpy as np

def verify():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    parquets = glob.glob(os.path.join(base_dir, '**', 'events.parquet'), recursive=True)
    
    passed = True
    for p in parquets:
        try:
            df = pd.read_parquet(p)
            if 'magnitude' not in df.columns:
                print(f"FAILED: {os.path.basename(os.path.dirname(p))} is missing magnitude")
                passed = False
                continue
                
            mag = df['magnitude'].dropna()
            if len(mag) == 0:
                print(f"FAILED: {os.path.basename(os.path.dirname(p))} has empty magnitude")
                passed = False
                continue
                
            unique_mags = len(mag.round(4).unique())
            max_mag = mag.abs().max()
            min_mag = mag.abs().min()
            
            if unique_mags <= 3 and max_mag > 0 and 'RENKO' not in p:
                print(f"FAILED: {os.path.basename(os.path.dirname(p))} has constant magnitude! Max: {max_mag:.4f}, Unique: {unique_mags}")
                passed = False
            else:
                print(f"PASSED: {os.path.basename(os.path.dirname(p))} (Max: {max_mag:.2f}, Min: {min_mag:.2f}, Unique: {unique_mags})")
                
            if 'mfe' not in df.columns or 'mae' not in df.columns:
                print(f"FAILED: {os.path.basename(os.path.dirname(p))} is missing mfe/mae")
                passed = False
                
        except Exception as e:
            print(f"ERROR reading {p}: {e}")
            passed = False
            
    if passed:
        print("\nALL DOSSIERS PASSED VERIFICATION!")
    else:
        print("\nVERIFICATION FAILED!")

if __name__ == '__main__':
    verify()
