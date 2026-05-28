"""Build Exit Dataset for Filtered Trades.

Uses the build_dataset function from training.build_exit_dataset to process
the filtered IS and OOS CSVs, outputting NPZ files for Exit ML training.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.build_exit_dataset import build_dataset

def main():
    csv_is = REPO / 'reports/findings/full_system/filtered_is_atr2.csv'
    csv_oos = REPO / 'reports/findings/full_system/filtered_oos_atr2.csv'
    
    out_dir = REPO / 'DATA/ATLAS_NT8/exit_dataset'
    
    print("=== Building Filtered IS Exit Dataset ===")
    build_dataset(csv_is, out_dir / 'filtered_exit_is.npz', 
                  features_root='DATA/ATLAS/FEATURES_5s_v2', 
                  bars_dir=REPO / 'DATA/ATLAS/5s')
    
    print("\n=== Building Filtered OOS Exit Dataset ===")
    build_dataset(csv_oos, out_dir / 'filtered_exit_oos.npz', 
                  features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', 
                  bars_dir=REPO / 'DATA/ATLAS_NT8/5s')

if __name__ == '__main__':
    main()
