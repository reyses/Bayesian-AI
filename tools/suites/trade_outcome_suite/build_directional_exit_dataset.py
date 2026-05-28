import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.build_exit_dataset import build_dataset

def main():
    csv_is = REPO / 'reports/findings/directional_ml/dir_trades_is.csv'
    csv_oos = REPO / 'reports/findings/directional_ml/dir_trades_oos.csv'
    
    out_dir = REPO / 'DATA/ATLAS_NT8/exit_dataset'
    
    print("=== Building Directional IS Exit Dataset ===")
    build_dataset(csv_is, out_dir / 'directional_exit_is.npz', features_root='DATA/ATLAS/FEATURES_5s_v2', bars_dir=REPO / 'DATA/ATLAS/5s')
    
    print("\n=== Building Directional OOS Exit Dataset ===")
    build_dataset(csv_oos, out_dir / 'directional_exit_oos.npz', features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', bars_dir=REPO / 'DATA/ATLAS_NT8/5s')

if __name__ == '__main__':
    main()
