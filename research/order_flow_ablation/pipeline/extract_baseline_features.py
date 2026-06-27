import os
import sys
import pandas as pd
import numpy as np
from datetime import timezone
# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

def extract_features():
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, "FEATURES_5s_v2")
    labels_csv = os.path.join(atlas_root, "regime_labels_2d.csv")
    
    import glob
    raw_dir = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/RAW/GLBX-20260131-LBJYPKPMWM"
    files = sorted(glob.glob(os.path.join(raw_dir, "*.trades.*.dbn.zst")))
    
    days = []
    for f in files:
        basename = os.path.basename(f)
        day_str = basename.split('-')[2].split('.')[0]
        day_fmt = f"{day_str[:4]}_{day_str[4:6]}_{day_str[6:8]}"
        if os.path.exists(os.path.join(features_root, "L0", f"{day_fmt}.parquet")):
            days.append(day_fmt)
        
    print(f"Extracting features for {len(days)} days...")
    
    fps = MultiDayForwardPassSystem(
        atlas_root=atlas_root,
        features_root=features_root,
        labels_csv=labels_csv,
        days=days,
        layers=['L0', 'L1', 'L2', 'L3', 'L4']
    )
    
    iterator = iter(fps)
    
    timestamps = []
    closes = []
    feature_vectors = []
    
    count = 0
    try:
        while True:
            bar_state = next(iterator)
            if bar_state.v2_vector is not None and not np.isnan(bar_state.v2_vector).any():
                timestamps.append(bar_state.timestamp)
                closes.append(bar_state.price)
                feature_vectors.append(bar_state.v2_vector.flatten())
                count += 1
                if count % 100000 == 0:
                    print(f"Processed {count} valid bars...")
    except StopIteration:
        pass
        
    if not timestamps:
        print("No valid bars extracted!")
        return
        
    print(f"Building DataFrame from {len(timestamps)} bars...")
    
    # Construct dataframe efficiently
    v2_matrix = np.vstack(feature_vectors)
    from core_v2.features import assemble_v2_grid
    grid = assemble_v2_grid(v2_matrix)
    flat_grid = grid.reshape(grid.shape[0], -1)
    df = pd.DataFrame(flat_grid, columns=[f"F_{i}" for i in range(416)])
    df.insert(0, 'close', closes)
    
    # Handle timestamp
    df['datetime'] = pd.to_datetime(timestamps, unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    out_dir = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_features_416D.parquet")
    df.to_parquet(out_path)
    print(f"Saved {len(df)} 416D feature bars to {out_path}")

if __name__ == '__main__':
    extract_features()
