import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.train_trajectory_entry import TrajectoryLSTM, build_trajectory_dataset

def process_dataset(csv_in, csv_out, features_root, atlas_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {csv_in}...")
    
    X_grid_val, X_tod_val, X_reg_val, X_dense_val, y_val = build_trajectory_dataset(csv_in, features_root=features_root, atlas_root=atlas_root)
    
    v_grid = torch.tensor(X_grid_val, dtype=torch.float32)
    v_tod = torch.tensor(X_tod_val, dtype=torch.float32)
    v_reg = torch.tensor(X_reg_val, dtype=torch.long)
    v_dense = torch.tensor(X_dense_val, dtype=torch.float32)
    v_y = torch.tensor(y_val, dtype=torch.float32)
    
    val_ds = TensorDataset(v_grid, v_tod, v_reg, v_dense, v_y)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    print("Loading model...")
    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load('checkpoints/trajectory_entry/best_model.pt', map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for b_grid, b_tod, b_reg, b_dense, b_y in val_ld:
            b_grid, b_tod, b_reg, b_dense = b_grid.to(device), b_tod.to(device), b_reg.to(device), b_dense.to(device)
            logits = model(b_grid, b_tod, b_reg, b_dense)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().flatten())
            
    df = pd.read_csv(csv_in)
    
    # Filter the exact rows that survived the builder
    # The builder skips the first 60 bars of the day
    from core_v2.features import load_features
    valid_indices = []
    for day, day_trades in df.groupby('day'):
        feats = load_features([day], root=features_root)
        if feats is None:
            continue
        ts = feats['timestamp'].values.astype(np.int64)
        for idx, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            idx_arr = np.where(ts == entry_ts)[0]
            if len(idx_arr) == 0:
                continue
            end_idx = idx_arr[0]
            start_idx = end_idx - 60 + 1
            if start_idx < 0:
                continue
            valid_indices.append(idx)
            
    df_eval = df.loc[valid_indices].copy()
    df_eval['ml_prob_long'] = all_preds
    df_eval['model_dir'] = np.where(df_eval['ml_prob_long'] >= 0.5, 'LONG', 'SHORT')
    
    # Overwrite the original trade data with the ML's directional choices
    inversions = 0
    for idx, row in df_eval.iterrows():
        if row['model_dir'] != row['leg_dir']:
            inversions += 1
            # Invert the PnL because it took the opposite side
            df_eval.at[idx, 'pnl_usd'] = -row['pnl_usd']
            df_eval.at[idx, 'pnl_pts'] = -row['pnl_pts']
            
    df_eval['leg_dir'] = df_eval['model_dir'] # Completely replace ZigZag direction
    
    print(f"Processed {len(df_eval)} trades. ML inverted ZigZag {inversions} times ({inversions/len(df_eval)*100:.1f}%).")
    df_eval.to_csv(csv_out, index=False)
    print(f"Saved directional trades to {csv_out}\\n")

def main():
    os.makedirs('reports/findings/directional_ml', exist_ok=True)
    process_dataset('reports/findings/multi_atr/multi_atr_is.csv', 'reports/findings/directional_ml/dir_trades_is.csv', features_root='DATA/ATLAS/FEATURES_5s_v2', atlas_root='DATA/ATLAS')
    process_dataset('reports/findings/multi_atr/multi_atr_oos.csv', 'reports/findings/directional_ml/dir_trades_oos.csv', features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', atlas_root='DATA/ATLAS_NT8')

if __name__ == '__main__':
    import os
    main()
