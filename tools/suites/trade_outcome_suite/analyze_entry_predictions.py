import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.train_trajectory_entry import TrajectoryLSTM, build_trajectory_dataset
from core_v2.features import load_features

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading OOS Dataset...")
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos.csv'
    
    # Run the builder to get the exact identical dataset the model evaluated
    X_grid_val, X_tod_val, X_reg_val, X_dense_val, y_val = build_trajectory_dataset(csv_oos, features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', atlas_root='DATA/ATLAS_NT8')
    
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
            
    df = pd.read_csv(csv_oos)
    valid_indices = []
    
    for day, day_trades in df.groupby('day'):
        feats = load_features([day], root='DATA/ATLAS_NT8/FEATURES_5s_v2')
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
            
    if len(valid_indices) != len(all_preds):
        print(f"ERROR: valid_indices ({len(valid_indices)}) != all_preds ({len(all_preds)})")
        return
        
    df_eval = df.loc[valid_indices].copy()
    df_eval['ml_prob_long'] = all_preds
    
    # 1. Model predicted direction
    df_eval['model_dir'] = np.where(df_eval['ml_prob_long'] >= 0.5, 'LONG', 'SHORT')
    
    # 2. PnL Mapping
    # If the model predicts the exact same direction as ZigZag (leg_dir), we get the raw PnL.
    # If the model predicts the OPPOSITE direction, we get the inverse PnL.
    def get_directional_pnl(row):
        if row['model_dir'] == row['leg_dir']:
            return row['pnl_usd']
        else:
            # We invert the PnL. If a Long hit a -$200 stop loss, a Short would have hit a +$200 profit target
            return -row['pnl_usd']
            
    df_eval['dir_pnl'] = df_eval.apply(get_directional_pnl, axis=1)
    df_eval['dir_win'] = (df_eval['dir_pnl'] > 0).astype(int)
    
    print("\\n=========================================================")
    print("ENTRY MODEL ANALYSIS: DIRECTIONAL PREDICTOR")
    print("=========================================================")
    
    print(f"Total OOS Triggers Evaluated: {len(df_eval)}")
    
    print("\\n1. RAW ZIGZAG (BASELINE)")
    print(f"Raw Win Rate: {(df_eval['pnl_usd'] > 0).mean()*100:.1f}%")
    print(f"Raw PnL: ${df_eval['pnl_usd'].sum():.2f}")
    
    print("\\n2. ML DIRECTIONAL STRATEGY")
    print(f"ML Win Rate: {df_eval['dir_win'].mean()*100:.1f}%")
    print(f"ML Total PnL: ${df_eval['dir_pnl'].sum():.2f}")
    
    long_preds = df_eval[df_eval['model_dir'] == 'LONG']
    short_preds = df_eval[df_eval['model_dir'] == 'SHORT']
    
    print(f"\\nPredicted LONG: {len(long_preds)} times (WR: {long_preds['dir_win'].mean()*100:.1f}%, PnL: ${long_preds['dir_pnl'].sum():.2f})")
    print(f"Predicted SHORT: {len(short_preds)} times (WR: {short_preds['dir_win'].mean()*100:.1f}%, PnL: ${short_preds['dir_pnl'].sum():.2f})")
    
    # Were there cases where ZigZag was wrong and ML inverted it to save the day?
    inverted = df_eval[df_eval['model_dir'] != df_eval['leg_dir']]
    print(f"\\nTimes ML disagreed with ZigZag: {len(inverted)} ({len(inverted)/len(df_eval)*100:.1f}%)")
    print(f"PnL gained/lost explicitly from inversions: ${inverted['dir_pnl'].sum():.2f} (vs if we followed ZigZag: ${inverted['pnl_usd'].sum():.2f})")

if __name__ == '__main__':
    main()
