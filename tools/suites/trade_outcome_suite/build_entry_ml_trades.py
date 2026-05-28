import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.suites.trade_outcome_suite.train_pf10_entry import TrajectoryLSTM, build_trajectory_dataset

def filter_dataset(csv_path, features_root, atlas_root, out_csv, model_path, threshold=0.58):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Filtering {csv_path} with threshold {threshold}...")
    
    # 1. Load Original Trades
    trades = pd.read_csv(csv_path)
    
    # 2. Build Dataset (to align indices)
    X_grid, X_tod, X_reg, X_dense, y, w = build_trajectory_dataset(
        csv_path, features_root=features_root, atlas_root=atlas_root)
    
    # We must match the exactly preserved trades. `build_trajectory_dataset` skips trades 
    # if features are missing. We replicate the matching logic here.
    from core_v2.features import load_features
    
    kept_indices = []
    days = trades['day'].unique()
    for day in tqdm(days, desc="Aligning Trades"):
        day_trades = trades[trades['day'] == day]
        feats = load_features(days=[day], root=features_root, require_all=False)
        if feats.empty:
            continue
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
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
            kept_indices.append(idx)
            
    assert len(kept_indices) == len(y), f"Mismatch: {len(kept_indices)} trades vs {len(y)} samples"
    
    trades_aligned = trades.loc[kept_indices].reset_index(drop=True)
    dir_arr = (trades_aligned['leg_dir'] == 'LONG').astype(int).values
    
    # 3. Predict Direction
    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    v_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
    v_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
    v_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
    v_dense = torch.tensor(X_dense, dtype=torch.float32).to(device)
    
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_grid), 512):
            logits = model(v_grid[i:i+512], v_tod[i:i+512], v_reg[i:i+512], v_dense[i:i+512])
            probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
    probs = np.concatenate(probs_list)
    
    # 4. Calculate Agreement and Filter (INVERSE FILTER: Disagreement >= threshold)
    agreement = np.where(dir_arr == 1, probs, 1.0 - probs)
    mask = agreement < (1.0 - threshold)
    
    filtered_trades = trades_aligned[mask].copy()
    
    print(f"Original: {len(trades)} | Aligned: {len(trades_aligned)} | Filtered: {len(filtered_trades)} ({(len(filtered_trades)/len(trades_aligned)*100):.1f}%)")
    
    # Ensure directory exists
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_trades.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}\n")

if __name__ == '__main__':
    model_path = REPO / 'checkpoints/trajectory_pf10_entry/best_model.pt'
    
    # OOS
    filter_dataset(
        csv_path=REPO / 'reports/findings/multi_atr/multi_atr_oos_atr2.csv',
        features_root='DATA/ATLAS_NT8/FEATURES_5s_v2',
        atlas_root='DATA/ATLAS_NT8',
        out_csv=REPO / 'reports/findings/trade_outcome_table/entry_ml_filtered_OOS.csv',
        model_path=model_path,
        threshold=0.58
    )
    
    # IS
    filter_dataset(
        csv_path=REPO / 'reports/findings/multi_atr/multi_atr_is_atr2.csv',
        features_root='DATA/ATLAS/FEATURES_5s_v2',
        atlas_root='DATA/ATLAS',
        out_csv=REPO / 'reports/findings/trade_outcome_table/entry_ml_filtered_IS.csv',
        model_path=model_path,
        threshold=0.58
    )
