"""Filter Trades for Specialized Exit ML.

Runs the Directional Entry Model on IS and OOS ATR 2 trades.
Filters for trades where the model DISAGREES with the strategy (agreement < 0.42).
Saves these specific subset trades to CSV so we can build an Exit ML dataset 
exclusively from them.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from core_v2.features import load_features
from tools.suites.trade_outcome_suite.train_pf10_entry import TrajectoryLSTM, build_trajectory_dataset

def process_file(csv_in, csv_out, entry_model_path, features_root, atlas_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nProcessing {csv_in}...")
    
    # 1. Build trajectory dataset to get X_grid
    X_grid, X_tod, X_reg, X_dense, y, w = build_trajectory_dataset(
        csv_in, features_root=features_root, atlas_root=atlas_root)
    
    # 2. Align trades (some are dropped if no features)
    trades = pd.read_csv(csv_in)
    if 'trade_id' not in trades.columns:
        trades['trade_id'] = np.arange(len(trades))
        
    pnl_list, dir_list, trade_idx_list = [], [], []
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
        
        for _, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            idx_arr = np.where(ts == entry_ts)[0]
            if len(idx_arr) == 0:
                continue
            end_idx = idx_arr[0]
            start_idx = end_idx - 60 + 1
            if start_idx < 0:
                continue
            pnl_list.append(float(trade['pnl_usd']))
            dir_list.append(1 if trade['leg_dir'] == 'LONG' else 0)
            trade_idx_list.append(trade['trade_id'])
            
    dir_arr = np.array(dir_list)
    trade_idx_arr = np.array(trade_idx_list)
    
    assert len(dir_arr) == len(y), f"Mismatch: {len(dir_arr)} trades vs {len(y)} samples"
    
    # 3. Load Entry Model and predict
    print("Loading Entry Model...")
    entry_model = TrajectoryLSTM().to(device)
    entry_model.load_state_dict(torch.load(entry_model_path, map_location=device, weights_only=True))
    entry_model.eval()
    
    v_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
    v_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
    v_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
    v_dense = torch.tensor(X_dense, dtype=torch.float32).to(device)
    
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_grid), 512):
            logits = entry_model(v_grid[i:i+512], v_tod[i:i+512], v_reg[i:i+512], v_dense[i:i+512])
            probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
    probs = np.concatenate(probs_list)  # P(Long)
    
    # 4. Filter for disagreement
    agreement = np.where(dir_arr == 1, probs, 1.0 - probs)
    disagreement_thresh = 0.58
    agreement_thresh = 1.0 - disagreement_thresh  # 0.42
    
    keep_mask = agreement < agreement_thresh
    filtered_trade_ids = trade_idx_arr[keep_mask]
    
    filtered_trades = trades[trades['trade_id'].isin(filtered_trade_ids)]
    
    print(f"Original Trades: {len(trades)}")
    print(f"Filtered Trades: {len(filtered_trades)} ({len(filtered_trades)/len(trades)*100:.1f}%)")
    
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    filtered_trades.to_csv(csv_out, index=False)
    print(f"Saved to {csv_out}")

if __name__ == '__main__':
    entry_model = 'checkpoints/trajectory_pf10_entry/best_model.pt'
    
    csv_is_in = 'reports/findings/multi_atr/multi_atr_is_atr2.csv'
    csv_oos_in = 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    
    csv_is_out = 'reports/findings/full_system/filtered_is_atr2.csv'
    csv_oos_out = 'reports/findings/full_system/filtered_oos_atr2.csv'
    
    process_file(csv_is_in, csv_is_out, entry_model, 'DATA/ATLAS/FEATURES_5s_v2', 'DATA/ATLAS')
    process_file(csv_oos_in, csv_oos_out, entry_model, 'DATA/ATLAS_NT8/FEATURES_5s_v2', 'DATA/ATLAS_NT8')
