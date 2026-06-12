import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
from training.utils.state import regime_to_idx
from training.models.cnn.model import GRID_FLAT_IDX, L0_IDX, GRID_H, GRID_W
from training.train_trajectory_entry import TrajectoryLSTM, _load_regime_lookup

def export_filtered_dataset(input_csv: str, output_csv: str, features_root: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load('checkpoints/trajectory_entry/best_model.pt', map_location=device, weights_only=True))
    model.eval()

    trades = pd.read_csv(input_csv)
    regime_lookup = _load_regime_lookup()
    days = trades['day'].unique()
    preds_map = {}
    
    for day in tqdm(days, desc=f"Inferring {os.path.basename(input_csv)}"):
        day_trades = trades[trades['day'] == day]
        
        feats = load_features(days=[day], root=features_root, require_all=False)
        if feats.empty:
            continue
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
            
        ts = feats['timestamp'].values.astype(np.int64)
        
        v2_matrix = np.zeros((len(feats), len(FEATURE_NAMES)), dtype=np.float32)
        feat_cols = set(feats.columns)
        for j, name in enumerate(FEATURE_NAMES):
            if name in feat_cols:
                v2_matrix[:, j] = feats[name].values.astype(np.float32)
                
        grids_all = v2_matrix[:, GRID_FLAT_IDX].reshape(-1, GRID_H, GRID_W)
        tods_all = v2_matrix[:, L0_IDX].reshape(-1, 1)
        
        iso = day.replace('_', '-')
        regime_2d = regime_lookup.get(iso, 'UNKNOWN')
        regime_idx = regime_to_idx(regime_2d)
        
        X_grid_list, X_tod_list, X_reg_list, idx_list = [], [], [], []
        
        for idx, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            idx_arr = np.where(ts == entry_ts)[0]
            if len(idx_arr) == 0:
                continue
            end_idx = idx_arr[0]
            start_idx = end_idx - 60 + 1
            
            if start_idx < 0:
                continue
                
            traj_grid = grids_all[start_idx:end_idx+1]
            tod_val = tods_all[end_idx]
            leg_dir = 1 if trade['leg_dir'] == 'LONG' else -1
            traj_grid = traj_grid * leg_dir
            
            X_grid_list.append(traj_grid)
            X_tod_list.append(tod_val)
            X_reg_list.append(regime_idx)
            idx_list.append(idx)
            
        if X_grid_list:
            X_grid = np.stack(X_grid_list, axis=0)
            X_tod = np.stack(X_tod_list, axis=0)
            X_grid = np.nan_to_num(X_grid, nan=0.0, posinf=0.0, neginf=0.0)
            X_tod = np.nan_to_num(X_tod, nan=0.0, posinf=0.0, neginf=0.0)
            X_reg = np.array(X_reg_list, dtype=np.int64)
            
            with torch.no_grad():
                t_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
                t_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
                t_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
                
                logits = model(t_grid, t_tod, t_reg)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                for trade_idx, prob in zip(idx_list, probs):
                    preds_map[trade_idx] = prob
                    
    trades['p_winner'] = trades.index.map(preds_map)
    valid_trades = trades.dropna(subset=['p_winner']).copy()
    
    # Filter only allowed trades
    allowed = valid_trades[valid_trades['p_winner'] > 0.5].copy()
    
    print(f"{os.path.basename(input_csv)} -> Evaluated: {len(valid_trades)}, Allowed: {len(allowed)}")
    
    allowed.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == '__main__':
    # IS
    export_filtered_dataset(
        input_csv='reports/findings/strategy_runs/zigzag_is_atr4.csv',
        output_csv='reports/findings/strategy_runs/zigzag_lstm_is_atr4.csv',
        features_root=DEFAULT_FEATURES_ROOT
    )
    
    # OOS
    export_filtered_dataset(
        input_csv='reports/findings/strategy_runs/zigzag_oos_atr4.csv',
        output_csv='reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv',
        features_root='DATA/ATLAS_NT8/FEATURES_5s_v2'
    )
