import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.suites.trade_outcome_suite.train_pf10_entry import TrajectoryLSTM
from core_v2.features import load_features, FEATURE_NAMES
from training.train_trajectory_entry import GRID_FLAT_IDX, L0_IDX, GRID_H, GRID_W, N_REGIMES, REGIME_EMBED

def load_regime_lookup():
    p = 'reports/findings/regime/regime_daily_metrics.csv'
    if not os.path.exists(p): return {}
    df = pd.read_csv(p)
    return dict(zip(df['day'], df['regime']))

def regime_to_idx(r):
    mapping = {'TREND_UP': 0, 'TREND_DOWN': 1, 'CHOP': 2, 'VOLATILE': 3, 'UNKNOWN': 4}
    return mapping.get(r, 4)

def evaluate_thresholds():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos.csv'
    model_path = 'checkpoints/trajectory_pf10_entry/best_model.pt'
    
    trades = pd.read_csv(csv_oos)
    regime_lookup = load_regime_lookup()
    days = trades['day'].unique()
    
    X_grid_list, X_tod_list, X_reg_list, X_dense_list, pnl_list, dir_list = [], [], [], [], [], []
    
    seq_len = 60
    features_root = 'DATA/ATLAS_NT8/FEATURES_5s_v2'
    
    for day in tqdm(days, desc="Building Trajectories"):
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
        
        for _, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            idx_arr = np.where(ts == entry_ts)[0]
            if len(idx_arr) == 0:
                continue
            end_idx = idx_arr[0]
            start_idx = end_idx - seq_len + 1
            if start_idx < 0:
                continue
                
            traj_grid = grids_all[start_idx:end_idx+1]
            tod_val = tods_all[end_idx]
            
            true_pivot_ts = int(trade['true_pivot_ts'])
            pivot_idx_arr = np.where(ts == true_pivot_ts)[0]
            if len(pivot_idx_arr) > 0:
                pivot_idx = pivot_idx_arr[0]
                anchor_grid = grids_all[pivot_idx]
            else:
                anchor_grid = traj_grid[0]
                
            delta_grid = traj_grid - anchor_grid
            two_channel_grid = np.stack([traj_grid, delta_grid], axis=1)
            
            multi_atr_cols = ['dir_x1', 'dist_x1', 'dir_x2', 'dist_x2', 'dir_x4', 'dist_x4', 'dir_x8', 'dist_x8', 'dir_x10', 'dist_x10']
            dense_state = trade[multi_atr_cols].values.astype(np.float32)
            
            X_grid_list.append(two_channel_grid)
            X_tod_list.append(tod_val)
            X_reg_list.append(regime_idx)
            X_dense_list.append(dense_state)
            
            pnl_list.append(float(trade.get('pnl_usd', 0.0)))
            dir_list.append(1 if trade['leg_dir'] == 'LONG' else 0)

    X_grid = np.stack(X_grid_list, axis=0)
    X_tod = np.stack(X_tod_list, axis=0)
    X_grid = np.nan_to_num(X_grid, nan=0.0, posinf=0.0, neginf=0.0)
    X_tod = np.nan_to_num(X_tod, nan=0.0, posinf=0.0, neginf=0.0)
    X_reg = np.array(X_reg_list, dtype=np.int64)
    X_dense = np.stack(X_dense_list, axis=0)
    
    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    v_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
    v_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
    v_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
    v_dense = torch.tensor(X_dense, dtype=torch.float32).to(device)
    
    print("Running Inference...")
    with torch.no_grad():
        logits = model(v_grid, v_tod, v_reg, v_dense)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    valid_pnl = np.array(pnl_list)
    valid_dir = np.array(dir_list)
    
    print("\n--- Profit Factor Analysis by Threshold (OOS) ---")
    
    scores = np.where(valid_dir == 1, probs, 1.0 - probs)
    
    for thresh in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        accepted = scores > thresh
        sel_pnl = valid_pnl[accepted]
        
        n_trades = len(sel_pnl)
        if n_trades == 0:
            print(f"Threshold > {thresh:.2f}: 0 trades accepted.")
            continue
            
        gross_profit = sel_pnl[sel_pnl > 0].sum()
        gross_loss = abs(sel_pnl[sel_pnl <= 0].sum())
        
        pf = gross_profit / gross_loss if gross_loss > 0 else 999.9
        net = gross_profit - gross_loss
        wr = (sel_pnl > 0).mean() * 100
        
        print(f"Threshold > {thresh:.2f} | Trades: {n_trades:4d} | Net PnL: ${net:7,.0f} | WR: {wr:4.1f}% | Profit Factor: {pf:5.2f}")

    print("\n--- Weighted Directional Override (Compounded Strategy) ---")
    # Instead of filtering, we OVERRIDE the direction.
    # ML chooses LONG if probs > 0.5, else SHORT.
    ml_dir = (probs > 0.5).astype(int)
    
    # If ML agrees with Original, PnL is unchanged.
    # If ML disagrees, PnL is inverted. Since original PnL is net of $6 friction:
    # Gross original PnL = valid_pnl + 6
    # Inverted Gross PnL = -(valid_pnl + 6)
    # Inverted Net PnL = -(valid_pnl + 6) - 6 = -valid_pnl - 12
    override_pnl = np.where(ml_dir == valid_dir, valid_pnl, -valid_pnl - 12.0)
    
    n_total = len(override_pnl)
    net_override = override_pnl.sum()
    wr_override = (override_pnl > 0).mean() * 100
    
    n_inverted = (ml_dir != valid_dir).sum()
    inv_pct = (n_inverted / n_total) * 100
    
    print(f"Total Trades: {n_total}")
    print(f"Trades Inverted by ML: {n_inverted} ({inv_pct:.1f}%)")
    print(f"Override Net PnL: ${net_override:,.0f}")
    print(f"Override Win Rate: {wr_override:.1f}%")

    print("\n--- 3-Option Logic (Continue, Flip, Skip) ---")
    # scores is P(Original Direction)
    # If scores > thresh_upper: CONTINUE
    # If scores < thresh_lower: FLIP
    # Else: SKIP
    
    for thresh in [0.55, 0.60, 0.65]:
        lower_thresh = 1.0 - thresh
        
        # Continue mask
        mask_continue = scores > thresh
        pnl_continue = valid_pnl[mask_continue]
        
        # Flip mask
        mask_flip = scores < lower_thresh
        pnl_flip = -valid_pnl[mask_flip] - 12.0 # Invert PnL and subtract 2x friction
        
        # Totals
        n_continue = len(pnl_continue)
        net_continue = pnl_continue.sum()
        
        n_flip = len(pnl_flip)
        net_flip = pnl_flip.sum()
        
        n_total = n_continue + n_flip
        net_total = net_continue + net_flip
        
        if n_total > 0:
            total_wins = (pnl_continue > 0).sum() + (pnl_flip > 0).sum()
            wr_total = (total_wins / n_total) * 100
        else:
            wr_total = 0.0
            
        print(f"Confidence Target: > {thresh:.2f} (or < {lower_thresh:.2f})")
        print(f"  CONTINUE | Trades: {n_continue:3d} | Net: ${net_continue:6,.0f}")
        print(f"  FLIP     | Trades: {n_flip:3d} | Net: ${net_flip:6,.0f}")
        print(f"  COMBINED | Trades: {n_total:3d} | Net: ${net_total:6,.0f} | WR: {wr_total:4.1f}%\n")

if __name__ == '__main__':
    evaluate_thresholds()
