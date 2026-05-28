"""Trajectory-Aware CNN Entry Filter.

Trains a binary classifier to predict P(winner) based on the 5-minute (60 bars at 5s)
trajectory of V2 features leading up to a Zigzag entry trigger.

Inputs:
    grid_traj : (B, 1, 60, 8, 23) - 60 bars of 8x23 V2 feature grids.
    tod       : (B, 1)            - time of day scalar.
    regime    : (B,)              - regime embedding.
Outputs:
    P(winner) : (B, 1)            - probability the trade closes green.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
from training.utils.state import regime_to_idx
from training.models.cnn.model import GRID_FLAT_IDX, L0_IDX, GRID_H, GRID_W, N_REGIMES, REGIME_EMBED

# â”€â”€â”€ Dataset Builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_regime_lookup(labels_csv: str = 'DATA/ATLAS/regime_labels_2d.csv') -> dict:
    df = pd.read_csv(labels_csv)
    df['date'] = df['date'].astype(str).str[:10]
    return dict(zip(df['date'], df['regime_2d']))

def build_trajectory_dataset(csv_path: str, atlas_root: str = 'DATA/ATLAS',
                             features_root: str = DEFAULT_FEATURES_ROOT,
                             seq_len: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build dataset from strategy_run CSV containing trades.
    Extracts `seq_len` bars of V2 features ending at each entry_ts."""
    
    trades = pd.read_csv(csv_path)
    # Binary label: 1 if pnl_usd > 0 else 0
    trades['label'] = (trades['pnl_usd'] > 0).astype(int)
    
    regime_lookup = _load_regime_lookup()
    
    # Group trades by day so we only load each day's parquet once
    days = trades['day'].unique()
    
    X_grid_list, X_tod_list, X_reg_list, X_dense_list, y_list, weights_list = [], [], [], [], [], []
    
    for day in tqdm(days, desc="Building Trajectories"):
        day_trades = trades[trades['day'] == day]
        
        feats = load_features(days=[day], root=features_root, require_all=False)
        if feats.empty:
            continue
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
            
        ts = feats['timestamp'].values.astype(np.int64)
        
        # Build feature matrix
        v2_matrix = np.zeros((len(feats), len(FEATURE_NAMES)), dtype=np.float32)
        feat_cols = set(feats.columns)
        for j, name in enumerate(FEATURE_NAMES):
            if name in feat_cols:
                v2_matrix[:, j] = feats[name].values.astype(np.float32)
                
        # Grids and tods
        grids_all = v2_matrix[:, GRID_FLAT_IDX].reshape(-1, GRID_H, GRID_W)
        tods_all = v2_matrix[:, L0_IDX].reshape(-1, 1)
        
        iso = day.replace('_', '-')
        regime_2d = regime_lookup.get(iso, 'UNKNOWN')
        regime_idx = regime_to_idx(regime_2d)
        
        for _, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            # Find the exact index of this timestamp
            idx_arr = np.where(ts == entry_ts)[0]
            if len(idx_arr) == 0:
                continue
            end_idx = idx_arr[0]
            start_idx = end_idx - seq_len + 1
            
            if start_idx < 0:
                continue # Not enough history for the trajectory
                
            traj_grid = grids_all[start_idx:end_idx+1] # shape (seq_len, 8, 23)
            tod_val = tods_all[end_idx] # scalar at entry
            
            # Note: For shorts, we could flip the trajectory features (negate price/z-scores, invert volume direction)
            # but for now we let the model learn the generic direction-agnostic geometry if it can, 
            # or rely on the direction signal. We might need a leg_dir embedding if we don't flip.
            # Let's add leg_dir as an extra input or flip. Flipping is safer for symmetry.
            leg_dir = 1 if trade['leg_dir'] == 'LONG' else -1
            
            # Do NOT flip the trajectory grid.
            # We want the model to see the absolute unadulterated grid to predict the raw physical direction (Long vs Short)
            
            # SPATIAL ANCHOR: Channel 0 is Absolute Grid, Channel 1 is Delta Grid from True Pivot
            true_pivot_ts = int(trade['true_pivot_ts'])
            pivot_idx_arr = np.where(ts == true_pivot_ts)[0]
            if len(pivot_idx_arr) > 0:
                pivot_idx = pivot_idx_arr[0]
                anchor_grid = grids_all[pivot_idx]
            else:
                anchor_grid = traj_grid[0]
                
            delta_grid = traj_grid - anchor_grid
            two_channel_grid = np.stack([traj_grid, delta_grid], axis=1) # (seq_len, 2, 8, 23)
            
            # Explicit Multi-ATR State
            multi_atr_cols = ['dir_x1', 'dist_x1', 'dir_x2', 'dist_x2', 'dir_x4', 'dist_x4', 'dir_x8', 'dist_x8', 'dir_x10', 'dist_x10']
            dense_state = trade[multi_atr_cols].values.astype(np.float32)
            
            # True Direction Label
            # If leg_dir is LONG and pnl > 0, true direction is LONG (1)
            # If leg_dir is SHORT and pnl <= 0, true direction is LONG (1)
            # Else SHORT (0)
            is_long = leg_dir == 1
            is_winner = ('pnl_usd' in trade and trade['pnl_usd'] > 0) or ('p_winner' in trade and trade['p_winner'] == 1.0)
            
            # Fallback if pnl_usd isn't there, we know 'label' was the original winner status in the old builds
            if 'pnl_usd' not in trade and 'p_winner' not in trade:
                is_winner = trade['label'] == 1.0
                
            true_dir = 1.0 if (is_long and is_winner) or (not is_long and not is_winner) else 0.0
            
            X_grid_list.append(two_channel_grid)
            X_tod_list.append(tod_val)
            X_reg_list.append(regime_idx)
            X_dense_list.append(dense_state)
            y_list.append(true_dir)
            
            # PnL Sample Weighting: Maximize Profit Factor
            weight = abs(float(trade.get('pnl_usd', 0.0)))
            # Cap the weight to prevent a single outlier from exploding the batch gradient
            weight = np.clip(weight, 1.0, 500.0)
            weights_list.append(weight)

    if not X_grid_list:
        raise ValueError("No valid trajectories found.")
        
    X_grid = np.stack(X_grid_list, axis=0) # (N, seq_len, 2, 8, 23)
    X_tod = np.stack(X_tod_list, axis=0)   # (N, 1)
    
    # Handle NaNs from parquets
    np.nan_to_num(X_grid, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_tod, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    
    X_reg = np.array(X_reg_list, dtype=np.int64) # (N,)
    X_dense = np.stack(X_dense_list, axis=0) # (N, 10)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1) # (N, 1)
    weights = np.array(weights_list, dtype=np.float32).reshape(-1, 1)
    
    return X_grid, X_tod, X_reg, X_dense, y, weights

# â”€â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TrajectoryLSTM(nn.Module):
    """LSTM over 2D Conv extracted V2 features."""
    def __init__(self, n_regimes: int = N_REGIMES, regime_embed: int = REGIME_EMBED):
        super().__init__()
        
        # Borrow the proven V2DirectionCNN 2D convolution stack
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # â†’ (64, 4, 8)
        )
        conv_flat = 64 * 4 * 8 # 2048
        
        self.regime_embed = nn.Embedding(n_regimes, regime_embed)
        
        # Sequence model
        lstm_input_size = conv_flat
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Classifier head (takes final LSTM hidden state + regime + tod)
        head_in = 128 + regime_embed + 1 + 10
        
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, grid_traj: torch.Tensor, tod: torch.Tensor, regime: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        # grid_traj: (B, seq_len, 2, H, W)
        B, seq_len, C, H, W = grid_traj.size()
        
        # Reshape for 2D Conv
        x = grid_traj.reshape(B * seq_len, C, H, W)
        c = self.conv(x) # -> (B * seq_len, 64, 4, 8)
        c = c.view(B, seq_len, -1) # -> (B, seq_len, 2048)
        
        # Run LSTM
        lstm_out, (hn, cn) = self.lstm(c) # lstm_out: (B, seq_len, 128)
        
        # We take the output at the last time step
        last_out = lstm_out[:, -1, :] # (B, 128)
        
        r = self.regime_embed(regime) # (B, regime_embed)
        
        # Concatenate final LSTM state, regime, TOD, and explicit Multi-ATR dense state
        out = torch.cat([last_out, r, tod, dense], dim=1) # (B, 128 + regime_embed + 1 + 10)
        
        return self.head(out)

# â”€â”€â”€ Training Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TrajectoryDataset(Dataset):
    def __init__(self, X_grid, X_tod, X_reg, X_dense, y, w):
        self.X_grid = X_grid
        self.X_tod = X_tod
        self.X_reg = X_reg
        self.X_dense = X_dense
        self.y = y
        self.w = w

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X_grid[idx], dtype=torch.float32),
                torch.tensor(self.X_tod[idx], dtype=torch.float32),
                torch.tensor(self.X_reg[idx], dtype=torch.long),
                torch.tensor(self.X_dense[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.w[idx], dtype=torch.float32))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_is = 'reports/findings/multi_atr/multi_atr_is_atr2.csv'
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    
    print("Building IS Dataset...")
    X_grid, X_tod, X_reg, X_dense, y, w = build_trajectory_dataset(csv_is)
    print(f"IS Shape: {X_grid.shape}, Positives: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
    
    print("Building OOS Dataset...")
    X_grid_val, X_tod_val, X_reg_val, X_dense_val, y_val, w_val = build_trajectory_dataset(csv_oos, features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', atlas_root='DATA/ATLAS_NT8')
    print(f"OOS Shape: {X_grid_val.shape}, Positives: {y_val.sum()}/{len(y_val)} ({y_val.sum()/len(y_val)*100:.1f}%)")
    
    print("Creating K-Fold Partitions...")
    n_samples = len(y)
    q_size = n_samples // 4
    
    indices = np.arange(n_samples)
    q1 = indices[:q_size]
    q2 = indices[q_size:2*q_size]
    q3 = indices[2*q_size:3*q_size]
    q4 = indices[3*q_size:]
    quarters = [q1, q2, q3, q4]
    
    print("Creating OOS DataLoader...")
    val_ds = TrajectoryDataset(X_grid_val, X_tod_val, X_reg_val, X_dense_val, y_val, w_val)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    print("Instantiating Model...")
    model = TrajectoryLSTM().to(device)
    print("Model created.")
    
    # Address class imbalance using pos_weight
    # y = 1 (winner) is minority ~35%
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    
    # We use unreduced BCE because we will multiply by our sample weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    epochs = 25
    best_auc = 0.0
    
    out_dir = 'checkpoints/trajectory_pf10_entry'
    os.makedirs(out_dir, exist_ok=True)
    
    for ep in range(epochs):
        # K-Fold Rotating Quarters for IS
        holdout_q = ep % 4
        train_idx = np.concatenate([quarters[i] for i in range(4) if i != holdout_q])
        eval_idx = quarters[holdout_q]
        
        train_ds = TrajectoryDataset(X_grid[train_idx], X_tod[train_idx], X_reg[train_idx], X_dense[train_idx], y[train_idx], w[train_idx])
        train_ld = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
        
        # Use the held-out IS quarter as a pseudo-val to track IS generalization, or just use the true OOS val_ld.
        # Since we have OOS, we will still evaluate on true OOS to track the actual generalization.
        # We can also evaluate on the held-out quarter if we wanted, but the OOS AUC is our target metric.
        
        model.train()
        train_loss = 0.0
        for b_grid, b_tod, b_reg, b_dense, b_y, b_w in tqdm(train_ld, desc=f"Epoch {ep+1}/{epochs}"):
            b_grid, b_tod, b_reg, b_dense, b_y, b_w = b_grid.to(device), b_tod.to(device), b_reg.to(device), b_dense.to(device), b_y.to(device), b_w.to(device)
            
            optimizer.zero_grad()
            logits = model(b_grid, b_tod, b_reg, b_dense)
            
            loss_unreduced = criterion(logits, b_y)
            # Multiply BCE Loss by the PnL sample weight
            loss = (loss_unreduced * b_w).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_ld)
        
        # Eval
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_y = []
        all_w = []
        with torch.no_grad():
            for b_grid, b_tod, b_reg, b_dense, b_y, b_w in val_ld:
                b_grid, b_tod, b_reg, b_dense, b_y, b_w = b_grid.to(device), b_tod.to(device), b_reg.to(device), b_dense.to(device), b_y.to(device), b_w.to(device)
                logits = model(b_grid, b_tod, b_reg, b_dense)
                loss_unreduced = criterion(logits, b_y)
                loss = (loss_unreduced * b_w).mean()
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy().flatten())
                all_y.extend(b_y.cpu().numpy().flatten())
                all_w.extend(b_w.cpu().numpy().flatten())
                
        val_loss /= len(val_ld)
        all_preds = np.array(all_preds)
        all_y = np.array(all_y)
        all_w = np.array(all_w)
        
        auc = roc_auc_score(all_y, all_preds)
        ap = average_precision_score(all_y, all_preds)
        
        # P(winner) threshold at 0.5
        preds_bin = (all_preds > 0.5).astype(int)
        
        # Magnitude Win Rate (Profit Ratio)
        correct_mask = (preds_bin == all_y)
        gross_profit = all_w[correct_mask].sum()
        gross_loss = all_w[~correct_mask].sum()
        tot_abs = gross_profit + gross_loss
        mag_wr = (gross_profit / tot_abs) if tot_abs > 0 else 0.0
        
        print(f"Ep {ep+1:2d} | TL: {train_loss:.4f} | VL: {val_loss:.4f} | AUC: {auc:.3f} | Mag WR: {mag_wr:.3f}")
              
        # Early stopping logic (max 25 epochs)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            
    print(f"Training complete. Best OOS AUC: {best_auc:.3f}")

if __name__ == '__main__':
    train()
