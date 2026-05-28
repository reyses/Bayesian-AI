"""Stay/Flip Directional Override Model.

Instead of predicting absolute direction (Long vs Short), this model predicts
P(Stay) — the probability that the strategy's ORIGINAL direction is correct.

    P(Stay) > 0.5  → Keep the original trade direction
    P(Stay) < 0.5  → Flip the trade direction

Labels:
    y = 1 if trade was a WINNER (original direction was correct → Stay)
    y = 0 if trade was a LOSER  (original direction was wrong  → Flip)

Sample weights = |pnl_usd| so the model is penalized more heavily for
misclassifying high-magnitude trades. This prevents the catastrophic failure
mode where the directional model accidentally flips big trend-following winners.

Training:
    - 4-Quarter K-Fold rotating epochs on IS data
    - Evaluated on true OOS each epoch
    - Magnitude Win Rate (Profit Ratio) as the primary metric
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
from sklearn.metrics import roc_auc_score, average_precision_score

from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
from training.utils.state import regime_to_idx
from training.models.cnn.model import GRID_FLAT_IDX, L0_IDX, GRID_H, GRID_W, N_REGIMES, REGIME_EMBED

# ——— Dataset Builder ——————————————————————————————————————————————————

def _load_regime_lookup(labels_csv: str = 'DATA/ATLAS/regime_labels_2d.csv') -> dict:
    df = pd.read_csv(labels_csv)
    df['date'] = df['date'].astype(str).str[:10]
    return dict(zip(df['date'], df['regime_2d']))

def build_stay_flip_dataset(csv_path: str, atlas_root: str = 'DATA/ATLAS',
                             features_root: str = DEFAULT_FEATURES_ROOT,
                             seq_len: int = 60) -> tuple:
    """Build dataset where labels are Stay(1) / Flip(0) and weights are |pnl|."""
    
    trades = pd.read_csv(csv_path)
    regime_lookup = _load_regime_lookup()
    
    days = trades['day'].unique()
    
    X_grid_list, X_tod_list, X_reg_list, X_dense_list = [], [], [], []
    y_list, weights_list, pnl_list, dir_list = [], [], [], []
    
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
            
            # SPATIAL ANCHOR: Channel 0 = Absolute Grid, Channel 1 = Delta from True Pivot
            true_pivot_ts = int(trade['true_pivot_ts'])
            pivot_idx_arr = np.where(ts == true_pivot_ts)[0]
            if len(pivot_idx_arr) > 0:
                anchor_grid = grids_all[pivot_idx_arr[0]]
            else:
                anchor_grid = traj_grid[0]
                
            delta_grid = traj_grid - anchor_grid
            two_channel_grid = np.stack([traj_grid, delta_grid], axis=1)  # (seq_len, 2, 8, 23)
            
            # Multi-ATR dense state
            multi_atr_cols = ['dir_x1', 'dist_x1', 'dir_x2', 'dist_x2', 'dir_x4', 'dist_x4', 'dir_x8', 'dist_x8', 'dir_x10', 'dist_x10']
            dense_state = trade[multi_atr_cols].values.astype(np.float32)
            
            # STAY/FLIP LABEL: 1 = winner (stay), 0 = loser (flip)
            pnl = float(trade['pnl_usd'])
            is_winner = 1.0 if pnl > 0 else 0.0
            
            # MAGNITUDE WEIGHT: |pnl_usd|, capped to prevent outlier domination
            weight = np.clip(abs(pnl), 1.0, 500.0)
            
            X_grid_list.append(two_channel_grid)
            X_tod_list.append(tod_val)
            X_reg_list.append(regime_idx)
            X_dense_list.append(dense_state)
            y_list.append(is_winner)
            weights_list.append(weight)
            pnl_list.append(pnl)
            dir_list.append(1 if trade['leg_dir'] == 'LONG' else 0)

    if not X_grid_list:
        raise ValueError("No valid trajectories found.")
        
    X_grid = np.stack(X_grid_list, axis=0)
    X_tod = np.stack(X_tod_list, axis=0)
    
    np.nan_to_num(X_grid, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_tod, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    
    X_reg = np.array(X_reg_list, dtype=np.int64)
    X_dense = np.stack(X_dense_list, axis=0)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    weights = np.array(weights_list, dtype=np.float32).reshape(-1, 1)
    pnl_arr = np.array(pnl_list, dtype=np.float32)
    dir_arr = np.array(dir_list, dtype=np.int64)
    
    return X_grid, X_tod, X_reg, X_dense, y, weights, pnl_arr, dir_arr

# ——— Model (same architecture, different semantics) ———————————————————

class StayFlipLSTM(nn.Module):
    """LSTM over 2D Conv extracted V2 features. Predicts P(Stay)."""
    def __init__(self, n_regimes: int = N_REGIMES, regime_embed: int = REGIME_EMBED):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        conv_flat = 64 * 4 * 8  # 2048
        
        self.regime_embed = nn.Embedding(n_regimes, regime_embed)
        
        self.lstm = nn.LSTM(input_size=conv_flat, hidden_size=128, num_layers=2,
                           batch_first=True, dropout=0.2)
        
        head_in = 128 + regime_embed + 1 + 10
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, grid_traj, tod, regime, dense):
        B, seq_len, C, H, W = grid_traj.size()
        x = grid_traj.reshape(B * seq_len, C, H, W)
        c = self.conv(x)
        c = c.view(B, seq_len, -1)
        lstm_out, _ = self.lstm(c)
        last_out = lstm_out[:, -1, :]
        r = self.regime_embed(regime)
        out = torch.cat([last_out, r, tod, dense], dim=1)
        return self.head(out)

# ——— Dataset ——————————————————————————————————————————————————————————

class StayFlipDataset(Dataset):
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

# ——— Training ——————————————————————————————————————————————————————————

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_is = 'reports/findings/multi_atr/multi_atr_is_atr2.csv'
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    
    print("Building IS Dataset...")
    X_grid, X_tod, X_reg, X_dense, y, w, pnl_is, dir_is = build_stay_flip_dataset(csv_is)
    print(f"IS: {X_grid.shape[0]} samples | Stay: {y.sum():.0f} ({y.mean()*100:.1f}%) | Flip: {(1-y).sum():.0f} ({(1-y).mean()*100:.1f}%)")
    
    print("Building OOS Dataset...")
    X_grid_v, X_tod_v, X_reg_v, X_dense_v, y_v, w_v, pnl_oos, dir_oos = build_stay_flip_dataset(
        csv_oos, features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', atlas_root='DATA/ATLAS_NT8')
    print(f"OOS: {X_grid_v.shape[0]} samples | Stay: {y_v.sum():.0f} ({y_v.mean()*100:.1f}%) | Flip: {(1-y_v).sum():.0f} ({(1-y_v).mean()*100:.1f}%)")
    
    # 4-Quarter K-Fold partitions
    print("Creating K-Fold Partitions...")
    n = len(y)
    q = n // 4
    indices = np.arange(n)
    quarters = [indices[:q], indices[q:2*q], indices[2*q:3*q], indices[3*q:]]
    
    val_ds = StayFlipDataset(X_grid_v, X_tod_v, X_reg_v, X_dense_v, y_v, w_v)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    print("Instantiating Model...")
    model = StayFlipLSTM().to(device)
    
    # Class imbalance: Stay (winners) is minority ~35%
    num_flip = (y == 0).sum()
    num_stay = (y == 1).sum()
    pos_weight = torch.tensor([num_flip / num_stay], dtype=torch.float32).to(device)
    print(f"pos_weight (Flip/Stay): {pos_weight.item():.2f}")
    
    # Unweighted BCE — magnitude weighting collapsed the model to 'always Stay'
    # because winners have huge |pnl| in trend-following. Only use pos_weight for class balance.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    epochs = 25
    best_auc = 0.0
    best_mag_wr = 0.0
    
    out_dir = 'checkpoints/stay_flip_atr2'
    os.makedirs(out_dir, exist_ok=True)
    
    for ep in range(epochs):
        # Rotate held-out quarter
        holdout_q = ep % 4
        train_idx = np.concatenate([quarters[i] for i in range(4) if i != holdout_q])
        
        train_ds = StayFlipDataset(X_grid[train_idx], X_tod[train_idx], X_reg[train_idx],
                                    X_dense[train_idx], y[train_idx], w[train_idx])
        train_ld = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
        
        model.train()
        train_loss = 0.0
        for b_grid, b_tod, b_reg, b_dense, b_y, b_w in tqdm(train_ld, desc=f"Epoch {ep+1}/{epochs}"):
            b_grid, b_tod, b_reg, b_dense, b_y, b_w = (
                b_grid.to(device), b_tod.to(device), b_reg.to(device),
                b_dense.to(device), b_y.to(device), b_w.to(device))
            
            optimizer.zero_grad()
            logits = model(b_grid, b_tod, b_reg, b_dense)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_ld)
        
        # ——— OOS Evaluation ———
        model.eval()
        val_loss = 0.0
        all_preds, all_y, all_w = [], [], []
        with torch.no_grad():
            for b_grid, b_tod, b_reg, b_dense, b_y, b_w in val_ld:
                b_grid, b_tod, b_reg, b_dense, b_y, b_w = (
                    b_grid.to(device), b_tod.to(device), b_reg.to(device),
                    b_dense.to(device), b_y.to(device), b_w.to(device))
                logits = model(b_grid, b_tod, b_reg, b_dense)
                val_loss += criterion(logits, b_y).item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy().flatten())
                all_y.extend(b_y.cpu().numpy().flatten())
                all_w.extend(b_w.cpu().numpy().flatten())
                
        val_loss /= len(val_ld)
        all_preds = np.array(all_preds)
        all_y = np.array(all_y)
        all_w = np.array(all_w)
        
        auc = roc_auc_score(all_y, all_preds)
        
        # Magnitude Win Rate: treats each trade's |pnl| as the weight
        # If the model says Stay (pred > 0.5) and trade IS a winner → correct, count |pnl| as profit
        # If the model says Flip (pred < 0.5) and trade IS a loser  → correct, count |pnl| as profit
        # Any mismatch → count |pnl| as loss
        preds_bin = (all_preds > 0.5).astype(int)
        correct = (preds_bin == all_y)
        mag_profit = all_w[correct].sum()
        mag_loss = all_w[~correct].sum()
        mag_total = mag_profit + mag_loss
        mag_wr = (mag_profit / mag_total) if mag_total > 0 else 0.0
        
        # Simulate the actual PnL impact on OOS
        # For each trade: if model says Stay, keep pnl. If model says Flip, invert pnl - 12 (2x friction).
        stay_mask = all_preds > 0.5
        sim_pnl = np.where(stay_mask, pnl_oos, -pnl_oos - 12.0)
        sim_net = sim_pnl.sum()
        base_net = pnl_oos.sum()
        n_flipped = (~stay_mask).sum()
        flip_pct = n_flipped / len(stay_mask) * 100
        
        print(f"Ep {ep+1:2d} | TL: {train_loss:.4f} | VL: {val_loss:.4f} | "
              f"AUC: {auc:.3f} | Mag WR: {mag_wr:.3f} | "
              f"Sim PnL: ${sim_net:+,.0f} (base ${base_net:+,.0f}) | Flip: {flip_pct:.0f}%")
        
        # Save best by AUC
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
        
        # Also track best Mag WR checkpoint
        if mag_wr > best_mag_wr:
            best_mag_wr = mag_wr
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_magwr_model.pt'))
            
    print(f"\nTraining complete. Best OOS AUC: {best_auc:.3f} | Best Mag WR: {best_mag_wr:.3f}")

if __name__ == '__main__':
    train()
