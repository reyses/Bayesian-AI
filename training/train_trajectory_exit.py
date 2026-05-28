import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

# ─── Model ───────────────────────────────────────────────────────────────────

class TrajectoryExitModel(nn.Module):
    """LSTM over 2-Channel 2D Conv extracted V2 features (Absolute + Delta)."""
    def __init__(self):
        super().__init__()
        
        # 2-channel CNN (Absolute State + Anchored Delta Field)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # → (64, 4, 8)
        )
        conv_flat = 64 * 4 * 8 # 2048
        
        # Sequence model
        lstm_input_size = conv_flat
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Dense input: [open_pnl, mae, mfe, time_in_trade, trade_velocity, dist_r_atr]
        n_dense = 6
        
        # Classifier head (takes final LSTM hidden state + dense trade metrics)
        head_in = 128 + n_dense
        
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, grid_traj: torch.Tensor, dense_metrics: torch.Tensor) -> torch.Tensor:
        # grid_traj: (B, seq_len, 2, 8, 23)
        B, seq_len, C, H, W = grid_traj.size()
        
        # Reshape for 2D Conv -> (B * seq_len, 2, 8, 23)
        x = grid_traj.reshape(B * seq_len, C, H, W)
        c = self.conv(x) # -> (B * seq_len, 64, 4, 8)
        c = c.view(B, seq_len, -1) # -> (B, seq_len, 2048)
        
        # Run LSTM
        lstm_out, (hn, cn) = self.lstm(c) # lstm_out: (B, seq_len, 128)
        
        # Output at the last time step
        last_out = lstm_out[:, -1, :] # (B, 128)
        
        # Concatenate final LSTM state and dense trade metrics
        out = torch.cat([last_out, dense_metrics], dim=1) # (B, 133)
        
        return self.head(out)

# ─── Training Loop ───────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    is_path = 'DATA/ATLAS_NT8/exit_dataset/exit_is.npz'
    oos_path = 'DATA/ATLAS_NT8/exit_dataset/exit_oos.npz'
    
    print("Loading datasets...")
    is_data = np.load(is_path)
    X_grid_is = is_data['X_grid']
    X_dense_is = is_data['X_dense']
    y_is = is_data['y']
    
    oos_data = np.load(oos_path)
    X_grid_oos = oos_data['X_grid']
    X_dense_oos = oos_data['X_dense']
    y_oos = oos_data['y']
    
    print(f"IS Shape: {X_grid_is.shape}, Dense: {X_dense_is.shape}, Positives: {y_is.sum()}/{len(y_is)} ({y_is.sum()/len(y_is)*100:.1f}%)")
    print(f"OOS Shape: {X_grid_oos.shape}, Dense: {X_dense_oos.shape}, Positives: {y_oos.sum()}/{len(y_oos)} ({y_oos.sum()/len(y_oos)*100:.1f}%)")
    
    # Scale dense features standardly? We can just divide by 100 or something. 
    # MFE/MAE/PnL are usually within [-200, 200]. Time is ~0-200. Velocity is ~[-5, 5].
    # Neural nets like small inputs. Let's do a simple min/max or z-score based on IS data.
    dense_mean = np.mean(X_dense_is, axis=0)
    dense_std = np.std(X_dense_is, axis=0) + 1e-8
    
    X_dense_is = (X_dense_is - dense_mean) / dense_std
    X_dense_oos = (X_dense_oos - dense_mean) / dense_std
    
    # Tensors
    t_grid = torch.tensor(X_grid_is, dtype=torch.float32)
    t_dense = torch.tensor(X_dense_is, dtype=torch.float32)
    t_y = torch.tensor(y_is, dtype=torch.float32).unsqueeze(1)
    
    v_grid = torch.tensor(X_grid_oos, dtype=torch.float32)
    v_dense = torch.tensor(X_dense_oos, dtype=torch.float32)
    v_y = torch.tensor(y_oos, dtype=torch.float32).unsqueeze(1)
    
    train_ds = TensorDataset(t_grid, t_dense, t_y)
    val_ds = TensorDataset(v_grid, v_dense, v_y)
    
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    model = TrajectoryExitModel().to(device)
    
    # Address class imbalance
    num_neg = (y_is == 0).sum()
    num_pos = (y_is == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    
    epochs = 20
    best_auc = 0.0
    
    out_dir = 'checkpoints/trajectory_exit'
    os.makedirs(out_dir, exist_ok=True)
    
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for b_grid, b_dense, b_y in train_ld:
            b_grid, b_dense, b_y = b_grid.to(device), b_dense.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            logits = model(b_grid, b_dense)
            loss = criterion(logits, b_y)
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
        with torch.no_grad():
            for b_grid, b_dense, b_y in val_ld:
                b_grid, b_dense, b_y = b_grid.to(device), b_dense.to(device), b_y.to(device)
                logits = model(b_grid, b_dense)
                loss = criterion(logits, b_y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy().flatten())
                all_y.extend(b_y.cpu().numpy().flatten())
                
        val_loss /= len(val_ld)
        all_preds = np.array(all_preds)
        all_y = np.array(all_y)
        
        auc = roc_auc_score(all_y, all_preds)
        ap = average_precision_score(all_y, all_preds)
        
        # Exit threshold: P(Hold) < 0.20 -> BAIL (predict 0)
        # So we predict 1 (Hold) if P(Hold) > 0.20
        preds_bin = (all_preds > 0.20).astype(int)
        precision = precision_score(all_y, preds_bin, zero_division=0)
        recall = recall_score(all_y, preds_bin, zero_division=0)
        
        held_samples = preds_bin.sum()
        total_samples = len(all_y)
        held_pct = held_samples / total_samples * 100
        
        print(f"Ep {ep+1:2d} | TL: {train_loss:.4f} | VL: {val_loss:.4f} | AUC: {auc:.3f} | AP: {ap:.3f} | "
              f"Prec(Hold): {precision:.3f} | Rec(Hold): {recall:.3f} | Held: {held_pct:.1f}%")
              
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            
    print(f"Training complete. Best OOS AUC: {best_auc:.3f}")

if __name__ == '__main__':
    train()
