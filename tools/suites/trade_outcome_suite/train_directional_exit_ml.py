import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.train_trajectory_exit import TrajectoryExitModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    is_path = 'DATA/ATLAS_NT8/exit_dataset/directional_exit_is.npz'
    oos_path = 'DATA/ATLAS_NT8/exit_dataset/directional_exit_oos.npz'
    
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
    
    dense_mean = np.mean(X_dense_is, axis=0)
    dense_std = np.std(X_dense_is, axis=0) + 1e-8
    
    X_dense_is = (X_dense_is - dense_mean) / dense_std
    X_dense_oos = (X_dense_oos - dense_mean) / dense_std
    
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
    
    num_neg = (y_is == 0).sum()
    num_pos = (y_is == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    best_pr_auc = -1
    os.makedirs('checkpoints/directional_exit', exist_ok=True)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_grid, b_dense, b_y in train_ld:
            b_grid, b_dense, b_y = b_grid.to(device), b_dense.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            logits = model(b_grid, b_dense)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_ld)
        
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
        
        try:
            auc = roc_auc_score(all_y, all_preds)
            pr_auc = average_precision_score(all_y, all_preds)
        except ValueError:
            auc, pr_auc = 0.0, 0.0
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), 'checkpoints/directional_exit/best_model.pt')
            print("  [*] Saved best model!")
            
        scheduler.step(pr_auc)

if __name__ == '__main__':
    train()
