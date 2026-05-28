import os
from pathlib import Path

file_path = "training/train_trajectory_entry.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Update CSV paths in train()
content = content.replace(
    "csv_is = 'reports/findings/strategy_runs/zigzag_is_atr4.csv'",
    "csv_is = 'reports/findings/multi_atr/multi_atr_is.csv'"
)
content = content.replace(
    "csv_oos = 'reports/findings/strategy_runs/zigzag_oos_atr4.csv'",
    "csv_oos = 'reports/findings/multi_atr/multi_atr_oos.csv'"
)

# 2. Add X_dense_list to build_trajectory_dataset
content = content.replace(
    "X_grid_list, X_tod_list, X_reg_list, y_list = [], [], [], []",
    "X_grid_list, X_tod_list, X_reg_list, X_dense_list, y_list = [], [], [], [], []"
)

# 3. Modify the grid logic
old_grid_logic = """            # SPATIAL ANCHOR: Channel 0 is Absolute Grid, Channel 1 is Delta Grid from T-60
            anchor_grid = traj_grid[0]
            delta_grid = traj_grid - anchor_grid
            two_channel_grid = np.stack([traj_grid, delta_grid], axis=1) # (seq_len, 2, 8, 23)
            
            X_grid_list.append(two_channel_grid)
            X_tod_list.append(tod_val)
            X_reg_list.append(regime_idx)
            y_list.append(trade['label'])"""

new_grid_logic = """            # SPATIAL ANCHOR: Channel 0 is Absolute Grid, Channel 1 is Delta Grid from True Pivot
            true_pivot_ts = int(trade['true_pivot_ts'])
            pivot_idx_arr = np.where(ts == true_pivot_ts)[0]
            if len(pivot_idx_arr) > 0:
                pivot_idx = pivot_idx_arr[0]
                anchor_grid = grids_all[pivot_idx] * leg_dir
            else:
                anchor_grid = traj_grid[0]
                
            delta_grid = traj_grid - anchor_grid
            two_channel_grid = np.stack([traj_grid, delta_grid], axis=1) # (seq_len, 2, 8, 23)
            
            # Explicit Multi-ATR State
            multi_atr_cols = ['dir_x1', 'dist_x1', 'dir_x2', 'dist_x2', 'dir_x4', 'dist_x4', 'dir_x8', 'dist_x8', 'dir_x10', 'dist_x10']
            dense_state = trade[multi_atr_cols].values.astype(np.float32)
            
            X_grid_list.append(two_channel_grid)
            X_tod_list.append(tod_val)
            X_reg_list.append(regime_idx)
            X_dense_list.append(dense_state)
            y_list.append(trade['label'])"""
content = content.replace(old_grid_logic, new_grid_logic)

# 4. Return X_dense
content = content.replace(
    "X_reg = np.array(X_reg_list, dtype=np.int64) # (N,)",
    "X_reg = np.array(X_reg_list, dtype=np.int64) # (N,)\n    X_dense = np.stack(X_dense_list, axis=0) # (N, 10)"
)
content = content.replace(
    "return X_grid, X_tod, X_reg, y",
    "return X_grid, X_tod, X_reg, X_dense, y"
)

# 5. Update TrajectoryLSTM to accept dense features
content = content.replace(
    "head_in = 128 + regime_embed + 1",
    "head_in = 128 + regime_embed + 1 + 10"
)

old_forward = """    def forward(self, grid_traj: torch.Tensor, tod: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
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
        
        # Concatenate final LSTM state, regime, and TOD
        out = torch.cat([last_out, r, tod], dim=1) # (B, 128 + regime_embed + 1)
        
        return self.head(out)"""

new_forward = """    def forward(self, grid_traj: torch.Tensor, tod: torch.Tensor, regime: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
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
        
        return self.head(out)"""

content = content.replace(old_forward, new_forward)

# 6. Update Dataset
old_dataset = """class TrajectoryDataset(Dataset):
    def __init__(self, X_grid, X_tod, X_reg, y):
        self.X_grid = torch.tensor(X_grid, dtype=torch.float32)
        self.X_tod = torch.tensor(X_tod, dtype=torch.float32)
        self.X_reg = torch.tensor(X_reg, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_grid[idx], self.X_tod[idx], self.X_reg[idx], self.y[idx]"""

new_dataset = """class TrajectoryDataset(Dataset):
    def __init__(self, X_grid, X_tod, X_reg, X_dense, y):
        self.X_grid = torch.tensor(X_grid, dtype=torch.float32)
        self.X_tod = torch.tensor(X_tod, dtype=torch.float32)
        self.X_reg = torch.tensor(X_reg, dtype=torch.long)
        self.X_dense = torch.tensor(X_dense, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_grid[idx], self.X_tod[idx], self.X_reg[idx], self.X_dense[idx], self.y[idx]"""

content = content.replace(old_dataset, new_dataset)

# 7. Update train loop unpacking
content = content.replace(
    "X_grid_is, X_tod_is, X_reg_is, y_is = build_trajectory_dataset(csv_is, seq_len=60, is_oos=False)",
    "X_grid_is, X_tod_is, X_reg_is, X_dense_is, y_is = build_trajectory_dataset(csv_is, seq_len=60, is_oos=False)"
)
content = content.replace(
    "X_grid_oos, X_tod_oos, X_reg_oos, y_oos = build_trajectory_dataset(csv_oos, seq_len=60, is_oos=True)",
    "X_grid_oos, X_tod_oos, X_reg_oos, X_dense_oos, y_oos = build_trajectory_dataset(csv_oos, seq_len=60, is_oos=True)"
)

content = content.replace(
    "dataset_is = TrajectoryDataset(X_grid_is, X_tod_is, X_reg_is, y_is)",
    "dataset_is = TrajectoryDataset(X_grid_is, X_tod_is, X_reg_is, X_dense_is, y_is)"
)
content = content.replace(
    "dataset_oos = TrajectoryDataset(X_grid_oos, X_tod_oos, X_reg_oos, y_oos)",
    "dataset_oos = TrajectoryDataset(X_grid_oos, X_tod_oos, X_reg_oos, X_dense_oos, y_oos)"
)

# 8. Update forward pass in train loop
old_train_forward = """            inputs_grid, inputs_tod, inputs_reg, labels = data
            inputs_grid, inputs_tod, inputs_reg, labels = inputs_grid.to(device), inputs_tod.to(device), inputs_reg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_grid, inputs_tod, inputs_reg)"""

new_train_forward = """            inputs_grid, inputs_tod, inputs_reg, inputs_dense, labels = data
            inputs_grid, inputs_tod, inputs_reg, inputs_dense, labels = inputs_grid.to(device), inputs_tod.to(device), inputs_reg.to(device), inputs_dense.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_grid, inputs_tod, inputs_reg, inputs_dense)"""
content = content.replace(old_train_forward, new_train_forward)

old_eval_forward = """            inputs_grid, inputs_tod, inputs_reg, labels = data
            inputs_grid, inputs_tod, inputs_reg, labels = inputs_grid.to(device), inputs_tod.to(device), inputs_reg.to(device), labels.to(device)
            
            outputs = model(inputs_grid, inputs_tod, inputs_reg)"""

new_eval_forward = """            inputs_grid, inputs_tod, inputs_reg, inputs_dense, labels = data
            inputs_grid, inputs_tod, inputs_reg, inputs_dense, labels = inputs_grid.to(device), inputs_tod.to(device), inputs_reg.to(device), inputs_dense.to(device), labels.to(device)
            
            outputs = model(inputs_grid, inputs_tod, inputs_reg, inputs_dense)"""
content = content.replace(old_eval_forward, new_eval_forward)

with open(file_path, "w") as f:
    f.write(content)

print("Updated train_trajectory_entry.py successfully")
