import os

file_path = "training/train_trajectory_entry.py"
with open(file_path, "r") as f:
    content = f.read()

old_block = """            # Simple flip strategy: multiply grid by leg_dir. 
            # (In V2, velocities, accelerations, and z-scores naturally invert. 
            # Non-directional features like vwap or volume magnitude might get incorrectly negated, 
            # but CNNs can handle it. For a pure physics entry filter, we will negate).
            traj_grid = traj_grid * leg_dir
            
            # SPATIAL ANCHOR: Channel 0 is Absolute Grid, Channel 1 is Delta Grid from True Pivot
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

new_block = """            # Do NOT flip the trajectory grid.
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
            y_list.append(true_dir)"""

content = content.replace(old_block, new_block)

with open(file_path, "w") as f:
    f.write(content)

print("Updated train_trajectory_entry.py for Directional Prediction")
