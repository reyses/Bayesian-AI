import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
from collections import deque
from datetime import datetime, timezone
import gc
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd

from network import MasterNetwork
from vtrace_reconciliation import VTraceReconciliation
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

def get_nmp_signals(grid_tensor, param_indices):
    device = grid_tensor.device
    N_AGENTS = grid_tensor.size(0)
    
    # 1. Parameter Mappings
    map_zfit = torch.tensor([1.5, 2.0, 3.0], device=device)
    map_vr_entry = torch.tensor([0.7, 1.0, 1.3], device=device)
    map_freight = torch.tensor([75.0, 100.0, 150.0], device=device)
    map_wick = torch.tensor([0.70, 0.83, 0.95], device=device)
    map_vr_max = torch.tensor([0.25, 0.35, 0.50], device=device)
    map_vr_bail = torch.tensor([0.50, 0.65, 0.80], device=device)
    
    limit_zfit = map_zfit[param_indices[:, 0]]
    limit_vr_entry = map_vr_entry[param_indices[:, 5]]
    limit_freight = map_freight[param_indices[:, 8]]
    limit_wick = map_wick[param_indices[:, 9]]
    limit_vr_max = map_vr_max[param_indices[:, 6]]
    
    # 2. Extract Features from V2_Grid
    # Dummy indices for prototype - these would map to actual V2_Grid slots
    current_z = torch.abs(grid_tensor[:, 0, -1, 0])
    current_vr = grid_tensor[:, 0, -1, 2]
    current_vel = torch.abs(grid_tensor[:, 1, -1, 3])
    current_wick = grid_tensor[:, 2, -1, 2]
    
    # 3. Vectorized NMP Logic
    is_extreme = current_z > limit_zfit
    is_calm = current_vr < limit_vr_entry
    no_freight = current_vel < limit_freight
    wick_reject = current_wick > limit_wick
    
    entry_signal = is_extreme & is_calm & no_freight & wick_reject
    exit_signal = current_vr > limit_vr_max
    
    raw_z = grid_tensor[:, 0, -1, 0]
    direction = torch.where(raw_z > 0, torch.tensor(2, device=device), torch.tensor(1, device=device))
    direction = torch.where(entry_signal, direction, torch.tensor(0, device=device))
    
    return direction, exit_signal

def run_gpu_smoke_test():
    print("[INFO] Booting 10-Headed Parameter Screening Engine...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Compute Device: {device}")
    
    master_net = MasterNetwork().to(device)
    optimizer = optim.Adam(master_net.parameters(), lr=0.003)
    vtrace = VTraceReconciliation()
    
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    l0_dir = os.path.join(features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    all_days = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    smoke_week = all_days[:5]
    print(f"[INFO] Target Screening Week : {smoke_week}")
    
    N_AGENTS = 1024
    max_epochs = 5
    trade_logs = []
    
    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(max_epochs):
        print(f"\n==============================================")
        print(f" EPOCH {epoch+1}/{max_epochs} | Simulating 10-Headed Matrix")
        print(f"==============================================")
        
        in_position = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
        directions = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
        entry_prices = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
        
        trade_entry_grids = torch.zeros((N_AGENTS, 8, 60, 23), dtype=torch.float32, device=device)
        trade_entry_l0s = torch.zeros((N_AGENTS, 60, 3), dtype=torch.float32, device=device)
        trade_entry_params = torch.zeros((N_AGENTS, 10), dtype=torch.int64, device=device)
        trade_entry_pis = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
        
        gross_profits = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
        gross_losses = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
        trade_counts = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
        
        buffer_grids, buffer_l0s, buffer_params, buffer_regrets, buffer_pis = [], [], [], [], []
        
        fps = MultiDayForwardPassSystem(
            atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=smoke_week
        )
        state_queue = deque(maxlen=60)
        master_net.eval()
        
        epsilon = max(0.01, 0.5 - (epoch / 10.0))
        temperature = max(0.1, 1.0 - (epoch / 20.0))
        
        bar_count = 0
        for bar_state in tqdm(fps, total=86400, desc=f"Epoch {epoch+1} Sim", leave=False):
            bar_count += 1
            v2_vec = bar_state.v2_vector
            if len(v2_vec) < 185: continue
            
            dt = datetime.fromtimestamp(bar_state.timestamp, tz=timezone.utc)
            day_norm = dt.weekday() / 4.0
            sec_in_day = int(bar_state.timestamp) % 86400
            tod_norm = sec_in_day / 86400.0
            
            l0 = [v2_vec[0], tod_norm, day_norm]
            grid = v2_vec[1:185].reshape(8, 23)
            state_queue.append((l0, grid))
            if len(state_queue) < 60: continue
            
            is_eod_maintenance = (75300 <= sec_in_day < 79200)
            
            l0_tensor = torch.tensor(np.array([s[0] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
            grid_tensor = torch.tensor(np.array([s[1] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
            grid_tensor = grid_tensor.permute(0, 2, 1, 3)
            l0_tensor = torch.nan_to_num(l0_tensor, nan=0.0)
            grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0)
            
            with torch.no_grad():
                heads, _ = master_net(grid_tensor, l0_tensor)
                
                joint_pi = torch.ones(N_AGENTS, dtype=torch.float32, device=device)
                param_indices = torch.zeros((N_AGENTS, 10), dtype=torch.int64, device=device)
                explore_mask = torch.rand(N_AGENTS, device=device) < epsilon
                
                for i, head_q in enumerate(heads):
                    q_exp = head_q.expand(N_AGENTS, 3)
                    probs = F.softmax(q_exp / temperature, dim=1)
                    random_acts = torch.randint(0, 3, (N_AGENTS,), device=device)
                    greedy_acts = torch.argmax(q_exp, dim=1)
                    acts = torch.where(explore_mask, random_acts, greedy_acts)
                    pi_val = torch.where(explore_mask, torch.tensor(0.333, device=device), probs.gather(1, greedy_acts.unsqueeze(1)).squeeze(1))
                    
                    joint_pi = joint_pi * pi_val
                    param_indices[:, i] = acts
            
            # Physics Proxy Evaluation
            nmp_actions, nmp_exits = get_nmp_signals(grid_tensor.expand(N_AGENTS, -1, -1, -1), param_indices)
            
            if torch.any(in_position):
                current_price = bar_state.price
                price_diff = current_price - entry_prices
                pnl = torch.where(directions == 1, price_diff * 50.0, -price_diff * 50.0)
                
                hit_sl = pnl < -50.0
                hit_tp = pnl > 100.0
                eod_exit = torch.tensor(is_eod_maintenance, device=device).expand(N_AGENTS)
                
                exit_mask = in_position & (hit_sl | hit_tp | eod_exit | nmp_exits)
                
                if torch.any(exit_mask):
                    exit_pnls = pnl[exit_mask] - 5.0
                    agents_exiting = exit_mask.nonzero(as_tuple=True)[0]
                    
                    profits = torch.where(exit_pnls > 0, exit_pnls, torch.tensor(0.0, device=device))
                    losses = torch.where(exit_pnls <= 0, torch.abs(exit_pnls), torch.tensor(0.0, device=device))
                    gross_profits[agents_exiting] += profits
                    gross_losses[agents_exiting] += losses
                    trade_counts[agents_exiting] += 1
                    
                    regrets = torch.where(exit_pnls < 20.0, (exit_pnls * 10.0) - 500.0, exit_pnls)
                    
                    # Store exact feature combinations mapped to chosen parameters
                    exited_params = trade_entry_params[exit_mask].cpu().numpy()
                    for p_arr in exited_params:
                        trade_logs.append({'zfit': p_arr[0], 'lambda': p_arr[1], 'pnl': float(exit_pnls[0])})
                    
                    buffer_grids.append(trade_entry_grids[exit_mask].clone().cpu())
                    buffer_l0s.append(trade_entry_l0s[exit_mask].clone().cpu())
                    buffer_params.append(trade_entry_params[exit_mask].clone().cpu())
                    buffer_pis.append(trade_entry_pis[exit_mask].clone().cpu())
                    buffer_regrets.append(regrets.clone().cpu())
                    
                    in_position[exit_mask] = False
                    directions[exit_mask] = 0
                    entry_prices[exit_mask] = 0.0
            
            if not is_eod_maintenance:
                entry_mask = (~in_position) & ((nmp_actions == 1) | (nmp_actions == 2))
                if torch.any(entry_mask):
                    in_position[entry_mask] = True
                    directions[entry_mask] = nmp_actions[entry_mask].to(torch.int32)
                    entry_prices[entry_mask] = bar_state.price
                    
                    expanded_grid = grid_tensor.expand(N_AGENTS, -1, -1, -1)
                    expanded_l0 = l0_tensor.expand(N_AGENTS, -1, -1)
                    trade_entry_grids[entry_mask] = expanded_grid[entry_mask]
                    trade_entry_l0s[entry_mask] = expanded_l0[entry_mask]
                    trade_entry_params[entry_mask] = param_indices[entry_mask]
                    trade_entry_pis[entry_mask] = joint_pi[entry_mask]

            # V-Trace Optimization (Daily Batch to prevent 26GB RAM OOM)
            if is_eod_maintenance and len(buffer_grids) > 2000:
                print(f"       [OPT] Running V-Trace Backpropagation on {len(buffer_grids)} trades...")
                master_net.train()
                
                all_grids = torch.cat(buffer_grids, dim=0)
                all_l0s = torch.cat(buffer_l0s, dim=0)
                all_params = torch.cat(buffer_params, dim=0)
                all_pis = torch.cat(buffer_pis, dim=0)
                all_regrets = torch.cat(buffer_regrets, dim=0)
                
                batch_size = 2048
                dataset_size = len(all_grids)
                indices = torch.randperm(dataset_size, device=device)
                
                total_loss = 0.0
                for i in range(0, dataset_size, batch_size):
                    idx = indices[i:i+batch_size]
                    b_grids = all_grids[idx].to(device)
                    b_l0s = all_l0s[idx].to(device)
                    b_params = all_params[idx].to(device)
                    b_pis = all_pis[idx].to(device)
                    b_regrets = all_regrets[idx].to(device)
                    
                    optimizer.zero_grad()
                    heads, _ = master_net(b_grids, b_l0s)
                    
                    joint_target_pi = torch.ones(len(idx), dtype=torch.float32, device=device)
                    head_loss_sum = 0.0
                    
                    for h_idx, head_q in enumerate(heads):
                        param_actions = b_params[:, h_idx]
                        current_q = head_q.gather(1, param_actions.unsqueeze(1)).squeeze(1)
                        target_probs = F.softmax(head_q / temperature, dim=1)
                        target_pi = target_probs.gather(1, param_actions.unsqueeze(1)).squeeze(1)
                        
                        joint_target_pi = joint_target_pi * target_pi
                        q_loss = F.mse_loss(current_q, b_regrets, reduction='none')
                        head_loss_sum += q_loss
                    
                    loss = vtrace.apply_gradient_correction(head_loss_sum / 10.0, joint_target_pi, b_pis)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(master_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    
                print(f"       [OPT] V-Trace Complete! Avg Loss: {total_loss / max(1, (dataset_size//batch_size)):.4f}")
                
                # Clear RAM Buffers
                buffer_grids.clear()
                buffer_l0s.clear()
                buffer_params.clear()
                buffer_regrets.clear()
                buffer_pis.clear()
                gc.collect()
                torch.cuda.empty_cache()

        mean_profit = torch.mean(gross_profits).item()
        mean_loss = torch.mean(gross_losses).item()
        metric_n = 999.0 if mean_loss == 0.0 else (mean_profit / mean_loss) - 1.0
        
        print(f"       [IS EVAL] Metric (n): {metric_n:.4f}")
        
        # Mid-Epoch Brain Save
        torch.save({
            'epoch': epoch + 1,
            'lstm': master_net.lstm.state_dict(),
            'heads': master_net.state_dict()
        }, f'checkpoints/screening_brain_ep{epoch+1}.pth')
        print(f"       [SAVE] Neural Network weights backed up to checkpoints/screening_brain_ep{epoch+1}.pth")

if __name__ == "__main__":
    run_gpu_smoke_test()
