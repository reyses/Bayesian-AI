import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
import time
import sys
import gc
import pandas as pd
from tqdm import tqdm
from collections import deque
from datetime import datetime, timezone
import threading
import queue
import ctypes

from network_research_A import ResearchANetwork
from vtrace_reconciliation import VTraceReconciliation
from curriculum_config import load_config, save_segment_metrics
from curriculum_metrics import evaluate_is_metrics, OOSDiagnosticsSuite

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem, ForwardPassSystem

def load_transfer_weights(master_net, pth_path, device):
    """Injects the starter brain's LSTM weights into Research A, ignoring shape mismatches."""
    print(f"[TRANSFER] Injecting Starter Brain: {pth_path}")
    state = torch.load(pth_path, map_location=device)
    
    src_state = state['heads'] if (isinstance(state, dict) and 'heads' in state) else state
    dest_state = master_net.state_dict()
    matched_state = {}
    mismatched_keys = []
    
    for k, v in src_state.items():
        if k in dest_state:
            if v.shape == dest_state[k].shape:
                matched_state[k] = v
            else:
                mismatched_keys.append(f"{k} (checkpoint: {v.shape}, model: {dest_state[k].shape})")
                
    if mismatched_keys:
        print(f"[TRANSFER] Shape mismatches found and skipped:\n  " + "\n  ".join(mismatched_keys))
        
    master_net.load_state_dict(matched_state, strict=False)
    print(f"[TRANSFER] LSTM Core and matching layers successfully mapped. Loaded {len(matched_state)} layers.")

def get_available_chunks(features_root, chunk_size=5):
    l0_dir = os.path.join(features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    all_days = [os.path.basename(f).replace('.parquet', '') for f in files]
    chunks = [all_days[i:i+chunk_size] for i in range(0, len(all_days), chunk_size)]
    return chunks

def physics_proxy_research_A(grid_tensor, zfit_indices, vr_indices, patience_indices):
    """
    Translates the neural outputs of Research A into real physical signals.
    grid_tensor: [N_AGENTS, 8, 60, 23]
    zfit_indices: [N_AGENTS] (values 0..8)
    vr_indices: [N_AGENTS] (values 0..8)
    patience_indices: [N_AGENTS] (values 0..8)
    
    Returns:
    direction: [N_AGENTS] (0: Hold, 1: Long, 2: Short)
    exit_signal: [N_AGENTS] (bool tensor)
    """
    device = grid_tensor.device
    N_AGENTS = grid_tensor.size(0)
    
    # 1. Parameter Mappings (9-Level Response Surfaces)
    map_zfit = torch.linspace(1.5, 3.5, 9, device=device)
    map_vr = torch.linspace(0.1, 0.9, 9, device=device)
    
    limit_zfit = map_zfit[zfit_indices]
    limit_vr = map_vr[vr_indices]
    
    # 2. Extract Features from V2_Grid
    current_z = torch.abs(grid_tensor[:, 0, -1, 0])
    current_vr = grid_tensor[:, 0, -1, 2]
    current_vel = torch.abs(grid_tensor[:, 1, -1, 3])
    current_wick = grid_tensor[:, 2, -1, 2]
    
    # 3. Vectorized NMP Logic
    is_extreme = current_z > limit_zfit
    
    # Nominal fallback limits for features not optimized by the network
    limit_freight = torch.tensor(100.0, device=device).expand(N_AGENTS)
    limit_wick = torch.tensor(0.83, device=device).expand(N_AGENTS)
    
    limit_vr_entry = limit_vr
    limit_vr_max = torch.clamp(limit_vr + 0.2, max=0.95)
    
    is_calm = current_vr < limit_vr_entry
    no_freight = current_vel < limit_freight
    wick_reject = current_wick > limit_wick
    
    entry_signal = is_extreme & is_calm & no_freight & wick_reject
    exit_signal = current_vr > limit_vr_max
    
    raw_z = grid_tensor[:, 0, -1, 0]
    direction = torch.where(raw_z > 0, torch.tensor(2, device=device), torch.tensor(1, device=device))
    direction = torch.where(entry_signal, direction, torch.tensor(0, device=device))
    
    return direction, exit_signal

def learner_worker(learner_queue, master_net, optimizer, vtrace, temperature, device, epoch_diagnostics=None, meta_critic_multiplier=1.0):
    """Background thread that continuously pulls trade batches and runs V-Trace backprop."""
    while True:
        item = learner_queue.get()
        if item is None:  # Shutdown sentinel
            break
            
        all_grids, all_l0s, all_params, all_pis, all_regrets = item
        dataset_size = len(all_grids)
        batch_size = 512
        indices = torch.randperm(dataset_size, device='cpu')
        
        master_net.train()
        for idx_opt in range(0, dataset_size, batch_size):
            idx = indices[idx_opt:idx_opt+batch_size]
            b_grids = all_grids[idx].to(device)
            b_l0s = all_l0s[idx].to(device)
            b_params = all_params[idx].to(device)
            b_pis = all_pis[idx].to(device)
            b_regrets = all_regrets[idx].to(device)
            
            optimizer.zero_grad()
            q_action, _ = master_net(b_grids, b_l0s)
            
            # Since param_actions was [Batch, 4], the first index is the action head.
            param_actions = b_params[:, 0]
            current_q = q_action.gather(1, param_actions.unsqueeze(1)).squeeze(1)
            target_probs = F.softmax(q_action / temperature, dim=1)
            target_pi = target_probs.gather(1, param_actions.unsqueeze(1)).squeeze(1)
            
            # Log Scaling Regrets
            log_regrets = torch.sign(b_regrets) * torch.log1p(torch.abs(b_regrets))
            
            q_loss = F.mse_loss(current_q, log_regrets, reduction='none')
            
            loss = vtrace.apply_gradient_correction(q_loss, target_pi, b_pis[:, 0] if b_pis.dim() > 1 else b_pis)
            loss = loss * meta_critic_multiplier
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(master_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch_diagnostics is not None:
                epoch_diagnostics["loss_sum"] += loss.item()
                epoch_diagnostics["grad_norm_sum"] += total_norm.item()
                epoch_diagnostics["batches"] += 1

def run_quadrant_sim(fps, master_net, optimizer, vtrace, config, device, epoch_idx=0, N_AGENTS=128, is_eval=False, meta_critic_multiplier=1.0):
    trade_pnls = []
    trade_durations = []
    trade_mfe_avail = []
    trade_mfe_trade = []
    trade_mae = []
    
    if is_eval:
        N_AGENTS = 1
        master_net.eval()
        temperature = 1.0
    else:
        master_net.train()
        base_epsilon = config.get("epsilon_start", 0.5) * (config.get("epsilon_decay", 0.95) ** epoch_idx)
        epsilon_floor = config.get("epsilon_min", 0.05) + config.get("epsilon_offset", 0.0)
        base_epsilon = max(base_epsilon, epsilon_floor)
        temperature = max(0.1, 1.0 - (epoch_idx / 20.0))
        
    in_position = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
    directions = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
    entry_prices = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    position_age = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
    
    if is_eval:
        efactor = torch.ones(N_AGENTS, device=device)
        ecuriosity = torch.zeros(N_AGENTS, device=device)
        epsilon = torch.zeros(N_AGENTS, device=device)
    else:
        # EFactor: Natural aggressiveness profile per agent N(1.0, 1.0) clamped to [0.05, 0.95] to prevent 100% clustering
        efactor = torch.normal(mean=1.0, std=1.0, size=(N_AGENTS,), device=device).clamp(0.05, 0.95)
        
        # ECuriosity: Natural curiosity profile per agent (Epsilon modifier)
        ecuriosity = torch.normal(mean=1.0, std=1.0, size=(N_AGENTS,), device=device).clamp(0.05, 1.95)
        
        epsilon = (base_epsilon * ecuriosity).clamp(0.0, 1.0)
        
    if N_AGENTS >= 4:
        q_size = N_AGENTS // 4
        q1_mask = torch.arange(N_AGENTS, device=device) < q_size
        q2_mask = (torch.arange(N_AGENTS, device=device) >= q_size) & (torch.arange(N_AGENTS, device=device) < (q_size * 2))
        q3_mask = (torch.arange(N_AGENTS, device=device) >= (q_size * 2)) & (torch.arange(N_AGENTS, device=device) < (q_size * 3))
        q4_mask = torch.arange(N_AGENTS, device=device) >= (q_size * 3)
        param_indices = torch.zeros(N_AGENTS, dtype=torch.long, device=device)
        param_indices[q2_mask] = 1
        param_indices[q3_mask] = 2
        param_indices[q4_mask] = 3
    else:
        q1_mask = torch.ones(N_AGENTS, dtype=torch.bool, device=device)
        q2_mask = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
        q3_mask = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
        q4_mask = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
        param_indices = torch.zeros(N_AGENTS, dtype=torch.long, device=device)
    trade_entry_grids = torch.zeros((N_AGENTS, 8, 60, 23), dtype=torch.float32, device=device)
    trade_entry_l0s = torch.zeros((N_AGENTS, 60, 3), dtype=torch.float32, device=device)
    trade_entry_params = torch.zeros((N_AGENTS, 1), dtype=torch.int64, device=device)
    trade_entry_pis = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    
    gross_profits = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    gross_losses = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    trade_counts = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
    
    learner_queue = None
    learner_thread = None
    epoch_diagnostics = {"loss_sum": 0.0, "grad_norm_sum": 0.0, "batches": 0} if not is_eval else None
    if not is_eval:
        # Maxsize=10 means at most ~20,000 trades buffered in RAM if Learner is too slow
        learner_queue = queue.Queue(maxsize=10)
        learner_thread = threading.Thread(
            target=learner_worker, 
            args=(learner_queue, master_net, optimizer, vtrace, temperature, device, epoch_diagnostics, meta_critic_multiplier)
        )
        learner_thread.start()
        
    buffer_grids, buffer_l0s, buffer_params, buffer_regrets, buffer_pis = [], [], [], [], []
    trade_pnls, trade_durations = [], []
    
    # Process ticks sequentially from FPS day by day
    for day in tqdm(fps._days, desc="Evaluating Days" if is_eval else f"Training Epoch {epoch_idx+1} Days", leave=False):
        try:
            ticker = ForwardPassSystem(day=day, atlas_root=fps._atlas_root,
                                       features_root=fps._features_root,
                                       labels_csv=fps._labels_csv)
        except FileNotFoundError:
            continue
            
        ts_arr = ticker._feats['timestamp'].values.astype(np.int64)
        N_BARS = len(ts_arr)
        if N_BARS < 60:
            continue
            
        v2_matrix = ticker._v2_matrix
        sec_in_day = ts_arr % 86400
        tod_norms = sec_in_day / 86400.0
        
        # weekday norm
        dt_series = pd.to_datetime(ts_arr, unit='s', utc=True)
        day_norms = dt_series.weekday.values / 4.0
        
        l0_all = np.column_stack([v2_matrix[:, 0], tod_norms, day_norms])
        grid_all = v2_matrix[:, 1:185].reshape(-1, 8, 23)
        
        # Pre-compute price
        is_1m_start = (ticker._ts1m[0] % 60 == 0) if len(ticker._ts1m) > 0 else False
        search_ts_1m = ts_arr - 60 if is_1m_start else ts_arr
        idx1 = np.searchsorted(ticker._ts1m, search_ts_1m, side='right') - 1
        price_arr = np.zeros(N_BARS, dtype=np.float32)
        valid_mask = (idx1 >= 0) & (idx1 < len(ticker._ts1m))
        price_arr[valid_mask] = ticker._c1[idx1[valid_mask]]
        
        # Convert to GPU tensors once per day
        l0_all_tensor = torch.tensor(l0_all, dtype=torch.float32, device=device)
        grid_all_tensor = torch.tensor(grid_all, dtype=torch.float32, device=device)
        
        l0_all_tensor = torch.nan_to_num(l0_all_tensor, nan=0.0)
        grid_all_tensor = torch.nan_to_num(grid_all_tensor, nan=0.0)
        
        for i in range(59, N_BARS):
            current_price = float(price_arr[i])
            sec = sec_in_day[i]
            is_eod_maintenance = (75300 <= sec < 79200)
            
            # Slices are zero-copy views on GPU
            grid_tensor = grid_all_tensor[i-59:i+1].permute(1, 0, 2).unsqueeze(0) # [1, 8, 60, 23]
            l0_tensor = l0_all_tensor[i-59:i+1].unsqueeze(0) # [1, 60, 3]
            
            with torch.no_grad():
                q_action, _ = master_net(grid_tensor, l0_tensor)
                
                explore_mask = torch.rand(N_AGENTS, device=device) < epsilon
                
                head_size = q_action.size(1)
                q_exp = q_action.expand(N_AGENTS, head_size)
                probs = F.softmax(q_exp / temperature, dim=1)
                random_acts = torch.randint(0, head_size, (N_AGENTS,), device=device)
                greedy_acts = torch.argmax(q_exp, dim=1)
                nn_action = torch.where(explore_mask, random_acts, greedy_acts)
                joint_pi = torch.where(explore_mask, torch.tensor(1.0 / head_size, device=device), probs.gather(1, greedy_acts.unsqueeze(1)).squeeze(1))
                    
            nn_wants_flat = (nn_action == 0) | ((directions == 1) & (nn_action == 2)) | ((directions == 2) & (nn_action == 1))
            
            # Pure Q4 Routing (NN Entry + NN Exit)
            wants_entry_long = (nn_action == 1)
            wants_entry_short = (nn_action == 2)
            custom_exit = nn_wants_flat
            
            # Check exits
            if torch.any(in_position):
                position_age[in_position] += 1
                price_diff = current_price - entry_prices
                pnl = torch.where(directions == 1, price_diff * 50.0, -price_diff * 50.0)
                
                hit_sl = pnl < -50.0
                hit_tp = pnl > 100.0
                eod_exit = torch.tensor(is_eod_maintenance, device=device).expand(N_AGENTS)
                
                exit_mask = in_position & (hit_sl | hit_tp | eod_exit | custom_exit)
                
                if torch.any(exit_mask):
                    # Subtract $5.00 round-trip costs as requested
                    exit_pnls = pnl[exit_mask] - 5.0
                    exit_durs = position_age[exit_mask]
                    agents_exiting = exit_mask.nonzero(as_tuple=True)[0]
                    
                    profits = torch.where(exit_pnls > 0, exit_pnls, torch.tensor(0.0, device=device))
                    losses = torch.where(exit_pnls <= 0, torch.abs(exit_pnls), torch.tensor(0.0, device=device))
                    gross_profits[agents_exiting] += profits
                    gross_losses[agents_exiting] += losses
                    trade_counts[agents_exiting] += 1
                    
                    F_FRICTION = 18.0
                    LAMBDA = 0.4
                    H_LOOKAHEAD = 27
                    
                    optimal_pnls = torch.zeros_like(exit_pnls)
                    
                    # Diagnostics lists
                    mfe_avail_list = []
                    mfe_trade_list = []
                    mae_list = []
                    
                    for local_idx in range(len(agents_exiting)):
                        agent_id = agents_exiting[local_idx].item()
                        dur = exit_durs[local_idx].item()
                        entry_bar_idx = i - dur
                        lookahead_end = min(entry_bar_idx + H_LOOKAHEAD, N_BARS)
                        
                        price_path_avail = price_arr[entry_bar_idx:lookahead_end]
                        price_path_trade = price_arr[entry_bar_idx:i+1] # up to current bar
                        agent_dir = directions[agent_id].item()
                        entry_p = entry_prices[agent_id].item()
                        
                        if agent_dir == 1: # Long
                            max_fav_avail = np.max(price_path_avail)
                            mfe_avail = ((max_fav_avail - entry_p) * 50.0) - 5.0
                            
                            max_fav_trade = np.max(price_path_trade)
                            mfe_trade = ((max_fav_trade - entry_p) * 50.0) - 5.0
                            
                            min_price = np.min(price_path_trade)
                            mae = ((entry_p - min_price) * 50.0)
                        else: # Short
                            min_fav_avail = np.min(price_path_avail)
                            mfe_avail = ((entry_p - min_fav_avail) * 50.0) - 5.0
                            
                            min_fav_trade = np.min(price_path_trade)
                            mfe_trade = ((entry_p - min_fav_trade) * 50.0) - 5.0
                            
                            max_price = np.max(price_path_trade)
                            mae = ((max_price - entry_p) * 50.0)
                            
                        optimal_pnls[local_idx] = mfe_avail
                        mfe_avail_list.append(max(0.0, mfe_avail))
                        mfe_trade_list.append(max(0.0, mfe_trade))
                        mae_list.append(max(0.0, mae))

                    pnl_adj = exit_pnls - F_FRICTION
                    base_reward = torch.sign(pnl_adj) * torch.log1p(torch.abs(pnl_adj))
                    gap = torch.clamp(optimal_pnls - exit_pnls, min=0.0)
                    regret_term = torch.log1p(gap)

                    regrets = base_reward - LAMBDA * regret_term
                    
                    if not is_eval:
                        buffer_grids.append(trade_entry_grids[exit_mask].clone().cpu())
                        buffer_l0s.append(trade_entry_l0s[exit_mask].clone().cpu())
                        buffer_params.append(trade_entry_params[exit_mask].clone().cpu())
                        buffer_pis.append(trade_entry_pis[exit_mask].clone().cpu())
                        buffer_regrets.append(regrets.clone().cpu())
                    else:
                        trade_pnls.extend(exit_pnls.cpu().numpy().tolist())
                        trade_durations.extend(exit_durs.cpu().numpy().tolist())
                        if "mfe_avail" not in locals():
                            # Return lists through the method signature if evaluating
                            pass
                        try:
                            # Use a hack to attach it to trade_pnls or return a dictionary
                            # Actually let's return a dictionary from run_quadrant_sim if is_eval
                            pass
                        except:
                            pass
                        
                        trade_mfe_avail.extend(mfe_avail_list)
                        trade_mfe_trade.extend(mfe_trade_list)
                        trade_mae.extend(mae_list)
                        
                    in_position[exit_mask] = False
                    directions[exit_mask] = 0
                    entry_prices[exit_mask] = 0.0
                    position_age[exit_mask] = 0
                    
            # Check entries
            if not is_eod_maintenance:
                base_spawn = 1.0 if is_eval else epsilon
                spawn_prob = base_spawn * efactor
                dice = torch.rand(N_AGENTS, device=device)
                authorized_to_enter = dice < spawn_prob
                
                entry_mask = (~in_position) & (wants_entry_long | wants_entry_short) & authorized_to_enter
                if torch.any(entry_mask):
                    in_position[entry_mask] = True
                    directions[entry_mask] = torch.where(wants_entry_long[entry_mask], torch.tensor(1, dtype=torch.int32, device=device), torch.tensor(2, dtype=torch.int32, device=device))
                    entry_prices[entry_mask] = current_price
                    position_age[entry_mask] = 0
                    
                    expanded_grid = grid_tensor.expand(N_AGENTS, -1, -1, -1)
                    expanded_l0 = l0_tensor.expand(N_AGENTS, -1, -1)
                    trade_entry_grids[entry_mask] = expanded_grid[entry_mask]
                    trade_entry_l0s[entry_mask] = expanded_l0[entry_mask]
                    trade_entry_params[entry_mask, 0] = nn_action[entry_mask]
                    trade_entry_pis[entry_mask] = joint_pi[entry_mask]
                    
            # Train optimization using V-Trace Reconciliation (Async)
            total_buffered = sum(x.size(0) for x in buffer_grids) if len(buffer_grids) > 0 else 0
            if not is_eval and (total_buffered >= 1024 or (is_eod_maintenance and total_buffered >= 128)):
                all_grids = torch.cat(buffer_grids, dim=0)
                all_l0s = torch.cat(buffer_l0s, dim=0)
                all_params = torch.cat(buffer_params, dim=0)
                all_pis = torch.cat(buffer_pis, dim=0)
                all_regrets = torch.cat(buffer_regrets, dim=0)
                
                # Push to learner background thread (blocks if queue is full)
                learner_queue.put((all_grids, all_l0s, all_params, all_pis, all_regrets))
                
                buffer_grids.clear()
                buffer_l0s.clear()
                buffer_params.clear()
                buffer_regrets.clear()
                buffer_pis.clear()

    if not is_eval:
        # Flush remaining trades
        total_buffered = sum(x.size(0) for x in buffer_grids) if len(buffer_grids) > 0 else 0
        if total_buffered > 0:
            all_grids = torch.cat(buffer_grids, dim=0)
            all_l0s = torch.cat(buffer_l0s, dim=0)
            all_params = torch.cat(buffer_params, dim=0)
            all_pis = torch.cat(buffer_pis, dim=0)
            all_regrets = torch.cat(buffer_regrets, dim=0)
            learner_queue.put((all_grids, all_l0s, all_params, all_pis, all_regrets))
            
        # Shutdown Learner Thread
        learner_queue.put(None)
        learner_thread.join()
        
        batches = max(1, epoch_diagnostics["batches"])
        avg_loss = epoch_diagnostics["loss_sum"] / batches
        avg_grad = epoch_diagnostics["grad_norm_sum"] / batches
        print(f"\n[DIAGNOSTICS] Avg Loss: {avg_loss:.4f} | Avg Grad Norm: {avg_grad:.4f}\n")

    if is_eval:
        return trade_pnls, trade_durations, trade_mfe_avail, trade_mfe_trade, trade_mae
    return trade_pnls, trade_durations

def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[INFO] Windows sleep prevention activated.")
    except Exception as e:
        print(f"[WARN] Could not prevent sleep: {e}")

def allow_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("[INFO] Windows sleep prevention deactivated.")
    except Exception as e:
        pass

def run_walk_forward_curriculum():
    prevent_sleep()
    print("[INFO] Booting Research A Walk-Forward Curriculum...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    chunks = get_available_chunks(features_root, chunk_size=5)
    print(f"[INFO] Constructed {len(chunks)} chronological segments.")
    
    master_net = ResearchANetwork().to(device)
    vtrace = VTraceReconciliation()
    
    os.makedirs('checkpoints', exist_ok=True)

    # Auto-Resume from latest epoch if it exists
    latest_epoch_path = 'checkpoints/research_A_segment_1_latest_epoch.pth'
    if os.path.exists(latest_epoch_path):
        print(f"[TRANSFER] Resuming from {latest_epoch_path}")
        try:
            checkpoint = torch.load(latest_epoch_path, map_location=device, weights_only=True)
            master_net.lstm.load_state_dict(checkpoint['lstm'])
            master_net.load_state_dict(checkpoint['heads'], strict=False)
        except Exception as e:
            print(f"[WARN] Failed to load latest epoch: {e}. Starting fresh.")
    else:
        print("[INFO] No existing checkpoint found. Starting with fresh random initialization!")
    
    # We do a short test: only 1 week IS and 1 week OOS (Step 1)
    for idx in range(1):
        train_segment = chunks[idx]
        eval_segment = chunks[idx + 1]
        
        print(f"\n==============================================")
        print(f" WALK-FORWARD STEP {idx+1} (1-Week IS / 1-Week OOS)")
        print(f" TRAIN ON: {train_segment[0]} -> {train_segment[-1]}")
        print(f" EVALUATE ON: {eval_segment[0]} -> {eval_segment[-1]}")
        print(f"==============================================")
        
        config = load_config()
        lr = config.get("learning_rate", 0.001)
        optimizer = optim.Adam(master_net.parameters(), lr=lr)
        epochs_per_segment = config.get("epochs_per_segment", 5)
        
        diagnostics_suite = OOSDiagnosticsSuite()
        
        for epoch in range(1, epochs_per_segment + 1):
            
            print(f"[TRAIN] Running Epoch {epoch}/{epochs_per_segment} with LR={lr}...")
            fps_train = MultiDayForwardPassSystem(
                atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=train_segment
            )
            is_trade_pnls, is_trade_durations = run_quadrant_sim(fps_train, master_net, optimizer, vtrace, config, device, epoch_idx=epoch-1, is_eval=False)
            
            is_metrics = evaluate_is_metrics(is_trade_pnls, is_trade_durations)
            print(f"\n[IS EVAL] n: {is_metrics.get('metric_n', 0):.4f} | PnL: {is_metrics.get('total_pnl', 0):.2f} | PnL Mode CI: {is_metrics.get('pnl_mode_ci', (0,0))} | Dur Mode CI: {is_metrics.get('dur_mode_ci', (0,0))}")
            
            # Save a temporary checkpoint for the latest epoch regardless of passing status
            torch.save({
                'segment_idx': idx + 1,
                'epoch_idx': epoch,
                'lstm': master_net.lstm.state_dict(),
                'heads': master_net.state_dict()
            }, f'checkpoints/research_A_segment_{idx+1}_latest_epoch.pth')
            
        # IS loop finished. Unconditional checkpoint save.
        print(f"[INFO] IS budget reached ({epochs_per_segment} epochs). Saving unconditional checkpoint.")
        torch.save({
            'segment_idx': idx + 1,
            'lstm': master_net.lstm.state_dict(),
            'heads': master_net.state_dict()
        }, f'checkpoints/research_A_segment_{idx+1}.pth')

        # Run OOS Eval exactly once
        print(f"\n[EVAL] Running OOS Evaluation on Next Segment (Week 2)...")
        fps_eval = MultiDayForwardPassSystem(
            atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=eval_segment
        )
        trade_pnls, trade_durations, trade_mfe_avail, trade_mfe_trade, trade_mae = run_quadrant_sim(fps_eval, master_net, optimizer, vtrace, config, device, epoch_idx=epochs_per_segment, is_eval=True)
        
        seg_diag = diagnostics_suite.add_segment_data(trade_pnls, trade_durations, trade_mfe_avail, trade_mfe_trade, trade_mae)
        print(f"\n[OOS DIAGNOSTICS] Trades: {seg_diag.get('trade_count', 0)} | MaxDD: {seg_diag.get('max_drawdown', 0):.2f} | PnL Mode CI: {seg_diag.get('pnl_mode_ci', (0,0))}")
        print(f"[OOS DIAGNOSTICS] Cap vs Avail: {seg_diag.get('cap_vs_avail', 0):.2%} | Cap vs Trade: {seg_diag.get('cap_vs_trade', 0):.2%} | Avg MAE: {seg_diag.get('avg_mae', 0):.2f}")
        
        pooled_diag = diagnostics_suite.get_pooled_diagnostics()
        print(f"\n[POOLED AGGREGATE]")
        for k, v in pooled_diag.items():
            print(f"  {k}: {v}")
            
        # Memory Cleanup
        try:
            del fps_train
            del fps_eval
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        run_walk_forward_curriculum()
    finally:
        allow_sleep()
    
    print("[INFO] Curriculum Complete. Releasing GPU resources and exiting.")
    torch.cuda.empty_cache()
    sys.exit(0)
