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
import scipy.stats as stats

from network_research_A import ResearchANetwork
from vtrace_reconciliation import VTraceReconciliation
from curriculum_config import load_config, save_segment_metrics
from curriculum_metrics import evaluate_is_metrics, OOSDiagnosticsSuite

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem, ForwardPassSystem
from core_v2.features import TF_HIERARCHY_V2, FEATURE_NAMES_V2

DELTA_FEATURE_NAMES = [
    'z_se', 'hurst', 'reversion_prob', 'swing_noise', 'z_high', 'z_low', 'SE_high', 'SE_low',
    'price_mean_w', 'vwap_w', 'price_velocity_w', 'price_accel_w'
]

# Canonical Index Resolution
TF_1M_IDX = TF_HIERARCHY_V2.index('1m')
TF_5M_IDX = TF_HIERARCHY_V2.index('5m')

DELTA_FEAT_INDICES = []
for fname in DELTA_FEATURE_NAMES:
    try:
        idx = FEATURE_NAMES_V2.index(fname)
        DELTA_FEAT_INDICES.append(idx)
    except ValueError:
        pass # In case there's an exact naming deviation
DELTA_FEAT_INDICES = torch.tensor(DELTA_FEAT_INDICES, dtype=torch.long)

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
    try:
        while True:
            item = learner_queue.get()
            if item is None:  # Shutdown sentinel
                break
                
            all_grids, all_l0s, all_deltas, all_scalars, all_params, all_pis, all_regrets = item
            dataset_size = len(all_grids)
            batch_size = 512
            indices = torch.randperm(dataset_size, device='cpu')
            
            master_net.train()
            for idx_opt in range(0, dataset_size, batch_size):
                idx = indices[idx_opt:idx_opt+batch_size]
                b_grids = all_grids[idx].to(device)
                b_l0s = all_l0s[idx].to(device)
                b_deltas = all_deltas[idx].to(device)
                b_scalars = all_scalars[idx].to(device)
                b_params = all_params[idx].to(device)
                b_pis = all_pis[idx].to(device)
                b_regrets = all_regrets[idx].to(device)
                
                optimizer.zero_grad()
                q_action, _ = master_net(b_grids, b_l0s, delta_features=b_deltas, scalar_context=b_scalars)
                
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
    except Exception as e:
        import traceback
        with open("learner_crash.log", "w") as f:
            f.write(traceback.format_exc())
        print(f"[FATAL] Learner thread crashed: {e}. Check learner_crash.log for traceback.")

def run_quadrant_sim(fps, master_net, optimizer, vtrace, config, device, epoch_idx=0, is_eval=False):
    global global_all_dpnls, global_all_mtm_rewards, global_holds_unl_neg_rewards, global_long_trades
    global_all_dpnls = []
    global_all_mtm_rewards = []
    global_holds_unl_neg_rewards = []
    global_long_trades = []
    
    print(f"\n[EVAL]" if is_eval else f"\n[TRAIN] Running Epoch {epoch_idx} with LR={optimizer.param_groups[0]['lr']}...")
    trade_pnls = []
    trade_durations = []
    trade_mfe_avail = []
    trade_mfe_trade = []
    trade_mae = []
    N_AGENTS = 128
    meta_critic_multiplier = 1.0
    
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
    last_pnls = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    last_q_max = torch.zeros(N_AGENTS, dtype=torch.float32, device=device) # For TD Bootstrapping
    entry_stacks = torch.zeros((N_AGENTS, 8, 25), dtype=torch.float32, device=device)
    
    terminal_regret = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    exited_last_step = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
    
    if is_eval:
        efactor = torch.ones(N_AGENTS, device=device)
        ecuriosity = torch.zeros(N_AGENTS, device=device)
        epsilon = torch.zeros(N_AGENTS, device=device)
    else:
        # EFactor: Natural aggressiveness profile per agent U(0.05, 0.95) to prevent clustering at clamp bounds
        efactor = torch.rand(N_AGENTS, device=device) * 0.90 + 0.05
        
        # ECuriosity: Natural curiosity profile per agent (Epsilon modifier) U(0.05, 1.95)
        ecuriosity = torch.rand(N_AGENTS, device=device) * 1.90 + 0.05
        
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
    trade_entry_grids = torch.zeros((N_AGENTS, 8, 60, 25), dtype=torch.float32, device=device)
    trade_entry_l0s = torch.zeros((N_AGENTS, 60, 3), dtype=torch.float32, device=device)
    trade_entry_params = torch.full((N_AGENTS, 1), -1, dtype=torch.int64, device=device)
    trade_entry_pis = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    trade_entry_deltas = torch.zeros((N_AGENTS, 24), dtype=torch.float32, device=device)
    trade_entry_scalars = torch.zeros((N_AGENTS, 3), dtype=torch.float32, device=device)
    
    gross_profits = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    gross_losses = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    trade_counts = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
    
    learner_queue = None
    learner_thread = None
    epoch_diagnostics = {"loss_sum": 0.0, "grad_norm_sum": 0.0, "batches": 0} if not is_eval else None
    if not is_eval:
        learner_queue = queue.Queue(maxsize=3)
        learner_thread = threading.Thread(
            target=learner_worker,
            args=(learner_queue, master_net, optimizer, vtrace, config.get('temperature', 0.1), device, epoch_diagnostics, meta_critic_multiplier),
            daemon=True
        )
        learner_thread.start()
        
    buffer_grids, buffer_l0s, buffer_deltas, buffer_scalars, buffer_params, buffer_regrets, buffer_pis = [], [], [], [], [], [], []
    
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
        
        # Native Anti-Scramble Grid Parity
        grid_all = ticker._v2_grid
        
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
        
        # --- VECTORIZED GPU FORWARD PASS ---
        # Unfold the sequence to create sliding windows of length 60
        grid_windows = grid_all_tensor.unfold(0, 60, 1).transpose(2, 3).contiguous()
        l0_windows = l0_all_tensor.unfold(0, 60, 1).transpose(1, 2).contiguous()
        
        num_windows = grid_windows.size(0)
        batch_size_fp = 4096
        
        # Precompute all neural network forward passes for the day
        master_net.eval()
        all_q_actions = []
        with torch.no_grad():
            for start_idx in range(0, num_windows, batch_size_fp):
                end_idx = min(start_idx + batch_size_fp, num_windows)
                gb = grid_windows[start_idx:end_idx]
                lb = l0_windows[start_idx:end_idx]
                q_act, _ = master_net(gb, lb)
                all_q_actions.append(q_act)
        all_q_actions = torch.cat(all_q_actions, dim=0)
        if not is_eval:
            master_net.train()
        # -----------------------------------
        
        for i in range(59, N_BARS):
            current_price = float(price_arr[i])
            sec = sec_in_day[i]
            is_eod_maintenance = (75300 <= sec < 79200)
            
            # Use precomputed window for buffer storage
            grid_tensor = grid_windows[i-59].unsqueeze(0)
            l0_tensor = l0_windows[i-59].unsqueeze(0)
            
            # --- Position-Aware Delta Feature Computation ---
            # Current stack is the last bar in the window: grid_tensor[0, :, -1, :]
            current_stack = grid_tensor[0, :, -1, :]
            
            # Extract chosen subset [TF_idx, Feat_idx]
            curr_1m = current_stack[TF_1M_IDX, DELTA_FEAT_INDICES]
            curr_5m = current_stack[TF_5M_IDX, DELTA_FEAT_INDICES]
            curr_flat = torch.cat([curr_1m, curr_5m], dim=-1) # [24]
            
            # Entry stack
            entry_1m = entry_stacks[:, TF_1M_IDX, DELTA_FEAT_INDICES]
            entry_5m = entry_stacks[:, TF_5M_IDX, DELTA_FEAT_INDICES]
            entry_flat = torch.cat([entry_1m, entry_5m], dim=-1) # [N_AGENTS, 24]
            
            # Delta logic
            raw_delta = curr_flat.unsqueeze(0).expand(N_AGENTS, -1) - entry_flat
            delta_features = torch.where(in_position.unsqueeze(1), raw_delta, torch.zeros_like(raw_delta))
            
            # Scalar Context: direction (-1, 0, 1), unl_pnl_norm, age_norm
            dir_signed = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
            dir_signed[directions == 1] = 1.0
            dir_signed[directions == 2] = -1.0
            
            price_diff = current_price - entry_prices
            unl_pnl_raw = torch.where(directions == 1, price_diff * 50.0, -price_diff * 50.0)
            unl_pnl = torch.where(in_position, unl_pnl_raw, torch.zeros_like(unl_pnl_raw))
            unl_pnl_norm = torch.clamp(unl_pnl / 100.0, -2.0, 2.0) # Assume $100 TP target
            age_norm = torch.clamp(position_age.float() / 60.0, 0.0, 5.0)
            
            scalar_context = torch.stack([dir_signed, unl_pnl_norm, age_norm], dim=-1)
            # ------------------------------------------------
            
            with torch.no_grad():
                # We fetch precomputed flat forward pass for flat agents
                q_flat = all_q_actions[i-59].expand(N_AGENTS, -1)
                
                # For in-position agents, we must compute live dynamic q_actions
                # We reuse the cached LSTM state (final_step) since the CNN+LSTM depends only on grid/l0
                if torch.any(in_position):
                    # In network_research_A, we know the forward handles this post-LSTM.
                    # Since we can't extract the LSTM hidden easily without a network rewrite,
                    # we will just do a fast batched inference for in_position agents!
                    pos_mask = in_position
                    pos_count = pos_mask.sum().item()
                    gb = grid_tensor.expand(pos_count, -1, -1, -1)
                    lb = l0_tensor.expand(pos_count, -1, -1)
                    db = delta_features[pos_mask]
                    sb = scalar_context[pos_mask]
                    q_pos, _ = master_net(gb, lb, delta_features=db, scalar_context=sb)
                    
                    q_action = q_flat.clone()
                    q_action[pos_mask] = q_pos
                else:
                    q_action = q_flat
                
                # Save max Q for the next step TD target
                current_q_max, greedy_acts = torch.max(q_action, dim=1)
                
                # --- Sanity Print for Delta Features ---
                if i % 5000 == 0 and torch.any(in_position):
                    active_agent = in_position.nonzero(as_tuple=True)[0][0]
                    print(f"\n[DEBUG] Bar {i} | Agent {active_agent} | Age: {position_age[active_agent]} | Delta Max: {delta_features[active_agent].abs().max().item():.4f}")
                # ---------------------------------------
                
                explore_mask = torch.rand(N_AGENTS, device=device) < epsilon
                
                head_size = q_action.size(1)
                probs = F.softmax(q_action / temperature, dim=1)
                random_acts = torch.randint(0, head_size, (N_AGENTS,), device=device)
                
                nn_action = torch.where(explore_mask, random_acts, greedy_acts)
                joint_pi = torch.where(explore_mask, torch.tensor(1.0 / head_size, device=device), probs.gather(1, greedy_acts.unsqueeze(1)).squeeze(1))
                    
            nn_wants_flat = (nn_action == 0) | ((directions == 1) & (nn_action == 2)) | ((directions == 2) & (nn_action == 1))
            
            # Pure Q4 Routing (NN Entry + NN Exit)
            wants_entry_long = (nn_action == 1)
            wants_entry_short = (nn_action == 2)
            custom_exit = nn_wants_flat
            
            # --- G0 TRACKING LOGIC ---
            dpnl_full = unl_pnl - last_pnls
            mtm_full = torch.where(in_position, torch.sign(dpnl_full) * torch.log1p(torch.abs(dpnl_full)), torch.zeros_like(dpnl_full))
            
            if torch.any(in_position):
                active_agents = in_position.nonzero(as_tuple=True)[0]
                for ag in active_agents:
                    ag_id = ag.item()
                    d = dpnl_full[ag_id].item()
                    m = mtm_full[ag_id].item()
                    global_all_dpnls.append(d)
                    global_all_mtm_rewards.append(m)
                    
                    # If holding a loser
                    act = nn_action[ag_id].item()
                    dir_val = directions[ag_id].item()
                    c_exit = (act == 0) or (dir_val == 1 and act == 2) or (dir_val == 2 and act == 1)
                    if not c_exit and unl_pnl[ag_id].item() < 0:
                        global_holds_unl_neg_rewards.append(m)
            # --------------------------
            
            # Sequential Logging & Dense MTM Rewards
            # 1. Resolve pending transitions (agents that were active in the PREVIOUS step)
            active_last_step = (trade_entry_params[:, 0] != -1) # Using -1 as null
            if not is_eval and torch.any(active_last_step):
                # We have previous transitions to log!
                # TD Target: r_t + gamma * max Q(s_{t+1})
                
                # Dense MTM for HOLD steps
                dpnl = unl_pnl - last_pnls
                # If they were in position, reward is MTM. If they were flat, reward is 0.
                mtm_reward = torch.where(in_position, torch.sign(dpnl) * torch.log1p(torch.abs(dpnl)), torch.zeros_like(dpnl))
                
                # TD Target logic
                GAMMA = 0.997
                target_q = mtm_reward + GAMMA * current_q_max
                
                # Override TD target for terminal exit actions with the pure regret penalty (no bootstrap)
                target_q = torch.where(exited_last_step, terminal_regret, target_q)
                
                buffer_grids.append(trade_entry_grids[active_last_step].clone().cpu())
                buffer_l0s.append(trade_entry_l0s[active_last_step].clone().cpu())
                buffer_deltas.append(trade_entry_deltas[active_last_step].clone().cpu())
                buffer_scalars.append(trade_entry_scalars[active_last_step].clone().cpu())
                buffer_params.append(trade_entry_params[active_last_step].clone().cpu())
                buffer_pis.append(trade_entry_pis[active_last_step].clone().cpu())
                buffer_regrets.append(target_q[active_last_step].clone().cpu())
                
                # Clear terminal state latches
                exited_last_step.fill_(False)
                terminal_regret.fill_(0.0)
            
            # Wipe transition latch
            trade_entry_params.fill_(-1)
            
            # Check exits
            if torch.any(in_position):
                position_age[in_position] += 1
                
                hit_sl = unl_pnl < -50.0
                hit_tp = unl_pnl > 100.0
                
                is_last_bar = (i == N_BARS - 1)
                force_exit = is_eod_maintenance or is_last_bar
                eod_exit = torch.tensor(force_exit, device=device).expand(N_AGENTS)
                
                exit_mask = in_position & (hit_sl | hit_tp | eod_exit | custom_exit)
                
                if torch.any(exit_mask):
                    exit_pnls = unl_pnl[exit_mask] - 5.0
                    exit_durs = position_age[exit_mask]
                    agents_exiting = exit_mask.nonzero(as_tuple=True)[0]
                    
                    # --- G0 TRACKING LOGIC (C) ---
                    for local_idx in range(len(agents_exiting)):
                        agent_id = agents_exiting[local_idx].item()
                        if directions[agent_id].item() == 1: # Long
                            p_diff = current_price - entry_prices[agent_id].item()
                            global_long_trades.append((p_diff, exit_pnls[local_idx].item()))
                    # -----------------------------
                    
                    profits = torch.where(exit_pnls > 0, exit_pnls, torch.tensor(0.0, device=device))
                    losses = torch.where(exit_pnls <= 0, torch.abs(exit_pnls), torch.tensor(0.0, device=device))
                    gross_profits[agents_exiting] += profits
                    gross_losses[agents_exiting] += losses
                    trade_counts[agents_exiting] += 1
                    
                    F_FRICTION = 18.0
                    LAMBDA = 0.4
                    H_LOOKAHEAD = 27
                    
                    optimal_pnls = torch.zeros_like(exit_pnls)
                    
                    mfe_avail_list = []
                    mfe_trade_list = []
                    mae_list = []
                    
                    # CAUSALITY FIREWALL: 
                    # The following loop looks ahead into future bars `price_path_avail` to find optimal exits.
                    # This is strictly used for the `gap_after_exit` penalty and NEVER feeds back into network input.
                    for local_idx in range(len(agents_exiting)):
                        agent_id = agents_exiting[local_idx].item()
                        dur = exit_durs[local_idx].item()
                        entry_bar_idx = i - dur
                        lookahead_end = min(i + H_LOOKAHEAD, N_BARS) # Post-exit optimal availability
                        
                        price_path_avail = price_arr[i:lookahead_end]
                        price_path_trade = price_arr[entry_bar_idx:i+1] # up to current bar
                        agent_dir = directions[agent_id].item()
                        entry_p = entry_prices[agent_id].item()
                        
                        if agent_dir == 1: # Long
                            max_fav_avail = np.max(price_path_avail)
                            mfe_avail = ((max_fav_avail - current_price) * 50.0)
                            max_fav_trade = np.max(price_path_trade)
                            mfe_trade = ((max_fav_trade - entry_p) * 50.0) - 5.0
                            min_price = np.min(price_path_trade)
                            mae = ((entry_p - min_price) * 50.0)
                        else: # Short
                            min_fav_avail = np.min(price_path_avail)
                            mfe_avail = ((current_price - min_fav_avail) * 50.0)
                            min_fav_trade = np.min(price_path_trade)
                            mfe_trade = ((entry_p - min_fav_trade) * 50.0) - 5.0
                            max_price = np.max(price_path_trade)
                            mae = ((max_price - entry_p) * 50.0)
                            
                        optimal_pnls[local_idx] = mfe_avail
                        mfe_avail_list.append(max(0.0, mfe_avail))
                        mfe_trade_list.append(max(0.0, mfe_trade))
                        mae_list.append(max(0.0, mae))

                    # Terminal Exit Penalty
                    gap = torch.clamp(optimal_pnls, min=0.0) # Opportunity missed after exit
                    regret_term = -LAMBDA * torch.log1p(gap)
                    
                    if not is_eval:
                        # Store the terminal regret to be logged as the precise target_q for this exit action on the next step
                        terminal_regret[agents_exiting] = regret_term.to(device)
                        exited_last_step[exit_mask] = True
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
                    unl_pnl[exit_mask] = 0.0
                    
            # Update the last_pnl latch
            last_pnls = unl_pnl.clone()
                    
            # Check entries
            if not is_eod_maintenance and i < N_BARS - 1:
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
                    
                    # Store entry snapshot
                    entry_stacks[entry_mask] = current_stack.expand(N_AGENTS, -1, -1)[entry_mask]
                    
            # 2. Latch current transitions for NEXT step's target calculation
            if not is_eval and not is_eod_maintenance and i < N_BARS - 1:
                # We log decisions for ALL agents (both in-position and flat!)
                latch_mask = torch.ones(N_AGENTS, dtype=torch.bool, device=device)
                expanded_grid = grid_tensor.expand(N_AGENTS, -1, -1, -1)
                expanded_l0 = l0_tensor.expand(N_AGENTS, -1, -1)
                trade_entry_grids[latch_mask] = expanded_grid[latch_mask]
                trade_entry_l0s[latch_mask] = expanded_l0[latch_mask]
                trade_entry_deltas[latch_mask] = delta_features[latch_mask]
                trade_entry_scalars[latch_mask] = scalar_context[latch_mask]
                trade_entry_params[latch_mask, 0] = nn_action[latch_mask]
                trade_entry_pis[latch_mask] = joint_pi[latch_mask]
                    
            # Train optimization using V-Trace Reconciliation (Async)
            total_buffered = sum(x.size(0) for x in buffer_grids) if len(buffer_grids) > 0 else 0
            if not is_eval and (total_buffered >= 256 or (is_eod_maintenance and total_buffered >= 64)):
                all_grids = torch.cat(buffer_grids, dim=0)
                all_l0s = torch.cat(buffer_l0s, dim=0)
                all_deltas = torch.cat(buffer_deltas, dim=0)
                all_scalars = torch.cat(buffer_scalars, dim=0)
                all_params = torch.cat(buffer_params, dim=0)
                all_pis = torch.cat(buffer_pis, dim=0)
                all_regrets = torch.cat(buffer_regrets, dim=0)
                
                # Push to learner background thread (blocks if queue is full)
                try:
                    learner_queue.put((all_grids, all_l0s, all_deltas, all_scalars, all_params, all_pis, all_regrets), timeout=60)
                except queue.Full:
                    print("\n[FATAL] Learner queue timeout. Background thread hung/crashed!")
                    sys.exit(1)
                
                buffer_grids.clear()
                buffer_l0s.clear()
                buffer_deltas.clear()
                buffer_scalars.clear()
                buffer_params.clear()
                buffer_regrets.clear()
                buffer_pis.clear()
                
        # Force garbage collection of memory-heavy Pandas/PyArrow dataframes
        import gc
        gc.collect()

    if not is_eval:
        # Flush remaining trades
        total_buffered = sum(x.size(0) for x in buffer_grids) if len(buffer_grids) > 0 else 0
        if total_buffered > 0:
            all_grids = torch.cat(buffer_grids, dim=0)
            all_l0s = torch.cat(buffer_l0s, dim=0)
            all_deltas = torch.cat(buffer_deltas, dim=0)
            all_scalars = torch.cat(buffer_scalars, dim=0)
            all_params = torch.cat(buffer_params, dim=0)
            all_pis = torch.cat(buffer_pis, dim=0)
            all_regrets = torch.cat(buffer_regrets, dim=0)
            learner_queue.put((all_grids, all_l0s, all_deltas, all_scalars, all_params, all_pis, all_regrets))
            
        # Shutdown Learner Thread
        learner_queue.put(None)
        learner_thread.join()
        
        batches = max(1, epoch_diagnostics["batches"])
        avg_loss = epoch_diagnostics["loss_sum"] / batches
        avg_grad = epoch_diagnostics["grad_norm_sum"] / batches
        print(f"\n[DIAGNOSTICS] Avg Loss: {avg_loss:.4f} | Avg Grad Norm: {avg_grad:.4f}\n")

    if is_eval:
        print("\n" + "="*50)
        print("STEP-0 GATE: Reward/Direction Sign Sentinel Report")
        print("="*50)
        
        # A
        if global_holds_unl_neg_rewards:
            mean_a = np.mean(global_holds_unl_neg_rewards)
            print(f"A. Sign-bug sentinel: mean(mtm_reward) over action=HOLD & unl_pnl<0 : {mean_a:.4f}")
        else:
            print("A. Sign-bug sentinel: No data (no hold&neg events)")
            print("A. Sign-bug sentinel: No data (no hold&neg events)")
            
        # B
        if len(global_all_dpnls) > 1:
            corr_b, _ = stats.pearsonr(global_all_dpnls, global_all_mtm_rewards)
            print(f"B. Reward-vs-dpnl correlation: {corr_b:.4f}")
        else:
            print("B. Reward-vs-dpnl correlation: Not enough data")
            
        # C
        if global_long_trades:
            profited_on_rise = sum(1 for (diff, pnl) in global_long_trades if (diff > 0 and pnl > 0) or (diff <= 0 and pnl <= 0))
            pct_c = profited_on_rise / len(global_long_trades) * 100
            print(f"C. Direction mapping check: {pct_c:.1f}% of Longs profited correctly based on price movement.")
        else:
            print("C. Direction mapping check: No long trades recorded.")
        print("="*50 + "\n")
        
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
    
    # Just run Segment 1 Evaluation (Index 0)
    idx = 0
    train_segment, eval_segment = chunks[idx], chunks[idx+1]
    
    print(f"\n==============================================")
    print(f" FAST DIAGNOSTIC G0: EVAL ONLY (Segment {idx+1})")
    print(f" EVALUATE ON: {eval_segment[0]} -> {eval_segment[-1]}")
    print(f"==============================================\n")
    
    config = load_config()
    fps_eval = MultiDayForwardPassSystem(
        atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=eval_segment
    )
    # Using 0 epochs since we don't train
    run_quadrant_sim(fps_eval, master_net, None, vtrace, config, device, epoch_idx=0, is_eval=True)

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
