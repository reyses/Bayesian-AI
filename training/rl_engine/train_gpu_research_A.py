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
import subprocess

def get_dynamic_n_agents(target_total_mb=10500, mb_per_agent=35, min_agents=4, max_agents=128):
    try:
        torch.cuda.empty_cache()
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']).decode('utf-8')
        used_mb = int(output.strip().split('\n')[0])
        available_mb = target_total_mb - used_mb
        if available_mb <= 0:
            return min_agents
        calc = available_mb // mb_per_agent
        return int(max(min_agents, min(max_agents, calc)))
    except Exception as e:
        print(f"[WARNING] Dynamic scaling failed: {e}. Defaulting to {min_agents} agents.")
        return min_agents

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

def learner_worker(learner_queue, master_net, optimizer, vtrace, temperature, device, epoch_diagnostics=None, meta_critic_multiplier=1.0, accum_steps=4):
    """Background thread that continuously pulls trade batches and runs V-Trace backprop."""
    try:
        scaler = torch.cuda.amp.GradScaler()
        while True:
            item = learner_queue.get()
            if item is None:  # Shutdown sentinel
                break
                
            all_grids, all_l0s, all_deltas, all_scalars, all_params, all_pis, all_regrets = item
            dataset_size = len(all_grids)
            batch_size = 512
            indices = torch.randperm(dataset_size, device='cpu')
            
            master_net.train()
            optimizer.zero_grad()
            step_count = 0
            
            for idx_opt in range(0, dataset_size, batch_size):
                idx = indices[idx_opt:idx_opt+batch_size]
                b_grids = all_grids[idx].to(device)
                b_l0s = all_l0s[idx].to(device)
                b_deltas = all_deltas[idx].to(device)
                b_scalars = all_scalars[idx].to(device)
                b_params = all_params[idx].to(device)
                b_pis = all_pis[idx].to(device)
                b_regrets = all_regrets[idx].to(device)
                
                with torch.cuda.amp.autocast():
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
                    loss = loss / accum_steps

                scaler.scale(loss).backward()
                step_count += 1
                
                if step_count % accum_steps == 0 or (idx_opt + batch_size >= dataset_size):
                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(master_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                    if epoch_diagnostics is not None:
                        epoch_diagnostics["loss_sum"] += loss.item() * accum_steps
                        epoch_diagnostics["grad_norm_sum"] += total_norm.item()
                        epoch_diagnostics["batches"] += 1
    except Exception as e:
        import traceback
        with open("learner_crash.log", "w") as f:
            f.write(traceback.format_exc())
        print(f"[FATAL] Learner thread crashed: {e}. Check learner_crash.log for traceback.")

def run_quadrant_sim(fps, master_net, optimizer, vtrace, config, device, epoch_idx=0, N_AGENTS=128, is_eval=False, meta_critic_multiplier=1.0, accum_steps=4):
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
    last_pnls = torch.zeros(N_AGENTS, dtype=torch.float32, device=device)
    last_q_max = torch.zeros(N_AGENTS, dtype=torch.float32, device=device) # For TD Bootstrapping
    entry_stacks = torch.zeros((N_AGENTS, 8, 25), dtype=torch.float32, device=device)
    
    trade_was_explored = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
    win_ticks = 0
    win_exit_ticks = 0
    loss_ticks = 0
    loss_exit_ticks = 0
    
    greedy_trade_metadata = []
    
    exited_last_step = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
    
    daily_bar_probs = []
    
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
            args=(learner_queue, master_net, optimizer, vtrace, config.get('temperature', 0.1), device, epoch_diagnostics, meta_critic_multiplier, accum_steps),
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
        segment_risk_tensor = torch.tensor(ticker._segment_risk, dtype=torch.float32, device=device)
        
        l0_all_tensor = torch.nan_to_num(l0_all_tensor, nan=0.0)
        grid_all_tensor = torch.nan_to_num(grid_all_tensor, nan=0.0)
        segment_risk_tensor = torch.nan_to_num(segment_risk_tensor, nan=0.0)
        
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
                
                # Base scalar context for entries (no unl_pnl, no age)
                # Lookahead exposure (segment_risk_tensor) has been removed!
                base_scalar = torch.zeros(end_idx - start_idx, 3, device=device)
                
                q_act, _ = master_net(gb, lb, scalar_context=base_scalar)
                all_q_actions.append(q_act)
        all_q_actions = torch.cat(all_q_actions, dim=0)
        if not is_eval:
            master_net.train()
        # -----------------------------------
        
        day_probs_list = []
        
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
            
            # Lookahead exposure (segment_risk_tensor) has been removed!
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
                if is_eval:
                    day_probs_list.append(probs[0].cpu().numpy())
                random_acts = torch.randint(0, head_size, (N_AGENTS,), device=device)
                
                nn_action = torch.where(explore_mask, random_acts, greedy_acts)
                joint_pi = torch.where(explore_mask, torch.tensor(1.0 / head_size, device=device), probs.gather(1, greedy_acts.unsqueeze(1)).squeeze(1))
                    
            nn_wants_flat = (nn_action == 0) | ((directions == 1) & (nn_action == 2)) | ((directions == 2) & (nn_action == 1))
            
            if not is_eval and torch.any(in_position):
                greedy_inpos = in_position & ~explore_mask
                if torch.any(greedy_inpos):
                    win_mask = greedy_inpos & (unl_pnl > 0)
                    loss_mask = greedy_inpos & (unl_pnl < 0)
                    
                    win_ticks += win_mask.sum().item()
                    win_exit_ticks += (win_mask & nn_wants_flat).sum().item()
                    
                    loss_ticks += loss_mask.sum().item()
                    loss_exit_ticks += (loss_mask & nn_wants_flat).sum().item()
                    
            trade_was_explored[in_position] |= explore_mask[in_position]
            
            # Pure Q4 Routing (NN Entry + NN Exit)
            wants_entry_long = (nn_action == 1)
            wants_entry_short = (nn_action == 2)
            custom_exit = nn_wants_flat
            
            # Sequential Logging & Dense MTM Rewards
            # 1. Resolve pending transitions (agents that were active in the PREVIOUS step)
            active_last_step = (trade_entry_params[:, 0] != -1) # Using -1 as null
            if not is_eval and torch.any(active_last_step):
                # We have previous transitions to log!
                # TD Target: r_t + gamma * max Q(s_{t+1})
                
                # Core Reward: Asymmetric Per-Step MTM
                dpnl = unl_pnl - last_pnls
                SCALE = 100.0
                BETA = 2.0
                
                r_t = torch.zeros_like(dpnl)
                pos_mask = (dpnl >= 0)
                neg_mask = (dpnl < 0)
                
                r_t[pos_mask] = dpnl[pos_mask] / SCALE
                r_t[neg_mask] = -BETA * torch.abs(dpnl[neg_mask]) / SCALE
                
                # Risk-Sensitivity Tilt (Underwater carrying cost)
                KAPPA = 0.5
                P = 2.0
                DEADBAND = 10.0
                
                deep_losers = (unl_pnl < -DEADBAND)
                if torch.any(deep_losers):
                    penalty = -KAPPA * ((torch.abs(unl_pnl[deep_losers]) - DEADBAND) / SCALE)**P
                    r_t[deep_losers] += penalty
                
                pristine_mask = (segment_risk_tensor[i, 0] == 1.0)
                chaos_mask = (segment_risk_tensor[i, 1] == 1.0)
                
                inactivity_penalty = torch.full_like(r_t, -0.001)
                
                # Heavy penalty for sitting out Pristine trends
                if pristine_mask:
                    inactivity_penalty -= 0.005
                
                # Small reward for staying out of Chaos
                if chaos_mask:
                    inactivity_penalty += 0.001
                
                r_t = torch.where(~in_position, inactivity_penalty, r_t)
                
                # TD Target logic
                GAMMA = 0.997
                target_q = r_t + GAMMA * current_q_max
                
                # Exiting simply stops accruing reward. Terminal future Q bootstrap is 0.
                target_q = torch.where(exited_last_step, r_t, target_q)
                
                buffer_grids.append(trade_entry_grids[active_last_step].clone().cpu())
                buffer_l0s.append(trade_entry_l0s[active_last_step].clone().cpu())
                buffer_deltas.append(trade_entry_deltas[active_last_step].clone().cpu())
                buffer_scalars.append(trade_entry_scalars[active_last_step].clone().cpu())
                buffer_params.append(trade_entry_params[active_last_step].clone().cpu())
                buffer_pis.append(trade_entry_pis[active_last_step].clone().cpu())
                buffer_regrets.append(target_q[active_last_step].clone().cpu())
                
                # Clear terminal state latches
                exited_last_step.fill_(False)
            
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
                    
                    profits = torch.where(exit_pnls > 0, exit_pnls, torch.tensor(0.0, device=device))
                    losses = torch.where(exit_pnls <= 0, torch.abs(exit_pnls), torch.tensor(0.0, device=device))
                    gross_profits[agents_exiting] += profits
                    gross_losses[agents_exiting] += losses
                    trade_counts[agents_exiting] += 1
                    
                    H_LOOKAHEAD = 27
                    
                    mfe_avail_list = []
                    mfe_trade_list = []
                    mae_list = []
                    
                    # Compute MFE / MAE strictly as read-only diagnostic scorecards (no reward path input)
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
                            
                        mfe_avail_list.append(max(0.0, mfe_avail))
                        mfe_trade_list.append(max(0.0, mfe_trade))
                        mae_list.append(max(0.0, mae))

                    if not is_eval:
                        exited_last_step[exit_mask] = True
                        
                    cpu_pnls = exit_pnls.cpu().numpy()
                    cpu_durs = exit_durs.cpu().numpy()
                    cpu_was_exp = trade_was_explored[exit_mask].cpu().numpy()
                    
                    for k_idx in range(len(agents_exiting)):
                        if not is_eval and cpu_was_exp[k_idx]:
                            continue
                            
                        trade_pnls.append(cpu_pnls[k_idx])
                        trade_durations.append(cpu_durs[k_idx])
                        trade_mfe_avail.append(mfe_avail_list[k_idx])
                        trade_mfe_trade.append(mfe_trade_list[k_idx])
                        trade_mae.append(mae_list[k_idx])
                        
                        agent_id = agents_exiting[k_idx].item()
                        agent_dir = directions[agent_id].item()
                        exit_bar = i
                        entry_bar = i - cpu_durs[k_idx]
                        greedy_trade_metadata.append((day, entry_bar, exit_bar, agent_dir))
                        
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
                    trade_was_explored[entry_mask] = explore_mask[entry_mask]
                    directions[entry_mask] = torch.where(wants_entry_long[entry_mask], torch.tensor(1, dtype=torch.int32, device=device), torch.tensor(2, dtype=torch.int32, device=device))
                    entry_prices[entry_mask] = current_price
                    position_age[entry_mask] = 0
                    
                    # Store entry snapshot
                    entry_stacks[entry_mask] = current_stack.expand(N_AGENTS, -1, -1)[entry_mask]
                    
            # 2. Latch current transitions for NEXT step's target calculation
            if not is_eval and not is_eod_maintenance and i < N_BARS - 1:
                # Sub-sample flat actions to save massive memory overhead, but log all active/entry/exit actions
                is_active = in_position | exited_last_step | (trade_entry_params[:, 0] != -1)
                random_sample = torch.rand(N_AGENTS, device=device) < 0.01
                latch_mask = is_active | random_sample
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
            if not is_eval and (total_buffered >= 8192 or (is_eod_maintenance and total_buffered >= 64)):
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
                
        if is_eval:
            daily_bar_probs.append({ 'day': day, 'probs': np.array(day_probs_list), 'start_idx': 59, 'end_idx': N_BARS-1 })
                
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

    return {
        'pnls': trade_pnls,
        'durations': trade_durations,
        'mfe_avail': trade_mfe_avail,
        'mfe_trade': trade_mfe_trade,
        'mae': trade_mae,
        'metadata': greedy_trade_metadata,
        'win_ticks': win_ticks,
        'win_exit_ticks': win_exit_ticks,
        'loss_ticks': loss_ticks,
        'loss_exit_ticks': loss_exit_ticks,
        'daily_bar_probs': daily_bar_probs
    }

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

def evaluate_is_mastery_gate(is_metrics):
    pnls = np.array(is_metrics['pnls'])
    metadata = is_metrics['metadata']
    durs = np.array(is_metrics['durations'])
    mae = np.array(is_metrics['mae'])
    mfe_avail = np.array(is_metrics['mfe_avail'])
    
    unique_trades = set(metadata)
    effective_N = len(unique_trades)
    raw_N = len(pnls)
    
    if effective_N < 100:
        return False, effective_N, raw_N, ["Sparsity (N_eff < 100)"]
        
    failed_conditions = []
    
    # 1. Edge floor CI > 0
    mean_pnl = np.mean(pnls)
    stderr = np.std(pnls) / np.sqrt(raw_N) if raw_N > 1 else 0
    ci_lower = mean_pnl - 1.96 * stderr
    if ci_lower <= 0:
        failed_conditions.append(f"Edge Floor CI ({ci_lower:.2f} <= 0)")
        
    # 2. Disposition <= 0
    win_ticks = is_metrics['win_ticks']
    win_exit_ticks = is_metrics['win_exit_ticks']
    loss_ticks = is_metrics['loss_ticks']
    loss_exit_ticks = is_metrics['loss_exit_ticks']
    
    if win_ticks > 0 and loss_ticks > 0:
        exit_rate_win = win_exit_ticks / win_ticks
        exit_rate_loss = loss_exit_ticks / loss_ticks
        disposition = exit_rate_win - exit_rate_loss
        if disposition > 0:
            failed_conditions.append(f"Disposition ({disposition:.4f} > 0)")
    else:
        failed_conditions.append("Disposition (No ticks)")
        
    # 3. Hold-gap <= 0
    winners = (pnls > 0)
    losers = (pnls <= 0)
    if np.sum(winners) > 0 and np.sum(losers) > 0:
        mean_dur_win = np.mean(durs[winners])
        mean_dur_loss = np.mean(durs[losers])
        hold_gap = mean_dur_loss - mean_dur_win
        if hold_gap > 0:
            failed_conditions.append(f"Hold Gap ({hold_gap:.1f} > 0)")
            
    # 4. Loser MAE <= 15
    if np.sum(losers) > 0:
        mean_loser_mae = np.mean(mae[losers])
        if mean_loser_mae > 15.0:
            failed_conditions.append(f"Loser MAE ({mean_loser_mae:.2f} > 15)")
            
    # 5. Tail CVaR >= -40
    pct_5 = max(1, int(raw_N * 0.05))
    worst_5_pct = np.sort(pnls)[:pct_5]
    es_5 = np.mean(worst_5_pct)
    if es_5 < -40.0:
        failed_conditions.append(f"Tail CVaR ({es_5:.2f} < -40)")
        
    # 6. Capture >= 0.0
    sum_mfe_avail = np.sum(mfe_avail)
    if sum_mfe_avail > 0:
        capture = np.sum(pnls) / sum_mfe_avail
        if capture < 0.0:
            failed_conditions.append(f"Capture ({capture:.4f} < 0)")
    else:
        failed_conditions.append("Capture (0 MFE Avail)")
            
    return len(failed_conditions) == 0, effective_N, raw_N, failed_conditions

def run_walk_forward_curriculum(start_segment=1, target_vram=10500, accum_steps=4):
    prevent_sleep()
    print(f"[INFO] Booting Research A Walk-Forward Curriculum... (Starting at Segment {start_segment})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    chunks = get_available_chunks(features_root, chunk_size=5)
    print(f"[INFO] Constructed {len(chunks)} chronological segments.")
    
    master_net = ResearchANetwork(lstm_hidden=128).to(device)
    vtrace = VTraceReconciliation(rho_bar=2.0128, c_bar=3.3986)
    
    os.makedirs('checkpoints', exist_ok=True)

    # Auto-Resume from latest epoch if it exists
    if start_segment > 1:
        # Try to load the completed previous segment or its latest epoch
        latest_epoch_path = f'checkpoints/research_A_segment_{start_segment-1}.pth'
        if not os.path.exists(latest_epoch_path):
            latest_epoch_path = f'checkpoints/research_A_segment_{start_segment-1}_latest_epoch.pth'
    else:
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
        print(f"[INFO] No existing checkpoint found at {latest_epoch_path}. Starting with fresh random initialization!")
    
    # Walk forward across all available segments starting from requested segment
    for idx in range(start_segment - 1, len(chunks) - 1):
        train_segment = chunks[idx]
        eval_segment = chunks[idx + 1]
        
        print(f"\n==============================================")
        print(f" WALK-FORWARD STEP {idx+1} (1-Week IS / 1-Week OOS)")
        print(f" TRAIN ON: {train_segment[0]} -> {train_segment[-1]}")
        print(f" EVALUATE ON: {eval_segment[0]} -> {eval_segment[-1]}")
        print(f"==============================================")
        
        config = load_config()
        lr = 0.00015 # Optuna Trial 520 Winner
        optimizer = optim.Adam(master_net.parameters(), lr=lr)
        
        MAX_EPOCHS_PER_SEGMENT = 15
        K_STREAK_REQUIRED = 3
        
        diagnostics_suite = OOSDiagnosticsSuite()
        
        epoch = 1
        pass_streak = 0
        
        while epoch <= MAX_EPOCHS_PER_SEGMENT:
            config = load_config()
            current_target_vram = config.get("target_vram", target_vram)
            
            print(f"[TRAIN] Running Epoch {epoch}/{MAX_EPOCHS_PER_SEGMENT} with LR={lr}...")
            fps_train = MultiDayForwardPassSystem(
                atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=train_segment
            )
            
            dynamic_is_agents = get_dynamic_n_agents(target_total_mb=current_target_vram, max_agents=128)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Active Load Check -> Spawning {dynamic_is_agents} parallel RL agents to keep VRAM < {current_target_vram/1000:.1f}GB...")
            is_metrics_dict = run_quadrant_sim(fps_train, master_net, optimizer, vtrace, config, device, epoch_idx=epoch-1, is_eval=False, N_AGENTS=dynamic_is_agents, accum_steps=accum_steps)
            
            # IS Mastery Gate Evaluation
            gate_passed, eff_n, raw_n, failed_conds = evaluate_is_mastery_gate(is_metrics_dict)
            
            print(f"\n[IS GATE] Epoch {epoch} | Effective N: {eff_n} (Raw: {raw_n})")
            print(f"[EPOCH DIAGNOSTIC] Days Processed in this Epoch: {train_segment}")
            if gate_passed:
                pass_streak += 1
                print(f"[IS GATE] -> PASS! (Streak: {pass_streak}/{K_STREAK_REQUIRED})")
            else:
                pass_streak = 0
                print(f"[IS GATE] -> FAIL: {', '.join(failed_conds)} (Streak Reset)")
            
            # Save a temporary checkpoint for the latest epoch regardless of passing status
            torch.save({
                'segment_idx': idx + 1,
                'epoch_idx': epoch,
                'lstm': master_net.lstm.state_dict(),
                'heads': master_net.state_dict()
            }, f'checkpoints/research_A_segment_{idx+1}_latest_epoch.pth')
            
            if pass_streak >= K_STREAK_REQUIRED:
                print(f"[INFO] [MASTERED] Curriculum Gate Passed! Advancing to OOS.")
                break
                
            epoch += 1
            
        if pass_streak < K_STREAK_REQUIRED:
            print(f"[INFO] [UNMASTERED] Hard cap reached ({MAX_EPOCHS_PER_SEGMENT} epochs). Failed conditions: {', '.join(failed_conds)}. Advancing anyway.")

        torch.save({
            'segment_idx': idx + 1,
            'lstm': master_net.lstm.state_dict(),
            'heads': master_net.state_dict()
        }, f'checkpoints/research_A_segment_{idx+1}.pth')

        # Out-of-Sample Eval
        fps_oos = MultiDayForwardPassSystem(atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=eval_segment)
        dynamic_oos_agents = get_dynamic_n_agents(fps_oos)
        
        oos_dict = run_quadrant_sim(fps_oos, master_net, None, vtrace, config, device, epoch_idx=epoch, is_eval=True, N_AGENTS=dynamic_oos_agents, accum_steps=accum_steps)
        
        trade_pnls = oos_dict['pnls']
        trade_durations = oos_dict['durations']
        trade_mfe_avail = oos_dict['mfe_avail']
        trade_mfe_trade = oos_dict['mfe_trade']
        trade_mae = oos_dict['mae']
        trade_metadata = oos_dict['metadata']
        
        seg_diag = diagnostics_suite.add_segment_data(
            trade_pnls, trade_durations, trade_mfe_avail, trade_mfe_trade, trade_mae,
            metadata=trade_metadata, segment_id=idx+1,
            train_dates=f"{train_segment[0]} -> {train_segment[-1]}",
            eval_dates=f"{eval_segment[0]} -> {eval_segment[-1]}"
        )
        print(f"\n[OOS DIAGNOSTICS] Trades: {seg_diag.get('trade_count', 0)} | MaxDD: {seg_diag.get('max_drawdown', 0):.2f} | PnL Mode CI: {seg_diag.get('pnl_mode_ci', (0,0))}")
        print(f"[OOS DIAGNOSTICS] Cap vs Avail: {seg_diag.get('cap_vs_avail', 0):.2%} | Cap vs Trade: {seg_diag.get('cap_vs_trade', 0):.2%} | Avg MAE: {seg_diag.get('avg_mae', 0):.2f}")
        
        # Dump raw OOS trade data for monitor_training.py to plot
        import json as _json
        _oos_dump = {
            'segment': idx + 1,
            'train_dates': f"{train_segment[0]} -> {train_segment[-1]}",
            'eval_dates': f"{eval_segment[0]} -> {eval_segment[-1]}",
            'pnls': [float(x) for x in trade_pnls],
            'durations': [float(x) for x in trade_durations],
            'metadata': trade_metadata
        }
        with open('oos_trade_data.json', 'w') as _f:
            _json.dump(_oos_dump, _f)
        
        pooled_diag = diagnostics_suite.get_pooled_diagnostics()
        print(f"\n[POOLED AGGREGATE]")
        for k, v in pooled_diag.items():
            print(f"  {k}: {v}")
            
        # Trigger Telegram Auto-Report
        try:
            import sys
            import subprocess
            from pathlib import Path
            repo_root = Path(__file__).resolve().parent.parent.parent
            autostats_path = repo_root / "autostats.py"
            autoplot_path = repo_root / "autoplot.py"
            
            sys.stdout.flush()
            # Launch both commands asynchronously passing the current segment index
            subprocess.Popen([sys.executable, str(autostats_path), str(idx + 1)], cwd=str(repo_root))
            subprocess.Popen([sys.executable, str(autoplot_path), str(idx + 1)], cwd=str(repo_root))
            print(f"[INFO] Dispatched Telegram update in background (autostats and autoplot) for Segment {idx + 1}.")
        except Exception as e:
            print(f"[ERROR] Failed to trigger Telegram update: {e}")
            
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
    import argparse
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Research A Walk-Forward Trainer")
    parser.add_argument("--start-segment", type=int, default=1, help="Segment to start/resume from")
    parser.add_argument("--target-vram", type=int, default=10500, help="Target total GPU memory usage in MB for dynamic agent scaling")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps to simulate larger batch sizes and save VRAM")
    args = parser.parse_args()
    
    try:
        run_walk_forward_curriculum(start_segment=args.start_segment, target_vram=args.target_vram, accum_steps=args.accum_steps)
    finally:
        allow_sleep()
    
    print("[INFO] Curriculum Complete. Releasing GPU resources and exiting.")
    torch.cuda.empty_cache()
    sys.exit(0)
