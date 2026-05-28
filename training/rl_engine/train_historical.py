import argparse
import json
import torch
import torch.optim as optim
import h5py
import numpy as np
import os
import glob
from collections import deque
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc
from datetime import datetime, timezone
from network import MasterNetwork
from hdf5_shadow_queue import HDF5ShadowQueue
from vtrace_reconciliation import VTraceReconciliation
from parallel_worlds import ParallelWorlds

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from core_v2.ledger import Ledger
from core_v2.exits import default_exit_suite

# ------------------------------------------------------------------
# HDF5 Disk-Streaming PyTorch Dataset
# Bypasses 12GB VRAM by lazily loading batches directly from the M.2 SSD
# ------------------------------------------------------------------
class OutOfCoreDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.length = 0
        if os.path.exists(h5_path):
            with h5py.File(self.h5_path, 'r') as f:
                if 'v2_grids' in f:
                    self.length = f['v2_grids'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            v2_grid = f['v2_grids'][idx]
            l0_feature = f['l0_features'][idx]
            action = f['actions'][idx]
            regret = f['regrets'][idx]
            pi = f['behavior_pi'][idx]
        
        return (
            torch.tensor(v2_grid, dtype=torch.float32),
            torch.tensor(l0_feature, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(regret, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32)
        )

# ------------------------------------------------------------------
# OOS Evaluation Metrics Breakdown
# ------------------------------------------------------------------
def print_oos_trade_breakdown(records):
    if not records:
        print("       [OOS] Trade Breakdown: No trades to analyze.")
        return
        
    durations = [r['duration'] for r in records]
    
    # 5 buckets
    import numpy as np
    counts, bin_edges = np.histogram(durations, bins=5)
    
    print(f"\n       [OOS] Trade Duration Breakdown (5 Buckets):")
    buckets_data = []
    
    for i in range(5):
        b_min = bin_edges[i]
        b_max = bin_edges[i+1]
        
        if i == 4:
            b_trades = [r for r in records if b_min <= r['duration'] <= b_max]
        else:
            b_trades = [r for r in records if b_min <= r['duration'] < b_max]
            
        b_count = len(b_trades)
        b_pnls = [r['pnl'] for r in b_trades]
        b_pnl_sum = sum(b_pnls)
        b_pnl_avg = b_pnl_sum / b_count if b_count > 0 else 0.0
        
        buckets_data.append({
            'min': b_min, 'max': b_max, 'count': b_count, 'trades': b_trades, 'avg_pnl': b_pnl_avg
        })
        
        if b_count > 0:
            print(f"         [{b_min:6.1f}s - {b_max:6.1f}s] : {b_count:5d} trades | Avg PnL: ${b_pnl_avg:6.2f}")
        else:
            print(f"         [{b_min:6.1f}s - {b_max:6.1f}s] :     0 trades")
            
    # Find mode bucket
    mode_idx = np.argmax(counts)
    mode_bucket = buckets_data[mode_idx]
    
    if mode_bucket['count'] > 1: # Need > 1 for standard deviation
        mode_trades = mode_bucket['trades']
        m_durations = [r['duration'] for r in mode_trades]
        m_pnls = [r['pnl'] for r in mode_trades]
        
        d_mean = np.mean(m_durations)
        d_se = np.std(m_durations, ddof=1) / np.sqrt(len(m_durations))
        d_ci_lower = d_mean - 1.96 * d_se
        d_ci_upper = d_mean + 1.96 * d_se
        
        p_mean = np.mean(m_pnls)
        p_se = np.std(m_pnls, ddof=1) / np.sqrt(len(m_pnls))
        p_ci_lower = p_mean - 1.96 * p_se
        p_ci_upper = p_mean + 1.96 * p_se
        
        print(f"\n       [OOS] Mode Bucket Confidence Intervals (95%):")
        print(f"         Mode Range: {mode_bucket['min']:.1f}s to {mode_bucket['max']:.1f}s ({mode_bucket['count']} trades)")
        print(f"         Duration  : Mean = {d_mean:.1f}s | 95% CI [{d_ci_lower:.1f}s, {d_ci_upper:.1f}s]")
        print(f"         PnL       : Mean = ${p_mean:.2f} | 95% CI [${p_ci_lower:.2f}, ${p_ci_upper:.2f}]")
    else:
        print(f"\n       [OOS] Mode Bucket Confidence Intervals: Not enough trades in mode to calculate 95% CI.")

# ------------------------------------------------------------------
# Training Harness
# ------------------------------------------------------------------
import gc

def run_training(is_smoke_test=False, agent_type="ENTRY_NMP", learning_rate=5e-3, run_id="default", max_epochs=99):
    print(f"[INFO] Booting PW-CRL Training Engine (V2 NATIVE)...")
    print(f"[INFO] Agent Type: {agent_type} | LR: {learning_rate} | Max Epochs: {max_epochs}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Compute Device: {device}")

    # 1. Initialize Network & V-Trace
    master_net = MasterNetwork().to(device)
    
    ckpt_path = f"C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/master_net_{run_id}.pth"
    if os.path.exists(ckpt_path):
        print(f"[INFO] Loading baseline PyTorch weights from {ckpt_path}")
        master_net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
    optimizer = optim.Adam(master_net.parameters(), lr=learning_rate)
    vtrace = VTraceReconciliation()
    
    # 2. Shadow Queue Storage
    import uuid
    db_path = f"C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/experiences_{run_id}.h5"
    if is_smoke_test and os.path.exists(db_path):
        os.remove(db_path) # Clear DB for clean smoke test
    
    queue = HDF5ShadowQueue(db_path)

    # 3. Path setup for V2 Features
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    l0_dir = os.path.join(features_root, 'L0')
    if not os.path.exists(l0_dir):
        print(f"[ERROR] Features directory not found: {l0_dir}")
        return
        
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    all_days = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    if not all_days:
        print(f"[ERROR] No parquet days found in {l0_dir}")
        return

    # Group days into "weeks" (5 trading days per segment)
    chunk_size = 5
    weeks = [all_days[i:i + chunk_size] for i in range(0, len(all_days), chunk_size)]
    if is_smoke_test:
        weeks = weeks[:3]
        
    print(f"[INFO] Discovered {len(all_days)} days, grouped into {len(weeks)} segments.")

    target_n = 0.0
    previous_n = 0.0
    
    # Preload the default structural exit suite for the environment
    exit_suite = default_exit_suite()

    # Iterate over segments. Reserve the next week for OOS validation.
    for segment in range(len(weeks) - 1):
        current_week = weeks[segment]
        next_week = weeks[segment + 1]
        
        print(f"\n========================================================")
        print(f"       STARTING CURRICULUM SEGMENT {segment+1} (Target: {target_n:.4f})")
        print(f"       IS Days: {current_week}")
        print(f"========================================================")
        
        epochs = 0
        segment_passed = False
        oos_history = []
        
        while epochs < max_epochs and not segment_passed:
            epochs += 1
            print(f"\n--- Epoch {epochs}/{max_epochs} ---")
            
            # --- DYNAMIC LEARNING RATE INJECTION ---
            try:
                config_path = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/hyperparams.json"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as cf:
                        dyn_config = json.load(cf)
                        if "lr" in dyn_config and dyn_config["lr"] != optimizer.param_groups[0]['lr']:
                            old_lr = optimizer.param_groups[0]['lr']
                            new_lr = dyn_config["lr"]
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            print(f"[HOTFIX] Dynamic Learning Rate updated: {old_lr} -> {new_lr}")
            except Exception as e:
                print(f"[WARN] Failed to load dynamic hyperparams.json: {e}")
            
            gross_profit = 0.0
            gross_loss = 0.0
            num_trades = 0

            # --- PHASE 1: GENERATE EXPERIENCES (V2 Native Forward Pass) ---
            print(f"[DATA] Simulating IS Trajectories through V2 FPS...")
            master_net.eval()
            
            fps = MultiDayForwardPassSystem(
                atlas_root=atlas_root,
                features_root=features_root,
                labels_csv=labels_csv,
                days=current_week
            )
            
            ledger = Ledger()
            state_queue = deque(maxlen=60)
            
            # Temporary storage for the state at trade entry
            active_trade_context = None

            for bar_idx, bar_state in enumerate(fps):
                if bar_idx > 0 and bar_idx % 2000 == 0:
                    print(f"       [IS] Processed {bar_idx} bars... (Current Trades: {num_trades})")
                
                # 1. Digest the Feature Tree (185 features)
                v2_vec = bar_state.v2_vector
                
                # Protect against malformed schema
                if len(v2_vec) < 185:
                    continue
                    
                dt = datetime.fromtimestamp(bar_state.timestamp, tz=timezone.utc)
                day_norm = dt.weekday() / 4.0 # Monday=0.0, Friday=1.0
                sec_in_day = int(bar_state.timestamp) % 86400
                tod_norm = sec_in_day / 86400.0
                
                l0 = [v2_vec[0], tod_norm, day_norm] # [3]
                grid = v2_vec[1:185].reshape(8, 23) # [8, 23]
                
                state_queue.append((l0, grid))
                
                if len(state_queue) < 60:
                    continue # Waiting for full sequence
                
                # 2. Agent Decision & Environment Mechanics
                if ledger.is_flat:
                    # Agent observes the full 60-sequence
                    l0_tensor = torch.tensor(np.array([s[0] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device) # [1, 60, 1]
                    grid_tensor = torch.tensor(np.array([s[1] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device) # [1, 60, 8, 23]
                    grid_tensor = grid_tensor.permute(0, 2, 1, 3) # [1, 8, 60, 23]
                    
                    # Fill NaNs from TF warmup phase
                    l0_tensor = torch.nan_to_num(l0_tensor, nan=0.0)
                    grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0)
                    
                    sec_in_day = int(bar_state.timestamp) % 86400
                    is_eod_maintenance = (75300 <= sec_in_day < 79200)
                    
                    if is_eod_maintenance:
                        continue # Block entries during maintenance window
                    
                    if agent_type == 'EXIT_NMP':
                        # EXIT_NMP trains purely on exits. Random structural entry logic (2% chance to jump in).
                        if np.random.rand() < 0.02:
                            action = np.random.choice([1, 2])
                            pi = 0.5
                        else:
                            action = 0
                            pi = 1.0
                    else:
                        # NMP, ENTRY_NMP, YOLO use the network for entry
                        with torch.no_grad():
                            q_values, _ = master_net(grid_tensor, l0_tensor)
                            
                            temperature = max(0.1, 1.0 - (epochs / 50.0))
                            probs = F.softmax(q_values / temperature, dim=1)
                            
                            epsilon = max(0.01, 0.5 - (epochs / 30.0))
                            if np.random.rand() < epsilon:
                                action = np.random.randint(0, 3)
                                pi = 0.33
                            else:
                                action = torch.multinomial(probs, 1).item()
                                pi = probs[0, action].item()
                    
                    if action in [1, 2]: # 1: Long, 2: Short
                        direction = 'long' if action == 1 else 'short'
                        
                        active_trade_context = {
                            'action': action,
                            'l0_feature': l0_tensor.cpu().numpy()[0], # [60, 1]
                            'v2_grid': grid_tensor.cpu().numpy()[0], # [8, 60, 23]
                            'pi': pi
                        }
                        
                        ledger.add_position(
                            direction=direction,
                            entry_price=bar_state.price,
                            entry_ts=bar_state.timestamp,
                            entry_tier='RL_AGENT',
                            entry_features=bar_state.v2_vector
                        )
                else:
                    # Position is open. Feed the tick to the physics engine.
                    ledger.update_bar(bar_state.v2_vector, bar_state.price, bar_state.timestamp)
                    pos = ledger.primary
                    
                    exit_reason = None
                    
                    sec_in_day = int(bar_state.timestamp) % 86400
                    is_eod_maintenance = (75300 <= sec_in_day < 79200)
                    
                    if is_eod_maintenance:
                        exit_reason = 'EOD_MAINTENANCE'
                    
                    # 1. Evaluate Neural Exit if permitted by Agent Architecture
                    if not exit_reason and agent_type in ['NMP', 'YOLO', 'EXIT_NMP']:
                        l0_tensor = torch.tensor(np.array([s[0] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
                        grid_tensor = torch.tensor(np.array([s[1] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
                        grid_tensor = grid_tensor.permute(0, 2, 1, 3)
                        l0_tensor = torch.nan_to_num(l0_tensor, nan=0.0)
                        grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0)
                        
                        with torch.no_grad():
                            q_values, _ = master_net(grid_tensor, l0_tensor)
                            nn_action = torch.argmax(q_values, dim=1).item()
                            
                        if nn_action == 0:
                            exit_reason = 'NN_Exit_Flat'
                        elif pos.direction == 'long' and nn_action == 2:
                            exit_reason = 'NN_Exit_Reverse'
                        elif pos.direction == 'short' and nn_action == 1:
                            exit_reason = 'NN_Exit_Reverse'

                    # 2. Evaluate Structural Exits (Guardrails) if permitted
                    if not exit_reason and agent_type != 'YOLO':
                        for rule in exit_suite:
                            exit_reason = rule.evaluate(bar_state, pos)
                            if exit_reason:
                                break
                            
                    if exit_reason:
                        # Structural Exit Triggered! Capture right-tail PnL.
                        record = ledger.remove_position(pos.contract_id, bar_state.price, bar_state.timestamp, exit_reason)
                        trade_pnl = record['pnl']
                        
                        num_trades += 1
                        if trade_pnl > 0:
                            gross_profit += trade_pnl
                        else:
                            gross_loss += abs(trade_pnl)
                            
                        # RL Reward Shaping
                        # Asymmetric penalization to protect limited equity (Applied to ALL agents, even YOLO)
                        if exit_reason == 'EOD_MAINTENANCE' and trade_pnl < 0:
                            # FATAL PENALTY: Holding a loss into the daily maintenance window
                            regret = float(trade_pnl * 5.0) - 500.0
                        elif trade_pnl < 0:
                            regret = float(trade_pnl * 2.0) - 100.0
                        elif trade_pnl < 20.0:
                            regret = float(trade_pnl) - 50.0
                        else:
                            regret = float(trade_pnl)
                        
                        # Persist to disk buffer
                        if active_trade_context is not None:
                            queue.write_terminal_trajectory(
                                v2_grid=active_trade_context['v2_grid'],
                                l0_feature=active_trade_context['l0_feature'],
                                action=active_trade_context['action'],
                                regret=regret,
                                behavior_pi=active_trade_context['pi']
                            )
                        
                        active_trade_context = None

            print(f"[DATA] HDF5 Database populated. Total experiences: {queue.get_dataset_size()}")

            # --- PHASE 2: V-TRACE GRADIENT OPTIMIZATION (Disk to GPU) ---
            print(f"\n[TRAIN] Executing V-trace Mini-batch optimization...")
            if queue.get_dataset_size() > 0:
                master_net.train()
                dataset = OutOfCoreDataset(db_path)
                dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0) 

                total_loss = 0.0
                for batch_idx, (grids, l0s, actions, regrets, behaviors) in enumerate(dataloader):
                    grids = torch.nan_to_num(grids.to(device), nan=0.0)
                    l0s = torch.nan_to_num(l0s.to(device), nan=0.0)
                    actions = actions.to(device)
                    regrets = regrets.to(device)
                    behaviors = behaviors.to(device)

                    optimizer.zero_grad()

                    q_values, _ = master_net(grids, l0s)
                    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    td_loss = F.smooth_l1_loss(q_taken, regrets, reduction='none')
                    target_pi = F.softmax(q_values, dim=1).gather(1, actions.unsqueeze(1)).squeeze(1)

                    corrected_loss = vtrace.apply_gradient_correction(td_loss, target_pi, behaviors)
                    
                    corrected_loss.backward()
                    
                    # Prevent network shattering from massive raw PnL regret dollars
                    clip_threshold = 10.0 if is_smoke_test else 1.0
                    torch.nn.utils.clip_grad_norm_(master_net.parameters(), max_norm=clip_threshold)
                    
                    optimizer.step()
                    
                    total_loss += corrected_loss.item()
                    
                    if batch_idx % 2 == 0:
                        print(f"       Batch {batch_idx} | V-Trace Loss: {corrected_loss.item():.4f}")
            else:
                print(f"       [WARN] No trades were taken during this epoch. Skipping optimization.")

            # --- PHASE 3: CURRICULUM KPI EVALUATION ---
            print(f"\n[KPI] Evaluating Curriculum Segment {segment+1}...")
            print(f"       Total Trades : {num_trades}")
            print(f"       Gross Profit : ${gross_profit:.2f}")
            print(f"       Gross Loss   : ${gross_loss:.2f}")
            
            if num_trades >= 10: # Relaxed constraint for real physics testing
                current_n = 999.0 if gross_loss == 0.0 else (gross_profit / gross_loss) - 1.0
                print(f"       Metric (n)   : {current_n:.4f} (IS Target is ignored for graduation)")
                
                # --- PHASE 3.5: WALK-FORWARD OUT-OF-SAMPLE (OOS) VALIDATION ---
                print(f"\n[VALIDATION] Executing Out-Of-Sample Walk-Forward on {next_week}...")
                master_net.eval()
                
                oos_profit = 0.0
                oos_loss = 0.0
                oos_trades = 0
                oos_records = []
                
                fps_oos = MultiDayForwardPassSystem(
                    atlas_root=atlas_root,
                    features_root=features_root,
                    labels_csv=labels_csv,
                    days=next_week
                )
                
                ledger_oos = Ledger()
                state_queue_oos = deque(maxlen=60)
                
                with torch.no_grad():
                    for bar_idx, bar_state in enumerate(fps_oos):
                        if bar_idx > 0 and bar_idx % 2000 == 0:
                            print(f"       [OOS] Processed {bar_idx} bars... (Current OOS Trades: {oos_trades})")
                            
                        v2_vec = bar_state.v2_vector
                        if len(v2_vec) < 185: continue
                            
                        dt = datetime.fromtimestamp(bar_state.timestamp, tz=timezone.utc)
                        day_norm = dt.weekday() / 4.0
                        sec_in_day = int(bar_state.timestamp) % 86400
                        tod_norm = sec_in_day / 86400.0
                        
                        l0 = [v2_vec[0], tod_norm, day_norm]
                        grid = v2_vec[1:185].reshape(8, 23)
                        state_queue_oos.append((l0, grid))
                        
                        if len(state_queue_oos) < 60: continue
                        
                        if ledger_oos.is_flat:
                            l0_t = torch.tensor(np.array([s[0] for s in state_queue_oos]), dtype=torch.float32).unsqueeze(0).to(device)
                            grid_t = torch.tensor(np.array([s[1] for s in state_queue_oos]), dtype=torch.float32).unsqueeze(0).to(device)
                            grid_t = grid_t.permute(0, 2, 1, 3) 
                            
                            l0_t = torch.nan_to_num(l0_t, nan=0.0)
                            grid_t = torch.nan_to_num(grid_t, nan=0.0)
                            
                            sec_in_day = int(bar_state.timestamp) % 86400
                            is_eod_maintenance = (75300 <= sec_in_day < 79200)
                            
                            if is_eod_maintenance:
                                continue # Block entries during maintenance window
                                
                            if agent_type == 'EXIT_NMP':
                                # Validation: purely random entry
                                if np.random.rand() < 0.02:
                                    action = np.random.choice([1, 2])
                                else:
                                    action = 0
                            else:
                                q_vals, _ = master_net(grid_t, l0_t)
                                action = torch.argmax(q_vals, dim=1).item()
                            
                            if action in [1, 2]:
                                ledger_oos.add_position(
                                    direction='long' if action == 1 else 'short',
                                    entry_price=bar_state.price,
                                    entry_ts=bar_state.timestamp,
                                    entry_tier='RL_AGENT',
                                    entry_features=bar_state.v2_vector
                                )
                        else:
                            ledger_oos.update_bar(bar_state.v2_vector, bar_state.price, bar_state.timestamp)
                            pos = ledger_oos.primary
                            
                            exit_reason = None
                            
                            sec_in_day = int(bar_state.timestamp) % 86400
                            is_eod_maintenance = (75300 <= sec_in_day < 79200)
                            
                            if is_eod_maintenance:
                                exit_reason = 'EOD_MAINTENANCE'
                            
                            if not exit_reason and agent_type in ['NMP', 'YOLO', 'EXIT_NMP']:
                                l0_t = torch.tensor(np.array([s[0] for s in state_queue_oos]), dtype=torch.float32).unsqueeze(0).to(device)
                                grid_t = torch.tensor(np.array([s[1] for s in state_queue_oos]), dtype=torch.float32).unsqueeze(0).to(device)
                                grid_t = grid_t.permute(0, 2, 1, 3)
                                l0_t = torch.nan_to_num(l0_t, nan=0.0)
                                grid_t = torch.nan_to_num(grid_t, nan=0.0)
                                
                                q_vals, _ = master_net(grid_t, l0_t)
                                nn_action = torch.argmax(q_vals, dim=1).item()
                                
                                if nn_action == 0:
                                    exit_reason = 'NN_Exit_Flat'
                                elif pos.direction == 'long' and nn_action == 2:
                                    exit_reason = 'NN_Exit_Reverse'
                                elif pos.direction == 'short' and nn_action == 1:
                                    exit_reason = 'NN_Exit_Reverse'

                            if not exit_reason and agent_type != 'YOLO':
                                for rule in exit_suite:
                                    exit_reason = rule.evaluate(bar_state, pos)
                                    if exit_reason: break
                                    
                            if exit_reason:
                                record = ledger_oos.remove_position(pos.contract_id, bar_state.price, bar_state.timestamp, exit_reason)
                                trade_pnl = record['pnl']
                                duration = record['exit_ts'] - record['entry_ts']
                                oos_records.append({'pnl': trade_pnl, 'duration': duration})
                                oos_trades += 1
                                if trade_pnl > 0:
                                    oos_profit += trade_pnl
                                else:
                                    oos_loss += abs(trade_pnl)
                
                oos_n = 999.0 if oos_loss == 0.0 else (oos_profit / oos_loss) - 1.0
                oos_net_pnl = oos_profit - oos_loss
                oos_history.append(oos_n)
                
                print(f"       OOS Trades : {oos_trades}")
                print(f"       OOS Net PnL: ${oos_net_pnl:.2f}")
                print(f"       OOS Metric (n) : {oos_n:.4f}")
                
                print_oos_trade_breakdown(oos_records)
                
                # Dynamic Pity Pass Logic
                if epochs >= 30 and oos_n < target_n:
                    avg_oos = sum(oos_history[-30:]) / 30.0
                    dynamic_target = max(0.0, avg_oos + 0.10) # ABSOLUTE FLOOR: Never pass a net-negative model
                    print(f"       [PITY PASS] Epoch 30+ reached. Dynamic Target = {dynamic_target:.4f} (Avg: {avg_oos:.4f} + 10%)")
                    
                    if oos_n >= dynamic_target:
                        if oos_net_pnl >= 200.0 * len(next_week):
                            print(f"[SUCCESS] Walk-Forward Validation Passed via Dynamic Target!")
                            target_n = dynamic_target + 0.10
                            print(f"[CURRICULUM] Advancing to target_n = {target_n:.4f}")
                            segment_passed = True
                        else:
                            print(f"[FAIL] OOS Capital Velocity Failed! ${oos_net_pnl:.2f} < ${200.0 * len(next_week):.2f}. Retraining...")
                    else:
                        print(f"[FAIL] OOS n ({oos_n:.4f}) still below Dynamic Target ({dynamic_target:.4f}). Retraining...")
                else:
                    if oos_n >= target_n:
                        if oos_net_pnl >= 200.0 * len(next_week):
                            print(f"[SUCCESS] Walk-Forward Validation Passed! Model has generalized.")
                            target_n = oos_n + 0.10
                            print(f"[CURRICULUM] Advancing to target_n = {target_n:.4f}")
                            segment_passed = True
                        else:
                            print(f"[FAIL] OOS Capital Velocity Failed! ${oos_net_pnl:.2f} < ${200.0 * len(next_week):.2f}. Retraining...")
                    else:
                        print(f"[FAIL] OVERFIT DETECTED! OOS n ({oos_n:.4f}) < Target ({target_n:.4f}). Retraining required...")
                        
            else:
                is_net_pnl = gross_profit - gross_loss
                if is_net_pnl < 200.0 * len(this_week):
                    print(f"[FAIL] IS Capital Velocity Failed! ${is_net_pnl:.2f} < ${200.0 * len(this_week):.2f}. Retraining...")
                else:
                    print(f"[FAIL] Volumetric constraint violated ({num_trades} < 10). Retraining...")
                
        if segment_passed and is_smoke_test:
            # FAIL-FAST: Step down gradient upon finding a structurally sound path
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"[OPTIMIZER] LR Stepped Down. New LR: {optimizer.param_groups[0]['lr']:.6f}")
                
        if not segment_passed:
            print(f"\n[FATAL] Segment {segment+1} failed after {max_epochs} epochs. Emergency Exit triggered.")
            break

        # Memory Cleanup after each Epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Continuous state-saving: Checkpoint model weights after every epoch
        checkpoint_path = f"master_net_{run_id}_{agent_type}_epoch_{epochs}.pth"
        torch.save(master_net.state_dict(), checkpoint_path)
        print(f"[CHECKPOINT] Model weights saved to {checkpoint_path}")

    # --- PHASE 4: ONNX EXPORT ---
    print(f"\n[DEPLOY] Exporting trained Master Network to ONNX for NT8 C# Native Interop...")
    master_net.eval()
    
    # Save PyTorch baseline weights for real test transfer learning
    torch.save(master_net.state_dict(), ckpt_path)
    print(f"[SUCCESS] Baseline PyTorch weights saved to: {ckpt_path}")
    
    # Create dummy inputs for ONNX tracing
    dummy_grid = torch.randn(1, 8, 60, 23, device=device)
    dummy_l0 = torch.randn(1, 60, 3, device=device)
    onnx_path = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/master_net.onnx"
    
    torch.onnx.export(
        master_net, 
        (dummy_grid, dummy_l0), 
        onnx_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['v2_grid', 'l0_feature'],
        output_names=['q_values', 'hidden_state_out'],
        dynamic_axes={'v2_grid': {0: 'batch_size'}, 'l0_feature': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}}
    )
    
    print(f"[SUCCESS] ONNX Model saved successfully to: {onnx_path}")
    print(f"[SUCCESS] Python Execution Pipeline Verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PW-CRL Historical Training Engine")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short smoke test simulation")
    parser.add_argument("--agent-type", type=str, default="ENTRY_NMP", choices=["NMP", "ENTRY_NMP", "EXIT_NMP", "YOLO"])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--run-id", type=str, default="default_run")
    parser.add_argument("--max-epochs", type=int, default=99)
    args = parser.parse_args()
    
    run_training(
        is_smoke_test=args.smoke_test,
        agent_type=args.agent_type,
        learning_rate=args.lr,
        run_id=args.run_id,
        max_epochs=args.max_epochs
    )
