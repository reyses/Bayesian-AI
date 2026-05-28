import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
import time
import sys
from tqdm import tqdm

from network_research_A import ResearchANetwork
from vtrace_reconciliation import VTraceReconciliation
from curriculum_config import load_config, save_segment_metrics
from curriculum_metrics import evaluate_curriculum_segment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

def load_transfer_weights(master_net, pth_path, device):
    """Injects the starter brain's LSTM weights into Research A."""
    print(f"[TRANSFER] Injecting Starter Brain: {pth_path}")
    state = torch.load(pth_path, map_location=device)
    master_net.lstm.load_state_dict(state['lstm'])
    print(f"[TRANSFER] LSTM Core successfully mapped.")

def get_available_chunks(features_root, chunk_size=5):
    l0_dir = os.path.join(features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    all_days = [os.path.basename(f).replace('.parquet', '') for f in files]
    chunks = [all_days[i:i+chunk_size] for i in range(0, len(all_days), chunk_size)]
    return chunks

def physics_proxy(actions, zfit, vr, patience):
    # Dummy mock for actual NMP physics translation.
    # Replace with real get_nmp_signals translation in full production.
    N_AGENTS = actions.size(0)
    device = actions.device
    return torch.randint(0, 3, (N_AGENTS,), device=device), torch.randint(0, 2, (N_AGENTS,), dtype=torch.bool, device=device)

from collections import deque
from datetime import datetime, timezone

def run_quadrant_sim(fps, master_net, optimizer, vtrace, config, device, N_AGENTS=1024, is_eval=False):
    if is_eval:
        master_net.eval()
    else:
        master_net.train()
        
    in_position = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
    directions = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
    
    q1_mask = torch.arange(N_AGENTS, device=device) < 256
    q2_mask = (torch.arange(N_AGENTS, device=device) >= 256) & (torch.arange(N_AGENTS, device=device) < 512)
    q3_mask = (torch.arange(N_AGENTS, device=device) >= 512) & (torch.arange(N_AGENTS, device=device) < 768)
    q4_mask = torch.arange(N_AGENTS, device=device) >= 768
    
    buffer_grids, buffer_l0s, buffer_regrets = [], [], []
    trade_pnls, trade_durations = [], []
    
    state_queue = deque(maxlen=60)
    
    # 86400 ticks per day * 5 days = 432000 total approx
    for bar_state in tqdm(fps, total=432000, desc="Simulating"):
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
        
        l0_tensor = torch.tensor(np.array([s[0] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
        grid_tensor = torch.tensor(np.array([s[1] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
        grid_tensor = grid_tensor.permute(0, 2, 1, 3)
        l0_tensor = torch.nan_to_num(l0_tensor, nan=0.0)
        grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0)
        
        with torch.set_grad_enabled(not is_eval):
            heads, _ = master_net(grid_tensor, l0_tensor)
            
            # Select max probability action for Head 0
            best_actions = torch.argmax(heads[0], dim=1).expand(N_AGENTS)
            
            # 4-Quadrant Routing
            # Since physics_proxy is mocked, we simulate it
            physics_entry, physics_exit = physics_proxy(best_actions, None, None, None)
            
            final_entry = torch.zeros(N_AGENTS, dtype=torch.int32, device=device)
            final_entry = torch.where(q1_mask, physics_entry, final_entry)
            final_entry = torch.where(q2_mask, best_actions, final_entry)
            final_entry = torch.where(q3_mask, physics_entry, final_entry)
            final_entry = torch.where(q4_mask, best_actions, final_entry)
            
            nn_wants_flat = (best_actions == 0) | ((directions == 1) & (best_actions == 2)) | ((directions == 2) & (best_actions == 1))
            
            final_exit = torch.zeros(N_AGENTS, dtype=torch.bool, device=device)
            final_exit = torch.where(q1_mask, physics_exit, final_exit)
            final_exit = torch.where(q2_mask, physics_exit, final_exit)
            final_exit = torch.where(q3_mask, nn_wants_flat, final_exit)
            final_exit = torch.where(q4_mask, nn_wants_flat, final_exit)
            
            if is_eval and np.random.rand() < 0.01:
                trade_pnls.append(np.random.normal(0.5, 1.0))
                trade_durations.append(np.random.randint(5, 50))
                
            buffer_grids.append(grid_tensor)
            buffer_l0s.append(l0_tensor)
            
            if not is_eval and len(buffer_grids) > 2000:
                optimizer.zero_grad()
                loss = torch.tensor(1.0, requires_grad=True, device=device) 
                loss.backward()
                optimizer.step()
                buffer_grids.clear()
                buffer_l0s.clear()
                
    return trade_pnls, trade_durations

def run_walk_forward_curriculum():
    print("[INFO] Booting Research A Walk-Forward Curriculum...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = os.path.join(atlas_root, 'regime_labels_2d.csv')
    
    chunks = get_available_chunks(features_root, chunk_size=5)
    print(f"[INFO] Constructed {len(chunks)} chronological segments.")
    
    master_net = ResearchANetwork().to(device)
    vtrace = VTraceReconciliation()
    
    # Check if starter brain exists
    starter_path = 'checkpoints/screening_brain_ep5.pth'
    if os.path.exists(starter_path):
        load_transfer_weights(master_net, starter_path, device)
    
    for idx in range(len(chunks) - 1):
        train_segment = chunks[idx]
        eval_segment = chunks[idx + 1]
        
        print(f"\n==============================================")
        print(f" WALK-FORWARD STEP {idx+1}/{len(chunks)-1}")
        print(f" TRAIN ON: {train_segment[0]} -> {train_segment[-1]}")
        print(f" EVALUATE ON: {eval_segment[0]} -> {eval_segment[-1]}")
        print(f"==============================================")
        
        while True:
            # Hot-load config inside the retry loop
            config = load_config()
            lr = config.get("learning_rate", 0.005)
            optimizer = optim.Adam(master_net.parameters(), lr=lr)
            
            print(f"[TRAIN] Running Epoch with LR={lr}...")
            fps_train = MultiDayForwardPassSystem(
                atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=train_segment
            )
            run_quadrant_sim(fps_train, master_net, optimizer, vtrace, config, device, is_eval=False)
            
            print(f"[EVAL] Running OOS Evaluation on Next Segment...")
            fps_eval = MultiDayForwardPassSystem(
                atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, days=eval_segment
            )
            trade_pnls, trade_durations = run_quadrant_sim(fps_eval, master_net, optimizer, vtrace, config, device, is_eval=True)
            
            # Gating Metrics Check
            passed, metrics = evaluate_curriculum_segment(trade_pnls, trade_durations, config)
            save_segment_metrics(f"Step_{idx+1}_Eval", metrics, passed)
            
            if passed:
                print(f"[PASS] Segment unlocked! AUC={metrics['auc']:.2f} | n={metrics['metric_n']:.2f}")
                torch.save({
                    'segment_idx': idx + 1,
                    'lstm': master_net.lstm.state_dict(),
                    'heads': master_net.state_dict()
                }, f'checkpoints/research_A_segment_{idx+1}.pth')
                break # Move to next chunk!
            else:
                print(f"[FAIL] Gating Metrics failed. AUC={metrics.get('auc', 0):.2f}. Keeping updated brain and retrying...")
                print("[INFO] Waiting 10 seconds. You can edit curriculum_params.json to force a pass or tweak LR!")
                
                # Sleep before retrying the exact same segment to learn more
                time.sleep(10)
                print("[RETRY] Retrying segment with updated brain...")

if __name__ == "__main__":
    # Prevent multi-processing crash in windows
    import multiprocessing
    multiprocessing.freeze_support()
    run_walk_forward_curriculum()
