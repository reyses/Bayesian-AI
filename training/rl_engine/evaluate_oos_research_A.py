import torch
import numpy as np
import os
import glob
from collections import deque
import torch.nn.functional as F

from network_research_A import ResearchANetwork

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from core_v2.ledger import Ledger
from core_v2.exits import default_exit_suite

def run_oos_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Booting OOS Diagnostic Engine (Research A)...")
    print(f"[INFO] Compute Device: {device}")

    # 1. Load Pre-trained Network
    master_net = ResearchANetwork().to(device)
    ckpt_path = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/checkpoints/research_A_segment_68.pth"
    
    if os.path.exists(ckpt_path):
        print(f"[INFO] Loading frozen baseline weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        if 'lstm' in state and 'heads' in state:
            master_net.lstm.load_state_dict(state['lstm'])
            master_net.load_state_dict(state['heads'], strict=False)
        else:
            master_net.load_state_dict(state, strict=False)
    else:
        print(f"[ERROR] Checkpoint not found at {ckpt_path}! Cannot run diagnostics.")
        return
        
    master_net.eval()

    # 2. Path setup for V2 Features
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS_NT8"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    labels_csv = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/regime_labels_2d.csv"
    
    l0_dir = os.path.join(features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    all_days = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    # 3. Extract purely OOS data
    # All days in ATLAS_NT8 are OOS.
    oos_days = all_days
    print(f"[INFO] Discovered {len(oos_days)} purely OOS days in ATLAS_NT8.")
    
    if len(oos_days) == 0:
        print("[ERROR] No OOS data available.")
        return

    exit_suite = default_exit_suite()
    
    # 4. Stream data through FPS
    print(f"\n[EVAL] Executing pure Forward Pass System over {len(oos_days)} days...")
    
    fps = MultiDayForwardPassSystem(
        atlas_root=atlas_root,
        features_root=features_root,
        labels_csv=labels_csv,
        days=oos_days
    )
    
    ledger = Ledger()
    state_queue = deque(maxlen=60)
    
    gross_profit = 0.0
    gross_loss = 0.0
    num_trades = 0
    
    with torch.no_grad():
        for bar_idx, bar_state in enumerate(fps):
            if bar_idx > 0 and bar_idx % 10000 == 0:
                print(f"       Processed {bar_idx} bars... (Current OOS Trades: {num_trades}, PnL: ${gross_profit - gross_loss:.2f})")
            
            v2_vec = bar_state.v2_vector
            if len(v2_vec) < 185:
                continue
                
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(bar_state.timestamp, tz=timezone.utc)
            day_norm = dt.weekday() / 4.0
            sec_in_day = int(bar_state.timestamp) % 86400
            tod_norm = sec_in_day / 86400.0
            
            l0 = [v2_vec[0], tod_norm, day_norm] # [3]
            grid = v2_vec[1:185].reshape(8, 23) # [8, 23]
            
            state_queue.append((l0, grid))
            
            if len(state_queue) < 60:
                continue
            
            if ledger.is_flat:
                # Build tensors
                l0_tensor = torch.tensor(np.array([s[0] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
                grid_tensor = torch.tensor(np.array([s[1] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)
                grid_tensor = grid_tensor.permute(0, 2, 1, 3)
                
                l0_tensor = torch.nan_to_num(l0_tensor, nan=0.0)
                grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0)
                
                heads, _ = master_net(grid_tensor, l0_tensor)
                
                # Head 1 is the Directional Action Space
                q_action = heads[0]
                
                # PURE EXPLOITATION: No probabilities, no multinomial, pure argmax.
                action = q_action.argmax(dim=1).item()
                
                if action in [1, 2]:
                    direction = 'long' if action == 1 else 'short'
                    ledger.add_position(
                        direction=direction,
                        entry_price=bar_state.price,
                        entry_ts=bar_state.timestamp,
                        entry_tier='RL_AGENT',
                        entry_features=bar_state.v2_vector
                    )
            else:
                ledger.update_bar(bar_state.v2_vector, bar_state.price, bar_state.timestamp)
                pos = ledger.primary
                
                exit_reason = None
                for rule in exit_suite:
                    exit_reason = rule.evaluate(bar_state, pos)
                    if exit_reason:
                        break
                        
                if exit_reason:
                    record = ledger.remove_position(pos.contract_id, bar_state.price, bar_state.timestamp, exit_reason)
                    trade_pnl = record['pnl']
                    
                    num_trades += 1
                    if trade_pnl > 0:
                        gross_profit += trade_pnl
                    else:
                        gross_loss += abs(trade_pnl)

    # 5. Final Report
    print(f"\n========================================================")
    print(f"               FULL OOS DIAGNOSTIC REPORT (RESEARCH A)")
    print(f"========================================================")
    print(f" Days Evaluated : {len(oos_days)}")
    print(f" Total Trades   : {num_trades}")
    print(f" Gross Profit   : ${gross_profit:.2f}")
    print(f" Gross Loss     : ${gross_loss:.2f}")
    
    net_pnl = gross_profit - gross_loss
    profit_factor = 999.0 if gross_loss == 0.0 else gross_profit / gross_loss
    metric_n = profit_factor - 1.0
    
    print(f" Net PnL        : ${net_pnl:.2f}")
    print(f" Profit Factor  : {profit_factor:.4f}")
    print(f" Metric (n)     : {metric_n:.4f}")
    print(f"========================================================")

if __name__ == "__main__":
    run_oos_diagnostic()
