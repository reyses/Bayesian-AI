import torch
import numpy as np
import os
import glob
from collections import deque
import torch.nn.functional as F
import argparse
import pandas as pd

from network_research_A import ResearchANetwork

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from core_v2.features import assemble_v2_grid
from core_v2.ledger import Ledger
from core_v2.exits import default_exit_suite

def compute_atr_pts(day_1m_path: str, period: int = 14) -> float:
    if not os.path.exists(day_1m_path):
        return 10.0
    df = pd.read_parquet(day_1m_path)
    h = df['high'].values; l = df['low'].values; c = df['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())

def run_diagnostic(target, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Booting Diagnostic Engine (Research A) for target: {target.upper()}")
    print(f"[INFO] Compute Device: {device}")

    # 1. Load Pre-trained Network
    master_net = ResearchANetwork().to(device)
    
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
    if target == 'is':
        atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
        features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    else:
        atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS_NT8"
        features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
        
    labels_csv = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/regime_labels_2d.csv"
    
    l0_dir = os.path.join(features_root, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    if target == 'is':
        days = [d for d in days if d.startswith('2025_')]
    else:
        days = [d for d in days if d.startswith('2026_')]
        
    print(f"[INFO] Discovered {len(days)} days for {target.upper()}.")
    
    if len(days) == 0:
        print("[ERROR] No data available.")
        return

    exit_suite = default_exit_suite()
    
    # 4. Stream data through FPS
    print(f"\n[EVAL] Executing pure Forward Pass System over {len(days)} days...")
    
    fps = MultiDayForwardPassSystem(
        atlas_root=atlas_root,
        features_root=features_root,
        labels_csv=labels_csv,
        days=days
    )
    
    ledger = Ledger()
    state_queue = deque(maxlen=60)
    
    gross_profit = 0.0
    gross_loss = 0.0
    num_trades = 0
    all_trades = []
    
    # Pre-compute ATRs
    atr_cache = {}
    for d in days:
        day_1m_path = os.path.join(atlas_root, '1m', f'{d}.parquet')
        atr_cache[d] = compute_atr_pts(day_1m_path)

    with torch.no_grad():
        for bar_idx, bar_state in enumerate(fps):
            if bar_idx > 0 and bar_idx % 10000 == 0:
                print(f"       Processed {bar_idx} bars... (Current Trades: {num_trades}, PnL: ${gross_profit - gross_loss:.2f})")
            
            v2_vec = bar_state.v2_vector
            if len(v2_vec) < 185:
                continue
                
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(bar_state.timestamp, tz=timezone.utc)
            day_norm = dt.weekday() / 4.0
            sec_in_day = int(bar_state.timestamp) % 86400
            tod_norm = sec_in_day / 86400.0
            
            l0 = [v2_vec[0], tod_norm, day_norm] # [3]
            # Store the FULL raw v2 row (canonical FEATURE_NAMES order). The CNN grid
            # is assembled via assemble_v2_grid (name-keyed placement) EXACTLY as the
            # training path does (ticker._v2_grid) — NOT a raw reshape. A reshape
            # scrambled the (tf, feature) mapping and used 23 features/TF instead of 25.
            state_queue.append((l0, np.asarray(v2_vec, dtype=np.float32)))

            if len(state_queue) < 60:
                continue

            if ledger.is_flat:
                # Build tensors with train/serve parity
                l0_tensor = torch.tensor(np.array([s[0] for s in state_queue]), dtype=torch.float32).unsqueeze(0).to(device)

                raw_window = np.stack([s[1] for s in state_queue])  # [60, N_FEATURES]
                # TODO(perf): assemble_v2_grid does ~200 FEATURE_NAMES.index() lookups
                # per call, and this runs on every flat bar. If OOS replay is slow,
                # precompute the (tf_idx, feat_idx) -> flat_idx map once and reuse it.
                window_grid = assemble_v2_grid(raw_window)           # [60, 8, 25]
                # [60, 8, 25] -> [8, 60, 25] -> [1, 8, 60, 25] to match network input
                grid_tensor = torch.tensor(window_grid, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0).to(device)

                l0_tensor = torch.nan_to_num(l0_tensor, nan=0.0)
                grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0)
                
                # Network returns (q_action [B,3], hidden_state). Was written for an
                # older multi-head API (heads[0] + argmax(dim=1)) which crashed.
                q_action, _ = master_net(grid_tensor, l0_tensor)  # [1, 3]

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
                    trade_pnl_usd = record['pnl']
                    
                    if record['dir'] == 'long':
                        trade_pnl_pts = record['exit_price'] - record['entry_price']
                    else:
                        trade_pnl_pts = record['entry_price'] - record['exit_price']
                    
                    num_trades += 1
                    if trade_pnl_usd > 0:
                        gross_profit += trade_pnl_usd
                    else:
                        gross_loss += abs(trade_pnl_usd)
                        
                    atr = atr_cache.get(bar_state.day, 10.0)
                    r_price = max(4, int(round(atr / 0.25 * 4))) * 0.25  # using mult=4
                        
                    all_trades.append({
                        'day': bar_state.day,
                        'entry_ts': int(record['entry_ts']),
                        'leg_dir': 'LONG' if record['dir'] == 'long' else 'SHORT',
                        'entry_price': record['entry_price'],
                        'exit_ts': int(record['exit_ts']),
                        'exit_price': record['exit_price'],
                        'pnl_pts': trade_pnl_pts,
                        'pnl_usd': trade_pnl_usd,
                        'r_price': r_price,
                        'atr_pts': atr
                    })

    # 5. Final Report
    print(f"\n========================================================")
    print(f"               FULL DIAGNOSTIC REPORT (RESEARCH A - {target.upper()})")
    print(f"========================================================")
    print(f" Days Evaluated : {len(days)}")
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
    
    if len(all_trades) > 0:
        out_dir = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/reports/findings/strategy_runs"
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"research_A_{target}.csv")
        df = pd.DataFrame(all_trades)
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved {len(df)} trades to {out_csv}")
    else:
        print("[WARN] No trades were generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, choices=['is', 'oos'], required=True)
    parser.add_argument('--ckpt', type=str, required=True, help="Path to weights")
    args = parser.parse_args()
    run_diagnostic(args.target, args.ckpt)
