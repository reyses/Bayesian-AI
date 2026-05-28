"""Stay/Flip OOS Simulator.

Loads the trained Stay/Flip model and evaluates it on OOS trades.
For each trade, the model outputs P(Stay):
    P(Stay) > threshold → keep original direction, PnL unchanged
    P(Stay) < threshold → flip direction, PnL = -(original_pnl + friction) - friction

Reports:
    - Magnitude Win Rate (Profit Ratio): gross_profit / (gross_profit + gross_loss)
      0.5 = breakeven, > 0.5 = profitable, 1.0 = pure profit
    - Profit Factor
    - Net PnL
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.suites.trade_outcome_suite.train_stay_flip import StayFlipLSTM, build_stay_flip_dataset
from sklearn.metrics import roc_auc_score

FRICTION = 6.0  # Round-trip friction per trade

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    model_dir = 'checkpoints/stay_flip_atr2'
    
    print("Building OOS Dataset...")
    X_grid, X_tod, X_reg, X_dense, y, w, pnl, leg_dir = build_stay_flip_dataset(
        csv_oos, features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', atlas_root='DATA/ATLAS_NT8')
    
    n_total = len(y)
    print(f"OOS: {n_total} trades | Winners: {y.sum():.0f} ({y.mean()*100:.1f}%) | Losers: {(1-y).sum():.0f}")
    
    # Base strategy stats
    base_net = pnl.sum()
    base_gp = pnl[pnl > 0].sum()
    base_gl = abs(pnl[pnl <= 0].sum())
    base_mag_wr = base_gp / (base_gp + base_gl) if (base_gp + base_gl) > 0 else 0.0
    base_pf = base_gp / base_gl if base_gl > 0 else 999.9
    print(f"\n=== BASE STRATEGY (ATR 2 OOS) ===")
    print(f"Trades: {n_total} | Net PnL: ${base_net:+,.0f} | Mag WR: {base_mag_wr:.3f} | PF: {base_pf:.2f}")
    
    # Try both checkpoints
    for ckpt_name in ['best_model.pt', 'best_magwr_model.pt']:
        model_path = os.path.join(model_dir, ckpt_name)
        if not os.path.exists(model_path):
            print(f"\nCheckpoint {ckpt_name} not found, skipping.")
            continue
            
        print(f"\n{'='*60}")
        print(f"=== STAY/FLIP MODEL: {ckpt_name} ===")
        print(f"{'='*60}")
        
        model = StayFlipLSTM().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Batched inference
        v_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
        v_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
        v_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
        v_dense = torch.tensor(X_dense, dtype=torch.float32).to(device)
        
        probs_list = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                logits = model(v_grid[i:i+batch_size], v_tod[i:i+batch_size],
                             v_reg[i:i+batch_size], v_dense[i:i+batch_size])
                probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
        probs = np.concatenate(probs_list)  # P(Stay)
        
        # AUC: How well does the model separate winners from losers?
        auc = roc_auc_score(y.flatten(), probs)
        print(f"OOS AUC (Stay/Flip): {auc:.3f}")
        
        # ——— Threshold Analysis ———
        print(f"\n--- Stay/Flip Override by Confidence Threshold ---")
        print(f"{'Threshold':>10} | {'Trades':>6} | {'Flipped':>7} | {'Net PnL':>10} | {'Mag WR':>7} | {'PF':>6}")
        print("-" * 65)
        
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            # Stay if P(Stay) >= thresh, Flip if P(Stay) < (1-thresh)
            # Skip (no trade) if in between
            stay_mask = probs >= thresh
            flip_mask = probs < (1.0 - thresh)
            skip_mask = ~stay_mask & ~flip_mask
            
            stay_pnl = pnl[stay_mask]
            flip_pnl = -pnl[flip_mask] - 2 * FRICTION  # Invert and pay friction both ways
            
            combined_pnl = np.concatenate([stay_pnl, flip_pnl]) if len(stay_pnl) + len(flip_pnl) > 0 else np.array([])
            
            n_trades = len(combined_pnl)
            n_flipped = flip_mask.sum()
            
            if n_trades == 0:
                print(f"{thresh:>10.2f} | {'0 trades':>40}")
                continue
            
            net = combined_pnl.sum()
            gp = combined_pnl[combined_pnl > 0].sum()
            gl = abs(combined_pnl[combined_pnl <= 0].sum())
            mag_wr = gp / (gp + gl) if (gp + gl) > 0 else 0.0
            pf = gp / gl if gl > 0 else 999.9
            
            print(f"{thresh:>10.2f} | {n_trades:>6d} | {n_flipped:>7d} | ${net:>+9,.0f} | {mag_wr:>7.3f} | {pf:>6.2f}")
        
        # ——— Simple Override (all trades, hard 0.5 decision) ———
        print(f"\n--- Simple Override (all trades, threshold=0.5) ---")
        stay_all = probs >= 0.5
        override_pnl = np.where(stay_all, pnl, -pnl - 2 * FRICTION)
        
        net_override = override_pnl.sum()
        gp_o = override_pnl[override_pnl > 0].sum()
        gl_o = abs(override_pnl[override_pnl <= 0].sum())
        mag_wr_o = gp_o / (gp_o + gl_o) if (gp_o + gl_o) > 0 else 0.0
        pf_o = gp_o / gl_o if gl_o > 0 else 999.9
        n_flipped_all = (~stay_all).sum()
        
        print(f"Total: {n_total} | Flipped: {n_flipped_all} ({n_flipped_all/n_total*100:.1f}%)")
        print(f"Net PnL: ${net_override:+,.0f} | Mag WR: {mag_wr_o:.3f} | PF: {pf_o:.2f}")
        print(f"vs Base: ${net_override - base_net:+,.0f} delta")
        
        # ——— Confidence Distribution ———
        print(f"\n--- P(Stay) Distribution ---")
        for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                       (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
            mask = (probs >= lo) & (probs < hi)
            n = mask.sum()
            if n > 0:
                bucket_pnl = pnl[mask]
                bucket_wr = (bucket_pnl > 0).mean()
                bucket_net = bucket_pnl.sum()
                gp_b = bucket_pnl[bucket_pnl > 0].sum()
                gl_b = abs(bucket_pnl[bucket_pnl <= 0].sum())
                mag_b = gp_b / (gp_b + gl_b) if (gp_b + gl_b) > 0 else 0.0
                print(f"  [{lo:.1f}, {hi:.1f}) | n={n:5d} | Net: ${bucket_net:+8,.0f} | Mag WR: {mag_b:.3f}")

if __name__ == '__main__':
    evaluate()
