"""Directional Agreement Filter.

Uses the 0.632 AUC directional model as a FILTER (not override):
    - If model's predicted direction AGREES with strategy → TAKE the trade
    - If model's predicted direction DISAGREES → SKIP the trade
    - No flipping, just selective entry

Also tests the INVERSE filter (take only when model disagrees),
and bucket analysis to find where the signal concentrates.
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

from tools.suites.trade_outcome_suite.train_pf10_entry import TrajectoryLSTM, build_trajectory_dataset
from sklearn.metrics import roc_auc_score

def mag_wr(pnl_arr):
    """Magnitude Win Rate: gross_profit / (gross_profit + gross_loss). 0.5 = breakeven."""
    if len(pnl_arr) == 0:
        return 0.0
    gp = pnl_arr[pnl_arr > 0].sum()
    gl = abs(pnl_arr[pnl_arr <= 0].sum())
    total = gp + gl
    return (gp / total) if total > 0 else 0.0

def pf(pnl_arr):
    """Profit Factor."""
    if len(pnl_arr) == 0:
        return 0.0
    gp = pnl_arr[pnl_arr > 0].sum()
    gl = abs(pnl_arr[pnl_arr <= 0].sum())
    return (gp / gl) if gl > 0 else 999.9

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    model_path = 'checkpoints/trajectory_pf10_entry/best_model.pt'
    
    print("Building OOS Dataset...")
    X_grid, X_tod, X_reg, X_dense, y, w = build_trajectory_dataset(
        csv_oos, features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', atlas_root='DATA/ATLAS_NT8')
    
    # Load raw trades for PnL and direction
    trades = pd.read_csv(csv_oos)
    # We need to align trades with the dataset (some trades get dropped in build_trajectory_dataset)
    # Rebuild the mapping
    from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
    
    pnl_list, dir_list = [], []
    days = trades['day'].unique()
    for day in tqdm(days, desc="Aligning PnL"):
        day_trades = trades[trades['day'] == day]
        feats = load_features(days=[day], root='DATA/ATLAS_NT8/FEATURES_5s_v2', require_all=False)
        if feats.empty:
            continue
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
        ts = feats['timestamp'].values.astype(np.int64)
        
        for _, trade in day_trades.iterrows():
            entry_ts = int(trade['entry_ts'])
            idx_arr = np.where(ts == entry_ts)[0]
            if len(idx_arr) == 0:
                continue
            end_idx = idx_arr[0]
            start_idx = end_idx - 60 + 1
            if start_idx < 0:
                continue
            pnl_list.append(float(trade['pnl_usd']))
            dir_list.append(1 if trade['leg_dir'] == 'LONG' else 0)
    
    pnl_arr = np.array(pnl_list)
    dir_arr = np.array(dir_list)
    
    assert len(pnl_arr) == len(y), f"Mismatch: {len(pnl_arr)} pnl vs {len(y)} samples"
    
    # Load model
    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Batched inference
    v_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
    v_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
    v_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
    v_dense = torch.tensor(X_dense, dtype=torch.float32).to(device)
    
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_grid), 512):
            logits = model(v_grid[i:i+512], v_tod[i:i+512], v_reg[i:i+512], v_dense[i:i+512])
            probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
    probs = np.concatenate(probs_list)  # P(Long) from the directional model
    
    n_total = len(probs)
    
    # ——— Base Strategy ———
    print(f"\n{'='*70}")
    print(f"=== BASE STRATEGY (ATR 2 OOS) — {n_total} trades ===")
    print(f"{'='*70}")
    print(f"Net PnL: ${pnl_arr.sum():+,.0f} | Mag WR: {mag_wr(pnl_arr):.3f} | PF: {pf(pnl_arr):.2f}")
    print(f"Winners: {(pnl_arr > 0).sum()} ({(pnl_arr > 0).mean()*100:.1f}%) | Losers: {(pnl_arr <= 0).sum()}")
    
    # ——— Agreement Score ———
    # P(model agrees with strategy direction)
    agreement = np.where(dir_arr == 1, probs, 1.0 - probs)
    
    # AUCs
    is_winner = (pnl_arr > 0).astype(int)
    true_dir = y.flatten()  # True market direction from training labels
    
    dir_auc = roc_auc_score(true_dir, probs)
    filter_auc = roc_auc_score(is_winner, agreement)
    print(f"\nDirectional AUC (Long vs Short): {dir_auc:.3f}")
    print(f"Filter AUC (Agreement predicts Winner): {filter_auc:.3f}")
    
    # ——— Agreement Filter (TAKE when model agrees) ———
    print(f"\n{'='*70}")
    print(f"=== AGREEMENT FILTER: Take trades where model agrees with strategy ===")
    print(f"{'='*70}")
    print(f"{'Threshold':>10} | {'Trades':>6} | {'Kept%':>5} | {'Net PnL':>10} | {'Mag WR':>7} | {'PF':>6} | {'Avg$':>7}")
    print("-" * 75)
    
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = agreement >= thresh
        sel = pnl_arr[mask]
        n = len(sel)
        if n < 5:
            print(f"{thresh:>10.2f} | {n:>6d} | {'<5 trades':>50}")
            continue
        kept_pct = n / n_total * 100
        avg = sel.mean()
        print(f"{thresh:>10.2f} | {n:>6d} | {kept_pct:>4.0f}% | ${sel.sum():>+9,.0f} | {mag_wr(sel):>7.3f} | {pf(sel):>6.2f} | ${avg:>+6.0f}")
    
    # ——— INVERSE Filter (TAKE when model DISAGREES) ———
    print(f"\n{'='*70}")
    print(f"=== INVERSE FILTER: Take trades where model DISAGREES with strategy ===")
    print(f"{'='*70}")
    print(f"{'Threshold':>10} | {'Trades':>6} | {'Kept%':>5} | {'Net PnL':>10} | {'Mag WR':>7} | {'PF':>6} | {'Avg$':>7}")
    print("-" * 75)
    
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = agreement < (1.0 - thresh)  # Strong disagreement
        sel = pnl_arr[mask]
        n = len(sel)
        if n < 5:
            print(f"{thresh:>10.2f} | {n:>6d} | {'<5 trades':>50}")
            continue
        kept_pct = n / n_total * 100
        avg = sel.mean()
        print(f"{thresh:>10.2f} | {n:>6d} | {kept_pct:>4.0f}% | ${sel.sum():>+9,.0f} | {mag_wr(sel):>7.3f} | {pf(sel):>6.2f} | ${avg:>+6.0f}")
    
    # ——— Confidence Bucket Analysis ———
    print(f"\n{'='*70}")
    print(f"=== P(Long) BUCKET ANALYSIS ===")
    print(f"{'='*70}")
    print(f"{'Bucket':>12} | {'N':>5} | {'Net PnL':>10} | {'Mag WR':>7} | {'PF':>6} | {'Avg$':>7} | {'%Long':>6} | {'%Win':>5}")
    print("-" * 80)
    
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        if n < 5:
            continue
        sel = pnl_arr[mask]
        pct_long = dir_arr[mask].mean() * 100
        pct_win = (sel > 0).mean() * 100
        avg = sel.mean()
        print(f"  [{lo:.1f},{hi:.1f}) | {n:>5d} | ${sel.sum():>+9,.0f} | {mag_wr(sel):>7.3f} | {pf(sel):>6.2f} | ${avg:>+6.0f} | {pct_long:>5.0f}% | {pct_win:>4.0f}%")
    
    # ——— Agreement Bucket Analysis ———
    print(f"\n{'='*70}")
    print(f"=== AGREEMENT SCORE BUCKET ANALYSIS ===")
    print(f"{'='*70}")
    print(f"{'Bucket':>12} | {'N':>5} | {'Net PnL':>10} | {'Mag WR':>7} | {'PF':>6} | {'Avg$':>7} | {'%Win':>5}")
    print("-" * 75)
    
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        mask = (agreement >= lo) & (agreement < hi)
        n = mask.sum()
        if n < 5:
            continue
        sel = pnl_arr[mask]
        pct_win = (sel > 0).mean() * 100
        avg = sel.mean()
        print(f"  [{lo:.1f},{hi:.1f}) | {n:>5d} | ${sel.sum():>+9,.0f} | {mag_wr(sel):>7.3f} | {pf(sel):>6.2f} | ${avg:>+6.0f} | {pct_win:>4.0f}%")

if __name__ == '__main__':
    evaluate()
