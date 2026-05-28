"""Full System Simulator.

1. Entry Filter: Uses Directional Entry ML to predict P(Long). We only take trades
   where the model DISAGREES with the strategy direction (agreement < 0.42).
   This is our "disagreement filter" that turned a -$9k system into +$2k.
   
2. Exit ML: For the trades that pass the filter, we evaluate them bar-by-bar
   using the Directional Exit ML. If P(Hold) < threshold, we bail early.
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

from core_v2.features import load_features, FEATURE_NAMES
from training.models.cnn.model import GRID_FLAT_IDX, GRID_H, GRID_W, N_REGIMES, REGIME_EMBED
from tools.suites.trade_outcome_suite.train_pf10_entry import TrajectoryLSTM, build_trajectory_dataset
from training.train_trajectory_exit import TrajectoryExitModel

DOLLAR_PER_POINT = 2.0
FRICTION_USD = 6.0

def simulate_full_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_oos = 'reports/findings/multi_atr/multi_atr_oos_atr2.csv'
    entry_model_path = 'checkpoints/trajectory_pf10_entry/best_model.pt'
    exit_model_path = 'checkpoints/filtered_exit/best_model.pt'
    features_root = 'DATA/ATLAS_NT8/FEATURES_5s_v2'
    bars_dir = 'DATA/ATLAS_NT8/5s'
    out_csv = 'reports/findings/full_system/full_system_oos.csv'
    
    # ---------------------------------------------------------
    # STEP 1: Directional Entry Filter
    # ---------------------------------------------------------
    print("\n--- STEP 1: Directional Entry Filter ---")
    print("Building OOS Dataset for Entry ML...")
    X_grid, X_tod, X_reg, X_dense, y, w = build_trajectory_dataset(
        csv_oos, features_root=features_root, atlas_root='DATA/ATLAS_NT8')
    
    trades = pd.read_csv(csv_oos)
    if 'trade_id' not in trades.columns:
        trades['trade_id'] = np.arange(len(trades))
        
    pnl_list, dir_list, trade_idx_list = [], [], []
    days = trades['day'].unique()
    for day in tqdm(days, desc="Aligning Trades"):
        day_trades = trades[trades['day'] == day]
        feats = load_features(days=[day], root=features_root, require_all=False)
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
            trade_idx_list.append(trade['trade_id'])
            
    pnl_arr = np.array(pnl_list)
    dir_arr = np.array(dir_list)
    trade_idx_arr = np.array(trade_idx_list)
    
    assert len(pnl_arr) == len(y), f"Mismatch: {len(pnl_arr)} pnl vs {len(y)} samples"
    
    print("Loading Entry Model...")
    entry_model = TrajectoryLSTM().to(device)
    entry_model.load_state_dict(torch.load(entry_model_path, map_location=device, weights_only=True))
    entry_model.eval()
    
    v_grid = torch.tensor(X_grid, dtype=torch.float32).to(device)
    v_tod = torch.tensor(X_tod, dtype=torch.float32).to(device)
    v_reg = torch.tensor(X_reg, dtype=torch.long).to(device)
    v_dense = torch.tensor(X_dense, dtype=torch.float32).to(device)
    
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_grid), 512):
            logits = entry_model(v_grid[i:i+512], v_tod[i:i+512], v_reg[i:i+512], v_dense[i:i+512])
            probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
    probs = np.concatenate(probs_list)  # P(Long)
    
    # Calculate agreement and filter
    agreement = np.where(dir_arr == 1, probs, 1.0 - probs)
    disagreement_thresh = 0.58
    agreement_thresh = 1.0 - disagreement_thresh  # 0.42
    
    keep_mask = agreement < agreement_thresh
    filtered_trade_ids = trade_idx_arr[keep_mask]
    
    filtered_trades = trades[trades['trade_id'].isin(filtered_trade_ids)]
    
    base_net = pnl_arr.sum()
    filt_net = pnl_arr[keep_mask].sum()
    
    print(f"\nTotal Trades: {len(trades)} | Net: ${base_net:+,.0f}")
    print(f"Filtered Trades (Disagreement >= {disagreement_thresh:.2f}): {len(filtered_trades)} | Net: ${filt_net:+,.0f}")
    print(f"Kept {len(filtered_trades)/len(trades)*100:.1f}% of trades.")
    
    if len(filtered_trades) == 0:
        print("No trades passed the filter!")
        return

    # ---------------------------------------------------------
    # STEP 2: Exit ML
    # ---------------------------------------------------------
    print("\n--- STEP 2: Exit ML on Filtered Trades ---")
    print("Loading Exit Model...")
    exit_model = TrajectoryExitModel().to(device)
    exit_model.load_state_dict(torch.load(exit_model_path, map_location=device, weights_only=True))
    exit_model.eval()
    
    # Load scaling stats for Exit ML
    is_path = REPO / 'DATA/ATLAS_NT8/exit_dataset/cascaded_exit_is.npz'
    is_data = np.load(is_path)
    X_dense_is = is_data['X_dense']
    dense_mean = np.mean(X_dense_is, axis=0)
    dense_std = np.std(X_dense_is, axis=0) + 1e-8
    
    bars_dir_path = Path(bars_dir)
    results = []
    days = filtered_trades['day'].unique()
    
    for day in tqdm(days, desc="Simulating Exits"):
        day_trades = filtered_trades[filtered_trades['day'] == day]
        
        bp = bars_dir_path / f'{day}.parquet'
        if not bp.exists():
            continue
        b = pd.read_parquet(bp).sort_values('timestamp').reset_index(drop=True)
        ts_bars = b['timestamp'].values.astype(np.int64)
        hi = b['high'].values.astype(np.float64)
        lo = b['low'].values.astype(np.float64)
        close = b['close'].values.astype(np.float64)
        
        feats = load_features(days=[day], root=features_root, require_all=False)
        if feats.empty:
            continue
        feats = feats.sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
            feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
            
        ts_feats = feats['timestamp'].values.astype(np.int64)
        
        v2_matrix = np.zeros((len(feats), len(FEATURE_NAMES)), dtype=np.float32)
        feat_cols = set(feats.columns)
        for j, name in enumerate(FEATURE_NAMES):
            if name in feat_cols:
                v2_matrix[:, j] = feats[name].values.astype(np.float32)
                
        grids_all = v2_matrix[:, GRID_FLAT_IDX].reshape(-1, GRID_H, GRID_W)
        
        for _, trade in day_trades.iterrows():
            entry_ts, exit_ts = int(trade['entry_ts']), int(trade['exit_ts'])
            ep = float(trade['entry_price'])
            leg_dir = 1 if trade['leg_dir'] == 'LONG' else -1
            orig_pnl = float(trade['pnl_usd'])
            
            r_price = float(trade.get('r_price', ep))
            atr_pts = float(trade.get('atr_pts', 1.0))
            
            ei_bar = int(np.searchsorted(ts_bars, entry_ts, side='left'))
            if ei_bar >= len(ts_bars) or ts_bars[ei_bar] != entry_ts:
                ei_bar = int(np.searchsorted(ts_bars, entry_ts, side='right') - 1)
            xi_bar = int(np.searchsorted(ts_bars, exit_ts, side='right') - 1)
            xi_bar = max(xi_bar, ei_bar)
            
            ei_feat = int(np.searchsorted(ts_feats, entry_ts, side='left'))
            if ei_feat >= len(ts_feats) or ts_feats[ei_feat] != entry_ts:
                ei_feat = int(np.searchsorted(ts_feats, entry_ts, side='right') - 1)
            
            if ei_feat < 59:
                continue
                
            duration = xi_bar - ei_bar
            orig_duration = duration
            
            base_res = {
                'trade_id': trade['trade_id'], 'day': day, 'hour': pd.to_datetime(entry_ts, unit='s').hour,
                'leg_dir': leg_dir, 'entry_price': ep,
                'orig_pnl': orig_pnl, 'orig_duration': orig_duration,
            }
            if duration <= 1:
                for thresh in [0.30, 0.40, 0.50]:
                    base_res[f'new_pnl_{int(thresh*100)}'] = orig_pnl
                    base_res[f'cut_{int(thresh*100)}'] = 0
                results.append(base_res)
                continue
                
            sh, sl, sc = hi[ei_bar:xi_bar+1], lo[ei_bar:xi_bar+1], close[ei_bar:xi_bar+1]
            if leg_dir == 1:
                min_pnl = (sl - ep) * DOLLAR_PER_POINT
                max_pnl = (sh - ep) * DOLLAR_PER_POINT
                open_pnl = (sc - ep) * DOLLAR_PER_POINT
            else:
                min_pnl = (ep - sh) * DOLLAR_PER_POINT
                max_pnl = (ep - sl) * DOLLAR_PER_POINT
                open_pnl = (ep - sc) * DOLLAR_PER_POINT
                
            entry_grid = grids_all[ei_feat] * leg_dir
            
            eval_points = list(range(3, duration, 3))
            if not eval_points:
                eval_points = [duration - 1]
                
            X_grid_batch = []
            X_dense_batch = []
            
            for t_idx in eval_points:
                t_bar = ei_bar + t_idx
                t_ts = ts_bars[t_bar]
                t_feat = int(np.searchsorted(ts_feats, t_ts, side='right') - 1)
                
                if t_feat < 59 or t_feat >= len(grids_all):
                    continue
                    
                start_feat = t_feat - 60 + 1
                traj_grid = grids_all[start_feat:t_feat+1] * leg_dir
                delta_grid = traj_grid - entry_grid
                two_channel = np.stack([traj_grid, delta_grid], axis=1)
                
                cur_open_pnl = open_pnl[t_idx]
                cur_mae = np.min(min_pnl[:t_idx+1])
                cur_mfe = np.max(max_pnl[:t_idx+1])
                trade_vel = cur_open_pnl / t_idx
                
                sc_now = close[ei_bar + t_idx]
                if leg_dir == 1:
                    dist_to_r = sc_now - r_price
                else:
                    dist_to_r = r_price - sc_now
                dist_r_atr = dist_to_r / atr_pts if atr_pts > 0 else 0.0
                
                dense_vec = np.array([cur_open_pnl, cur_mae, cur_mfe, t_idx, trade_vel, dist_r_atr], dtype=np.float32)
                
                X_grid_batch.append(two_channel)
                X_dense_batch.append(dense_vec)
                
            if not X_grid_batch:
                for thresh in [0.30, 0.40, 0.50]:
                    base_res[f'new_pnl_{int(thresh*100)}'] = orig_pnl
                    base_res[f'cut_{int(thresh*100)}'] = 0
                results.append(base_res)
                continue
                
            X_grid_batch = np.stack(X_grid_batch, axis=0)
            X_dense_batch = np.stack(X_dense_batch, axis=0)
            X_dense_batch = (X_dense_batch - dense_mean) / dense_std
            
            t_grid = torch.tensor(X_grid_batch, dtype=torch.float32).to(device)
            t_dense = torch.tensor(X_dense_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits = exit_model(t_grid, t_dense)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
            res = base_res.copy()
            for thresh in [0.30, 0.40, 0.50]:
                bail_idx = np.where(probs < thresh)[0]
                if len(bail_idx) > 0:
                    first_bail = bail_idx[0]
                    t_idx_exit = eval_points[first_bail]
                    res[f'new_pnl_{int(thresh*100)}'] = open_pnl[t_idx_exit] - FRICTION_USD
                    res[f'cut_{int(thresh*100)}'] = 1
                else:
                    res[f'new_pnl_{int(thresh*100)}'] = orig_pnl
                    res[f'cut_{int(thresh*100)}'] = 0
                    
            results.append(res)
            
    df = pd.DataFrame(results)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"\nSimulation saved to {out_csv}")
    
    # Print summary metrics
    print("\n--- PERFORMANCE SUMMARY (Entry Filter + Exit ML) ---")
    orig_sum = df['orig_pnl'].sum()
    orig_gp = df['orig_pnl'][df['orig_pnl'] > 0].sum()
    orig_gl = abs(df['orig_pnl'][df['orig_pnl'] <= 0].sum())
    orig_magwr = orig_gp / (orig_gp + orig_gl) if (orig_gp + orig_gl) > 0 else 0
    orig_pf = orig_gp / orig_gl if orig_gl > 0 else 999.9
    print(f"Base Filtered Trades (No Exit ML): ${orig_sum:,.0f} | Mag WR: {orig_magwr:.3f} | PF: {orig_pf:.2f}")
    
    for thresh in [0.30, 0.40, 0.50]:
        new_pnl = df[f'new_pnl_{int(thresh*100)}']
        new_sum = new_pnl.sum()
        cuts = df[f'cut_{int(thresh*100)}'].sum()
        
        new_gp = new_pnl[new_pnl > 0].sum()
        new_gl = abs(new_pnl[new_pnl <= 0].sum())
        new_magwr = new_gp / (new_gp + new_gl) if (new_gp + new_gl) > 0 else 0
        new_pf = new_gp / new_gl if new_gl > 0 else 999.9
        
        print(f"Exit ML Threshold {thresh:.2f}: ${new_sum:,.0f} (Delta: ${new_sum - orig_sum:,.0f}) | Mag WR: {new_magwr:.3f} | PF: {new_pf:.2f} | Exits: {cuts} / {len(df)} ({100*cuts/len(df):.1f}%)")

if __name__ == '__main__':
    simulate_full_system()
