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
from training.models.cnn.model import GRID_FLAT_IDX, GRID_H, GRID_W
from training.train_trajectory_exit import TrajectoryExitModel

DOLLAR_PER_POINT = 2.0
FRICTION_USD = 6.0

def simulate_exits(csv_path: str, model_path: str, features_root: str, bars_dir: str, out_csv: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TrajectoryExitModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    is_path = REPO / 'DATA/ATLAS_NT8/exit_dataset/directional_exit_is.npz'
    is_data = np.load(is_path)
    X_dense_is = is_data['X_dense']
    dense_mean = np.mean(X_dense_is, axis=0)
    dense_std = np.std(X_dense_is, axis=0) + 1e-8
    
    trades = pd.read_csv(csv_path)
    if 'trade_id' not in trades.columns:
        trades['trade_id'] = np.arange(len(trades))
        
    days = trades['day'].unique()
    bars_dir = Path(bars_dir)
    results = []
    
    for day in tqdm(days, desc="Simulating Dual ML OOS Trades"):
        day_trades = trades[trades['day'] == day]
        
        bp = bars_dir / f'{day}.parquet'
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
            
            # Dir trades has r_price and atr_pts natively if we used zigzag underlying
            # Oh wait, we might have dropped r_price and atr_pts during inversion!
            # Let's handle missing gracefully
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
                    base_res[f'mae_at_cut_{int(thresh*100)}'] = np.nan
                    base_res[f'dur_{int(thresh*100)}'] = orig_duration
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
                    base_res[f'mae_at_cut_{int(thresh*100)}'] = np.nan
                    base_res[f'dur_{int(thresh*100)}'] = orig_duration
                results.append(base_res)
                continue
                
            X_grid_batch = np.stack(X_grid_batch, axis=0)
            X_dense_batch = np.stack(X_dense_batch, axis=0)
            X_dense_batch = (X_dense_batch - dense_mean) / dense_std
            
            t_grid = torch.tensor(X_grid_batch, dtype=torch.float32).to(device)
            t_dense = torch.tensor(X_dense_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits = model(t_grid, t_dense)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
            res = base_res.copy()
            for thresh in [0.30, 0.40, 0.50]:
                bail_idx = np.where(probs < thresh)[0]
                if len(bail_idx) > 0:
                    first_bail = bail_idx[0]
                    t_idx_exit = eval_points[first_bail]
                    res[f'new_pnl_{int(thresh*100)}'] = open_pnl[t_idx_exit] - FRICTION_USD
                    res[f'cut_{int(thresh*100)}'] = 1
                    res[f'mae_at_cut_{int(thresh*100)}'] = np.min(min_pnl[:t_idx_exit+1])
                    res[f'dur_{int(thresh*100)}'] = t_idx_exit
                else:
                    res[f'new_pnl_{int(thresh*100)}'] = orig_pnl
                    res[f'cut_{int(thresh*100)}'] = 0
                    res[f'mae_at_cut_{int(thresh*100)}'] = np.nan
                    res[f'dur_{int(thresh*100)}'] = orig_duration
                    
            results.append(res)
            
    df = pd.DataFrame(results)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"\nSimulation saved to {out_csv}")
    
    # Print summary metrics
    print("\n--- PERFORMANCE SUMMARY (OOS Dual ML) ---")
    orig_sum = df['orig_pnl'].sum()
    print(f"Base Directional ML (No Exit ML): ${orig_sum:,.0f} | WR: {100*(df['orig_pnl'] > 0).mean():.1f}%")
    
    for thresh in [0.30, 0.40, 0.50]:
        new_sum = df[f'new_pnl_{int(thresh*100)}'].sum()
        cuts = df[f'cut_{int(thresh*100)}'].sum()
        wr = 100 * (df[f'new_pnl_{int(thresh*100)}'] > 0).mean()
        print(f"Exit ML Threshold {thresh:.2f}: ${new_sum:,.0f} | WR: {wr:.1f}% | Exits: {cuts} / {len(df)} ({100*cuts/len(df):.1f}%)")

if __name__ == '__main__':
    csv_oos = REPO / 'reports/findings/directional_ml/dir_trades_oos.csv'
    model_path = REPO / 'checkpoints/directional_exit/best_model.pt'
    out_csv = REPO / 'reports/findings/directional_ml/dual_ml_oos.csv'
    
    simulate_exits(str(csv_oos), str(model_path), 'DATA/ATLAS_NT8/FEATURES_5s_v2', str(REPO / 'DATA/ATLAS_NT8/5s'), str(out_csv))
