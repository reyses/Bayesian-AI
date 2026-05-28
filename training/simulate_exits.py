import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
from training.models.cnn.model import GRID_FLAT_IDX, GRID_H, GRID_W
from training.train_trajectory_exit import TrajectoryExitModel

REPO = Path(__file__).resolve().parents[1]
DOLLAR_PER_POINT = 2.0
FRICTION_USD = 6.0

def simulate_exits(csv_path: str, model_path: str, features_root: str, bars_dir: str, out_csv: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = TrajectoryExitModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Need to normalize dense features the exact same way as training.
    # We will load IS dataset to get the mean/std
    is_path = REPO / 'DATA/ATLAS_NT8/exit_dataset/exit_is.npz'
    is_data = np.load(is_path)
    X_dense_is = is_data['X_dense']
    dense_mean = np.mean(X_dense_is, axis=0)
    dense_std = np.std(X_dense_is, axis=0) + 1e-8
    
    trades = pd.read_csv(csv_path)
    # Add trade_id if missing
    if 'trade_id' not in trades.columns:
        trades['trade_id'] = np.arange(len(trades))
        
    days = trades['day'].unique()
    bars_dir = Path(bars_dir)
    
    results = []
    
    for day in tqdm(days, desc="Simulating OOS Trades"):
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
            r_price = float(trade['r_price'])
            atr_pts = float(trade['atr_pts'])
            
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
            if duration <= 1:
                results.append({
                    'trade_id': trade['trade_id'], 'day': day, 'hour': pd.to_datetime(entry_ts, unit='s').hour,
                    'leg_dir': leg_dir, 'entry_price': ep,
                    'orig_pnl': orig_pnl, 'orig_duration': orig_duration,
                    'new_pnl_20': orig_pnl, 'cut_20': 0, 'mae_at_cut_20': np.nan, 'dur_20': orig_duration,
                    'new_pnl_30': orig_pnl, 'cut_30': 0, 'mae_at_cut_30': np.nan, 'dur_30': orig_duration,
                    'new_pnl_40': orig_pnl, 'cut_40': 0, 'mae_at_cut_40': np.nan, 'dur_40': orig_duration,
                })
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
            
            # We evaluate every 3 bars (15 seconds) to be fast but precise enough
            # We don't evaluate at t=0 because we just entered
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
                results.append({
                    'trade_id': trade['trade_id'], 'day': day, 'hour': pd.to_datetime(entry_ts, unit='s').hour,
                    'leg_dir': leg_dir, 'entry_price': ep,
                    'orig_pnl': orig_pnl, 'orig_duration': orig_duration,
                    'new_pnl_20': orig_pnl, 'cut_20': 0, 'mae_at_cut_20': np.nan, 'dur_20': orig_duration,
                    'new_pnl_30': orig_pnl, 'cut_30': 0, 'mae_at_cut_30': np.nan, 'dur_30': orig_duration,
                    'new_pnl_40': orig_pnl, 'cut_40': 0, 'mae_at_cut_40': np.nan, 'dur_40': orig_duration,
                })
                continue
                
            X_grid_batch = np.stack(X_grid_batch, axis=0)
            X_dense_batch = np.stack(X_dense_batch, axis=0)
            
            # Normalize dense
            X_dense_batch = (X_dense_batch - dense_mean) / dense_std
            
            t_grid = torch.tensor(X_grid_batch, dtype=torch.float32).to(device)
            t_dense = torch.tensor(X_dense_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits = model(t_grid, t_dense)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
            # Evaluate thresholds
            res = {
                'trade_id': trade['trade_id'], 'day': day, 'hour': pd.to_datetime(entry_ts, unit='s').hour,
                'leg_dir': leg_dir, 'entry_price': ep,
                'orig_pnl': orig_pnl, 'orig_duration': orig_duration,
            }
            
            for thresh in [0.20, 0.30, 0.40]:
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
    print(f"Simulation saved to {out_csv}")

if __name__ == '__main__':
    csv_oos = REPO / 'reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv'
    model_path = REPO / 'checkpoints/trajectory_exit/best_model.pt'
    out_csv = REPO / 'reports/findings/exit_simulation_oos.csv'
    
    simulate_exits(str(csv_oos), str(model_path), 'DATA/ATLAS_NT8/FEATURES_5s_v2', str(REPO / 'DATA/ATLAS_NT8/5s'), str(out_csv))
