import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
from training.models.cnn.model import GRID_FLAT_IDX, GRID_H, GRID_W

REPO = Path(__file__).resolve().parents[1]
DOLLAR_PER_POINT = 2.0
FRICTION_USD = 6.0

def build_dataset(csv_path: str, out_file: str, features_root: str, bars_dir: str, seq_len: int = 60, samples_per_trade: int = 10):
    trades = pd.read_csv(csv_path)
    days = trades['day'].unique()
    
    bars_dir = Path(bars_dir)
    
    X_grid_list = []
    X_dense_list = []
    y_list = []
    w_list = []
    
    skipped = 0
    
    for day in tqdm(days, desc=f"Building {Path(out_file).name}"):
        day_trades = trades[trades['day'] == day]
        
        # Load 5s price bars for accurate PnL tracking
        bp = bars_dir / f'{day}.parquet'
        if not bp.exists():
            skipped += len(day_trades)
            continue
        b = pd.read_parquet(bp).sort_values('timestamp').reset_index(drop=True)
        ts_bars = b['timestamp'].values.astype(np.int64)
        hi = b['high'].values.astype(np.float64)
        lo = b['low'].values.astype(np.float64)
        close = b['close'].values.astype(np.float64)
        
        # Load V2 features
        feats = load_features(days=[day], root=features_root, require_all=False)
        if feats.empty:
            skipped += len(day_trades)
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
            final_pnl = float(trade['pnl_usd'])
            r_price = float(trade['r_price'])
            atr_pts = float(trade['atr_pts'])
            
            # Find indices in 5s bars
            ei_bar = int(np.searchsorted(ts_bars, entry_ts, side='left'))
            if ei_bar >= len(ts_bars) or ts_bars[ei_bar] != entry_ts:
                ei_bar = int(np.searchsorted(ts_bars, entry_ts, side='right') - 1)
            xi_bar = int(np.searchsorted(ts_bars, exit_ts, side='right') - 1)
            xi_bar = max(xi_bar, ei_bar)
            
            # Find indices in features
            ei_feat = int(np.searchsorted(ts_feats, entry_ts, side='left'))
            if ei_feat >= len(ts_feats) or ts_feats[ei_feat] != entry_ts:
                ei_feat = int(np.searchsorted(ts_feats, entry_ts, side='right') - 1)
            
            if ei_feat < seq_len - 1:
                continue # Not enough feature history
                
            duration = xi_bar - ei_bar
            if duration <= 1:
                continue # Too short to sample
                
            # Compute exact PnL path
            sh, sl, sc = hi[ei_bar:xi_bar+1], lo[ei_bar:xi_bar+1], close[ei_bar:xi_bar+1]
            if leg_dir == 1:
                min_pnl = (sl - ep) * DOLLAR_PER_POINT
                max_pnl = (sh - ep) * DOLLAR_PER_POINT
                open_pnl = (sc - ep) * DOLLAR_PER_POINT
            else:
                min_pnl = (ep - sh) * DOLLAR_PER_POINT
                max_pnl = (ep - sl) * DOLLAR_PER_POINT
                open_pnl = (ep - sc) * DOLLAR_PER_POINT
                
            # Sample points within the trade
            # Always sample the worst point (MAE), the best point (MFE), and random points
            worst_bar = np.argmin(min_pnl)
            best_bar = np.argmax(max_pnl)
            
            samples = set([worst_bar, best_bar])
            possible_choices = list(set(range(1, duration)) - samples)
            needed = min(samples_per_trade, duration) - len(samples)
            if needed > 0 and possible_choices:
                chosen = np.random.choice(possible_choices, min(needed, len(possible_choices)), replace=False)
                samples.update(chosen)
            # Grid at entry for delta
            entry_grid = grids_all[ei_feat] * leg_dir # (8, 23)
            
            for t_idx in samples:
                # t_idx is relative to ei_bar.
                t_bar = ei_bar + t_idx
                t_ts = ts_bars[t_bar]
                
                # Find matching feature index
                t_feat = int(np.searchsorted(ts_feats, t_ts, side='right') - 1)
                
                # Check bounds
                if t_feat < seq_len - 1 or t_feat >= len(grids_all):
                    continue
                    
                # The trailing 60 bars of features
                start_feat = t_feat - seq_len + 1
                traj_grid = grids_all[start_feat:t_feat+1] * leg_dir # (60, 8, 23)
                
                # Channel 1: Delta from entry
                delta_grid = traj_grid - entry_grid # (60, 8, 23)
                
                # Stack to (60, 2, 8, 23)
                two_channel_grid = np.stack([traj_grid, delta_grid], axis=1)
                
                # Dense context features at time t
                cur_open_pnl = open_pnl[t_idx]
                cur_mae = np.min(min_pnl[:t_idx+1])
                cur_mfe = np.max(max_pnl[:t_idx+1])
                bars_in_trade = t_idx
                trade_velocity = cur_open_pnl / bars_in_trade if bars_in_trade > 0 else 0.0
                
                sc_now = close[ei_bar + t_idx]
                if leg_dir == 1:
                    dist_to_r = sc_now - r_price
                else:
                    dist_to_r = r_price - sc_now
                dist_r_atr = dist_to_r / atr_pts if atr_pts > 0 else 0.0
                
                # Dense vector
                dense_vec = np.array([cur_open_pnl, cur_mae, cur_mfe, bars_in_trade, trade_velocity, dist_r_atr], dtype=np.float32)
                
                # Label: Does holding make more money than bailing right now?
                # Bail value = cur_open_pnl - $6 friction. Hold value = final_pnl (friction already included).
                bail_val = cur_open_pnl - FRICTION_USD
                label = 1 if final_pnl > bail_val else 0
                weight = abs(final_pnl - bail_val)
                
                # Asymmetric penalty: if the trade ultimately closed green and we should have held, 
                # apply a 3x multiplier to brutally penalize false bails on massive winners.
                if final_pnl > 0 and label == 1:
                    weight *= 3.0
                
                X_grid_list.append(two_channel_grid)
                X_dense_list.append(dense_vec)
                y_list.append(label)
                w_list.append(weight)
                
    if not X_grid_list:
        print("No samples generated.")
        return
        
    X_grid = np.stack(X_grid_list, axis=0) # (N, 60, 2, 8, 23)
    X_grid = np.nan_to_num(X_grid, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_dense = np.stack(X_dense_list, axis=0) # (N, 5)
    X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = np.array(y_list, dtype=np.float32) # (N,)
    w = np.array(w_list, dtype=np.float32) # (N,)
    
    out_dir = Path(out_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(out_file, X_grid=X_grid, X_dense=X_dense, y=y, w=w)
    print(f"Saved {len(y)} samples to {out_file}")
    print(f"Class balance: {y.sum()} HOLD ({y.sum()/len(y)*100:.1f}%) / {len(y) - y.sum()} BAIL")

if __name__ == '__main__':
    csv_is = REPO / 'reports/findings/trade_outcome_table/entry_ml_filtered_IS.csv'
    csv_oos = REPO / 'reports/findings/trade_outcome_table/entry_ml_filtered_OOS.csv'
    
    out_dir = REPO / 'DATA/ATLAS_NT8/exit_dataset'
    
    print("=== Building IS Dataset ===")
    build_dataset(csv_is, out_dir / 'cascaded_exit_is.npz', features_root='DATA/ATLAS/FEATURES_5s_v2', bars_dir=REPO / 'DATA/ATLAS/5s')
    
    print("\n=== Building OOS Dataset ===")
    build_dataset(csv_oos, out_dir / 'cascaded_exit_oos.npz', features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', bars_dir=REPO / 'DATA/ATLAS_NT8/5s')

