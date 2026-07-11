import os
import glob
import pandas as pd
import numpy as np
import importlib.util
from datetime import datetime
import concurrent.futures
import subprocess
import sys

sys.path.append(r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI")
from core_v2.telemetry.reporter import TelemetryReporter
from tools.fspace_ml.pytorch_stepwise import pytorch_stepwise_forward

base_dir = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI"

def get_all_strategies():
    tests_dir = os.path.join(base_dir, "research", "nt8_catalog", "tests")
    folders = [f.path for f in os.scandir(tests_dir) if f.is_dir() and not f.name.startswith('.')]
    
    strategies = {}
    for folder in folders:
        strategy_name = os.path.basename(folder)
        py_files = glob.glob(os.path.join(folder, "ag_deepdive_*.py"))
        if not py_files: continue
        strategies[strategy_name] = (py_files[0], folder)
    return strategies

def extract_features_for_event(event, df_5s, df_1s):
    I_rth = event.get('event_idx', -1)
    I_exit_rth = event.get('exit_idx', -1)
    
    if I_rth == -1: return None
    
    day_str = event['day'].replace('_', '-') 
    rth_start = pd.Timestamp(f"{day_str} 08:30:00", tz='America/Chicago')
    
    event_ts = rth_start.timestamp() + (I_rth * 5)
    exit_ts = rth_start.timestamp() + (I_exit_rth * 5) if I_exit_rth != -1 else -1
    
    match_5s = df_5s.index[df_5s['timestamp'] == event_ts]
    if len(match_5s) == 0: return None
    I_5 = match_5s[0]
    
    match_exit = df_5s.index[df_5s['timestamp'] == exit_ts] if exit_ts != -1 else []
    I_exit = match_exit[0] if len(match_exit) > 0 else -1
    
    if I_5 < 180 or I_5 >= len(df_5s) - 12:
        return None
        
    match_1s = df_1s.index[df_1s['timestamp'] == event_ts]
    if len(match_1s) == 0: return None
    I_1 = match_1s[0]
    
    if I_1 < 5 or I_1 >= len(df_1s):
        return None
        
    features = {}
    features['hit'] = event['hit']
    features['magnitude'] = event['magnitude']
    
    cols_1s = [c for c in df_1s.columns if c != 'timestamp']
    cols_5s = [c for c in df_5s.columns if c != 'timestamp']
    
    # Phase 1
    for offset in range(1, 6):
        row = df_1s.iloc[I_1 - offset][cols_1s]
        for c in cols_1s: features[f"Ph1_1s_Tminus{offset}_{c}"] = row[c]
            
    for offset in [1, 2, 3]:
        row = df_5s.iloc[I_5 - offset][cols_5s]
        for c in cols_5s: features[f"Ph1_5s_Tminus{offset}_{c}"] = row[c]
            
    for offset in [6, 9, 12]:
        row = df_5s.iloc[I_5 - offset][cols_5s]
        for c in cols_5s: features[f"Ph1_15s_Tminus{offset//3}_{c}"] = row[c]
            
    row = df_5s.iloc[I_5 - 60][cols_5s]
    for c in cols_5s: features[f"Ph1_5m_Tminus1_{c}"] = row[c]
        
    # Phase 2
    if I_exit != -1 and I_exit > I_5:
        hold_end = min(I_exit, I_5 + 720) 
        during_slice = df_5s.iloc[I_5 : hold_end][cols_5s]
        if len(during_slice) > 0:
            mx = during_slice.max()
            mn = during_slice.min()
            for c in cols_5s:
                features[f"Ph2_Max_{c}"] = mx[c]
                features[f"Ph2_Min_{c}"] = mn[c]
                
    # Phase 3
    if I_exit != -1 and I_exit >= 3:
        row = df_5s.iloc[I_exit - 1][cols_5s]
        for c in cols_5s: features[f"Ph3_ExitMinus1_{c}"] = row[c]
            
    # Phase 4
    if I_exit != -1 and I_exit + 12 < len(df_5s):
        row = df_5s.iloc[I_exit + 12][cols_5s]
        for c in cols_5s: features[f"Ph4_ExitPlus1m_{c}"] = row[c]
            
    return features

def load_features(day, grid):
    features_dir = os.path.join(base_dir, f"DATA/ATLAS/FEATURES_{grid}_v2")
    if not os.path.exists(features_dir): return None
    
    layer_folders = [f.path for f in os.scandir(features_dir) if f.is_dir()]
    dfs = []
    
    for folder in layer_folders:
        fpath = os.path.join(folder, f"{day}.parquet")
        if os.path.exists(fpath):
            try:
                df = pd.read_parquet(fpath)
                dfs.append(df)
            except Exception:
                pass
                
    if not dfs: return None
    combined = pd.concat(dfs, axis=1)
    
    # Drop duplicate columns (all layers might have saved 'timestamp')
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    if 'timestamp' not in combined.columns:
        combined['timestamp'] = combined.index.astype(np.int64) // 10**9
        
    return combined

def worker_process_day(worker_args):
    day_idx, all_days, strategy_paths = worker_args
    day = all_days[day_idx]
    day_results = {s_name: [] for s_name in strategy_paths.keys()}
    
    df_5s = load_features(day, '5s')
    df_1s = load_features(day, '1s')
    
    if df_5s is None or df_1s is None:
        return day, day_results

    base_dir_local = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    l0_dir_local = os.path.join(base_dir_local, 'DATA/ATLAS/5s')

    for s_name, s_path in strategy_paths.items():
        try:
            spec = importlib.util.spec_from_file_location(f"module_{s_name}", s_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            events = []
            if hasattr(module, 'process_day'):
                varnames = module.process_day.__code__.co_varnames
                
                # Check if it strictly takes `args` tuple
                if 'args' in varnames and 'day' not in varnames:
                    # Attempt to compute prior day profile
                    yest_profile = None
                    if day_idx > 0:
                        yest_day = all_days[day_idx - 1]
                        yest_path = os.path.join(l0_dir_local, f"{yest_day}.parquet")
                        if hasattr(module, 'compute_prior_day_ohlc'):
                            yest_profile = module.compute_prior_day_ohlc(yest_path)
                        elif hasattr(module, 'compute_daily_profile'):
                            yest_profile = module.compute_daily_profile(yest_path)
                    
                    if yest_profile is not None:
                        events_out = module.process_day((day, yest_profile))
                    else:
                        # Fallback for ADX-08 which takes args but unpacks day=args directly
                        events_out = module.process_day(day)
                elif 'df' in varnames:
                    events_out = module.process_day(day)
                else:
                    events_out = module.process_day(day)
                
                if events_out:
                    events = events_out
            
            for event in events:
                event['day'] = day
                feat = extract_features_for_event(event, df_5s, df_1s)
                if feat:
                    day_results[s_name].append(feat)
                    
        except Exception as e:
            pass # Silently skip errors to avoid crashing worker
            
    return day, day_results

def process_strategy_ml(strategy_name, df_dataset, folder):
    if len(df_dataset) < 10:
        return
        
    df_dataset = df_dataset.dropna(axis=1, how='all')
    df_dataset = df_dataset.fillna(0)
    
    y = df_dataset['hit'].values
    if len(np.unique(y)) < 2:
        return
        
    X_df = df_dataset.drop(columns=['hit', 'magnitude'])
    X = X_df.values
    
    reporter = TelemetryReporter(f"ml_{strategy_name}")
    reporter.update(0, 1, f"GPU Stepwise LR: {X.shape[0]} samples, {X.shape[1]} features...")
    
    # PYTORCH GPU FORWARD SELECTION
    n_features_to_select = min(15, X.shape[1] // 2)
    selected_indices, pseudo_r2 = pytorch_stepwise_forward(X, y, n_features_to_select=n_features_to_select)
    
    reporter.clear()
    
    if not selected_indices:
        return
        
    selected_features = X_df.columns[selected_indices].tolist()
    
    # Write to dossier
    dossier_path = os.path.join(folder, 'augmentation', 'fspace_doe_report.md')
    if os.path.exists(dossier_path):
        result_lines = [
            f"",
            f"## ML Feature Extraction & Selection",
            f"- **Target:** Binary 'Hit' (Win Rate)",
            f"- **Total Samples:** {len(X)}",
            f"- **Total Dimensionality Explored:** {X.shape[1]} (Fractal Slice)",
            f"- **Pseudo R-Squared (McFadden):** {pseudo_r2:.4f}",
            f"- **Compute Engine:** PyTorch CUDA",
            f"",
            f"### Top Selected Features (Stepwise Forward Elimination)",
        ]
        for f in selected_features:
            result_lines.append(f"- `{f}`")
            
        with open(dossier_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(result_lines))

def main():
    # 1. Start Telemetry GUI
    if os.name == 'nt':
        subprocess.Popen([sys.executable, "core_v2/telemetry/gui.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    telemetry = TelemetryReporter("extraction_pipeline")
    
    # 2. Setup Data
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    strategies_info = get_all_strategies()
    strategy_paths = {k: v[0] for k, v in strategies_info.items()}
    
    ckpt_dir = os.path.join(base_dir, "DATA/ATLAS/ML_CHECKPOINTS")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Find days that are already completed across ALL strategies
    # Actually, tracking per strategy per day is safer, but for simplicity we group by month
    
    # 3. Process by Month to enable checkpointing
    months = sorted(list(set([d[:7] for d in days]))) # e.g. "2024_01"
    
    for m_idx, month in enumerate(months):
        month_days = [d for d in days if d.startswith(month)]
        
        telemetry.update(m_idx, len(months), f"Extracting {month}...")
        
        # Accumulators for this month
        month_features = {s: [] for s in strategies_info.keys()}
        
        needs_processing = False
        for s_name in strategies_info.keys():
            if not os.path.exists(os.path.join(ckpt_dir, f"{s_name}_{month}.parquet")):
                needs_processing = True
                break
                
        if not needs_processing:
            continue # Skip entire month if all strategies have this month checkpointed!
            
        # Parallel extraction for the days in this month
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as executor:
            # Pass worker_args tuple: (day_idx, days, strategy_paths)
            futures = {executor.submit(worker_process_day, (days.index(d), days, strategy_paths)): d for d in month_days}
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                d, day_results = future.result()
                for s_name, feats in day_results.items():
                    month_features[s_name].extend(feats)
                completed += 1
                telemetry.update(m_idx + (completed/len(month_days)), len(months), f"Extracting {month} ({completed}/{len(month_days)} days)")
                
        # Flush month to disk for checkpointing
        for s_name, feats in month_features.items():
            if feats:
                df = pd.DataFrame(feats)
                df.to_parquet(os.path.join(ckpt_dir, f"{s_name}_{month}.parquet"))
            else:
                # Create empty file to mark as done
                pd.DataFrame().to_parquet(os.path.join(ckpt_dir, f"{s_name}_{month}.parquet"))
                
    telemetry.update(len(months), len(months), "Data Extraction Complete. Booting PyTorch GPUs...")
    
    # 4. ML Processing (Load all months, run PyTorch GPU)
    for s_idx, (s_name, (_, folder)) in enumerate(strategies_info.items()):
        telemetry.update(s_idx, len(strategies_info), f"ML: {s_name}")
        
        # Load all checkpoints for this strategy
        s_ckpts = glob.glob(os.path.join(ckpt_dir, f"{s_name}_*.parquet"))
        dfs = []
        for c in s_ckpts:
            df = pd.read_parquet(c)
            if not df.empty: dfs.append(df)
            
        if not dfs: continue
        
        full_df = pd.concat(dfs, ignore_index=True)
        process_strategy_ml(s_name, full_df, folder)
        
    telemetry.clear()
    print("Full Pipeline Complete.")

if __name__ == '__main__':
    # Needed for Windows multiprocessing
    main()
