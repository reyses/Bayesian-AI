import os
import sys
import glob
import time
import json
import subprocess
import psutil
import concurrent.futures
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI")
try:
    from telegram_mcp import send_telegram_alert, inject_prompt
except ImportError:
    def send_telegram_alert(msg): print(msg)
    def inject_prompt(msg): print(msg)

from core_v2.telemetry.reporter import TelemetryReporter
from core_v2.features import load_features

# Add local path so Windows multiprocessing can pickle the functions
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import stage2_parallel_chaos

def get_safe_parallel_days():
    return 1 # Hardcoded to 1 to prevent PCIe/CUDA context thrashing across multiple PyTorch instances

def process_day_stage1(day):
    from datetime import datetime
    stage1_file = f"artifacts/stage1_segments_{day}.json"
    if os.path.exists(stage1_file):
        return day, True, "Skipped", "Skipped", "0.0 mins"
    try:
        t0 = time.time()
        dt_start = datetime.now().strftime("%I:%M %p")
        
        subprocess.run([sys.executable, "research/Regression segments/stage1_speed_pass.py", "--day", day, "--hours", "24"], check=True)
        
        t1 = time.time()
        dt_end = datetime.now().strftime("%I:%M %p")
        dur_mins = (t1 - t0) / 60.0
        dur_str = f"{dur_mins:.1f} mins"
        
        return day, os.path.exists(stage1_file), dt_start, dt_end, dur_str
    except Exception as e:
        return day, False, "", "", str(e)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=None, help="Format YYYY_MM_DD")
    args = parser.parse_args()

    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    l0_dir = os.path.join(atlas_root, 'FEATURES_5s_v2', 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    if args.start_date:
        days = [d for d in days if d >= args.start_date]

    print(f"[RUNNER] Found {len(days)} total days of ATLAS data.")
    
    subprocess.Popen([sys.executable, "core_v2/telemetry/gui.py"], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    
    # ---- PHASE 1: STAGE 1 (PARALLEL DAYS) ----
    max_days = get_safe_parallel_days()
    print(f"\n========================================")
    print(f"[PHASE 1] Parallel Stage 1 (Dynamic Pool: {max_days} days)")
    print(f"========================================")
    send_telegram_alert(f"🚀 Phase 1: Scanning remaining days of data for pristine regimes...")
    
    phase1_reporter = TelemetryReporter("phase1_progress")
    completed_days = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_days) as executor:
        futures = {executor.submit(process_day_stage1, day): day for day in days}
        for future in concurrent.futures.as_completed(futures):
            completed_days += 1
            phase1_reporter.update(completed_days, len(days), "Phase 1: Stage 1 GPU Scan")
            day = futures[future]
            try:
                day_res, success, t_start, t_end, dur_str = future.result()
                if success:
                    print(f"  -> Finished Stage 1 for {day} ({dur_str})")
                    # If it was skipped, don't spam Telegram. Only alert on actual processed days.
                    if t_start != "Skipped":
                        send_telegram_alert(f"✔️ Day {day} finished! ({completed_days}/{len(days)})\n⏱️ {t_start} -> {t_end}\n⏳ Duration: {dur_str}")
            except Exception as e:
                print(f"[ERROR] Day {day} failed Stage 1: {e}")

    phase1_reporter.clear()
    
    # ---- PHASE 2: STAGE 2 (GLOBAL CHAOS BLOCKS) ----
    print(f"\n========================================")
    print(f"[PHASE 2] Collecting all Chaos Blocks across the year...")
    print(f"========================================")
    
    global_worker_args = []
    final_segments = []
    groups = None
    
    # Build the worker args by opening day features ONE at a time to save RAM
    for day in days:
        stage1_file = f"artifacts/stage1_segments_{day}.json"
        stage2_file = f"artifacts/stage2_segments_{day}.json"
        
        if os.path.exists(stage2_file):
            with open(stage2_file, 'r') as f:
                final_segments.extend(json.load(f))
            continue

        if not os.path.exists(stage1_file):
            continue
            
        with open(stage1_file, 'r') as f:
            all_segs = json.load(f)
            
        pristine = [s for s in all_segs if s['status'] == 'PRISTINE']
        chaos = [s for s in all_segs if s['status'] == 'UNPROCESSED_CHAOS']
        final_segments.extend(pristine)
        
        if len(chaos) > 0:
            features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
            try:
                df = load_features([day], root=features_root)
                ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f'{day}.parquet'))
            except:
                print(f"[WARN] Failed to load data for {day}, skipping its chaos blocks.")
                continue
                
            min_len = min(len(df), len(ohlcv))
            df = df.iloc[:min_len]
            ohlcv = ohlcv.iloc[:min_len]
            
            features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
            if groups is None:
                groups = stage2_parallel_chaos.build_groups_from_columns(features_cols)
                
            scaler = StandardScaler()
            X_global = scaler.fit_transform(df[features_cols].values)
            
            valid_idx = ~np.isnan(X_global).any(axis=1)
            X_global = X_global[valid_idx]
            close_prices = ohlcv['close'].values[valid_idx]
            
            for block in chaos:
                s_idx = block['start_idx']
                e_idx = block['end_idx']
                X_chunk = X_global[s_idx:e_idx]
                Y_chunk = close_prices[s_idx:e_idx]
                global_worker_args.append((
                    len(global_worker_args), 
                    block, 
                    day, 
                    block['error_band_used'], 
                    groups, 
                    X_chunk, 
                    Y_chunk
                ))
                
            # GC drop the large dataframes
            del df, ohlcv, X_global, close_prices
            
    print(f"[PHASE 2] Harvested {len(final_segments)} Pristine blocks directly.")
    print(f"[PHASE 2] Aggregated {len(global_worker_args)} Global Chaos blocks to resolve.")
    send_telegram_alert(f"🎯 Phase 1 Complete!\n\nHarvested {len(final_segments)} pristine blocks directly.\nNow pooling {len(global_worker_args)} scattered chaos gaps across {os.cpu_count()*2} CPU threads for Phase 2!")
    
    if len(global_worker_args) > 0:
        available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
        safe_ram_mb = max(0, available_ram_mb - 1024) # Keep 1GB explicitly free
        max_workers_by_ram = max(1, int(safe_ram_mb / 150)) # Assume ~150MB per worker
        max_workers_by_cpu = max(1, os.cpu_count() * 2)
        
        num_workers = min(max_workers_by_ram, max_workers_by_cpu)
        print(f"[PHASE 2] Dynamic Scaling: {available_ram_mb:.0f}MB Free RAM -> Spawning {num_workers} global workers.")
        
        phase2_reporter = TelemetryReporter("phase2_progress")
        
        # Windows requires top-level module functions for Pool, so we use stage2_parallel_chaos.process_chaos_block
        with Pool(processes=num_workers) as pool:
            # We use imap_unordered for fast iteration and progress reporting
            results = []
            for i, res in enumerate(pool.imap_unordered(stage2_parallel_chaos.process_chaos_block, global_worker_args)):
                results.append(res)
                phase2_reporter.update(i+1, len(global_worker_args), f"Phase 2: Resolving Chaos Blocks")
                
        for res in results:
            final_segments.extend(res)
            
        phase2_reporter.clear()
        
    final_segments.sort(key=lambda x: (x['day'], x['start_idx']))
    
    output_json = "artifacts/stage2_year_segments.json"
    with open(output_json, 'w') as f:
        json.dump(final_segments, f, indent=2)
        
    print(f"\n[RUNNER] YEAR COMPLETE! Collected {len(final_segments)} total segments.")
    send_telegram_alert(f"✅ YEAR COMPLETE! Harvested {len(final_segments)} total segments.")
    inject_prompt(f"System: FULL YEAR PROCESSING COMPLETE! Harvested {len(final_segments)} total segments.")

if __name__ == "__main__":
    main()
