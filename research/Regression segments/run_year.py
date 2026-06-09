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
    
    # ---- PHASE 2: STAGE 2 (PER DAY) ----
    print(f"\n========================================")
    print(f"[PHASE 2] Resolving Chaos Blocks day-by-day...")
    print(f"========================================")
    
    final_segments = []
    
    # Process stage 2 sequentially day by day to save RAM
    for day in days:
        stage1_file = f"artifacts/stage1_segments_{day}.json"
        stage2_file = f"artifacts/stage2_segments_{day}.json"
        
        if not os.path.exists(stage1_file):
            continue
            
        if os.path.exists(stage2_file):
            with open(stage2_file, 'r') as f:
                final_segments.extend(json.load(f))
            continue
            
        print(f"\n[RUNNER] Starting STAGE 2 for {day}")
        t1 = time.time()
        subprocess.run([sys.executable, "research/Regression segments/stage2_parallel_chaos.py", "--day", day], check=True)
        print(f"[RUNNER] Stage 2 finished in {(time.time() - t1)/60:.1f} mins.")
        
        if os.path.exists(stage2_file):
            with open(stage2_file, 'r') as f:
                final_segments.extend(json.load(f))
                
        time.sleep(1) # Brief pause to allow OS memory reclamation
        
    final_segments.sort(key=lambda x: (x['day'], x['start_idx']))
    
    output_json = "artifacts/stage2_year_segments.json"
    with open(output_json, 'w') as f:
        json.dump(final_segments, f, indent=2)
        
    print(f"\n[RUNNER] YEAR COMPLETE! Collected {len(final_segments)} total segments.")
    send_telegram_alert(f"✅ YEAR COMPLETE! Harvested {len(final_segments)} total segments.")
    inject_prompt(f"System: FULL YEAR PROCESSING COMPLETE! Harvested {len(final_segments)} total segments.")

if __name__ == "__main__":
    main()
