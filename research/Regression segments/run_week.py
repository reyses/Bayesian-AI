import os
import subprocess
import time
import json
import sys

# Dynamically locate repository root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_root)
try:
    from telegram_mcp import send_telegram_alert, inject_prompt
except ImportError:
    def send_telegram_alert(msg): print(msg)
    def inject_prompt(msg): print(msg)

def main():
    days = ['2026_03_02', '2026_03_03', '2026_03_04', '2026_03_05', '2026_03_06']
    all_segments = []
    
    for day in days:
        stage2_file = f"artifacts/stage2_segments_{day}.json"
        
        if os.path.exists(stage2_file):
            print(f"[RUNNER] Skipping {day}, already fully processed.")
            with open(stage2_file, 'r') as f:
                all_segments.extend(json.load(f))
            continue
            
        print(f"\n========================================")
        print(f"[RUNNER] Starting STAGE 1 for {day}")
        print(f"========================================")
        t0 = time.time()
        
        stage1_file = f"artifacts/stage1_segments_{day}.json"
        if not os.path.exists(stage1_file):
            subprocess.run([sys.executable, "research/Regression segments/stage1_speed_pass.py", "--day", day, "--hours", "24"], check=True)
            
        print(f"[RUNNER] Stage 1 finished in {(time.time() - t0)/60:.1f} mins.")
        
        print(f"\n========================================")
        print(f"[RUNNER] Starting STAGE 2 for {day}")
        print(f"========================================")
        t1 = time.time()
        subprocess.run([sys.executable, "research/Regression segments/stage2_parallel_chaos.py", "--day", day], check=True)
        print(f"[RUNNER] Stage 2 finished in {(time.time() - t1)/60:.1f} mins.")
        
        with open(stage2_file, 'r') as f:
            all_segments.extend(json.load(f))
            
        send_telegram_alert(f"✅ Finished processing {day}!")
        inject_prompt(f"System: Automatically waking up! Just finished processing {day}.")
            
    # Combine all
    with open("artifacts/stage2_week_segments.json", "w") as f:
        json.dump(all_segments, f, indent=2)
        
    print(f"\n[RUNNER] All 5 days processed and combined into stage2_week_segments.json!")
    print(f"[RUNNER] Total segments collected: {len(all_segments)}")

if __name__ == "__main__":
    main()
