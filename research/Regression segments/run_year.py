import os
import subprocess
import time
import json
import sys
import glob

sys.path.append(r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI")
try:
    from telegram_mcp import send_telegram_alert, inject_prompt
except ImportError:
    def send_telegram_alert(msg): print(msg)
    def inject_prompt(msg): print(msg)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=None, help="Format YYYY_MM_DD to skip directly to a specific date")
    args = parser.parse_args()

    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    l0_dir = os.path.join(atlas_root, 'FEATURES_5s_v2', 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    if args.start_date:
        days = [d for d in days if d >= args.start_date]
    
    print(f"[RUNNER] Found {len(days)} total days of ATLAS data to process.")
    
    import threading
    import tkinter as tk
    from tkinter import ttk

    class ProgressBarWindow:
        def __init__(self, total_days):
            self.total_days = total_days
            self.current_day = 0
            self.root = None
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        def _run(self):
            self.root = tk.Tk()
            self.root.title("Pipeline Progress")
            self.root.geometry("400x120")
            self.root.attributes('-topmost', True)
            
            self.label = tk.Label(self.root, text="Starting...", font=("Segoe UI", 12))
            self.label.pack(pady=10)
            
            self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
            self.progress.pack(pady=5)
            
            self.status = tk.Label(self.root, text="Initializing...", font=("Segoe UI", 9), fg="gray")
            self.status.pack(pady=5)
            
            self._update_gui()
            self.root.mainloop()

        def _update_gui(self):
            if self.root:
                pct = (self.current_day / max(1, self.total_days)) * 100
                self.progress["value"] = pct
                self.label.config(text=f"Completed {self.current_day} / {self.total_days} days ({pct:.1f}%)")
                self.root.after(1000, self._update_gui)

        def update_progress(self, current_day, status_text=""):
            self.current_day = current_day
            if self.root and hasattr(self, 'status'):
                try:
                    self.status.config(text=status_text)
                except:
                    pass

    pbar_win = ProgressBarWindow(len(days))
    
    all_segments = []
    
    for i, day in enumerate(days):
        pbar_win.update_progress(i, status_text=f"Processing {day}...")
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
        
        if not os.path.exists(stage1_file):
            print(f"[RUNNER] Stage 1 produced no segments (likely a holiday/empty day). Skipping {day}.")
            continue
        
        print(f"\n========================================")
        print(f"[RUNNER] Starting STAGE 2 for {day}")
        print(f"========================================")
        t1 = time.time()
        subprocess.run([sys.executable, "research/Regression segments/stage2_parallel_chaos.py", "--day", day], check=True)
        print(f"[RUNNER] Stage 2 finished in {(time.time() - t1)/60:.1f} mins.")
        
        with open(stage2_file, 'r') as f:
            all_segments.extend(json.load(f))
            
        send_telegram_alert(f"✅ Finished processing {day}!")
        time.sleep(3)  # Buffer to allow OS to reclaim PyTorch/CUDA shared memory handles
        # To avoid spamming the AI with wakeups every 20 minutes for a year,
        # we will only inject prompt every 30 days or at the very end.
        if (days.index(day) + 1) % 30 == 0:
            inject_prompt(f"System: Automatically waking up! Just finished processing month milestone {day}.")
            
    # Combine all
    with open("artifacts/stage2_year_segments.json", "w") as f:
        json.dump(all_segments, f, indent=2)
        
    print(f"\n[RUNNER] All {len(days)} days processed and combined into stage2_year_segments.json!")
    print(f"[RUNNER] Total segments collected: {len(all_segments)}")
    inject_prompt(f"System: FULL YEAR PROCESSING COMPLETE! Harvested {len(all_segments)} total segments.")
    pbar_win.update_progress(len(days), status_text="Finished!")

if __name__ == "__main__":
    main()
