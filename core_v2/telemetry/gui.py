import os
import glob
import json
import time
import tkinter as tk
from tkinter import ttk

class MetricRow:
    def __init__(self, parent_frame, metric_id):
        self.metric_id = metric_id
        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="x", pady=5, padx=10)
        
        self.label = tk.Label(self.frame, text="Loading...", font=("Segoe UI", 10, "bold"))
        self.label.pack(anchor="w")
        
        self.progress = ttk.Progressbar(self.frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(fill="x", pady=2)
        
        self.status = tk.Label(self.frame, text="Initializing...", font=("Segoe UI", 9), fg="gray")
        self.status.pack(anchor="w")
        
        self.start_time = time.time()
        self.start_value = None
        self.last_update_ts = 0

    def update(self, data):
        current = data.get("current", 0)
        total = data.get("total", 1)
        label = data.get("label", self.metric_id)
        
        if self.start_value is None:
            self.start_value = current
            self.start_time = time.time()
            
        pct = (current / max(1, total)) * 100
        self.progress["value"] = pct
        self.label.config(text=label)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        processed = current - self.start_value
        
        if processed > 0 and elapsed > 0:
            velocity = processed / elapsed
            remaining = total - current
            eta_seconds = remaining / velocity
            
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds/3600:.1f} hours"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.1f} mins"
            else:
                eta_str = f"{eta_seconds:.0f} secs"
                
            vel_str = f"{velocity:.1f}/s"
        else:
            eta_str = "Calculating..."
            vel_str = "..."
            
        self.status.config(text=f"Progress: {current:.0f} / {total:.0f} ({pct:.1f}%) | Speed: {vel_str} | ETA: {eta_str}")
        self.last_update_ts = time.time()

    def destroy(self):
        self.frame.destroy()

class TelemetryViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bayesian-AI Telemetry")
        self.root.geometry("450x300")
        self.root.attributes('-topmost', True)
        self.root.configure(bg="#f0f0f0")
        
        header = tk.Label(self.root, text="Live Pipeline Progress", font=("Segoe UI", 12, "bold"), bg="#f0f0f0")
        header.pack(pady=10)
        
        self.canvas = tk.Canvas(self.root, bg="#f0f0f0")
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        
        self.rows_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        
        self.rows_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        
        self.metrics = {}  # metric_id -> MetricRow
        self._poll_json()

    def _poll_json(self):
        files = glob.glob("artifacts/telemetry_*.json")
        active_ids = set()
        
        current_time = time.time()
        
        for f in files:
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                metric_id = data.get("metric_id")
                if not metric_id:
                    continue
                    
                # Check heartbeat: if no update in 10 seconds, worker is dead/finished
                ts = data.get("timestamp", current_time)
                if current_time - ts > 10.0:
                    try:
                        os.remove(f)
                    except:
                        pass
                    continue
                    
                active_ids.add(metric_id)
                if metric_id not in self.metrics:
                    self.metrics[metric_id] = MetricRow(self.rows_frame, metric_id)
                    
                self.metrics[metric_id].update(data)
            except Exception:
                pass # Ignore temporary read locks
                
        # Remove stale or deleted metrics
        for metric_id in list(self.metrics.keys()):
            if metric_id not in active_ids:
                self.metrics[metric_id].destroy()
                del self.metrics[metric_id]
                
        self.root.after(500, self._poll_json)

if __name__ == "__main__":
    viewer = TelemetryViewer()
    viewer.root.mainloop()
