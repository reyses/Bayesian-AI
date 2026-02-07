"""
Bayesian-AI - Live Training Dashboard
Real-time monitoring of training progress.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import time
import pandas as pd
import threading
import sys
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.style as mplstyle

# Try to use a faster style if available, else default
try:
    mplstyle.use('fast')
except:
    pass

class Tooltip:
    """
    It creates a tooltip for a given widget as the mouse goes on it.
    """
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

class LiveDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian-AI Live Training Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2b2b2b")

        # Data Path
        self.json_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'training_progress.json')
        self.training_dir = os.path.join(os.path.dirname(__file__), '..', 'training')
        self.last_update = 0
        self.is_running = True

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#2b2b2b", foreground="white", fieldbackground="#3b3b3b")
        style.configure("TLabel", background="#2b2b2b", foreground="white", font=("Arial", 10))
        style.configure("TButton", background="#4080ff", foreground="white", borderwidth=0)
        style.map("TButton", background=[('active', '#5090ff')])
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("Metric.TLabel", font=("Consolas", 12))

        # Main Layout
        self.create_layout()
        
        # Start Polling Thread
        self.poll_thread = threading.Thread(target=self.poll_data, daemon=True)
        self.poll_thread.start()

        # Update GUI loop
        self.root.after(1000, self.update_gui)

    def create_layout(self):
        # 2x2 Grid
        self.root.columnconfigure(0, weight=6)
        self.root.columnconfigure(1, weight=4)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Panel 1: Chart (Top Left)
        self.frame_chart = ttk.Frame(self.root, padding=5)
        self.frame_chart.grid(row=0, column=0, sticky="nsew")
        
        self.fig_chart = Figure(figsize=(5, 4), dpi=100, facecolor="#2b2b2b")
        self.ax_chart = self.fig_chart.add_subplot(111)
        self.ax_chart.set_facecolor("#2b2b2b")
        self.ax_chart.tick_params(colors='white')
        self.ax_chart.xaxis.label.set_color('white')
        self.ax_chart.yaxis.label.set_color('white')
        self.ax_chart.set_title("Live Market Chart (15m)", color="white")
        
        self.canvas_chart = FigureCanvasTkAgg(self.fig_chart, self.frame_chart)
        self.canvas_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Panel 2: Metrics (Top Right)
        self.frame_metrics = ttk.Frame(self.root, padding=10)
        self.frame_metrics.grid(row=0, column=1, sticky="nsew")
        
        ttk.Label(self.frame_metrics, text="Training Metrics", style="Header.TLabel").pack(anchor="w", pady=5)
        
        self.lbl_iter = ttk.Label(self.frame_metrics, text="Iteration: -- / --", style="Metric.TLabel")
        self.lbl_iter.pack(anchor="w")
        
        self.lbl_time = ttk.Label(self.frame_metrics, text="Elapsed: --:--:--", style="Metric.TLabel")
        self.lbl_time.pack(anchor="w")
        
        self.lbl_eta = ttk.Label(self.frame_metrics, text="ETA: --:--:--", style="Metric.TLabel")
        self.lbl_eta.pack(anchor="w")
        
        ttk.Separator(self.frame_metrics, orient='horizontal').pack(fill='x', pady=10)
        
        self.lbl_states = ttk.Label(self.frame_metrics, text="States Learned: 0", style="Metric.TLabel")
        self.lbl_states.pack(anchor="w")
        
        self.lbl_conf = ttk.Label(self.frame_metrics, text="High Conf States: 0", style="Metric.TLabel")
        self.lbl_conf.pack(anchor="w")
        
        self.lbl_trades = ttk.Label(self.frame_metrics, text="Total Trades: 0", style="Metric.TLabel")
        self.lbl_trades.pack(anchor="w")
        
        self.lbl_wr = ttk.Label(self.frame_metrics, text="Win Rate: 0.0%", style="Metric.TLabel")
        self.lbl_wr.pack(anchor="w")

        # Panel 3: P&L Chart (Bottom Left)
        self.frame_pnl = ttk.Frame(self.root, padding=5)
        self.frame_pnl.grid(row=1, column=0, sticky="nsew")
        
        self.fig_pnl = Figure(figsize=(5, 4), dpi=100, facecolor="#2b2b2b")
        self.ax_pnl = self.fig_pnl.add_subplot(111)
        self.ax_pnl.set_facecolor("#2b2b2b")
        self.ax_pnl.tick_params(colors='white')
        self.ax_pnl.set_title("Cumulative P&L", color="white")
        
        self.canvas_pnl = FigureCanvasTkAgg(self.fig_pnl, self.frame_pnl)
        self.canvas_pnl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Panel 4: Controls & Stats (Bottom Right)
        self.frame_controls = ttk.Frame(self.root, padding=10)
        self.frame_controls.grid(row=1, column=1, sticky="nsew")
        
        ttk.Label(self.frame_controls, text="Controls", style="Header.TLabel").pack(anchor="w", pady=5)
        
        # Status Indicator
        self.lbl_status = ttk.Label(self.frame_controls, text="Status: RUNNING", style="Header.TLabel", foreground="#00ff00")
        self.lbl_status.pack(anchor="w", pady=2)

        btn_frame = ttk.Frame(self.frame_controls)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_pause = ttk.Button(btn_frame, text="Pause", command=self.pause_training)
        self.btn_pause.pack(side="left", padx=5)
        Tooltip(self.btn_pause, "Pause training by creating a PAUSE signal file")

        self.btn_resume = ttk.Button(btn_frame, text="Resume", command=self.resume_training)
        self.btn_resume.pack(side="left", padx=5)
        Tooltip(self.btn_resume, "Resume training by removing the PAUSE signal file")

        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self.stop_training)
        self.btn_stop.pack(side="left", padx=5)
        Tooltip(self.btn_stop, "Stop training gracefully by creating a STOP signal file")

        self.btn_export = ttk.Button(btn_frame, text="Export PNG", command=self.export_chart)
        self.btn_export.pack(side="left", padx=5)
        Tooltip(self.btn_export, "Save the current chart as a PNG image")
        
        ttk.Separator(self.frame_controls, orient='horizontal').pack(fill='x', pady=10)
        
        self.txt_log = tk.Text(self.frame_controls, height=10, bg="#1e1e1e", fg="white", borderwidth=0)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.log("Dashboard initialized.")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.txt_log.see(tk.END)

    def poll_data(self):
        while self.is_running:
            try:
                if os.path.exists(self.json_path):
                    mtime = os.path.getmtime(self.json_path)
                    if mtime > self.last_update:
                        with open(self.json_path, 'r') as f:
                            self.data = json.load(f)
                        self.last_update = mtime
                        self.new_data_event = True
            except Exception as e:
                print(f"Polling error: {e}")
            
            time.sleep(1)

    def update_gui(self):
        if hasattr(self, 'new_data_event') and self.new_data_event:
            self.new_data_event = False
            self.refresh_dashboard()
        
        # Check status files
        pause_file = os.path.join(self.training_dir, 'PAUSE')
        stop_file = os.path.join(self.training_dir, 'STOP')

        if os.path.exists(stop_file):
            self.lbl_status.config(text="Status: STOPPED", foreground="red")
            self.btn_pause.state(['disabled'])
            self.btn_resume.state(['disabled'])
            self.btn_stop.state(['disabled'])
        elif os.path.exists(pause_file):
            self.lbl_status.config(text="Status: PAUSED", foreground="orange")
            self.btn_pause.state(['disabled'])
            self.btn_resume.state(['!disabled'])
        else:
            self.lbl_status.config(text="Status: RUNNING", foreground="#00ff00")
            self.btn_pause.state(['!disabled'])
            self.btn_resume.state(['disabled'])

        self.root.after(1000, self.update_gui)

    def pause_training(self):
        try:
            with open(os.path.join(self.training_dir, 'PAUSE'), 'w') as f:
                f.write('PAUSE')
            self.log("Signal sent: PAUSE")
        except Exception as e:
            self.log(f"Error pausing: {e}")

    def resume_training(self):
        try:
            p = os.path.join(self.training_dir, 'PAUSE')
            if os.path.exists(p):
                os.remove(p)
            self.log("Signal sent: RESUME")
        except Exception as e:
            self.log(f"Error resuming: {e}")

    def stop_training(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop training?"):
            try:
                with open(os.path.join(self.training_dir, 'STOP'), 'w') as f:
                    f.write('STOP')
                self.log("Signal sent: STOP")
            except Exception as e:
                self.log(f"Error stopping: {e}")

    def export_chart(self):
        try:
            filename = f"dashboard_export_{int(time.time())}.png"
            self.fig_pnl.savefig(filename)
            self.log(f"Chart exported to {filename}")
            messagebox.showinfo("Export", f"Chart saved to {filename}")
        except Exception as e:
            self.log(f"Error exporting: {e}")

    def refresh_dashboard(self):
        d = self.data
        
        # Metrics
        self.lbl_iter.config(text=f"Iteration: {d.get('iteration', '?')} / {d.get('total_iterations', '?')}")
        elapsed = int(d.get('elapsed_seconds', 0))
        self.lbl_time.config(text=f"Elapsed: {datetime.timedelta(seconds=elapsed)}")
        
        iter_current = d.get('iteration', 0)
        iter_total = d.get('total_iterations', 1)
        if iter_current > 0:
            eta_seconds = int((elapsed / iter_current) * (iter_total - iter_current))
            self.lbl_eta.config(text=f"ETA: {datetime.timedelta(seconds=eta_seconds)}")
        
        self.lbl_states.config(text=f"States Learned: {d.get('states_learned', 0)}")
        self.lbl_conf.config(text=f"High Conf States: {d.get('high_confidence_states', 0)}")
        
        trades = d.get('trades', [])
        self.lbl_trades.config(text=f"Total Trades: {len(trades)}")
        
        wins = sum(1 for t in trades if t.get('result') == 'WIN')
        total = len(trades)
        wr = (wins/total*100) if total > 0 else 0
        self.lbl_wr.config(text=f"Win Rate: {wr:.1f}%")
        
        # Charts
        self.update_charts(d)
        
        self.log(f"Updated: Iter {iter_current}")

    def update_charts(self, data):
        # 1. P&L Chart
        trades = data.get('trades', [])
        if trades:
            df = pd.DataFrame(trades)
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            self.ax_pnl.clear()
            self.ax_pnl.set_facecolor("#2b2b2b")
            self.ax_pnl.plot(range(len(df)), df['cumulative_pnl'], color='#00ff00')
            self.ax_pnl.axhline(0, color='gray', linestyle='--', alpha=0.5)
            self.ax_pnl.set_title("Cumulative P&L", color="white")
            self.ax_pnl.tick_params(colors='white')
            
            # Fill areas
            self.ax_pnl.fill_between(range(len(df)), df['cumulative_pnl'], 0, where=(df['cumulative_pnl'] >= 0), facecolor='green', alpha=0.3)
            self.ax_pnl.fill_between(range(len(df)), df['cumulative_pnl'], 0, where=(df['cumulative_pnl'] < 0), facecolor='red', alpha=0.3)
            
            self.canvas_pnl.draw()

        # 2. Market Chart
        candles = data.get('recent_candles', [])
        if candles:
            df = pd.DataFrame(candles)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.ax_chart.clear()
            self.ax_chart.set_facecolor("#2b2b2b")
            self.ax_chart.set_title("Live Market Chart (15m)", color="white")
            self.ax_chart.tick_params(colors='white')
            
            # Simple line chart for now (Candlestick requires mplfinance or complex plotting)
            self.ax_chart.plot(df['timestamp'], df['close'], color='cyan', label='Close')
            
            # Overlay trades if timestamps match? (Complex logic omitted for MVP)
            
            self.canvas_chart.draw()

if __name__ == "__main__":
    if not os.environ.get('DISPLAY', '') and sys.platform != 'win32':
        print("No display found. Dashboard requires a GUI environment.")
        sys.exit(1)
        
    root = tk.Tk()
    app = LiveDashboard(root)
    root.mainloop()
