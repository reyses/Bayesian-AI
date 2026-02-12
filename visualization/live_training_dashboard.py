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
import numpy as np
import threading
import sys
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.style as mplstyle

try:
    mplstyle.use('fast')
except:
    pass


class Tooltip:
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

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
                       background="#333333", foreground="#ffffff", relief='solid', borderwidth=1,
                       font=("Arial", "9", "normal"))
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
        self.root.geometry("1400x850")
        self.root.configure(bg="#1e1e1e")

        # Data Path
        self.json_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'training_progress.json')
        self.training_dir = os.path.join(os.path.dirname(__file__), '..', 'training')
        self.pause_file = os.path.join(self.training_dir, 'PAUSE')
        self.stop_file = os.path.join(self.training_dir, 'STOP')
        self.last_update = 0
        self.is_running = True
        self.remote_status = "RUNNING"

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#1e1e1e", foreground="white", fieldbackground="#2b2b2b")
        style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Arial", 10))
        style.configure("TButton", background="#4080ff", foreground="white", borderwidth=0)
        style.map("TButton", background=[('active', '#5090ff')])
        style.configure("Header.TLabel", font=("Arial", 13, "bold"), background="#1e1e1e")
        style.configure("BigMetric.TLabel", font=("Consolas", 20, "bold"), background="#1e1e1e")
        style.configure("Metric.TLabel", font=("Consolas", 11), background="#1e1e1e")
        style.configure("SmallMetric.TLabel", font=("Consolas", 9), background="#1e1e1e", foreground="#aaaaaa")
        style.configure("Card.TFrame", background="#2b2b2b")

        self.create_layout()

        # Start Polling Thread
        self.poll_thread = threading.Thread(target=self.poll_data, daemon=True)
        self.poll_thread.start()

        # Update GUI loop
        self.root.after(1000, self.update_gui)

    def create_layout(self):
        """
        Layout:
        Row 0: Top metric cards (P&L, Win Rate, Sharpe, Trades) — spans full width
        Row 1 col 0: Cumulative P&L chart (left)
        Row 1 col 1: Per-day P&L bar chart (right)
        Row 2 col 0: Day-by-day table (left)
        Row 2 col 1: Controls + log (right)
        """
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=3)
        self.root.rowconfigure(2, weight=2)

        # === ROW 0: Top metric cards ===
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))
        for i in range(8):
            top_frame.columnconfigure(i, weight=1)

        # Card: Progress
        self._card_progress = self._make_card(top_frame, 0, "PROGRESS", "Day -- / --", "-- | ETA: --")
        # Card: Total P&L
        self._card_pnl = self._make_card(top_frame, 1, "TOTAL P&L", "$0.00", "Today: $0.00")
        # Card: Win Rate
        self._card_wr = self._make_card(top_frame, 2, "WIN RATE", "0.0%", "Today: 0.0%")
        # Card: Sharpe
        self._card_sharpe = self._make_card(top_frame, 3, "SHARPE", "0.00", "Today: 0.00")
        # Card: Trades
        self._card_trades = self._make_card(top_frame, 4, "TRADES", "0", "Today: 0")
        # Card: States
        self._card_states = self._make_card(top_frame, 5, "STATES", "0", "High Conf: 0")
        # Card: Drawdown
        self._card_dd = self._make_card(top_frame, 6, "MAX DD", "$0.00", "Avg Dur: --")
        # Card: Best Params
        self._card_params = self._make_card(top_frame, 7, "BEST PARAMS", "--", "TP/SL: --")

        # === ROW 1 LEFT: Cumulative P&L Chart ===
        self.frame_pnl = ttk.Frame(self.root, padding=5)
        self.frame_pnl.grid(row=1, column=0, sticky="nsew", padx=(5, 2))

        self.fig_pnl = Figure(figsize=(6, 3), dpi=100, facecolor="#1e1e1e")
        self.ax_pnl = self.fig_pnl.add_subplot(111)
        self._style_axis(self.ax_pnl, "Cumulative P&L (Best Iteration)")
        self.canvas_pnl = FigureCanvasTkAgg(self.fig_pnl, self.frame_pnl)
        self.canvas_pnl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === ROW 1 RIGHT: Per-Day P&L Bar Chart ===
        self.frame_daily = ttk.Frame(self.root, padding=5)
        self.frame_daily.grid(row=1, column=1, sticky="nsew", padx=(2, 5))

        self.fig_daily = Figure(figsize=(6, 3), dpi=100, facecolor="#1e1e1e")
        self.ax_daily = self.fig_daily.add_subplot(111)
        self._style_axis(self.ax_daily, "Daily P&L")
        self.canvas_daily = FigureCanvasTkAgg(self.fig_daily, self.frame_daily)
        self.canvas_daily.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === ROW 2 LEFT: Day-by-day table ===
        table_frame = ttk.Frame(self.root, padding=5)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=(5, 2))

        ttk.Label(table_frame, text="Day-by-Day Results", style="Header.TLabel").pack(anchor="w")

        cols = ("Day", "Date", "Trades", "WR%", "P&L", "Sharpe")
        self.day_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.day_tree.heading(c, text=c)
            self.day_tree.column(c, width=80, anchor="center")
        self.day_tree.column("Date", width=100)
        self.day_tree.pack(fill=tk.BOTH, expand=True)

        # Tag colors for table rows
        self.day_tree.tag_configure('profit', foreground='#00ff00')
        self.day_tree.tag_configure('loss', foreground='#ff4444')

        # === ROW 2 RIGHT: Controls + Log ===
        self.frame_controls = ttk.Frame(self.root, padding=10)
        self.frame_controls.grid(row=2, column=1, sticky="nsew", padx=(2, 5))

        # Status
        self.lbl_status = ttk.Label(self.frame_controls, text="Status: RUNNING",
                                     style="Header.TLabel", foreground="#00ff00")
        self.lbl_status.pack(anchor="w", pady=2)

        btn_frame = ttk.Frame(self.frame_controls)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_pause = ttk.Button(btn_frame, text="⏸️ Pause", command=self.pause_training)
        self.btn_pause.pack(side="left", padx=5)
        Tooltip(self.btn_pause, "Pause training by creating a PAUSE signal file")

        self.btn_resume = ttk.Button(btn_frame, text="▶️ Resume", command=self.resume_training)
        self.btn_resume.pack(side="left", padx=5)
        Tooltip(self.btn_resume, "Resume training by removing the PAUSE signal file")

        self.btn_stop = ttk.Button(btn_frame, text="🛑 Stop", command=self.stop_training)
        self.btn_stop.pack(side="left", padx=5)
        Tooltip(self.btn_stop, "Stop training gracefully by creating a STOP signal file")

        self.btn_export = ttk.Button(btn_frame, text="📸 Export PNG", command=self.export_chart)
        self.btn_export.pack(side="left", padx=5)

        ttk.Separator(self.frame_controls, orient='horizontal').pack(fill='x', pady=5)

        self.txt_log = tk.Text(self.frame_controls, height=8, bg="#111111", fg="#cccccc",
                               borderwidth=0, font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.log("Dashboard initialized. Waiting for training data...")

    def _make_card(self, parent, col, title, value_text, sub_text):
        """Create a metric card widget. Returns (value_label, sub_label)."""
        frame = tk.Frame(parent, bg="#2b2b2b", padx=8, pady=6, highlightthickness=1,
                        highlightbackground="#3b3b3b")
        frame.grid(row=0, column=col, sticky="nsew", padx=3, pady=3)

        tk.Label(frame, text=title, font=("Arial", 8, "bold"), fg="#888888",
                bg="#2b2b2b").pack(anchor="w")
        lbl_val = tk.Label(frame, text=value_text, font=("Consolas", 16, "bold"),
                          fg="white", bg="#2b2b2b")
        lbl_val.pack(anchor="w")
        lbl_sub = tk.Label(frame, text=sub_text, font=("Consolas", 9),
                          fg="#888888", bg="#2b2b2b")
        lbl_sub.pack(anchor="w")
        return lbl_val, lbl_sub

    def _style_axis(self, ax, title):
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors='#888888', labelsize=8)
        ax.set_title(title, color="white", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color('#3b3b3b')

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

                if os.path.exists(self.stop_file):
                    self.remote_status = "STOPPED"
                elif os.path.exists(self.pause_file):
                    self.remote_status = "PAUSED"
                else:
                    self.remote_status = "RUNNING"
            except Exception as e:
                print(f"Polling error: {e}")
            time.sleep(1)

    def update_gui(self):
        if hasattr(self, 'new_data_event') and self.new_data_event:
            self.new_data_event = False
            self.refresh_dashboard()

        if self.remote_status == "STOPPED":
            self.lbl_status.config(text="Status: STOPPED", foreground="red")
        elif self.remote_status == "PAUSED":
            self.lbl_status.config(text="Status: PAUSED", foreground="orange")
        else:
            self.lbl_status.config(text="Status: RUNNING", foreground="#00ff00")

        self.root.after(1000, self.update_gui)

    def pause_training(self):
        try:
            with open(self.pause_file, 'w') as f:
                f.write('PAUSE')
            self.log("Signal sent: PAUSE")
        except Exception as e:
            self.log(f"Error pausing: {e}")

    def resume_training(self):
        try:
            if os.path.exists(self.pause_file):
                os.remove(self.pause_file)
            self.log("Signal sent: RESUME")
        except Exception as e:
            self.log(f"Error resuming: {e}")

    def stop_training(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop training?"):
            try:
                with open(self.stop_file, 'w') as f:
                    f.write('STOP')
                self.log("Signal sent: STOP")
            except Exception as e:
                self.log(f"Error stopping: {e}")

    def export_chart(self):
        try:
            filename = f"dashboard_export_{int(time.time())}.png"
            self.fig_pnl.savefig(filename, facecolor="#1e1e1e")
            self.log(f"Chart exported to {filename}")
            messagebox.showinfo("Export", f"Chart saved to {filename}")
        except Exception as e:
            self.log(f"Error exporting: {e}")

    def refresh_dashboard(self):
        d = self.data

        # === TOP CARDS ===
        iter_current = d.get('iteration', 0)
        iter_total = d.get('total_iterations', 1)
        elapsed = int(d.get('elapsed_seconds', 0))
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        eta_str = "--"
        if iter_current > 0:
            eta_seconds = int((elapsed / iter_current) * (iter_total - iter_current))
            eta_str = str(datetime.timedelta(seconds=eta_seconds))
        current_date = d.get('current_date', '')

        self._card_progress[0].config(text=f"Day {iter_current} / {iter_total}")
        self._card_progress[1].config(text=f"{current_date} | ETA: {eta_str}")

        # P&L
        total_pnl = d.get('total_pnl', 0.0)
        today_pnl = d.get('today_pnl', 0.0)
        pnl_color = "#00ff00" if total_pnl >= 0 else "#ff4444"
        self._card_pnl[0].config(text=f"${total_pnl:,.2f}", fg=pnl_color)
        today_pnl_color = "#00ff00" if today_pnl >= 0 else "#ff4444"
        self._card_pnl[1].config(text=f"Today: ${today_pnl:,.2f}", fg=today_pnl_color)

        # Win Rate
        cum_wr = d.get('cumulative_win_rate', 0.0)
        today_wr = d.get('today_win_rate', 0.0) * 100
        wr_color = "#00ff00" if cum_wr >= 50 else "#ff4444"
        self._card_wr[0].config(text=f"{cum_wr:.1f}%", fg=wr_color)
        self._card_wr[1].config(text=f"Today: {today_wr:.1f}%")

        # Sharpe
        cum_sharpe = d.get('cumulative_sharpe', 0.0)
        today_sharpe = d.get('today_sharpe', 0.0)
        sharpe_color = "#00ff00" if cum_sharpe > 0 else "#ff4444"
        self._card_sharpe[0].config(text=f"{cum_sharpe:.2f}", fg=sharpe_color)
        self._card_sharpe[1].config(text=f"Today: {today_sharpe:.2f}")

        # Trades
        total_trades = d.get('total_trades', 0)
        today_trades = d.get('today_trades', 0)
        self._card_trades[0].config(text=f"{total_trades}")
        self._card_trades[1].config(text=f"Today: {today_trades}")

        # States
        states = d.get('states_learned', 0)
        high_conf = d.get('high_confidence_states', 0)
        self._card_states[0].config(text=f"{states:,}")
        self._card_states[1].config(text=f"High Conf: {high_conf}")

        # Drawdown + Avg Duration
        max_dd = d.get('max_drawdown', 0.0)
        avg_dur = d.get('avg_duration', 0.0)
        dur_str = f"{avg_dur:.0f}s" if avg_dur < 120 else f"{avg_dur/60:.1f}m"
        self._card_dd[0].config(text=f"${max_dd:,.2f}", fg="#ff4444" if max_dd > 0 else "white")
        self._card_dd[1].config(text=f"Avg Dur: {dur_str}")

        # Best Params
        bp = d.get('best_params', {})
        if bp:
            tp_sl = f"TP:{bp.get('TP','?')} SL:{bp.get('SL','?')}"
            thresh = f"Thr:{bp.get('Threshold','?')} {bp.get('MaxHold','')}"
            self._card_params[0].config(text=tp_sl, font=("Consolas", 11, "bold"))
            self._card_params[1].config(text=thresh)
        else:
            self._card_params[0].config(text="--")
            self._card_params[1].config(text="Waiting...")

        # === CHARTS ===
        self.update_charts(d)

        # === DAY TABLE ===
        self.update_day_table(d)

        self.log(f"Day {iter_current}/{iter_total} | {current_date} | "
                 f"P&L: ${total_pnl:,.2f} | WR: {cum_wr:.1f}% | Trades: {total_trades}")

    def update_charts(self, data):
        # 1. Cumulative P&L
        trades = data.get('trades', [])
        if trades:
            pnls = [t.get('pnl', 0) for t in trades]
            cum_pnl = np.cumsum(pnls)

            self.ax_pnl.clear()
            self._style_axis(self.ax_pnl, "Cumulative P&L (Best Iteration)")

            x = range(len(cum_pnl))
            self.ax_pnl.plot(x, cum_pnl, color='#00ff00', linewidth=1.5)
            self.ax_pnl.axhline(0, color='#555555', linestyle='--', linewidth=0.5)
            self.ax_pnl.fill_between(x, cum_pnl, 0,
                                      where=(cum_pnl >= 0), facecolor='#00ff00', alpha=0.15)
            self.ax_pnl.fill_between(x, cum_pnl, 0,
                                      where=(cum_pnl < 0), facecolor='#ff4444', alpha=0.15)
            self.ax_pnl.set_xlabel("Trade #", color="#888888", fontsize=8)
            self.ax_pnl.set_ylabel("P&L ($)", color="#888888", fontsize=8)
            self.fig_pnl.tight_layout()
            self.canvas_pnl.draw()

        # 2. Daily P&L bar chart
        day_summaries = data.get('day_summaries', [])
        if day_summaries:
            days = [s.get('day', 0) for s in day_summaries]
            pnls = [s.get('pnl', 0) for s in day_summaries]
            colors = ['#00ff00' if p >= 0 else '#ff4444' for p in pnls]

            self.ax_daily.clear()
            self._style_axis(self.ax_daily, "Daily P&L")
            self.ax_daily.bar(days, pnls, color=colors, alpha=0.8, width=0.7)
            self.ax_daily.axhline(0, color='#555555', linestyle='--', linewidth=0.5)
            self.ax_daily.set_xlabel("Day", color="#888888", fontsize=8)
            self.ax_daily.set_ylabel("P&L ($)", color="#888888", fontsize=8)
            self.fig_daily.tight_layout()
            self.canvas_daily.draw()

    def update_day_table(self, data):
        # Clear existing rows
        for item in self.day_tree.get_children():
            self.day_tree.delete(item)

        day_summaries = data.get('day_summaries', [])
        for s in day_summaries:
            pnl = s.get('pnl', 0)
            wr = s.get('win_rate', 0) * 100
            tag = 'profit' if pnl >= 0 else 'loss'
            self.day_tree.insert('', 'end', values=(
                s.get('day', ''),
                s.get('date', ''),
                s.get('trades', 0),
                f"{wr:.1f}",
                f"${pnl:,.2f}",
                f"{s.get('sharpe', 0):.2f}",
            ), tags=(tag,))

        # Auto-scroll to bottom
        children = self.day_tree.get_children()
        if children:
            self.day_tree.see(children[-1])


if __name__ == "__main__":
    if not os.environ.get('DISPLAY', '') and sys.platform != 'win32':
        print("No display found. Dashboard requires a GUI environment.")
        sys.exit(1)

    root = tk.Tk()
    app = LiveDashboard(root)
    root.mainloop()
