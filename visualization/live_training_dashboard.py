"""
Fractal Command Center (Live Dashboard)
Visualizes the Pattern Discovery, Clustering, and Fission process in real-time.
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import datetime
import numpy as np

class FractalDashboard:
    def __init__(self, root, queue):
        self.root = root
        self.queue = queue
        self.root.title("BAYESIAN-AI: FRACTAL COMMAND CENTER")
        self.root.geometry("1400x900")

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#00ff00", font=("Consolas", 10))
        style.configure("Header.TLabel", font=("Consolas", 14, "bold"), foreground="#ffffff")

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Data Stores
        self.templates = {} # ID -> {z, mom, pnl, count}
        self.fission_events = []
        self._transition_arrows = [] # Keep track of arrow artists

        # Constants
        self.ARROW_TRANSITION_PROB_THRESHOLD = 0.5
        self.ARROW_LENGTH_FACTOR = 0.9
        self.ARROW_HEAD_WIDTH = 0.1

        self.COL_ID = "ID"
        self.COL_COUNT = "Count"
        self.COL_PNL = "PnL"
        self.COL_STATUS = "Status"

        self._sort_col = self.COL_PNL
        self._sort_reverse = True

        self._setup_layout()
        self.root.after(100, self._process_queue)

    def _setup_layout(self):
        # Top: Metrics Bar
        self.top_bar = ttk.Frame(self.main_frame, height=50)
        self.top_bar.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_status = ttk.Label(self.top_bar, text="SYSTEM STATUS: INITIALIZING", style="Header.TLabel")
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_stats = ttk.Label(self.top_bar, text="TEMPLATES: 0 | FISSIONS: 0 | TOTAL PnL: $0")
        self.lbl_stats.pack(side=tk.RIGHT)

        # Split: Left (Visuals) | Right (Logs)
        split_frame = ttk.Frame(self.main_frame)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # LEFT: Physics Map
        left_pane = ttk.Frame(split_frame, width=900)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left_pane, text="PHYSICS MANIFOLD (Z-Score vs Momentum)", style="Header.TLabel").pack(anchor=tk.W)

        self.fig, self.ax = plt.subplots(figsize=(8, 6), facecolor='#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_xlabel("Z-Score (Sigma)")
        self.ax.set_ylabel("Momentum Strength")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=50, alpha=0.8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_pane)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # RIGHT: Fission Log & Leaderboard
        right_pane = ttk.Frame(split_frame, width=400)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        # Leaderboard
        ttk.Label(right_pane, text="TOP PERFORMING TEMPLATES", style="Header.TLabel").pack(anchor=tk.W)
        cols = (self.COL_ID, self.COL_COUNT, self.COL_PNL, self.COL_STATUS)
        self.tree_ranks = ttk.Treeview(right_pane, columns=cols, show='headings', height=15)
        for col in cols:
            self.tree_ranks.heading(col, text=col, command=lambda c=col: self._on_header_click(c))
            self.tree_ranks.column(col, width=80)
        self.tree_ranks.pack(fill=tk.X, pady=5)

        # Fission Log
        ttk.Label(right_pane, text="FISSION EVENTS & ALERTS", style="Header.TLabel").pack(anchor=tk.W, pady=(10,0))
        self.log_text = tk.Text(right_pane, height=15, bg="#000000", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(500, self._process_queue)

    def _handle_message(self, msg):
        msg_type = msg.get('type')

        if msg_type == 'TEMPLATE_UPDATE':
            # msg: {'id': 42, 'z': 2.5, 'mom': 5.0, 'pnl': 150.0, 'count': 50}
            tid = msg['id']
            self.templates[tid] = msg
            self._update_leaderboard()
            self._update_plot()
            self._log(f"UPDATED: Template {tid} | PnL: ${msg['pnl']:.0f}")

        elif msg_type == 'FISSION_EVENT':
            # msg: {'parent_id': 42, 'children_count': 3, 'reason': 'Variance'}
            txt = f"⚠️ FISSION: Template {msg['parent_id']} shattered into {msg['children_count']} subsets ({msg['reason']})"
            self._log(txt, error=True)
            self.fission_events.append(msg)
            self.lbl_stats.config(text=self._get_stats_str())

        elif msg_type == 'STATUS':
            self.lbl_status.config(text=f"SYSTEM STATUS: {msg['text']}")

    def _update_plot(self):
        # Refresh Scatter Plot
        if not self.templates: return

        # Clear existing arrows using managed list
        for artist in self._transition_arrows:
            try:
                artist.remove()
            except ValueError:
                pass # Already removed
        self._transition_arrows.clear()

        z_vals = np.array([d.get('z', 0) for d in self.templates.values()])
        m_vals = np.array([d.get('mom', 0) for d in self.templates.values()])

        # Clip momentum outliers using IQR so plot isn't dominated by extremes
        if len(m_vals) > 4:
            q1, q3 = np.percentile(m_vals, [25, 75])
            iqr = q3 - q1
            m_lo = q1 - 1.5 * iqr
            m_hi = q3 + 1.5 * iqr
            m_vals = np.clip(m_vals, m_lo, m_hi)
        # Color by Risk Score if available, else PnL (fallback to old behavior if risk not present)
        # Risk Score: 0 (Green) -> 1 (Red).
        # We need a colormap. 'RdYlGn_r' (Red-Yellow-Green reversed) maps 0 to Green, 1 to Red.

        c_vals = []
        use_risk_color = False
        for d in self.templates.values():
            if 'risk_score' in d:
                c_vals.append(d['risk_score'])
                use_risk_color = True
            else:
                c_vals.append(d.get('pnl', 0)) # Fallback to PnL

        offsets = np.c_[z_vals, m_vals]
        self.scatter.set_offsets(offsets)
        self.scatter.set_array(np.array(c_vals))

        if use_risk_color:
            self.scatter.set_cmap('RdYlGn_r')
            self.scatter.set_clim(0.0, 1.0)
        else:
            self.scatter.set_cmap('viridis')
            self.scatter.autoscale() # Reset clim for PnL

        # Draw Navigation Arrows
        for tid, data in self.templates.items():
            trans = data.get('transitions', {})
            if not trans: continue

            x1, y1 = data.get('z', 0), data.get('mom', 0)

            for next_id, prob in trans.items():
                if prob > self.ARROW_TRANSITION_PROB_THRESHOLD and next_id in self.templates:
                    next_data = self.templates[next_id]
                    x2, y2 = next_data.get('z', 0), next_data.get('mom', 0)

                    # Draw Arrow
                    arrow = self.ax.arrow(
                        x1, y1,
                        (x2-x1) * self.ARROW_LENGTH_FACTOR,
                        (y2-y1) * self.ARROW_LENGTH_FACTOR,
                        head_width=self.ARROW_HEAD_WIDTH,
                        head_length=self.ARROW_HEAD_WIDTH,
                        fc='white', ec='white', alpha=0.6,
                        length_includes_head=True
                    )
                    self._transition_arrows.append(arrow)

        # Rescale axes to fit new data
        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def _on_header_click(self, col):
        if col == self._sort_col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = True
        self._update_leaderboard()

    def _update_leaderboard(self):
        # Clear
        for i in self.tree_ranks.get_children():
            self.tree_ranks.delete(i)

        # Sort dynamically
        key_map = {self.COL_ID: "id", self.COL_COUNT: "count", self.COL_PNL: "pnl"}
        sort_key = key_map.get(self._sort_col, "pnl")

        sorted_templates = sorted(
            self.templates.values(),
            key=lambda x: x.get(sort_key, 0),
            reverse=self._sort_reverse
        )

        # Top 15
        for t in sorted_templates[:15]:
            self.tree_ranks.insert("", tk.END, values=(t['id'], t.get('count',0), f"${t.get('pnl',0):.0f}", "ACTIVE"))

        # Update Stats Label
        self.lbl_stats.config(text=self._get_stats_str())

    def _log(self, text, error=False):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = "🔴 " if error else "🟢 "
        self.log_text.insert(tk.END, f"[{timestamp}] {prefix}{text}\n")
        self.log_text.see(tk.END)

    def _get_stats_str(self):
        total_pnl = sum(t.get('pnl', 0) for t in self.templates.values())
        return f"TEMPLATES: {len(self.templates)} | FISSIONS: {len(self.fission_events)} | TOTAL PnL: ${total_pnl:.0f}"

def launch_dashboard(queue):
    root = tk.Tk()
    app = FractalDashboard(root, queue)
    root.mainloop()

if __name__ == '__main__':
    # Test Run
    import threading, time
    q = queue.Queue()

    def simulate_feed():
        time.sleep(2)
        q.put({'type': 'STATUS', 'text': 'SCANNING ATLAS...'})
        time.sleep(1)
        q.put({'type': 'TEMPLATE_UPDATE', 'id': 101, 'z': 2.1, 'mom': 4.5, 'pnl': 120, 'count': 45})
        time.sleep(0.5)
        q.put({'type': 'TEMPLATE_UPDATE', 'id': 102, 'z': -1.5, 'mom': -3.2, 'pnl': -50, 'count': 30})
        time.sleep(1)
        q.put({'type': 'FISSION_EVENT', 'parent_id': 101, 'children_count': 3, 'reason': 'Variance limit'})

    t = threading.Thread(target=simulate_feed, daemon=True)
    t.start()

    launch_dashboard(q)
