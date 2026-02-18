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
        cols = ("ID", "Count", "PnL", "Status")
        self.tree_ranks = ttk.Treeview(right_pane, columns=cols, show='headings', height=15)
        for col in cols:
            self.tree_ranks.heading(col, text=col)
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

        # Clear existing arrows
        for artist in self.ax.patches:
            artist.remove()

        z_vals = [d.get('z', 0) for d in self.templates.values()]
        m_vals = [d.get('mom', 0) for d in self.templates.values()]
        p_vals = [d.get('pnl', 0) for d in self.templates.values()]

        # Update scatter data instead of clearing and re-plotting
        offsets = np.c_[z_vals, m_vals]
        self.scatter.set_offsets(offsets)
        self.scatter.set_array(np.array(p_vals))

        # Draw Navigation Arrows (Transitions > 50%)
        # Note: We need full transition map here.
        # Since templates only stores scalar data in msg, we need to pass transition info
        # Let's assume TEMPLATE_UPDATE might contain 'transitions' or we fetch from somewhere else.
        # Ideally, we should receive transition updates.

        # Currently the dashboard message is simple dict.
        # Let's assume orchestrator passes 'transitions' dict in TEMPLATE_UPDATE msg
        # msg: {'id': 42, ..., 'transitions': {99: 0.8}}

        for tid, data in self.templates.items():
            trans = data.get('transitions', {})
            if not trans: continue

            x1, y1 = data.get('z', 0), data.get('mom', 0)

            for next_id, prob in trans.items():
                if prob > 0.5 and next_id in self.templates:
                    next_data = self.templates[next_id]
                    x2, y2 = next_data.get('z', 0), next_data.get('mom', 0)

                    # Draw Arrow
                    self.ax.arrow(x1, y1, (x2-x1)*0.9, (y2-y1)*0.9,
                                  head_width=0.1, head_length=0.1, fc='white', ec='white', alpha=0.6,
                                  length_includes_head=True)

        # Rescale axes to fit new data
        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def _update_leaderboard(self):
        # Clear
        for i in self.tree_ranks.get_children():
            self.tree_ranks.delete(i)

        # Sort by PnL
        sorted_templates = sorted(self.templates.values(), key=lambda x: x.get('pnl', 0), reverse=True)

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
