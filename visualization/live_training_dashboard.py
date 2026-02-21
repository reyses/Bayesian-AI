"""
Fractal Command Center (Live Dashboard)
DMAIC Analyze Layer — real-time Pareto of profit gap across all phases.
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import datetime
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = '#1e1e1e'
FG_GREEN = '#00ff00'
FG_WHITE = '#ffffff'
FG_RED   = '#ff4444'
FG_AMBER = '#ffaa00'
FG_BLUE  = '#44aaff'
FG_GREY  = '#888888'

PARETO_COLORS = {
    'Missed':    '#ff4444',
    'Wrong Dir': '#ff8800',
    'Too Early': '#ffdd00',
    'Noise':     '#888888',
}

TOP_TEMPLATES_LIMIT = 50


class FractalDashboard:
    def __init__(self, root, queue):
        self.root  = root
        self.queue = queue
        self.root.title("BAYESIAN-AI: FRACTAL COMMAND CENTER")
        self.root.geometry("1600x950")
        self.root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame",       background=BG)
        style.configure("TLabel",       background=BG, foreground=FG_GREEN,  font=("Consolas", 10))
        style.configure("Header.TLabel",background=BG, foreground=FG_WHITE,  font=("Consolas", 13, "bold"))
        style.configure("Dim.TLabel",   background=BG, foreground=FG_GREY,   font=("Consolas", 9))
        style.configure("Accent.TLabel",background=BG, foreground=FG_AMBER,  font=("Consolas", 11, "bold"))
        style.configure("Good.TLabel",  background=BG, foreground=FG_GREEN,  font=("Consolas", 11, "bold"))
        style.configure("Bad.TLabel",   background=BG, foreground=FG_RED,    font=("Consolas", 11, "bold"))

        # ── Data stores ───────────────────────────────────────────────────────
        self.templates        = {}   # id -> {z, mom, pnl, count, ...}
        self.fission_events   = []
        self._transition_arrows = []

        # Sorting state for leaderboard
        self._sort_col = "PnL"
        self._sort_reverse = True

        # Oracle attribution
        self.attribution = {
            'ideal': 0.0, 'actual': 0.0,
            'missed': 0.0, 'wrong_dir': 0.0,
            'too_early': 0.0, 'noise': 0.0,
        }

        self._running = True   # set False on SHUTDOWN to stop rescheduling
        self._setup_layout()
        self.root.after(100, self._process_queue)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _setup_layout(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(self.main_frame)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.lbl_status = ttk.Label(top, text="SYSTEM STATUS: INITIALIZING", style="Header.TLabel")
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_stats = ttk.Label(top, text="TEMPLATES: 0 | FISSIONS: 0 | PnL: $0", style="TLabel")
        self.lbl_stats.pack(side=tk.RIGHT)

        # ── Three-column body ─────────────────────────────────────────────────
        body = ttk.Frame(self.main_frame)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Column weights
        body.columnconfigure(0, weight=5)   # Physics manifold
        body.columnconfigure(1, weight=4)   # Pareto
        body.columnconfigure(2, weight=3)   # Right panel
        body.rowconfigure(0, weight=3)
        body.rowconfigure(1, weight=2)

        # ── Col 0: Physics Manifold ───────────────────────────────────────────
        phys_frame = ttk.Frame(body)
        phys_frame.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=(0,4))

        ttk.Label(phys_frame, text="PHYSICS MANIFOLD  (Z-Score vs Momentum)",
                  style="Header.TLabel").pack(anchor=tk.W)

        self.fig_phys, self.ax_phys = plt.subplots(figsize=(6, 6), facecolor=BG)
        self.ax_phys.set_facecolor(BG)
        for spine in self.ax_phys.spines.values():
            spine.set_color(FG_GREY)
        self.ax_phys.tick_params(colors=FG_GREY)
        self.ax_phys.xaxis.label.set_color(FG_GREY)
        self.ax_phys.yaxis.label.set_color(FG_GREY)
        self.ax_phys.set_xlabel("Z-Score (Sigma)")
        self.ax_phys.set_ylabel("Momentum Strength")
        self.ax_phys.grid(True, linestyle='--', alpha=0.2, color=FG_GREY)
        self.scatter = self.ax_phys.scatter([], [], c=[], cmap='viridis', s=50, alpha=0.8)

        canvas_phys = FigureCanvasTkAgg(self.fig_phys, master=phys_frame)
        canvas_phys.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_phys = canvas_phys

        # ── Col 1: Pareto Chart ───────────────────────────────────────────────
        pareto_frame = ttk.Frame(body)
        pareto_frame.grid(row=0, column=1, rowspan=2, sticky='nsew', padx=4)

        ttk.Label(pareto_frame, text="PARETO: PROFIT GAP (DMAIC ANALYZE)",
                  style="Header.TLabel").pack(anchor=tk.W)

        # Profit gap summary numbers
        nums = ttk.Frame(pareto_frame)
        nums.pack(fill=tk.X, pady=(4, 2))

        ttk.Label(nums, text="Ideal profit:", style="Dim.TLabel").grid(row=0, column=0, sticky='w')
        self.lbl_ideal  = ttk.Label(nums, text="$0", style="Accent.TLabel")
        self.lbl_ideal.grid(row=0, column=1, sticky='w', padx=6)

        ttk.Label(nums, text="Actual profit:", style="Dim.TLabel").grid(row=1, column=0, sticky='w')
        self.lbl_actual = ttk.Label(nums, text="$0", style="Good.TLabel")
        self.lbl_actual.grid(row=1, column=1, sticky='w', padx=6)

        ttk.Label(nums, text="Captured:", style="Dim.TLabel").grid(row=2, column=0, sticky='w')
        self.lbl_captured = ttk.Label(nums, text="0.0%", style="Bad.TLabel")
        self.lbl_captured.grid(row=2, column=1, sticky='w', padx=6)

        # Pareto bar chart
        self.fig_pareto, self.ax_pareto = plt.subplots(figsize=(5, 5), facecolor=BG)
        self.ax_pareto.set_facecolor(BG)
        for spine in self.ax_pareto.spines.values():
            spine.set_color(FG_GREY)
        self.ax_pareto.tick_params(colors=FG_WHITE, labelsize=10)
        self.ax_pareto.set_title("Where is the profit gap?", color=FG_GREY, fontsize=10)
        self.ax_pareto.grid(axis='x', linestyle='--', alpha=0.2, color=FG_GREY)
        self._pareto_bars = None

        canvas_pareto = FigureCanvasTkAgg(self.fig_pareto, master=pareto_frame)
        canvas_pareto.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_pareto = canvas_pareto

        # ── Col 2: Leaderboard + Log (full height) ────────────────────────────
        right_pane = ttk.Frame(body)
        right_pane.grid(row=0, column=2, rowspan=2, sticky='nsew')

        ttk.Label(right_pane, text="TOP TEMPLATES", style="Header.TLabel").pack(anchor=tk.W)
        cols = ("ID", "Trades", "Win%", "PnL")
        self.tree_ranks = ttk.Treeview(right_pane, columns=cols, show='headings', height=14)
        self.tree_ranks.tag_configure('positive', foreground=FG_GREEN)
        self.tree_ranks.tag_configure('negative', foreground=FG_RED)
        for col in cols:
            self.tree_ranks.heading(col, text=col,
                command=lambda c=col: self._on_header_click(c))
            self.tree_ranks.column(col, width=65)
        self.tree_ranks.pack(fill=tk.X)

        ttk.Label(right_pane, text="EVENTS & ALERTS", style="Header.TLabel").pack(anchor=tk.W, pady=(10, 0))
        self.log_text = tk.Text(right_pane, bg="#000000", fg=FG_GREEN,
                                font=("Consolas", 9), wrap=tk.WORD)
        self.log_text.tag_config('error', foreground=FG_RED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ── Queue processing ──────────────────────────────────────────────────────
    def _process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        finally:
            # Only reschedule while running — stops dangling callbacks after SHUTDOWN
            if self._running:
                self.root.after(500, self._process_queue)

    def _handle_message(self, msg):
        t = msg.get('type')

        if t == 'TEMPLATE_UPDATE':
            tid = msg['id']
            self.templates[tid] = msg
            self._update_leaderboard()
            self._update_manifold()
            self._log(f"Template {tid} | PnL: ${msg.get('pnl', 0):.0f}")

        elif t == 'FISSION_EVENT':
            self.fission_events.append(msg)
            self._log(f"FISSION: {msg['parent_id']} -> {msg['children_count']} ({msg['reason']})", error=True)
            self.lbl_stats.config(text=self._stats_str())

        elif t == 'STATUS':
            self.lbl_status.config(text=f"SYSTEM STATUS: {msg['text']}")

        elif t == 'PHASE_PROGRESS':
            step = msg.get('step', '')
            pct  = msg.get('pct', 0)
            if step:
                self._log(f"{step}  {pct:.0f}%")

        elif t == 'ORACLE_ATTRIBUTION':
            # {'type':'ORACLE_ATTRIBUTION', 'ideal':X, 'actual':X,
            #  'missed':X, 'wrong_dir':X, 'too_early':X, 'noise':X}
            for k in ('ideal', 'actual', 'missed', 'wrong_dir', 'too_early', 'noise'):
                self.attribution[k] = float(msg.get(k, 0))
            self._update_pareto()
            self._log(f"Oracle attribution updated | Captured: {self._capture_pct():.1f}%")

        elif t == 'SHUTDOWN':
            self._running = False
            # Close all matplotlib figures while still in the main loop so tkinter
            # Image objects are deleted here, not from the GC in a daemon thread.
            try:
                plt.close('all')
            except Exception:
                pass
            self.root.quit()

    # ── Pareto chart ──────────────────────────────────────────────────────────
    def _capture_pct(self):
        ideal = self.attribution['ideal']
        return (self.attribution['actual'] / ideal * 100) if ideal > 0 else 0.0

    def _update_pareto(self):
        a = self.attribution
        ideal  = a['ideal']
        actual = a['actual']

        # Update summary labels
        self.lbl_ideal.config(text=f"${ideal:,.0f}")
        self.lbl_actual.config(text=f"${actual:,.0f}")
        cap = self._capture_pct()
        self.lbl_captured.config(text=f"{cap:.1f}%",
                                  style="Good.TLabel" if cap >= 20 else "Bad.TLabel")

        if ideal <= 0:
            return

        # Pareto bars: descending by dollar loss
        buckets = {
            'Missed':    a['missed'],
            'Too Early': a['too_early'],
            'Wrong Dir': a['wrong_dir'],
            'Noise':     a['noise'],
        }
        buckets = dict(sorted(buckets.items(), key=lambda x: x[1], reverse=True))

        labels = list(buckets.keys())
        values = [v / ideal * 100 for v in buckets.values()]
        colors = [PARETO_COLORS[l] for l in labels]

        self.ax_pareto.cla()
        self.ax_pareto.set_facecolor(BG)
        self.ax_pareto.tick_params(colors=FG_WHITE, labelsize=10)
        self.ax_pareto.set_title("Where is the profit gap?", color=FG_GREY, fontsize=10)
        self.ax_pareto.grid(axis='x', linestyle='--', alpha=0.2, color=FG_GREY)

        bars = self.ax_pareto.barh(labels, values, color=colors, height=0.5)

        # Annotate bars with $ amount and %
        for bar, lbl, val, pct in zip(bars, labels, buckets.values(), values):
            self.ax_pareto.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}  ({pct:.1f}%)",
                va='center', ha='left', color=FG_WHITE, fontsize=9
            )

        self.ax_pareto.set_xlabel("% of ideal profit lost", color=FG_GREY)
        self.ax_pareto.invert_yaxis()
        self.ax_pareto.set_xlim(0, max(values) * 1.45 if values else 100)

        # Cumulative Pareto line on twin axis
        ax2 = self.ax_pareto.twiny()
        ax2.set_facecolor(BG)
        ax2.tick_params(colors=FG_GREY, labelsize=8)
        cumulative = np.cumsum(values)
        ax2.plot(cumulative, range(len(labels)), color=FG_BLUE, marker='o',
                 markersize=5, linewidth=1.5, alpha=0.8)
        ax2.axvline(80, color=FG_AMBER, linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Cumulative %", color=FG_GREY)

        self.fig_pareto.tight_layout()
        self.canvas_pareto.draw()

    # ── Physics manifold ──────────────────────────────────────────────────────
    def _update_manifold(self):
        if not self.templates:
            return

        for artist in self._transition_arrows:
            try: artist.remove()
            except ValueError: pass
        self._transition_arrows.clear()

        z_vals = np.array([d.get('z', 0) for d in self.templates.values()])
        m_vals = np.array([d.get('mom', 0) for d in self.templates.values()])

        if len(m_vals) > 4:
            q1, q3 = np.percentile(m_vals, [25, 75])
            iqr = q3 - q1
            m_vals = np.clip(m_vals, q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        c_vals = []
        use_risk = False
        for d in self.templates.values():
            if 'risk_score' in d:
                c_vals.append(d['risk_score']); use_risk = True
            else:
                c_vals.append(d.get('pnl', 0))

        self.scatter.set_offsets(np.c_[z_vals, m_vals])
        self.scatter.set_array(np.array(c_vals))
        if use_risk:
            self.scatter.set_cmap('RdYlGn_r')
            self.scatter.set_clim(0.0, 1.0)
        else:
            self.scatter.set_cmap('viridis')
            self.scatter.autoscale()

        for tid, data in self.templates.items():
            for next_id, prob in data.get('transitions', {}).items():
                if prob > 0.5 and next_id in self.templates:
                    nd = self.templates[next_id]
                    x1, y1 = data.get('z', 0), data.get('mom', 0)
                    x2, y2 = nd.get('z', 0), nd.get('mom', 0)
                    arrow = self.ax_phys.arrow(
                        x1, y1, (x2-x1)*0.9, (y2-y1)*0.9,
                        head_width=0.1, head_length=0.1,
                        fc='white', ec='white', alpha=0.5,
                        length_includes_head=True
                    )
                    self._transition_arrows.append(arrow)

        self.ax_phys.relim()
        self.ax_phys.autoscale_view()
        self.canvas_phys.draw()

    # ── Leaderboard ───────────────────────────────────────────────────────────
    def _on_header_click(self, col):
        """Sort leaderboard by clicked column."""
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = True  # Default to descending for new col

        # Update headers with arrows
        for c in ("ID", "Trades", "Win%", "PnL"):
            text = c
            if c == self._sort_col:
                text += " ↓" if self._sort_reverse else " ↑"
            self.tree_ranks.heading(c, text=text)

        self._update_leaderboard()

    def _update_leaderboard(self):
        for i in self.tree_ranks.get_children():
            self.tree_ranks.delete(i)

        # Map column names to data keys
        key_map = {"ID": "id", "Trades": "count", "Win%": "win_rate", "PnL": "pnl"}
        sort_key = key_map.get(self._sort_col, "pnl")

        top = sorted(
            self.templates.values(),
            key=lambda x: x.get(sort_key, 0),
            reverse=self._sort_reverse
        )[:TOP_TEMPLATES_LIMIT]

        for t in top:
            pnl = t.get('pnl', 0)
            tag = 'positive' if pnl > 0 else 'negative' if pnl < 0 else ''
            win_pct = t.get('win_rate', 0) * 100
            self.tree_ranks.insert("", tk.END, values=(
                t['id'], t.get('count', 0),
                f"{win_pct:.0f}%", f"${pnl:.0f}"
            ), tags=(tag,))
        self.lbl_stats.config(text=self._stats_str())

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log(self, text, error=False):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        tag = "ERR " if error else "    "
        line_tags = ('error',) if error else ()
        self.log_text.insert(tk.END, f"[{ts}] {tag}{text}\n", line_tags)
        self.log_text.see(tk.END)

    def _stats_str(self):
        total_pnl = sum(t.get('pnl', 0) for t in self.templates.values())
        return f"TEMPLATES: {len(self.templates)} | FISSIONS: {len(self.fission_events)} | PnL: ${total_pnl:.0f}"


# ── Lightweight progress popup with PnL control chart (default UI) ────────────
class ProgressPopup:
    """
    460x490 progress window with live PnL control chart.
    Stays open after training completes — close manually when done.
    """

    _CHART_W = 420
    _CHART_H = 130

    def __init__(self, root, q):
        self.root = root
        self.q    = q
        self._pnl_history = []
        self._done = False

        self.root.title("Bayesian-AI Training")
        self.root.geometry("460x490+60+60")
        self.root.configure(bg=BG)
        self.root.resizable(True, False)
        self.root.attributes('-topmost', True)

        style = ttk.Style()
        style.configure("Popup.Horizontal.TProgressbar",
                         troughcolor='#333333', background='#00cc44', thickness=18)

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(root, text="BAYESIAN-AI TRAINING", bg=BG, fg=FG_WHITE,
                 font=('Consolas', 12, 'bold')).pack(pady=(14, 2))

        # Phase name (bold, amber) — e.g. "FORWARD PASS"
        self._phase_var = tk.StringVar(value="Initializing...")
        tk.Label(root, textvariable=self._phase_var, bg=BG, fg=FG_AMBER,
                 font=('Consolas', 11, 'bold')).pack()

        # Progress detail line — e.g. "Day 126 / 250" or sub-step name
        self._step_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self._step_var, bg=BG, fg=FG_GREY,
                 font=('Consolas', 9)).pack(pady=(1, 3))

        # ── Progress bar ──────────────────────────────────────────────────────
        self._pbar = ttk.Progressbar(root, style="Popup.Horizontal.TProgressbar",
                                     orient='horizontal', length=420, mode='determinate')
        self._pbar.pack()

        # Percentage label — prominent, right-aligned feel
        self._pct_var = tk.StringVar(value="0%")
        tk.Label(root, textvariable=self._pct_var, bg=BG, fg=FG_WHITE,
                 font=('Consolas', 10, 'bold')).pack(pady=(3, 10))

        # ── Stats row ─────────────────────────────────────────────────────────
        stats_frame = tk.Frame(root, bg=BG)
        stats_frame.pack(fill='x', padx=20)
        for col, lbl in enumerate(("Net PnL", "Win Rate", "Trades")):
            tk.Label(stats_frame, text=lbl, bg=BG, fg=FG_GREY,
                     font=('Consolas', 8)).grid(row=0, column=col, padx=20)

        self._pnl_var    = tk.StringVar(value="$0")
        self._wr_var     = tk.StringVar(value="—")
        self._trades_var = tk.StringVar(value="0")

        self._pnl_lbl = tk.Label(stats_frame, textvariable=self._pnl_var, bg=BG,
                                  fg=FG_GREEN, font=('Consolas', 14, 'bold'))
        self._pnl_lbl.grid(row=1, column=0, padx=20)
        tk.Label(stats_frame, textvariable=self._wr_var, bg=BG,
                 fg=FG_WHITE, font=('Consolas', 14, 'bold')).grid(row=1, column=1, padx=20)
        tk.Label(stats_frame, textvariable=self._trades_var, bg=BG,
                 fg=FG_WHITE, font=('Consolas', 14, 'bold')).grid(row=1, column=2, padx=20)

        # ── PnL control chart ─────────────────────────────────────────────────
        tk.Label(root, text="PnL Curve", bg=BG, fg=FG_GREY,
                 font=('Consolas', 8)).pack(pady=(14, 2))
        self._canvas = tk.Canvas(root, width=self._CHART_W, height=self._CHART_H,
                                 bg='#141414', highlightthickness=1,
                                 highlightbackground='#333333')
        self._canvas.pack(padx=20)

        # ── Status footer ─────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Running...")
        tk.Label(root, textvariable=self._status_var, bg=BG, fg=FG_GREY,
                 font=('Consolas', 8)).pack(pady=(8, 10))

        self.root.after(250, self._poll)

    # ── Chart ─────────────────────────────────────────────────────────────────
    def _redraw_chart(self):
        c = self._canvas
        c.delete('all')
        pts = self._pnl_history
        if len(pts) < 2:
            c.create_text(self._CHART_W // 2, self._CHART_H // 2,
                          text="Waiting for data...", fill=FG_GREY,
                          font=('Consolas', 9))
            return

        W, H, pad = self._CHART_W, self._CHART_H, 6
        mn, mx = min(pts), max(pts)
        span = mx - mn if mx != mn else 1.0

        # Zero baseline
        zero_y = H - pad - max(0.0, (0 - mn) / span) * (H - 2 * pad)
        zero_y = max(pad, min(H - pad, zero_y))
        c.create_line(pad, zero_y, W - pad, zero_y, fill='#444444', dash=(3, 3))

        # Polyline coords
        coords = []
        for i, v in enumerate(pts):
            x = pad + i / (len(pts) - 1) * (W - 2 * pad)
            y = H - pad - ((v - mn) / span) * (H - 2 * pad)
            coords.extend([x, y])

        # Shaded fill under curve
        fill_pts = [pad, zero_y] + coords + [W - pad, zero_y]
        shade = '#002200' if pts[-1] >= 0 else '#220000'
        c.create_polygon(fill_pts, fill=shade, outline='')

        # Curve line
        color = FG_GREEN if pts[-1] >= 0 else FG_RED
        c.create_line(coords, fill=color, width=2, smooth=True)

        # Current value label at right end
        last_x = coords[-2]
        last_y = coords[-1]
        sign = '+' if pts[-1] >= 0 else ''
        c.create_text(last_x - 2, last_y - 9,
                      text=f"{sign}${pts[-1]:,.0f}",
                      fill=color, font=('Consolas', 7, 'bold'), anchor='e')

    # ── Queue polling ─────────────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                msg   = self.q.get_nowait()
                mtype = msg.get('type', '')
                if mtype == 'PHASE_PROGRESS':
                    phase  = msg.get('phase', '')
                    step   = msg.get('step', '')
                    pct    = float(msg.get('pct', 0))
                    pnl    = msg.get('pnl')
                    trades = msg.get('trades')
                    wr     = msg.get('wr')

                    # Derive a clean phase label and a day/detail sub-line
                    # step format: "FORWARD_PASS  day 126/250" or "FORWARD_PASS COMPLETE" etc.
                    import re as _re
                    _day_m = _re.search(r'day\s+(\d+)/(\d+)', step, _re.I)
                    if _day_m:
                        _cur, _tot = int(_day_m.group(1)), int(_day_m.group(2))
                        phase_label = "FORWARD PASS"
                        detail      = f"Day {_cur} / {_tot}"
                    elif step == 'FORWARD_PASS COMPLETE':
                        phase_label = "FORWARD PASS"
                        detail      = "Complete"
                    elif step == 'STRATEGY_SELECTION':
                        phase_label = "STRATEGY SELECTION"
                        detail      = ""
                    elif step == 'FORWARD_PASS':
                        phase_label = "FORWARD PASS"
                        detail      = "Starting..."
                    else:
                        phase_label = phase or step
                        detail      = step if phase else ""

                    self._phase_var.set(phase_label)
                    self._step_var.set(detail)
                    self._pbar['value'] = pct
                    self._pct_var.set(f"{pct:.1f}%")

                    if pnl is not None:
                        sign = '+' if pnl >= 0 else ''
                        self._pnl_var.set(f"{sign}${pnl:,.0f}")
                        self._pnl_lbl.config(fg=FG_GREEN if pnl >= 0 else FG_RED)
                        self._pnl_history.append(pnl)
                        self._redraw_chart()
                    if wr is not None:
                        self._wr_var.set(f"{wr:.1f}%")
                    if trades is not None:
                        self._trades_var.set(f"{trades:,}")

                    if step == 'FORWARD_PASS COMPLETE':
                        self._done = True
                        self._status_var.set("COMPLETE — close window when ready")
                        self._pct_var.set("100%")
                        self.root.attributes('-topmost', False)

                elif mtype == 'SHUTDOWN':
                    if not self._done:
                        self._status_var.set("Stopped — close window when ready")
                    return  # stop polling; window stays open
        except Exception:
            pass
        self.root.after(250, self._poll)


def launch_popup(queue):
    """Launch the lightweight progress popup in its own Tk mainloop."""
    root = tk.Tk()
    ProgressPopup(root, queue)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


# ── Full dashboard entry point ─────────────────────────────────────────────────
def launch_dashboard(queue):
    root = tk.Tk()
    app = FractalDashboard(root, queue)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    import threading, time
    q = queue.Queue()

    def _sim():
        time.sleep(1)
        q.put({'type': 'STATUS', 'text': 'SCANNING ATLAS...'})
        q.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze', 'step': 'PATTERN_DISCOVERY', 'pct': 20})
        time.sleep(1)
        for i, (tid, z, m, pnl, wr) in enumerate([
            (150, 1.8, 4.2, 5016, 0.57), (391, -2.1, -3.8, 5177, 0.55),
            (463, 2.4, 5.1, 4763, 0.56), (173, -1.6, -2.9, 3265, 0.66),
        ]):
            q.put({'type': 'TEMPLATE_UPDATE', 'id': tid, 'z': z, 'mom': m,
                   'pnl': pnl, 'count': 300+i*50, 'win_rate': wr})
            time.sleep(0.3)
        q.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze', 'step': 'FORWARD_PASS', 'pct': 65})
        time.sleep(1)
        q.put({'type': 'ORACLE_ATTRIBUTION',
               'ideal': 842400, 'actual': 18661,
               'missed': 620000, 'too_early': 124800,
               'wrong_dir': 28110, 'noise': 5568})
        q.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze', 'step': 'COMPLETE', 'pct': 100})
        time.sleep(1)
        q.put({'type': 'FISSION_EVENT', 'parent_id': 150, 'children_count': 3, 'reason': 'Variance'})

    threading.Thread(target=_sim, daemon=True).start()
    launch_dashboard(q)
