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

        # Forward pass live stats
        self.fp_stats = {'day': 0, 'n_days': 0, 'pnl': 0.0, 'trades': 0, 'wr': 0.0, 'pct': 0.0}

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

        # Forward pass progress row
        ttk.Label(nums, text="Forward pass:", style="Dim.TLabel").grid(row=3, column=0, sticky='w', pady=(6,0))
        self.lbl_fp_day = ttk.Label(nums, text="—", style="TLabel")
        self.lbl_fp_day.grid(row=3, column=1, sticky='w', padx=6, pady=(6,0))
        self.fp_progress = ttk.Progressbar(nums, orient='horizontal', length=220, mode='determinate')
        self.fp_progress.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(2,0))
        self.lbl_fp_stats = ttk.Label(nums, text="", style="Dim.TLabel")
        self.lbl_fp_stats.grid(row=5, column=0, columnspan=2, sticky='w')

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

        elif t == 'FORWARD_PASS_STATS':
            fp = msg
            self.fp_stats = fp
            day, n_days = fp['day'], fp['n_days']
            pnl, trades, wr, pct = fp['pnl'], fp['trades'], fp['wr'], fp['pct']
            self.fp_progress['value'] = pct
            self.lbl_fp_day.config(text=f"day {day}/{n_days}  ({pct:.0f}%)")
            self.lbl_fp_stats.config(
                text=f"PnL: ${pnl:,.0f}  |  Trades: {trades}  |  WR: {wr*100:.1f}%"
            )
            # Live-update actual PnL in pareto section
            self.lbl_actual.config(text=f"${pnl:,.0f}",
                                   style="Good.TLabel" if pnl >= 0 else "Bad.TLabel")
            if day % 10 == 0 or day == n_days:
                self._log(f"FWD day {day}/{n_days} | PnL: ${pnl:,.0f} | Trades: {trades} | WR: {wr*100:.1f}%")

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
            self.root.quit()  # returns control to launch_dashboard finally block

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


# ── Entry point ───────────────────────────────────────────────────────────────
def launch_progress_popup(queue):
    """Lightweight progress popup covering all training phases."""
    root = tk.Tk()
    root.title("Training Progress")
    root.geometry("460x175")
    root.resizable(False, False)
    root.attributes('-topmost', True)

    style = ttk.Style(root)
    style.theme_use('clam')
    BG = '#1a1a2e'
    style.configure('TLabel',    background=BG, foreground='#e0e0e0', font=('Consolas', 10))
    style.configure('Big.TLabel', background=BG, foreground='#00ff88', font=('Consolas', 12, 'bold'))
    style.configure('Dim.TLabel', background=BG, foreground='#888888', font=('Consolas', 9))
    style.configure('TFrame',    background=BG)
    style.configure('green.Horizontal.TProgressbar', troughcolor='#2a2a3e', background='#00ff88')
    style.configure('blue.Horizontal.TProgressbar',  troughcolor='#2a2a3e', background='#4488ff')
    root.configure(bg=BG)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill='both', expand=True)

    # Row 0: phase title + step
    lbl_phase = ttk.Label(frame, text="Initializing...", style='Big.TLabel')
    lbl_phase.grid(row=0, column=0, sticky='w')
    lbl_step  = ttk.Label(frame, text="", style='Dim.TLabel')
    lbl_step.grid(row=0, column=1, sticky='e')

    # Row 1: progress label left, secondary stat right
    lbl_prog  = ttk.Label(frame, text="", style='TLabel')
    lbl_prog.grid(row=1, column=0, sticky='w', pady=(6, 2))
    lbl_right = ttk.Label(frame, text="", style='TLabel')
    lbl_right.grid(row=1, column=1, sticky='e', pady=(6, 2))

    # Row 2: progress bar
    pbar = ttk.Progressbar(frame, orient='horizontal', length=436, mode='indeterminate',
                           style='blue.Horizontal.TProgressbar')
    pbar.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(0, 4))
    pbar.start(40)  # spin until Phase 4 gives us a real pct

    # Row 3: stat line 1
    lbl_stat1 = ttk.Label(frame, text="Waiting for first message...", style='TLabel')
    lbl_stat1.grid(row=3, column=0, columnspan=2, sticky='w')

    # Row 4: stat line 2
    lbl_stat2 = ttk.Label(frame, text="", style='Dim.TLabel')
    lbl_stat2.grid(row=4, column=0, columnspan=2, sticky='w')

    frame.columnconfigure(1, weight=1)

    # Mutable state
    state = {
        'phase': '', 'tmpls': 0, 'fissions': 0,
        'tmpl_r2_sum': 0.0, 'tmpl_wr_sum': 0.0,
        'p3_total': 0,
    }
    _running = [True]

    def _set_color(pnl):
        c = '#00ff88' if pnl >= 0 else '#ff4444'
        style.configure('Big.TLabel', foreground=c)

    def _to_determinate(pct, color='green'):
        pbar.stop()
        pbar['mode']  = 'determinate'
        pbar['style'] = f'{color}.Horizontal.TProgressbar'
        pbar['value'] = pct

    def poll():
        if not _running[0]:
            return
        try:
            while True:
                msg = queue.get_nowait()
                t = msg.get('type', '')

                # ── Phase start announcement ──────────────────────────────────
                if t == 'PHASE_START':
                    ph    = msg.get('phase', 0)
                    label = msg.get('label', f'Phase {ph}')
                    total = msg.get('total', 0)
                    done  = msg.get('done', 0)
                    state['phase'] = str(ph)
                    if ph == 3:
                        state['p3_total'] = total
                        state['tmpls']    = done
                    lbl_phase.config(text=label)
                    lbl_step.config(text='')
                    lbl_stat2.config(text='')
                    if total > 0:
                        pct = done / total * 100
                        _to_determinate(pct, color='blue')
                        lbl_prog.config(text=f"{done} / {total}  ({pct:.0f}%)")
                    else:
                        pbar.stop()
                        pbar['mode']  = 'indeterminate'
                        pbar['style'] = 'blue.Horizontal.TProgressbar'
                        pbar.start(40)
                        lbl_prog.config(text="Starting...")
                    lbl_right.config(text='')
                    lbl_stat1.config(text='')

                # ── Phase 2: per-TF level completed ──────────────────────────
                elif t == 'PHASE_UPDATE' and msg.get('phase') == 2:
                    done  = msg.get('done', 0)
                    total = msg.get('total', 1)
                    tf    = msg.get('tf', '')
                    pats  = msg.get('patterns', 0)
                    pct   = done / total * 100
                    _to_determinate(pct, color='blue')
                    lbl_prog.config(text=f"TF {done} / {total}  ({pct:.0f}%)")
                    lbl_right.config(text=f"{pats:,} patterns")
                    lbl_stat1.config(text=f"Last completed: {tf}")

                # ── Phase 3: per-template optimization ───────────────────────
                elif t == 'TEMPLATE_UPDATE':
                    if state['phase'] != '3':
                        state['phase']       = '3'
                        state['fissions']    = 0
                        state['tmpl_r2_sum'] = 0.0
                        state['tmpl_wr_sum'] = 0.0
                        lbl_phase.config(text='Phase 3 — Template Optimization')
                        lbl_step.config(text='Optuna TPE')
                    done  = msg.get('done',  state['tmpls'] + 1)
                    total = msg.get('total', state['p3_total']) or done
                    state['tmpls'] = done
                    state['tmpl_r2_sum'] += msg.get('adj_r2', 0.0)
                    state['tmpl_wr_sum'] += msg.get('win_rate', 0.0)
                    avg_wr = state['tmpl_wr_sum'] / done * 100
                    avg_r2 = state['tmpl_r2_sum'] / done
                    pct = min(done / total * 100, 100) if total else 0
                    _to_determinate(pct, color='blue')
                    lbl_prog.config(text=f"{done} / {total}  ({pct:.0f}%)")
                    lbl_right.config(text=f"Fissions: {state['fissions']}")
                    lbl_stat1.config(text=f"Avg WR: {avg_wr:.1f}%")
                    lbl_stat2.config(text=f"Avg Adj R²: {avg_r2:.4f}")

                elif t == 'FISSION_EVENT':
                    state['fissions'] += 1
                    lbl_right.config(text=f"Fissions: {state['fissions']}")

                # ── Phase 4: forward pass ─────────────────────────────────────
                elif t == 'FORWARD_PASS_STATS':
                    if state['phase'] != 'ForwardPass':
                        state['phase'] = 'ForwardPass'
                        lbl_phase.config(text='Phase 4 — Forward Pass')
                        lbl_step.config(text='')
                    day, n_days = msg['day'], msg['n_days']
                    pnl, trades, wr, pct = msg['pnl'], msg['trades'], msg['wr'], msg['pct']
                    _to_determinate(pct)
                    lbl_prog.config(text=f"Day {day} / {n_days}  ({pct:.0f}%)")
                    lbl_right.config(text=f"WR {wr*100:.1f}%  |  {trades} trades")
                    sign = '+' if pnl >= 0 else ''
                    lbl_stat1.config(text=f"PnL: {sign}${pnl:,.0f}")
                    _set_color(pnl)
                    lbl_stat2.config(text="")

                elif t == 'ORACLE_ATTRIBUTION':
                    ideal  = msg.get('ideal', 0)
                    actual = msg.get('actual', 0)
                    cap    = actual / ideal * 100 if ideal else 0
                    _set_color(actual)
                    lbl_stat2.config(text=f"Ideal: ${ideal:,.0f}  |  Capture: {cap:.1f}%")

                elif t == 'SHUTDOWN':
                    _running[0] = False
                    root.quit()
                    return
        except Exception:
            pass
        root.after(300, poll)

    root.after(300, poll)
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def launch_dashboard(queue):
    root = tk.Tk()
    app = FractalDashboard(root, queue)
    try:
        root.mainloop()
    finally:
        # Close matplotlib figures on main thread BEFORE destroy() to prevent
        # PhotoImage.__del__ being called from a GC thread ("main thread not in main loop")
        try:
            plt.close('all')
        except Exception:
            pass
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
