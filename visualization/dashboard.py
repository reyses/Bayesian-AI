"""
Fractal Command Center (Live Dashboard)
DMAIC Analyze Layer  -- real-time Pareto of profit gap across all phases.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import datetime
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
BG = "#1e1e1e"
FG_GREEN = "#00ff00"
FG_WHITE = "#ffffff"
FG_RED = "#ff4444"
FG_AMBER = "#ffaa00"
FG_BLUE = "#44aaff"
FG_GREY = "#888888"

PARETO_COLORS = {
    "Missed": "#ff4444",
    "Wrong Dir": "#ff8800",
    "Too Early": "#ffdd00",
    "Noise": "#888888",
}

TOP_TEMPLATES_LIMIT = 50


class Tooltip:
    """
    Creates a tooltip for a given widget as the mouse hovers over it.
    """

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return

        try:
            x, y, cx, cy = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += cy + self.widget.winfo_rooty() + 25
        except (TypeError, ValueError, AttributeError):
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#333333",
            foreground="#ffffff",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Consolas", 8),
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class FractalDashboard:
    def __init__(self, root, queue):
        self.root = root
        self.queue = queue
        self.root.title("BAYESIAN-AI: FRACTAL COMMAND CENTER")
        self.root.geometry("1600x950")
        self.root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure(
            "TLabel", background=BG, foreground=FG_GREEN, font=("Consolas", 10)
        )
        style.configure(
            "Header.TLabel",
            background=BG,
            foreground=FG_WHITE,
            font=("Consolas", 13, "bold"),
        )
        style.configure(
            "Dim.TLabel", background=BG, foreground=FG_GREY, font=("Consolas", 9)
        )
        style.configure(
            "Accent.TLabel",
            background=BG,
            foreground=FG_AMBER,
            font=("Consolas", 11, "bold"),
        )
        style.configure(
            "Good.TLabel",
            background=BG,
            foreground=FG_GREEN,
            font=("Consolas", 11, "bold"),
        )
        style.configure(
            "Bad.TLabel",
            background=BG,
            foreground=FG_RED,
            font=("Consolas", 11, "bold"),
        )

        # ── Data stores ───────────────────────────────────────────────────────
        self.templates = {}  # id -> {z, mom, pnl, count, ...}
        self.fission_events = []
        self._transition_arrows = []
        self._scatter_ids = []  # Parallel list to scatter points
        self._annot = None  # Tooltip annotation

        # Sorting state for leaderboard
        self._sort_col = "PnL"
        self._sort_reverse = True

        # Oracle attribution
        self.attribution = {
            "ideal": 0.0,
            "actual": 0.0,
            "missed": 0.0,
            "wrong_dir": 0.0,
            "too_early": 0.0,
            "noise": 0.0,
        }

        self._running = True  # set False on SHUTDOWN to stop rescheduling
        self._setup_layout()
        self.root.after(100, self._process_queue)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _setup_layout(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(self.main_frame)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.lbl_status = ttk.Label(
            top, text="SYSTEM STATUS: INITIALIZING", style="Header.TLabel"
        )
        self.lbl_status.pack(side=tk.LEFT)
        Tooltip(self.lbl_status, "Current system operational state and active process.")

        self.lbl_stats = ttk.Label(
            top, text="TEMPLATES: 0 | FISSIONS: 0 | PnL: $0", style="TLabel"
        )
        self.lbl_stats.pack(side=tk.RIGHT)
        Tooltip(
            self.lbl_stats,
            "Key Metrics:\n• Templates: Active strategy patterns\n• Fissions: Adaptation events count\n• PnL: Total Realized Profit/Loss",
        )

        # ── Three-column body ─────────────────────────────────────────────────
        body = ttk.Frame(self.main_frame)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Column weights
        body.columnconfigure(0, weight=5)  # Physics manifold
        body.columnconfigure(1, weight=4)  # Pareto
        body.columnconfigure(2, weight=3)  # Right panel
        body.rowconfigure(0, weight=3)
        body.rowconfigure(1, weight=2)

        # ── Col 0: Physics Manifold ───────────────────────────────────────────
        phys_frame = ttk.Frame(body)
        phys_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 4))

        self.lbl_phys_header = ttk.Label(
            phys_frame,
            text="PHYSICS MANIFOLD  (Z-Score vs Momentum)",
            style="Header.TLabel",
        )
        self.lbl_phys_header.pack(anchor=tk.W)
        Tooltip(
            self.lbl_phys_header,
            "Visualizes strategy templates based on statistical properties.\n\n• X-Axis: Z-Score (Significance)\n• Y-Axis: Momentum (Trend Strength)\n• Color: Profit/Loss or Risk Score",
        )

        self.fig_phys, self.ax_phys = plt.subplots(figsize=(6, 6), facecolor=BG)
        self.ax_phys.set_facecolor(BG)
        for spine in self.ax_phys.spines.values():
            spine.set_color(FG_GREY)
        self.ax_phys.tick_params(colors=FG_GREY)
        self.ax_phys.xaxis.label.set_color(FG_GREY)
        self.ax_phys.yaxis.label.set_color(FG_GREY)
        self.ax_phys.set_xlabel("Z-Score (Sigma)")
        self.ax_phys.set_ylabel("Momentum Strength")
        self.ax_phys.grid(True, linestyle="--", alpha=0.2, color=FG_GREY)
        self.scatter = self.ax_phys.scatter(
            [], [], c=[], cmap="viridis", s=50, alpha=0.8
        )

        # Tooltip annotation (hidden by default)
        self._annot = self.ax_phys.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="black", ec=FG_GREEN, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=FG_GREEN),
            color=FG_GREEN,
            fontsize=8,
            visible=False,
        )

        canvas_phys = FigureCanvasTkAgg(self.fig_phys, master=phys_frame)
        canvas_phys.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_phys = canvas_phys

        # Bind hover event for tooltips
        self.canvas_phys.mpl_connect("motion_notify_event", self._on_hover)
        self._add_chart_context_menu(self.canvas_phys, self.fig_phys, "physics_manifold")

        # ── Col 1: Pareto Chart ───────────────────────────────────────────────
        pareto_frame = ttk.Frame(body)
        pareto_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=4)

        self.lbl_pareto_header = ttk.Label(
            pareto_frame,
            text="PARETO: PROFIT GAP (DMAIC ANALYZE)",
            style="Header.TLabel",
        )
        self.lbl_pareto_header.pack(anchor=tk.W)
        Tooltip(
            self.lbl_pareto_header,
            "DMAIC Analyze Phase: Breakdown of lost potential profit.\nIdentifies where the strategy is leaking value (e.g., missed trades, wrong direction).",
        )

        # Profit gap summary numbers
        nums = ttk.Frame(pareto_frame)
        nums.pack(fill=tk.X, pady=(4, 2))

        ttk.Label(nums, text="Ideal profit:", style="Dim.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.lbl_ideal = ttk.Label(nums, text="$0", style="Accent.TLabel")
        self.lbl_ideal.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(nums, text="Actual profit:", style="Dim.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        self.lbl_actual = ttk.Label(nums, text="$0", style="Good.TLabel")
        self.lbl_actual.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(nums, text="Captured:", style="Dim.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self.lbl_captured = ttk.Label(nums, text="0.0%", style="Bad.TLabel")
        self.lbl_captured.grid(row=2, column=1, sticky="w", padx=6)

        # Pareto bar chart
        self.fig_pareto, self.ax_pareto = plt.subplots(figsize=(5, 5), facecolor=BG)
        self.ax_pareto.set_facecolor(BG)
        for spine in self.ax_pareto.spines.values():
            spine.set_color(FG_GREY)
        self.ax_pareto.tick_params(colors=FG_WHITE, labelsize=10)
        self.ax_pareto.set_title("Where is the profit gap?", color=FG_GREY, fontsize=10)
        self.ax_pareto.grid(axis="x", linestyle="--", alpha=0.2, color=FG_GREY)
        self._pareto_bars = None

        canvas_pareto = FigureCanvasTkAgg(self.fig_pareto, master=pareto_frame)
        canvas_pareto.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_pareto = canvas_pareto
        self._add_chart_context_menu(self.canvas_pareto, self.fig_pareto, "pareto_analysis")

        # ── Col 2: Leaderboard + Log (full height) ────────────────────────────
        right_pane = ttk.Frame(body)
        right_pane.grid(row=0, column=2, rowspan=2, sticky="nsew")

        self.lbl_templates_header = ttk.Label(
            right_pane, text="TOP TEMPLATES", style="Header.TLabel"
        )
        self.lbl_templates_header.pack(anchor=tk.W)
        Tooltip(
            self.lbl_templates_header,
            "Leaderboard of top performing strategy templates.\nClick column headers to sort by ID, Trade Count, Win Rate, or PnL.",
        )

        cols = ("ID", "Trades", "Win%", "PnL")
        self.tree_ranks = ttk.Treeview(
            right_pane, columns=cols, show="headings", height=14
        )
        self.tree_ranks.tag_configure("positive", foreground=FG_GREEN)
        self.tree_ranks.tag_configure("negative", foreground=FG_RED)
        for col in cols:
            self.tree_ranks.heading(
                col, text=col, command=lambda c=col: self._on_header_click(c)
            )
            self.tree_ranks.column(col, width=65)
        self.tree_ranks.pack(fill=tk.X)

        ttk.Label(right_pane, text="EVENTS & ALERTS", style="Header.TLabel").pack(
            anchor=tk.W, pady=(10, 0)
        )
        self.log_text = tk.Text(
            right_pane, bg="#000000", fg=FG_GREEN, font=("Consolas", 9), wrap=tk.WORD
        )
        self.log_text.tag_config("error", foreground=FG_RED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Context menu for log
        self.menu_log = tk.Menu(self.root, tearoff=0)
        self.menu_log.add_command(label="Copy Selection", command=self._log_copy_sel)
        self.menu_log.add_command(label="Copy All", command=self._log_copy_all)
        self.menu_log.add_separator()
        self.menu_log.add_command(label="Clear Log", command=self._log_clear)
        self.log_text.bind(
            "<Button-3>", lambda e: self.menu_log.tk_popup(e.x_root, e.y_root)
        )

        # Context menu for Leaderboard
        self.menu_tree = tk.Menu(self.root, tearoff=0)
        self.menu_tree.add_command(label="Copy Template ID", command=self._tree_copy_id)
        self.menu_tree.add_command(label="Copy Row Data", command=self._tree_copy_row)
        self.tree_ranks.bind(
            "<Button-3>", lambda e: self.menu_tree.tk_popup(e.x_root, e.y_root)
        )

    # ── Chart interactions ────────────────────────────────────────────────────
    def _add_chart_context_menu(self, canvas, fig, default_name):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Save Chart as Image...",
            command=lambda: self._save_chart(fig, default_name),
        )
        canvas.get_tk_widget().bind(
            "<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root)
        )

    def _save_chart(self, fig, default_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{default_name}_{timestamp}.png",
            filetypes=[("PNG", "*.png"), ("All Files", "*.*")],
        )
        if fn:
            try:
                fig.savefig(fn, dpi=DEFAULT_CHART_DPI, bbox_inches="tight", facecolor=BG)
                self._log(f"Chart saved: {fn}")
            except Exception as e:
                self._log(f"Error saving chart: {e}", error=True)

    # ── Log interactions ──────────────────────────────────────────────────────
    def _tree_copy_id(self):
        sel = self.tree_ranks.selection()
        if sel:
            # The iid is the template ID, which is what we want to copy.
            # sel is a tuple of selected iids, we'll take the first one.
            self.root.clipboard_clear()
            self.root.clipboard_append(sel[0])

    def _tree_copy_row(self):
        sel = self.tree_ranks.selection()
        if sel:
            vals = self.tree_ranks.item(sel[0])["values"]
            text = " | ".join(str(v) for v in vals)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _log_copy_sel(self):
        try:
            txt = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
        except tk.TclError:
            pass

    def _log_copy_all(self):
        txt = self.log_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)

    def _log_clear(self):
        self.log_text.delete("1.0", tk.END)

    # ── Queue processing ──────────────────────────────────────────────────────
    def _process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        finally:
            # Only reschedule while running  -- stops dangling callbacks after SHUTDOWN
            if self._running:
                self.root.after(500, self._process_queue)

    def _handle_message(self, msg):
        t = msg.get("type")

        if t == "TEMPLATE_UPDATE":
            tid = msg["id"]
            self.templates[tid] = msg
            self._update_leaderboard()
            self._update_manifold()
            self._log(f"Template {tid} | PnL: ${msg.get('pnl', 0):.0f}")

        elif t == "FISSION_EVENT":
            self.fission_events.append(msg)
            self._log(
                f"FISSION: {msg['parent_id']} -> {msg['children_count']} ({msg['reason']})",
                error=True,
            )
            self.lbl_stats.config(text=self._stats_str())

        elif t == "STATUS":
            self.lbl_status.config(text=f"SYSTEM STATUS: {msg['text']}")

        elif t == "PHASE_PROGRESS":
            step = msg.get("step", "")
            pct = msg.get("pct", 0)
            if step:
                self._log(f"{step}  {pct:.0f}%")

        elif t == "ORACLE_ATTRIBUTION":
            # {'type':'ORACLE_ATTRIBUTION', 'ideal':X, 'actual':X,
            #  'missed':X, 'wrong_dir':X, 'too_early':X, 'noise':X}
            for k in ("ideal", "actual", "missed", "wrong_dir", "too_early", "noise"):
                self.attribution[k] = float(msg.get(k, 0))
            self._update_pareto()
            self._log(
                f"Oracle attribution updated | Captured: {self._capture_pct():.1f}%"
            )

        elif t == "SHUTDOWN":
            self._running = False
            # Close all matplotlib figures while still in the main loop so tkinter
            # Image objects are deleted here, not from the GC in a daemon thread.
            try:
                plt.close("all")
            except Exception:
                pass
            self.root.quit()

    # ── Pareto chart ──────────────────────────────────────────────────────────
    def _capture_pct(self):
        ideal = self.attribution["ideal"]
        return (self.attribution["actual"] / ideal * 100) if ideal > 0 else 0.0

    def _update_pareto(self):
        a = self.attribution
        ideal = a["ideal"]
        actual = a["actual"]

        # Update summary labels
        self.lbl_ideal.config(text=f"${ideal:,.0f}")
        self.lbl_actual.config(text=f"${actual:,.0f}")
        cap = self._capture_pct()
        self.lbl_captured.config(
            text=f"{cap:.1f}%", style="Good.TLabel" if cap >= 20 else "Bad.TLabel"
        )

        if ideal <= 0:
            return

        # Pareto bars: descending by dollar loss
        buckets = {
            "Missed": a["missed"],
            "Too Early": a["too_early"],
            "Wrong Dir": a["wrong_dir"],
            "Noise": a["noise"],
        }
        buckets = dict(sorted(buckets.items(), key=lambda x: x[1], reverse=True))

        labels = list(buckets.keys())
        values = [v / ideal * 100 for v in buckets.values()]
        colors = [PARETO_COLORS[l] for l in labels]

        self.ax_pareto.cla()
        self.ax_pareto.set_facecolor(BG)
        self.ax_pareto.tick_params(colors=FG_WHITE, labelsize=10)
        self.ax_pareto.set_title("Where is the profit gap?", color=FG_GREY, fontsize=10)
        self.ax_pareto.grid(axis="x", linestyle="--", alpha=0.2, color=FG_GREY)

        bars = self.ax_pareto.barh(labels, values, color=colors, height=0.5)

        # Annotate bars with $ amount and %
        for bar, lbl, val, pct in zip(bars, labels, buckets.values(), values):
            self.ax_pareto.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}  ({pct:.1f}%)",
                va="center",
                ha="left",
                color=FG_WHITE,
                fontsize=9,
            )

        self.ax_pareto.set_xlabel("% of ideal profit lost", color=FG_GREY)
        self.ax_pareto.invert_yaxis()
        self.ax_pareto.set_xlim(0, max(values) * 1.45 if values else 100)

        # Cumulative Pareto line on twin axis
        ax2 = self.ax_pareto.twiny()
        ax2.set_facecolor(BG)
        ax2.tick_params(colors=FG_GREY, labelsize=8)
        cumulative = np.cumsum(values)
        ax2.plot(
            cumulative,
            range(len(labels)),
            color=FG_BLUE,
            marker="o",
            markersize=5,
            linewidth=1.5,
            alpha=0.8,
        )
        ax2.axvline(80, color=FG_AMBER, linestyle="--", alpha=0.5, linewidth=1)
        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Cumulative %", color=FG_GREY)

        self.fig_pareto.tight_layout()
        self.canvas_pareto.draw()

    # ── Physics manifold ──────────────────────────────────────────────────────
    def _update_manifold(self):
        if not self.templates:
            return

        for artist in self._transition_arrows:
            try:
                artist.remove()
            except ValueError:
                pass
        self._transition_arrows.clear()

        # Extract data into lists to ensure order synchronization
        data_list = list(self.templates.values())
        self._scatter_ids = [d.get("id") for d in data_list]

        z_vals = np.array([d.get("z", 0) for d in data_list])
        m_vals = np.array([d.get("mom", 0) for d in data_list])

        if len(m_vals) > 4:
            q1, q3 = np.percentile(m_vals, [25, 75])
            iqr = q3 - q1
            m_vals = np.clip(m_vals, q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        c_vals = []
        use_risk = False
        for d in data_list:
            if "risk_score" in d:
                c_vals.append(d["risk_score"])
                use_risk = True
            else:
                c_vals.append(d.get("pnl", 0))

        self.scatter.set_offsets(np.c_[z_vals, m_vals])
        self.scatter.set_array(np.array(c_vals))
        if use_risk:
            self.scatter.set_cmap("RdYlGn_r")
            self.scatter.set_clim(0.0, 1.0)
        else:
            self.scatter.set_cmap("viridis")
            self.scatter.autoscale()

        for tid, data in self.templates.items():
            for next_id, prob in data.get("transitions", {}).items():
                if prob > 0.5 and next_id in self.templates:
                    nd = self.templates[next_id]
                    x1, y1 = data.get("z", 0), data.get("mom", 0)
                    x2, y2 = nd.get("z", 0), nd.get("mom", 0)
                    arrow = self.ax_phys.arrow(
                        x1,
                        y1,
                        (x2 - x1) * 0.9,
                        (y2 - y1) * 0.9,
                        head_width=0.1,
                        head_length=0.1,
                        fc="white",
                        ec="white",
                        alpha=0.5,
                        length_includes_head=True,
                    )
                    self._transition_arrows.append(arrow)

        self.ax_phys.relim()
        self.ax_phys.autoscale_view()
        self.canvas_phys.draw()

    def _on_hover(self, event):
        """Show tooltip when hovering over a manifold point."""
        vis = self._annot.get_visible()
        if event.inaxes == self.ax_phys:
            cont, ind = self.scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                if idx < len(self._scatter_ids):
                    tid = self._scatter_ids[idx]
                    tmpl = self.templates.get(tid)
                    if tmpl:
                        text = (
                            f"ID: {tid}\n"
                            f"PnL: ${tmpl.get('pnl',0):.0f}\n"
                            f"Win%: {tmpl.get('win_rate',0)*100:.0f}%\n"
                            f"Trades: {tmpl.get('count',0)}"
                        )
                        self._annot.xy = (event.xdata, event.ydata)
                        self._annot.set_text(text)
                        self._annot.set_visible(True)
                        self.canvas_phys.draw_idle()
                        return
        if vis:
            self._annot.set_visible(False)
            self.canvas_phys.draw_idle()

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
                text += " v" if self._sort_reverse else " ^"
            self.tree_ranks.heading(c, text=text)

        self._update_leaderboard()

    def _update_leaderboard(self):
        # Preserve selection
        selected_iids = self.tree_ranks.selection()

        for i in self.tree_ranks.get_children():
            self.tree_ranks.delete(i)

        # Map column names to data keys
        key_map = {"ID": "id", "Trades": "count", "Win%": "win_rate", "PnL": "pnl"}
        sort_key = key_map.get(self._sort_col, "pnl")

        top = sorted(
            self.templates.values(),
            key=lambda x: x.get(sort_key, 0),
            reverse=self._sort_reverse,
        )[:TOP_TEMPLATES_LIMIT]

        for t in top:
            pnl = t.get("pnl", 0)
            tag = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
            win_pct = t.get("win_rate", 0) * 100
            self.tree_ranks.insert(
                "",
                tk.END,
                iid=str(t["id"]),
                values=(t["id"], t.get("count", 0), f"{win_pct:.0f}%", f"${pnl:.0f}"),
                tags=(tag,),
            )

        # Restore selection
        to_select = [iid for iid in selected_iids if self.tree_ranks.exists(iid)]
        if to_select:
            self.tree_ranks.selection_set(to_select)

        self.lbl_stats.config(text=self._stats_str())

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log(self, text, error=False):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        tag = "ERR " if error else "    "
        line_tags = ("error",) if error else ()
        self.log_text.insert(tk.END, f"[{ts}] {tag}{text}\n", line_tags)
        self.log_text.see(tk.END)

    def _stats_str(self):
        total_pnl = sum(t.get("pnl", 0) for t in self.templates.values())
        return f"TEMPLATES: {len(self.templates)} | FISSIONS: {len(self.fission_events)} | PnL: ${total_pnl:.0f}"


# ── Lightweight progress popup with PnL control chart (default UI) ────────────
class ProgressPopup:
    """
    460x490 progress window with live PnL control chart.
    Stays open after training completes  -- close manually when done.
    """

    _CHART_W = 420
    _CHART_H = 130

    def __init__(self, root, q, shared_state=None):
        self.root = root
        self.q = q
        self._shared_state = shared_state  # None = training mode
        self._pnl_history = [0]  # cumulative PnL per trade (chart tracks by trade)
        self._current_mode = shared_state.get('mode', 'is') if shared_state else 'is'
        self._done = False

        self.root.title("Bayesian-AI LIVE" if shared_state else "Bayesian-AI Training")
        self.root.geometry("620x880+60+60")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.attributes("-topmost", True)

        style = ttk.Style()
        style.configure(
            "Popup.Horizontal.TProgressbar",
            troughcolor="#333333",
            background="#00cc44",
            thickness=18,
        )

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(
            root,
            text="BAYESIAN-AI TRAINING",
            bg=BG,
            fg=FG_WHITE,
            font=("Consolas", 12, "bold"),
        ).pack(pady=(14, 2))

        # ── Aggression slider (live mode only) ────────────────────────────────
        if self._shared_state is not None:
            agg_frame = tk.Frame(root, bg=BG)
            agg_frame.pack(fill="x", padx=20, pady=(4, 0))

            self._agg_label_var = tk.StringVar(value="Aggression: 50%")
            tk.Label(
                agg_frame, textvariable=self._agg_label_var,
                bg=BG, fg=FG_AMBER, font=("Consolas", 9, "bold"),
            ).pack(side=tk.LEFT, padx=(0, 8))

            self._agg_scale = tk.Scale(
                agg_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                bg=BG, fg=FG_WHITE, troughcolor="#333333",
                highlightbackground=BG, font=("Consolas", 8),
                showvalue=False, length=250,
                command=self._on_aggression_change,
            )
            self._agg_scale.set(int(self._shared_state.get('aggression', 0.5) * 100))
            self._agg_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # ── Manual BUY / SELL / FLATTEN buttons ──────────────────────────
            btn_frame = tk.Frame(root, bg=BG)
            btn_frame.pack(fill="x", padx=20, pady=(6, 0))

            self._buy_btn = tk.Button(
                btn_frame, text="BUY", bg="#006600", fg=FG_WHITE,
                activebackground="#00aa00", font=("Consolas", 10, "bold"),
                width=8, command=lambda: self._manual_order('BUY'),
            )
            self._buy_btn.pack(side=tk.LEFT, padx=(0, 6))

            self._sell_btn = tk.Button(
                btn_frame, text="SELL", bg="#880000", fg=FG_WHITE,
                activebackground="#cc0000", font=("Consolas", 10, "bold"),
                width=8, command=lambda: self._manual_order('SELL'),
            )
            self._sell_btn.pack(side=tk.LEFT, padx=(0, 6))

            self._flatten_btn = tk.Button(
                btn_frame, text="FLAT", bg="#333333", fg=FG_WHITE,
                activebackground="#555555", font=("Consolas", 10, "bold"),
                width=10, command=lambda: self._manual_order('FLATTEN'),
            )
            self._flatten_btn.pack(side=tk.LEFT, padx=(0, 6))

            tk.Button(
                btn_frame, text="SAVE", bg="#004488", fg=FG_WHITE,
                activebackground="#0066cc", font=("Consolas", 10, "bold"),
                width=8, command=self._request_save,
            ).pack(side=tk.RIGHT)

            tk.Button(
                btn_frame, text="SNAP", bg="#664400", fg=FG_WHITE,
                activebackground="#886600", font=("Consolas", 10, "bold"),
                width=6, command=self._take_screenshot,
            ).pack(side=tk.RIGHT, padx=4)

            _pp_on = self._shared_state.get('ping_pong', False)
            self._pp_btn = tk.Button(
                btn_frame, text="PING-PONG",
                bg="#008800" if _pp_on else "#444444", fg=FG_WHITE,
                activebackground="#00cc00" if _pp_on else "#666666",
                font=("Consolas", 10, "bold"),
                width=10, command=self._toggle_ping_pong,
            )
            self._pp_btn.pack(side=tk.RIGHT, padx=(0, 6))

            # ── NT8 Account Equity row ───────────────────────────────────
            eq_frame = tk.Frame(root, bg=BG)
            eq_frame.pack(fill="x", padx=20, pady=(8, 0))
            for col, lbl in enumerate(("Cash Value", "Unrealized", "Net Liq")):
                tk.Label(
                    eq_frame, text=lbl, bg=BG, fg=FG_GREY, font=("Consolas", 8),
                ).grid(row=0, column=col, padx=14)

            self._cash_var = tk.StringVar(value="--")
            self._unreal_var = tk.StringVar(value="--")
            self._netliq_var = tk.StringVar(value="--")

            tk.Label(
                eq_frame, textvariable=self._cash_var, bg=BG, fg=FG_WHITE,
                font=("Consolas", 12, "bold"),
            ).grid(row=1, column=0, padx=14)
            self._unreal_lbl = tk.Label(
                eq_frame, textvariable=self._unreal_var, bg=BG, fg=FG_WHITE,
                font=("Consolas", 12, "bold"),
            )
            self._unreal_lbl.grid(row=1, column=1, padx=14)
            self._netliq_lbl = tk.Label(
                eq_frame, textvariable=self._netliq_var, bg=BG, fg=FG_BLUE,
                font=("Consolas", 12, "bold"),
            )
            self._netliq_lbl.grid(row=1, column=2, padx=14)

        # Phase name (bold, amber)  -- e.g. "FORWARD PASS"
        self._phase_var = tk.StringVar(value="Initializing...")
        tk.Label(
            root,
            textvariable=self._phase_var,
            bg=BG,
            fg=FG_AMBER,
            font=("Consolas", 11, "bold"),
        ).pack()

        # Progress detail line  -- e.g. "Day 126 / 250" or sub-step name
        self._step_var = tk.StringVar(value="")
        tk.Label(
            root, textvariable=self._step_var, bg=BG, fg=FG_GREY, font=("Consolas", 9)
        ).pack(pady=(1, 3))

        # ── Progress bar ──────────────────────────────────────────────────────
        self._pbar = ttk.Progressbar(
            root,
            style="Popup.Horizontal.TProgressbar",
            orient="horizontal",
            length=420,
            mode="determinate",
        )
        self._pbar.pack(fill=tk.X, padx=20)

        # Trade health label  -- shows position ticks or trade count
        self._pct_var = tk.StringVar(value="0%")
        self._pct_lbl = tk.Label(
            root,
            textvariable=self._pct_var,
            bg=BG,
            fg=FG_WHITE,
            font=("Consolas", 10, "bold"),
        )
        self._pct_lbl.pack(pady=(3, 10))

        # ── Stats (compact single row) ───────────────────────────────────────
        stats_frame = tk.Frame(root, bg=BG)
        stats_frame.pack(fill="x", padx=10, pady=(2, 0))
        _stat_labels = ("PnL", "WR", "Trades", "PF", "Win$", "Loss$")
        for col, lbl in enumerate(_stat_labels):
            tk.Label(stats_frame, text=lbl, bg=BG, fg=FG_GREY,
                     font=("Consolas", 7)).grid(row=0, column=col, padx=6)

        self._pnl_var = tk.StringVar(value="$0")
        self._wr_var = tk.StringVar(value="--")
        self._trades_var = tk.StringVar(value="0")
        self._pf_var = tk.StringVar(value="--")
        self._gw_var = tk.StringVar(value="$0")
        self._gl_var = tk.StringVar(value="$0")

        self._pnl_lbl = tk.Label(stats_frame, textvariable=self._pnl_var,
                                  bg=BG, fg=FG_GREEN, font=("Consolas", 12, "bold"))
        self._pnl_lbl.grid(row=1, column=0, padx=6)
        tk.Label(stats_frame, textvariable=self._wr_var, bg=BG, fg=FG_WHITE,
                 font=("Consolas", 12, "bold")).grid(row=1, column=1, padx=6)
        tk.Label(stats_frame, textvariable=self._trades_var, bg=BG, fg=FG_WHITE,
                 font=("Consolas", 12, "bold")).grid(row=1, column=2, padx=6)
        self._pf_lbl = tk.Label(stats_frame, textvariable=self._pf_var,
                                 bg=BG, fg=FG_WHITE, font=("Consolas", 12, "bold"))
        self._pf_lbl.grid(row=1, column=3, padx=6)
        self._gw_lbl = tk.Label(stats_frame, textvariable=self._gw_var,
                                 bg=BG, fg=FG_GREEN, font=("Consolas", 12, "bold"))
        self._gw_lbl.grid(row=1, column=4, padx=6)
        self._gl_lbl = tk.Label(stats_frame, textvariable=self._gl_var,
                                 bg=BG, fg=FG_RED, font=("Consolas", 12, "bold"))
        self._gl_lbl.grid(row=1, column=5, padx=6)

        # ── PnL control chart ─────────────────────────────────────────────────
        tk.Label(root, text="PnL by Trade", bg=BG, fg=FG_GREY, font=("Consolas", 8)).pack(
            pady=(14, 2)
        )
        self._canvas = tk.Canvas(
            root,
            width=self._CHART_W,
            height=self._CHART_H,
            bg="#141414",
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self._canvas.pack(padx=20, fill=tk.X, expand=False)
        self._canvas.bind("<Configure>", lambda e: self._redraw_chart())

        # ── Capture rate quartiles (between PnL and Price charts) ─────────
        exit_frame = tk.Frame(root, bg=BG)
        exit_frame.pack(fill="x", padx=20, pady=(4, 0))
        _exit_labels = ("Reversed", "Q1 0-25%", "Q2 25-50%", "Q3 50-75%", "Q4 75-100%", "100%+")
        _exit_colors = ("#ff2222", FG_RED, FG_AMBER, FG_AMBER, FG_GREEN, "#00ffff")
        for col, lbl in enumerate(_exit_labels):
            tk.Label(
                exit_frame, text=lbl, bg=BG, fg=FG_GREY, font=("Consolas", 7)
            ).grid(row=0, column=col, padx=6)

        self._cap_rev_var = tk.StringVar(value="--")
        self._cap_q1_var = tk.StringVar(value="--")
        self._cap_q2_var = tk.StringVar(value="--")
        self._cap_q3_var = tk.StringVar(value="--")
        self._cap_q4_var = tk.StringVar(value="--")
        self._cap_plus_var = tk.StringVar(value="--")

        for col, (var, clr) in enumerate(zip(
            (self._cap_rev_var, self._cap_q1_var, self._cap_q2_var,
             self._cap_q3_var, self._cap_q4_var, self._cap_plus_var),
            _exit_colors,
        )):
            tk.Label(
                exit_frame, textvariable=var, bg=BG, fg=clr,
                font=("Consolas", 10, "bold"),
            ).grid(row=1, column=col, padx=6)

        # ── Live Price Chart ─────────────────────────────────────────────
        self._price_history = []  # last N prices for line chart
        self._active_entry_price = None  # horizontal line while in trade
        self._active_entry_side = None
        self._MAX_PRICE_PTS = 200  # rolling window
        self._trade_markers = []  # (price_index, action, side, price, pnl)
        self._active_side = None  # 'long'/'short' when in position, None when flat

        price_header = tk.Frame(root, bg=BG)
        price_header.pack(fill="x", padx=20, pady=(10, 2))
        tk.Label(price_header, text="Price", bg=BG, fg=FG_GREY,
                 font=("Consolas", 8)).pack(side="left")
        self._price_var = tk.StringVar(value="--")
        self._price_lbl = tk.Label(
            price_header, textvariable=self._price_var, bg=BG, fg=FG_WHITE,
            font=("Consolas", 10, "bold"))
        self._price_lbl.pack(side="right")

        self._price_canvas = tk.Canvas(
            root, width=self._CHART_W, height=self._CHART_H,
            bg="#141414", highlightthickness=1, highlightbackground="#333333")
        self._price_canvas.pack(padx=20, fill=tk.X, expand=False)
        self._price_canvas.bind("<Configure>", lambda e: self._redraw_price_chart())

        # ── DMI chart ─────────────────────────────────────────────────────────
        _dmi_frame = tk.Frame(root, bg=BG)
        _dmi_frame.pack(fill="x", padx=20, pady=(6, 0))
        tk.Label(_dmi_frame, text="DMI", bg=BG, fg=FG_GREY,
                 font=("Consolas", 8)).pack(side="left")
        self._dmi_label_var = tk.StringVar(value="--")
        tk.Label(_dmi_frame, textvariable=self._dmi_label_var, bg=BG, fg=FG_WHITE,
                 font=("Consolas", 8)).pack(side="right")

        self._DMI_CHART_H = 100
        self._dmi_canvas = tk.Canvas(
            root, width=self._CHART_W, height=self._DMI_CHART_H,
            bg="#141414", highlightthickness=1, highlightbackground="#333333")
        self._dmi_canvas.pack(padx=20, fill=tk.X, expand=False)
        self._dmi_canvas.bind("<Configure>", lambda e: self._redraw_dmi_chart())
        self._dmi_plus_history = []
        self._dmi_minus_history = []
        self._MAX_DMI_PTS = self._MAX_PRICE_PTS

        # ── Status footer ─────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Running...")
        tk.Label(
            root, textvariable=self._status_var, bg=BG, fg=FG_GREY, font=("Consolas", 8)
        ).pack(pady=(8, 10))

        self.root.after(250, self._poll)

    # ── Chart ─────────────────────────────────────────────────────────────────
    def _redraw_chart(self):
        c = self._canvas
        c.delete("all")
        pts = self._pnl_history
        W = max(100, c.winfo_width())
        H = max(40, c.winfo_height())
        if len(pts) < 2:
            c.create_text(
                W // 2,
                H // 2,
                text="Waiting for data...",
                fill=FG_GREY,
                font=("Consolas", 9),
            )
            return

        pad = 6
        mn, mx = min(pts), max(pts)
        span = mx - mn if mx != mn else 1.0

        # Zero baseline
        zero_y = H - pad - max(0.0, (0 - mn) / span) * (H - 2 * pad)
        zero_y = max(pad, min(H - pad, zero_y))
        c.create_line(pad, zero_y, W - pad, zero_y, fill="#444444", dash=(3, 3))

        # Polyline coords
        coords = []
        for i, v in enumerate(pts):
            x = pad + i / (len(pts) - 1) * (W - 2 * pad)
            y = H - pad - ((v - mn) / span) * (H - 2 * pad)
            coords.extend([x, y])

        # Shaded fill under curve
        fill_pts = [pad, zero_y] + coords + [W - pad, zero_y]
        shade = "#002200" if pts[-1] >= 0 else "#220000"
        c.create_polygon(fill_pts, fill=shade, outline="")

        # Curve line
        color = FG_GREEN if pts[-1] >= 0 else FG_RED
        c.create_line(coords, fill=color, width=2, smooth=True)

        # Current value label at right end
        last_x = coords[-2]
        last_y = coords[-1]
        sign = "+" if pts[-1] >= 0 else ""
        c.create_text(
            last_x - 2,
            last_y - 9,
            text=f"{sign}${pts[-1]:,.0f}",
            fill=color,
            font=("Consolas", 7, "bold"),
            anchor="e",
        )

        # Trade numbers along bottom (~8 evenly spaced)
        n_trades = len(pts) - 1  # first entry is $0 baseline
        if n_trades >= 2:
            n_labels = min(8, n_trades)
            step = max(1, n_trades // n_labels)
            for j in range(1, len(pts), step):
                frac = j / (len(pts) - 1)
                x = pad + frac * (W - 2 * pad)
                c.create_text(x, H - 1, text=f"T{j}", fill="#555555",
                              font=("Consolas", 6), anchor="s")

    # ── Price line chart ──────────────────────────────────────────────────
    def _redraw_price_chart(self):
        c = self._price_canvas
        c.delete("all")
        pts = self._price_history
        W = max(100, c.winfo_width())
        H = max(40, c.winfo_height())
        if len(pts) < 2:
            c.create_text(W // 2, H // 2, text="Waiting for bars...",
                          fill=FG_GREY, font=("Consolas", 9))
            return

        pad = 6
        mn, mx = min(pts), max(pts)
        span = mx - mn if mx != mn else 1.0

        # Price grid lines (3 levels)
        for frac in (0.25, 0.5, 0.75):
            gy = pad + frac * (H - 2 * pad)
            price_at = mx - frac * span
            c.create_line(pad, gy, W - pad, gy, fill="#282828", dash=(2, 4))
            c.create_text(W - pad + 2, gy, text=f"{price_at:,.0f}",
                          fill="#444444", font=("Consolas", 5), anchor="w")

        # Polyline
        coords = []
        for i, v in enumerate(pts):
            x = pad + i / (len(pts) - 1) * (W - 2 * pad)
            y = H - pad - ((v - mn) / span) * (H - 2 * pad)
            coords.extend([x, y])

        # Neutral blue  -- no buy/sell connotation
        color = "#4A9EFF"
        c.create_line(coords, fill=color, width=2, smooth=True)

        # Current price dot at right end
        c.create_oval(coords[-2] - 3, coords[-1] - 3,
                      coords[-2] + 3, coords[-1] + 3,
                      fill=color, outline="")

        # Active trade: horizontal entry line (persists until exit)
        if self._active_entry_price is not None and mn <= self._active_entry_price <= mx:
            _ey = H - pad - ((self._active_entry_price - mn) / span) * (H - 2 * pad)
            _ec = "#00FF00" if self._active_entry_side == 'long' else "#FF4444"
            c.create_line(pad, _ey, W - pad, _ey, fill=_ec, width=1, dash=(4, 3))
            c.create_text(pad + 4, _ey - 8,
                          text=f"{self._active_entry_side.upper()} @ {self._active_entry_price:,.2f}",
                          fill=_ec, font=("Consolas", 6), anchor="w")

        # Trade markers: entry = up triangle (direction), exit = down triangle (outcome)
        n_pts = len(pts)
        for (idx, action, side, mprice, mpnl) in self._trade_markers:
            if idx < 0 or idx >= n_pts:
                continue
            mx_pos = pad + idx / max(1, n_pts - 1) * (W - 2 * pad)
            my_pos = H - pad - ((mprice - mn) / span) * (H - 2 * pad)
            sz = 5
            if action == 'entry':
                # Up triangle: green=LONG, red=SHORT
                mc = "#00FF00" if side == 'long' else "#FF4444"
                c.create_polygon(mx_pos, my_pos - sz, mx_pos - sz, my_pos + sz,
                                 mx_pos + sz, my_pos + sz, fill=mc, outline="#000")
            elif action == 'skip':
                # Yellow triangle: up=LONG skip, down=SHORT skip
                sz = 3
                if side == 'long':
                    c.create_polygon(mx_pos, my_pos - sz, mx_pos - sz, my_pos + sz,
                                     mx_pos + sz, my_pos + sz, fill="#FFAA00", outline="")
                else:
                    c.create_polygon(mx_pos, my_pos + sz, mx_pos - sz, my_pos - sz,
                                     mx_pos + sz, my_pos - sz, fill="#FFAA00", outline="")
            elif action == 'peak_skip':
                # Purple triangle: up=LONG skip, down=SHORT skip
                sz = 3
                if side == 'long':
                    c.create_polygon(mx_pos, my_pos - sz, mx_pos - sz, my_pos + sz,
                                     mx_pos + sz, my_pos + sz, fill="#CC44FF", outline="")
                else:
                    c.create_polygon(mx_pos, my_pos + sz, mx_pos - sz, my_pos - sz,
                                     mx_pos + sz, my_pos - sz, fill="#CC44FF", outline="")
            elif action == 'peak_entry':
                # Purple up triangle: peak detection entry
                mc = "#CC44FF"
                c.create_polygon(mx_pos, my_pos - sz, mx_pos - sz, my_pos + sz,
                                 mx_pos + sz, my_pos + sz, fill=mc, outline="#000")
            else:
                # Down triangle: green=win, red=loss
                mc = "#00FF00" if mpnl and mpnl > 0 else "#FF4444"
                c.create_polygon(mx_pos, my_pos + sz, mx_pos - sz, my_pos - sz,
                                 mx_pos + sz, my_pos - sz, fill=mc, outline="#000")

    def _save_chart_data(self):
        """Save price history + trade markers to CSV at end of phase."""
        import os, csv
        from datetime import datetime
        _mode = self._current_mode if hasattr(self, '_current_mode') else 'is'
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _dir = os.path.join('reports', _mode)
        os.makedirs(_dir, exist_ok=True)

        # Save trade markers
        _markers_path = os.path.join(_dir, f'chart_markers_{_ts}.csv')
        try:
            with open(_markers_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['idx', 'action', 'side', 'price', 'pnl'])
                for m in self._trade_markers:
                    w.writerow(m)
            print(f"  [CHART] Markers saved: {_markers_path} ({len(self._trade_markers)} markers)")
        except Exception as e:
            print(f"  [CHART] Marker save failed: {e}")

        # Save price history
        _price_path = os.path.join(_dir, f'chart_prices_{_ts}.csv')
        try:
            with open(_price_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['idx', 'price'])
                for i, p in enumerate(self._price_history):
                    w.writerow([i, p])
            print(f"  [CHART] Prices saved: {_price_path} ({len(self._price_history)} points)")
        except Exception as e:
            print(f"  [CHART] Price save failed: {e}")

        # Save canvas as PostScript -> convert if possible
        try:
            _ps_path = os.path.join(_dir, f'chart_{_mode}_{_ts}.ps')
            self._price_canvas.postscript(file=_ps_path, colormode='color')
            self._canvas.postscript(file=_ps_path.replace('chart_', 'pnl_'), colormode='color')
            print(f"  [CHART] Canvas saved: {_ps_path}")
        except Exception as e:
            print(f"  [CHART] Canvas save failed: {e}")

    def _redraw_dmi_chart(self):
        """Draw DMI+/DMI- overlay chart."""
        c = self._dmi_canvas
        c.delete("all")
        W = max(100, c.winfo_width())
        H = max(30, c.winfo_height())
        pad = 4

        dp = self._dmi_plus_history
        dm = self._dmi_minus_history
        n = min(len(dp), len(dm))
        if n < 2:
            c.create_text(W // 2, H // 2, text="Waiting for DMI...",
                          fill="#444444", font=("Consolas", 7))
            return

        # Auto-scale to data range, enforce minimum span for visibility
        all_vals = dp[-n:] + dm[-n:]
        mn = max(0, min(all_vals) - 2)
        mx = max(all_vals) + 2
        # Minimum span of 40 so lines don't look flat when DMI is stable
        MIN_SPAN = 40.0
        span = mx - mn if mx != mn else 1.0
        if span < MIN_SPAN:
            mid = (mn + mx) / 2
            mn = max(0, mid - MIN_SPAN / 2)
            mx = mn + MIN_SPAN
            span = MIN_SPAN

        def _y(v):
            return H - pad - ((v - mn) / span) * (H - 2 * pad)

        # Grid: 20 and 30 reference lines
        for ref in (20, 30):
            if mn <= ref <= mx:
                gy = _y(ref)
                c.create_line(pad, gy, W - pad, gy, fill="#282828", dash=(2, 4))
                c.create_text(W - pad + 2, gy, text=str(ref),
                              fill="#444444", font=("Consolas", 5), anchor="w")

        # DMI+ line (green)
        coords_p = []
        for i in range(n):
            x = pad + i / max(1, n - 1) * (W - 2 * pad)
            coords_p.extend([x, _y(dp[len(dp) - n + i])])
        if len(coords_p) >= 4:
            c.create_line(coords_p, fill="#00CC00", width=1, smooth=True)

        # DMI- line (red)
        coords_m = []
        for i in range(n):
            x = pad + i / max(1, n - 1) * (W - 2 * pad)
            coords_m.extend([x, _y(dm[len(dm) - n + i])])
        if len(coords_m) >= 4:
            c.create_line(coords_m, fill="#CC0000", width=1, smooth=True)

        # Current values at right edge
        if dp and dm:
            c.create_text(W - 2, _y(dp[-1]) - 6, text=f"+{dp[-1]:.0f}",
                          fill="#00CC00", font=("Consolas", 6), anchor="e")
            c.create_text(W - 2, _y(dm[-1]) + 6, text=f"-{dm[-1]:.0f}",
                          fill="#CC0000", font=("Consolas", 6), anchor="e")

        # Overlay trade markers on DMI chart (same indices as price chart)
        # Shows where entries/exits/skips occurred relative to DMI state
        n_pts = len(self._price_history)  # markers indexed to price history
        _mid_y = H // 2  # place markers at vertical center of DMI chart
        for (idx, action, side, mprice, mpnl) in self._trade_markers:
            if idx < 0 or idx >= n_pts:
                continue
            mx = pad + idx / max(1, n_pts - 1) * (W - 2 * pad)
            sz = 3
            if action == 'entry':
                mc = "#00FF00" if side == 'long' else "#FF4444"
                c.create_line(mx, pad, mx, H - pad, fill=mc, width=1, dash=(1, 3))
            elif action == 'peak_entry':
                c.create_line(mx, pad, mx, H - pad, fill="#CC44FF", width=1, dash=(1, 3))
            elif action == 'peak_skip':
                c.create_polygon(mx, _mid_y - sz, mx - sz, _mid_y,
                                 mx, _mid_y + sz, mx + sz, _mid_y,
                                 fill="#CC44FF", outline="")
            elif action == 'skip':
                c.create_polygon(mx, _mid_y - sz, mx - sz, _mid_y,
                                 mx, _mid_y + sz, mx + sz, _mid_y,
                                 fill="#FFAA00", outline="")
            elif action == 'exit':
                mc = "#00FF00" if mpnl and mpnl > 0 else "#FF4444"
                c.create_line(mx, pad, mx, H - pad, fill=mc, width=1, dash=(2, 2))

    def _on_aggression_change(self, val):
        """Slider callback  -- update shared state so engine reads it."""
        v = int(val) / 100.0
        if self._shared_state is not None:
            self._shared_state['aggression'] = v
        labels = {0: "SNIPER", 25: "CAUTIOUS", 50: "BALANCED",
                  75: "AGGRESSIVE", 100: "YOLO"}
        nearest = min(labels.keys(), key=lambda k: abs(k - int(val)))
        self._agg_label_var.set(f"Aggression: {int(val)}% ({labels[nearest]})")

    def _manual_order(self, action: str):
        """BUY/SELL/FLATTEN button callback  -- engine picks it up instantly.

        When in position, BUY/SELL visually show FLIP but still send the
        original action. Engine handles flatten + re-enter automatically.
        """
        if self._shared_state is not None:
            self._shared_state['manual_order'] = action
            self._status_var.set(f"{action} sent...")

    def _take_screenshot(self):
        """Capture full dashboard window as PNG  -- uses Win32 API for exact bounds."""
        import os
        from datetime import datetime
        _dir = os.path.join('reports', 'screenshots')
        os.makedirs(_dir, exist_ok=True)
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _mode = self._current_mode if hasattr(self, '_current_mode') else 'live'
        _path = os.path.join(_dir, f'{_mode}_{_ts}.png')
        try:
            from PIL import ImageGrab
            self.root.update_idletasks()
            # Get the HWND and use it for exact window rect (handles DPI scaling)
            img = ImageGrab.grab(all_screens=True)
            img.save(_path)
            self._status_var.set(f"Screenshot: {_path}")
            print(f"  [SCREENSHOT] {_path}")
        except ImportError:
            _ps = _path.replace('.png', '.ps')
            self._price_canvas.postscript(file=_ps, colormode='color')
            self._status_var.set(f"Screenshot (PS): {_ps}")
            print(f"  [SCREENSHOT] {_ps} (install Pillow for PNG)")
        except Exception as e:
            self._status_var.set(f"Screenshot failed: {e}")

    def _request_save(self):
        """SAVE button  -- ask engine to prepare for shutdown."""
        if self._shared_state is not None:
            self._shared_state['prepare_shutdown'] = True
            self._status_var.set("Preparing for shutdown...")

    def _toggle_ping_pong(self):
        """Toggle ping-pong mode on/off."""
        if self._shared_state is None:
            return
        current = self._shared_state.get('ping_pong', False)
        new_val = not current
        self._shared_state['ping_pong'] = new_val
        self._pp_btn.config(
            bg="#008800" if new_val else "#444444",
            activebackground="#00cc00" if new_val else "#666666",
        )
        self._status_var.set(f"Ping-pong: {'ON' if new_val else 'OFF'}")

    # ── Queue polling ─────────────────────────────────────────────────────────
    def _poll(self):
        # Detect mode change (IS -> OOS) and reset chart
        if self._shared_state is not None:
            _new_mode = self._shared_state.get('mode', self._current_mode)
            if _new_mode != self._current_mode:
                self._current_mode = _new_mode
                self._pnl_history = [0]
                self._price_history = []
                self._trade_markers = []
                self._active_entry_price = None
                self._active_entry_side = None
                self._dmi_plus_history = []
                self._dmi_minus_history = []
                self._redraw_chart()
                self._redraw_price_chart()
                self.root.title(f"Bayesian-AI  {_new_mode.upper()} Training")
        try:
            _max_msgs = 100  # process many messages per cycle (redraw only once at end)
            _msg_count = 0
            while _msg_count < _max_msgs:
                msg = self.q.get_nowait()
                _msg_count += 1
                mtype = msg.get("type", "")
                if mtype == "PHASE_PROGRESS":
                    phase = msg.get("phase", "")
                    step = msg.get("step", "")
                    pct = float(msg.get("pct", 0))
                    pnl = msg.get("pnl")
                    trades = msg.get("trades")
                    wr = msg.get("wr")

                    # Derive a clean phase label and a day/detail sub-line
                    import re as _re

                    _day_m = _re.search(r"day\s+(\d+)/(\d+)", step, _re.I)
                    _lvl_m = _re.search(r"lvl\s+(\d+)/(\d+)", step, _re.I)
                    _tmpl_m = _re.search(r"tmpl\s+(\d+)/(\d+)", step, _re.I)
                    _mode_tag = self._current_mode.upper() if hasattr(self, '_current_mode') else 'IS'
                    if _day_m:
                        _cur, _tot = int(_day_m.group(1)), int(_day_m.group(2))
                        phase_label = f"{_mode_tag} FORWARD PASS"
                        detail = f"Day {_cur} / {_tot}"
                    elif step == "FORWARD_PASS COMPLETE":
                        phase_label = f"{_mode_tag} FORWARD PASS"
                        detail = "Complete"
                    elif step == "FORWARD_PASS":
                        phase_label = f"{_mode_tag} FORWARD PASS"
                        detail = "Starting..."
                    elif _lvl_m:
                        phase_label = "PATTERN DISCOVERY"
                        detail = f"TF {_lvl_m.group(1)} / {_lvl_m.group(2)}"
                    elif step == "CLUSTERING":
                        phase_label = "CLUSTERING"
                        detail = "Building templates..."
                    elif _tmpl_m:
                        phase_label = "OPTIMIZATION"
                        detail = f"Tmpl {_tmpl_m.group(1)} / {_tmpl_m.group(2)}"
                    elif step == "STRATEGY_SELECTION":
                        phase_label = "STRATEGY SELECTION"
                        detail = ""
                    else:
                        phase_label = phase or step
                        detail = step if phase else ""

                    self._phase_var.set(phase_label)
                    self._step_var.set(detail)
                    self._pbar["value"] = pct
                    # Bar label: trade life decay or entry belief
                    if step and step.startswith('WARN'):
                        # Manual trade against belief  -- flash warning
                        self._pct_var.set(step)
                        self._pct_lbl.config(fg="#ff4444")
                    elif step and step.startswith('life'):
                        # In position  -- trade life decaying 100% -> 0%
                        self._pct_var.set(step)
                        _clr = (FG_GREEN if pct >= 60 else
                                "#ffaa00" if pct >= 30 else FG_RED)
                        self._pct_lbl.config(fg=_clr)
                    elif step and step.startswith('belief'):
                        # Flat  -- entry belief charging up
                        self._pct_var.set(step)
                        _clr = ("#ffaa00" if pct >= 60 else
                                "#888888" if pct >= 20 else "#444444")
                        self._pct_lbl.config(fg=_clr)
                    else:
                        self._pct_var.set(step if step else f"{pct:.1f}%")
                        self._pct_lbl.config(fg=FG_WHITE)

                    if pnl is not None:
                        sign = "+" if pnl >= 0 else ""
                        self._pnl_var.set(f"{sign}${pnl:,.0f}")
                        self._pnl_lbl.config(fg=FG_GREEN if pnl >= 0 else FG_RED)
                        self._pnl_history.append(pnl)
                        self._redraw_chart()
                    if wr is not None:
                        self._wr_var.set(f"{wr:.1f}%")
                    if trades is not None:
                        self._trades_var.set(f"{trades:,}")

                    # Profit factor & gross W/L
                    _pf = msg.get('pf')
                    _gw = msg.get('gross_w')
                    _gl = msg.get('gross_l')
                    if _pf is not None:
                        self._pf_var.set(f"{_pf:.2f}")
                        self._pf_lbl.config(fg=FG_GREEN if _pf >= 1.0 else FG_RED)
                    if _gw is not None:
                        self._gw_var.set(f"+${_gw:,.0f}")
                    if _gl is not None:
                        self._gl_var.set(f"-${_gl:,.0f}")

                    # Capture rate quartile buckets
                    _c_rev = msg.get('cap_reversed', 0)
                    _c_q1 = msg.get('cap_q1', 0)
                    _c_q2 = msg.get('cap_q2', 0)
                    _c_q3 = msg.get('cap_q3', 0)
                    _c_q4 = msg.get('cap_q4', 0)
                    _c_plus = msg.get('cap_100plus', 0)
                    _c_tot = _c_rev + _c_q1 + _c_q2 + _c_q3 + _c_q4 + _c_plus
                    if _c_tot > 0:
                        self._cap_rev_var.set(f"{_c_rev}  ({_c_rev/_c_tot*100:.0f}%)")
                        self._cap_q1_var.set(f"{_c_q1}  ({_c_q1/_c_tot*100:.0f}%)")
                        self._cap_q2_var.set(f"{_c_q2}  ({_c_q2/_c_tot*100:.0f}%)")
                        self._cap_q3_var.set(f"{_c_q3}  ({_c_q3/_c_tot*100:.0f}%)")
                        self._cap_q4_var.set(f"{_c_q4}  ({_c_q4/_c_tot*100:.0f}%)")
                        self._cap_plus_var.set(f"{_c_plus}  ({_c_plus/_c_tot*100:.0f}%)")

                    if step == "FORWARD_PASS COMPLETE":
                        self._done = True
                        self._status_var.set("COMPLETE  -- close window when ready")
                        self._pct_var.set("100%")
                        self.root.attributes("-topmost", False)
                        # Save chart data + screenshot
                        self._save_chart_data()

                elif mtype == "ACCOUNT_UPDATE":
                    cash = float(msg.get("cash_value", 0))
                    unreal = float(msg.get("unrealized_pnl", 0))
                    netliq = float(msg.get("net_liquidation", 0))

                    # Update equity labels (live mode only)
                    if hasattr(self, '_cash_var'):
                        self._cash_var.set(f"${cash:,.0f}")
                        sign_u = "+" if unreal >= 0 else ""
                        self._unreal_var.set(f"{sign_u}${unreal:,.0f}")
                        self._unreal_lbl.config(
                            fg=FG_GREEN if unreal >= 0 else FG_RED)
                        self._netliq_var.set(f"${netliq:,.0f}")
                        self._netliq_lbl.config(
                            fg=FG_GREEN if unreal >= 0 else FG_BLUE)

                elif mtype == "TICK_UPDATE":
                    price = msg.get("price")
                    if price is not None:
                        p = float(price)
                        self._price_var.set(f"{p:,.2f}")
                        # Flash green/red
                        prev = getattr(self, '_prev_price', None)
                        if prev is not None:
                            self._price_lbl.config(
                                fg=FG_GREEN if p >= prev else FG_RED)
                        self._prev_price = p
                        # Feed price chart
                        self._price_history.append(p)
                        if len(self._price_history) > self._MAX_PRICE_PTS:
                            trim = len(self._price_history) - self._MAX_PRICE_PTS
                            self._price_history = self._price_history[-self._MAX_PRICE_PTS:]
                            # Shift marker indices and drop expired ones
                            self._trade_markers = [
                                (idx - trim, a, s, mp, mpnl)
                                for (idx, a, s, mp, mpnl) in self._trade_markers
                                if idx - trim >= 0
                            ]
                        self._needs_redraw = True

                    # Feed DMI chart
                    _dp = msg.get('dmi_plus')
                    _dm = msg.get('dmi_minus')
                    if _dp is not None and _dm is not None:
                        self._dmi_plus_history.append(float(_dp))
                        self._dmi_minus_history.append(float(_dm))
                        if len(self._dmi_plus_history) > self._MAX_DMI_PTS:
                            self._dmi_plus_history = self._dmi_plus_history[-self._MAX_DMI_PTS:]
                            self._dmi_minus_history = self._dmi_minus_history[-self._MAX_DMI_PTS:]
                        self._dmi_label_var.set(
                            f"+{float(_dp):.0f} / -{float(_dm):.0f}")
                        self._needs_redraw = True

                    # Update unrealized PnL from tick (instant, don't wait for NT8)
                    _tick_unreal = msg.get('unrealized_pnl')
                    if _tick_unreal is not None and hasattr(self, '_unreal_var'):
                        _u = float(_tick_unreal)
                        sign_u = "+" if _u >= 0 else ""
                        self._unreal_var.set(f"{sign_u}${_u:,.0f}")
                        self._unreal_lbl.config(
                            fg=FG_GREEN if _u >= 0 else FG_RED)

                elif mtype == "TRADE_MARKER":
                    action = msg.get("action", "").lower()
                    side = msg.get("side", "")
                    mprice = msg.get("price", 0)
                    mpnl = msg.get("pnl", 0)
                    # Track position side for FLIP routing + entry line
                    if action == 'entry' or action == 'ENTRY':
                        self._active_side = side
                        self._active_entry_price = mprice
                        self._active_entry_side = side
                    elif action == 'exit' or action == 'EXIT':
                        self._active_side = None
                        self._active_entry_price = None
                        self._active_entry_side = None
                    # Update button states: FLAT shows position, BUY/SELL become FLIP
                    if hasattr(self, '_flatten_btn'):
                        if action == 'entry':
                            _clr = "#006600" if side == 'long' else "#aa0000"
                            self._flatten_btn.config(
                                text=f"FLAT {side.upper()}", bg=_clr,
                                activebackground=_clr)
                            # Highlight the FLIP direction, dim the same-dir button
                            if side == 'long':
                                # SELL = flip to SHORT (highlighted)
                                self._sell_btn.config(
                                    text="FLIP SHORT", bg="#aa5500",
                                    activebackground="#cc6600")
                                self._buy_btn.config(
                                    text="BUY", bg="#333333",
                                    activebackground="#555555")
                            else:
                                # BUY = flip to LONG (highlighted)
                                self._buy_btn.config(
                                    text="FLIP LONG", bg="#aa5500",
                                    activebackground="#cc6600")
                                self._sell_btn.config(
                                    text="SELL", bg="#333333",
                                    activebackground="#555555")
                        elif action == 'exit':
                            self._flatten_btn.config(
                                text="FLAT", bg="#333333",
                                activebackground="#555555")
                            self._buy_btn.config(
                                text="BUY", bg="#006600",
                                activebackground="#00aa00")
                            self._sell_btn.config(
                                text="SELL", bg="#880000",
                                activebackground="#cc0000")
                    # Store marker at current price_history index
                    idx = len(self._price_history) - 1
                    if idx >= 0:
                        self._trade_markers.append((idx, action, side, mprice, mpnl))
                        # Trim markers that fell off the rolling window
                        cutoff = len(self._price_history) - self._MAX_PRICE_PTS
                        if cutoff > 0:
                            self._trade_markers = [
                                m for m in self._trade_markers if m[0] >= cutoff
                            ]
                        self._needs_redraw = True

                elif mtype == "SHUTDOWN_READY":
                    # Engine reports whether it's safe to close
                    status = msg.get("status", "unknown")
                    self._status_var.set(status)

                elif mtype == "SHUTDOWN":
                    self.root.quit()  # break mainloop; launcher handles cleanup
                    return
        except queue.Empty:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        # Single redraw per poll cycle (not per message)
        if getattr(self, '_needs_redraw', False):
            self._redraw_price_chart()
            self._redraw_dmi_chart()
            self._needs_redraw = False
        self.root.after(1000, self._poll)  # poll every 1 second


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


if __name__ == "__main__":
    import threading, time

    q = queue.Queue()

    def _sim():
        time.sleep(1)
        q.put({"type": "STATUS", "text": "SCANNING ATLAS..."})
        q.put(
            {
                "type": "PHASE_PROGRESS",
                "phase": "Analyze",
                "step": "PATTERN_DISCOVERY",
                "pct": 20,
            }
        )
        time.sleep(1)
        for i, (tid, z, m, pnl, wr) in enumerate(
            [
                (150, 1.8, 4.2, 5016, 0.57),
                (391, -2.1, -3.8, 5177, 0.55),
                (463, 2.4, 5.1, 4763, 0.56),
                (173, -1.6, -2.9, 3265, 0.66),
            ]
        ):
            q.put(
                {
                    "type": "TEMPLATE_UPDATE",
                    "id": tid,
                    "z": z,
                    "mom": m,
                    "pnl": pnl,
                    "count": 300 + i * 50,
                    "win_rate": wr,
                }
            )
            time.sleep(0.3)
        q.put(
            {
                "type": "PHASE_PROGRESS",
                "phase": "Analyze",
                "step": "FORWARD_PASS",
                "pct": 65,
            }
        )
        time.sleep(1)
        q.put(
            {
                "type": "ORACLE_ATTRIBUTION",
                "ideal": 842400,
                "actual": 18661,
                "missed": 620000,
                "too_early": 124800,
                "wrong_dir": 28110,
                "noise": 5568,
            }
        )
        q.put(
            {
                "type": "PHASE_PROGRESS",
                "phase": "Analyze",
                "step": "COMPLETE",
                "pct": 100,
            }
        )
        time.sleep(1)
        q.put(
            {
                "type": "FISSION_EVENT",
                "parent_id": 150,
                "children_count": 3,
                "reason": "Variance",
            }
        )

    threading.Thread(target=_sim, daemon=True).start()
    launch_dashboard(q)
