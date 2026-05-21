"""
Trading Dashboard V2 — clean, physics-only, real-time.

Layout:
  ┌──────────────────────────────────────────────────┐
  │ STATUS BAR: price | z | vr | position | daily PnL│
  ├────────────────────────────┬─────────────────────┤
  │                            │  SESSION STATS       │
  │   1m CANDLESTICK CHART     │  $/day  trades  WR   │
  │   + regression bands       │  tier breakdown      │
  │   + entry/exit markers     │                     │
  │                            │  TRADE LOG           │
  │                            │  last 10 trades      │
  ├────────────────────────────┼─────────────────────┤
  │   PnL EQUITY CURVE         │  CONTROLS            │
  │   cumulative daily PnL     │  FLATTEN  BUY  SELL  │
  └────────────────────────────┴─────────────────────┘
"""
import tkinter as tk
from tkinter import ttk
import queue
import time
import numpy as np
from collections import deque

# Colors
BG = '#0d1117'
BG_CARD = '#161b22'
FG = '#c9d1d9'
GREEN = '#3fb950'
RED = '#f85149'
BLUE = '#58a6ff'
AMBER = '#d29922'
GREY = '#484f58'
WHITE = '#ffffff'
CYAN = '#39d2c0'


class TradingDashboard:
    """Clean trading dashboard for physics-only engine."""

    def __init__(self, root, gui_queue, shared_state=None):
        self.root = root
        self.queue = gui_queue
        self.shared_state = shared_state or {}

        self.root.title('BAYESIAN-AI v2')
        # Geometry is restored from disk in _load_geometry() (called below).
        self.root.configure(bg=BG)
        self.root.minsize(900, 500)

        # Data
        self.prices = deque(maxlen=500)
        self.z_history = deque(maxlen=500)
        self.pnl_curve = deque(maxlen=500)
        self.trades = []  # last N trades
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.last_price = 0.0
        self.last_z = 0.0
        self.last_vr = 0.0
        self.in_position = False
        self.direction = ''
        self.tier = ''
        self.entry_price = 0.0
        self.unrealized = 0.0
        self.account_size = 0.0
        self.account_realized = 0.0
        self.account_unrealized = 0.0
        # Engine state tracking
        self.engine_state = 'INIT'
        self.bar_count = 0
        self.last_bar_ts = 0
        self.bar_rate = 0.0  # bars/min
        self.activity = ''
        # Recent NT8 fill display
        self.last_nt8_fill_text = ''

        # L5 engine state (pushed by engine via 'L5_STATE' messages)
        self.l5_r_price = None        # zigzag reversal threshold (points)
        self.l5_zz_dir = None         # 'up' / 'down'
        self.l5_zz_extreme = None     # running extreme price
        self.l5_b10_mode = None       # 'normal' / 'cautious'
        self.l5_last_b7 = None        # last B7 pred_R
        self.l5_last_b9 = None        # last B9 pred

        # Skipped-signal tracking: an R-trigger fired but no entry taken.
        # skip_flags is a parallel deque to `prices` -- each bar gets a
        # None (no skip) or a direction string (skipped). They slide in
        # lockstep so chart markers stay aligned.
        self.skip_flags = deque(maxlen=500)
        self.skip_count = 0
        self.last_skip_text = ''

        # Geometry persistence + debounced autosave
        self._geom_save_job = None
        self._load_geometry()

        self._build_ui()
        self.root.bind('<Configure>', self._on_configure)
        self.root.after(100, self._poll_queue)

    # ── Window geometry persistence ──

    _GEOMETRY_FILE = 'live/state/dashboard_geometry.json'

    def _load_geometry(self):
        """Restore window position + size from the last session."""
        import os, json
        try:
            if os.path.exists(self._GEOMETRY_FILE):
                with open(self._GEOMETRY_FILE) as f:
                    geom = json.load(f).get('geometry')
                if geom:
                    self.root.geometry(geom)
                    return
        except Exception:
            pass
        self.root.geometry('1200x700')   # default

    def _on_configure(self, event):
        """Debounced geometry save -- fires 800ms after the last move/resize."""
        if event.widget is not self.root:
            return
        if self._geom_save_job is not None:
            self.root.after_cancel(self._geom_save_job)
        self._geom_save_job = self.root.after(800, self._save_geometry)

    def _save_geometry(self):
        import os, json
        self._geom_save_job = None
        try:
            os.makedirs(os.path.dirname(self._GEOMETRY_FILE), exist_ok=True)
            with open(self._GEOMETRY_FILE, 'w') as f:
                json.dump({'geometry': self.root.geometry()}, f)
        except Exception:
            pass

    # ── Screen capture ──

    def _take_screenshot(self):
        """Capture the dashboard window as a PNG (reports/screenshots/)."""
        import os
        from datetime import datetime
        out_dir = os.path.join('reports', 'screenshots')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out_dir, f'dashboard_{ts}.png')
        try:
            from PIL import ImageGrab
            self.root.update_idletasks()
            # Exact window bounds (handles multi-monitor + DPI)
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
            img.save(path)
            if hasattr(self, 'lbl_save_status'):
                self.lbl_save_status.config(text=f'shot: {os.path.basename(path)}')
            print(f'  [SCREENSHOT] {path}')
        except ImportError:
            print('  [SCREENSHOT] Pillow not installed -- pip install Pillow')
            if hasattr(self, 'lbl_save_status'):
                self.lbl_save_status.config(text='shot: need Pillow')
        except Exception as e:
            print(f'  [SCREENSHOT] failed: {e}')
            if hasattr(self, 'lbl_save_status'):
                self.lbl_save_status.config(text=f'shot failed')

    def _build_ui(self):
        """Build the dashboard layout."""
        # ── Status bar ──
        status_frame = tk.Frame(self.root, bg=BG_CARD, height=50)
        status_frame.pack(fill=tk.X, padx=4, pady=(4, 2))
        status_frame.pack_propagate(False)

        self.lbl_price = tk.Label(status_frame, text='--', font=('Consolas', 20, 'bold'),
                                  bg=BG_CARD, fg=WHITE)
        self.lbl_price.pack(side=tk.LEFT, padx=15)

        self.lbl_z = tk.Label(status_frame, text='z: --', font=('Consolas', 13),
                              bg=BG_CARD, fg=CYAN)
        self.lbl_z.pack(side=tk.LEFT, padx=10)

        self.lbl_vr = tk.Label(status_frame, text='vr: --', font=('Consolas', 13),
                               bg=BG_CARD, fg=CYAN)
        self.lbl_vr.pack(side=tk.LEFT, padx=10)

        self.lbl_position = tk.Label(status_frame, text='FLAT', font=('Consolas', 14, 'bold'),
                                     bg=BG_CARD, fg=GREY)
        self.lbl_position.pack(side=tk.LEFT, padx=20)

        self.lbl_daily = tk.Label(status_frame, text='$0', font=('Consolas', 18, 'bold'),
                                  bg=BG_CARD, fg=GREEN)
        self.lbl_daily.pack(side=tk.RIGHT, padx=15)

        self.lbl_unrealized = tk.Label(status_frame, text='', font=('Consolas', 12),
                                       bg=BG_CARD, fg=GREY)
        self.lbl_unrealized.pack(side=tk.RIGHT, padx=5)

        # ── Engine health bar (state + bar flow) ──
        health_frame = tk.Frame(self.root, bg=BG_CARD, height=24)
        health_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        health_frame.pack_propagate(False)

        self.lbl_state = tk.Label(health_frame, text='● INIT',
                                   font=('Consolas', 10, 'bold'),
                                   bg=BG_CARD, fg=AMBER)
        self.lbl_state.pack(side=tk.LEFT, padx=10)

        self.lbl_bar_flow = tk.Label(health_frame,
                                      text='bars: 0  |  last: --  |  rate: --',
                                      font=('Consolas', 10),
                                      bg=BG_CARD, fg=GREY)
        self.lbl_bar_flow.pack(side=tk.LEFT, padx=10)

        self.lbl_activity = tk.Label(health_frame, text='',
                                      font=('Consolas', 10),
                                      bg=BG_CARD, fg=CYAN)
        self.lbl_activity.pack(side=tk.RIGHT, padx=10)

        # ── Body: left (chart area) + right (stats + trades + controls) ──
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # Left: price chart + equity curve
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Price chart canvas (z overlay drawn on same canvas, right axis)
        self.price_canvas = tk.Canvas(left, bg=BG_CARD, highlightthickness=0)
        self.price_canvas.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 2))

        # Equity curve canvas
        self.equity_canvas = tk.Canvas(left, bg=BG_CARD, highlightthickness=0, height=120)
        self.equity_canvas.pack(fill=tk.X, padx=(0, 2))

        # Right panel
        right = tk.Frame(body, bg=BG, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # NT8 Account card
        acct_card = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        acct_card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(acct_card, text='ACCOUNT (NT8)', font=('Consolas', 10, 'bold'),
                 bg=BG_CARD, fg=GREY).pack(anchor=tk.W)

        self.lbl_acct_size = tk.Label(acct_card, text='$0', font=('Consolas', 14, 'bold'),
                                       bg=BG_CARD, fg=WHITE)
        self.lbl_acct_size.pack(anchor=tk.W)

        acct_row = tk.Frame(acct_card, bg=BG_CARD)
        acct_row.pack(fill=tk.X, pady=2)
        self.lbl_acct_realized = tk.Label(acct_row, text='Day $0',
                                           font=('Consolas', 11),
                                           bg=BG_CARD, fg=GREEN)
        self.lbl_acct_realized.pack(side=tk.LEFT)
        self.lbl_acct_unrealized = tk.Label(acct_row, text='Unrl $0',
                                             font=('Consolas', 11),
                                             bg=BG_CARD, fg=GREY)
        self.lbl_acct_unrealized.pack(side=tk.RIGHT)

        # Session stats card
        stats_card = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        stats_card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(stats_card, text='SESSION', font=('Consolas', 10, 'bold'),
                 bg=BG_CARD, fg=GREY).pack(anchor=tk.W)

        self.lbl_stats_pnl = tk.Label(stats_card, text='$0/day', font=('Consolas', 16, 'bold'),
                                       bg=BG_CARD, fg=GREEN)
        self.lbl_stats_pnl.pack(anchor=tk.W)

        stats_row = tk.Frame(stats_card, bg=BG_CARD)
        stats_row.pack(fill=tk.X, pady=4)
        self.lbl_trades = tk.Label(stats_row, text='0 trades', font=('Consolas', 11),
                                   bg=BG_CARD, fg=FG)
        self.lbl_trades.pack(side=tk.LEFT)
        self.lbl_wr = tk.Label(stats_row, text='0% WR', font=('Consolas', 11),
                               bg=BG_CARD, fg=FG)
        self.lbl_wr.pack(side=tk.RIGHT)

        # Tier breakdown
        tier_card = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        tier_card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(tier_card, text='TIERS', font=('Consolas', 10, 'bold'),
                 bg=BG_CARD, fg=GREY).pack(anchor=tk.W)

        self.tier_text = tk.Text(tier_card, bg=BG_CARD, fg=FG, font=('Consolas', 9),
                                 height=6, wrap=tk.NONE, borderwidth=0, highlightthickness=0)
        self.tier_text.pack(fill=tk.X)
        self.tier_text.config(state=tk.DISABLED)

        # Trade log
        log_card = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        log_card.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        tk.Label(log_card, text='TRADE LOG', font=('Consolas', 10, 'bold'),
                 bg=BG_CARD, fg=GREY).pack(anchor=tk.W)

        self.log_text = tk.Text(log_card, bg=BG_CARD, fg=FG, font=('Consolas', 9),
                                height=12, wrap=tk.NONE, borderwidth=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        self.log_text.tag_configure('win', foreground=GREEN)
        self.log_text.tag_configure('loss', foreground=RED)

        # Controls
        ctrl_card = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        ctrl_card.pack(fill=tk.X)

        btn_frame = tk.Frame(ctrl_card, bg=BG_CARD)
        btn_frame.pack(fill=tk.X)

        self.btn_flatten = tk.Button(btn_frame, text='FLATTEN', bg='#b62324', fg=WHITE,
                                     font=('Consolas', 11, 'bold'), width=10,
                                     command=self._on_flatten)
        self.btn_flatten.pack(side=tk.LEFT, padx=2)

        self.btn_buy = tk.Button(btn_frame, text='BUY', bg='#1a7f37', fg=WHITE,
                                 font=('Consolas', 11, 'bold'), width=8,
                                 command=self._on_buy)
        self.btn_buy.pack(side=tk.LEFT, padx=2)

        self.btn_sell = tk.Button(btn_frame, text='SELL', bg='#b62324', fg=WHITE,
                                  font=('Consolas', 11, 'bold'), width=8,
                                  command=self._on_sell)
        self.btn_sell.pack(side=tk.LEFT, padx=2)

        # Second row — save button
        btn_row2 = tk.Frame(ctrl_card, bg=BG_CARD)
        btn_row2.pack(fill=tk.X, pady=(4, 0))

        self.btn_save = tk.Button(btn_row2, text='SAVE NOW', bg=BLUE, fg=WHITE,
                                   font=('Consolas', 11, 'bold'), width=10,
                                   command=self._on_save)
        self.btn_save.pack(side=tk.LEFT, padx=2)

        self.btn_capture = tk.Button(btn_row2, text='CAPTURE', bg=CYAN, fg=BG,
                                      font=('Consolas', 11, 'bold'), width=9,
                                      command=self._take_screenshot)
        self.btn_capture.pack(side=tk.LEFT, padx=2)

        self.btn_close = tk.Button(btn_row2, text='CLOSE', bg='#8b1a1a', fg=WHITE,
                                    font=('Consolas', 11, 'bold'), width=8,
                                    command=self._on_close_clicked)
        self.btn_close.pack(side=tk.LEFT, padx=2)

        self.lbl_save_status = tk.Label(btn_row2, text='',
                                         font=('Consolas', 9),
                                         bg=BG_CARD, fg=GREEN)
        self.lbl_save_status.pack(side=tk.LEFT, padx=5)

    # ── Queue polling ──

    def _poll_queue(self):
        """Process GUI messages from engine."""
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        # Re-enable manual buttons once engine clears the pending flag
        # (engine clears it on fill confirmation from NT8).
        if (hasattr(self, 'btn_flatten')
                and str(self.btn_flatten['state']) == 'disabled'
                and not self.shared_state.get('manual_order_pending')):
            for btn in (self.btn_flatten, self.btn_buy, self.btn_sell):
                try:
                    btn.config(state='normal')
                except Exception:
                    pass

        if not self.shared_state.get('shutdown_confirmed'):
            self.root.after(50, self._poll_queue)

    def _handle_msg(self, msg):
        msg_type = msg.get('type', '')

        if msg_type == 'ACCOUNT_UPDATE':
            self.account_size = msg.get('cash_value', 0)
            self.account_realized = msg.get('realized_pnl', 0)
            self.account_unrealized = msg.get('unrealized_pnl', 0)
            self._update_account()
            return

        if msg_type == 'ENGINE_STATE':
            self.engine_state = msg.get('state', '')
            self.bar_count = msg.get('bar_count', 0)
            self.last_bar_ts = msg.get('last_bar_ts', 0)
            self.bar_rate = msg.get('bar_rate', 0)
            self.activity = msg.get('activity', '')
            self._update_engine_state()
            return

        if msg_type == 'L5_STATE':
            # L5 engine internals -- the indicators that actually drive
            # trade decisions (zigzag/R-trigger/B7/B9/B10).
            self.l5_r_price = msg.get('r_price')
            self.l5_zz_dir = msg.get('zz_dir')
            self.l5_zz_extreme = msg.get('zz_extreme')
            self.l5_b10_mode = msg.get('b10_mode')
            if msg.get('b7_pred') is not None:
                self.l5_last_b7 = msg.get('b7_pred')
            if msg.get('b9_pred') is not None:
                self.l5_last_b9 = msg.get('b9_pred')
            return

        if msg_type == 'SIGNAL_SKIP':
            # An R-trigger fired but no entry was taken (B7 low conviction,
            # V2 warmup, etc.). Mark the current bar so the chart shows it.
            self.skip_count = msg.get('skip_count', self.skip_count + 1)
            direction = msg.get('direction', '?')
            reason = msg.get('reason', '')
            if self.skip_flags:
                self.skip_flags[-1] = direction
            pr = msg.get('pred_R')
            thr = msg.get('thr')
            if pr is not None and thr is not None:
                self.last_skip_text = f'{direction} {reason} ({pr:.2f}<{thr:.2f})'
            else:
                self.last_skip_text = f'{direction} {reason}'
            return

        if msg_type == 'TICK_UPDATE':
            price = msg.get('price', 0)
            self.last_price = price
            self.last_z = msg.get('z_se', 0)
            self.last_vr = msg.get('vr', 0)
            self.in_position = msg.get('in_position', False)
            self.direction = msg.get('direction', '')
            self.tier = msg.get('tier', '')
            self.entry_price = msg.get('entry_price', 0.0)
            self.daily_pnl = msg.get('daily_pnl', 0)
            self.unrealized = msg.get('unrealized', 0)

            self.prices.append(price)
            self.z_history.append(self.last_z)
            self.skip_flags.append(None)   # parallel to prices; SIGNAL_SKIP sets [-1]
            self._update_status()
            self._draw_price_chart()

        elif msg_type == 'NT8_TRADE':
            # Ground truth from NT8 — always display these
            side = msg.get('side', '')
            entry = msg.get('entry_price', 0)
            exit_p = msg.get('exit_price', 0)
            pnl = msg.get('pnl', 0)
            is_chain = msg.get('is_chain', False)
            self.trade_count += 1
            if pnl > 0:
                self.win_count += 1
            self.trades.append({
                'time': time.strftime('%H:%M:%S'),
                'side': side,
                'price': exit_p,
                'entry': entry,
                'pnl': pnl,
                'action': 'CHAIN' if is_chain else 'EXIT',
            })
            # Track for status bar
            sign = '+' if pnl >= 0 else ''
            tag = 'CHN' if is_chain else 'TRD'
            self.last_nt8_fill_text = (f'{tag} {side} {entry:.2f}->{exit_p:.2f} '
                                        f'{sign}${pnl:.0f}')
            self.pnl_curve.append(self.daily_pnl)
            self._update_log()
            self._draw_equity()

        elif msg_type == 'TRADE_MARKER':
            action = msg.get('action', '')
            side = msg.get('side', '')
            price = msg.get('price', 0)
            pnl = msg.get('pnl', 0)
            # ENTRY markers only — EXITs now come from NT8_TRADE
            if action == 'ENTRY':
                pass  # could annotate chart, but don't touch trade log

        elif msg_type == 'STATS':
            pass  # stats handled via TICK_UPDATE

    def _update_engine_state(self):
        """Update engine health bar — state + bar flow + activity."""
        state_colors = {
            'INIT': AMBER, 'WARMUP': AMBER, 'SYNCING': AMBER,
            'TRADING': GREEN, 'CATCH_UP': BLUE,
            'BROKER_DISCONNECTED': RED, 'STALE': RED, 'SHUTDOWN': GREY,
        }
        color = state_colors.get(self.engine_state, GREY)
        self.lbl_state.config(text=f'● {self.engine_state}', fg=color)

        # Bar flow
        if self.last_bar_ts > 0:
            t = time.strftime('%H:%M:%S', time.gmtime(self.last_bar_ts))
            rate_str = f'{self.bar_rate:.1f}/min' if self.bar_rate > 0 else '--'
            self.lbl_bar_flow.config(
                text=f'bars: {self.bar_count:,}  |  last: {t}  |  rate: {rate_str}')
        else:
            self.lbl_bar_flow.config(text=f'bars: {self.bar_count}  |  waiting...')

        # Activity: current position + last NT8 fill
        act_parts = []
        if self.activity:
            act_parts.append(self.activity)
        if self.last_nt8_fill_text:
            act_parts.append(self.last_nt8_fill_text)
        self.lbl_activity.config(text='  |  '.join(act_parts))

    def _update_account(self):
        """Update NT8 account card."""
        self.lbl_acct_size.config(text=f'${self.account_size:,.0f}')
        r_color = GREEN if self.account_realized >= 0 else RED
        u_color = GREEN if self.account_unrealized >= 0 else (
            RED if self.account_unrealized < 0 else GREY)
        self.lbl_acct_realized.config(text=f'Day ${self.account_realized:+,.0f}',
                                       fg=r_color)
        self.lbl_acct_unrealized.config(text=f'Unrl ${self.account_unrealized:+,.0f}',
                                         fg=u_color)

    def _update_status(self):
        """Update status bar."""
        self.lbl_price.config(text=f'{self.last_price:,.2f}')

        # Z color
        z = self.last_z
        z_color = RED if abs(z) > 2 else (AMBER if abs(z) > 1 else CYAN)
        self.lbl_z.config(text=f'z: {z:+.1f}', fg=z_color)

        # VR color
        vr = self.last_vr
        vr_color = GREEN if vr < 0.5 else (AMBER if vr < 1.0 else RED)
        self.lbl_vr.config(text=f'vr: {vr:.2f}', fg=vr_color)

        # Position
        if self.in_position:
            pos_text = f'{self.direction.upper()} {self.tier}'
            pos_color = GREEN if self.direction == 'long' else RED
        else:
            pos_text = 'FLAT'
            pos_color = GREY
        self.lbl_position.config(text=pos_text, fg=pos_color)

        # Daily PnL
        pnl_color = GREEN if self.daily_pnl >= 0 else RED
        self.lbl_daily.config(text=f'${self.daily_pnl:+,.0f}', fg=pnl_color)

        # Unrealized
        if self.in_position and self.unrealized != 0:
            u_color = GREEN if self.unrealized > 0 else RED
            self.lbl_unrealized.config(text=f'unrl ${self.unrealized:+,.0f}', fg=u_color)
        else:
            self.lbl_unrealized.config(text='')

        # Session stats
        self.lbl_stats_pnl.config(text=f'${self.daily_pnl:+,.0f}',
                                   fg=GREEN if self.daily_pnl >= 0 else RED)
        self.lbl_trades.config(text=f'{self.trade_count} trades')
        wr = self.win_count / self.trade_count * 100 if self.trade_count > 0 else 0
        self.lbl_wr.config(text=f'{wr:.0f}% WR')

    def _draw_price_chart(self):
        """Price chart with z_se overlay (right axis)."""
        c = self.price_canvas
        c.delete('all')
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10 or len(self.prices) < 2:
            return

        prices = list(self.prices)
        zvals = list(self.z_history) if len(self.z_history) >= 2 else None
        n = len(prices)
        p_min = min(prices)
        p_max = max(prices)
        p_range = max(p_max - p_min, 0.01)

        # Z-axis: fixed -4 to +4
        z_min, z_max = -4.0, 4.0
        z_range = z_max - z_min

        margin_l = 50   # room for price labels
        margin_r = 40   # room for z labels
        margin_t = 15
        margin_b = 15
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b

        def x_at(i):
            return margin_l + (i / max(n - 1, 1)) * chart_w

        def y_price(p):
            return margin_t + (1 - (p - p_min) / p_range) * chart_h

        def y_z(z):
            return margin_t + (1 - (z - z_min) / z_range) * chart_h

        # ── Shaded NMP bands (|z| > 2) ──
        y_p2 = y_z(2.0)
        y_n2 = y_z(-2.0)
        c.create_rectangle(margin_l, margin_t, w - margin_r, y_p2,
                           fill='#1a2332', outline='')
        c.create_rectangle(margin_l, y_n2, w - margin_r, margin_t + chart_h,
                           fill='#1a2332', outline='')
        # Threshold lines at +/-2
        c.create_line(margin_l, y_p2, w - margin_r, y_p2,
                      fill=RED, dash=(3, 3), width=1)
        c.create_line(margin_l, y_n2, w - margin_r, y_n2,
                      fill=GREEN, dash=(3, 3), width=1)
        # Zero z line
        y_zero = y_z(0)
        c.create_line(margin_l, y_zero, w - margin_r, y_zero,
                      fill='#2a2f38', dash=(2, 4), width=1)

        # ── Z line (behind price) ──
        if zvals:
            for i in range(len(zvals) - 1):
                x1 = x_at(i)
                x2 = x_at(i + 1)
                y1 = y_z(max(min(zvals[i], z_max), z_min))
                y2 = y_z(max(min(zvals[i + 1], z_max), z_min))
                color = '#d97706' if zvals[i + 1] > 0 else '#65a30d'
                c.create_line(x1, y1, x2, y2, fill=color, width=1)

        # ── Price line (on top, bright) ──
        for i in range(n - 1):
            c.create_line(x_at(i), y_price(prices[i]),
                          x_at(i + 1), y_price(prices[i + 1]),
                          fill=BLUE, width=2)

        # ── L5 zigzag overlay: running extreme + R-trigger threshold ──
        # These are the indicators that actually drive L5 entries: the
        # zigzag tracks an extreme; an R-trigger fires when price reverses
        # by r_price from it.
        if (self.l5_r_price is not None and self.l5_zz_extreme is not None
                and self.l5_zz_dir is not None):
            ext = self.l5_zz_extreme
            rp = self.l5_r_price
            # Reversal threshold: opposite side of the extreme by r_price
            if self.l5_zz_dir == 'up':      # tracking a HIGH -> SHORT trigger below
                thr = ext - rp
            else:                            # tracking a LOW -> LONG trigger above
                thr = ext + rp
            # Extreme line (amber dashed) -- draw if within visible range
            if p_min <= ext <= p_max:
                ye = y_price(ext)
                c.create_line(margin_l, ye, w - margin_r, ye,
                              fill=AMBER, dash=(5, 3), width=1)
                c.create_text(margin_l + 4, ye - 7, text='zz-extreme',
                              anchor=tk.W, fill=AMBER, font=('Consolas', 7))
            # R-trigger threshold line (cyan dashed)
            if p_min <= thr <= p_max:
                yt = y_price(thr)
                c.create_line(margin_l, yt, w - margin_r, yt,
                              fill=CYAN, dash=(5, 3), width=1)
                c.create_text(margin_l + 4, yt - 7,
                              text=f'R-trigger {thr:.2f}',
                              anchor=tk.W, fill=CYAN, font=('Consolas', 7))
            # L5 status block (bottom-left)
            mode = (self.l5_b10_mode or 'normal').upper()
            line1 = f'L5  r={rp:.1f}  dir={self.l5_zz_dir.upper()}  {mode}'
            b7s = f'{self.l5_last_b7:.2f}' if self.l5_last_b7 is not None else '--'
            b9s = (f'{self.l5_last_b9:+.1f}' if self.l5_last_b9 is not None else '--')
            line2 = f'B7 {b7s}   B9 {b9s}   skips {self.skip_count}'
            c.create_text(margin_l + 4, margin_t + chart_h - 32,
                          text=line1, anchor=tk.W, fill=CYAN,
                          font=('Consolas', 8, 'bold'))
            c.create_text(margin_l + 4, margin_t + chart_h - 20,
                          text=line2, anchor=tk.W, fill=FG,
                          font=('Consolas', 8))
            if self.last_skip_text:
                c.create_text(margin_l + 4, margin_t + chart_h - 8,
                              text=f'last skip: {self.last_skip_text}',
                              anchor=tk.W, fill=AMBER, font=('Consolas', 8))

        # ── Skipped-signal markers ──
        # An R-trigger fired here but no entry was taken. Hollow amber
        # triangle pointing the would-be trade direction (down=short skip,
        # up=long skip). skip_flags is parallel to `prices`.
        skips = list(self.skip_flags)
        for i, sk in enumerate(skips):
            if sk is None or i >= n:
                continue
            xs = x_at(i)
            ys = y_price(prices[i])
            if sk == 'short':   # would-be short -> triangle points down
                pts = [xs - 4, ys - 7, xs + 4, ys - 7, xs, ys - 1]
            else:               # long (or '?') -> triangle points up
                pts = [xs - 4, ys + 7, xs + 4, ys + 7, xs, ys + 1]
            c.create_polygon(pts, outline=AMBER, fill='', width=1)

        # ── Price axis labels (left) ──
        c.create_text(margin_l - 5, margin_t, text=f'{p_max:.2f}',
                      anchor=tk.E, fill=BLUE, font=('Consolas', 8))
        c.create_text(margin_l - 5, margin_t + chart_h, text=f'{p_min:.2f}',
                      anchor=tk.E, fill=BLUE, font=('Consolas', 8))

        # ── Z axis labels (right) ──
        c.create_text(w - margin_r + 5, y_p2, text='+2',
                      anchor=tk.W, fill=RED, font=('Consolas', 7))
        c.create_text(w - margin_r + 5, y_n2, text='-2',
                      anchor=tk.W, fill=GREEN, font=('Consolas', 7))
        c.create_text(w - margin_r + 5, y_zero, text='0',
                      anchor=tk.W, fill=GREY, font=('Consolas', 7))

        # ── Current z value (top-right) ──
        if zvals:
            curr_z = zvals[-1]
            z_color = RED if curr_z > 2 else GREEN if curr_z < -2 else AMBER
            c.create_text(w - margin_r + 5, margin_t + 8,
                          text=f'z {curr_z:+.2f}',
                          anchor=tk.W, fill=z_color, font=('Consolas', 8, 'bold'))

        # ── Entry price line or current price dotted ──
        if self.in_position and self.entry_price > 0:
            ep = self.entry_price
            if p_min <= ep <= p_max:
                y_ep = y_price(ep)
                ep_color = GREEN if self.direction == 'long' else RED
                c.create_line(margin_l, y_ep, w - margin_r, y_ep,
                              fill=ep_color, dash=(4, 3), width=1)
                c.create_text(margin_l - 5, y_ep, text=f'{ep:.2f}',
                              anchor=tk.E, fill=ep_color, font=('Consolas', 8, 'bold'))
        else:
            y_curr = y_price(prices[-1])
            c.create_line(margin_l, y_curr, w - margin_r, y_curr,
                          fill=GREY, dash=(2, 4))

    def _draw_equity(self):
        """Equity curve."""
        c = self.equity_canvas
        c.delete('all')
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10 or len(self.pnl_curve) < 2:
            return

        pnls = list(self.pnl_curve)
        n = len(pnls)
        p_min = min(min(pnls), 0)
        p_max = max(max(pnls), 0)
        p_range = max(p_max - p_min, 1)

        margin = 30
        chart_w = w - margin * 2
        chart_h = h - margin

        # Zero line
        y_zero = margin + (1 - (0 - p_min) / p_range) * (chart_h - 5)
        c.create_line(margin, y_zero, w - margin, y_zero,
                      fill=GREY, dash=(2, 4))

        # Equity line
        for i in range(len(pnls) - 1):
            x1 = margin + (i / (n - 1)) * chart_w
            x2 = margin + ((i + 1) / (n - 1)) * chart_w
            y1 = margin + (1 - (pnls[i] - p_min) / p_range) * (chart_h - 5)
            y2 = margin + (1 - (pnls[i + 1] - p_min) / p_range) * (chart_h - 5)
            color = GREEN if pnls[i + 1] >= 0 else RED
            c.create_line(x1, y1, x2, y2, fill=color, width=2)

    def _update_log(self):
        """Update trade log text."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        for t in reversed(self.trades[-15:]):
            pnl = t.get('pnl', 0)
            tag = 'win' if pnl > 0 else 'loss'
            line = f'{t["time"]} ${pnl:>+7.1f}\n'
            self.log_text.insert(tk.END, line, tag)
        self.log_text.config(state=tk.DISABLED)

    # ── Button handlers ──

    def _lock_manual_buttons(self, action: str):
        """Disable manual buttons until engine confirms fill.
        Watchdog re-enables after 8s in case fill message is lost."""
        self.shared_state['manual_order'] = action
        self.shared_state['manual_order_pending'] = True
        # Disable all manual buttons
        for btn in (self.btn_flatten, self.btn_buy, self.btn_sell):
            try:
                btn.config(state='disabled')
            except Exception:
                pass
        # Watchdog: re-enable after 8s even if fill never comes
        self.root.after(8000, self._force_unlock_if_still_pending)

    def _force_unlock_if_still_pending(self):
        """Safety: re-enable buttons if pending flag hung."""
        if self.shared_state.get('manual_order_pending'):
            self.shared_state['manual_order_pending'] = False
            try:
                self.lbl_save_status.config(text='fill timeout — re-armed',
                                            fg='#ffaa00')
                self.root.after(3000,
                                lambda: self.lbl_save_status.config(text=''))
            except Exception:
                pass

    def _on_flatten(self):
        self._lock_manual_buttons('FLATTEN')

    def _on_buy(self):
        self._lock_manual_buttons('BUY')

    def _on_sell(self):
        self._lock_manual_buttons('SELL')

    def _on_save(self):
        """Request engine to flush all buffers to disk immediately."""
        self.shared_state['save_now'] = True
        self.lbl_save_status.config(text='saving...', fg=AMBER)
        # Clear the status after 2 seconds
        self.root.after(2000, lambda: self.lbl_save_status.config(text='saved',
                                                                    fg=GREEN))
        self.root.after(5000, lambda: self.lbl_save_status.config(text=''))

    def _on_close_clicked(self):
        """CLOSE button -- graceful engine shutdown.

        Sets the shutdown flag the engine's trade loop polls. The engine
        then flattens open positions, saves the checkpoint, disconnects,
        and force-exits the process (os._exit) -- which closes this window
        too. We save geometry first and show a status; a 20s fallback
        destroys the window if the engine never takes the process down
        (e.g. engine crashed before Step 7).
        """
        self._save_geometry()
        self.shared_state['shutdown'] = True
        try:
            self.lbl_save_status.config(
                text='closing -- engine flattening + saving...', fg=AMBER)
            self.btn_close.config(state='disabled', text='CLOSING')
            for b in (self.btn_flatten, self.btn_buy, self.btn_sell,
                       self.btn_save, self.btn_capture):
                b.config(state='disabled')
        except Exception:
            pass
        # Fallback: if the engine doesn't os._exit within 20s, close anyway.
        self.root.after(20000, self.root.destroy)


# Alias for engine_v2 compatibility
ProgressPopup = TradingDashboard
