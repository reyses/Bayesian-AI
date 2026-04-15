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
        self.root.geometry('1200x700')
        self.root.configure(bg=BG)
        self.root.minsize(900, 500)

        # Data
        self.prices = deque(maxlen=500)
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

        self._build_ui()
        self.root.after(100, self._poll_queue)

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

        # Price chart canvas
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
                                   font=('Consolas', 11, 'bold'), width=14,
                                   command=self._on_save)
        self.btn_save.pack(side=tk.LEFT, padx=2)

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

        if msg_type == 'TICK_UPDATE':
            price = msg.get('price', 0)
            self.last_price = price
            self.last_z = msg.get('z_se', 0)
            self.last_vr = msg.get('vr', 0)
            self.in_position = msg.get('in_position', False)
            self.direction = msg.get('direction', '')
            self.tier = msg.get('tier', '')
            self.daily_pnl = msg.get('daily_pnl', 0)
            self.unrealized = msg.get('unrealized', 0)

            self.prices.append(price)
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

        # Activity
        self.lbl_activity.config(text=self.activity)

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
        """Simple price line chart."""
        c = self.price_canvas
        c.delete('all')
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10 or len(self.prices) < 2:
            return

        prices = list(self.prices)
        n = len(prices)
        p_min = min(prices)
        p_max = max(prices)
        p_range = max(p_max - p_min, 0.01)

        margin = 40
        chart_w = w - margin * 2
        chart_h = h - margin * 2

        # Price line
        points = []
        for i, p in enumerate(prices):
            x = margin + (i / (n - 1)) * chart_w
            y = margin + (1 - (p - p_min) / p_range) * chart_h
            points.append((x, y))

        for i in range(len(points) - 1):
            c.create_line(points[i][0], points[i][1],
                          points[i + 1][0], points[i + 1][1],
                          fill=BLUE, width=1)

        # Price labels
        c.create_text(margin - 5, margin, text=f'{p_max:.0f}',
                      anchor=tk.E, fill=GREY, font=('Consolas', 8))
        c.create_text(margin - 5, h - margin, text=f'{p_min:.0f}',
                      anchor=tk.E, fill=GREY, font=('Consolas', 8))

        # Current price line
        y_curr = margin + (1 - (prices[-1] - p_min) / p_range) * chart_h
        c.create_line(margin, y_curr, w - margin, y_curr,
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

    def _on_flatten(self):
        self.shared_state['manual_order'] = 'FLATTEN'

    def _on_buy(self):
        self.shared_state['manual_order'] = 'BUY'

    def _on_sell(self):
        self.shared_state['manual_order'] = 'SELL'

    def _on_save(self):
        """Request engine to flush all buffers to disk immediately."""
        self.shared_state['save_now'] = True
        self.lbl_save_status.config(text='saving...', fg=AMBER)
        # Clear the status after 2 seconds
        self.root.after(2000, lambda: self.lbl_save_status.config(text='saved',
                                                                    fg=GREEN))
        self.root.after(5000, lambda: self.lbl_save_status.config(text=''))


# Alias for engine_v2 compatibility
ProgressPopup = TradingDashboard
