"""Non-blocking bridge between LiveEngine and the Tk popup dashboard."""

import time
from typing import Optional
import queue


class GUIBridge:
    """Encapsulates all GUI message formatting and throttling.

    LiveEngine never touches gui_queue directly  -- all pushes go through
    this bridge. If no GUI queue is provided, all pushes silently no-op.
    """

    def __init__(self, gui_queue: Optional[queue.Queue] = None):
        self._q = gui_queue
        self._last_stats_push = 0.0

    def push(self, msg: dict):
        """Non-blocking push. Drops if full or no GUI."""
        if self._q is None:
            return
        try:
            self._q.put_nowait(msg)
        except Exception:
            pass

    def push_tick(self, price: float, bars: int):
        self.push({'type': 'TICK_UPDATE', 'price': price, 'bars': bars})

    def push_trade_marker(self, action: str, side: str, price: float,
                          pnl: float = 0.0):
        msg = {'type': 'TRADE_MARKER', 'action': action,
               'side': side, 'price': price}
        if pnl:
            msg['pnl'] = pnl
        self.push(msg)

    def push_stats(self, session_pnl: float, session_wins: int,
                   session_trades: int, gross_win: float, gross_loss: float,
                   exit_buckets: dict, belief_pct: float, in_position: bool,
                   daily_pnl: float):
        """Throttled stats push  -- max 1/second."""
        now = time.time()
        if now - self._last_stats_push < 1.0:
            return
        self._last_stats_push = now

        wr = session_wins / session_trades * 100 if session_trades > 0 else 0.0
        pf = gross_win / abs(gross_loss) if gross_loss != 0 else 0.0
        eb = exit_buckets

        if in_position:
            _bar_label = f'life {belief_pct:.0f}%'
        else:
            _bar_label = (f'belief {belief_pct:.0f}%' if belief_pct > 0
                          else f'trade {session_trades}')

        self.push({
            'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
            'step': _bar_label, 'pct': belief_pct,
            'pnl': daily_pnl, 'wr': wr, 'trades': session_trades, 'pf': pf,
            'exit_optimal': eb['optimal'], 'exit_partial': eb['partial'],
            'exit_early': eb['early'], 'exit_reversed': eb['reversed'],
            'gross_w': gross_win, 'gross_l': abs(gross_loss),
        })

    def push_account(self, cash: float, realized: float,
                     unrealized: float, net_liq: float):
        self.push({'type': 'ACCOUNT_UPDATE', 'cash_value': cash,
                   'realized_pnl': realized, 'unrealized_pnl': unrealized,
                   'net_liquidation': net_liq})

    def push_shutdown_ready(self, status: str):
        self.push({'type': 'SHUTDOWN_READY', 'status': status})

    def push_phase(self, step: str, pct: float):
        self.push({'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
                   'step': step, 'pct': pct})
