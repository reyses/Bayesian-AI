"""Watchdog  -- end-of-session flatten + live operational safety.

Purpose:
  - Flatten positions before market maintenance windows
  - Flatten on program shutdown / connection loss (live only)
  - NOT for "stuck trade" detection (other exits handle that)

In IS/OOS: only fires near end-of-day (last N minutes before session close).
In live: also fires on maintenance windows, shutdown, disconnection.
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


# CME MNQ session close = 16:00 CT (22:00 UTC), maintenance 16:15-16:30 CT
# RTH close = 16:00 CT, Globex close = 17:00 CT
SESSION_END_BUFFER_BARS = 20  # 5 minutes at 15s  -- flatten before close


class WatchdogCheck:

    def __init__(self, tick_threshold: float = 8, bar_threshold: int = 5,
                 worker_threshold: int = 5, config=None):
        self.tick_threshold = tick_threshold
        self.bar_threshold = bar_threshold
        self.worker_threshold = worker_threshold
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._mfe_progress_pct = config.watchdog_mfe_progress_pct
        self._mfe_floor_ticks = config.watchdog_mfe_floor_ticks

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 exit_signal: dict = None) -> Optional[ExitResult]:
        """End-of-session flatten only.

        The old "stuck trade" logic (adverse ticks + workers against) is removed.
        Other exits (regime_decay, tidal_wave, belief_flip, giveback, 15s flip)
        handle mid-trade thesis invalidation. Watchdog is purely operational.
        """
        # End-of-session flatten: check if we're near session close
        # This is signaled by the trainer/live engine setting 'session_ending'
        # in exit_signal, or by the bars_remaining field.
        if exit_signal is not None and exit_signal.get('session_ending'):
            if pos.side == 'long':
                _pnl = (bar_close - pos.entry_price) / tick_size
            else:
                _pnl = (pos.entry_price - bar_close) / tick_size

            return ExitResult(
                action=ExitAction.WATCHDOG,
                exit_price=bar_close,
                reason=f"Watchdog: session ending, flatten "
                       f"(held {pos.bars_held} bars, PnL={_pnl:.1f}t)",
                pnl_ticks=_pnl,
                bars_held=pos.bars_held,
            )

        return None
