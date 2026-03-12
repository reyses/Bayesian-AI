"""Watchdog — detect stuck trades going nowhere."""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class WatchdogCheck:

    def __init__(self, tick_threshold: float = 8, bar_threshold: int = 5,
                 worker_threshold: int = 5):
        self.tick_threshold = tick_threshold
        self.bar_threshold = bar_threshold
        self.worker_threshold = worker_threshold

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 exit_signal: dict = None) -> Optional[ExitResult]:
        if pos.bars_held < self.bar_threshold:
            return None

        if pos.side == 'long':
            adverse_ticks = (pos.entry_price - bar_close) / tick_size
        else:
            adverse_ticks = (bar_close - pos.entry_price) / tick_size

        if adverse_ticks <= self.tick_threshold:
            return None

        if pos.side == 'long':
            mfe_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            mfe_ticks = (pos.entry_price - pos.peak_favorable) / tick_size

        if mfe_ticks >= pos.trail_activation_ticks * 0.5:
            return None

        workers_against = 0
        if exit_signal is not None:
            workers_against = exit_signal.get('workers_against', 0)

        if workers_against >= self.worker_threshold or mfe_ticks < 2:
            return ExitResult(
                action=ExitAction.WATCHDOG,
                exit_price=bar_close,
                reason=f"Watchdog: {adverse_ticks:.0f} ticks adverse, "
                       f"MFE only {mfe_ticks:.0f} ticks",
                pnl_ticks=(bar_close - pos.entry_price) / tick_size
                          if pos.side == 'long'
                          else (pos.entry_price - bar_close) / tick_size,
                bars_held=pos.bars_held,
            )

        return None
