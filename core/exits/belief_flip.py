"""Belief Flip Exit — TBN signals urgent exit when workers flip conviction."""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class BeliefFlipExit:

    @staticmethod
    def evaluate(pos: PositionState, bar_close: float, tick_size: float,
                 exit_signal: dict = None) -> Optional[ExitResult]:
        if exit_signal is None:
            return None
        if not exit_signal.get('urgent_exit', False):
            return None
        return ExitResult(
            action=ExitAction.TRAIL_STOP,
            exit_price=bar_close,
            reason=f"Belief flip: {exit_signal.get('reason', 'urgent')}",
            pnl_ticks=(bar_close - pos.entry_price) / tick_size
                      if pos.side == 'long'
                      else (pos.entry_price - bar_close) / tick_size,
            bars_held=pos.bars_held,
            trail_level=pos.stop_loss,
        )
