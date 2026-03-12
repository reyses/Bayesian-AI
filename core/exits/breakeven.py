"""Breakeven Lock — move SL to entry once MFE exceeds threshold."""
from core.exit_engine import PositionState


class BreakevenLock:

    def __init__(self, activation_ticks: float = 4):
        self.activation_ticks = activation_ticks

    def apply(self, pos: PositionState, tick_size: float) -> None:
        """Adjust pos.stop_loss in-place. No ExitResult — this is an SL adjustment."""
        if pos.breakeven_locked:
            return
        if pos.side == 'long':
            favorable = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            favorable = (pos.entry_price - pos.peak_favorable) / tick_size
        if favorable >= self.activation_ticks:
            buffer = 1 * tick_size
            if pos.side == 'long':
                be_level = pos.entry_price + buffer
                pos.stop_loss = max(pos.stop_loss, be_level)
            else:
                be_level = pos.entry_price - buffer
                pos.stop_loss = min(pos.stop_loss, be_level)
            pos.breakeven_locked = True
