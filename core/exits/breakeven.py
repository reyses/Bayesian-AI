"""Trailing Stop — continuously ratchet SL behind peak favorable price.

Direction-aware: longs and shorts have inverted cycle shapes.
  SHORT cycle: sharp drop → gradual recovery (slow reversal)
  LONG  cycle: gradual rise → sharp drop (fast reversal)

So longs get tighter trail + faster activation to catch the sharp giveback.
"""
from core.exit_engine import PositionState


class BreakevenLock:
    """Trailing stop disguised as BreakevenLock for backward compatibility."""

    def __init__(self, activation_ticks: float = 4, buffer_ticks: float = 1.0,
                 trail_pct_short: float = 0.50, trail_pct_long: float = 0.65):
        self.activation_ticks = activation_ticks
        self.buffer_ticks = buffer_ticks
        self.trail_pct_short = trail_pct_short
        self.trail_pct_long = trail_pct_long

    def apply(self, pos: PositionState, tick_size: float) -> None:
        """Adjust pos.stop_loss in-place. No ExitResult — this is an SL adjustment."""
        if pos.side == 'long':
            mfe_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            mfe_ticks = (pos.entry_price - pos.peak_favorable) / tick_size

        if mfe_ticks < self.activation_ticks:
            return

        # Longs get tighter trail — sharp drops eat profit fast
        trail_pct = self.trail_pct_long if pos.side == 'long' else self.trail_pct_short
        trail_ticks = max(self.buffer_ticks, mfe_ticks * trail_pct)
        trail_distance = trail_ticks * tick_size

        if pos.side == 'long':
            new_sl = pos.entry_price + trail_distance
            pos.stop_loss = max(pos.stop_loss, new_sl)
        else:
            new_sl = pos.entry_price - trail_distance
            pos.stop_loss = min(pos.stop_loss, new_sl)

        # Mark as active (for reporting compatibility)
        pos.breakeven_locked = True
