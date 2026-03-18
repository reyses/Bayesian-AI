"""Trailing Stop  -- ratchet SL behind peak favorable price once expected profit is reached.

Activation is based on per-template expected MFE (p75_mfe_ticks), not a fixed
threshold. This lets trades develop to their statistical potential before
protection kicks in.

Once activated:
  - SL moves to entry + (MFE * trail_pct) -> locks in profit
  - Only ratchets UP (longs) or DOWN (shorts)  -- never weakens

Direction-aware trailing:
  SHORT cycle: sharp drop -> gradual recovery (slow reversal)
  LONG  cycle: gradual rise -> sharp drop (fast reversal)
  So longs get tighter trail to catch the sharp giveback.
"""
from core.exit_engine import PositionState


class TrailingStop:
    """Profit-protecting trailing stop. Activates when MFE reaches a
    configurable fraction of the template's expected profit (p75_mfe)."""

    def __init__(self, activation_pct: float = 0.80,
                 activation_floor_ticks: float = 20.0,
                 activation_ceiling_ticks: float = 400.0,
                 buffer_ticks: float = 2.0,
                 trail_pct_short: float = 0.50,
                 trail_pct_long: float = 0.65):
        self.activation_pct = activation_pct          # fraction of anchor_mfe to activate
        self.activation_floor_ticks = activation_floor_ticks  # minimum activation ($5 for MNQ)
        self.activation_ceiling_ticks = activation_ceiling_ticks  # max activation ($100 MNQ)
        self.buffer_ticks = buffer_ticks              # minimum trail distance from peak
        self.trail_pct_short = trail_pct_short
        self.trail_pct_long = trail_pct_long

    def apply(self, pos: PositionState, tick_size: float) -> None:
        """Adjust pos.stop_loss in-place. No ExitResult  -- this is an SL adjustment."""
        if pos.side == 'long':
            mfe_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            mfe_ticks = (pos.entry_price - pos.peak_favorable) / tick_size

        # Activation: wait until MFE reaches activation_pct of expected profit
        # anchor_mfe_ticks comes from template p75_mfe (TF-scaled in open_position)
        _anchor = getattr(pos, 'anchor_mfe_ticks', 0.0)
        _activation = max(
            self.activation_floor_ticks,
            min(self.activation_ceiling_ticks,
                _anchor * self.activation_pct if _anchor > 0 else self.activation_floor_ticks)
        )

        if mfe_ticks < _activation:
            return

        # Trail distance: keep trail_pct of profit locked in
        # Cascade mode: if strong trend (Hurst > 0.55 + ADX > 30), widen trail
        # to let the move develop. Normal days: tight trail captures small profits.
        # Research: Feb 9 resonance cascade = $27K from wide trail on trend day.
        trail_pct = self.trail_pct_long if pos.side == 'long' else self.trail_pct_short

        # Check for cascade conditions via position's DMI confirmation
        if pos.dmi_direction_confirmed and mfe_ticks > _activation * 1.5:
            # Trade confirmed + well past activation = trending strongly
            # Reduce trail_pct by 30% -> let more profit run
            trail_pct *= 0.70

        trail_ticks = max(self.buffer_ticks, mfe_ticks * trail_pct)
        trail_distance = trail_ticks * tick_size

        if pos.side == 'long':
            new_sl = pos.entry_price + trail_distance
            pos.stop_loss = max(pos.stop_loss, new_sl)
        else:
            new_sl = pos.entry_price - trail_distance
            pos.stop_loss = min(pos.stop_loss, new_sl)

        pos.breakeven_locked = True


# Backward compatibility alias
BreakevenLock = TrailingStop
