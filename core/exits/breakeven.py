"""Trailing Stop  -- ratchet SL behind peak favorable price.

Two-phase activation:
  Phase 1 (development): trade must reach BASE_ACTIVATION_TICKS (40t = $10 MNQ)
      before any BE protection. Lets the trade develop without noise kills.
  Phase 2 (sensor-driven tightening): once activated, trail distance adapts
      based on 1m sensor state:
      - Volume confirming + DMI aligned -> wide trail (room to breathe)
      - Volume fading -> medium trail (caution)
      - Volume opposing + DMI flipping -> tight trail (lock it NOW)

Research basis (EE analysis, 2026-03-18):
  - Old activation at ~5,000 ticks: never fires (trade MFE avg = 42 ticks)
  - Old activation at 2 ticks: kills trades on every pullback
  - 40 ticks: safe breadth, lets trade prove itself before protection
"""
from core.exit_engine import PositionState

# Base activation: MFE must reach this before BE can lock.
# 40 ticks = $10 MNQ. Below this the trade hasn't proved itself.
BASE_ACTIVATION_TICKS = 40.0

# Trail percentages: how much of MFE to protect once activated.
# Sensor state modulates these.
TRAIL_PCT_WIDE = 0.50       # sensors confirm -> give room
TRAIL_PCT_MEDIUM = 0.65     # neutral -> moderate protection
TRAIL_PCT_TIGHT = 0.80      # sensors opposing -> lock most of it

# Minimum buffer: never trail closer than this to peak
BUFFER_TICKS = 2.0


class TrailingStop:
    """Sensor-adaptive trailing stop.

    Activates at 40 ticks MFE. Trail distance driven by volume/direction
    relationship from exit_signal sensors.
    """

    def __init__(self, activation_pct: float = 0.80,
                 activation_floor_ticks: float = BASE_ACTIVATION_TICKS,
                 activation_ceiling_ticks: float = 400.0,
                 buffer_ticks: float = BUFFER_TICKS,
                 trail_pct_short: float = TRAIL_PCT_MEDIUM,
                 trail_pct_long: float = TRAIL_PCT_MEDIUM):
        self.activation_pct = activation_pct
        self.activation_floor_ticks = activation_floor_ticks
        self.activation_ceiling_ticks = activation_ceiling_ticks
        self.buffer_ticks = buffer_ticks
        self.trail_pct_short = trail_pct_short
        self.trail_pct_long = trail_pct_long

    def apply(self, pos: PositionState, tick_size: float,
              exit_signal: dict = None) -> None:
        """Adjust pos.stop_loss in-place based on MFE + sensor state.

        Args:
            pos: current position state
            tick_size: instrument tick size (0.25 for MNQ)
            exit_signal: TBN exit signal dict with sensor data (optional)
        """
        if pos.side == 'long':
            mfe_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            mfe_ticks = (pos.entry_price - pos.peak_favorable) / tick_size

        # ── Activation threshold ──
        # Use the higher of: fixed floor OR anchor-based
        _anchor = getattr(pos, 'anchor_mfe_ticks', 0.0)
        _activation = max(
            self.activation_floor_ticks,
            min(self.activation_ceiling_ticks,
                _anchor * self.activation_pct if _anchor > 0 else self.activation_floor_ticks)
        )

        if mfe_ticks < _activation:
            return

        # ── Sensor-driven trail percentage ──
        # Default: medium trail
        trail_pct = self.trail_pct_long if pos.side == 'long' else self.trail_pct_short

        if exit_signal is not None:
            # Count sensors opposing trade direction
            _n_against = sum([
                exit_signal.get('vel_1s_against', False),
                exit_signal.get('vol_1m_against', False),
                exit_signal.get('dmi_1m_against', False),
                exit_signal.get('fm_1m_against', False),
            ])

            if _n_against >= 2:
                # Multiple sensors opposing -> tighten aggressively
                trail_pct = TRAIL_PCT_TIGHT
            elif _n_against == 0 and pos.dmi_direction_confirmed:
                # All sensors aligned + DMI confirmed -> wide trail, let it run
                trail_pct = TRAIL_PCT_WIDE
            # else: 1 sensor opposing or no DMI confirmation -> medium (default)

        # ── Compute trail and ratchet SL ──
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
