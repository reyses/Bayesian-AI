"""Stateful Peak Exit -- inverted entry signal as exit trigger.

The exit fires when the OPPOSITE trade's entry conditions are met:
same sensors, inverted direction. "Would the system enter against me?"

Sensor fusion (multi-TF):
  - 1s velocity: fast turn detection (flipped against trade = alert)
  - 1m volume:   institutional flow (against trade = confirming reversal)
  - 1m DMI:      direction structure (crossed against = trend flipped)
  - 1m F_momentum: institutional momentum (against = force reversed)

Research basis (human seeds, 2026-03-18):
  - Real peaks: 1m volume collapses (-57 aligned), 1m momentum leaves (-69)
  - Fake peaks: 1m volume still flowing (+127), momentum building against (+66)
  - DMI peak -> volume drop -> N bars -> DMI cross (the sequence)
  - Exit should fire at volume drop, not wait for DMI cross (too late)

Integration:
  - Checked FIRST in cascade for peak trades (template_id == -100)
  - Checked after giveback for non-peak trades (enhancement)
  - Requires min_hold_bars before firing (prevent 1-bar stutter exits)
  - SL always active as safety net below this
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


# -- Thresholds (named constants, calibrate from IS data) --

# Minimum bars before peak exit can fire.
# Research: 1-bar exits are the stutter problem. 6-10 bar sweet spot.
# Set to 3 = let the trade develop but don't hold through obvious reversals.
MIN_HOLD_BARS_PEAK = 3

# For non-peak trades, be more conservative (they entered on pattern match)
MIN_HOLD_BARS_TEMPLATE = 5

# Sensor agreement thresholds:
# Full confidence (all 4 sensors agree) -> exit immediately
# Strong (3 of 4) -> exit if trade has been profitable
# Partial (2 of 4) -> only exit if giving back significantly
FULL_SENSOR_COUNT = 4
STRONG_SENSOR_COUNT = 3

# Minimum peak before inverted exit can fire (prevent exiting flat trades)
# Trade must have reached at least this many ticks profit before we consider
# "protecting" it via inverted exit. Below this, SL handles it.
MIN_PEAK_TICKS = 4.0

# Cooldown: after a peak exit fires, suppress new peak entries for N bars.
# This prevents re-entering the same decaying peak.
COOLDOWN_BARS = 6


class PeakStateExit:
    """Exit when the opposite direction's entry conditions are met.

    Uses 4 sensors from exit_signal (populated by TBN):
      vel_1s_against:  1s velocity flipped against trade direction
      vol_1m_against:  1m volume flowing against trade direction
      dmi_1m_against:  1m DMI crossed against trade direction
      fm_1m_against:   1m F_momentum against trade direction
    """

    def __init__(self, config=None):
        self.cooldown_until = 0  # bar index when cooldown expires

    def evaluate(
        self,
        pos: PositionState,
        bar_close: float,
        tick_size: float,
        current_bar_index: int,
        exit_signal: dict = None,
        belief_network=None,
    ) -> Optional[ExitResult]:
        """Check if inverted entry fires against current position.

        Two signals:
        1. Sensor fusion (4 sensors from 1s + 1m) — existing logic
        2. Cascade fade (4+ TF workers peak against trade) — new, PF=2.44

        Returns ExitResult if exit triggered, None if HOLD.
        """
        if exit_signal is None:
            return None

        # -- Minimum hold: prevent 1-bar stutter exits --
        is_peak_trade = (pos.template_id == -100)
        min_bars = MIN_HOLD_BARS_PEAK if is_peak_trade else MIN_HOLD_BARS_TEMPLATE
        if pos.bars_held < min_bars:
            return None

        # -- Minimum peak: only protect trades that have been profitable --
        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
            current_ticks = (bar_close - pos.entry_price) / tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / tick_size
            current_ticks = (pos.entry_price - bar_close) / tick_size

        if peak_ticks < MIN_PEAK_TICKS:
            return None  # trade never proved itself -- SL handles these

        # -- Signal 1: Cascade fade (multi-TF peak agreement against trade) --
        # Research: when 4+ TFs peak in the same direction, entering OPPOSITE
        # has PF=2.44. If 4+ TFs peak AGAINST our trade, the move we're riding
        # is exhausted at every scale. Exit immediately.
        if belief_network is not None:
            _cascade_against = self._check_cascade_fade(pos, belief_network)
            if _cascade_against >= 4:
                return ExitResult(
                    action=ExitAction.PEAK_STATE_EXIT,
                    exit_price=bar_close,
                    reason=f"Cascade fade: {_cascade_against}/5 TFs peaked against "
                           f"({pos.side}) peak={peak_ticks:.1f}t now={current_ticks:.1f}t",
                    pnl_ticks=current_ticks,
                    bars_held=pos.bars_held,
                )

        # -- Signal 2: Sensor fusion (existing 4-sensor logic) --
        sig_vel_1s = exit_signal.get('vel_1s_against', False)
        sig_vol_1m = exit_signal.get('vol_1m_against', False)
        sig_dmi_1m = exit_signal.get('dmi_1m_against', False)
        sig_fm_1m = exit_signal.get('fm_1m_against', False)

        n_sensors = sum([sig_vel_1s, sig_vol_1m, sig_dmi_1m, sig_fm_1m])

        # -- Decision logic --
        exit_reason = None

        if n_sensors >= FULL_SENSOR_COUNT:
            # All 4 sensors say "enter against me" -> exit immediately
            exit_reason = 'full_inverted'

        elif n_sensors >= STRONG_SENSOR_COUNT:
            # 3 of 4 agree -> exit only if significantly giving back
            gave_back_pct = (peak_ticks - current_ticks) / peak_ticks if peak_ticks > 0 else 0
            if gave_back_pct >= 0.30:
                exit_reason = 'strong_inverted'

        if exit_reason is None:
            return None

        # -- Set cooldown to prevent re-entry on same peak --
        self.cooldown_until = current_bar_index + COOLDOWN_BARS

        # -- Build sensor detail string --
        sensors = []
        if sig_vel_1s:
            sensors.append('1s_vel')
        if sig_vol_1m:
            sensors.append('1m_vol')
        if sig_dmi_1m:
            sensors.append('1m_dmi')
        if sig_fm_1m:
            sensors.append('1m_fm')

        return ExitResult(
            action=ExitAction.PEAK_STATE_EXIT,
            exit_price=bar_close,
            reason=f"Peak state exit ({exit_reason}): {n_sensors}/4 sensors "
                   f"[{'+'.join(sensors)}] peak={peak_ticks:.1f}t "
                   f"now={current_ticks:.1f}t",
            pnl_ticks=current_ticks,
            bars_held=pos.bars_held,
        )

    def _check_cascade_fade(self, pos, belief_network) -> int:
        """Count how many TF workers show peaks against the trade direction.

        Research (2026-03-21): when 4+ TFs peak against trade = PF 2.44 fade.
        The move we're riding is exhausted at every scale.

        Checks TF workers: 1m, 5m, 15m, 1h, 4h.
        A worker "peaks against" when its F_momentum is decaying AND
        its direction (DMI) opposes the trade side.
        """
        _cascade_tfs = [60, 300, 900, 3600, 14400]  # 1m, 5m, 15m, 1h, 4h
        _trade_sign = 1.0 if pos.side == 'long' else -1.0
        _against = 0

        for tf_secs in _cascade_tfs:
            worker = belief_network.workers.get(tf_secs)
            if worker is None or not worker._states:
                continue
            idx = worker._last_tf_bar_idx
            if idx < 1 or idx >= len(worker._states):
                continue

            curr = worker._states[idx]
            prev = worker._states[idx - 1]
            ms_curr = curr['state'] if isinstance(curr, dict) else curr
            ms_prev = prev['state'] if isinstance(prev, dict) else prev

            # Worker peaks against trade when:
            # 1. DMI opposes trade direction
            dmi_p = getattr(ms_curr, 'dmi_plus', 0.0) or 0.0
            dmi_m = getattr(ms_curr, 'dmi_minus', 0.0) or 0.0
            dmi_against = (dmi_m - dmi_p) * _trade_sign > 0

            # 2. F_momentum decaying (peak exhaustion at this scale)
            fm_curr = abs(getattr(ms_curr, 'F_momentum', 0.0) or 0.0)
            fm_prev = abs(getattr(ms_prev, 'F_momentum', 0.0) or 0.0)
            fm_decaying = fm_curr < fm_prev * 0.90 if fm_prev > 0.5 else False

            if dmi_against and fm_decaying:
                _against += 1

        return _against

    def in_cooldown(self, bar_index: int) -> bool:
        """Check if we're in cooldown (suppress re-entry on same peak)."""
        return bar_index < self.cooldown_until
