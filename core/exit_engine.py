"""
Unified Exit Engine — cascade orchestrator
============================================
Owns: position state, MFE tracking, sub-bar wick resolution, cascade ordering.
Delegates each exit check to its standalone module in core/exits/.

Usage:
    exit_eng = ExitEngine(mode='training', tick_size=0.25, tick_value=0.50)
    pos = exit_eng.open_position(side, entry_price, bar_idx, tid, lib_entry)
    result = exit_eng.evaluate(pos, bar_high, bar_low, bar_close, bar_idx, ...)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List


class ExitAction(Enum):
    HOLD = 'hold'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    TRAIL_STOP = 'trail_stop'
    MAX_HOLD = 'max_hold'
    ENVELOPE_DECAY = 'envelope_decay'
    BAND_URGENT = 'band_urgent'
    WATCHDOG = 'watchdog'
    BREAKEVEN_LOCK = 'breakeven_lock'
    PEAK_GIVEBACK = 'peak_giveback'
    MAINTENANCE_FLAT = 'maintenance_flat'


@dataclass
class ExitResult:
    action: ExitAction
    exit_price: float
    reason: str
    pnl_ticks: float = 0.0
    bars_held: int = 0
    trail_level: float = 0.0
    envelope_level: float = 0.0
    band_zone: str = ''
    band_action: str = ''  # 'tighten', 'widen', 'urgent', 'none'


@dataclass
class PositionState:
    """Unified position state — used by ExitEngine and exit modules."""
    side: str                    # 'long' or 'short'
    entry_price: float
    entry_bar_index: int
    template_id: int
    tick_size: float = 0.25
    tick_value: float = 0.50     # MNQ: $0.50 per tick

    # Template-specific parameters (from pattern_library)
    sl_ticks: float = 0.0
    tp_ticks: float = 0.0
    trail_activation_ticks: float = 0.0
    trailing_stop_ticks: float = 0.0
    max_hold_bars: int = 120

    # Dynamic state (updated each bar)
    peak_favorable: float = 0.0
    bars_held: int = 0
    breakeven_locked: bool = False
    envelope_active: bool = False
    envelope_level: float = 0.0

    # Position lifecycle
    entry_time: float = 0.0
    stop_loss: float = 0.0          # absolute price level
    profit_target: float = 0.0      # absolute price level (0 = none)
    envelope_T0: float = 0.0
    envelope_halflife: float = 20.0

    # 30m worker flip: once fired, stays True for remainder of trade
    slow_flip_active: bool = False

    # Anchor: expected trade outcome (price + time)
    anchor_mfe_ticks: float = 0.0   # template p75_mfe, brain-adjusted
    anchor_mfe_bars: float = 0.0    # template avg_mfe_bar, brain-adjusted


class ExitEngine:
    """
    Unified exit engine for training and live.

    Mode differences (data availability only):
        training: receives intra-bar high/low for wick checking
        live:     receives 15s bar OHLC (use high/low, not just close)

    All logic is identical regardless of mode.
    """

    def __init__(
        self,
        mode: str = 'live',
        tick_size: float = 0.25,
        tick_value: float = 0.50,
    ):
        assert mode in ('training', 'live'), f"Invalid mode: {mode}"
        self.mode = mode
        self.tick_size = tick_size
        self.tick_value = tick_value

        # ── Exit modules ──
        from core.exits.stop_loss import StopLossCheck
        from core.exits.take_profit import TakeProfitCheck
        from core.exits.breakeven import BreakevenLock
        from core.exits.envelope import EnvelopeDecay
        from core.exits.giveback import PeakGiveback
        from core.exits.band_exit import BandUrgentExit
        from core.exits.watchdog import WatchdogCheck
        from core.exits.belief_flip import BeliefFlipExit

        self.stop_loss = StopLossCheck()
        self.take_profit = TakeProfitCheck()
        self.breakeven = BreakevenLock(activation_ticks=4)
        self.envelope = EnvelopeDecay(half_life_bars=20, floor_pct=0.3, min_bars=5)
        self.giveback = PeakGiveback(min_mfe_ticks=16, giveback_pct=0.70)
        self.band_exit = BandUrgentExit()
        self.watchdog = WatchdogCheck(tick_threshold=8, bar_threshold=5,
                                       worker_threshold=5)
        self.belief_flip = BeliefFlipExit()

        # Self-tuning state — two independent counters
        self._tune_too_early = 0
        self._tune_too_late = 0
        self._tune_total = 0
        self._tune_hl_min = 8
        self._tune_hl_max = 60
        self._tune_gb_min = 0.55
        self._tune_gb_max = 0.90
        self._tune_window = 30

    # ── Convenience aliases for tuned params ──

    @property
    def envelope_half_life_bars(self):
        return self.envelope.half_life_bars

    @envelope_half_life_bars.setter
    def envelope_half_life_bars(self, val):
        self.envelope.half_life_bars = val

    @property
    def giveback_pct(self):
        return self.giveback.giveback_pct

    @giveback_pct.setter
    def giveback_pct(self, val):
        self.giveback.giveback_pct = val

    @property
    def envelope_floor_pct(self):
        return self.envelope.floor_pct

    @property
    def envelope_min_bars(self):
        return self.envelope.min_bars

    @property
    def giveback_min_mfe_ticks(self):
        return self.giveback.min_mfe_ticks

    @property
    def be_activation_ticks(self):
        return self.breakeven.activation_ticks

    @be_activation_ticks.setter
    def be_activation_ticks(self, val):
        self.breakeven.activation_ticks = val

    # ==================================================================
    # SELF-TUNING
    # ==================================================================

    def record_trade_outcome(self, trade_mfe_ticks: float, actual_pnl_ticks: float,
                             capture_rate: float):
        """Feed closed-trade stats back to auto-tune exit parameters."""
        self._tune_total += 1

        if 0 < capture_rate < 0.20 and trade_mfe_ticks < 8:
            self._tune_too_early += 1

        if trade_mfe_ticks >= 16:
            gave_back = (trade_mfe_ticks - actual_pnl_ticks) / trade_mfe_ticks
            if gave_back >= 0.50:
                self._tune_too_late += 1

        if self._tune_total % self._tune_window == 0 and self._tune_total > 0:
            if self._tune_too_early >= 3:
                self.envelope_half_life_bars = min(
                    self._tune_hl_max,
                    self.envelope_half_life_bars * 1.10)
            if self._tune_too_late >= 3:
                self.giveback_pct = max(
                    self._tune_gb_min,
                    self.giveback_pct - 0.05)
            if self._tune_too_early < 3 and self._tune_too_late < 3:
                self.envelope_half_life_bars += (20 - self.envelope_half_life_bars) * 0.1
                self.giveback_pct += (0.70 - self.giveback_pct) * 0.1
            self._tune_too_early = 0
            self._tune_too_late = 0

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def open_position(
        self,
        side: str,
        entry_price: float,
        entry_bar_index: int,
        template_id: int,
        sl_ticks: float,
        tp_ticks: float,
        trail_ticks: float = 0.0,
        trail_activation_ticks: float = 0.0,
        max_hold_bars: int = 120,
        lib_entry: dict = None,
    ) -> PositionState:
        """Initialize a new position with pre-computed exit parameters."""
        if lib_entry:
            _p75_bar = lib_entry.get('p75_mfe_bar', 0.0)
            if _p75_bar > 0:
                max_hold_bars = int(_p75_bar * 2.5)
            else:
                max_hold_bars = lib_entry.get('max_hold_bars', max_hold_bars)

        _anchor_mfe = 0.0
        _anchor_bars = 0.0
        if lib_entry:
            _anchor_mfe = lib_entry.get('p75_mfe_ticks', 0.0)
            _anchor_bars = lib_entry.get('avg_mfe_bar', 0.0)

        pos = PositionState(
            side=side,
            entry_price=entry_price,
            entry_bar_index=entry_bar_index,
            template_id=template_id,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            sl_ticks=float(sl_ticks),
            tp_ticks=float(tp_ticks),
            trailing_stop_ticks=float(trail_ticks) if trail_ticks else float(sl_ticks),
            trail_activation_ticks=float(trail_activation_ticks or 0),
            max_hold_bars=max_hold_bars,
            anchor_mfe_ticks=float(_anchor_mfe),
            anchor_mfe_bars=float(_anchor_bars),
        )

        _sd = sl_ticks * self.tick_size
        _td = tp_ticks * self.tick_size
        if side == 'long':
            pos.stop_loss = entry_price - _sd
            pos.profit_target = entry_price + _td if tp_ticks > 0 else 0.0
        else:
            pos.stop_loss = entry_price + _sd
            pos.profit_target = entry_price - _td if tp_ticks > 0 else 0.0

        pos.peak_favorable = entry_price
        pos.envelope_active = True
        pos.envelope_level = tp_ticks * self.tick_size
        pos.envelope_T0 = float(sl_ticks)

        import time as _time
        pos.entry_time = _time.time()

        return pos

    def evaluate(
        self,
        pos: PositionState,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        current_bar_index: int,
        band_context: dict = None,
        net_force: float = 0.0,
        exit_signal: dict = None,
        sub_bar_highs: list = None,
        sub_bar_lows: list = None,
        noise_ticks: float = 0.0,
    ) -> ExitResult:
        """
        Evaluate all exit conditions for current bar.

        Cascade order (first trigger wins):
        1. Stop loss       2. Take profit     3. Watchdog
        4. Band urgent     5. Breakeven lock   6. Envelope decay
        7. Peak giveback   8. Belief flip      9. HOLD
        """
        pos.bars_held = current_bar_index - pos.entry_bar_index
        pos.bars_in_trade = pos.bars_held  # sync alias

        # ── Resolve worst/best price this bar ──
        if sub_bar_highs is not None and sub_bar_lows is not None and len(sub_bar_highs) > 0:
            worst_price = min(sub_bar_lows) if pos.side == 'long' else max(sub_bar_highs)
            best_price = max(sub_bar_highs) if pos.side == 'long' else min(sub_bar_lows)
        else:
            worst_price = bar_low if pos.side == 'long' else bar_high
            best_price = bar_high if pos.side == 'long' else bar_low

        # ── Update MFE tracking ──
        if pos.side == 'long':
            pos.peak_favorable = max(pos.peak_favorable, best_price)
        else:
            pos.peak_favorable = min(pos.peak_favorable, best_price)

        # ── Cascade ──
        ts = self.tick_size

        # 1. Stop Loss
        r = self.stop_loss.evaluate(pos, worst_price, ts)
        if r: return r

        # 2. Take Profit
        r = self.take_profit.evaluate(pos, best_price, ts)
        if r: return r

        # 3. Watchdog
        r = self.watchdog.evaluate(pos, bar_close, ts, exit_signal)
        if r: return r

        # 4. Band Urgent
        r = self.band_exit.evaluate(pos, bar_close, ts, band_context)
        if r: return r

        # 5. Breakeven Lock (adjusts SL in-place, no exit)
        self.breakeven.apply(pos, ts)

        # 6. Envelope Decay
        r = self.envelope.evaluate(pos, bar_close, ts, net_force, band_context,
                                   noise_ticks)
        if r: return r

        # 7. Peak Giveback
        r = self.giveback.evaluate(pos, bar_close, ts, exit_signal, noise_ticks)
        if r: return r

        # 8. Belief Flip
        r = self.belief_flip.evaluate(pos, bar_close, ts, exit_signal)
        if r: return r

        # 9. HOLD
        return ExitResult(
            action=ExitAction.HOLD,
            exit_price=0.0,
            reason='hold',
            bars_held=pos.bars_held,
            trail_level=pos.stop_loss,
            envelope_level=pos.envelope_level,
            band_zone=band_context.get('band_summary', '') if band_context else '',
        )

    # ==================================================================
    # CME Maintenance Flatten
    # ==================================================================

    MAINT_FLATTEN_MINUTE = 16 * 60 + 45   # 16:45 ET
    MAINT_REOPEN_MINUTE = 18 * 60         # 18:00 ET

    @staticmethod
    def is_maintenance_window(bar_timestamp: float) -> bool:
        """Check if bar falls in CME maintenance flatten window (16:45-18:00 ET)."""
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        _et = datetime.fromtimestamp(bar_timestamp, tz=timezone.utc).astimezone(
            ZoneInfo('US/Eastern'))
        _minute_of_day = _et.hour * 60 + _et.minute
        return (ExitEngine.MAINT_FLATTEN_MINUTE
                <= _minute_of_day
                < ExitEngine.MAINT_REOPEN_MINUTE)

    # ==================================================================
    # PRIVATE -- Utilities (kept for backward compat)
    # ==================================================================

    def _calc_pnl_ticks(self, pos: PositionState, exit_price: float) -> float:
        if pos.side == 'long':
            return (exit_price - pos.entry_price) / self.tick_size
        else:
            return (pos.entry_price - exit_price) / self.tick_size
