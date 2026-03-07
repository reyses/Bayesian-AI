"""
Unified Exit Engine
===================
Single exit decision module used by both training orchestrator and live engine.
Ensures training metrics accurately reflect live execution behavior.

Usage:
    # In orchestrator (training)
    exit_eng = ExitEngine(mode='training', tick_size=0.25, tick_value=0.50)

    # In live engine
    exit_eng = ExitEngine(mode='live', tick_size=0.25, tick_value=0.50)

    # Same API, same logic, same results
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
    """Unified position state — used by ExitEngine AND WaveRider (single object)."""
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
    current_trail: float = 0.0
    peak_favorable: float = 0.0
    bars_held: int = 0
    breakeven_locked: bool = False
    envelope_active: bool = False
    envelope_level: float = 0.0

    # Position lifecycle (formerly wave_rider.Position)
    entry_time: float = 0.0
    stop_loss: float = 0.0          # absolute price level
    profit_target: float = 0.0      # absolute price level (0 = none)
    high_water_mark: float = 0.0
    entry_layer_state: object = None
    entry_dmi_inverse: bool = False
    original_trail_ticks: float = 0.0
    last_adjustment_reason: str = ''
    breakeven_level: float = 0.0
    bars_in_trade: int = 0
    envelope_T0: float = 0.0
    envelope_halflife: float = 20.0

    # CST structural integrity
    cst_centroid: object = None
    cst_basin_mean: float = 0.0
    cst_basin_std: float = 0.0
    cst_ancestry: object = None


def make_position(entry_price: float, side: str, tick_size: float = 0.25,
                   tick_value: float = 0.50, stop_distance_ticks: int = 20,
                   profit_target_ticks: int = 0, trailing_stop_ticks: int = 0,
                   trail_activation_ticks: int = 0, template_id=0,
                   state=None, cst_centroid=None, cst_basin_mean: float = 0.0,
                   cst_basin_std: float = 0.0, cst_ancestry=None) -> PositionState:
    """Create a PositionState with absolute price levels. Replaces WaveRider.open_position()."""
    import time as _time
    sd = stop_distance_ticks * tick_size
    sl = (entry_price + sd) if side == 'short' else (entry_price - sd)
    pt = 0.0
    if profit_target_ticks:
        ptd = profit_target_ticks * tick_size
        pt = (entry_price - ptd) if side == 'short' else (entry_price + ptd)
    return PositionState(
        side=side, entry_price=entry_price, entry_bar_index=0,
        template_id=template_id or 0, tick_size=tick_size, tick_value=tick_value,
        sl_ticks=float(stop_distance_ticks), tp_ticks=float(profit_target_ticks or 0),
        trailing_stop_ticks=float(trailing_stop_ticks or 0),
        trail_activation_ticks=float(trail_activation_ticks or 0),
        original_trail_ticks=float(trailing_stop_ticks or 0),
        stop_loss=sl, profit_target=pt,
        high_water_mark=entry_price, peak_favorable=entry_price, current_trail=sl,
        entry_time=_time.time(), entry_layer_state=state,
        envelope_T0=float(stop_distance_ticks), envelope_halflife=20.0,
        envelope_active=True,
        cst_centroid=cst_centroid, cst_basin_mean=cst_basin_mean,
        cst_basin_std=cst_basin_std, cst_ancestry=cst_ancestry,
    )


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

        # Envelope decay parameters
        self.envelope_half_life_bars = 20
        self.envelope_floor_pct = 0.3
        self.envelope_min_bars = 5

        # Peak giveback parameters
        self.giveback_min_mfe_ticks = 16   # must have reached at least 16 ticks profit
        self.giveback_pct = 0.70           # exit if gave back 70% of peak profit

        # Self-tuning state — two independent counters
        self._tune_too_early = 0   # exited before move materialized
        self._tune_too_late = 0    # held past peak, gave back profit
        self._tune_total = 0
        self._tune_hl_min = 8
        self._tune_hl_max = 60
        self._tune_gb_min = 0.55   # giveback_pct floor
        self._tune_gb_max = 0.90   # giveback_pct ceiling
        self._tune_window = 30     # recalibrate every N trades

        # Watchdog parameters
        self.watchdog_tick_threshold = 8
        self.watchdog_bar_threshold = 5
        self.watchdog_worker_threshold = 5


    # ==================================================================
    # SELF-TUNING
    # ==================================================================

    def record_trade_outcome(self, trade_mfe_ticks: float, actual_pnl_ticks: float,
                             capture_rate: float):
        """Feed closed-trade stats back to auto-tune exit parameters.

        Two independent signals, each tuning a different knob:
          too_early → grow halflife (be more patient)
          too_late  → shrink giveback_pct (cut losers faster at peak)

        Called after every trade close. Recalibrates every _tune_window trades.
        """
        self._tune_total += 1

        # Too early: low capture AND trade never reached a good peak
        if 0 < capture_rate < 0.20 and trade_mfe_ticks < 8:
            self._tune_too_early += 1

        # Too late: reached good peak but gave most of it back
        if trade_mfe_ticks >= 16:
            gave_back = (trade_mfe_ticks - actual_pnl_ticks) / trade_mfe_ticks
            if gave_back >= 0.50:
                self._tune_too_late += 1

        # Recalibrate envelope every N trades
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
        lib_entry: dict,
        atr_ticks: float = 0.0,
        network_tp: Optional[float] = None,
    ) -> PositionState:
        """
        Initialize a new position with template-specific exit parameters.

        STOP SIZING: Uses cluster-fitted metrics from pattern_library.
        Falls back to ATR-based only if cluster metrics are missing.
        """
        pos = PositionState(
            side=side,
            entry_price=entry_price,
            entry_bar_index=entry_bar_index,
            template_id=template_id,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
        )

        # -- Stop Loss sizing (cluster-fitted, not ATR) --
        p25_mae = lib_entry.get('p25_mae')
        mean_mae = lib_entry.get('mean_mae')
        reg_sigma = lib_entry.get('regression_sigma')
        _atr = atr_ticks if atr_ticks > 0 else lib_entry.get('atr', 20.0)

        if p25_mae is not None and p25_mae > 0:
            pos.sl_ticks = p25_mae * 3.0
        elif mean_mae is not None and mean_mae > 0:
            pos.sl_ticks = mean_mae * 2.0
        elif reg_sigma is not None and reg_sigma > 0:
            pos.sl_ticks = reg_sigma * 1.1  # already in ticks from lib_entry
        else:
            pos.sl_ticks = _atr * 3.0  # ATR fallback

        pos.sl_ticks = max(pos.sl_ticks, 8.0)
        pos.sl_ticks = min(pos.sl_ticks, 80.0)

        # -- Take Profit sizing (cascade) --
        mfe_coeff = lib_entry.get('mfe_coeff')
        p75_mfe = lib_entry.get('p75_mfe')

        if network_tp is not None and network_tp > 0:
            pos.tp_ticks = network_tp
        elif mfe_coeff is not None and mfe_coeff > 0:
            pos.tp_ticks = mfe_coeff
        elif p75_mfe is not None and p75_mfe > 0:
            pos.tp_ticks = p75_mfe
        else:
            pos.tp_ticks = _atr * 3.0

        pos.tp_ticks = max(pos.tp_ticks, 4.0)
        pos.tp_ticks = min(pos.tp_ticks, 200.0)

        # -- Trail activation --
        if p25_mae is not None and p25_mae > 0:
            pos.trail_activation_ticks = p25_mae * 0.3
        else:
            pos.trail_activation_ticks = _atr * 0.6

        pos.trail_activation_ticks = max(pos.trail_activation_ticks, 3.0)

        # -- Trailing stop distance (starts at SL distance) --
        pos.trailing_stop_ticks = pos.sl_ticks

        # -- Max hold --
        pos.max_hold_bars = lib_entry.get('max_hold_bars', 120)

        # -- Initial trail at entry --
        if side == 'long':
            pos.current_trail = entry_price - (pos.sl_ticks * self.tick_size)
            pos.peak_favorable = entry_price
        else:
            pos.current_trail = entry_price + (pos.sl_ticks * self.tick_size)
            pos.peak_favorable = entry_price

        # -- Envelope initialization --
        pos.envelope_active = True
        pos.envelope_level = pos.tp_ticks * self.tick_size

        # -- Populate backward-compat fields (formerly wave_rider.Position) --
        import time as _time
        pos.entry_time = _time.time()
        pos.high_water_mark = entry_price
        pos.original_trail_ticks = pos.trailing_stop_ticks
        pos.envelope_T0 = float(pos.sl_ticks)
        # Absolute price levels
        if side == 'long':
            pos.stop_loss = entry_price - (pos.sl_ticks * self.tick_size)
            pos.profit_target = entry_price + (pos.tp_ticks * self.tick_size) if pos.tp_ticks > 0 else 0.0
        else:
            pos.stop_loss = entry_price + (pos.sl_ticks * self.tick_size)
            pos.profit_target = entry_price - (pos.tp_ticks * self.tick_size) if pos.tp_ticks > 0 else 0.0

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
    ) -> ExitResult:
        """
        Evaluate all exit conditions for current bar.

        Evaluation order (first trigger wins):
        1. Stop loss (hard stop)
        2. Take profit
        3. Watchdog (stuck trade)
        4. Max hold
        5. Band-aware urgent exit
        6. Envelope decay
        7. Trail stop update + check
        8. Breakeven lock
        9. HOLD
        """
        pos.bars_held = current_bar_index - pos.entry_bar_index
        pos.bars_in_trade = pos.bars_held  # sync alias

        # -- Determine worst/best price this bar --
        if sub_bar_highs is not None and sub_bar_lows is not None and len(sub_bar_highs) > 0:
            worst_price = min(sub_bar_lows) if pos.side == 'long' else max(sub_bar_highs)
            best_price = max(sub_bar_highs) if pos.side == 'long' else min(sub_bar_lows)
        else:
            worst_price = bar_low if pos.side == 'long' else bar_high
            best_price = bar_high if pos.side == 'long' else bar_low

        # -- Update MFE tracking --
        if pos.side == 'long':
            pos.peak_favorable = max(pos.peak_favorable, best_price)
        else:
            pos.peak_favorable = min(pos.peak_favorable, best_price)
        pos.high_water_mark = pos.peak_favorable  # sync alias

        # -- 1. STOP LOSS --
        sl_price = self._get_stop_price(pos)
        if self._is_stopped(pos, worst_price, sl_price):
            return ExitResult(
                action=ExitAction.STOP_LOSS,
                exit_price=sl_price,
                reason=f"SL hit at {sl_price:.2f} (worst={worst_price:.2f})",
                pnl_ticks=self._calc_pnl_ticks(pos, sl_price),
                bars_held=pos.bars_held,
                trail_level=pos.current_trail,
            )

        # -- 2. TAKE PROFIT --
        tp_price = self._get_tp_price(pos)
        if self._is_tp_hit(pos, best_price, tp_price):
            return ExitResult(
                action=ExitAction.TAKE_PROFIT,
                exit_price=tp_price,
                reason=f"TP hit at {tp_price:.2f} (best={best_price:.2f})",
                pnl_ticks=self._calc_pnl_ticks(pos, tp_price),
                bars_held=pos.bars_held,
                trail_level=pos.current_trail,
            )

        # -- 3. WATCHDOG --
        watchdog = self._check_watchdog(pos, bar_close, exit_signal)
        if watchdog is not None:
            return watchdog

        # -- 4. MAX HOLD -- DISABLED: envelope_decay handles time-based exits
        #    better ($46/trade vs $24/trade). Max hold cut winners short.

        # -- 5. BAND-AWARE URGENT EXIT --
        band_result = self._check_band_exit(pos, bar_close, band_context)
        if band_result is not None:
            return band_result

        # -- 6. ENVELOPE DECAY --
        envelope_result = self._check_envelope(pos, bar_close, net_force, band_context)
        if envelope_result is not None:
            return envelope_result

        # -- 6b. PEAK GIVEBACK --
        giveback_result = self._check_peak_giveback(pos, bar_close)
        if giveback_result is not None:
            return giveback_result

        # -- 7. TRAIL STOP -- DISABLED: trail exits avg $3-4/trade with 84%
        #    "too early" rate.  Envelope decay is physics-aware and 12x more
        #    profitable per trade.  Still update HWM for breakeven logic.
        self._update_trail(pos, best_price, band_context)

        # -- 8. BREAKEVEN LOCK --
        self._check_breakeven(pos)

        # -- 9. Exit signal from belief network (tighten/urgent) --
        if exit_signal is not None and exit_signal.get('urgent_exit', False):
            return ExitResult(
                action=ExitAction.TRAIL_STOP,
                exit_price=bar_close,
                reason=f"Belief flip: {exit_signal.get('reason', 'urgent')}",
                pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                bars_held=pos.bars_held,
                trail_level=pos.current_trail,
            )

        # -- 10. HOLD --
        return ExitResult(
            action=ExitAction.HOLD,
            exit_price=0.0,
            reason='hold',
            bars_held=pos.bars_held,
            trail_level=pos.current_trail,
            envelope_level=pos.envelope_level,
            band_zone=band_context.get('band_summary', '') if band_context else '',
        )

    # ==================================================================
    # PRIVATE -- Stop Loss
    # ==================================================================

    def _get_stop_price(self, pos: PositionState) -> float:
        if pos.side == 'long':
            return pos.entry_price - (pos.sl_ticks * self.tick_size)
        else:
            return pos.entry_price + (pos.sl_ticks * self.tick_size)

    def _is_stopped(self, pos: PositionState, worst_price: float, sl_price: float) -> bool:
        if pos.side == 'long':
            return worst_price <= sl_price
        else:
            return worst_price >= sl_price

    # ==================================================================
    # PRIVATE -- Take Profit
    # ==================================================================

    def _get_tp_price(self, pos: PositionState) -> float:
        if pos.side == 'long':
            return pos.entry_price + (pos.tp_ticks * self.tick_size)
        else:
            return pos.entry_price - (pos.tp_ticks * self.tick_size)

    def _is_tp_hit(self, pos: PositionState, best_price: float, tp_price: float) -> bool:
        if pos.side == 'long':
            return best_price >= tp_price
        else:
            return best_price <= tp_price

    # ==================================================================
    # PRIVATE -- Trail Stop
    # ==================================================================

    def _update_trail(self, pos: PositionState, best_price: float,
                      band_context: dict = None):
        """Update trailing stop with band-aware tighten/widen."""
        if pos.side == 'long':
            favorable_move = (best_price - pos.entry_price) / self.tick_size
        else:
            favorable_move = (pos.entry_price - best_price) / self.tick_size

        if favorable_move < pos.trail_activation_ticks:
            return  # not enough profit to start trailing

        # Band-aware trail adjustment
        trail_mult = 1.0
        if band_context is not None:
            _sup = band_context.get('support_score', 0.0)
            _res = band_context.get('resistance_score', 0.0)

            if pos.side == 'long' and _res > 0.5:
                trail_mult = 0.6   # tighten at ceiling
            elif pos.side == 'long' and _sup > 0.5:
                trail_mult = 1.4   # widen at floor
            elif pos.side == 'short' and _sup > 0.5:
                trail_mult = 0.6   # tighten at floor
            elif pos.side == 'short' and _res > 0.5:
                trail_mult = 1.4   # widen at ceiling

        # Trail tightens as profit grows
        progress_ratio = favorable_move / max(1, pos.trail_activation_ticks)
        tightening = max(0.4, 1.0 - (progress_ratio - 1.0) * 0.15)
        trail_dist_ticks = pos.sl_ticks * tightening * trail_mult
        trail_dist = trail_dist_ticks * self.tick_size

        # Move trail (only tightens, never widens)
        if pos.side == 'long':
            new_trail = best_price - trail_dist
            pos.current_trail = max(pos.current_trail, new_trail)
        else:
            new_trail = best_price + trail_dist
            pos.current_trail = min(pos.current_trail, new_trail)

    def _is_trail_hit(self, pos: PositionState, worst_price: float) -> bool:
        if pos.side == 'long':
            return worst_price <= pos.current_trail
        else:
            return worst_price >= pos.current_trail

    # ==================================================================
    # PRIVATE -- Envelope Decay
    # ==================================================================

    def _check_envelope(self, pos: PositionState, bar_close: float,
                        net_force: float = 0.0,
                        band_context: dict = None) -> Optional[ExitResult]:
        """Half-life envelope decay with dynamic halflife.

        Halflife modulated by three fractal signals:
          1. Giveback ratio: shrinks when trade gives back from peak
          2. Net force: extends when force aligned, shrinks when adverse
          3. Band exhaustion: shrinks when multi-TF bands say move is exhausted
        """
        if not pos.envelope_active or pos.bars_held < self.envelope_min_bars:
            return None

        # Dynamic halflife: shrinks when trade is giving back from peak
        base_hl = self.envelope_half_life_bars
        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            current_ticks = (bar_close - pos.entry_price) / self.tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size
            current_ticks = (pos.entry_price - bar_close) / self.tick_size

        # Signal 1: giveback from peak
        hl_mult = 1.0
        if peak_ticks > 4:
            giveback_ratio = max(0, peak_ticks - current_ticks) / peak_ticks
            hl_mult *= max(0.5, 1.0 - giveback_ratio)

        # Signal 2: band exhaustion (fractal)
        # When bands across TFs say price is at resistance (LONG) or support (SHORT),
        # the move is exhausted → shrink halflife. Aligned → extend.
        if band_context is not None:
            sup = band_context.get('support_score', 0.0)
            res = band_context.get('resistance_score', 0.0)
            if pos.side == 'long':
                # Resistance = exhaustion for longs, support = wind at back
                exhaustion = res - sup
            else:
                # Support = exhaustion for shorts, resistance = wind at back
                exhaustion = sup - res
            # exhaustion > 0 → move exhausted → shrink halflife (0.5x at exhaustion=1)
            # exhaustion < 0 → move has room → extend halflife (1.5x at exhaustion=-1)
            band_mult = max(0.5, min(1.5, 1.0 - exhaustion * 0.5))
            hl_mult *= band_mult

        effective_hl = base_hl * max(0.3, hl_mult)

        # Decay factor
        decay = math.exp(-0.693 * pos.bars_held / max(1, effective_hl))

        # net_force modulation
        if net_force != 0.0:
            force_aligned = (
                (pos.side == 'long' and net_force > 0) or
                (pos.side == 'short' and net_force < 0)
            )
            if force_aligned:
                decay = min(decay * 1.3, 1.0)
            else:
                decay = decay * 0.7

        # Current envelope level
        initial_tp = pos.tp_ticks * self.tick_size
        floor = initial_tp * self.envelope_floor_pct
        current_envelope = floor + (initial_tp - floor) * decay
        pos.envelope_level = current_envelope

        # Only trigger after significant time (uses dynamic halflife)
        if pos.bars_held < effective_hl * 0.5:
            return None

        if pos.side == 'long':
            unrealized = bar_close - pos.entry_price
        else:
            unrealized = pos.entry_price - bar_close

        # Profitable but below decaying envelope floor
        if 0 < unrealized < current_envelope * 0.3:
            return ExitResult(
                action=ExitAction.ENVELOPE_DECAY,
                exit_price=bar_close,
                reason=f"Envelope decay: unrealized={unrealized:.2f} < envelope_floor",
                pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                bars_held=pos.bars_held,
                envelope_level=current_envelope,
            )

        return None

    # ==================================================================
    # PRIVATE -- Peak Giveback
    # ==================================================================

    def _check_peak_giveback(self, pos: PositionState, bar_close: float) -> Optional[ExitResult]:
        """Exit if trade reached a good peak then gave back most of the profit."""
        # How far did price get from entry (peak MFE in ticks)?
        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            current_ticks = (bar_close - pos.entry_price) / self.tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size
            current_ticks = (pos.entry_price - bar_close) / self.tick_size

        # Only trigger if the trade actually reached a meaningful peak
        if peak_ticks < self.giveback_min_mfe_ticks:
            return None

        # How much was given back?
        gave_back = peak_ticks - current_ticks
        if peak_ticks > 0 and gave_back / peak_ticks >= self.giveback_pct:
            return ExitResult(
                action=ExitAction.PEAK_GIVEBACK,
                exit_price=bar_close,
                reason=f"Peak giveback: peak={peak_ticks:.1f}t now={current_ticks:.1f}t gave_back={gave_back/peak_ticks:.0%}",
                pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                bars_held=pos.bars_held,
            )

        return None

    # ==================================================================
    # PRIVATE -- Band-Aware Exit
    # ==================================================================

    def _check_band_exit(self, pos: PositionState, bar_close: float,
                         band_context: dict = None) -> Optional[ExitResult]:
        """Band-aware urgent exit when support/resistance is broken."""
        if band_context is None:
            return None

        direction = band_context.get('direction')
        strength = band_context.get('strength', 0.0)

        if strength < 0.6:
            return None

        # LONG but bands say strong SHORT (support broken)
        if pos.side == 'long' and direction == 'short' and strength > 0.7:
            unrealized_ticks = (bar_close - pos.entry_price) / self.tick_size
            if unrealized_ticks < -2:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: LONG but support broken (str={strength:.2f})",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )

        # SHORT but bands say strong LONG (resistance broken)
        if pos.side == 'short' and direction == 'long' and strength > 0.7:
            unrealized_ticks = (pos.entry_price - bar_close) / self.tick_size
            if unrealized_ticks < -2:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: SHORT but resistance broken (str={strength:.2f})",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )

        return None

    # ==================================================================
    # PRIVATE -- Watchdog
    # ==================================================================

    def _check_watchdog(self, pos: PositionState, bar_close: float,
                        exit_signal: dict = None) -> Optional[ExitResult]:
        """Detect stuck trades going nowhere."""
        if pos.bars_held < self.watchdog_bar_threshold:
            return None

        # Adverse tick check
        if pos.side == 'long':
            adverse_ticks = (pos.entry_price - bar_close) / self.tick_size
        else:
            adverse_ticks = (bar_close - pos.entry_price) / self.tick_size

        if adverse_ticks > self.watchdog_tick_threshold:
            if pos.side == 'long':
                mfe_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            else:
                mfe_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size

            # Stuck: adverse and never made meaningful progress
            if mfe_ticks < pos.trail_activation_ticks * 0.5:
                # Also check workers_against from exit signal
                _workers_against = 0
                if exit_signal is not None:
                    _workers_against = exit_signal.get('workers_against', 0)

                if _workers_against >= self.watchdog_worker_threshold or mfe_ticks < 2:
                    return ExitResult(
                        action=ExitAction.WATCHDOG,
                        exit_price=bar_close,
                        reason=f"Watchdog: {adverse_ticks:.0f} ticks adverse, "
                               f"MFE only {mfe_ticks:.0f} ticks",
                        pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                        bars_held=pos.bars_held,
                    )

        return None

    # ==================================================================
    # PRIVATE -- Breakeven Lock
    # ==================================================================

    def _check_breakeven(self, pos: PositionState):
        """Lock trail to breakeven once profitable past activation threshold."""
        if pos.breakeven_locked:
            return

        if pos.side == 'long':
            favorable = (pos.peak_favorable - pos.entry_price) / self.tick_size
        else:
            favorable = (pos.entry_price - pos.peak_favorable) / self.tick_size

        if favorable >= pos.trail_activation_ticks * 0.6:
            if pos.side == 'long':
                be_level = pos.entry_price + (2 * self.tick_size)
                pos.current_trail = max(pos.current_trail, be_level)
            else:
                be_level = pos.entry_price - (2 * self.tick_size)
                pos.current_trail = min(pos.current_trail, be_level)
            pos.breakeven_locked = True

    # ==================================================================
    # PRIVATE -- Utilities
    # ==================================================================

    def _calc_pnl_ticks(self, pos: PositionState, exit_price: float) -> float:
        if pos.side == 'long':
            return (exit_price - pos.entry_price) / self.tick_size
        else:
            return (pos.entry_price - exit_price) / self.tick_size
