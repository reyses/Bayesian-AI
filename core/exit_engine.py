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

        # Breakeven lock: fixed activation threshold (ticks of favorable excursion)
        # Analysis EE found 4 ticks optimal: 77% of OOS losers reached MFE >= 4
        self.be_activation_ticks = 4

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
        sl_ticks: float,
        tp_ticks: float,
        trail_ticks: float = 0.0,
        trail_activation_ticks: float = 0.0,
        max_hold_bars: int = 120,
        lib_entry: dict = None,
    ) -> PositionState:
        """
        Initialize a new position with pre-computed exit parameters.

        Sizing is computed by ExecutionEngine._compute_sizing() — this method
        does NOT recompute. It sets up position state and absolute price levels.

        If lib_entry is passed, p75_mfe_bar is used for max_hold_bars override.
        """
        # Per-template exit timescale override
        if lib_entry:
            _p75_bar = lib_entry.get('p75_mfe_bar', 0.0)
            if _p75_bar > 0:
                max_hold_bars = int(_p75_bar * 2.5)
            else:
                max_hold_bars = lib_entry.get('max_hold_bars', max_hold_bars)

        # Anchor: template historical MFE (price + time)
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

        # Absolute price levels
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

        # -- 1. STOP LOSS --
        sl_price = self._get_stop_price(pos)
        if self._is_stopped(pos, worst_price, sl_price):
            return ExitResult(
                action=ExitAction.STOP_LOSS,
                exit_price=sl_price,
                reason=f"SL hit at {sl_price:.2f} (worst={worst_price:.2f})",
                pnl_ticks=self._calc_pnl_ticks(pos, sl_price),
                bars_held=pos.bars_held,
                trail_level=pos.stop_loss,
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
                trail_level=pos.stop_loss,
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

        # -- 5b. BREAKEVEN LOCK (adjust SL before envelope/giveback) --
        self._check_breakeven(pos)

        # -- 6. ENVELOPE DECAY --
        envelope_result = self._check_envelope(pos, bar_close, net_force, band_context, noise_ticks)
        if envelope_result is not None:
            return envelope_result

        # -- 6b. PEAK GIVEBACK --
        giveback_result = self._check_peak_giveback(pos, bar_close, exit_signal, noise_ticks)
        if giveback_result is not None:
            return giveback_result

        # -- 9. Exit signal from belief network (tighten/urgent) --
        if exit_signal is not None and exit_signal.get('urgent_exit', False):
            return ExitResult(
                action=ExitAction.TRAIL_STOP,
                exit_price=bar_close,
                reason=f"Belief flip: {exit_signal.get('reason', 'urgent')}",
                pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                bars_held=pos.bars_held,
                trail_level=pos.stop_loss,
            )

        # -- 10. HOLD --
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
    # PRIVATE -- Stop Loss
    # ==================================================================

    def _get_stop_price(self, pos: PositionState) -> float:
        # Use pos.stop_loss directly — it may have been tightened by
        # breakeven lock or other adjustments.
        return pos.stop_loss

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

    # ==================================================================
    # PRIVATE -- Envelope Decay
    # ==================================================================

    def _check_envelope(self, pos: PositionState, bar_close: float,
                        net_force: float = 0.0,
                        band_context: dict = None,
                        noise_ticks: float = 0.0) -> Optional[ExitResult]:
        """Half-life envelope decay with dynamic halflife.

        Halflife modulated by three fractal signals:
          1. Giveback ratio: shrinks when trade gives back from peak
          2. Net force: extends when force aligned, shrinks when adverse
          3. Band exhaustion: shrinks when multi-TF bands say move is exhausted
        """
        if not pos.envelope_active or pos.bars_held < self.envelope_min_bars:
            return None

        # Dynamic halflife: per-template when available, else global default
        # max_hold_bars is set from template's p75_mfe_bar at entry; /5 gives base HL
        if pos.max_hold_bars > 0 and pos.max_hold_bars != 120:
            base_hl = max(8.0, pos.max_hold_bars / 5.0)
        else:
            base_hl = self.envelope_half_life_bars
        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            current_ticks = (bar_close - pos.entry_price) / self.tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size
            current_ticks = (pos.entry_price - bar_close) / self.tick_size

        # Noise gate: trade still within normal market breathing — don't exit
        if noise_ticks > 0 and peak_ticks < noise_ticks:
            return None

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

        # Signal 3: anchor time patience (trade before expected MFE time)
        if pos.anchor_mfe_bars > 0 and pos.bars_held < pos.anchor_mfe_bars:
            anchor_progress = pos.bars_held / pos.anchor_mfe_bars
            # 2x patience at trade start, tapering to 1x at expected MFE time
            hl_mult *= (2.0 - anchor_progress)

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

        # Current envelope level (noise-aware floor)
        initial_tp = pos.tp_ticks * self.tick_size
        noise_floor = noise_ticks * self.tick_size if noise_ticks > 0 else 0
        floor = max(initial_tp * self.envelope_floor_pct, noise_floor)
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

    def _get_giveback_threshold(self, peak_ticks: float, noise_ticks: float = 0.0) -> float:
        """Tiered giveback: protect big winners aggressively,
        give small winners room to develop.

        Uses dynamic noise floor when available — the MFE must exceed
        the current market noise level before giveback activates.
        This prevents exiting on normal intra-wave pullbacks.

        Peak MFE (ticks)  →  Giveback trigger
        ────────────────     ────────────────
        2× noise+         →  40% (aggressive protection)
        1× noise - 2×     →  self.giveback_pct (self-tuned, ~55-70%)
        < noise           →  disabled (move within noise floor)
        """
        # Dynamic noise floor: use measured noise if available, else static default
        min_mfe = max(self.giveback_min_mfe_ticks, noise_ticks) if noise_ticks > 0 else self.giveback_min_mfe_ticks

        if peak_ticks >= min_mfe * 2:
            return 0.40
        elif peak_ticks >= min_mfe:
            return self.giveback_pct
        else:
            return 1.01  # >100% = never triggers

    def _check_peak_giveback(self, pos: PositionState, bar_close: float,
                             exit_signal: dict = None,
                             noise_ticks: float = 0.0) -> Optional[ExitResult]:
        """Exit if trade reached a good peak then gave back most of the profit.
        When 30m worker flips against trade direction (slow_flip_tighten),
        threshold tightens by 15pp to protect profit sooner.

        noise_ticks: dynamic noise floor from MarketState.swing_noise_ticks.
        If > 0, overrides the static giveback_min_mfe_ticks threshold —
        giveback only activates after MFE exceeds the current noise level.
        """
        # -- 30m slow-flip detection (sticky once set) --
        if exit_signal and exit_signal.get('slow_flip_tighten'):
            pos.slow_flip_active = True

        # How far did price get from entry (peak MFE in ticks)?
        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            current_ticks = (bar_close - pos.entry_price) / self.tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size
            current_ticks = (pos.entry_price - bar_close) / self.tick_size

        if peak_ticks <= 0:
            return None

        # Anchor patience: trade still developing (before expected time + below expected MFE)
        if (pos.anchor_mfe_ticks > 0 and pos.anchor_mfe_bars > 0
                and pos.bars_held < pos.anchor_mfe_bars
                and peak_ticks < pos.anchor_mfe_ticks * 0.3):
            return None

        # Tiered threshold based on peak size
        threshold = self._get_giveback_threshold(peak_ticks)

        # 30m flip tightens threshold by 15pp (protect profit when higher TF turns)
        if pos.slow_flip_active and threshold < 1.0:
            threshold = max(0.25, threshold - 0.15)

        gave_back = peak_ticks - current_ticks
        if gave_back / peak_ticks >= threshold:
            return ExitResult(
                action=ExitAction.PEAK_GIVEBACK,
                exit_price=bar_close,
                reason=f"Peak giveback: peak={peak_ticks:.1f}t now={current_ticks:.1f}t "
                       f"gave_back={gave_back/peak_ticks:.0%} (tier={threshold:.0%})",
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
        """Lock stop to breakeven once MFE exceeds be_activation_ticks.

        Uses a fixed threshold (default 4 ticks) instead of trail_activation.
        The old threshold (trail_activation * 0.6) was effectively unreachable
        when trail_activation was tied to enormous tp_ticks values.
        """
        if pos.breakeven_locked:
            return

        if pos.side == 'long':
            favorable = (pos.peak_favorable - pos.entry_price) / self.tick_size
        else:
            favorable = (pos.entry_price - pos.peak_favorable) / self.tick_size

        if favorable >= self.be_activation_ticks:
            # Lock SL to entry + 1 tick buffer (avoid slippage-induced loss)
            buffer = 1 * self.tick_size
            if pos.side == 'long':
                be_level = pos.entry_price + buffer
                pos.stop_loss = max(pos.stop_loss, be_level)
            else:
                be_level = pos.entry_price - buffer
                pos.stop_loss = min(pos.stop_loss, be_level)
            pos.breakeven_locked = True

    # ==================================================================
    # CME Maintenance Flatten
    # ==================================================================

    # CME daily halt: 17:00-18:00 ET. Flatten 15 min before.
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
    # PRIVATE -- Utilities
    # ==================================================================

    def _calc_pnl_ticks(self, pos: PositionState, exit_price: float) -> float:
        if pos.side == 'long':
            return (exit_price - pos.entry_price) / self.tick_size
        else:
            return (pos.entry_price - exit_price) / self.tick_size
