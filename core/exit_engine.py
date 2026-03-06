"""
Unified Exit Engine
===================
Single exit decision module used by both training orchestrator and live engine.
Ensures training metrics accurately reflect live execution behavior.

Usage:
    # In orchestrator (training)
    exit_eng = ExitEngine(mode='training', wave_rider=wr)

    # In live engine
    exit_eng = ExitEngine(mode='live', wave_rider=wr)

    # Same API, same logic, same results
    action = exit_eng.evaluate(position, bar, band_context)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple

class ExitAction(Enum):
    HOLD = 'hold'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    TRAIL_STOP = 'trail_stop'
    MAX_HOLD = 'max_hold'
    ENVELOPE_DECAY = 'envelope_decay'
    BAND_URGENT = 'band_urgent'
    WATCHDOG = 'watchdog'
    BREAKEVEN_LOCK = 'breakeven_lock'  # informational — tightens trail, doesn't exit

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
    """Minimal position representation — populated by caller."""
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
    max_hold_bars: int = 120

    # Dynamic state (updated each bar)
    current_trail: float = 0.0
    peak_favorable: float = 0.0
    bars_held: int = 0
    breakeven_locked: bool = False
    envelope_active: bool = False
    envelope_level: float = 0.0


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
        wave_rider = None,
        tick_size: float = 0.25,
        tick_value: float = 0.50,
    ):
        assert mode in ('training', 'live'), f"Invalid mode: {mode}"
        self.mode = mode
        self.wave_rider = wave_rider
        self.tick_size = tick_size
        self.tick_value = tick_value

        self.envelope_half_life_bars = 40
        self.envelope_floor_pct = 0.3
        self.watchdog_tick_threshold = 8
        self.watchdog_bar_threshold = 5
        self.watchdog_worker_threshold = 5

    def open_position(
        self,
        side: str,
        entry_price: float,
        entry_bar_index: int,
        template_id: int,
        lib_entry: dict,
    ) -> PositionState:
        """
        Initialize position with template-specific exit parameters.

        STOP SIZING: cluster-fitted metrics from pattern_library.
        Falls back to ATR only if cluster metrics are missing.
        """
        pos = PositionState(
            side=side,
            entry_price=entry_price,
            entry_bar_index=entry_bar_index,
            template_id=template_id,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
        )

        # ── Stop Loss sizing (cluster-fitted, not ATR) ────────
        # Priority: p25_mae × 3.0 → regression_sigma × 1.1 → ATR fallback
        p25_mae = lib_entry.get('p25_mae')
        reg_sigma = lib_entry.get('regression_sigma')
        atr = lib_entry.get('atr', 20.0)

        if p25_mae is not None and p25_mae > 0:
            pos.sl_ticks = p25_mae * 3.0
        elif reg_sigma is not None and reg_sigma > 0:
            pos.sl_ticks = (reg_sigma / self.tick_size) * 1.1
        else:
            pos.sl_ticks = atr * 2.0

        pos.sl_ticks = max(pos.sl_ticks, 8.0)
        pos.sl_ticks = min(pos.sl_ticks, 80.0)

        # ── Take Profit sizing (cascade) ──────────────────────
        # Priority: network_tp → mfe_coeff → p75_mfe → ATR fallback
        network_tp = lib_entry.get('network_tp')
        mfe_coeff = lib_entry.get('mfe_coeff')
        p75_mfe = lib_entry.get('p75_mfe')

        if network_tp is not None and network_tp > 0:
            pos.tp_ticks = network_tp
        elif mfe_coeff is not None and mfe_coeff > 0:
            pos.tp_ticks = mfe_coeff
        elif p75_mfe is not None and p75_mfe > 0:
            pos.tp_ticks = p75_mfe
        else:
            pos.tp_ticks = atr * 3.0

        pos.tp_ticks = max(pos.tp_ticks, 4.0)
        pos.tp_ticks = min(pos.tp_ticks, 200.0)

        # ── Trail activation ──────────────────────────────────
        if p25_mae is not None and p25_mae > 0:
            pos.trail_activation_ticks = p25_mae * 0.3
        else:
            pos.trail_activation_ticks = atr * 0.6
        pos.trail_activation_ticks = max(pos.trail_activation_ticks, 3.0)

        # ── Max hold (per-template) ───────────────────────────
        pos.max_hold_bars = lib_entry.get('max_hold_bars', 120)

        # ── Initial trail at entry ────────────────────────────
        if side == 'long':
            pos.current_trail = entry_price - (pos.sl_ticks * self.tick_size)
            pos.peak_favorable = entry_price
        else:
            pos.current_trail = entry_price + (pos.sl_ticks * self.tick_size)
            pos.peak_favorable = entry_price

        # ── Envelope initialization ───────────────────────────
        pos.envelope_active = True
        pos.envelope_level = pos.tp_ticks * self.tick_size

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
        worker_beliefs: dict = None,
        sub_bar_highs: list = None,
        sub_bar_lows: list = None,
    ) -> ExitResult:
        """
        Evaluate all exit conditions for current bar.

        IDENTICAL logic for training and live. The only difference:
        - Training may pass sub_bar_highs/lows for intra-bar wick checking
        - Live passes None for sub_bars → uses bar_high/bar_low instead

        Evaluation order (first trigger wins):
        1. Stop loss        2. Take profit       3. Watchdog
        4. Max hold         5. Band urgent exit   6. Envelope decay
        7. Trail stop       8. Breakeven lock     9. Wave rider signal
        10. HOLD
        """
        pos.bars_held = current_bar_index - pos.entry_bar_index

        # ── Determine worst/best price this bar ──────────────
        if sub_bar_highs is not None and sub_bar_lows is not None:
            worst_price = min(sub_bar_lows) if pos.side == 'long' else max(sub_bar_highs)
            best_price = max(sub_bar_highs) if pos.side == 'long' else min(sub_bar_lows)
        else:
            worst_price = bar_low if pos.side == 'long' else bar_high
            best_price = bar_high if pos.side == 'long' else bar_low

        # ── Update MFE tracking ──────────────────────────────
        if pos.side == 'long':
            pos.peak_favorable = max(pos.peak_favorable, best_price)
        else:
            pos.peak_favorable = min(pos.peak_favorable, best_price)

        # ── 1. STOP LOSS ─────────────────────────────────────
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

        # ── 2. TAKE PROFIT ───────────────────────────────────
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

        # ── 3. WATCHDOG ──────────────────────────────────────
        watchdog = self._check_watchdog(pos, bar_close, worker_beliefs)
        if watchdog is not None:
            return watchdog

        # ── 4. MAX HOLD ──────────────────────────────────────
        if pos.bars_held >= pos.max_hold_bars:
            return ExitResult(
                action=ExitAction.MAX_HOLD,
                exit_price=bar_close,
                reason=f"Max hold {pos.max_hold_bars} bars reached",
                pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                bars_held=pos.bars_held,
            )

        # ── 5. BAND-AWARE URGENT EXIT ────────────────────────
        band_result = self._check_band_exit(pos, bar_close, band_context)
        if band_result is not None:
            return band_result

        # ── 6. ENVELOPE DECAY ────────────────────────────────
        envelope_result = self._check_envelope(pos, bar_close, net_force)
        if envelope_result is not None:
            return envelope_result

        # ── 7. TRAIL STOP UPDATE + CHECK ─────────────────────
        self._update_trail(pos, best_price, band_context)
        if self._is_trail_hit(pos, worst_price):
            return ExitResult(
                action=ExitAction.TRAIL_STOP,
                exit_price=pos.current_trail,
                reason=f"Trail stop at {pos.current_trail:.2f}",
                pnl_ticks=self._calc_pnl_ticks(pos, pos.current_trail),
                bars_held=pos.bars_held,
                trail_level=pos.current_trail,
            )

        # ── 8. BREAKEVEN LOCK ────────────────────────────────
        self._check_breakeven(pos)

        # ── 9. WAVE RIDER EXIT SIGNAL ────────────────────────
        if self.wave_rider is not None:
            wr_signal = self.wave_rider.get_exit_signal(pos.side, pos.entry_price)
            if wr_signal and wr_signal.get('exit', False):
                return ExitResult(
                    action=ExitAction.TRAIL_STOP,
                    exit_price=bar_close,
                    reason=f"WaveRider: {wr_signal.get('reason', 'exit signal')}",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                    band_action=wr_signal.get('band_action', ''),
                )

        # ── 10. HOLD ─────────────────────────────────────────
        return ExitResult(
            action=ExitAction.HOLD,
            exit_price=0.0,
            reason='hold',
            bars_held=pos.bars_held,
            trail_level=pos.current_trail,
            envelope_level=pos.envelope_level,
            band_zone=band_context.get('band_summary', '') if band_context else '',
        )

    # ── PRIVATE: Stop Loss ────────────────────────────────────

    def _get_stop_price(self, pos):
        if pos.side == 'long':
            return pos.entry_price - (pos.sl_ticks * self.tick_size)
        return pos.entry_price + (pos.sl_ticks * self.tick_size)

    def _is_stopped(self, pos, worst_price, sl_price):
        if pos.side == 'long':
            return worst_price <= sl_price
        return worst_price >= sl_price

    # ── PRIVATE: Take Profit ──────────────────────────────────

    def _get_tp_price(self, pos):
        if pos.side == 'long':
            return pos.entry_price + (pos.tp_ticks * self.tick_size)
        return pos.entry_price - (pos.tp_ticks * self.tick_size)

    def _is_tp_hit(self, pos, best_price, tp_price):
        if pos.side == 'long':
            return best_price >= tp_price
        return best_price <= tp_price

    # ── PRIVATE: Trail Stop ───────────────────────────────────

    def _update_trail(self, pos, best_price, band_context=None):
        if pos.side == 'long':
            favorable_move = (best_price - pos.entry_price) / self.tick_size
        else:
            favorable_move = (pos.entry_price - best_price) / self.tick_size

        if favorable_move < pos.trail_activation_ticks:
            return

        # Band-aware trail adjustment
        trail_mult = 1.0
        if band_context is not None:
            at_resistance = band_context.get('at_resistance', False)
            at_support = band_context.get('at_support', False)
            if pos.side == 'long' and at_resistance:
                trail_mult = 0.6   # tighten at ceiling
            elif pos.side == 'long' and at_support:
                trail_mult = 1.4   # widen at floor
            elif pos.side == 'short' and at_support:
                trail_mult = 0.6
            elif pos.side == 'short' and at_resistance:
                trail_mult = 1.4

        progress_ratio = favorable_move / pos.trail_activation_ticks
        tightening = max(0.4, 1.0 - (progress_ratio - 1.0) * 0.15)
        trail_dist = pos.sl_ticks * tightening * trail_mult * self.tick_size

        if pos.side == 'long':
            new_trail = best_price - trail_dist
            pos.current_trail = max(pos.current_trail, new_trail)
        else:
            new_trail = best_price + trail_dist
            pos.current_trail = min(pos.current_trail, new_trail)

    def _is_trail_hit(self, pos, worst_price):
        if pos.side == 'long':
            return worst_price <= pos.current_trail
        return worst_price >= pos.current_trail

    # ── PRIVATE: Envelope Decay ───────────────────────────────

    def _check_envelope(self, pos, bar_close, net_force=0.0):
        if not pos.envelope_active or pos.bars_held < 5:
            return None

        decay = np.exp(-0.693 * pos.bars_held / self.envelope_half_life_bars)

        if net_force != 0.0:
            force_aligned = (
                (pos.side == 'long' and net_force > 0) or
                (pos.side == 'short' and net_force < 0)
            )
            decay = min(decay * 1.3, 1.0) if force_aligned else decay * 0.7

        initial_tp = pos.tp_ticks * self.tick_size
        floor = initial_tp * self.envelope_floor_pct
        current_envelope = floor + (initial_tp - floor) * decay
        pos.envelope_level = current_envelope

        if pos.bars_held < self.envelope_half_life_bars * 0.5:
            return None

        if pos.side == 'long':
            unrealized = bar_close - pos.entry_price
        else:
            unrealized = pos.entry_price - bar_close

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

    # ── PRIVATE: Band-Aware Exit ──────────────────────────────

    def _check_band_exit(self, pos, bar_close, band_context=None):
        if band_context is None:
            return None

        direction = band_context.get('direction')
        strength = band_context.get('strength', 0.0)
        if strength < 0.6:
            return None

        # LONG but multi-TF support broken
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

        # SHORT but multi-TF resistance broken
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

    # ── PRIVATE: Watchdog ─────────────────────────────────────

    def _check_watchdog(self, pos, bar_close, worker_beliefs=None):
        if pos.bars_held < self.watchdog_bar_threshold:
            return None

        if pos.side == 'long':
            adverse_ticks = (pos.entry_price - bar_close) / self.tick_size
        else:
            adverse_ticks = (bar_close - pos.entry_price) / self.tick_size

        if adverse_ticks > self.watchdog_tick_threshold:
            if pos.side == 'long':
                mfe_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            else:
                mfe_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size

            if mfe_ticks < pos.trail_activation_ticks * 0.5:
                return ExitResult(
                    action=ExitAction.WATCHDOG,
                    exit_price=bar_close,
                    reason=f"Watchdog: {adverse_ticks:.0f} ticks adverse, MFE only {mfe_ticks:.0f}",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                )

        if worker_beliefs is not None:
            flipped = sum(
                1 for wid, b in worker_beliefs.items()
                if (pos.side == 'long' and b.get('side') == 'short') or
                   (pos.side == 'short' and b.get('side') == 'long')
            )
            if flipped >= self.watchdog_worker_threshold:
                return ExitResult(
                    action=ExitAction.WATCHDOG,
                    exit_price=bar_close,
                    reason=f"Watchdog: {flipped}/{len(worker_beliefs)} workers flipped",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                )
        return None

    # ── PRIVATE: Breakeven Lock ───────────────────────────────

    def _check_breakeven(self, pos):
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

    # ── PRIVATE: Utilities ────────────────────────────────────

    def _calc_pnl_ticks(self, pos, exit_price):
        if pos.side == 'long':
            return (exit_price - pos.entry_price) / self.tick_size
        return (pos.entry_price - exit_price) / self.tick_size
