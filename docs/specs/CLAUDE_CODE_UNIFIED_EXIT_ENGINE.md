# CLAUDE CODE INSTRUCTIONS: Unified Exit Engine
# Single module called by BOTH orchestrator (training) and live engine
# Date: March 6, 2026
# Priority: HIGH — exit mismatch invalidates all training metrics

## THE PROBLEM

Training and live use completely different exit logic:
- Different stop sizing (cluster-fitted vs ATR)
- Different TP cascades
- Different trail activation thresholds
- Band-aware exits only in live
- Envelope decay only in live
- 1s wick checking only in training
- Auto-TP re-entry only in live

Training PnL is meaningless because exits don't match execution.

## THE SOLUTION

One module: `core/exit_engine.py`
One class: `ExitEngine`
Two modes: `mode='training'` or `mode='live'`

Both orchestrator and live instantiate the same class. Every exit decision
flows through the same code path. Mode flag controls only DATA AVAILABILITY,
not logic.

---

## FILE: `core/exit_engine.py` (new file, ~400-500 lines)

```python
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
    # Diagnostic fields for oracle_trade_log
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
    sl_ticks: float = 0.0       # cluster-fitted stop loss distance
    tp_ticks: float = 0.0       # target profit distance
    trail_activation_ticks: float = 0.0  # when trailing starts
    max_hold_bars: int = 120    # per-template max hold
    
    # Dynamic state (updated each bar)
    current_trail: float = 0.0  # current trailing stop level
    peak_favorable: float = 0.0 # best price seen (MFE tracking)
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
        mode: str = 'live',           # 'training' or 'live'
        wave_rider = None,            # existing WaveRider instance
        tick_size: float = 0.25,
        tick_value: float = 0.50,
    ):
        assert mode in ('training', 'live'), f"Invalid mode: {mode}"
        self.mode = mode
        self.wave_rider = wave_rider
        self.tick_size = tick_size
        self.tick_value = tick_value
        
        # ── Envelope decay parameters ─────────────────────────
        self.envelope_half_life_bars = 40    # bars until envelope shrinks 50%
        self.envelope_floor_pct = 0.3        # minimum envelope as % of initial
        
        # ── Watchdog parameters ───────────────────────────────
        self.watchdog_tick_threshold = 8     # ticks of adverse without progress
        self.watchdog_bar_threshold = 5      # bars without progress
        self.watchdog_worker_threshold = 5   # workers flipping against
    
    # ══════════════════════════════════════════════════════════════
    # PUBLIC API — called identically by orchestrator and live
    # ══════════════════════════════════════════════════════════════
    
    def open_position(
        self,
        side: str,
        entry_price: float,
        entry_bar_index: int,
        template_id: int,
        lib_entry: dict,
    ) -> PositionState:
        """
        Initialize a new position with template-specific exit parameters.
        
        STOP SIZING: Always uses cluster-fitted metrics from pattern_library.
        Falls back to ATR-based only if cluster metrics are missing.
        
        Args:
            side: 'long' or 'short'
            entry_price: fill price
            entry_bar_index: current bar index (for max_hold tracking)
            template_id: matched template ID
            lib_entry: pattern_library[template_id] dict
        
        Returns:
            PositionState ready for evaluate() calls
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
        atr = lib_entry.get('atr', 20.0)  # only as last resort
        
        if p25_mae is not None and p25_mae > 0:
            pos.sl_ticks = p25_mae * 3.0
        elif reg_sigma is not None and reg_sigma > 0:
            pos.sl_ticks = (reg_sigma / self.tick_size) * 1.1
        else:
            pos.sl_ticks = atr * 2.0  # ATR fallback, wider for safety
        
        pos.sl_ticks = max(pos.sl_ticks, 8.0)   # floor: 8 ticks minimum
        pos.sl_ticks = min(pos.sl_ticks, 80.0)   # ceiling: 80 ticks max
        
        # ── Take Profit sizing (cascade) ──────────────────────
        # Priority: network_tp → mfe_coeff → p75_mfe → ATR fallback
        network_tp = lib_entry.get('network_tp')
        mfe_coeff = lib_entry.get('mfe_coeff')
        p75_mfe = lib_entry.get('p75_mfe')
        
        if network_tp is not None and network_tp > 0:
            pos.tp_ticks = network_tp
        elif mfe_coeff is not None and mfe_coeff > 0:
            pos.tp_ticks = mfe_coeff  # already in ticks from clustering
        elif p75_mfe is not None and p75_mfe > 0:
            pos.tp_ticks = p75_mfe
        else:
            pos.tp_ticks = atr * 3.0  # ATR fallback
        
        pos.tp_ticks = max(pos.tp_ticks, 4.0)
        pos.tp_ticks = min(pos.tp_ticks, 200.0)
        
        # ── Trail activation (when trailing stop engages) ─────
        # Priority: p25_mae × 0.3 → ATR × 0.6
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
        pos.envelope_level = pos.tp_ticks * self.tick_size  # initial envelope = TP distance
        
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
        sub_bar_highs: list = None,   # 1s highs within this bar (training only)
        sub_bar_lows: list = None,    # 1s lows within this bar (training only)
    ) -> ExitResult:
        """
        Evaluate all exit conditions for current bar.
        
        IDENTICAL logic for training and live. The only difference:
        - Training may pass sub_bar_highs/lows for intra-bar wick checking
        - Live passes None for sub_bars → uses bar_high/bar_low instead
        
        Evaluation order (first trigger wins):
        1. Stop loss (hard stop — checked against worst price in bar)
        2. Take profit (checked against best price in bar)
        3. Watchdog (stuck trade detection)
        4. Max hold (time-based forced exit)
        5. Band-aware urgent exit (support/resistance broken)
        6. Envelope decay (time-decaying profit target)
        7. Trail stop update + check
        8. Breakeven lock (informational — tightens trail)
        9. Wave rider exit signal
        10. HOLD (no exit)
        
        Args:
            pos: current PositionState
            bar_high/low/close: current bar OHLC (both modes use H/L, not just close)
            current_bar_index: for bars_held calculation
            band_context: from get_band_confluence() — optional
            net_force: F_net from field engine — for envelope modulation
            worker_beliefs: TBN worker states — for watchdog
            sub_bar_highs/lows: intra-bar 1s data (training only, None in live)
        
        Returns:
            ExitResult with action and diagnostics
        """
        pos.bars_held = current_bar_index - pos.entry_bar_index
        
        # ── Determine worst/best price this bar ──────────────
        # If sub-bar data available (training), use it for precision
        # Otherwise use bar high/low (live — this is MORE ACCURATE than
        # only using close, which was the old live behavior)
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
                exit_price=sl_price,  # assume filled at stop level
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
        
        # ── 3. WATCHDOG (stuck trade) ────────────────────────
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
            wr_signal = self.wave_rider.get_exit_signal(
                pos.side, pos.entry_price
            )
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
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Stop Loss
    # ══════════════════════════════════════════════════════════════
    
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
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Take Profit
    # ══════════════════════════════════════════════════════════════
    
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
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Trail Stop
    # ══════════════════════════════════════════════════════════════
    
    def _update_trail(
        self,
        pos: PositionState,
        best_price: float,
        band_context: dict = None,
    ):
        """
        Update trailing stop. Incorporates band-aware tighten/widen.
        
        Trail only activates after price moves trail_activation_ticks
        in favorable direction.
        """
        # Check if trail should activate
        if pos.side == 'long':
            favorable_move = (best_price - pos.entry_price) / self.tick_size
        else:
            favorable_move = (pos.entry_price - best_price) / self.tick_size
        
        if favorable_move < pos.trail_activation_ticks:
            return  # not enough profit to start trailing
        
        # ── Band-aware trail adjustment ───────────────────────
        trail_mult = 1.0
        if band_context is not None:
            at_resistance = band_context.get('at_resistance', False)
            at_support = band_context.get('at_support', False)
            
            if pos.side == 'long' and at_resistance:
                trail_mult = 0.6   # tighten — we're at a ceiling
            elif pos.side == 'long' and at_support:
                trail_mult = 1.4   # widen — give room at floor
            elif pos.side == 'short' and at_support:
                trail_mult = 0.6   # tighten — we're at a floor
            elif pos.side == 'short' and at_resistance:
                trail_mult = 1.4   # widen — give room at ceiling
        
        # Trail distance: starts at SL distance, tightens as profit grows
        # At activation: trail_dist = sl_ticks
        # At 2x activation: trail_dist = sl_ticks * 0.7
        # At 3x activation: trail_dist = sl_ticks * 0.5
        progress_ratio = favorable_move / pos.trail_activation_ticks
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
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Envelope Decay
    # ══════════════════════════════════════════════════════════════
    
    def _check_envelope(
        self,
        pos: PositionState,
        bar_close: float,
        net_force: float = 0.0,
    ) -> Optional[ExitResult]:
        """
        Half-life envelope decay with F_net modulation.
        
        The envelope is a time-decaying profit target. As the trade ages,
        the envelope shrinks toward a floor. If price hasn't reached the
        decayed envelope level, exit (the edge is expiring).
        
        F_net modulation: if net_force agrees with position direction,
        decay slows (the market is pushing our way). If it disagrees,
        decay accelerates.
        """
        if not pos.envelope_active or pos.bars_held < 5:
            return None
        
        # Decay factor (exponential half-life)
        decay = np.exp(-0.693 * pos.bars_held / self.envelope_half_life_bars)
        
        # F_net modulation: favorable force slows decay, adverse accelerates
        if net_force != 0.0:
            force_aligned = (
                (pos.side == 'long' and net_force > 0) or
                (pos.side == 'short' and net_force < 0)
            )
            if force_aligned:
                decay = min(decay * 1.3, 1.0)   # slow decay
            else:
                decay = decay * 0.7               # accelerate decay
        
        # Current envelope level (decays from initial TP toward floor)
        initial_tp = pos.tp_ticks * self.tick_size
        floor = initial_tp * self.envelope_floor_pct
        current_envelope = floor + (initial_tp - floor) * decay
        pos.envelope_level = current_envelope
        
        # Check: is unrealized P&L below the decayed envelope?
        # Only triggers after significant time (bars_held > half_life * 0.5)
        if pos.bars_held < self.envelope_half_life_bars * 0.5:
            return None
        
        if pos.side == 'long':
            unrealized = bar_close - pos.entry_price
        else:
            unrealized = pos.entry_price - bar_close
        
        # If we're profitable but below the decaying envelope, AND
        # the envelope has decayed significantly, time to exit
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
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Band-Aware Exit
    # ══════════════════════════════════════════════════════════════
    
    def _check_band_exit(
        self,
        pos: PositionState,
        bar_close: float,
        band_context: dict = None,
    ) -> Optional[ExitResult]:
        """
        Band-aware urgent exit.
        
        If LONG and multi-TF bands show resistance broken (price above
        all bands) → unlikely to revert further, take profit.
        
        If LONG and support broken (price below all bands) → urgent exit,
        the floor fell out.
        
        Same logic inverted for SHORT.
        """
        if band_context is None:
            return None
        
        direction = band_context.get('direction')
        strength = band_context.get('strength', 0.0)
        
        # Only trigger on strong signals
        if strength < 0.6:
            return None
        
        # LONG position but bands say strong SHORT (support broken)
        if pos.side == 'long' and direction == 'short' and strength > 0.7:
            # Only urgent if we're also in loss
            unrealized_ticks = (bar_close - pos.entry_price) / self.tick_size
            if unrealized_ticks < -2:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: LONG but multi-TF support broken (str={strength:.2f})",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )
        
        # SHORT position but bands say strong LONG (resistance broken)
        if pos.side == 'short' and direction == 'long' and strength > 0.7:
            unrealized_ticks = (pos.entry_price - bar_close) / self.tick_size
            if unrealized_ticks < -2:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: SHORT but multi-TF resistance broken (str={strength:.2f})",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )
        
        return None
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Watchdog
    # ══════════════════════════════════════════════════════════════
    
    def _check_watchdog(
        self,
        pos: PositionState,
        bar_close: float,
        worker_beliefs: dict = None,
    ) -> Optional[ExitResult]:
        """
        Watchdog: detect stuck trades going nowhere.
        
        Triggers if:
        - Adverse ticks > threshold AND bars without progress > threshold
        - OR: majority of TBN workers have flipped against position
        """
        if pos.bars_held < self.watchdog_bar_threshold:
            return None
        
        # Adverse tick check
        if pos.side == 'long':
            adverse_ticks = (pos.entry_price - bar_close) / self.tick_size
        else:
            adverse_ticks = (bar_close - pos.entry_price) / self.tick_size
        
        if adverse_ticks > self.watchdog_tick_threshold:
            # Check if MFE has progressed recently
            if pos.side == 'long':
                mfe_ticks = (pos.peak_favorable - pos.entry_price) / self.tick_size
            else:
                mfe_ticks = (pos.entry_price - pos.peak_favorable) / self.tick_size
            
            # Stuck: adverse and never made meaningful progress
            if mfe_ticks < pos.trail_activation_ticks * 0.5:
                return ExitResult(
                    action=ExitAction.WATCHDOG,
                    exit_price=bar_close,
                    reason=f"Watchdog: {adverse_ticks:.0f} ticks adverse, "
                           f"MFE only {mfe_ticks:.0f} ticks",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                )
        
        # Worker belief flip check
        if worker_beliefs is not None:
            flipped = 0
            total = 0
            for wid, belief in worker_beliefs.items():
                total += 1
                if pos.side == 'long' and belief.get('side') == 'short':
                    flipped += 1
                elif pos.side == 'short' and belief.get('side') == 'long':
                    flipped += 1
            
            if total > 0 and flipped >= self.watchdog_worker_threshold:
                return ExitResult(
                    action=ExitAction.WATCHDOG,
                    exit_price=bar_close,
                    reason=f"Watchdog: {flipped}/{total} workers flipped against",
                    pnl_ticks=self._calc_pnl_ticks(pos, bar_close),
                    bars_held=pos.bars_held,
                )
        
        return None
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Breakeven Lock
    # ══════════════════════════════════════════════════════════════
    
    def _check_breakeven(self, pos: PositionState):
        """
        Lock trail to breakeven once profitable past activation threshold.
        Informational — modifies trail level, doesn't trigger exit.
        """
        if pos.breakeven_locked:
            return
        
        if pos.side == 'long':
            favorable = (pos.peak_favorable - pos.entry_price) / self.tick_size
        else:
            favorable = (pos.entry_price - pos.peak_favorable) / self.tick_size
        
        # Lock breakeven at 60% of trail activation (conservative)
        if favorable >= pos.trail_activation_ticks * 0.6:
            if pos.side == 'long':
                be_level = pos.entry_price + (2 * self.tick_size)  # 2 ticks above entry
                pos.current_trail = max(pos.current_trail, be_level)
            else:
                be_level = pos.entry_price - (2 * self.tick_size)
                pos.current_trail = min(pos.current_trail, be_level)
            pos.breakeven_locked = True
    
    # ══════════════════════════════════════════════════════════════
    # PRIVATE — Utilities
    # ══════════════════════════════════════════════════════════════
    
    def _calc_pnl_ticks(self, pos: PositionState, exit_price: float) -> float:
        if pos.side == 'long':
            return (exit_price - pos.entry_price) / self.tick_size
        else:
            return (pos.entry_price - exit_price) / self.tick_size
```

---

## INTEGRATION: Orchestrator

### File: `training/orchestrator.py`

**FIND** the existing exit handling block in `run_forward_pass()`.
Look for where `wave_rider.update_trail()`, `check_stops_hilo()`,
and `get_exit_signal()` are called.

**REPLACE** with:

```python
# ── Initialize unified exit engine (once, before day loop) ────
from core.exit_engine import ExitEngine, ExitAction
_exit_engine = ExitEngine(
    mode='training',
    wave_rider=self.wave_rider,
    tick_size=self.tick_size,
    tick_value=self.tick_value,
)

# ... inside the bar loop, when a position is open:

# ── Open position via exit engine ─────────────────────────────
# (replace existing position initialization)
_pos_state = _exit_engine.open_position(
    side=side,
    entry_price=entry_price,
    entry_bar_index=bar_idx,
    template_id=best_tid,
    lib_entry=lib_entry,
)

# ... each bar while position is held:

# Get band context from TBN (if band context module implemented)
_band_ctx = None
if hasattr(self, 'tbn') and hasattr(self.tbn, 'get_band_confluence'):
    _band_ctx = self.tbn.get_band_confluence()

# Get net force from field engine
_net_force = getattr(current_state, 'net_force', 0.0)  # renamed from F_net

# Get sub-bar data for 1s wick checking (training has this)
_sub_highs = None
_sub_lows = None
if hasattr(bar_data, 'sub_bar_highs'):
    _sub_highs = bar_data.sub_bar_highs
    _sub_lows = bar_data.sub_bar_lows

# Evaluate exit
_exit_result = _exit_engine.evaluate(
    pos=_pos_state,
    bar_high=bar_data.high,
    bar_low=bar_data.low,
    bar_close=bar_data.close,
    current_bar_index=bar_idx,
    band_context=_band_ctx,
    net_force=_net_force,
    worker_beliefs=None,  # populate if available
    sub_bar_highs=_sub_highs,
    sub_bar_lows=_sub_lows,
)

if _exit_result.action != ExitAction.HOLD:
    # Close position
    exit_price = _exit_result.exit_price
    exit_reason = _exit_result.reason
    pnl_ticks = _exit_result.pnl_ticks
    # ... continue with existing trade recording logic
```

---

## INTEGRATION: Live Engine

### File: `live/live_engine.py`

**FIND** the existing exit handling in `_manage_position()` or equivalent.

**REPLACE** with same pattern:

```python
# ── Initialize (once, in __init__ or _load_checkpoints) ──────
from core.exit_engine import ExitEngine, ExitAction
self._exit_engine = ExitEngine(
    mode='live',
    wave_rider=self._wave_rider,
    tick_size=self._tick_size,
    tick_value=self._tick_value,
)

# ── On entry ─────────────────────────────────────────────────
self._pos_state = self._exit_engine.open_position(
    side=side,
    entry_price=fill_price,
    entry_bar_index=self._bar_count,
    template_id=template_id,
    lib_entry=self._pattern_library[template_id],
)

# ── Each bar ─────────────────────────────────────────────────
_band_ctx = None
if hasattr(self._tbn, 'get_band_confluence'):
    _band_ctx = self._tbn.get_band_confluence()

_exit_result = self._exit_engine.evaluate(
    pos=self._pos_state,
    bar_high=bar.high,
    bar_low=bar.low,         # KEY: use high/low, not just close
    bar_close=bar.close,
    current_bar_index=self._bar_count,
    band_context=_band_ctx,
    net_force=getattr(self._current_state, 'net_force', 0.0),
    worker_beliefs=self._tbn.get_worker_beliefs() if self._tbn else None,
    sub_bar_highs=None,      # live doesn't have sub-bar data
    sub_bar_lows=None,
)

if _exit_result.action != ExitAction.HOLD:
    self._submit_exit_order(
        exit_price=_exit_result.exit_price,
        reason=_exit_result.reason,
    )
```

---

## WHAT GETS DELETED

After integration, remove these DUPLICATE exit functions:

### From `training/orchestrator.py`:
- `check_stops_hilo()` — replaced by exit_engine.evaluate() with sub_bar data
- Inline trail update logic — replaced by exit_engine._update_trail()
- Inline TP/SL checks — replaced by exit_engine evaluate cascade
- ATR-based stop sizing — replaced by cluster-fitted sizing

### From `live/live_engine.py`:
- `_manage_trail()` — replaced by exit_engine._update_trail()
- `_check_envelope_decay()` — replaced by exit_engine._check_envelope()
- `_check_band_exit()` — replaced by exit_engine._check_band_exit()
- `_check_watchdog()` — replaced by exit_engine._check_watchdog()
- ATR-based stop/trail/TP sizing — replaced by cluster-fitted

### From `training/wave_rider.py`:
- Keep `get_exit_signal()` — the exit engine calls it as one input
- Keep `update_trail()` — but the exit engine's trail logic takes priority
  (wave_rider trail becomes supplementary, not primary)

---

## VERIFICATION

### Test 1: Identical behavior check (before removing old code)
```bash
# Run forward pass with BOTH old and new exit logic
# Compare exit decisions bar-by-bar
# They should differ (that's expected — the old code was wrong)
# But the PATTERN should make sense:
#   - More band_urgent exits (new feature in training)
#   - More envelope_decay exits (new feature in training)
#   - Slightly different trail levels (band-aware tightening)
#   - Same SL exits if sub_bar data is passed (wick checking preserved)
```

### Test 2: Training/Live parity
```bash
# Run forward pass on a single day with exit_engine (training mode)
# Run live dry-run on the same day's data with exit_engine (live mode)
# Exit decisions should be IDENTICAL for the same bars
# (minus sub_bar precision — training may catch a few more wick stops)
```

### Test 3: PnL comparison
```bash
# Compare forward pass PnL with old exits vs new exits
# Expected: slightly lower PnL (envelope decay + band urgency = more exits)
# But REALISTIC — matches what live would actually do
```

### Test 4: Live dry run
```bash
python -m live --dry-run --no-gui
# Should start without errors
# Exit engine logs should show exit_action types
```

---

## ESTIMATED SCOPE

| Item | Lines |
|------|-------|
| `core/exit_engine.py` (new) | ~450 |
| `training/orchestrator.py` (integration) | ~30 (replace ~80 old) |
| `live/live_engine.py` (integration) | ~25 (replace ~120 old) |
| Deletions from old exit logic | ~-200 |

Net change: ~+100 lines (new module replaces scattered logic)

---

## KEY DESIGN DECISIONS

1. **Cluster-fitted stops everywhere.** ATR fallback exists but should rarely fire.
   If a template doesn't have p25_mae/regression_sigma, that's a data problem
   to fix upstream, not a reason to use generic ATR.

2. **Bar high/low in live, not just close.** The old live engine only checked
   close price against stops. This missed any wick that touched the stop and
   reversed. Now live uses bar_high/bar_low, which is available from NinjaTrader
   on every 15s bar. Still less precise than training's 1s sub-bars, but
   dramatically better than close-only.

3. **Band context is optional.** The exit engine works without it (returns None
   checks everywhere). This means you can deploy the exit engine BEFORE
   implementing the band confluence module.

4. **Wave rider is supplementary.** The exit engine has its own trail logic
   (which includes band awareness). Wave rider's exit signal is checked AFTER
   the exit engine's own trail — it adds a second opinion, not the primary trail.

5. **No auto-TP re-entry in this module.** Re-entry is an ENTRY decision,
   not an exit decision. The exit engine closes the position. If the live engine
   wants to re-enter, it calls the entry logic again. This keeps concerns separated.
