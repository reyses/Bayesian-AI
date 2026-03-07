# ═══════════════════════════════════════════════════════════════════════════════
# BAYESIANBRIDGE MASTER SPEC — MONOLITHIC CLAUDE CODE INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Date: March 6, 2026
# Status: Phase 4 (terminology refactor) COMPLETE. Phases 1, 2, 3 remain.
#
# CRITICAL CONTEXT:
# The codebase has ALREADY been refactored from quantum physics metaphors to
# standard statistical terminology. ALL old names below have been replaced:
#
#   ThreeBodyQuantumState → MarketState
#   QuantumFieldEngine    → StatisticalFieldEngine
#   QuantumRiskEngine     → MonteCarloRiskEngine
#   particle_position     → price
#   particle_velocity     → velocity
#   center_position       → regression_center
#   F_net                 → net_force
#   F_reversion           → mean_reversion_force
#   sigma_fractal         → regression_sigma
#   lagrange_zone         → band_zone
#   L1_STABLE             → INNER
#   L2_ROCHE              → UPPER_EXTREME
#   L3_ROCHE              → LOWER_EXTREME
#   ROCHE_SNAP            → BAND_REVERSAL
#   STRUCTURAL_DRIVE      → MOMENTUM_BREAK
#   tunnel_probability    → reversion_probability
#   escape_probability    → breakout_probability
#   event_horizon_upper   → upper_band_3sigma
#   event_horizon_lower   → lower_band_3sigma
#   upper_singularity     → upper_band_2sigma
#   lower_singularity     → lower_band_2sigma
#   spin_inverted         → reversal_confirmed
#   coherence             → entropy_normalized
#   resonance_coherence   → alignment_score
#   fractal_alignment_count → multi_tf_alignment_count
#   time_at_roche         → time_at_band_extreme
#
# USE THE NEW NAMES IN ALL CODE. If you see old names in the specs below,
# translate them to the new names listed above.
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION ORDER
# ─────────────────────────────────────────────────────────────────────────────
#
# PHASE 1: Unified Exit Engine        ← foundation (no dependencies)
#    ↓ GATE 1: forward pass runs, trade count within ±15%
# PHASE 2: Band Context               ← requires: exit engine (band_context param)
#    ↓ GATE 2: band_direction logged in oracle_trade_log.csv
# PHASE 3: Oracle Direction Learning   ← requires: exit engine + band context
#    ↓ GATE 3: pattern_library.pkl shows corrected biases
#
# Do NOT start Phase N+1 until Gate N passes.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# BACKUP PROTOCOL (run ONCE before starting Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
#
# mkdir -p backups/pre_refactor
# cp -r core/ backups/pre_refactor/core/
# cp -r training/ backups/pre_refactor/training/
# cp -r live/ backups/pre_refactor/live/
# cp -r checkpoints/ backups/pre_refactor/checkpoints/
#
# Before each phase:
# cp checkpoints/pattern_library.pkl checkpoints/pattern_library_pre_phaseN.pkl
# cp checkpoints/live_brain.pkl checkpoints/live_brain_pre_phaseN.pkl 2>/dev/null || true
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: UNIFIED EXIT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
# Problem: Training and live use completely different exit logic.
#   - Different stop sizing (cluster-fitted vs ATR)
#   - Different TP cascades
#   - Different trail activation thresholds
#   - Band-aware exits only in live
#   - Envelope decay only in live
#   - 1s wick checking only in training
#   - Auto-TP re-entry only in live
# Training PnL is meaningless because exits don't match execution.
#
# Solution: One module, two modes. Same code path, always.
# New file: core/exit_engine.py (~450 lines)
# Modifies: training/orchestrator.py, live/live_engine.py
# Deletes: scattered exit logic in both files (~200 lines removed)
# ═══════════════════════════════════════════════════════════════════════════════

## FILE: core/exit_engine.py (CREATE NEW)

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
```

## INTEGRATION: training/orchestrator.py

FIND the existing exit handling in `run_forward_pass()` where
`wave_rider.update_trail()`, `check_stops_hilo()`, and `get_exit_signal()`
are called.

REPLACE with:

```python
# ── Initialize unified exit engine (once, before day loop) ────
from core.exit_engine import ExitEngine, ExitAction
_exit_engine = ExitEngine(
    mode='training',
    wave_rider=self.wave_rider,
    tick_size=self.tick_size,
    tick_value=self.tick_value,
)

# ... when opening a position:
_pos_state = _exit_engine.open_position(
    side=side,
    entry_price=entry_price,
    entry_bar_index=bar_idx,
    template_id=best_tid,
    lib_entry=lib_entry,
)

# ... each bar while position is held:
_band_ctx = None
if hasattr(self, 'tbn') and hasattr(self.tbn, 'get_band_confluence'):
    _band_ctx = self.tbn.get_band_confluence()

_net_force = getattr(current_state, 'net_force', 0.0)

_sub_highs = None
_sub_lows = None
if hasattr(bar_data, 'sub_bar_highs'):
    _sub_highs = bar_data.sub_bar_highs
    _sub_lows = bar_data.sub_bar_lows

_exit_result = _exit_engine.evaluate(
    pos=_pos_state,
    bar_high=bar_data.high,
    bar_low=bar_data.low,
    bar_close=bar_data.close,
    current_bar_index=bar_idx,
    band_context=_band_ctx,
    net_force=_net_force,
    worker_beliefs=None,
    sub_bar_highs=_sub_highs,
    sub_bar_lows=_sub_lows,
)

if _exit_result.action != ExitAction.HOLD:
    exit_price = _exit_result.exit_price
    exit_reason = _exit_result.reason
    pnl_ticks = _exit_result.pnl_ticks
    # ... continue with existing trade recording logic
```

## INTEGRATION: live/live_engine.py

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
    bar_low=bar.low,
    bar_close=bar.close,
    current_bar_index=self._bar_count,
    band_context=_band_ctx,
    net_force=getattr(self._current_state, 'net_force', 0.0),
    worker_beliefs=self._tbn.get_worker_beliefs() if self._tbn else None,
    sub_bar_highs=None,
    sub_bar_lows=None,
)

if _exit_result.action != ExitAction.HOLD:
    self._submit_exit_order(
        exit_price=_exit_result.exit_price,
        reason=_exit_result.reason,
    )
```

## DELETE after integration:

From training/orchestrator.py:
- `check_stops_hilo()`
- Inline trail update logic
- Inline TP/SL checks
- ATR-based stop sizing

From live/live_engine.py:
- `_manage_trail()`
- `_check_envelope_decay()`
- `_check_band_exit()`
- `_check_watchdog()`
- ATR-based stop/trail/TP sizing

Keep from training/wave_rider.py:
- `get_exit_signal()` — exit engine calls it as supplementary input
- `update_trail()` — wave_rider trail becomes secondary to exit engine

## ─── GATE 1 VALIDATION ──────────────────────────────────────

```bash
# 1A: Forward pass completes without errors
python training/orchestrator.py --forward-pass --data DATA/ATLAS

# 1B: Trade count within ±15% of previous run

# 1C: Exit action breakdown — no single category >80%

# 1D: Live dry run starts
python -m live --dry-run --no-gui

# 1E: PnL within ±20% of baseline
```

PASS CRITERIA:
- [ ] Forward pass completes without exception
- [ ] Trade count within ±15%
- [ ] Exit action distribution balanced
- [ ] Live dry run starts
- [ ] PnL within ±20%

FAIL → ROLLBACK: Uncomment old exit logic, debug exit_engine.py


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: STANDARD ERROR BAND MULTI-TF CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
# Problem: Persistent SHORT bias. z_score sign in uptrend → always SHORT.
# Solution: Multi-TF Standard Error Band confluence for structural direction.
# ~100 lines across 3 files. No deletions. No structural changes.
# Depends on: Phase 1 (exit engine accepts band_context parameter)
# ═══════════════════════════════════════════════════════════════════════════════

## CHANGE 1: Add BandContext dataclass

### File: training/timeframe_belief_network.py — after WorkerBelief

```python
@dataclass
class BandContext:
    """Where price sits relative to Standard Error Bands at one TF."""
    tf_seconds: int
    z_score: float
    sigma: float                # regression_sigma at this TF
    center: float               # regression_center (fair value)
    band: int                   # -3 to +3 (discrete sigma level)
    band_position: float        # -1.0 to +1.0 (continuous)
    at_support: bool            # z <= -1.0
    at_resistance: bool         # z >= 1.0
    band_label: str             # '-2σ', '+1σ', 'center'
```

## CHANGE 2: Add band_context field to WorkerBelief

```python
@dataclass
class WorkerBelief:
    tf_seconds:    int
    dir_prob:      float
    pred_mfe:      float
    template_id:   int
    tf_bar_idx:    int
    conviction:    float
    wave_maturity: float = 0.0
    band_context:  Optional[BandContext] = None   # ← ADD THIS
```

## CHANGE 3: Populate band_context in TimeframeWorker._analyze()

After the existing physics blend section:

```python
# ── Band Context (Standard Error Bands) ──────────────────────
_z = float(getattr(state, 'z_score', 0.0))
_sigma = float(getattr(state, 'regression_sigma', 0.0))
_center = float(getattr(state, 'regression_center', 0.0))

_band_int = int(np.clip(np.round(_z), -3, 3))
_band_pos = float(np.clip(_z / 3.0, -1.0, 1.0))
_at_sup = _z <= -1.0
_at_res = _z >= 1.0

if abs(_z) < 0.5:
    _band_lbl = 'center'
else:
    _sign = '+' if _z > 0 else '-'
    _band_lbl = f'{_sign}{abs(_band_int)}σ'

_band_ctx = BandContext(
    tf_seconds=self.tf_seconds,
    z_score=_z,
    sigma=_sigma,
    center=_center,
    band=_band_int,
    band_position=_band_pos,
    at_support=_at_sup,
    at_resistance=_at_res,
    band_label=_band_lbl,
)
```

Include `band_context=_band_ctx` in the WorkerBelief constructor.

## CHANGE 4: Add get_band_confluence() to TimeframeBeliefNetwork

```python
def get_band_confluence(self) -> Optional[dict]:
    """
    Multi-TF Standard Error Band confluence.
    
    If majority of TFs show price at support bands (z <= -1σ) → LONG
    If majority show resistance bands (z >= +1σ) → SHORT
    If mixed → None (no signal)
    
    Higher TFs carry more weight.
    
    Returns:
        {direction, strength, support_score, resistance_score,
         active_bands, band_summary, per_tf}
        None if < 3 active workers have band data.
    """
    active_bands = {}
    for tf, worker in self.workers.items():
        b = worker.current_belief
        if b is not None and b.band_context is not None:
            active_bands[tf] = b.band_context
    
    if len(active_bands) < 3:
        return None
    
    support_score = 0.0
    resistance_score = 0.0
    total_weight = 0.0
    per_tf = {}
    summary_parts = []
    
    for tf, ctx in active_bands.items():
        w = self._weight_map.get(tf, 1.0)
        total_weight += w
        tf_label = self._TF_LABELS.get(tf, str(tf))
        per_tf[tf_label] = ctx
        
        if ctx.at_support:
            support_score += w * abs(ctx.z_score)
            summary_parts.append(f"{tf_label}:{ctx.band_label}")
        elif ctx.at_resistance:
            resistance_score += w * ctx.z_score
            summary_parts.append(f"{tf_label}:{ctx.band_label}")
        else:
            summary_parts.append(f"{tf_label}:center")
    
    if total_weight > 0:
        support_score /= total_weight
        resistance_score /= total_weight
    
    if support_score > resistance_score * 2 and support_score > 0.5:
        direction = 'long'
        strength = min(1.0, support_score / 3.0)
    elif resistance_score > support_score * 2 and resistance_score > 0.5:
        direction = 'short'
        strength = min(1.0, resistance_score / 3.0)
    else:
        direction = None
        strength = 0.0
    
    arrow = '→ LONG' if direction == 'long' else ('→ SHORT' if direction == 'short' else '→ MIXED')
    summary = ' | '.join(summary_parts) + f' {arrow}'
    
    return {
        'direction': direction,
        'strength': strength,
        'support_score': support_score,
        'resistance_score': resistance_score,
        'active_bands': len(active_bands),
        'band_summary': summary,
        'per_tf': per_tf,
    }
```

## CHANGE 5: Add band_confluence to BeliefState

```python
@dataclass
class BeliefState:
    direction:              str
    conviction:             float
    predicted_mfe:          float
    active_levels:          int
    wave_maturity:          float = 0.0
    decision_wave_maturity: float = 0.0
    tf_beliefs:     Dict[int, WorkerBelief] = field(default_factory=dict)
    band_confluence: Optional[dict] = None   # ← ADD
```

At end of `get_belief()`: `band_confluence = self.get_band_confluence()`

## CHANGE 6: Band confluence in direction cascades

### In live/live_engine.py — _determine_direction()

Insert Priority 3.5 (after template bias, before DMI):

```python
# Priority 3.5: Multi-TF band confluence
if side is None:
    band_signal = self._belief_network.get_band_confluence()
    if band_signal is not None and band_signal['direction'] is not None:
        side = band_signal['direction']
        _p_long = 0.5 + (0.3 if side == 'long' else -0.3) * band_signal['strength']
        return side, _p_long, 'band_confluence'
```

Replace final velocity fallback:

```python
# FINAL FALLBACK: band confluence > velocity sign
if side is None:
    band_signal = self._belief_network.get_band_confluence()
    if band_signal is not None and band_signal['direction'] is not None:
        side = band_signal['direction']
        return side, 0.55 if side == 'long' else 0.45, 'band_fallback'

vel = getattr(s, 'velocity', 0.0)
side = 'long' if vel >= 0 else 'short'
return side, 0.55 if side == 'long' else 0.45, 'velocity'
```

### In training/orchestrator.py — forward pass direction cascade

Add Priority 2.5 (before DMI):

```python
# Priority 2.5: Multi-TF band confluence
if side is None:
    _band = belief_network.get_band_confluence()
    if _band is not None and _band['direction'] is not None:
        side = _band['direction']
```

Replace velocity fallback same as live.

## CHANGE 7: Add band fields to oracle_trade_log.csv

In the `pending_oracle` dict:

```python
'band_direction': _band['direction'] if _band else None,
'band_strength': round(_band['strength'], 3) if _band else 0.0,
'band_summary': _band['band_summary'] if _band else '',
```

## ─── GATE 2 VALIDATION ──────────────────────────────────────

```bash
# 2A: Forward pass completes
python training/orchestrator.py --forward-pass --data DATA/ATLAS

# 2B: band_direction column populated
python -c "
import pandas as pd
df = pd.read_csv('oracle_trade_log.csv')
print(df['band_direction'].value_counts(dropna=False))
print(f'Non-null: {df[\"band_direction\"].notna().sum()}/{len(df)}')
"

# 2C: band_direction accuracy vs oracle
python -c "
import pandas as pd
df = pd.read_csv('oracle_trade_log.csv')
df = df[df['band_direction'].notna() & df['oracle_label'].notna()]
df['band_correct'] = (
    ((df['band_direction'] == 'long') & (df['oracle_label'] > 0)) |
    ((df['band_direction'] == 'short') & (df['oracle_label'] < 0))
)
print(f'Band accuracy: {df[\"band_correct\"].mean():.1%}')
"

# 2D: LONG/SHORT ratio closer to 1.0

# 2E: Live dry run
python -m live --dry-run --no-gui
```

PASS CRITERIA:
- [ ] band_direction populated (>50% non-null)
- [ ] band accuracy vs oracle >50%
- [ ] LONG/SHORT ratio closer to 1.0
- [ ] Live dry run starts


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ORACLE DIRECTION LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
# Problem: Forward pass discards oracle corrections. Live starts cold.
# Solution: IS forward pass absorbs oracle labels, writes corrected biases
#           to pattern_library.pkl, trains signed MFE regression per template.
# ~180 lines across 2 files. No new files. No structural changes.
# Depends on: Phase 1 (exit engine stable), Phase 2 (band context in trade log)
# ═══════════════════════════════════════════════════════════════════════════════

## CHANGE 1: Direction Correction Accumulator

### File: training/orchestrator.py — run_forward_pass()

WHERE: After the forward pass day loop ends, before the "Final Report" section.

```python
# ═══════════════════════════════════════════════════════════════════
# ORACLE DIRECTION LEARNING (supervised correction)
# ═══════════════════════════════════════════════════════════════════

if oracle_trade_records and not oos_mode:
    print("\n  Learning direction corrections from oracle...")
    
    _dir_corrections = defaultdict(lambda: {
        'long_correct': 0, 'long_wrong': 0,
        'short_correct': 0, 'short_wrong': 0,
        'long_pnl': 0.0, 'short_pnl': 0.0,
        'signed_mfe_samples': [],
    })
    
    for rec in oracle_trade_records:
        tid = rec.get('template_id')
        if tid is None or tid == -1:
            continue
        
        direction = rec.get('direction', '')
        oracle_label = rec.get('oracle_label', 0)
        actual_pnl = rec.get('actual_pnl', 0.0)
        oracle_mfe = rec.get('oracle_mfe', 0.0)
        oracle_mae = rec.get('oracle_mae', 0.0)
        
        acc = _dir_corrections[tid]
        oracle_says_long = oracle_label > 0
        oracle_says_short = oracle_label < 0
        
        if direction == 'LONG':
            acc['long_pnl'] += actual_pnl
            if oracle_says_long:
                acc['long_correct'] += 1
            elif oracle_says_short:
                acc['long_wrong'] += 1
        
        if direction == 'SHORT':
            acc['short_pnl'] += actual_pnl
            if oracle_says_short:
                acc['short_correct'] += 1
            elif oracle_says_long:
                acc['short_wrong'] += 1
        
        if oracle_label != 0:
            signed_mfe = oracle_mfe if oracle_label > 0 else -oracle_mae
            acc['signed_mfe_samples'].append({
                'signed_mfe': signed_mfe,
                'entry_depth': rec.get('entry_depth', 6),
                'dmi_diff': rec.get('dmi_diff', 0.0),
                'oracle_label': oracle_label,
            })
    
    _updated_count = 0
    _regression_count = 0
    
    for tid, acc in _dir_corrections.items():
        if tid not in self.pattern_library:
            continue
        lib = self.pattern_library[tid]
        
        # ── Corrected direction bias (70% forward pass, 30% original) ──
        long_total = acc['long_correct'] + acc['long_wrong']
        short_total = acc['short_correct'] + acc['short_wrong']
        total_dir_trades = long_total + short_total
        
        if total_dir_trades >= 3:
            fp_long_correct = acc['long_correct']
            fp_short_correct = acc['short_correct']
            fp_total_correct = fp_long_correct + fp_short_correct
            
            fp_long_bias = fp_long_correct / fp_total_correct if fp_total_correct > 0 else 0.5
            fp_short_bias = fp_short_correct / fp_total_correct if fp_total_correct > 0 else 0.5
            
            orig_long = lib.get('long_bias', 0.5)
            orig_short = lib.get('short_bias', 0.5)
            
            new_long = 0.7 * fp_long_bias + 0.3 * orig_long
            new_short = 0.7 * fp_short_bias + 0.3 * orig_short
            total = new_long + new_short
            if total > 0:
                new_long /= total
                new_short /= total
            
            lib['long_bias'] = round(new_long, 4)
            lib['short_bias'] = round(new_short, 4)
            lib['direction_source'] = 'oracle_corrected'
            _updated_count += 1
        
        # ── PnL-weighted direction signal ──
        if long_total >= 2 and short_total >= 2:
            lib['long_avg_pnl'] = round(acc['long_pnl'] / long_total, 2)
            lib['short_avg_pnl'] = round(acc['short_pnl'] / short_total, 2)
        
        # ── Signed MFE regression ──
        samples = acc['signed_mfe_samples']
        if len(samples) >= 15:
            try:
                from sklearn.linear_model import LinearRegression
                X = np.array([[s['entry_depth'], s['dmi_diff']] for s in samples])
                y = np.array([s['signed_mfe'] for s in samples])
                reg = LinearRegression().fit(X, y)
                lib['signed_mfe_coeff'] = reg.coef_.tolist()
                lib['signed_mfe_intercept'] = float(reg.intercept_)
                _regression_count += 1
            except Exception:
                pass
    
    print(f"  Direction corrections: {_updated_count} templates updated")
    print(f"  Signed MFE regression: {_regression_count} templates fitted")
    
    import pickle as _pkl_dir
    _lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
    with open(_lib_path, 'wb') as _f:
        _pkl_dir.dump(self.pattern_library, _f)
    print(f"  Updated pattern_library.pkl saved")
```

## CHANGE 2: Signed MFE in Direction Cascade

### File: training/orchestrator.py — direction decision block

Add Priority 0.5 (between oracle marker and logistic regression):

```python
# Priority 0.5: Signed MFE regression (learned from IS forward pass)
if side is None:
    _smfe_coeff = lib_entry.get('signed_mfe_coeff')
    if _smfe_coeff is not None:
        _entry_depth = getattr(best_candidate, 'depth', 6)
        _live_dmi = (getattr(best_candidate.state, 'dmi_plus', 0.0)
                   - getattr(best_candidate.state, 'dmi_minus', 0.0))
        _smfe_features = np.array([[_entry_depth, _live_dmi]])
        _pred_smfe = float(
            np.dot(_smfe_features, np.array(_smfe_coeff))
            + lib_entry.get('signed_mfe_intercept', 0.0)
        )
        if abs(_pred_smfe) > 0.5:
            side = 'long' if _pred_smfe > 0 else 'short'
```

### File: live/live_engine.py

VERIFY the existing `signed_mfe_coeff` check works. It was added but never
populated because forward pass never wrote coefficients. After Change 1,
it will automatically work.

## CHANGE 3: Brain Direction-Specific Win Rates

### File: training/orchestrator.py — direction cascade

Add Priority 1.5:

```python
# Priority 1.5: Brain direction-specific win rate
if side is None:
    _dir_long_prob = self.brain.get_dir_probability(best_tid, 'LONG')
    _dir_short_prob = self.brain.get_dir_probability(best_tid, 'SHORT')
    if _dir_long_prob is not None and _dir_short_prob is not None:
        if _dir_long_prob > _dir_short_prob + 0.10:
            side = 'long'
        elif _dir_short_prob > _dir_long_prob + 0.10:
            side = 'short'
```

### File: live/live_engine.py — _determine_direction()

Add Priority 0.5 (after live bias, before signed MFE):

```python
# Priority 0.5: Brain direction-specific win rate
_dir_long = self._brain.get_dir_probability(base_tid, 'LONG')
_dir_short = self._brain.get_dir_probability(base_tid, 'SHORT')
if _dir_long is not None and _dir_short is not None:
    if _dir_long > _dir_short + 0.10:
        return 'long', _dir_long, 'brain_dir'
    elif _dir_short > _dir_long + 0.10:
        return 'short', 1.0 - _dir_short, 'brain_dir'
```

## CHANGE 4: Save Brain for Live

### File: training/orchestrator.py — after pattern library save

```python
if not oos_mode:
    _brain_path = os.path.join(self.checkpoint_dir, 'pattern_forward_brain.pkl')
    self.brain.save(_brain_path)
    print(f"  Forward pass brain saved: {_brain_path}")
    print(f"    States: {len(self.brain.table)}")
    print(f"    Direction pairs: {len(self.brain.dir_table)}")
```

### File: live/live_engine.py — _load_checkpoints()

Prefer forward pass brain:

```python
live_brain_path = os.path.join(cpdir, 'live_brain.pkl')
forward_brain_path = os.path.join(cpdir, 'pattern_forward_brain.pkl')
training_brains = sorted(glob.glob(os.path.join(cpdir, 'pattern_*_brain.pkl')))

if os.path.exists(live_brain_path):
    self._brain.load(live_brain_path)
    logger.info(f"  Brain: live_brain.pkl ({len(self._brain.table)} states, "
                f"{len(self._brain.dir_table)} dir pairs)")
elif os.path.exists(forward_brain_path):
    self._brain.load(forward_brain_path)
    logger.info(f"  Brain: pattern_forward_brain.pkl ({len(self._brain.table)} states, "
                f"{len(self._brain.dir_table)} dir pairs) — IS-learned directions")
elif training_brains:
    self._brain.load(training_brains[-1])
    logger.info(f"  Brain: {os.path.basename(training_brains[-1])} (training base)")
else:
    logger.warning("  No brain checkpoint found — starting fresh")
```

## CHANGE 5: Direction Learning Report

### File: training/orchestrator.py — report section

After existing profit gap analysis, add:

```python
if _dir_corrections:
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DIRECTION LEARNING (oracle corrections absorbed)")
    report_lines.append("=" * 80)
    
    _total_corrected = sum(
        1 for acc in _dir_corrections.values()
        if (acc['long_correct'] + acc['long_wrong'] +
            acc['short_correct'] + acc['short_wrong']) >= 3
    )
    _total_smfe = sum(
        1 for acc in _dir_corrections.values()
        if len(acc['signed_mfe_samples']) >= 15
    )
    
    report_lines.append(f"  Templates with direction corrections: {_total_corrected}")
    report_lines.append(f"  Templates with signed MFE regression: {_total_smfe}")
    
    _corrections_list = []
    for tid, acc in _dir_corrections.items():
        if tid not in self.pattern_library:
            continue
        lib = self.pattern_library[tid]
        orig_long = lib.get('long_bias', 0.5)
        long_total = acc['long_correct'] + acc['long_wrong']
        short_total = acc['short_correct'] + acc['short_wrong']
        if long_total + short_total < 3:
            continue
        _corrections_list.append({
            'tid': tid,
            'orig_long_bias': orig_long,
            'new_long_bias': lib.get('long_bias', 0.5),
            'long_correct': acc['long_correct'],
            'long_wrong': acc['long_wrong'],
            'short_correct': acc['short_correct'],
            'short_wrong': acc['short_wrong'],
            'long_pnl': acc['long_pnl'],
            'short_pnl': acc['short_pnl'],
            'shift': abs(lib.get('long_bias', 0.5) - orig_long),
        })
    
    _corrections_list.sort(key=lambda x: -x['shift'])
    
    if _corrections_list:
        report_lines.append("")
        report_lines.append(f"  TOP 15 DIRECTION CORRECTIONS (biggest bias shift):")
        report_lines.append(f"  {'TID':>8} {'Orig':>6} {'New':>6} {'Shift':>6} "
                           f"{'L_ok':>5} {'L_bad':>6} {'S_ok':>5} {'S_bad':>6} "
                           f"{'L_PnL':>10} {'S_PnL':>10}")
        for r in _corrections_list[:15]:
            report_lines.append(
                f"  {r['tid']:>8} {r['orig_long_bias']:>6.2f} "
                f"{r['new_long_bias']:>6.2f} {r['shift']:>+5.2f} "
                f"{r['long_correct']:>5} {r['long_wrong']:>6} "
                f"{r['short_correct']:>5} {r['short_wrong']:>6} "
                f"${r['long_pnl']:>9,.0f} ${r['short_pnl']:>9,.0f}")
    
    _all_long_ok = sum(a['long_correct'] for a in _dir_corrections.values())
    _all_long_bad = sum(a['long_wrong'] for a in _dir_corrections.values())
    _all_short_ok = sum(a['short_correct'] for a in _dir_corrections.values())
    _all_short_bad = sum(a['short_wrong'] for a in _dir_corrections.values())
    _all_total = _all_long_ok + _all_long_bad + _all_short_ok + _all_short_bad
    _all_correct = _all_long_ok + _all_short_ok
    
    if _all_total > 0:
        report_lines.append("")
        report_lines.append(f"  DIRECTION ACCURACY (this run):")
        report_lines.append(f"    Correct: {_all_correct}/{_all_total} "
                           f"({_all_correct/_all_total*100:.1f}%)")
        report_lines.append(f"    LONG  correct: {_all_long_ok}  wrong: {_all_long_bad}")
        report_lines.append(f"    SHORT correct: {_all_short_ok}  wrong: {_all_short_bad}")
```

## ─── GATE 3 VALIDATION ──────────────────────────────────────

```bash
# 3A: Forward pass completes (Run 1 — builds corrections)
python training/orchestrator.py --forward-pass --data DATA/ATLAS

# 3B: Direction corrections section in report (>20 templates corrected)

# 3C: Pattern library has corrected biases
python -c "
import pickle
with open('checkpoints/pattern_library.pkl', 'rb') as f:
    lib = pickle.load(f)
corrected = sum(1 for v in lib.values() if v.get('direction_source') == 'oracle_corrected')
smfe = sum(1 for v in lib.values() if v.get('signed_mfe_coeff') is not None)
print(f'Total: {len(lib)}, Corrected: {corrected}, SMFE: {smfe}')
"

# 3D: Run 2 — uses corrected biases (PnL should improve)
python training/orchestrator.py --forward-pass --data DATA/ATLAS

# 3E: Velocity fallback <15% of direction decisions

# 3F: Live loads forward brain
python -m live --dry-run --no-gui
```

PASS CRITERIA:
- [ ] >20 templates corrected
- [ ] pattern_library.pkl has oracle_corrected entries
- [ ] pattern_forward_brain.pkl exists
- [ ] Run 2 PnL >= Run 1
- [ ] Velocity fallback <15%
- [ ] Live loads forward brain

BEFORE RUNNING PHASE 3, BACK UP:
cp checkpoints/pattern_library.pkl checkpoints/pattern_library_pre_phase3.pkl


# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED OUTCOMES AFTER ALL 3 PHASES
# ═══════════════════════════════════════════════════════════════════════════════
#
# | Metric                        | Before        | After            |
# |-------------------------------|---------------|------------------|
# | Training/live exit parity     | ~40% match    | 100% same code   |
# | LONG/SHORT ratio              | SHORT-biased  | Balanced ~0.8-1.2|
# | Velocity fallback usage       | ~40% of trades| <5%              |
# | Direction accuracy vs oracle  | ~50% (coin)   | >60%             |
# | Templates with corrected bias | 0             | >50% of active   |
# | Forward pass PnL              | X             | Closer to live   |
#
# TIMELINE: ~12-15 hours over 3 days. Do NOT compress.
# Each gate needs a clean forward pass (30-60 min) and inspection.
# ═══════════════════════════════════════════════════════════════════════════════
