"""Envelope Decay  -- time-based exit with dynamic halflife modulation."""
import math
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


_LN2 = 0.693  # ln(2)  -- structural constant for halflife decay


class EnvelopeDecay:

    def __init__(self, half_life_bars: float = 20, floor_pct: float = 0.3,
                 min_bars: int = 5, config=None):
        self.half_life_bars = half_life_bars
        self.floor_pct = floor_pct
        self.min_bars = min_bars
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._force_boost = config.envelope_force_boost
        self._force_penalty = config.envelope_force_penalty
        self._early_suppress_pct = config.envelope_early_suppress_pct
        self._floor_trigger_pct = config.envelope_floor_trigger_pct
        self._template_hl_divisor = config.envelope_template_hl_divisor
        self._template_hl_floor = config.envelope_template_hl_floor
        self._peak_min_ticks = config.envelope_peak_min_ticks
        self._giveback_hl_floor = config.envelope_giveback_hl_floor
        self._band_coeff = config.envelope_band_coeff
        self._band_mult_min = config.envelope_band_mult_min
        self._band_mult_max = config.envelope_band_mult_max
        self._anchor_patience_max = config.envelope_anchor_patience_max
        self._hl_mult_floor = config.envelope_hl_mult_floor
        self._adx_slope_boost = config.envelope_adx_slope_boost
        self._adx_slope_penalty = config.envelope_adx_slope_penalty

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 net_force: float = 0.0, band_context: dict = None,
                 noise_ticks: float = 0.0) -> Optional[ExitResult]:
        if not pos.envelope_active or pos.bars_held < self.min_bars:
            return None

        # Base halflife: per-template when available, else global default
        if pos.max_hold_bars > 0 and pos.max_hold_bars != 120:
            base_hl = max(self._template_hl_floor,
                          pos.max_hold_bars / self._template_hl_divisor)
        else:
            base_hl = self.half_life_bars

        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
            current_ticks = (bar_close - pos.entry_price) / tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / tick_size
            current_ticks = (pos.entry_price - bar_close) / tick_size

        # Noise gate: trade still within normal market breathing
        if noise_ticks > 0 and peak_ticks < noise_ticks:
            return None

        # Signal 1: giveback from peak
        hl_mult = 1.0
        if peak_ticks > self._peak_min_ticks:
            giveback_ratio = max(0, peak_ticks - current_ticks) / peak_ticks
            hl_mult *= max(self._giveback_hl_floor, 1.0 - giveback_ratio)

        # Signal 2: band exhaustion
        if band_context is not None:
            sup = band_context.get('support_score', 0.0)
            res = band_context.get('resistance_score', 0.0)
            if pos.side == 'long':
                exhaustion = res - sup
            else:
                exhaustion = sup - res
            band_mult = max(self._band_mult_min,
                            min(self._band_mult_max,
                                1.0 - exhaustion * self._band_coeff))
            hl_mult *= band_mult

        # Signal 3: anchor time patience
        if pos.anchor_mfe_bars > 0 and pos.bars_held < pos.anchor_mfe_bars:
            anchor_progress = pos.bars_held / pos.anchor_mfe_bars
            hl_mult *= (self._anchor_patience_max - anchor_progress)

        # Shape primitive halflife modulation (from exit primitive calibration)
        if pos.envelope_halflife_mult != 1.0:
            hl_mult *= pos.envelope_halflife_mult

        effective_hl = base_hl * max(self._hl_mult_floor, hl_mult)

        # ADX slope modulation: rising trend -> slow decay, falling -> speed up
        # exit_signal carries 'adx_slope' from TBN when available
        # (intentionally not gated behind config flag  -- always active when data present)
        if band_context is not None:
            _adx_slope = band_context.get('adx_slope', 0.0)
            if _adx_slope > 0:
                # Trend strengthening  -- slow down envelope decay (up to 50%)
                effective_hl *= 1.0 + min(0.5, _adx_slope * self._adx_slope_boost)
            elif _adx_slope < -1.0:
                # Trend weakening  -- speed up decay (up to 50% faster)
                effective_hl *= max(0.5, 1.0 + _adx_slope * self._adx_slope_penalty)

        # Decay factor
        decay = math.exp(-_LN2 * pos.bars_held / max(1, effective_hl))

        # Net force modulation
        if net_force != 0.0:
            force_aligned = (
                (pos.side == 'long' and net_force > 0) or
                (pos.side == 'short' and net_force < 0)
            )
            if force_aligned:
                decay = min(decay * self._force_boost, 1.0)
            else:
                decay = decay * self._force_penalty

        # Current envelope level (noise-aware floor)
        initial_tp = pos.tp_ticks * tick_size
        noise_floor = noise_ticks * tick_size if noise_ticks > 0 else 0
        floor = max(initial_tp * self.floor_pct, noise_floor)
        current_envelope = floor + (initial_tp - floor) * decay
        pos.envelope_level = current_envelope

        # Only trigger after significant time
        if pos.bars_held < effective_hl * self._early_suppress_pct:
            return None

        if pos.side == 'long':
            unrealized = bar_close - pos.entry_price
        else:
            unrealized = pos.entry_price - bar_close

        if 0 < unrealized < current_envelope * self._floor_trigger_pct:
            return ExitResult(
                action=ExitAction.ENVELOPE_DECAY,
                exit_price=bar_close,
                reason=f"Envelope decay: unrealized={unrealized:.2f} < envelope_floor",
                pnl_ticks=(bar_close - pos.entry_price) / tick_size
                          if pos.side == 'long'
                          else (pos.entry_price - bar_close) / tick_size,
                bars_held=pos.bars_held,
                envelope_level=current_envelope,
            )

        return None
