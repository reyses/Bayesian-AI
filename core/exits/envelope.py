"""Envelope Decay — time-based exit with dynamic halflife modulation."""
import math
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class EnvelopeDecay:

    def __init__(self, half_life_bars: float = 20, floor_pct: float = 0.3,
                 min_bars: int = 5):
        self.half_life_bars = half_life_bars
        self.floor_pct = floor_pct
        self.min_bars = min_bars

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 net_force: float = 0.0, band_context: dict = None,
                 noise_ticks: float = 0.0) -> Optional[ExitResult]:
        if not pos.envelope_active or pos.bars_held < self.min_bars:
            return None

        # Base halflife: per-template when available, else global default
        if pos.max_hold_bars > 0 and pos.max_hold_bars != 120:
            base_hl = max(8.0, pos.max_hold_bars / 5.0)
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
        if peak_ticks > 4:
            giveback_ratio = max(0, peak_ticks - current_ticks) / peak_ticks
            hl_mult *= max(0.5, 1.0 - giveback_ratio)

        # Signal 2: band exhaustion
        if band_context is not None:
            sup = band_context.get('support_score', 0.0)
            res = band_context.get('resistance_score', 0.0)
            if pos.side == 'long':
                exhaustion = res - sup
            else:
                exhaustion = sup - res
            band_mult = max(0.5, min(1.5, 1.0 - exhaustion * 0.5))
            hl_mult *= band_mult

        # Signal 3: anchor time patience
        if pos.anchor_mfe_bars > 0 and pos.bars_held < pos.anchor_mfe_bars:
            anchor_progress = pos.bars_held / pos.anchor_mfe_bars
            hl_mult *= (2.0 - anchor_progress)

        effective_hl = base_hl * max(0.3, hl_mult)

        # Decay factor
        decay = math.exp(-0.693 * pos.bars_held / max(1, effective_hl))

        # Net force modulation
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
        initial_tp = pos.tp_ticks * tick_size
        noise_floor = noise_ticks * tick_size if noise_ticks > 0 else 0
        floor = max(initial_tp * self.floor_pct, noise_floor)
        current_envelope = floor + (initial_tp - floor) * decay
        pos.envelope_level = current_envelope

        # Only trigger after significant time
        if pos.bars_held < effective_hl * 0.5:
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
                pnl_ticks=(bar_close - pos.entry_price) / tick_size
                          if pos.side == 'long'
                          else (pos.entry_price - bar_close) / tick_size,
                bars_held=pos.bars_held,
                envelope_level=current_envelope,
            )

        return None
