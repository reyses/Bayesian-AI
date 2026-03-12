"""Peak Giveback — exit when trade retraces too far from MFE peak.

Currently uses static tiered thresholds + self-tuning.
Future: physics-based dynamic thresholds from a separate peak module.
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class PeakGiveback:

    def __init__(self, min_mfe_ticks: float = 16, giveback_pct: float = 0.70):
        self.min_mfe_ticks = min_mfe_ticks
        self.giveback_pct = giveback_pct

    def get_threshold(self, peak_ticks: float, noise_ticks: float = 0.0) -> float:
        """Tiered giveback threshold.

        Peak MFE (ticks)  ->  Giveback trigger
        2x noise+         ->  40% (aggressive protection)
        1x noise - 2x     ->  self.giveback_pct (~55-70%)
        < noise            ->  disabled (move within noise floor)
        """
        min_mfe = (max(self.min_mfe_ticks, noise_ticks)
                   if noise_ticks > 0 else self.min_mfe_ticks)
        if peak_ticks >= min_mfe * 2:
            return 0.40
        elif peak_ticks >= min_mfe:
            return self.giveback_pct
        else:
            return 1.01  # >100% = never triggers

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 exit_signal: dict = None,
                 noise_ticks: float = 0.0) -> Optional[ExitResult]:
        # 30m slow-flip detection (sticky once set)
        if exit_signal and exit_signal.get('slow_flip_tighten'):
            pos.slow_flip_active = True

        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
            current_ticks = (bar_close - pos.entry_price) / tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / tick_size
            current_ticks = (pos.entry_price - bar_close) / tick_size

        if peak_ticks <= 0:
            return None

        # Anchor patience: trade still developing
        if (pos.anchor_mfe_ticks > 0 and pos.anchor_mfe_bars > 0
                and pos.bars_held < pos.anchor_mfe_bars
                and peak_ticks < pos.anchor_mfe_ticks * 0.3):
            return None

        threshold = self.get_threshold(peak_ticks, noise_ticks)

        # 30m flip tightens threshold by 15pp
        if pos.slow_flip_active and threshold < 1.0:
            threshold = max(0.25, threshold - 0.15)

        gave_back = peak_ticks - current_ticks
        if gave_back / peak_ticks >= threshold:
            pnl_ticks = (current_ticks if pos.side == 'long'
                         else (pos.entry_price - bar_close) / tick_size)
            return ExitResult(
                action=ExitAction.PEAK_GIVEBACK,
                exit_price=bar_close,
                reason=f"Peak giveback: peak={peak_ticks:.1f}t now={current_ticks:.1f}t "
                       f"gave_back={gave_back/peak_ticks:.0%} (tier={threshold:.0%})",
                pnl_ticks=(bar_close - pos.entry_price) / tick_size
                          if pos.side == 'long'
                          else (pos.entry_price - bar_close) / tick_size,
                bars_held=pos.bars_held,
            )

        return None
