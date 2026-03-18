"""Belief Flip Exit  -- TBN urgency signal + DI crossover trend reversal.

Two triggers:
1. TBN urgent_exit flag (workers flipped conviction against trade)
2. DI crossover against position (direct trend reversal signal from DMI)
   - Long: DI+ was > DI-, now DI- >= DI+ -> bearish reversal
   - Short: DI- was > DI+, now DI+ >= DI- -> bullish reversal
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class BeliefFlipExit:

    def __init__(self, di_gap_threshold: float = 5.0, min_bars: int = 3):
        self.di_gap_threshold = di_gap_threshold  # DI gap minimum for crossover (87% accurate at ≥5)
        self.min_bars = min_bars                   # minimum bars held before DI crossover allowed

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 exit_signal: dict = None) -> Optional[ExitResult]:
        if exit_signal is None:
            return None

        pnl_ticks = ((bar_close - pos.entry_price) / tick_size
                     if pos.side == 'long'
                     else (pos.entry_price - bar_close) / tick_size)

        # Trigger 1: TBN urgent exit signal
        if exit_signal.get('urgent_exit', False):
            return ExitResult(
                action=ExitAction.BELIEF_FLIP,
                exit_price=bar_close,
                reason=f"Belief flip: {exit_signal.get('reason', 'urgent')}",
                pnl_ticks=pnl_ticks,
                bars_held=pos.bars_held,
                trail_level=pos.stop_loss,
            )

        # Trigger 2: DI crossover against position (trend reversal)
        # Uses 5m DMI (87% accurate at gap≥5, vs 1m at 63%)
        if pos.bars_held >= self.min_bars:
            di_plus = exit_signal.get('di_plus', 0.0)
            di_minus = exit_signal.get('di_minus', 0.0)
            di_plus_prev = exit_signal.get('di_plus_prev', di_plus)
            di_minus_prev = exit_signal.get('di_minus_prev', di_minus)
            di_gap = abs(di_plus - di_minus)

            if pos.side == 'long':
                crossed_against = (di_plus_prev > di_minus_prev
                                   and di_minus >= di_plus)
            else:
                crossed_against = (di_minus_prev > di_plus_prev
                                   and di_plus >= di_minus)

            if crossed_against and di_gap >= self.di_gap_threshold:
                return ExitResult(
                    action=ExitAction.BELIEF_FLIP,
                    exit_price=bar_close,
                    reason=f"DI crossover against {pos.side}: "
                           f"DI+={di_plus:.1f} DI-={di_minus:.1f}",
                    pnl_ticks=pnl_ticks,
                    bars_held=pos.bars_held,
                    trail_level=pos.stop_loss,
                )

        return None
