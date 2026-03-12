"""Band Urgent Exit — exit when multi-TF support/resistance is broken against trade."""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class BandUrgentExit:

    @staticmethod
    def evaluate(pos: PositionState, bar_close: float, tick_size: float,
                 band_context: dict = None) -> Optional[ExitResult]:
        if band_context is None:
            return None
        direction = band_context.get('direction')
        strength = band_context.get('strength', 0.0)
        if strength < 0.6:
            return None

        if pos.side == 'long' and direction == 'short' and strength > 0.7:
            unrealized_ticks = (bar_close - pos.entry_price) / tick_size
            if unrealized_ticks < -2:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: LONG but support broken (str={strength:.2f})",
                    pnl_ticks=unrealized_ticks,
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )

        if pos.side == 'short' and direction == 'long' and strength > 0.7:
            unrealized_ticks = (pos.entry_price - bar_close) / tick_size
            if unrealized_ticks < -2:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: SHORT but resistance broken (str={strength:.2f})",
                    pnl_ticks=unrealized_ticks,
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )

        return None
