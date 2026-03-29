"""Band Urgent Exit  -- exit when multi-TF support/resistance is broken against trade."""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class BandUrgentExit:

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._min_strength = config.band_urgent_min_strength
        self._trigger_strength = config.band_urgent_trigger_strength
        self._loss_ticks = config.band_urgent_loss_ticks

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 band_context: dict = None) -> Optional[ExitResult]:
        if band_context is None:
            return None
        direction = band_context.get('direction')
        strength = band_context.get('strength', 0.0)
        if strength < self._min_strength:
            return None

        # Thesis invalidation: band direction flipped against position
        # loss_ticks=0 means fire regardless of PnL (pure thesis invalidation)
        # loss_ticks>0 means only fire if underwater by that amount
        if pos.side == 'long' and direction == 'short' and strength > self._trigger_strength:
            unrealized_ticks = (bar_close - pos.entry_price) / tick_size
            if self._loss_ticks <= 0 or unrealized_ticks < -self._loss_ticks:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: LONG but support broken (str={strength:.2f}, pnl={unrealized_ticks:.0f}t)",
                    pnl_ticks=unrealized_ticks,
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )

        if pos.side == 'short' and direction == 'long' and strength > self._trigger_strength:
            unrealized_ticks = (pos.entry_price - bar_close) / tick_size
            if self._loss_ticks <= 0 or unrealized_ticks < -self._loss_ticks:
                return ExitResult(
                    action=ExitAction.BAND_URGENT,
                    exit_price=bar_close,
                    reason=f"Band urgent: SHORT but resistance broken (str={strength:.2f}, pnl={unrealized_ticks:.0f}t)",
                    pnl_ticks=unrealized_ticks,
                    bars_held=pos.bars_held,
                    band_zone=band_context.get('band_summary', ''),
                    band_action='urgent',
                )

        return None
