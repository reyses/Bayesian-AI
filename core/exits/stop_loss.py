"""Stop Loss  -- hard stop at absolute price level."""
from core.exit_engine import ExitAction, ExitResult, PositionState


class StopLossCheck:

    @staticmethod
    def evaluate(pos: PositionState, worst_price: float,
                 tick_size: float) -> ExitResult | None:
        sl_price = pos.stop_loss
        if pos.side == 'long':
            hit = worst_price <= sl_price
        else:
            hit = worst_price >= sl_price
        if not hit:
            return None
        pnl_ticks = ((sl_price - pos.entry_price) / tick_size
                      if pos.side == 'long'
                      else (pos.entry_price - sl_price) / tick_size)
        return ExitResult(
            action=ExitAction.STOP_LOSS,
            exit_price=sl_price,
            reason=f"SL hit at {sl_price:.2f} (worst={worst_price:.2f})",
            pnl_ticks=pnl_ticks,
            bars_held=pos.bars_held,
            trail_level=pos.stop_loss,
        )
