"""Take Profit  -- hard target at absolute price level."""
from core.exit_engine import ExitAction, ExitResult, PositionState


class TakeProfitCheck:

    @staticmethod
    def evaluate(pos: PositionState, best_price: float,
                 tick_size: float) -> ExitResult | None:
        if pos.tp_ticks <= 0:
            return None
        if pos.side == 'long':
            tp_price = pos.entry_price + (pos.tp_ticks * tick_size)
            hit = best_price >= tp_price
        else:
            tp_price = pos.entry_price - (pos.tp_ticks * tick_size)
            hit = best_price <= tp_price
        if not hit:
            return None
        pnl_ticks = ((tp_price - pos.entry_price) / tick_size
                      if pos.side == 'long'
                      else (pos.entry_price - tp_price) / tick_size)
        return ExitResult(
            action=ExitAction.TAKE_PROFIT,
            exit_price=tp_price,
            reason=f"TP hit at {tp_price:.2f} (best={best_price:.2f})",
            pnl_ticks=pnl_ticks,
            bars_held=pos.bars_held,
            trail_level=pos.stop_loss,
        )
