"""Tick-exact TP / SL exit rules.

The default `TakeProfit` and `HardStop` evaluate `position.pnl(state.price)`
where `state.price` is the 5s BAR CLOSE. If price moved through the TP/SL
threshold INTRABAR, the exit fires but closes at the bar-close price — which
may be much further from entry than the threshold. This inflates both
winners and losers (intrabar overshoot bias).

These tick-exact variants walk the 5s OHLC: if high/low crossed the TP/SL
threshold during the bar, the trade is closed at EXACTLY the TP/SL price
(with slippage if configured). Returns a special tuple so the engine can
close at the threshold price instead of state.price.

Engine integration: needs an engine variant that respects an "exit at
custom price" instruction. The simplest path is a Python-level wrapper:
the exit rule computes the would-be exit price and stuffs it into
`position.extras['_force_exit_price']`; then a small engine patch closes
at that price if set.

Since modifying the engine is invasive, this module instead does the
clean thing: it returns the close price implicitly via a sentinel-keyed
position extra that the engine reads in `_tick`. We patch engine in
`training_iso_v2/engine.py` to honor this.
"""
from __future__ import annotations

from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.ledger import Position
from training_iso_v2.exits import ExitRule


TICK = 0.25
TICK_VALUE = 0.50  # $/tick MNQ


def _pnl_at_price(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == 'long':
        return (exit_price - entry_price) / TICK * TICK_VALUE
    return (entry_price - exit_price) / TICK * TICK_VALUE


class TickExactTP(ExitRule):
    """Closes at the EXACT TP price if the bar's high (LONG) / low (SHORT)
    crossed the threshold during the 5s bar. Falls back to bar-close pnl
    check (the legacy behavior) if no ohlcv_5s is available.

    Slippage in TICKS added against us at fill.
    """
    name = 'take_profit_exact'

    def __init__(self, usd: float = 20.0, slippage_ticks: float = 0.0):
        self.usd = float(abs(usd))
        self.slippage_ticks = float(slippage_ticks)

    def evaluate(self, state: BarState, position: Position) -> Optional[str]:
        # Compute target exit price from TP$
        target_pnl_ticks = self.usd / TICK_VALUE
        if position.direction == 'long':
            tp_price = position.entry_price + target_pnl_ticks * TICK
            tp_price += self.slippage_ticks * TICK  # need to print higher to fill
        else:
            tp_price = position.entry_price - target_pnl_ticks * TICK
            tp_price -= self.slippage_ticks * TICK

        bar = getattr(state, 'ohlcv_5s', None)
        if bar is not None:
            hi = float(bar['high']); lo = float(bar['low'])
            if position.direction == 'long' and hi >= tp_price:
                # Stuff exit price into extras; engine reads it
                position.extras['_force_exit_price'] = float(tp_price)
                return self.name
            if position.direction == 'short' and lo <= tp_price:
                position.extras['_force_exit_price'] = float(tp_price)
                return self.name
            return None
        # Fallback: bar close
        if position.pnl(state.price) >= self.usd:
            return self.name
        return None


class TickExactSL(ExitRule):
    """Closes at the EXACT SL price if the bar's low (LONG) / high (SHORT)
    crossed the threshold. Slippage in ticks against us at fill.
    """
    name = 'hard_stop_exact'

    def __init__(self, usd: float = -20.0, slippage_ticks: float = 1.0):
        self.usd = -abs(float(usd))   # always negative
        self.slippage_ticks = float(slippage_ticks)

    def evaluate(self, state: BarState, position: Position) -> Optional[str]:
        loss_ticks = abs(self.usd) / TICK_VALUE
        if position.direction == 'long':
            sl_price = position.entry_price - loss_ticks * TICK
            sl_price -= self.slippage_ticks * TICK  # worse fill on adverse move
        else:
            sl_price = position.entry_price + loss_ticks * TICK
            sl_price += self.slippage_ticks * TICK

        bar = getattr(state, 'ohlcv_5s', None)
        if bar is not None:
            hi = float(bar['high']); lo = float(bar['low'])
            if position.direction == 'long' and lo <= sl_price:
                position.extras['_force_exit_price'] = float(sl_price)
                return self.name
            if position.direction == 'short' and hi >= sl_price:
                position.extras['_force_exit_price'] = float(sl_price)
                return self.name
            return None
        if position.pnl(state.price) <= self.usd:
            return self.name
        return None
