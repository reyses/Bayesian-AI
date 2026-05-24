"""IsoOrchestrator — runs N parallel engines on one ticker.

Each engine has ONE strategy and its OWN ledger. All engines see the same
bar stream; one engine's open position does NOT block another's. After the
run, returns a dict of {tier_name: closed_trades_list}.

This is the iso semantic the legacy 2026-04-18 pipeline established:
isolation lets us measure each tier's standalone EV with no cross-tier
interference (no first-signal-wins, no chains, no priority cascade).
"""
from __future__ import annotations

from typing import Dict, Iterable, List

from core_v2.strategy_engine import Engine
from core_v2.exits import default_exit_suite
from training.ledger import ClosedTrade
from training.utils.state import BarState
from training.strategies.base import Strategy


def _build_engine(strategy: Strategy, exits=None,
                       cnn_filter=None, cnn_entry=None,
                       threshold_map=None, entry_extras_hook=None) -> Engine:
    """One engine per strategy."""
    if exits is None:
        exits = default_exit_suite()
    return Engine(strategies=[strategy], exits=exits,
                       cnn_filter=cnn_filter, cnn_entry=cnn_entry,
                       entry_extras_hook=entry_extras_hook,
                       threshold_map=threshold_map)


class IsoOrchestrator:
    """Runs N strategies in N parallel engines on one bar stream.

    Usage:
        orch = IsoOrchestrator(strategies=[KillShot(), Cascade(), FreightTrain()],
                                       threshold_map=thr_map)
        per_tier = orch.run(ticker)   # {'KILL_SHOT': [trades], 'CASCADE': [...], ...}
    """

    def __init__(self, strategies: List[Strategy], exits=None,
                 cnn_filter=None, cnn_entry=None,
                 threshold_map=None, entry_extras_hook=None,
                 eod_close: bool = True):
        self.engines = []
        for s in strategies:
            eng = _build_engine(s, exits=exits, cnn_filter=cnn_filter,
                                       cnn_entry=cnn_entry, threshold_map=threshold_map,
                                       entry_extras_hook=entry_extras_hook)
            eng.eod_close = eod_close
            self.engines.append((s.name, eng))
        self._last_state = None
        self._current_day = None

    def run(self, bars: Iterable[BarState]) -> Dict[str, List[ClosedTrade]]:
        for state in bars:
            # Day boundary force-close (each engine independently)
            if self._current_day is not None and state.day != self._current_day:
                if self._last_state is not None:
                    for _, eng in self.engines:
                        if eng.eod_close and not eng.ledger.is_flat:
                            eng.ledger.close(self._last_state.price,
                                                  self._last_state.timestamp,
                                                  'eod_close')
            self._current_day = state.day

            # Tick every engine on this bar
            for _, eng in self.engines:
                eng._tick(state)

            self._last_state = state

        # End of stream — flush each engine
        if self._last_state is not None:
            for _, eng in self.engines:
                if eng.eod_close and not eng.ledger.is_flat:
                    eng.ledger.close(self._last_state.price,
                                          self._last_state.timestamp,
                                          'eod_close')

        return {name: eng.ledger.closed for name, eng in self.engines}

    def total_pnl(self) -> float:
        return sum(t.pnl for _, eng in self.engines for t in eng.ledger.closed)

    def summary(self) -> Dict[str, Dict]:
        out = {}
        for name, eng in self.engines:
            trades = eng.ledger.closed
            n = len(trades)
            total = sum(t.pnl for t in trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            out[name] = {
                'n': n, 'total_pnl': total,
                'mean_pnl': total / max(n, 1),
                'count_wr': wins / max(n, 1),
            }
        return out
