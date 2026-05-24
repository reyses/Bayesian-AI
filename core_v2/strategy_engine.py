"""V2-native bar-loop engine.

Per-bar flow:
  1. ledger.update(price)
  2. If in position: walk exits (first match wins) → close if any fired
  3. If flat: walk strategies (first signal wins) → open if any fired
  4. (CNN filter+entry hooks added in Phase 6)

Strategies own NO state across bars; everything they need is in BarState.
The ledger owns position state. The engine owns the loop and dispatch order.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from training.utils.state import BarState
from core_v2.ledger import Ledger, ClosedTrade
from training.strategies.base import Strategy
from core_v2.exits import ExitRule


class Engine:
    def __init__(self, strategies: List[Strategy], exits: List[ExitRule],
                 cnn_filter=None, cnn_entry=None,
                 eod_close: bool = True,
                 entry_extras_hook=None,
                 threshold_map: Optional[dict] = None):
        """
        Args:
            strategies     : entry rules, evaluated in order; first match wins.
            exits          : exit rules, evaluated in order; first match wins.
            cnn_filter     : optional callable(state, signal) -> bool
                              True  = take the signal (open).
                              False = skip.
                              None  = no filter (take every signal).
            cnn_entry      : optional callable(state, ledger) -> EntrySignal | None
                              Run AFTER deterministic strategies fail. Lets CNN
                              spawn its own entry when no rule fired.
            eod_close      : if True, force-close at the last bar of each day.
            entry_extras_hook : optional callable(state) -> dict, attached to
                              Position.extras at open. Used to capture entry-time
                              signal values (e.g., entry_swing_noise) that the
                              exit rules need later.
        """
        self.strategies = strategies
        self.exits = exits
        self.cnn_filter = cnn_filter
        self.cnn_entry = cnn_entry
        self.eod_close = eod_close
        self.entry_extras_hook = entry_extras_hook
        self.threshold_map = threshold_map  # dict from threshold_optimizer.optimize_all_cells
        self.ledger = Ledger()
        self._last_state: Optional[BarState] = None
        self._current_day: Optional[str] = None

    def run(self, bars: Iterable[BarState]) -> List[ClosedTrade]:
        for state in bars:
            # Day boundary force-close
            if self.eod_close and self._current_day is not None and \
                  state.day != self._current_day and not self.ledger.is_flat:
                last = self._last_state
                if last is not None:
                    self.ledger.close(last.price, last.timestamp, 'eod_close')
            self._current_day = state.day

            self._tick(state)
            self._last_state = state

        # End of stream — flush
        if self.eod_close and not self.ledger.is_flat and self._last_state is not None:
            self.ledger.close(self._last_state.price,
                                  self._last_state.timestamp, 'eod_close')

        return self.ledger.closed

    # ------------------------------------------------------------------

    def _tick(self, state: BarState) -> None:
        # 1. Per-bar update
        self.ledger.update(state.price)

        # 2. Exit check
        if not self.ledger.is_flat:
            for rule in self.exits:
                reason = rule.evaluate(state, self.ledger.position)
                if reason is not None:
                    # Honor tick-exact exits via extras['_force_exit_price']
                    exit_price = state.price
                    pos = self.ledger.position
                    if pos is not None and pos.extras:
                        forced = pos.extras.pop('_force_exit_price', None)
                        if forced is not None:
                            exit_price = float(forced)
                    self.ledger.close(exit_price, state.timestamp, reason)
                    break

        # 3. Entry check (only if flat after exits)
        if self.ledger.is_flat:
            self._try_entry(state)

    def _try_entry(self, state: BarState) -> None:
        # Deterministic strategies first — first signal wins
        for strat in self.strategies:
            sig = strat.evaluate(state)
            if sig is None:
                continue
            # CNN filter (optional). True = take, False = skip.
            if self.cnn_filter is not None:
                if not self.cnn_filter(state, sig):
                    return  # filter rejected; do not fall through to other strats
            self._open(state, sig, cnn_filtered=(self.cnn_filter is not None),
                          cnn_generated=False)
            return

        # No deterministic rule fired → optional CNN entry generator
        if self.cnn_entry is not None:
            sig = self.cnn_entry(state, self.ledger)
            if sig is not None:
                self._open(state, sig, cnn_filtered=False, cnn_generated=True)

    def _open(self, state: BarState, sig, cnn_filtered: bool,
                cnn_generated: bool) -> None:
        extras = (sig.extras.copy() if sig.extras else {})
        if self.entry_extras_hook is not None:
            extras.update(self.entry_extras_hook(state))
        # Per-(regime, tier) adaptive thresholds from the optimizer
        if self.threshold_map is not None:
            from training.calibration.threshold_optimizer import lookup_thresholds
            extras['thresholds'] = lookup_thresholds(
                self.threshold_map, state.regime_idx, sig.tier)
        self.ledger.open(
            direction=sig.direction, price=state.price, ts=state.timestamp,
            tier=sig.tier, v2_vector=state.v2_vector,
            regime_idx=state.regime_idx, day=state.day,
            cnn_filtered=cnn_filtered, cnn_generated=cnn_generated,
            extras=extras,
        )
