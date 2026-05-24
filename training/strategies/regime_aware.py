"""Regime-aware NMP — REVERSION + (regime × direction) flip rule.

Discovered 2026-05-04 from per-cell EDA on 19,106 IS NMP trades:
    UP_SMOOTH × short  -> flip to long  (NMP shorts in uptrend bleed)
    UP_CHOPPY × short  -> flip to long
    DOWN_SMOOTH × long -> flip to short (NMP longs in downtrend bleed)

Walk-forward validated inside IS at +$88/day, 95% CI [+$21, +$165] — significant.
OOS delta +$68/day, all 3 flip cells positive direction.

The strategy fires on the same condition as ReversionFromExtreme but consults
the flip table at signal emission. When the (regime, direction) hits a flip
cell, the direction is inverted and the tier is renamed `NMP_FLIP` so the
ledger and downstream analysis can distinguish kept-vs-flipped entries.
"""
from __future__ import annotations

from typing import Optional, Set, Tuple

from training.utils.state import BarState, REGIME_VOCAB
from training.strategies.base import EntrySignal
from training.strategies.reversion import ReversionFromExtreme


# (regime_idx, original_direction) cells where the flip helps.
# Source: 2026-05-04 IS regret analysis on 19,106 NMP trades.
DEFAULT_FLIP_CELLS: Set[Tuple[int, str]] = {
    (REGIME_VOCAB.index('UP_SMOOTH'), 'short'),
    (REGIME_VOCAB.index('UP_CHOPPY'), 'short'),
    (REGIME_VOCAB.index('DOWN_SMOOTH'), 'long'),
}


class RegimeAwareReversion(ReversionFromExtreme):
    """REVERSION with regime-direction flip rule.

    Same trigger as ReversionFromExtreme; inverts direction when the
    (regime_idx, direction) pair matches a flip-cell.
    """
    name = 'NMP_REGIME'

    def __init__(self, flip_cells: Optional[Set[Tuple[int, str]]] = None,
                 flip_tier_name: str = 'NMP_FLIP',
                 keep_tier_name: str = 'NMP_KEEP',
                 **kwargs):
        super().__init__(**kwargs)
        self.flip_cells = flip_cells if flip_cells is not None else DEFAULT_FLIP_CELLS
        self.flip_tier_name = flip_tier_name
        self.keep_tier_name = keep_tier_name

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        sig = super().evaluate(state)
        if sig is None:
            return None
        key = (int(state.regime_idx), str(sig.direction))
        if key in self.flip_cells:
            # Flip direction; re-tag tier so we can audit
            new_dir = 'short' if sig.direction == 'long' else 'long'
            extras = dict(sig.extras or {})
            extras['flipped_from'] = sig.direction
            extras['regime_2d'] = state.regime_2d
            return EntrySignal(direction=new_dir, tier=self.flip_tier_name,
                                  extras=extras)
        # Kept as-is — relabel for clarity
        extras = dict(sig.extras or {})
        extras['regime_2d'] = state.regime_2d
        return EntrySignal(direction=sig.direction, tier=self.keep_tier_name,
                              extras=extras)
