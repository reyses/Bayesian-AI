"""Filtered Regime-Aware NMP — RegimeAwareReversion + per-cell quality gates.

For each (regime_idx, direction) cell with a known quality filter
(loaded from training_iso_v2/output/cell_filters.json), check the cell's top
discriminator at entry. If the value is in the loser tail per the
empirical-Bayes-learned threshold, SKIP the trade.

The filter is applied AFTER the flip rule — so the (regime, direction)
keys correspond to the FINAL post-flip direction, not the original
fade-thesis direction.

Filter spec format (per training_v2/cell_filters.py):
    {
        "feature": "L1_1m_bar_range",
        "threshold": 25.7,
        "skip_above": True,    # skip if value > threshold
    }
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Set, Tuple

from training.utils.state import BarState
from training.strategies.base import EntrySignal
from training.strategies.regime_aware import RegimeAwareReversion, DEFAULT_FLIP_CELLS


DEFAULT_FILTERS_PATH = 'training_iso_v2/output/cell_filters.json'


class FilteredRegimeAwareReversion(RegimeAwareReversion):
    """RegimeAwareReversion + per-cell entry quality filters.

    On signal emission:
        1. Run REVERSION trigger
        2. Apply (regime, direction) flip rule (from RegimeAwareReversion)
        3. NEW: check the (regime, FINAL_direction) filter; skip if value in loser tail
    """
    name = 'NMP_FILTERED'

    def __init__(self, filters_path: Optional[str] = None,
                 flip_cells: Optional[Set[Tuple[int, str]]] = None,
                 **kwargs):
        super().__init__(flip_cells=flip_cells or DEFAULT_FLIP_CELLS,
                              **kwargs)
        self.filters_path = filters_path or DEFAULT_FILTERS_PATH
        self.filters: Dict[Tuple[int, str], Dict] = {}
        self._load_filters()

    def _load_filters(self):
        if not os.path.exists(self.filters_path):
            self.filters = {}
            return
        with open(self.filters_path, 'r') as f:
            raw = json.load(f)
        # Keys come as "regime_idx|direction"
        self.filters = {}
        for key, spec in raw.items():
            try:
                r_str, d = key.split('|', 1)
                self.filters[(int(r_str), str(d))] = spec
            except ValueError:
                continue

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        sig = super().evaluate(state)
        if sig is None:
            return None
        # sig.direction is now the FINAL direction (post-flip).
        cell_key = (int(state.regime_idx), str(sig.direction))
        flt = self.filters.get(cell_key)
        if flt is None:
            return sig
        # Look up the feature value at entry
        feature = flt.get('feature')
        threshold = flt.get('threshold')
        skip_above = bool(flt.get('skip_above', False))
        if feature is None or threshold is None:
            return sig
        val = state.get(feature, default=float('nan'))
        if val != val:  # NaN — pass through
            return sig
        # Skip if in loser tail
        if skip_above and val > threshold:
            return None
        if not skip_above and val < threshold:
            return None
        # Tag as filtered for audit
        extras = dict(sig.extras or {})
        extras['filter_passed'] = True
        extras['filter_feature'] = feature
        extras['filter_threshold'] = threshold
        return EntrySignal(direction=sig.direction, tier=sig.tier, extras=extras)
