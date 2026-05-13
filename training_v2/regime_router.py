"""Regime → eligible-strategies router.

Day-level routing, not bar-level. The day's regime_2d label is constant
across the day (per atlas_regime_labeler_2d.py). We use it to enable/disable
strategies wholesale for the day.

EDA-driven mapping (2026-05-01 finding):
    UP_SMOOTH / DOWN_SMOOTH  → trend-follow (MA-align edge is here)
    UP_CHOPPY / DOWN_CHOPPY  → trend-follow with caution (smaller cohort)
    FLAT_SMOOTH              → low-eligibility (chop edge weak; reversion only)
    FLAT_CHOPPY              → reversion-favored
    UNKNOWN                  → no-trade

Phase 4 will plug specific strategy names; for Phase 1-3 the router just
returns "all enabled" so the deterministic strategies can be measured as-is.
"""
from __future__ import annotations

from typing import List

from training_v2.strategies.base import Strategy


# Regime → list of strategy NAMES allowed to fire today.
# Use '*' as a wildcard meaning "all strategies".
DEFAULT_ELIGIBILITY = {
    'UP_SMOOTH': ['*'],
    'UP_CHOPPY': ['*'],
    'DOWN_SMOOTH': ['*'],
    'DOWN_CHOPPY': ['*'],
    'FLAT_SMOOTH': ['*'],   # tighten in Phase 4
    'FLAT_CHOPPY': ['*'],   # tighten in Phase 4
    'UNKNOWN': ['*'],
}


class RegimeRouter:
    def __init__(self, eligibility=None):
        self.eligibility = dict(eligibility or DEFAULT_ELIGIBILITY)

    def filter(self, regime_2d: str, strategies: List[Strategy]) -> List[Strategy]:
        allowed = self.eligibility.get(regime_2d, ['*'])
        if '*' in allowed:
            return list(strategies)
        return [s for s in strategies if s.name in allowed]
