"""Shared NMP entry helper — V2-native. Used by the 9 ported tiers.

NMP fades extreme z. The seed condition (V2):
    |L3_1m_z_se_15| >= z_threshold[regime]
    AND L3_1m_reversion_prob_15 >= r_threshold[regime]

Direction:
    z > 0 → SHORT (price stretched up; fade DOWN to band)
    z < 0 → LONG  (price stretched down; fade UP to band)

Per-regime thresholds (optional): pass a dict mapping regime label to
{'z_thr': float, 'r_thr': float}. Loaded from
training_iso_v2/output/seed_thresholds_per_regime.json. Falls back to the
universal (z_threshold, r_threshold) defaults when the regime is missing
or no map is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from training_iso_v2.state import BarState, REGIME_VOCAB, is_trend_too_fast
from training_iso_v2.v2_cols import z_se_w, reversion_prob_w, price_velocity_w
from training_iso_v2.strategies.base import EntrySignal, Strategy


@dataclass
class NMPSeed:
    """Result of evaluating the NMP seed at a bar."""
    fired: bool
    direction: str          # 'long' | 'short' | ''
    z: float
    rprob: float


def _resolve_per_regime(per_regime_map: Optional[Dict],
                              regime_idx: int,
                              key: str,
                              default: float) -> float:
    """Look up `key` for the regime label of regime_idx in per_regime_map.
    Returns default when map is None, regime label is missing, or key absent."""
    if not per_regime_map:
        return default
    if regime_idx < 0 or regime_idx >= len(REGIME_VOCAB):
        return default
    label = REGIME_VOCAB[regime_idx]
    cell = per_regime_map.get(label)
    if not cell:
        return default
    return float(cell.get(key, default))


def evaluate_nmp_seed(state: BarState, tf: str = '1m',
                              z_threshold: float = 1.8,
                              r_threshold: float = 0.55,
                              fire_on: str = '1m',
                              per_regime: Optional[Dict] = None) -> NMPSeed:
    """Returns NMPSeed(fired=False) if not at fire boundary OR not qualified.

    When `per_regime` is provided, looks up regime-specific (z_thr, r_thr)
    from state.regime_idx; falls back to the function-level defaults when
    a key is missing.
    """
    ready = {
        '1m': state.is_1m_close,
        '5m': state.is_5m_close,
        '15m': state.is_15m_close,
    }.get(fire_on, False)
    if not ready:
        return NMPSeed(False, '', 0.0, 0.0)

    z = state.get(z_se_w(tf), 0.0)
    r = state.get(reversion_prob_w(tf), 0.0)

    z_thr = _resolve_per_regime(per_regime, state.regime_idx, 'z_thr', z_threshold)
    r_thr = _resolve_per_regime(per_regime, state.regime_idx, 'r_thr', r_threshold)

    if abs(z) < z_thr or r < r_thr:
        return NMPSeed(False, '', float(z), float(r))

    direction = 'short' if z > 0 else 'long'
    return NMPSeed(True, direction, float(z), float(r))


class NMPBaseStrategy(Strategy):
    """Common NMP entry trigger; subclasses override `_qualify` to add filters.

    A trend-velocity gate runs as the FINAL filter on every emitted signal:
    if the 1h price velocity is extreme (|vel| > trend_fast_thr), the entry
    is skipped because:
      - counter-trend entries (fade against fast macro) get overrun
      - pro-trend entries (chasing fast macro) are late to the move
    Validated empirically on 2026_02_12 + 2026_03_03 audit.

    Disable per-strategy by passing `trend_gate_mode=None`.

    ─── 2026-05-10 RETUNE PARAMS (data-validated, IS+OOS) ─────────────────
    z_band_lo, z_band_hi:  filter entries to |z_se| in [lo, hi]
                            empirically optimal: [1.5, 1.8] for most FADE tiers
                            None disables the band (default)
    veto_cells:  iterable of (direction, alignment_category) tuples to VETO
                  alignment_category in {'aligned', 'opposed', 'neutral'}
                  Category = 1h_z_se vs SEED direction (NOT signal direction
                  after flip; flip not used since FLIP rule overfit OOS).
                  e.g.  veto_cells=[('short', 'neutral')]  for FADE_CALM.
    """

    name = 'NMP_BASE'

    def __init__(self, tf: str = '1m', fire_on: str = '1m',
                 z_threshold: float = 1.8, r_threshold: float = 0.55,
                 per_regime: Optional[Dict] = None,
                 trend_fast_thr: float = 25.0,
                 trend_gate_mode: Optional[str] = 'symmetric',
                 z_band_lo: Optional[float] = None,
                 z_band_hi: Optional[float] = None,
                 veto_cells: Optional[list] = None):
        self.tf = tf
        self.fire_on = fire_on
        self.z_threshold = z_threshold
        self.r_threshold = r_threshold
        self.per_regime = per_regime    # optional regime-keyed threshold map
        self.trend_fast_thr = trend_fast_thr
        self.trend_gate_mode = trend_gate_mode
        self._vel_1h_col = price_velocity_w('1h')
        # 2026-05-10 retune params
        self.z_band_lo = z_band_lo
        self.z_band_hi = z_band_hi
        self.veto_cells = veto_cells or []

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        """Subclass hook. Default: emit the seed direction with this tier name."""
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z, 'reversion_prob': seed.rprob})

    def _passes_retune_filters(self, state: BarState, seed: NMPSeed) -> bool:
        """2026-05-10 validated retune: z-band + 1h-regime VETO cells."""
        # E1 z-band
        if self.z_band_lo is not None or self.z_band_hi is not None:
            z = abs(seed.z)
            if self.z_band_lo is not None and z < self.z_band_lo:
                return False
            if self.z_band_hi is not None and z > self.z_band_hi:
                return False
        # E2 VETO 1h-regime cells
        if self.veto_cells:
            z_1h = state.get('L3_1h_z_se_12', 0.0)
            if seed.direction == 'short':
                category = ('aligned' if z_1h >= +0.3
                              else 'opposed' if z_1h <= -0.3
                              else 'neutral')
            else:
                category = ('aligned' if z_1h <= -0.3
                              else 'opposed' if z_1h >= +0.3
                              else 'neutral')
            if (seed.direction, category) in self.veto_cells:
                return False
        return True

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        seed = evaluate_nmp_seed(state, self.tf, self.z_threshold,
                                            self.r_threshold, self.fire_on,
                                            per_regime=self.per_regime)
        if not seed.fired:
            return None
        # 2026-05-10 retune filters BEFORE subclass _qualify (cheap gate)
        if not self._passes_retune_filters(state, seed):
            return None
        sig = self._qualify(state, seed)
        if sig is None:
            return None
        # Trend-velocity gate (post-qualifier)
        if self.trend_gate_mode is not None:
            vel_1h = state.get(self._vel_1h_col, 0.0)
            if is_trend_too_fast(vel_1h, sig.direction,
                                          fast_thr=self.trend_fast_thr,
                                          mode=self.trend_gate_mode):
                return None
        return sig
