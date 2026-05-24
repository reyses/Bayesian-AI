"""FadeAtBand — fade entries at higher-TF ±k·σ band, target lower-TF mean.

User-validated framework (2026-05-08, chart-driven):
    The 15m mean is rarely touched by price; the 5m mean is touched often.
    So 5m mean is the realistic FADE TARGET, while the 15m ±2σ band is
    the OVEREXTENSION TRIGGER.

Strategy:
    WATCH    5s price relative to 15m_mean ± k·15m_sigma
    CONFIRM  price stays outside band for `confirm_bars` (filters spikes)
    ENTRY    fade direction (above band → SHORT; below band → LONG)
    TARGET   5m regression mean

Stateless trigger logic via per-bar consecutive-bar counters held in
`self._state` keyed by day.

NMP-style; FreightTrain-style filters are NOT applied.
"""
from __future__ import annotations

from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy
from training.utils.v2_cols import (price_mean_w, price_sigma_w,
                                                  hurst_w, swing_noise_w,
                                                  vol_sigma_w, price_velocity_w)


class FadeAtBand(Strategy):
    """Fade-trade triggered by 5s price reaching higher-TF ±k·σ band."""

    name = 'FADE_AT_BAND'

    def __init__(self,
                 band_tf: str = '15m',
                 target_tf: str = '5m',
                 k_sigma: float = 2.0,
                 confirm_bars: int = 6,
                 fire_on: str = '5s',
                 # ── Robustness filters (all optional, can be tightened later) ──
                 hurst_tf: str = '5m',
                 hurst_max: Optional[float] = 0.60,
                 swing_noise_tf: str = '1m',
                 swing_noise_max: Optional[float] = None,
                 trend_gate_tf: str = '15m',
                 max_counter_trend_vel: Optional[float] = 25.0,
                 require_divergence: bool = True,
                 divergence_pri_tf: str = '1m',
                 divergence_sec_tf: str = '5m'):
        self.band_tf = band_tf
        self.target_tf = target_tf
        self.k_sigma = k_sigma
        self.confirm_bars = confirm_bars
        self.fire_on = fire_on
        self._mean_col = price_mean_w(band_tf)
        self._sigma_col = price_sigma_w(band_tf)
        self._target_mean_col = price_mean_w(target_tf)
        # Robustness filter columns
        self._hurst_col = hurst_w(hurst_tf) if hurst_tf else None
        self.hurst_max = hurst_max
        self._sn_col = swing_noise_w(swing_noise_tf) if swing_noise_tf else None
        self.swing_noise_max = swing_noise_max
        self._trend_vel_col = (price_velocity_w(trend_gate_tf)
                                       if trend_gate_tf else None)
        self.max_counter_trend_vel = max_counter_trend_vel
        self.require_divergence = require_divergence
        self._div_pri_mean_col = (price_mean_w(divergence_pri_tf)
                                            if divergence_pri_tf else None)
        self._div_sec_mean_col = (price_mean_w(divergence_sec_tf)
                                             if divergence_sec_tf else None)
        # Per-day state
        self._consec_above = 0
        self._consec_below = 0
        self._cur_day = None

    # ── Filter helpers — return False to ABORT entry ────────────────────
    def _passes_hurst(self, state: BarState) -> bool:
        """Hurst < threshold = mean-reverting regime (good for fades)."""
        if self.hurst_max is None or self._hurst_col is None:
            return True
        h = state.get(self._hurst_col)
        if h is None or h != h:    # NaN → permissive
            return True
        return h < self.hurst_max

    def _passes_swing_noise(self, state: BarState) -> bool:
        """Skip if chop is exploding (fades get whipsawed in heavy chop)."""
        if self.swing_noise_max is None or self._sn_col is None:
            return True
        sn = state.get(self._sn_col)
        if sn is None or sn != sn:
            return True
        return sn < self.swing_noise_max

    def _passes_trend_gate(self, state: BarState, direction: str) -> bool:
        """Don't fade against a strongly trending macro.
        For SHORT entry: skip if 15m vel is strongly POSITIVE (don't fight up-trend).
        For LONG entry:  skip if 15m vel is strongly NEGATIVE.
        """
        if self.max_counter_trend_vel is None or self._trend_vel_col is None:
            return True
        v = state.get(self._trend_vel_col)
        if v is None or v != v:
            return True
        if direction == 'short' and v >= self.max_counter_trend_vel:
            return False
        if direction == 'long' and v <= -self.max_counter_trend_vel:
            return False
        return True

    def _passes_divergence(self, state: BarState, direction: str) -> bool:
        """Require 1m vs 5m mean divergence aligned with fade direction.
        For SHORT (we expect price to drop):
          1m mean must be ABOVE 5m mean (short-tier stretched up).
        For LONG: 1m mean BELOW 5m mean.
        """
        if not self.require_divergence:
            return True
        if (self._div_pri_mean_col is None
                  or self._div_sec_mean_col is None):
            return True
        m_pri = state.get(self._div_pri_mean_col)
        m_sec = state.get(self._div_sec_mean_col)
        if m_pri is None or m_sec is None or m_pri != m_pri or m_sec != m_sec:
            return True
        d = m_pri - m_sec
        if direction == 'short' and d <= 0:
            return False
        if direction == 'long' and d >= 0:
            return False
        return True

    def _all_filters_pass(self, state: BarState, direction: str) -> bool:
        return (self._passes_hurst(state)
                      and self._passes_swing_noise(state)
                      and self._passes_trend_gate(state, direction)
                      and self._passes_divergence(state, direction))

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        # Reset state at day boundary
        if state.day != self._cur_day:
            self._cur_day = state.day
            self._consec_above = 0
            self._consec_below = 0

        # Read band & price
        band_mean = state.get(self._mean_col)
        band_sigma = state.get(self._sigma_col)
        if band_mean is None or band_sigma is None:
            return None
        if band_mean != band_mean or band_sigma != band_sigma:    # NaN
            return None
        band_hi = band_mean + self.k_sigma * band_sigma
        band_lo = band_mean - self.k_sigma * band_sigma

        price = state.price

        # Update consecutive-bar counters
        if price > band_hi:
            self._consec_above += 1
            self._consec_below = 0
        elif price < band_lo:
            self._consec_below += 1
            self._consec_above = 0
        else:
            self._consec_above = 0
            self._consec_below = 0

        # Confirmed-fade triggers + robustness filters
        if self._consec_above >= self.confirm_bars:
            self._consec_above = 0    # fire once, reset
            if self._all_filters_pass(state, 'short'):
                return EntrySignal(
                    direction='short', tier=self.name,
                    extras={
                        'band_tf': self.band_tf,
                        'target_tf': self.target_tf,
                        'entry_band_hi': band_hi,
                        'target_mean_col': self._target_mean_col,
                    })
        if self._consec_below >= self.confirm_bars:
            self._consec_below = 0
            if self._all_filters_pass(state, 'long'):
                return EntrySignal(
                    direction='long', tier=self.name,
                    extras={
                        'band_tf': self.band_tf,
                        'target_tf': self.target_tf,
                        'entry_band_lo': band_lo,
                        'target_mean_col': self._target_mean_col,
                    })
        return None
