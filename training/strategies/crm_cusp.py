"""CRM-CUSP FADE — fade reversion at confirmed |z| local maximum.

Source: 2026-05-10 cusp research (reports/findings/cusp_research/summary.md).

Cusp definition (no lookahead):
    At each 1m close t, look back two bars. Bar t-1 is a |z| local max iff
        |z[t-2]| < |z[t-1]|  AND  |z[t-1]| > |z[t]|
    We confirm it at t (one-bar lag). Reversion direction = -sign(z[t-1]).

Edge (OOS, 68 days, z in [1.5, 1.8)):
    h=15m  cusp +$0.28 / thr-only −$0.39   sign flip
    h=30m  cusp +$0.39 / thr-only −$0.04
    h=60m  cusp +$0.82 / thr-only +$0.20

The cusp signal is a SUBSET of NMP threshold entries (~55% retention) — the
ones where reversion is already underway. Higher z buckets ([1.8+) showed
NEGATIVE per-trade across both splits, so this tier hard-gates to [1.5, 1.8).

Fires once per cusp event; cooldown of 5 minutes between fires prevents
back-to-back near-identical signals.
"""
from __future__ import annotations

from typing import Optional, List

from training.utils.state import BarState, is_trend_too_fast
from training.strategies.base import EntrySignal, Strategy
from training.utils.v2_cols import z_se_w, price_velocity_w


class CrmCuspFade(Strategy):
    """Fade reversion at |z| local-max within validated [1.5, 1.8) band."""
    name = 'CRM_CUSP_FADE'

    # Validated band from cusp research (2026-05-10). DO NOT widen past 1.8 —
    # cusps at higher |z| ARE losers, not winners.
    Z_BAND_LO = 1.5
    Z_BAND_HI = 1.8

    # Cooldown in 1m bars. Cusps cluster; without this, the same reversion
    # path can fire 3-4 signals in 5 minutes as |z| oscillates around its
    # peak. 5 minutes is the validated h=5m horizon where edge is weakest;
    # by 5m post-fire, the trade has played out.
    COOLDOWN_1M_BARS = 5

    # Trend-velocity gate (1h price velocity). Same as NMPBaseStrategy default.
    TREND_FAST_THR = 25.0

    def __init__(self):
        self._z_hist: List[float] = []   # last 3 1m z values
        self._cooldown: int = 0          # 1m bars remaining
        self._vel_1h_col = price_velocity_w('1h')
        self._z_col = z_se_w('1m')

    def reset(self):
        self._z_hist = []
        self._cooldown = 0

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not state.is_1m_close:
            return None

        z_now = state.get(self._z_col, 0.0)
        # Update history first; on the bar that triggers we still want it in
        # buffer so subsequent bars can see the full path.
        self._z_hist.append(float(z_now))
        if len(self._z_hist) > 3:
            self._z_hist.pop(0)

        # Decrement cooldown each 1m bar
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if len(self._z_hist) < 3:
            return None

        z_tm2, z_tm1, z_t = self._z_hist
        az_tm2, az_tm1, az_t = abs(z_tm2), abs(z_tm1), abs(z_t)

        # Cusp: |z| strictly peaked at t-1
        if not (az_tm2 < az_tm1 and az_tm1 > az_t):
            return None

        # |z| at cusp must be in the validated reversion band
        if not (self.Z_BAND_LO <= az_tm1 < self.Z_BAND_HI):
            return None

        # Reversion direction at cusp
        direction = 'short' if z_tm1 > 0 else 'long'

        # Trend-velocity gate — skip when 1h is moving too fast (overruns
        # counter-trend fades, late on pro-trend chases)
        vel_1h = state.get(self._vel_1h_col, 0.0)
        if is_trend_too_fast(vel_1h, direction,
                                       fast_thr=self.TREND_FAST_THR,
                                       mode='symmetric'):
            return None

        self._cooldown = self.COOLDOWN_1M_BARS
        return EntrySignal(
            direction=direction, tier=self.name,
            extras={'z_cusp': float(z_tm1),
                    'z_at_decision': float(z_t),
                    'vel_1h': float(vel_1h)})
