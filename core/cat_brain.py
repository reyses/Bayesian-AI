"""Cat Brain -- rolling delta regime classifier.

Smarter than a lizard (counter), dumber than a crow (k-NN).
Uses a rolling window of bar-over-bar deltas to classify the current
market regime and provide entry/exit guidance.

The cat doesn't calculate -- it sees recent movement patterns and reacts.
No cumulative state. No absolute thresholds. Pure delta-based.

Channels (all bar-over-bar deltas, computed over rolling window):
  - z_trend:      mean z-score over window (drift direction)
  - fm_direction:  F_momentum sign persistence (trending or oscillating)
  - vol_flow:     signed volume accumulation direction
  - dmi_slope:    DMI gap trend (widening = strengthening, narrowing = weakening)
  - entropy_trend: entropy change (rising = chaos, falling = order)
  - coherence_trend: oscillation coherence change (tightening or loosening)
  - prob_shift:   P_at_center delta (state probability mass moving)
  - sigma_change: regression band width change (expanding or contracting)

Regime classification:
  TRENDING_WITH:    strong directional move, sensors aligned
  TRENDING_AGAINST: strong move opposite to proposed trade
  TRANSITIONING:    regime changing (peak territory -- trade the reversal)
  CHOPPY:           no direction, oscillating (sit out)
  EXHAUSTING:       trend losing energy (exit territory)
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import numpy as np


# Regime labels
REGIME_TRENDING_WITH = 'TRENDING_WITH'
REGIME_TRENDING_AGAINST = 'TRENDING_AGAINST'
REGIME_TRANSITIONING = 'TRANSITIONING'
REGIME_CHOPPY = 'CHOPPY'
REGIME_EXHAUSTING = 'EXHAUSTING'


@dataclass
class CatState:
    """Current cat brain assessment."""
    regime: str = REGIME_CHOPPY
    direction_bias: float = 0.0     # -1.0 (strong SHORT) to +1.0 (strong LONG)
    confidence: float = 0.0         # 0.0 (no idea) to 1.0 (certain)
    should_trade: bool = False      # cat says "enter" or "sit out"
    exit_urgency: float = 0.0       # 0.0 (hold) to 1.0 (exit NOW)

    # Rolling delta channel values (for logging/debugging)
    z_trend: float = 0.0
    fm_direction: float = 0.0
    vol_flow: float = 0.0
    dmi_slope: float = 0.0
    entropy_trend: float = 0.0
    coherence_trend: float = 0.0
    prob_shift: float = 0.0
    sigma_change: float = 0.0


class CatBrain:
    """Rolling delta regime classifier.

    Consumes MarketState objects one bar at a time. Maintains a rolling
    window of deltas. Classifies the current regime from the delta pattern.

    Parameters
    ----------
    window : int
        Number of bars in the rolling window. Default 200.
        At 1m bars = 3.3 hours. At 15s bars = 50 minutes.
    """

    def __init__(self, window: int = 200):
        self._window = window

        # Rolling buffers for each channel (bar-over-bar deltas)
        self._z_scores = deque(maxlen=window)
        self._fm_deltas = deque(maxlen=window)
        self._vol_deltas = deque(maxlen=window)
        self._dmi_diffs = deque(maxlen=window)  # dmi_plus - dmi_minus
        self._entropy_vals = deque(maxlen=window)
        self._coherence_vals = deque(maxlen=window)
        self._prob0_vals = deque(maxlen=window)  # P_at_center
        self._sigma_vals = deque(maxlen=window)

        # Previous bar state (for computing deltas)
        self._prev_fm = 0.0
        self._prev_vol = 0.0
        self._prev_entropy = 0.0
        self._prev_coherence = 0.0
        self._prev_prob0 = 0.0
        self._prev_sigma = 0.0
        self._prev_dmi_diff = 0.0

        self._bars_seen = 0
        self._state = CatState()

    @property
    def state(self) -> CatState:
        return self._state

    @property
    def is_warmed_up(self) -> bool:
        return self._bars_seen >= self._window // 2  # usable after half window

    def update(self, market_state) -> CatState:
        """Process one bar's MarketState, update rolling deltas, classify regime."""
        # Extract fields from MarketState
        z = getattr(market_state, 'z_score', 0.0) or 0.0
        fm = getattr(market_state, 'F_momentum', 0.0) or 0.0
        vol = getattr(market_state, 'volume_delta', 0.0) or 0.0
        dmi_p = getattr(market_state, 'dmi_plus', 0.0) or 0.0
        dmi_m = getattr(market_state, 'dmi_minus', 0.0) or 0.0
        entropy = getattr(market_state, 'entropy_normalized', 0.0) or 0.0
        coherence = getattr(market_state, 'oscillation_entropy_normalized', 0.0) or 0.0
        prob0 = getattr(market_state, 'P_at_center', 0.0) or 0.0
        sigma = getattr(market_state, 'regression_sigma', 0.0) or 0.0

        dmi_diff = dmi_p - dmi_m

        # Compute deltas
        fm_delta = fm - self._prev_fm
        vol_delta = vol - self._prev_vol
        entropy_delta = entropy - self._prev_entropy
        coherence_delta = coherence - self._prev_coherence
        prob0_delta = prob0 - self._prev_prob0
        sigma_delta = sigma - self._prev_sigma
        dmi_diff_delta = dmi_diff - self._prev_dmi_diff

        # Store previous
        self._prev_fm = fm
        self._prev_vol = vol
        self._prev_entropy = entropy
        self._prev_coherence = coherence
        self._prev_prob0 = prob0
        self._prev_sigma = sigma
        self._prev_dmi_diff = dmi_diff

        # Append to rolling buffers
        self._z_scores.append(z)
        self._fm_deltas.append(fm_delta)
        self._vol_deltas.append(vol_delta)
        self._dmi_diffs.append(dmi_diff)
        self._entropy_vals.append(entropy)
        self._coherence_vals.append(coherence)
        self._prob0_vals.append(prob0)
        self._sigma_vals.append(sigma)

        self._bars_seen += 1

        if not self.is_warmed_up:
            self._state = CatState(regime=REGIME_CHOPPY, confidence=0.0)
            return self._state

        # ── Compute rolling delta channels ──
        z_arr = np.array(self._z_scores)
        fm_arr = np.array(self._fm_deltas)
        dmi_arr = np.array(self._dmi_diffs)
        ent_arr = np.array(self._entropy_vals)
        coh_arr = np.array(self._coherence_vals)
        sig_arr = np.array(self._sigma_vals)

        # Channel 1: z_trend -- mean z-score (positive = above center, negative = below)
        z_trend = float(np.mean(z_arr))

        # Channel 2: fm_direction -- what fraction of recent bars had positive F_momentum delta
        fm_positive_frac = float(np.mean(fm_arr > 0))
        fm_direction = fm_positive_frac * 2 - 1  # -1 to +1

        # Channel 3: vol_flow -- mean signed volume over window
        vol_flow = float(np.mean(self._vol_deltas))

        # Channel 4: dmi_slope -- is DMI gap widening or narrowing?
        if len(dmi_arr) >= 10:
            dmi_recent = float(np.mean(dmi_arr[-10:]))
            dmi_older = float(np.mean(dmi_arr[:10]))
            dmi_slope = dmi_recent - dmi_older
        else:
            dmi_slope = 0.0

        # Channel 5: entropy_trend -- rising entropy = increasing chaos
        if len(ent_arr) >= 10:
            entropy_trend = float(np.mean(ent_arr[-10:])) - float(np.mean(ent_arr[:10]))
        else:
            entropy_trend = 0.0

        # Channel 6: coherence_trend -- rising coherence = tightening oscillation
        if len(coh_arr) >= 10:
            coherence_trend = float(np.mean(coh_arr[-10:])) - float(np.mean(coh_arr[:10]))
        else:
            coherence_trend = 0.0

        # Channel 7: prob_shift -- is probability mass moving?
        prob_shift = float(prob0 - np.mean(self._prob0_vals))

        # Channel 8: sigma_change -- bands expanding or contracting?
        if len(sig_arr) >= 10:
            sigma_change = float(np.mean(sig_arr[-10:])) - float(np.mean(sig_arr[:10]))
        else:
            sigma_change = 0.0

        # ── Classify regime from channels ──
        abs_z = abs(z_trend)
        abs_dmi = abs(float(np.mean(dmi_arr[-20:]))) if len(dmi_arr) >= 20 else 0.0
        mean_coh = float(np.mean(coh_arr[-20:])) if len(coh_arr) >= 20 else 0.0

        # Direction bias: which way is the market leaning?
        direction_bias = 0.0
        if abs_z > 0.1:
            direction_bias += np.sign(z_trend) * min(abs_z, 1.0) * 0.4
        if abs_dmi > 5.0:
            dmi_sign = 1.0 if np.mean(dmi_arr[-20:]) > 0 else -1.0
            direction_bias += dmi_sign * min(abs_dmi / 30.0, 1.0) * 0.4
        if abs(fm_direction) > 0.1:
            direction_bias += fm_direction * 0.2
        direction_bias = float(np.clip(direction_bias, -1.0, 1.0))

        # Regime classification
        regime = REGIME_CHOPPY
        confidence = 0.0

        # TRENDING: strong z + DMI agrees + coherence low (not oscillating)
        if abs_z > 0.3 and abs_dmi > 10.0 and mean_coh < 0.6:
            regime = REGIME_TRENDING_WITH  # direction set by bias
            confidence = min(1.0, (abs_z + abs_dmi / 30.0) / 2.0)

        # TRANSITIONING: DMI slope reversing + entropy rising + coherence changing
        elif abs(dmi_slope) > 5.0 and abs(entropy_trend) > 0.05:
            regime = REGIME_TRANSITIONING
            confidence = min(1.0, abs(dmi_slope) / 15.0)

        # EXHAUSTING: strong z but DMI narrowing + coherence rising
        elif abs_z > 0.2 and dmi_slope * np.sign(z_trend) < -3.0:
            regime = REGIME_EXHAUSTING
            confidence = min(1.0, abs(dmi_slope) / 10.0)

        # CHOPPY: low z + low DMI + high coherence (tight oscillation)
        elif abs_z < 0.2 and abs_dmi < 10.0:
            regime = REGIME_CHOPPY
            confidence = min(1.0, (1.0 - abs_z) * (1.0 - abs_dmi / 30.0))

        # ── Trade guidance ──
        should_trade = regime in (REGIME_TRANSITIONING, REGIME_TRENDING_WITH)

        # Exit urgency: high when regime is exhausting or transitioning against
        exit_urgency = 0.0
        if regime == REGIME_EXHAUSTING:
            exit_urgency = confidence * 0.8
        elif regime == REGIME_TRANSITIONING:
            exit_urgency = confidence * 0.5

        self._state = CatState(
            regime=regime,
            direction_bias=direction_bias,
            confidence=confidence,
            should_trade=should_trade,
            exit_urgency=exit_urgency,
            z_trend=z_trend,
            fm_direction=fm_direction,
            vol_flow=vol_flow,
            dmi_slope=dmi_slope,
            entropy_trend=entropy_trend,
            coherence_trend=coherence_trend,
            prob_shift=prob_shift,
            sigma_change=sigma_change,
        )
        return self._state

    def should_enter_peak(self, peak_direction: str) -> tuple:
        """Should the cat approve this peak entry?

        Returns (approved: bool, reason: str).
        Peak fires are approved when:
          - Regime is TRANSITIONING (peak = the regime change)
          - Regime is TRENDING and peak direction matches trend
        Peak fires are blocked when:
          - Regime is CHOPPY (peak is noise)
          - Regime is TRENDING_AGAINST (peak is against the freight train)
        """
        if not self.is_warmed_up:
            return True, 'cat_warmup'  # let it through until warmed up

        s = self._state
        trade_sign = 1.0 if peak_direction == 'LONG' else -1.0

        # TRANSITIONING: this is peak territory -- always approve
        if s.regime == REGIME_TRANSITIONING:
            return True, f'cat_transition(conf={s.confidence:.2f})'

        # TRENDING: approve only if peak direction matches trend
        if s.regime == REGIME_TRENDING_WITH:
            if trade_sign * s.direction_bias > 0:
                return True, f'cat_trend_with(bias={s.direction_bias:+.2f})'
            else:
                return False, f'cat_trend_against(bias={s.direction_bias:+.2f})'

        # EXHAUSTING: approve reversal (peak catches the turn)
        if s.regime == REGIME_EXHAUSTING:
            if trade_sign * s.direction_bias < 0:  # reversal of exhausting trend
                return True, f'cat_exhaust_reversal(bias={s.direction_bias:+.2f})'
            else:
                return False, f'cat_exhaust_continuation(bias={s.direction_bias:+.2f})'

        # CHOPPY: block -- peak is noise
        if s.regime == REGIME_CHOPPY:
            return False, f'cat_choppy(conf={s.confidence:.2f})'

        return True, 'cat_default'

    def should_exit(self, trade_side: str) -> tuple:
        """Should the cat recommend exiting?

        Returns (should_exit: bool, urgency: float, reason: str).
        """
        if not self.is_warmed_up:
            return False, 0.0, 'cat_warmup'

        s = self._state
        trade_sign = 1.0 if trade_side == 'long' else -1.0

        # TRANSITIONING against trade: the regime is changing away from us
        if s.regime == REGIME_TRANSITIONING and trade_sign * s.direction_bias < 0:
            return True, 0.8, f'cat_regime_flip(bias={s.direction_bias:+.2f})'

        # EXHAUSTING: the trend we're riding is dying
        if s.regime == REGIME_EXHAUSTING and trade_sign * s.direction_bias > 0:
            return True, s.exit_urgency, f'cat_exhausting(urg={s.exit_urgency:.2f})'

        # Strong trend against: we're on the wrong side
        if s.regime == REGIME_TRENDING_WITH and trade_sign * s.direction_bias < -0.5:
            return True, 0.9, f'cat_wrong_side(bias={s.direction_bias:+.2f})'

        return False, 0.0, 'cat_hold'

    def get_direction(self) -> tuple:
        """Cat's opinion on direction.

        Returns (direction: str, confidence: float).
        """
        s = self._state
        if abs(s.direction_bias) < 0.1:
            return 'NEUTRAL', 0.0
        elif s.direction_bias > 0:
            return 'LONG', abs(s.direction_bias)
        else:
            return 'SHORT', abs(s.direction_bias)
