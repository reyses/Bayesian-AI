"""
PID Oscillation Analyzer — Sub-Minute Band Flip Detector

Watches the 15s quantum state stream. When the market enters a PID-controlled
oscillation regime (price bouncing between Standard Error Bands with low DMI),
identifies the band-touch flip points and emits PIDSignal objects.

Trade logic (Hawaiian Surfer at sub-minute scale):
  - Enter at band touch (price at 1σ or 2σ, oscillation confirmed)
  - Direction: toward center (L1_STABLE) if entering at outer band,
               toward outer band if entering after center cross
  - Exit: at opposite band or at center (depending on entry zone)
  - Stop: outside the band that was touched at entry
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

# Regime detection thresholds
PID_MIN_FORCE       = 0.3    # |term_pid| must exceed this
PID_MIN_OSC_COH     = 0.5    # oscillation_coherence threshold
PID_MAX_Z_ENTER     = 2.0    # don't enter if z >= 2.0 (nightmare field)
PID_MIN_BASE_COH    = 0.4    # base quantum coherence minimum
PID_MAX_ADX         = 30.0   # DMI low = PID regime (visual shows 20-26)
PID_MIN_REGIME_BARS = 3      # must see N consecutive PID bars before entering

# ── TENSION classification thresholds ───────────────────────────────────────
# A PID signal is classified TENSION (dangerous-but-profitable) when any of:
#   1. z_score near outer Roche (>= 1.5σ) — PID fighting possible breakout
#   2. term_pid very large (>= 1.0) — control force maxed out, system under strain
#   3. escape_probability elevated (>= 0.25) — quantum field says breakout is real
#   4. oscillation_coherence falling while regime persists — control degrading
# TENSION signals are logged in shadow but flagged separately.
# They are NEVER enabled for live trading until a dedicated analysis sprint.
PID_TENSION_Z_MIN        = 1.5    # z >= this → approaching outer Roche → TENSION
PID_TENSION_FORCE_MAX    = 1.0    # |term_pid| >= this → maxed-out control → TENSION
PID_TENSION_ESCAPE_MIN   = 0.25   # escape_probability >= this → TENSION
PID_TENSION_COH_DROP     = 0.15   # osc_coh dropped >= this vs 3-bar avg → TENSION


@dataclass
class PIDSignal:
    timestamp:     object       # bar timestamp
    direction:     str          # 'LONG' | 'SHORT'
    entry_price:   float
    target_price:  float        # center band or opposite sigma band
    stop_price:    float        # outside the touched band
    z_score:       float        # entry z_score
    band_touched:  str          # '1sig' | '2sig'
    regime_bars:   int          # how many consecutive PID bars before this signal
    osc_coherence: float        # oscillation_coherence at entry
    term_pid:      float        # PID control force at entry
    pid_class:     str          # 'STABLE' | 'TENSION'
    tension_reason: str         # '' | 'outer_roche' | 'maxed_force' | 'escape_risk' | 'coh_drop'


class PIDOscillationAnalyzer:
    def __init__(self, sigma_per_bar: float = None):
        """
        sigma_per_bar: the current day's sigma (from engine center mass computation).
        Updated per-day via reset().
        """
        self._sigma      = sigma_per_bar or 1.0
        self._regime_n   = 0     # consecutive PID bars seen
        self._signals    = []
        self._osc_coh_history = []   # rolling 3-bar osc_coh for TENSION coh_drop check

    def reset(self, sigma: float):
        """Call at start of each day with the day's regression sigma."""
        self._sigma    = sigma
        self._regime_n = 0
        self._signals  = []
        self._osc_coh_history = []

    def tick(self, state) -> Optional[PIDSignal]:
        """
        Feed one ThreeBodyQuantumState. Returns a PIDSignal if an entry is triggered,
        else None. Call once per 15s bar during the forward pass.
        Signal is classified as STABLE or TENSION for separate shadow analysis.
        """
        force    = abs(getattr(state, 'term_pid', 0.0))
        osc_coh  = getattr(state, 'oscillation_coherence', 0.0)
        base_coh = getattr(state, 'coherence', 0.0)
        adx      = getattr(state, 'adx_strength', 100.0)
        z        = state.z_score
        escape   = getattr(state, 'escape_probability', 0.0)

        # Check if this bar is in PID regime
        in_pid = (force    >= PID_MIN_FORCE
              and osc_coh  >= PID_MIN_OSC_COH
              and base_coh >= PID_MIN_BASE_COH
              and adx      <= PID_MAX_ADX
              and abs(z)   < PID_MAX_Z_ENTER)

        if in_pid:
            self._osc_coh_history.append(osc_coh)
            if len(self._osc_coh_history) > 3:
                self._osc_coh_history.pop(0)
            self._regime_n += 1
        else:
            self._regime_n = 0
            self._osc_coh_history.clear()
            return None

        if self._regime_n < PID_MIN_REGIME_BARS:
            return None   # not enough consecutive PID bars yet

        # Identify band touch and direction
        sigma     = self._sigma
        price     = state.particle_position
        center    = price - z * sigma   # L1_STABLE approximation

        if z <= -1.0:
            direction    = 'LONG'
            target_price = center
            stop_price   = price - 0.5 * sigma
            band_touched = '1sig' if abs(z) < 2.0 else '2sig'
        elif z >= 1.0:
            direction    = 'SHORT'
            target_price = center
            stop_price   = price + 0.5 * sigma
            band_touched = '1sig' if abs(z) < 2.0 else '2sig'
        else:
            return None   # near center, no directional edge

        # ── TENSION classification ───────────────────────────────────────────
        # Dangerous-but-profitable: high reward but misfire risk is large.
        # These are logged separately and NEVER enabled for live trading
        # until a dedicated analysis sprint.
        tension_reason = ''
        if abs(z) >= PID_TENSION_Z_MIN:
            tension_reason = 'outer_roche'      # approaching outer Roche limit
        elif force >= PID_TENSION_FORCE_MAX:
            tension_reason = 'maxed_force'      # PID control maxed out
        elif escape >= PID_TENSION_ESCAPE_MIN:
            tension_reason = 'escape_risk'      # quantum field says breakout is real
        elif (len(self._osc_coh_history) >= 3
              and (self._osc_coh_history[0] - osc_coh) >= PID_TENSION_COH_DROP):
            tension_reason = 'coh_drop'         # coherence degrading mid-regime

        pid_class = 'TENSION' if tension_reason else 'STABLE'

        signal = PIDSignal(
            timestamp     = state.timestamp,
            direction     = direction,
            entry_price   = price,
            target_price  = target_price,
            stop_price    = stop_price,
            z_score       = z,
            band_touched  = band_touched,
            regime_bars   = self._regime_n,
            osc_coherence = osc_coh,
            term_pid      = getattr(state, 'term_pid', 0.0),
            pid_class     = pid_class,
            tension_reason= tension_reason,
        )

        self._signals.append(signal)
        return signal

    @property
    def signals(self) -> List[PIDSignal]:
        return list(self._signals)
