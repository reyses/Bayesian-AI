"""
TrajectoryEngine — Reads the P(D) decay curve and makes trading decisions.

Takes calibrated trajectory curves from multiple TFs and determines:
  - Regime: TRENDING / INFLECTION / CHOP
  - Direction: LONG / SHORT / FLAT
  - Action: ENTER / HOLD / EXIT / WAIT

The engine does NOT use fixed thresholds. It reads the SHAPE of the curve
and the DISAGREEMENT between near and far horizons.

Usage:
  engine = TrajectoryEngine(calibrators={'1h': cal_1h, '1m': cal_1m, '1s': cal_1s})
  signal = engine.update('1m', raw_trajectory)  # called every 1m bar
  signal = engine.update('1s', raw_trajectory)  # called every 1s bar
  action = engine.decide()  # returns TradeAction
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TradeAction:
    """Output from TrajectoryEngine."""
    action: str       # 'ENTER', 'EXIT', 'HOLD', 'WAIT'
    direction: str    # 'long', 'short', 'flat'
    confidence: float # 0-1, how sure
    sight: int        # bars of clear runway
    reason: str       # human-readable


class TFState:
    """Tracks trajectory state for one timeframe."""

    def __init__(self, tf, horizons, chop_zones):
        self.tf = tf
        self.horizons = horizons
        self.chop_zones = chop_zones  # list of (low, high) per horizon
        self.n_horizons = len(horizons)

        # Rolling history of P(D) at n+1 for regime detection
        self.p_history = []   # last N bars of calibrated P(D) at n+1
        self.history_len = 10  # bars of lookback for regime

        # Current trajectory
        self.curve = None       # (K,) calibrated P(D) per horizon
        self.prev_curve = None  # previous bar's curve

    def update(self, calibrated_curve):
        """Update with new calibrated trajectory curve."""
        self.prev_curve = self.curve
        self.curve = calibrated_curve

        # Track n+1 history for regime detection
        if calibrated_curve is not None:
            self.p_history.append(calibrated_curve[0])
            if len(self.p_history) > self.history_len:
                self.p_history = self.p_history[-self.history_len:]

    @property
    def direction(self):
        """Current direction from n+1."""
        if self.curve is None:
            return 'flat'
        return 'long' if self.curve[0] > 0.5 else 'short'

    @property
    def n1_confidence(self):
        """Confidence at n+1 (0 = uncertain, 1 = certain)."""
        if self.curve is None:
            return 0.0
        return abs(self.curve[0] - 0.5) * 2

    @property
    def regime(self):
        """Detect regime from P(D) history at n+1.

        TRENDING: P(D) consistently above or below chop zone
        INFLECTION: P(D) rapidly crossing through 50%
        CHOP: P(D) stuck near 50% for multiple bars
        """
        if len(self.p_history) < 3:
            return 'UNKNOWN'

        recent = np.array(self.p_history[-5:])
        chop_low, chop_high = self.chop_zones[0]  # n+1 chop zone

        # Chop: most recent bars are within chop zone
        in_chop = ((recent >= chop_low) & (recent <= chop_high)).mean()
        if in_chop > 0.6:
            return 'CHOP'

        # Inflection: rapid crossing through 50%
        if len(recent) >= 3:
            crossed = False
            for i in range(1, len(recent)):
                if (recent[i-1] > 0.5 and recent[i] < 0.5) or \
                   (recent[i-1] < 0.5 and recent[i] > 0.5):
                    crossed = True
                    break
            if crossed:
                return 'INFLECTION'

        return 'TRENDING'

    @property
    def near_far_disagreement(self):
        """Disagreement between near (n+1,n+2) and far (n-1,n-2) horizons.

        Positive = near horizons seeing something far horizons don't (peak approaching).
        """
        if self.curve is None or self.n_horizons < 4:
            return 0.0
        near = np.mean(self.curve[:2])
        far = np.mean(self.curve[-2:])
        return near - far

    @property
    def sight_distance(self):
        """How many horizons are outside the chop zone in the same direction."""
        if self.curve is None:
            return 0
        direction = self.curve[0] > 0.5
        sight = 0
        for hi in range(self.n_horizons):
            chop_low, chop_high = self.chop_zones[hi]
            p = self.curve[hi]
            # Outside chop zone AND same direction as n+1
            if (p > chop_high and direction) or (p < chop_low and not direction):
                sight = hi + 1
            else:
                break
        return sight

    @property
    def trajectory_slope(self):
        """Rate of change of the curve (negative = decaying)."""
        if self.curve is None or self.n_horizons < 2:
            return 0.0
        # Linear fit slope across horizons
        x = np.arange(self.n_horizons)
        return float(np.polyfit(x, self.curve, 1)[0])


class TrajectoryEngine:
    """Multi-TF trajectory navigation engine."""

    def __init__(self, calibrators=None, horizons_per_tf=None):
        """
        calibrators: dict of tf -> TrajectoryCalibrator
        horizons_per_tf: dict of tf -> list of horizons
        """
        self.calibrators = calibrators or {}
        self.tf_states: Dict[str, TFState] = {}

        if horizons_per_tf:
            for tf, horizons in horizons_per_tf.items():
                chop_zones = self.calibrators[tf].chop_zones if tf in self.calibrators \
                    else [(0.45, 0.55)] * len(horizons)
                self.tf_states[tf] = TFState(tf, horizons, chop_zones)

        # Trade state
        self.in_trade = False
        self.trade_dir = ''
        self.entry_bar = 0
        self.bar_count = 0

    def update(self, tf, raw_curve):
        """Update one TF with its raw P(D) trajectory and return calibrated curve.

        raw_curve: (K,) raw P(long) per horizon from TrajectoryPredictor
        """
        if tf not in self.tf_states:
            return None

        # Calibrate
        if tf in self.calibrators:
            calibrated = self.calibrators[tf].calibrate(raw_curve)
        else:
            calibrated = raw_curve

        self.tf_states[tf].update(calibrated)
        return calibrated

    def decide(self):
        """Make trading decision from all TF states.

        Priority: 1h sets direction, 1m provides wave entry/exit, 1s confirms timing.
        """
        self.bar_count += 1

        # Get state per TF (may not all be present)
        s_1h = self.tf_states.get('1h')
        s_1m = self.tf_states.get('1m')
        s_1s = self.tf_states.get('1s')

        # Primary decision TF is 1m (or lowest available)
        primary = s_1m or s_1s or s_1h
        if primary is None or primary.curve is None:
            return TradeAction('WAIT', 'flat', 0, 0, 'no_data')

        # --- IN TRADE: check exits ---
        if self.in_trade:
            return self._check_exit(s_1h, s_1m, s_1s)

        # --- NOT IN TRADE: check entries ---
        return self._check_entry(s_1h, s_1m, s_1s)

    def _check_entry(self, s_1h, s_1m, s_1s):
        """Entry logic: 1h direction + 1m wave starting + 1s confirms."""

        # 1h must have direction (not chop)
        if s_1h and s_1h.regime == 'CHOP':
            return TradeAction('WAIT', 'flat', 0, 0, '1h_chop')

        # Structural direction from 1h (or 1m if no 1h)
        struct_dir = s_1h.direction if s_1h else (s_1m.direction if s_1m else 'flat')

        # 1m must be leaving chop zone in the 1h direction
        if s_1m:
            if s_1m.regime == 'CHOP':
                return TradeAction('WAIT', struct_dir, 0, 0, '1m_chop')

            # 1m direction must agree with 1h
            if s_1m.direction != struct_dir:
                return TradeAction('WAIT', struct_dir, 0, 0, '1m_against_1h')

            # 1m must have sight (runway)
            if s_1m.sight_distance < 2:
                return TradeAction('WAIT', struct_dir, s_1m.n1_confidence, s_1m.sight_distance,
                                   '1m_short_sight')

        # 1s timing confirmation (if available)
        if s_1s:
            if s_1s.direction != struct_dir:
                return TradeAction('WAIT', struct_dir, 0, 0, '1s_not_confirming')

        # All conditions met — enter
        confidence = s_1m.n1_confidence if s_1m else (s_1h.n1_confidence if s_1h else 0)
        sight = s_1m.sight_distance if s_1m else 0

        self.in_trade = True
        self.trade_dir = struct_dir
        self.entry_bar = self.bar_count

        return TradeAction('ENTER', struct_dir, confidence, sight,
                           f'1h={s_1h.direction if s_1h else "?"} '
                           f'1m_sight={sight} regime={s_1m.regime if s_1m else "?"}')

    def _check_exit(self, s_1h, s_1m, s_1s):
        """Exit logic: trajectory shape drives all exits."""

        # Primary: 1m trajectory
        if s_1m and s_1m.curve is not None:
            # 1. n+1 flipped direction — immediate exit
            if s_1m.direction != self.trade_dir:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                   f'1m_flipped n+1={s_1m.curve[0]:.2f}')

            # 2. Regime changed to INFLECTION — peak approaching
            if s_1m.regime == 'INFLECTION':
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                   f'1m_inflection')

            # 3. Regime changed to CHOP — signal lost
            if s_1m.regime == 'CHOP':
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, 0, 0, '1m_chop')

            # 4. Near-far disagreement spiking — peak imminent
            disagree = s_1m.near_far_disagreement
            if self.trade_dir == 'long' and disagree > 0.15:
                # Near horizons dropping while far still confident = peak
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                   f'1m_disagree={disagree:.2f}')
            elif self.trade_dir == 'short' and disagree < -0.15:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                   f'1m_disagree={disagree:.2f}')

            # 5. Sight distance contracted to 1 — only see one bar ahead
            if s_1m.sight_distance <= 1:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence,
                                   s_1m.sight_distance, '1m_sight=1')

        # 1h structural flip — hard exit regardless
        if s_1h and s_1h.direction != self.trade_dir and s_1h.regime != 'CHOP':
            self.in_trade = False
            return TradeAction('EXIT', self.trade_dir, 0, 0,
                               f'1h_flipped to {s_1h.direction}')

        # Hold
        sight = s_1m.sight_distance if s_1m else 0
        conf = s_1m.n1_confidence if s_1m else 0
        return TradeAction('HOLD', self.trade_dir, conf, sight, 'trajectory_holds')

    def on_exit(self):
        """Called when position is closed externally."""
        self.in_trade = False
        self.trade_dir = ''

    def on_fill(self, direction):
        """Called when entry fill is confirmed."""
        self.in_trade = True
        self.trade_dir = direction
        self.entry_bar = self.bar_count
