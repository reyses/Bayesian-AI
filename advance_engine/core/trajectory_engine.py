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

        # 1h structural bias — persists between 1h updates
        self._structural_dir = 'flat'     # 'long', 'short', 'flat'
        self._structural_conf = 0.0       # 0-1 how strong the bias is
        self._structural_sight = 0        # hours of runway
        self._structural_updated = 0      # bar count when last updated

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

        # Update structural bias when 1h updates
        if tf == '1h' and calibrated is not None:
            p_n1 = calibrated[0]
            conf = abs(p_n1 - 0.5) * 2
            direction = 'long' if p_n1 > 0.5 else 'short'

            # Sight: count horizons outside chop zone in same direction
            sight = 0
            for hi in range(len(calibrated)):
                if (direction == 'long' and calibrated[hi] > 0.55) or \
                   (direction == 'short' and calibrated[hi] < 0.45):
                    sight = hi + 1
                else:
                    break

            # Only set bias if confident enough
            if conf >= 0.5:
                self._structural_dir = direction
                self._structural_conf = conf
                self._structural_sight = sight
            else:
                self._structural_dir = 'flat'
                self._structural_conf = 0.0
                self._structural_sight = 0
            self._structural_updated = self.bar_count

        return calibrated

    def decide(self):
        """Make trading decision from all TF states.

        Priority: 1h sets direction, 1m provides wave entry/exit, 1s confirms timing.
        """
        self.bar_count += 1

        # Get state per TF (may not all be present)
        s_1h = self.tf_states.get('1h')
        s_15m = self.tf_states.get('15m')
        s_1m = self.tf_states.get('1m')
        s_15s = self.tf_states.get('15s')
        s_1s = self.tf_states.get('1s')

        # Primary decision TF is 1m (or lowest available)
        primary = s_1m or s_1s or s_1h
        if primary is None or primary.curve is None:
            return TradeAction('WAIT', 'flat', 0, 0, 'no_data')

        # --- IN TRADE: check exits ---
        if self.in_trade:
            return self._check_exit(s_1h, s_15m, s_1m, s_15s, s_1s)

        # --- NOT IN TRADE: check entries ---
        return self._check_entry(s_1h, s_15m, s_1m, s_15s, s_1s)

    def _check_entry(self, s_1h, s_15m, s_1m, s_15s, s_1s):
        """Entry logic with calibrated thresholds.

        1h:  compass heading (bias conf >= 0.50 for 80% calibrated)
        15m: session filter (conf >= 0.63 to disagree with 1h)
        1m:  wave entry (conf >= 0.50 with bias, >= 0.80 against)
        15s: fast confirmation (conf >= 0.43)
        1s:  DROPPED for direction (chop zone covers entire range)
        """
        bias_dir = self._structural_dir
        bias_conf = self._structural_conf

        # No structural direction = no trade
        if bias_dir == 'flat':
            return TradeAction('WAIT', 'flat', 0, 0, 'no_structural_bias')

        # 15m session filter: needs calibrated confidence >= 0.63 to override
        if s_15m and s_15m.curve is not None:
            if s_15m.direction != bias_dir and s_15m.n1_confidence > 0.63:
                return TradeAction('WAIT', bias_dir, 0, 0,
                                   f'15m_against_bias({s_15m.direction} conf={s_15m.n1_confidence:.2f})')

        if s_1m is None or s_1m.curve is None:
            return TradeAction('WAIT', bias_dir, 0, 0, '1m_no_data')

        # 1m regime check
        if s_1m.regime == 'CHOP':
            return TradeAction('WAIT', bias_dir, 0, 0, '1m_chop')

        # 1m direction vs structural bias (calibrated thresholds)
        with_bias = (s_1m.direction == bias_dir)

        if with_bias:
            min_conf = 0.50   # 80% calibrated accuracy
            min_sight = 2
        else:
            min_conf = 0.80   # ~90% calibrated accuracy to go against trend
            min_sight = 4

        if s_1m.n1_confidence < min_conf:
            return TradeAction('WAIT', s_1m.direction, s_1m.n1_confidence,
                               s_1m.sight_distance,
                               f'1m_conf_{s_1m.n1_confidence:.2f}<{min_conf}')

        if s_1m.sight_distance < min_sight:
            return TradeAction('WAIT', s_1m.direction, s_1m.n1_confidence,
                               s_1m.sight_distance,
                               f'1m_sight_{s_1m.sight_distance}<{min_sight}')

        # 15s fast confirmation (calibrated: conf >= 0.43 for 80%)
        if s_15s and s_15s.curve is not None:
            if s_15s.direction != s_1m.direction and s_15s.n1_confidence > 0.43:
                return TradeAction('WAIT', s_1m.direction, 0, 0,
                                   f'15s_not_confirming(conf={s_15s.n1_confidence:.2f})')

        # 1s: NOT used for direction (calibration shows chop zone = entire range)

        # Enter
        trade_dir = s_1m.direction
        confidence = s_1m.n1_confidence
        sight = s_1m.sight_distance

        self.in_trade = True
        self.trade_dir = trade_dir
        self.entry_bar = self.bar_count

        _bias_label = 'WITH' if with_bias else 'AGAINST'
        return TradeAction('ENTER', trade_dir, confidence, sight,
                           f'{_bias_label}_bias bias={bias_dir}({bias_conf:.2f}) '
                           f'sight={sight}')

    def _check_exit(self, s_1h, s_15m, s_1m, s_15s, s_1s):
        """Exit logic with calibrated thresholds.

        1h:  structural flip = immediate exit
        15m: session flip (conf >= 0.63) = exit
        15s: fast move detection (conf >= 0.43) = quick exit
        1m:  primary exit via trajectory shape
        1s:  NOT used for direction exits
        """
        bias_dir = self._structural_dir
        with_bias = (self.trade_dir == bias_dir)

        # Structural bias flipped against our trade — exit immediately
        if bias_dir != 'flat' and bias_dir != self.trade_dir:
            self.in_trade = False
            return TradeAction('EXIT', self.trade_dir, 0, 0,
                               f'structural_flip bias={bias_dir}')

        # 15m session turning against our trade (calibrated: 0.63 for reliable flip)
        if s_15m and s_15m.curve is not None:
            if s_15m.direction != self.trade_dir and s_15m.n1_confidence > 0.63:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_15m.n1_confidence, 0,
                                   f'15m_session_flip({s_15m.direction} conf={s_15m.n1_confidence:.2f})')

        # 15s fast exit (calibrated: 0.43 for 80% real move detection)
        if s_15s and s_15s.curve is not None:
            if s_15s.direction != self.trade_dir and s_15s.n1_confidence > 0.43:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_15s.n1_confidence, 0,
                                   f'15s_fast_exit({s_15s.direction} conf={s_15s.n1_confidence:.2f})')

        # Primary: 1m trajectory
        if s_1m and s_1m.curve is not None:
            # 1. n+1 flipped direction
            if s_1m.direction != self.trade_dir:
                # With bias: allow one bar of dip (oscillation)
                # Against bias: exit immediately
                if not with_bias:
                    self.in_trade = False
                    return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                       f'1m_flipped_against_bias n+1={s_1m.curve[0]:.2f}')
                # With bias: only exit if n+1 AND n+2 both flipped
                elif s_1m.n_horizons >= 2 and s_1m.curve[1] < 0.5 if self.trade_dir == 'long' \
                        else s_1m.curve[1] > 0.5:
                    self.in_trade = False
                    return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                       f'1m_flipped_confirmed n+1={s_1m.curve[0]:.2f}')

            # 2. Regime changed to INFLECTION
            if s_1m.regime == 'INFLECTION':
                if not with_bias:
                    self.in_trade = False
                    return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                       '1m_inflection_against_bias')
                # With bias: inflection might just be oscillation — only exit if strong
                elif s_1m.n1_confidence < 0.3:
                    self.in_trade = False
                    return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                       '1m_inflection')

            # 3. Regime changed to CHOP — signal lost
            if s_1m.regime == 'CHOP':
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, 0, 0, '1m_chop')

            # 4. Near-far disagreement spiking — peak imminent
            disagree = s_1m.near_far_disagreement
            disagree_thresh = 0.20 if with_bias else 0.12  # more patient with bias
            if self.trade_dir == 'long' and disagree > disagree_thresh:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                   f'1m_disagree={disagree:.2f}')
            elif self.trade_dir == 'short' and disagree < -disagree_thresh:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence, 0,
                                   f'1m_disagree={disagree:.2f}')

            # 5. Sight distance contracted
            min_sight = 1 if with_bias else 2  # more patient with bias
            if s_1m.sight_distance <= min_sight:
                self.in_trade = False
                return TradeAction('EXIT', self.trade_dir, s_1m.n1_confidence,
                                   s_1m.sight_distance,
                                   f'1m_sight={s_1m.sight_distance}')

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
