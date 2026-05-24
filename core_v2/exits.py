"""Exit signal evaluators.

Each ExitRule reads (BarState, Position) and returns an exit reason string
or None. Engine evaluates rules in declaration order; first match closes
the position.

V2-native — all reads via state.get('L3_<tf>_z_se_<N>') etc., no V1 indices.

Adaptive thresholds:
    HardStop / TakeProfit / Giveback / TimeStop FIRST consult
    `position.extras['thresholds']` (a dict possibly populated by the engine
    via threshold_optimizer.lookup_thresholds(regime, tier)). If absent or a
    field is missing, fall back to the rule's __init__ default. Rules that
    aren't tied to per-cell thresholds (ZSeReversal, RegimeFlip,
    SwingNoiseSpike) keep their constructor configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict

from training.utils.state import BarState, REGIME_VOCAB
from core_v2.ledger import Position
from training.utils.v2_cols import (z_se_w, swing_noise_w, price_velocity_w,
                                          reversion_prob_w)


# ── Tunables — Phase 2 baseline; recalibrate in Phase 3 ──────────────────

# Hard SL/TP in $ (per contract). MNQ: 1pt = $2; 1tick=$0.50.
HARD_STOP_USD = -25.0       # -$25 per contract
TAKE_PROFIT_USD = 60.0      # +$60 per contract
GIVEBACK_MIN_PEAK = 30.0    # arm giveback only if peak > $30
GIVEBACK_KEEP = 0.5         # keep 50% of peak; trigger if pnl < peak*0.5

# Time stop — 5s bars
MAX_HOLD_BARS = 360         # 30 min  (360 * 5s)

# z_se reversal — fade thesis dies when 1m z_se crosses 0
Z_REVERSAL_THRESHOLD = 0.0   # cross past 0 in opposite direction

# swing_noise spike — chop just started
SWING_NOISE_SPIKE_FACTOR = 2.0  # current > 2x entry value


class ExitRule(ABC):
    name: str = 'BASE'

    @abstractmethod
    def evaluate(self, state: BarState, position: Position) -> Optional[str]:
        """Return exit reason str if rule fires, else None."""


# Risk-management floor: optimizer is allowed to pick `sl_pts=999` on small
# samples (no path went deep red). Always enforce a hard cap regardless.
SL_PTS_FLOOR = 25.0     # ≈ -$50 — backstop for tail events not seen in IS


def _thr(position, key: str, default):
    thr = position.extras.get('thresholds') if position.extras else None
    if thr is None:
        return default
    return thr.get(key, default)


class HardStop(ExitRule):
    """Reads `sl_pts` (POINTS) from position.extras['thresholds']; converts
    to $; clamps to SL_PTS_FLOOR. Falls back to the constructor default $.
    """
    name = 'hard_stop'

    def __init__(self, usd: float = HARD_STOP_USD):
        self.usd = usd

    def evaluate(self, state, position):
        sl_pts = _thr(position, 'sl_pts', None)
        if sl_pts is not None:
            sl_pts = min(float(sl_pts), SL_PTS_FLOOR)
            cap_usd = -abs(sl_pts) * 2.0
        else:
            cap_usd = self.usd
        if position.pnl(state.price) <= cap_usd:
            return self.name
        return None


class TakeProfit(ExitRule):
    """Reads `tp_pts` (POINTS) from position.extras['thresholds']."""
    name = 'take_profit'

    def __init__(self, usd: float = TAKE_PROFIT_USD):
        self.usd = usd

    def evaluate(self, state, position):
        tp_pts = _thr(position, 'tp_pts', None)
        target_usd = (tp_pts * 2.0) if tp_pts is not None else self.usd
        if position.pnl(state.price) >= target_usd:
            return self.name
        return None


class Giveback(ExitRule):
    """Reads `gb_min` ($) and `gb_keep` (ratio) from position.extras['thresholds']."""
    name = 'giveback'

    def __init__(self, min_peak: float = GIVEBACK_MIN_PEAK,
                 keep: float = GIVEBACK_KEEP):
        self.min_peak = min_peak
        self.keep = keep

    def evaluate(self, state, position):
        min_peak = _thr(position, 'gb_min', self.min_peak)
        keep = _thr(position, 'gb_keep', self.keep)
        if position.peak_pnl < min_peak:
            return None
        if position.pnl(state.price) < position.peak_pnl * keep:
            return self.name
        return None


class TimeStop(ExitRule):
    """Reads `time_stop_bars` from position.extras['thresholds']."""
    name = 'time_stop'

    def __init__(self, max_bars: int = MAX_HOLD_BARS):
        self.max_bars = max_bars

    def evaluate(self, state, position):
        max_bars = _thr(position, 'time_stop_bars', self.max_bars)
        if position.bars_held >= int(max_bars):
            return self.name
        return None


class ZSeReversal(ExitRule):
    """1m mean-reversion thesis dies when z_se_w flips sign past 0
    in the direction opposite to the entry direction.

    For longs (we bet bounce up because z was negative), exit when z >= 0.
    For shorts (we bet drop because z was positive), exit when z <= 0.

    Skipped for trades flagged as RIDE (trend-follow thesis) — those entered
    in the SAME direction as the regime trend, so z is already on the
    "wrong side" at entry and this rule would fire on bar 1.

    Detection of RIDE: position.entry_tier in {'NMP_FLIP', 'MA_ALIGN'}
    or position.extras['flipped_from'] is present.
    """
    name = 'z_se_reversal'
    RIDE_TIERS = {'NMP_FLIP', 'MA_ALIGN', 'NMP_RIDE',
                       'RIDE_CALM', 'RIDE_MOMENTUM', 'RIDE_AGAINST'}

    def __init__(self, tf: str = '1m'):
        self.col = z_se_w(tf)

    def evaluate(self, state, position):
        # Skip for ride/trend-follow trades — fade-thesis exit doesn't apply
        if position.entry_tier in self.RIDE_TIERS:
            return None
        if position.extras and position.extras.get('flipped_from'):
            return None
        z = state.get(self.col, 0.0)
        if position.direction == 'long' and z >= 0:
            return self.name
        if position.direction == 'short' and z <= 0:
            return self.name
        return None


class SwingNoiseSpike(ExitRule):
    """Chop just kicked in. Entry was on a smooth read; current bar's
    swing_noise is N-fold higher → exit before chop eats the trade.
    """
    name = 'swing_noise_spike'

    def __init__(self, tf: str = '1m', factor: float = SWING_NOISE_SPIKE_FACTOR):
        self.col = swing_noise_w(tf)
        self.factor = factor

    def evaluate(self, state, position):
        entry_sn = position.extras.get('entry_swing_noise')
        if entry_sn is None or entry_sn <= 0:
            return None
        current = state.get(self.col, 0.0)
        if current >= entry_sn * self.factor:
            return self.name
        return None


class OUReversionDecay(ExitRule):
    """Ornstein-Uhlenbeck reversion-thesis decay exit.

    NMP entry depends on OU first-passage probability (`reversion_prob_w`):
    "given current OU calibration, probability the price returns to band".
    A high entry rprob means the OU decay rate (theta) was strong enough
    to predict reversion within the window.

    During the trade, the OU calibration updates each bar. If the rprob
    DROPS materially from its entry value, the decay rate has weakened —
    the mean-reversion thesis is dying. Exit before it fully fails.

    Skipped for RIDE trades (the thesis isn't reversion).

    rprob_decay_factor: exit when current rprob <= entry_rprob × factor.
        e.g., factor=0.6 with entry rprob=0.85 -> exit when rprob <= 0.51.
    """
    name = 'ou_reversion_decay'
    RIDE_TIERS = {'NMP_FLIP', 'MA_ALIGN', 'NMP_RIDE',
                       'RIDE_CALM', 'RIDE_MOMENTUM', 'RIDE_AGAINST'}

    def __init__(self, tf: str = '1m', decay_factor: float = 0.6,
                 min_entry_rprob: float = 0.5):
        self.col = reversion_prob_w(tf)
        self.decay_factor = decay_factor
        self.min_entry_rprob = min_entry_rprob

    def evaluate(self, state, position):
        if position.entry_tier in self.RIDE_TIERS:
            return None
        if position.extras and position.extras.get('flipped_from'):
            return None
        entry_rprob = (position.extras or {}).get('entry_reversion_prob')
        if entry_rprob is None or entry_rprob < self.min_entry_rprob:
            return None
        current = state.get(self.col, 0.0)
        # Thesis dying: current rprob has decayed materially below entry value
        if current <= entry_rprob * self.decay_factor:
            return self.name
        return None


class TargetMeanReached(ExitRule):
    """Exit when 5s price reaches the target TF's regression mean.

    Reads `target_mean_col` from position.extras (set by FadeAtBand).
    For LONG positions, exits when price >= target mean.
    For SHORT positions, exits when price <= target mean.

    The target_mean is the price level we're snapping back to — the
    5m (or whichever TF) regression mean. When price reaches it,
    the fade thesis has paid out fully.
    """
    name = 'target_mean'

    def evaluate(self, state, position):
        col = (position.extras or {}).get('target_mean_col')
        if not col:
            return None
        target = state.get(col)
        if target is None or target != target:    # NaN
            return None
        if position.direction == 'long' and state.price >= target:
            return self.name
        if position.direction == 'short' and state.price <= target:
            return self.name
        return None


class RegimeFlip(ExitRule):
    """Macro-TF velocity flipped against the position's direction."""
    name = 'regime_flip'

    def __init__(self, tf: str = '4h'):
        self.col = price_velocity_w(tf)

    def evaluate(self, state, position):
        v = state.get(self.col, 0.0)
        if position.direction == 'long' and v < 0:
            return self.name
        if position.direction == 'short' and v > 0:
            return self.name
        return None


class ZSeRetracement(ExitRule):
    """1m z_se retracement exit — DATA-DRIVEN.

    Discovered via peak-signature mining: at MFE, L3_1m_z_se_15 has retreated
    ~25-35% from its entry magnitude toward zero. The "trade is at peak"
    signature. For longs (entered at z<0): exit when z has risen toward 0
    by `pct` of entry magnitude. For shorts (entered at z>0): mirror.

    pct=0.30 default → exit when z has covered 30% of distance from
    entry value to 0.
    """
    name = 'z_se_retracement'
    RIDE_TIERS = {'NMP_FLIP', 'MA_ALIGN', 'NMP_RIDE',
                       'RIDE_CALM', 'RIDE_MOMENTUM', 'RIDE_AGAINST',
                       'NMP_RIDE_RAW'}

    def __init__(self, tf: str = '1m', pct: float = 0.30,
                 min_entry_abs_z: float = 1.0):
        self.col = z_se_w(tf)
        self.pct = pct
        self.min_entry_abs_z = min_entry_abs_z

    def evaluate(self, state, position):
        # Skip RIDE/flipped trades — fade-thesis rule doesn't apply when
        # the position rides with the band-tracking trend.
        if position.entry_tier in self.RIDE_TIERS:
            return None
        if position.extras and position.extras.get('flipped_from'):
            return None
        entry_z = (position.extras or {}).get('entry_z_se')
        if entry_z is None or abs(entry_z) < self.min_entry_abs_z:
            return None
        current = state.get(self.col, 0.0)
        target = entry_z * (1.0 - self.pct)   # closer to 0 by pct
        if entry_z > 0 and current <= target:
            return self.name
        if entry_z < 0 and current >= target:
            return self.name
        return None


class MFEPriceTarget(ExitRule):
    """Per-cell modal MFE price-target exit — DATA-DRIVEN.

    For every (tier × regime × direction) cell, the modal peak PnL ($) is
    known from peak-signature mining across 70k+ trades. When current PnL
    reaches the cell's modal MFE, the trade has hit the EMPIRICAL typical
    peak. Three modes:

    'hard' (default)
        Exit immediately when pnl >= cell target. Caps every trade at
        typical peak; misses the right-tail extension. Best signal-to-noise.

    'arm_giveback'
        Sets position.extras['mfe_armed'] = True when target is hit. A
        downstream giveback rule then watches for retracement past the
        peak. Captures right tail when present, locks in if it retraces.

    Loads `per_cell_targets` from training_iso_v2/output/mfe_targets_per_cell.json
    via the engine's wiring (passed at __init__).
    """
    name = 'mfe_price_target'

    def __init__(self, per_cell_targets: Dict = None,
                 fallback_usd: Optional[float] = None,
                 mode: str = 'hard',
                 min_target_usd: float = 5.0):
        self.targets = per_cell_targets or {}
        self.fallback_usd = fallback_usd
        self.mode = mode
        self.min_target_usd = min_target_usd

    def _cell_key(self, state: BarState, position: Position) -> str:
        ridx = state.regime_idx
        regime = (REGIME_VOCAB[ridx] if 0 <= ridx < len(REGIME_VOCAB)
                       else 'UNKNOWN')
        return f'{position.entry_tier}|{regime}|{position.direction}'

    def evaluate(self, state, position):
        cell_key = self._cell_key(state, position)
        cell = self.targets.get(cell_key)
        if cell is None:
            target = self.fallback_usd
            if target is None:
                return None
        else:
            target = float(cell.get('price_target_usd', 0.0))
        if target < self.min_target_usd:
            return None       # too tight; would fire on entry slippage
        pnl = position.pnl(state.price)
        if pnl < target:
            return None
        if self.mode == 'hard':
            return self.name
        if self.mode == 'arm_giveback':
            position.extras['mfe_armed'] = True
            position.extras['mfe_armed_pnl'] = pnl
            return None
        return None


class MFEArmedGiveback(ExitRule):
    """Companion to MFEPriceTarget(mode='arm_giveback').

    Once mfe_armed=True is set in position.extras, this rule fires when
    pnl retraces past the giveback threshold. The threshold is FLOORED
    at the original arm price (the cell's target) so we never exit below
    the empirical peak target — that would defeat the whole point of the
    price-target signal.

    threshold = max(peak * keep, mfe_armed_pnl)

    With keep=0.7 (default), retracement of 30% from peak triggers exit,
    BUT not below the original target. Tighter than the legacy 50%
    giveback because we want to lock in the right-tail capture rather
    than wait through a deep retracement.
    """
    name = 'mfe_giveback'

    def __init__(self, keep: float = 0.7):
        self.keep = keep

    def evaluate(self, state, position):
        if not (position.extras and position.extras.get('mfe_armed')):
            return None
        pnl = position.pnl(state.price)
        peak = max(position.peak_pnl,
                       position.extras.get('mfe_armed_pnl', 0.0))
        target_floor = float(position.extras.get('mfe_armed_pnl', 0.0))
        threshold = max(peak * self.keep, target_floor)
        if pnl < threshold:
            return self.name
        return None


class Z15sOvershoot(ExitRule):
    """15s z_se sign-flip overshoot exit — DATA-DRIVEN.

    Discovered via peak-signature mining: at MFE, L3_15s_z_se_12 has
    overshot to the OPPOSITE side of zero from its entry side. e.g., for
    a fade-short (entry 1m z>0, fade direction=short), 15s z entry was
    typically slightly positive (0.6); at MFE it's negative (-1.3+). This
    is a fast-timeframe momentum-exhaustion peak signature.

    For trade direction LONG (we want price up), exit when 15s z has gone
    BELOW `-min_overshoot` (overshoot to the SHORT side). For SHORT,
    exit when 15s z is ABOVE `+min_overshoot`.
    """
    name = 'z_15s_overshoot'
    RIDE_TIERS = {'NMP_FLIP', 'MA_ALIGN', 'NMP_RIDE',
                       'RIDE_CALM', 'RIDE_MOMENTUM', 'RIDE_AGAINST',
                       'NMP_RIDE_RAW'}

    def __init__(self, tf: str = '15s', min_overshoot: float = 1.0):
        self.col = z_se_w(tf)
        self.min_overshoot = min_overshoot

    def evaluate(self, state, position):
        if position.entry_tier in self.RIDE_TIERS:
            return None
        if position.extras and position.extras.get('flipped_from'):
            return None
        z = state.get(self.col, 0.0)
        if position.direction == 'long' and z <= -self.min_overshoot:
            return self.name
        if position.direction == 'short' and z >= self.min_overshoot:
            return self.name
        return None


def default_exit_suite(mfe_targets: Optional[Dict] = None,
                              mfe_mode: str = 'arm_giveback') -> List[ExitRule]:
    """RESEARCH exit suite — DATA-DRIVEN, NO dollar caps.

    Updated 2026-05-06 with three layers, all derived from peak-signature
    mining of 70k+ IS trades:

      LAYER 1 (PRICE target, per-cell):
        MFEPriceTarget       — when pnl reaches cell's modal MFE ($), arm a
                                  giveback. Catches the right tail; locks in if
                                  it retraces. Uses per-cell map from
                                  training_iso_v2/output/mfe_targets_per_cell.json
                                  (default mode='arm_giveback').
        MFEArmedGiveback    — fires once armed and pnl retraces 50% off peak.

      LAYER 2 (FEATURE signatures at MFE):
        ZSeRetracement      — 1m z retraced 30% from entry toward 0
        Z15sOvershoot       — 15s z flipped sign past entry direction

      LAYER 3 (legacy thesis fallback):
        OUReversionDecay    — rprob × 0.6 (selection-biased; rarely fires)
        ZSeReversal         — 1m z crosses 0
        SwingNoiseSpike     — chop expansion
        TimeStop            — 60 min cap

    Order matters; first match wins. The price-target exit gates on PRICE
    (the user's "we hit the right price exit"). Feature exits gate on
    pattern. Both fire independently — whichever lands first.

    For LIVE/production deployment use `production_exit_suite()` which
    prepends HardStop/TakeProfit/Giveback to cap dollar exposure.
    """
    return [
        # LAYER 0: TargetMeanReached — fires for FadeAtBand entries when
        # price reaches the target TF's mean. Pure physics exit.
        TargetMeanReached(),
        # LAYER 1: per-cell price target (if mfe_targets provided)
        MFEPriceTarget(per_cell_targets=mfe_targets, mode=mfe_mode),
        MFEArmedGiveback(keep=0.7),    # 30% retracement, floored at target
        # LAYER 2: data-driven feature exits (peak-signature mining)
        ZSeRetracement(tf='1m', pct=0.30, min_entry_abs_z=1.0),
        Z15sOvershoot(tf='15s', min_overshoot=1.0),
        # LAYER 3: legacy thesis exits (fallback)
        OUReversionDecay(tf='1m', decay_factor=0.6),
        ZSeReversal(tf='1m'),
        SwingNoiseSpike(tf='1m'),
        TimeStop(max_bars=720),
    ]


def production_exit_suite() -> List[ExitRule]:
    """LIVE/SIM exit suite — thesis exits + dollar risk caps.

    Use this for deployment, NOT for tier discovery/research. Caps tail
    losses and locks in profit but distorts the underlying physics by
    truncating both ends of the PnL distribution.
    """
    return [
        HardStop(),
        TakeProfit(),
        Giveback(),
        OUReversionDecay(tf='1m', decay_factor=0.6),
        ZSeReversal(tf='1m'),
        SwingNoiseSpike(tf='1m'),
        TimeStop(),
    ]


# ── Bayesian conditional exit (2026-05-10) ──────────────────────────────
#
# Per-tier exit table keyed on (t_since_peak_bucket, capture_bucket).
# Returns P(current peak IS the FINAL peak); fires when P_final >= threshold.
#
# Validated tiers (empirical IS+OOS simulation 2026-05-10):
#   FADE_CALM:    +$30 net (small win, OOS +$71)
#   KILL_SHOT:    -$55 IS / +$12 OOS (neutral; OOS leans positive)
#
# Tiers where oracle hurts — DO NOT ENABLE:
#   NMP_FADE_RAW: -$4,964 net (selection bias; oracle interferes with tier exit)
#   FADE_AT_BAND: -$198 net (entry rule is broken; oracle can't fix)
#
# Behavior: only enabled for tiers in ENABLED_TIERS set. Otherwise no-op.
# When the oracle has no table cell for the queried state, returns None
# (no exit) — defer to other rules.

BAYES_CONDITIONAL_ENABLED_TIERS = {
    'FADE_CALM',
    # 'KILL_SHOT',  # neutral on IS; enable for OOS-leaning deployments
}


class BayesConditionalExit(ExitRule):
    """Per-tier conditional exit oracle.

    At each bar:
      1. Track peak_pnl + ts_at_peak (stored in position.extras).
      2. Compute (t_since_peak_bucket, capture_bucket).
      3. Look up P(current peak IS final) for this tier+state.
      4. Fire if P_final >= threshold.

    No-op for tiers not in ENABLED_TIERS set (avoid known negative impact).
    """
    name = 'bayes_conditional'

    # Buckets MUST match training_iso_v2/filters/bayes_conditional_exit.py
    T_BINS = (5, 15, 30, 60, 120, 300, 900)

    def __init__(self, enabled_tiers: set = None, threshold_fade: float = 0.85,
                  threshold_ride: float = 0.70):
        self.enabled_tiers = enabled_tiers or BAYES_CONDITIONAL_ENABLED_TIERS
        self.threshold_fade = threshold_fade
        self.threshold_ride = threshold_ride
        # Lazy-load the table on first use (avoid import-time CSV read)
        self._table = None

    def _load_table(self):
        if self._table is not None: return
        from training.filters.bayes_conditional_exit import (
            _load_conditional_table,
        )
        self._table = _load_conditional_table()

    @classmethod
    def _bucket_time(cls, s: float) -> int:
        for i, b in enumerate(cls.T_BINS):
            if s <= b: return i
        return len(cls.T_BINS)

    @staticmethod
    def _bucket_capture(r: float) -> int:
        if r <= 0: return 0
        if r < 0.3: return 1
        if r < 0.6: return 2
        if r < 0.9: return 3
        return 4

    def _threshold_for(self, tier: str) -> float:
        if tier.startswith('RIDE_') or tier == 'NMP_RIDE_RAW':
            return self.threshold_ride
        return self.threshold_fade

    def evaluate(self, state, position):
        tier = position.entry_tier
        if tier not in self.enabled_tiers:
            return None
        self._load_table()
        cell_table = self._table.get(tier, {})
        if not cell_table:
            return None

        cur_pnl = position.pnl(state.price)
        peak_pnl = position.peak_pnl

        # Track ts at peak in extras
        if cur_pnl > position.extras.get('_bayes_peak_pnl', -float('inf')):
            position.extras['_bayes_peak_pnl'] = cur_pnl
            position.extras['_bayes_peak_ts'] = state.timestamp

        if peak_pnl <= 0:
            return None

        peak_ts = position.extras.get('_bayes_peak_ts', position.entry_ts)
        t_since_peak = max(0, state.timestamp - peak_ts)
        capture = cur_pnl / peak_pnl if peak_pnl > 0 else 0
        t_b = self._bucket_time(t_since_peak)
        cap_b = self._bucket_capture(capture)
        p_final = cell_table.get((t_b, cap_b))
        if p_final is None:
            return None
        if p_final >= self._threshold_for(tier):
            return f'bayes_p{int(100*p_final):d}'
        return None
