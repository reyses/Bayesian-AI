"""
Timeframe Belief Network
========================
N workers, each monitoring a different timeframe level simultaneously.

ARCHITECTURE
------------
Each worker has TWO tasks:
  Task 1 (Aggregation):  accumulate 15s bars -> TF OHLCV bar -> QuantumState
  Task 2 (Analysis):     state -> cluster match -> regression -> P(LONG), pred_MFE

Update cadence (15s bar count per worker wakeup):
  1h   = every 240 bars  (light:   ~7 updates/day)
  30m  = every 120 bars  (light:  ~14 updates/day)
  15m  = every  60 bars  (light:  ~26 updates/day)
  5m   = every  20 bars  (medium: ~78 updates/day)
  3m   = every  12 bars  (medium:~133 updates/day)
  1m   = every   4 bars  (medium:~390 updates/day)
  30s  = every   2 bars  (heavy: ~780 updates/day)
  15s  = every   1 bar   (heavy:5300 updates/day -- uses top-K parallel matching)

PATH CONVICTION (psychohistory principle)
-----------------------------------------
The PATH to the leaf is the prediction, not the leaf alone.
If 1h -> 30m -> 15m -> 5m -> 1m -> 15s all agree on LONG,
the path conviction is very high -- the fractal tree has converged.

path_conviction = weighted_geometric_mean(P(correct_direction) across all active TF levels)
                  higher-TF beliefs carry more weight (they summarize more history)

DECISION TF
-----------
The default decision resolution is 5m: we decide "what is the NEXT 5m bar going to do?"
The 15s execution workers provide the precise entry trigger.
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from training.fractal_clustering import DNA_LIVE_DIMS

logger = logging.getLogger(__name__)

# Parent DNA Matching: 15m = anchor worker (parent signature matching)
ANCHOR_TF = 900


@dataclass
class WorkerBelief:
    """Current belief from one TF worker."""
    tf_seconds:    int
    dir_prob:      float    # P(LONG) from logistic regression [0..1]
    pred_mfe:      float    # OLS predicted MFE in price points
    template_id:   int
    tf_bar_idx:    int      # which TF bar produced this belief
    conviction:    float    # |dir_prob - 0.5| * 2  -> how sure this worker is [0..1]
    wave_maturity: float = 0.0  # P(wave near completion) [0..1]
    # Composite: 0.4*pattern_maturity + 0.3*min(1,|z|/3) + 0.3*tunnel_probability
    # High value = wave is well-developed/near exhaustion = higher entry risk
    z_score:       float = 0.0  # raw z_score from quantum state (for decay tracking)
    momentum:      float = 0.0  # raw momentum_strength from quantum state
    dna_agreement: float = 0.5  # DNA match quality [0=outside cell, 1=at centroid, 0.5=neutral]


@dataclass
class BeliefState:
    """Aggregated belief across all active TF workers."""
    direction:              str           # 'long' | 'short'
    conviction:             float         # weighted geometric mean across TF levels [0..1]
    predicted_mfe:          float         # MFE prediction in ticks (from decision-level worker)
    active_levels:          int           # how many TF levels contributed
    wave_maturity:          float = 0.0   # weighted avg wave_maturity across ALL active workers [0..1]
    decision_wave_maturity: float = 0.0   # wave_maturity at the DECISION TF only (e.g. 5m)
    # KEY DISTINCTION: a mature 30s wave may just be a forming 5m wave.
    # Use decision_wave_maturity (5m) to assess if the TRADEABLE wave is exhausted.
    # Use wave_maturity (weighted avg) for reference only.
    tf_beliefs:     Dict[int, WorkerBelief] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.conviction >= TimeframeBeliefNetwork.MIN_CONVICTION


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _logistic_prob(feat_s: np.ndarray, lib: dict) -> float:
    """
    P(LONG) from the cluster's logistic regression model.
    Fallback: use long_bias / short_bias aggregate if model not fitted.

    Prior correction: the logistic regression was trained on class-imbalanced data
    (e.g. 75% SHORT oracle markers). The intercept absorbed that base rate, so the
    model predicts "what fraction of TRAINING examples were LONG" rather than
    "what direction should THIS pattern trade." We subtract the training log-odds
    to recenter: P(LONG)=0.5 at the feature mean, letting features alone decide.
    """
    coeff = lib.get('dir_coeff')
    if coeff is not None:
        logit = float(np.dot(feat_s, coeff) + lib.get('dir_intercept', 0.0))
        # Prior correction: remove training class imbalance from intercept
        _lb = lib.get('long_bias', 0.5)
        _sb = lib.get('short_bias', 0.5)
        if _lb > 0.01 and _sb > 0.01:
            logit -= math.log(_lb / _sb)
        return _sigmoid(logit)
    # Fallback: convert bias fractions to a probability
    long_b  = lib.get('long_bias',  0.0)
    short_b = lib.get('short_bias', 0.0)
    total = long_b + short_b
    if total > 0.05:
        return long_b / total
    return 0.5  # truly no information


def _ols_mfe(feat_s: np.ndarray, lib: dict) -> float:
    """
    Predicted MFE in price points from the cluster's OLS model.
    Fallback: mean_mfe_ticks * tick_size.
    """
    coeff = lib.get('mfe_coeff')
    if coeff is not None:
        return float(np.dot(feat_s, coeff) + lib.get('mfe_intercept', 0.0))
    return lib.get('mean_mfe_ticks', 0.0) * 0.25  # ticks -> price points


class TimeframeWorker:
    """
    Monitors one timeframe level.

    Task 1 (Aggregation, per-day):
        Receives pre-computed ThreeBodyQuantumState list for its TF (light).
        State list is built by TimeframeBeliefNetwork.prepare_day() once per day.

    Task 2 (Analysis, event-driven):
        When its TF bar changes (every bars_per_update 15s bars):
        - Map current 15s bar index -> TF bar index -> state
        - Extract 14D features
        - Match to nearest cluster (leaf: top-K; others: top-1)
        - Run logistic regression -> P(LONG)
        - Run OLS regression     -> predicted_MFE
        - Update current_belief (sticky until next TF bar)

    At 15s (leaf), uses top-K matching = K independent parallel analysis
    threads logically -- averages their outputs for robustness.
    """

    LEAF_TOP_K = 3  # analysis threads at the leaf (15s or 1s)

    def __init__(self, tf_seconds: int, is_leaf: bool = False, base_resolution_seconds: int = 15):
        self.tf_seconds              = tf_seconds
        self.base_resolution_seconds = base_resolution_seconds
        self.bars_per_update         = max(1, tf_seconds // base_resolution_seconds)
        self.is_leaf                 = is_leaf

        # Trade context -- set by network when position is open
        self._trade_side: str = None            # 'long' or 'short'
        self._trade_profit_ticks: float = 0.0   # current P&L in ticks

        # Filled by prepare()
        self._states: list = []

        # Current belief -- sticky until TF bar changes
        self.current_belief: Optional[WorkerBelief] = None
        self._last_tf_bar_idx: int = -1
        self._last_state = None                 # last analyzed quantum state

        # EOD adaptive learning — persists across days (EMA)
        self._direction_accuracy: float = 0.5   # how often this worker's dir was right
        self._dmi_reliability: float = 0.5      # how often DMI agreement predicted wins
        self._conviction_scale: float = 1.0     # conviction multiplier from learned accuracy

        # Worker augmented playbook — local pattern journal from observed outcomes
        # Key: (lagrange_zone, z_bin, dmi_sign)  Value: {lw, ll, sw, sl}
        self._playbook: dict = {}

    def prepare(self, states: list):
        """Supply Task-1 result: pre-computed states for the day."""
        self._states          = states
        self.current_belief   = None
        self._last_tf_bar_idx = -1

    def review_day(self, day_trades: list, alpha: float = 0.3):
        """
        End-of-day learning: review trades, update accuracy stats.

        Each trade dict must have:
          - 'side': 'long' or 'short'
          - 'pnl': float (positive = win)
          - 'worker_snapshots': dict of {tf_label: {'d': dir_prob, 'dmi_agrees': bool}}

        Updates are EMA: (1-alpha) * old + alpha * today.
        """
        tf_label = self._tf_label()
        dir_correct = 0
        dir_total = 0
        dmi_correct = 0
        dmi_total = 0

        for t in day_trades:
            snap = t.get('worker_snapshots', {}).get(tf_label)
            if snap is None:
                continue

            side = t['side']
            won = t['pnl'] > 0
            d = snap.get('d', 0.5)

            # Direction accuracy: did this worker's dir_prob agree with the winning side?
            worker_said_long = d > 0.5
            trade_was_long = side == 'long'
            worker_agreed = (worker_said_long == trade_was_long)

            if (worker_agreed and won) or (not worker_agreed and not won):
                dir_correct += 1
            dir_total += 1

            # DMI reliability: when DMI agreed with trade, did it win?
            _dmi = snap.get('dmi', 0.0)
            if _dmi != 0.0:
                dmi_agrees = ((side == 'long' and _dmi > 0) or
                              (side == 'short' and _dmi < 0))
                if (dmi_agrees and won) or (not dmi_agrees and not won):
                    dmi_correct += 1
                dmi_total += 1

        if dir_total >= 3:  # need minimum sample
            today_dir = dir_correct / dir_total
            self._direction_accuracy = (1 - alpha) * self._direction_accuracy + alpha * today_dir
            # Map accuracy to conviction scale: 0.5 accuracy → 1.0x, 0.7 → 1.24x, 0.3 → 0.76x
            self._conviction_scale = 0.7 + 0.6 * self._direction_accuracy

        if dmi_total >= 3:
            today_dmi = dmi_correct / dmi_total
            self._dmi_reliability = (1 - alpha) * self._dmi_reliability + alpha * today_dmi

        # ── Playbook update: record outcome for this worker's state at entry ──
        for t in day_trades:
            snap = t.get('worker_snapshots', {}).get(tf_label)
            if snap is None:
                continue
            _lz = snap.get('lz', 'UNKNOWN')
            _z = snap.get('z', 0.0)
            _dmi_raw = snap.get('dmi', 0.0)
            key = self._playbook_key(_z, _lz, _dmi_raw)
            if key not in self._playbook:
                self._playbook[key] = {'lw': 0, 'll': 0, 'sw': 0, 'sl': 0,
                                       'early': 0, 'late': 0, 'eff_sum': 0.0, 'eff_n': 0}
            bucket = self._playbook[key]
            _side = t['side']
            _won = t['pnl'] > 0
            if _side == 'long':
                bucket['lw' if _won else 'll'] += 1
            else:
                bucket['sw' if _won else 'sl'] += 1
            # Regret data (bridged from wave_rider completed_reviews)
            _rt = t.get('regret_type', '')
            if _rt == 'closed_too_early':
                bucket['early'] += 1
            elif _rt == 'closed_too_late':
                bucket['late'] += 1
            _eff = t.get('exit_efficiency')
            if _eff is not None:
                bucket['eff_sum'] += _eff
                bucket['eff_n'] += 1

    def _tf_label(self) -> str:
        """Map tf_seconds to the label used in worker snapshots."""
        _MAP = {3600: '1h', 1800: '30m', 900: '15m', 300: '5m', 180: '3m',
                60: '1m', 30: '30s', 15: '15s', 5: '5s', 1: '1s'}
        return _MAP.get(self.tf_seconds, f'{self.tf_seconds}s')

    @staticmethod
    def _playbook_key(z_score: float, lagrange_zone: str, dmi_diff: float) -> tuple:
        """Discretized state signature for playbook lookup."""
        z_bin = round(z_score)
        dmi_sign = 'pos' if dmi_diff > 5.0 else ('neg' if dmi_diff < -5.0 else 'flat')
        return (lagrange_zone, z_bin, dmi_sign)

    def tick(self, bar_i: int, pattern_library: dict, scaler,
             valid_tids: list, centroids_scaled: np.ndarray,
             scaler_mean: np.ndarray = None, scaler_scale: np.ndarray = None) -> bool:
        """
        Called every 15s bar. Returns True if belief was updated.

        Task 1 check: has my TF bar changed?
        Task 2: if yes, re-run cluster match + regression.
        """
        if self.tf_seconds < self.base_resolution_seconds:
            # Sub-resolution worker (e.g. 5s worker in a 15s loop):
            # bar_i counts base-resolution bars; pick the LAST sub-bar within
            # each base period so we always use the most current fine-grained state.
            #   5s in 15s loop: bar_i=0 -> idx=2, bar_i=1 -> idx=5, ...
            #   1s in 15s loop: bar_i=0 -> idx=14, bar_i=1 -> idx=29, ...
            _ratio     = self.base_resolution_seconds // self.tf_seconds
            tf_bar_idx = bar_i * _ratio + (_ratio - 1)
        else:
            tf_bar_idx = bar_i // self.bars_per_update

        if tf_bar_idx == self._last_tf_bar_idx:
            return False   # TF bar unchanged -- belief still valid
        if not self._states or tf_bar_idx >= len(self._states):
            return False   # No state yet (warmup period)

        state_raw = self._states[tf_bar_idx]
        # batch_compute_states() wraps each result as {'bar_idx': i, 'state': ThreeBodyQuantumState, ...}
        # Unwrap so state_to_features() receives the actual physics object, not the dict wrapper.
        state = state_raw['state'] if isinstance(state_raw, dict) and 'state' in state_raw else state_raw
        self._last_tf_bar_idx = tf_bar_idx

        return self._analyze(state, tf_bar_idx, pattern_library, scaler,
                             valid_tids, centroids_scaled, scaler_mean, scaler_scale)

    def tick_at(self, tf_bar_idx: int, pattern_library: dict, scaler,
                valid_tids: list, centroids_scaled: np.ndarray,
                scaler_mean: np.ndarray = None, scaler_scale: np.ndarray = None) -> bool:
        """
        Tick at a specific native-resolution bar index (bypasses base-resolution mapping).
        Used by the 1s inner loop to tick sub-resolution workers at their natural cadence.
        """
        if tf_bar_idx == self._last_tf_bar_idx:
            return False
        if not self._states or tf_bar_idx >= len(self._states):
            return False
        state_raw = self._states[tf_bar_idx]
        state = state_raw['state'] if isinstance(state_raw, dict) and 'state' in state_raw else state_raw
        self._last_tf_bar_idx = tf_bar_idx
        return self._analyze(state, tf_bar_idx, pattern_library, scaler,
                             valid_tids, centroids_scaled, scaler_mean, scaler_scale)

    def _analyze(self, state, tf_bar_idx: int, pattern_library: dict, scaler,
                 valid_tids: list, centroids_scaled: np.ndarray,
                 scaler_mean: np.ndarray = None, scaler_scale: np.ndarray = None) -> bool:
        """Task 2: feature extraction, cluster matching, physics blend, belief update."""
        self._regret_conv_discount = 1.0  # reset each bar; playbook may lower it
        self._last_state = state
        feat = TimeframeBeliefNetwork.state_to_features(state, self.tf_seconds)

        # Optimization: Use direct NumPy vectorized scaling if available (40x faster than scaler.transform)
        if scaler_mean is not None:
             feat_s = (np.array(feat) - scaler_mean) / scaler_scale
        else:
             feat_s = scaler.transform([feat])[0]
        _dna_agreement = 0.5  # Default: neutral (no DNA data)
        _network = getattr(self, '_network', None)

        # --- PATH 1: DNA ANCHOR MATCHING (15m worker) ---
        if (self.tf_seconds == ANCHOR_TF and _network is not None
                and _network.dna_index.get(ANCHOR_TF)):
            dna_entries = _network.dna_index[ANCHOR_TF]
            _mask = DNA_LIVE_DIMS
            feat_live = feat_s[_mask]

            matched_tid = None
            best_dist = float('inf')
            for _tid, _cen16, _bmin, _bmax in dna_entries:
                if np.all(feat_live >= _bmin) and np.all(feat_live <= _bmax):
                    _d = np.linalg.norm(feat_live - _cen16[_mask])
                    if _d < best_dist:
                        best_dist = _d
                        matched_tid = _tid

            _network._active_template_id = matched_tid

            if matched_tid is None:
                self.current_belief = None
                return True  # Updated to None (no parent match)

            primary_tid = matched_tid
            lib = pattern_library[primary_tid]
            dir_prob = _logistic_prob(feat_s, lib)
            pred_mfe = _ols_mfe(feat_s, lib)
            _any_fitted = lib.get('dir_coeff') is not None
            _dna_agreement = 1.0  # Anchor matched = full agreement

        # --- PATH 2: DNA VERIFICATION (non-anchor workers) ---
        elif (_network is not None and _network._active_template_id is not None
              and _network.dna_index):
            active_tid = _network._active_template_id
            dna_entries = _network.dna_index.get(self.tf_seconds, [])
            dna_entry = None
            for _e in dna_entries:
                if _e[0] == active_tid:
                    dna_entry = _e
                    break

            if dna_entry is not None:
                _, _cen16, _bmin, _bmax = dna_entry
                _mask = DNA_LIVE_DIMS
                feat_live = feat_s[_mask]
                if np.all(feat_live >= _bmin) and np.all(feat_live <= _bmax):
                    cell_range = np.where((_bmax - _bmin) < 1e-9, 1.0, _bmax - _bmin)
                    normalized = (feat_live - _bmin) / cell_range
                    center_dist = np.linalg.norm(normalized - 0.5)
                    max_dist = np.sqrt(len(_mask)) * 0.5
                    _dna_agreement = float(np.clip(1.0 - center_dist / max_dist, 0.0, 1.0))
                else:
                    _dna_agreement = 0.0
            else:
                _dna_agreement = 0.5  # No DNA for this TF → neutral

            primary_tid = active_tid
            lib = pattern_library.get(primary_tid, {})
            if lib.get('dir_coeff') is not None:
                dir_prob = _logistic_prob(feat_s, lib)
                pred_mfe = _ols_mfe(feat_s, lib)
                _any_fitted = True
            elif len(centroids_scaled) > 0:
                dists = np.linalg.norm(centroids_scaled - feat_s, axis=1)
                best_idx = int(np.argmin(dists))
                primary_tid = valid_tids[best_idx]
                lib = pattern_library[primary_tid]
                dir_prob = _logistic_prob(feat_s, lib)
                pred_mfe = _ols_mfe(feat_s, lib)
                _any_fitted = lib.get('dir_coeff') is not None
            else:
                dir_prob = 0.5
                pred_mfe = 0.0
                _any_fitted = False

        # --- PATH 3: Legacy fallback (no DNA data or no active template) ---
        elif len(centroids_scaled) > 0:
            dists = np.linalg.norm(centroids_scaled - feat_s, axis=1)
            if self.is_leaf:
                top_k     = min(self.LEAF_TOP_K, len(dists))
                top_k_idx = np.argpartition(dists, top_k - 1)[:top_k]
                dir_probs  = []
                pred_mfes  = []
                for idx in top_k_idx:
                    lib = pattern_library[valid_tids[idx]]
                    dir_probs.append(_logistic_prob(feat_s, lib))
                    pred_mfes.append(_ols_mfe(feat_s, lib))
                dir_prob = float(np.mean(dir_probs))
                pred_mfe = float(np.mean(pred_mfes))
                primary_tid = valid_tids[np.argmin(dists)]
                _any_fitted = any(
                    pattern_library[valid_tids[i]].get('dir_coeff') is not None
                    for i in top_k_idx
                )
            else:
                best_idx    = int(np.argmin(dists))
                primary_tid = valid_tids[best_idx]
                lib         = pattern_library[primary_tid]
                dir_prob    = _logistic_prob(feat_s, lib)
                pred_mfe    = _ols_mfe(feat_s, lib)
                _any_fitted = lib.get('dir_coeff') is not None
        else:
            # No centroids and no DNA → physics only
            primary_tid = valid_tids[0] if valid_tids else 0
            dir_prob = 0.5
            pred_mfe = 0.0
            _any_fitted = False

        # ── Physics blend: Roche limit oscillation gives direction from z_score ──
        # The market oscillates between standard error bands at every timeframe.
        # z_score < 0  (below mean) → expect reversion UP  → P(LONG) > 0.5
        # z_score > 0  (above mean) → expect reversion DOWN → P(LONG) < 0.5
        #
        # Sensitivity scales with log(bars_aggregated) — higher TF workers have
        # more samples in their z_score so the signal is statistically stronger.
        #   1h (240 bars): sensitivity ≈ 1.00   → at z=±2 → dir_prob ≈ 0.88
        #   5m  (20 bars): sensitivity ≈ 0.67   → at z=±2 → dir_prob ≈ 0.76
        #   15s  (1 bar):  sensitivity ≈ 0.50   → at z=±2 → dir_prob ≈ 0.73
        _n_bars = max(1, self.bars_per_update)
        _phys_sensitivity = 0.5 + 0.5 * (np.log(_n_bars) / np.log(240))  # [0.5, 1.0]
        _z_raw    = float(getattr(state, 'z_score', 0.0))
        _phys_dir = _sigmoid(-_z_raw * _phys_sensitivity)

        # Blend with ML signal only if a fitted logistic regression exists.
        # Unfitted templates fall back to long_bias ≈ 0.59 (NQ bullish noise).
        # In that case, use pure physics to avoid degenerate uniform dir_prob values.
        # Feature 0 = abs(z) — loses z_score sign. Physics adds it via sigmoid(-z*sens).
        # Logistic coefficients are biased by 75% SHORT training data (prior correction
        # fixes intercept but not coefficients). At 50/50 the physics z-sign can flip
        # direction to LONG when pattern is below the mean (z < -1), while the logistic
        # still has veto power when features strongly favor SHORT.
        if _any_fitted:
            osc_coh = feat[15]
            ml_weight = 0.5 - 0.15 * osc_coh
            phys_weight = 0.5 + 0.15 * osc_coh
            dir_prob = ml_weight * dir_prob + phys_weight * _phys_dir
        else:
            dir_prob = _phys_dir

        # ── Playbook blend: local WR from observed outcomes ──────────────
        if self._playbook:
            _pb_z = float(getattr(state, 'z_score', 0.0))
            _pb_lz = getattr(state, 'lagrange_zone', 'UNKNOWN')
            _pb_dmi = getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)
            _pb_key = self._playbook_key(_pb_z, _pb_lz, _pb_dmi)
            _pb = self._playbook.get(_pb_key)
            if _pb is not None:
                _pb_total = _pb['lw'] + _pb['ll'] + _pb['sw'] + _pb['sl']
                if _pb_total >= 5:
                    _long_n = _pb['lw'] + _pb['ll']
                    _short_n = _pb['sw'] + _pb['sl']
                    if _long_n > 0 and _short_n > 0:
                        _local_long_wr = _pb['lw'] / _long_n
                        _local_short_wr = _pb['sw'] / _short_n
                        _local_dir = _local_long_wr / (_local_long_wr + _local_short_wr + 1e-9)
                    elif _long_n > 0:
                        _local_dir = _pb['lw'] / _long_n
                    else:
                        _local_dir = 1.0 - (_pb['sw'] / _short_n)
                    dir_prob = 0.7 * dir_prob + 0.3 * _local_dir

                # ── Regret-informed adjustments ──
                _pb_eff_n = _pb.get('eff_n', 0)
                if _pb_eff_n >= 5:
                    _avg_eff = _pb['eff_sum'] / _pb_eff_n
                    _early_pct = _pb.get('early', 0) / _pb_eff_n
                    # Poor exit efficiency → discount conviction downstream
                    if _avg_eff < 0.40:
                        self._regret_conv_discount = 0.7
                    # High early-exit rate → boost predicted MFE (hold longer)
                    if _early_pct > 0.50:
                        pred_mfe *= 1.3

        # Wave maturity: estimate of how "mature" (near completion) the current wave is.
        # High value = wave is well-developed / near exhaustion = higher entry risk.
        # Composite of the three strongest exhaustion signals from the quantum state:
        #   pattern_maturity  : engine's L7 development measure (0-1)
        #   |z_score| / 3.0   : approach to Roche limit (3 sigma = fully mature)
        #   tunnel_probability: P(revert to center) = how close to reversal
        _pm  = getattr(state, 'pattern_maturity',   0.0)
        _tp  = getattr(state, 'tunnel_probability', 0.0)
        _z   = abs(getattr(state, 'z_score',        0.0))
        wave_maturity = float(np.clip(
            0.4 * _pm + 0.3 * min(1.0, _z / 3.0) + 0.3 * _tp,
            0.0, 1.0
        ))

        conviction = abs(dir_prob - 0.5) * 2.0

        # ── DNA agreement modulation ──
        # DNA score scales conviction: 0.0 (outside cell) → 0.5x, 1.0 (at centroid) → 1.0x
        conviction *= (0.5 + 0.5 * _dna_agreement)

        # ── Hurst-based conviction scaling ──
        hurst = feat[8]
        hurst_scale = 0.7 + (hurst - 0.3) * (0.6 / 0.4)
        hurst_scale = max(0.7, min(1.3, hurst_scale))
        conviction *= hurst_scale

        # ── Price-aware conviction modulation (2 layers) ──
        if self._trade_side is not None:
            _agrees = ((self._trade_side == 'long' and dir_prob > 0.5) or
                       (self._trade_side == 'short' and dir_prob < 0.5))
            _winning = self._trade_profit_ticks > 0

            # Layer 1: Direction + P&L agreement
            if _agrees and _winning:
                conviction = min(1.0, conviction * 1.3)
            elif not _agrees and not _winning:
                conviction = min(1.0, conviction * 1.2)
            elif not _agrees and _winning:
                conviction *= 0.7

            # Layer 2: DMI/ADX trend signal, scaled by TF sample size
            _dmi_diff = getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)
            _dmi_agrees = ((self._trade_side == 'long' and _dmi_diff > 0) or
                           (self._trade_side == 'short' and _dmi_diff < 0))
            _adx = getattr(state, 'adx_strength', 0.0)
            _tf_reliability = min(1.0, math.log(max(2, self.bars_per_update)) / math.log(240))

            if not _dmi_agrees and _adx > 25:
                conviction *= (1.0 - 0.3 * _tf_reliability)
            elif _dmi_agrees and _adx > 25:
                conviction = min(1.0, conviction * (1.0 + 0.15 * _tf_reliability))

        # Layer 3: EOD-learned conviction scale (adapts across days)
        conviction = min(1.0, conviction * self._conviction_scale)

        # Layer 4: Regret-informed discount (poor exit efficiency zones)
        conviction *= self._regret_conv_discount

        self.current_belief = WorkerBelief(
            tf_seconds    = self.tf_seconds,
            dir_prob      = dir_prob,
            pred_mfe      = pred_mfe,
            template_id   = primary_tid,
            tf_bar_idx    = tf_bar_idx,
            conviction    = conviction,
            wave_maturity = wave_maturity,
            z_score       = getattr(state, 'z_score', 0.0),
            momentum      = getattr(state, 'momentum_strength', 0.0),
            dna_agreement = _dna_agreement,
        )
        return True


class TimeframeBeliefNetwork:
    """
    8 workers monitoring different timeframe levels simultaneously.

    The PATH through the fractal hierarchy is the prediction:
    a leaf signal that emerges from a consistent top-down conviction
    is much more reliable than an isolated 15s bar pattern match.

    Path conviction: weighted geometric mean of P(direction) across
    all active TF levels.  Higher-TF workers carry more weight
    (they summarise days/hours of market structure).
    """

    TIMEFRAMES_SECONDS = [3600, 1800, 900, 300, 180, 60, 30, 15, 5, 1]
    TF_WEIGHTS         = [4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.1]
    MIN_CONVICTION     = 0.48   # skip trade if path conviction below this (physics at z=0 gives 0.50)
    MIN_ACTIVE_LEVELS  = 3      # need >=3 active TF levels for a signal
    DEFAULT_DECISION_TF = 300   # 5m: default scale at which to read predicted_mfe

    # Dynamic Exit Thresholds
    # belief_flip disabled (threshold > 1.0 = unreachable):
    # Forward pass showed 149 belief_flip exits at avg -$72.96 = -$10,872 total.
    # OOS data confirmed winners flip workers MORE than losers across all TFs,
    # meaning urgent_exit was cutting winners not protecting against losers.
    # Trail stop handles exits naturally; re-enable only with directional evidence.
    # Physics Decay Exit — bottom-up cascade parameters
    # Workers track z_score drift from entry; if reality diverges from
    # expected mean-reversion trajectory, the cascade score rises.
    # tau narrows from alpha_max -> alpha_min over the pattern horizon T_k bars.
    # Fast TFs (1s, 5s) detect reversals first -> weighted more for exit.
    DECAY_ALPHA_MAX  = 3.0    # initial tolerance band (wide at trade start)
    DECAY_ALPHA_MIN  = 1.0    # final tolerance band (narrow at pattern horizon)
    DECAY_THETA_EXIT = 1.5    # cascade score threshold -> trigger exit

    URGENT_EXIT_CONVICTION_THRESHOLD = 1.01   # effectively disabled
    # Pre-fix: tighten fired on (not is_confident) OR wave_mature > 0.65 → every bar
    # during a normal move triggered tightening, collapsing the trail to 2 ticks.
    # Now: only tighten on extreme maturity (0.85) — wave is clearly exhausting.
    # Low conviction during a trade is normal and does NOT predict reversal.
    TIGHTEN_TRAIL_WAVE_MATURITY_THRESHOLD = 0.85
    WIDEN_TRAIL_WAVE_MATURITY_THRESHOLD = 0.30

    _TF_LABELS = {3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
                  180:'3m',  60:'1m',   30:'30s',   15:'15s',
                  5:'5s',    1:'1s'}

    def __init__(self, pattern_library: dict, scaler, engine,
                 valid_tids: list, centroids_scaled: np.ndarray,
                 decision_tf: int = DEFAULT_DECISION_TF,
                 base_resolution_seconds: int = 15,
                 dna_index: Dict[int, list] = None):
        self.pattern_library   = pattern_library
        # Pre-convert regression coefficients from lists to numpy arrays
        # Eliminates per-tick np.array() allocation in _logistic_prob/_ols_mfe
        for _tid, _lib in self.pattern_library.items():
            if isinstance(_lib.get('dir_coeff'), list):
                _lib['dir_coeff'] = np.array(_lib['dir_coeff'])
            if isinstance(_lib.get('mfe_coeff'), list):
                _lib['mfe_coeff'] = np.array(_lib['mfe_coeff'])
        self.scaler            = scaler
        self.engine            = engine
        self.valid_tids        = valid_tids
        self.centroids_scaled  = centroids_scaled
        self.decision_tf       = decision_tf
        self.base_resolution_seconds = base_resolution_seconds

        # Optimization: Pre-extract scaler params for fast vectorization
        # Avoids sklearn validation overhead in tight loops
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            self.scaler_mean = self.scaler.mean_.astype(np.float64)
            self.scaler_scale = self.scaler.scale_.astype(np.float64)
        else:
            self.scaler_mean = None
            self.scaler_scale = None

        # active_timeframes: TFs that can be computed by resampling from df_micro.
        # Sub-resolution TFs (< base_resolution_seconds) need external data (df_5s/df_1s)
        # but workers are always created for all TFs -- they just stay silent when no data.
        self.active_timeframes = [tf for tf in self.TIMEFRAMES_SECONDS if tf >= base_resolution_seconds]

        all_weights = dict(zip(self.TIMEFRAMES_SECONDS, self.TF_WEIGHTS))
        self._weight_map = dict(all_weights)   # weights for ALL TFs, not just active

        leaf_tf = base_resolution_seconds      # the base TF is the leaf

        self.workers: Dict[int, TimeframeWorker] = {
            tf: TimeframeWorker(tf, is_leaf=(tf == leaf_tf), base_resolution_seconds=base_resolution_seconds)
            for tf in self.TIMEFRAMES_SECONDS  # ALL TFs, including sub-resolution
        }

        # Parent DNA Matching: cross-worker coordination
        self.dna_index = dna_index or {}
        self._active_template_id = None  # Set by 15m anchor worker
        # Give each worker a reference to the network for DNA coordination
        for worker in self.workers.values():
            worker._network = self

        # Active-trade time-scale state (set at entry, cleared at exit)
        self._trade_avg_mfe_bar = 0.0
        self._trade_p75_mfe_bar = 0.0
        self._trade_bars_held   = 0
        self._trade_expected_mfe_ticks = 0.0

        # Physics decay tracking state (populated while a trade is open)
        self._decay_trade_side: Optional[str] = None
        self._decay_entry_bar: int = 0
        self._decay_pattern_horizon: int = 1
        self._decay_entry_physics: Dict[int, dict] = {}  # {tf_secs: {'z','m','d'}}
        self._current_bar: int = 0

    # ------------------------------------------------------------------
    # TRADE TIME-SCALE MANAGEMENT
    # ------------------------------------------------------------------

    def set_active_trade_timescale(self, avg_mfe_bar: float, p75_mfe_bar: float,
                                    expected_mfe_ticks: float = 0.0):
        """
        Call at trade entry with the matched template's time-scale stats.
        avg_mfe_bar / p75_mfe_bar are in 15s bars (0-based bar index where
        MFE historically peaked).  0.0 means unknown → time signals silent.
        expected_mfe_ticks: template's mean MFE in ticks (for capture % calc).
        """
        self._trade_avg_mfe_bar = avg_mfe_bar
        self._trade_p75_mfe_bar = p75_mfe_bar
        self._trade_bars_held   = 0
        self._trade_expected_mfe_ticks = expected_mfe_ticks

    def tick_trade_bar(self):
        """Increment hold counter — call once per 15s bar while position is open."""
        self._trade_bars_held += 1

    def clear_active_trade_timescale(self):
        """Call at trade exit to reset time-scale state."""
        self._trade_avg_mfe_bar = 0.0
        self._trade_p75_mfe_bar = 0.0
        self._trade_bars_held   = 0
        self._trade_expected_mfe_ticks = 0.0

    def set_trade_context(self, side: str, profit_ticks: float):
        """Propagate trade side + P&L to all workers for price-aware conviction."""
        for w in self.workers.values():
            w._trade_side = side
            w._trade_profit_ticks = profit_ticks

    def clear_trade_context(self):
        """Clear trade context from all workers at trade exit."""
        for w in self.workers.values():
            w._trade_side = None
            w._trade_profit_ticks = 0.0

    def end_of_day_review(self, day_trades: list):
        """
        After each trading day, let every worker review the day's trades
        and update their learned accuracy (direction, DMI reliability).

        day_trades: list of dicts with keys:
          - 'side': 'long' or 'short'
          - 'pnl': float
          - 'worker_snapshots': {tf_label: {'d': dir_prob, 'dmi_agrees': bool}}
        """
        if not day_trades:
            return
        for w in self.workers.values():
            w.review_day(day_trades)

    def compute_p_profitable(self, side: str, template_win_rate: float = 0.5) -> float:
        """
        P(profitable) from template win rate (prior) + live DMI/momentum (likelihood).

        Aggregates across all workers, weighted by TF reliability:
          - DMI agreement × ADX strength (60%) — is the trend with you?
          - Momentum agreement × magnitude (40%) — does volume confirm?
          - TF reliability: 15s ≈ 0%, 1h+ ≈ 100%

        Returns float in [0.0, 1.0].
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for w in self.workers.values():
            state = w._last_state
            if state is None:
                continue

            # TF reliability: 0.0 (15s) → 1.0 (1h+)
            tf_rel = min(1.0, math.log(max(2, w.bars_per_update)) / math.log(240))
            if tf_rel < 0.01:
                continue  # skip noise-level workers

            dmi_diff = getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)
            adx = getattr(state, 'adx_strength', 0.0)
            mom = getattr(state, 'momentum_strength', 0.0)

            signal = 0.0

            # DMI component (60%): direction agreement scaled by trend strength
            # Learned DMI reliability adjusts weight: 0.5 (default) → 1.0x, 0.7 → 1.4x, 0.3 → 0.6x
            dmi_agrees = ((side == 'long' and dmi_diff > 0) or
                          (side == 'short' and dmi_diff < 0))
            _dmi_weight = min(1.0, adx / 50.0)
            _dmi_learned = w._dmi_reliability * 2.0  # 0.5 default → 1.0x
            signal += (1.0 if dmi_agrees else -1.0) * _dmi_weight * 0.6 * _dmi_learned

            # Momentum component (40%): volume-weighted velocity agreement
            mom_agrees = ((side == 'long' and mom > 0) or
                          (side == 'short' and mom < 0))
            _mom_weight = min(1.0, math.log1p(abs(mom)) / 3.0)
            signal += (1.0 if mom_agrees else -1.0) * _mom_weight * 0.4

            # Worker's overall learned accuracy scales its vote weight
            weighted_sum += signal * tf_rel * w._conviction_scale
            weight_total += tf_rel

        if weight_total < 0.01:
            return template_win_rate  # no worker data → fall back to prior

        live_score = weighted_sum / weight_total  # [-1, +1]

        # Combine: prior (template WR) shifts baseline, live signals update
        raw = (template_win_rate - 0.5) * 2.0 + live_score
        return 1.0 / (1.0 + math.exp(-3.0 * raw))

    # ------------------------------------------------------------------
    # DAY SETUP
    # ------------------------------------------------------------------

    def prepare_day(self, df_micro: pd.DataFrame, states_micro: list = None,
                    df_5s: pd.DataFrame = None, df_1s: pd.DataFrame = None):
        """
        Task 1 for all workers: pre-aggregate the day's micro bars (15s or 1s)
        to each TF level and compute quantum states (once per day, fast).

        Micro states can be supplied directly (states_micro) if already computed
        by the main forward pass, avoiding redundant work.
        """
        # Ensure DatetimeIndex for pandas resample
        df = df_micro.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df.index = pd.to_datetime(df.index, unit='s')

        # Supply base resolution states directly if available
        base_tf = self.base_resolution_seconds
        if states_micro is not None and base_tf in self.workers:
            self.workers[base_tf].prepare(states_micro)

        for tf_secs in self.active_timeframes:
            if tf_secs == base_tf:
                if states_micro is None:
                    # Compute fresh from micro bars
                    try:
                        s = self.engine.batch_compute_states(df_micro, use_cuda=True)
                        self.workers[base_tf].prepare(s)
                    except Exception as e:
                        logger.warning(f"TBN: {base_tf}s state compute failed: {e}")
                        self.workers[base_tf].prepare([])
                continue

            try:
                tf_df = df.resample(f'{tf_secs}s').agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min',   'close': 'last',
                    'volume': 'sum'
                }).dropna()

                if len(tf_df) < 5:
                    # Too few bars -- 1h workers may have <5 bars near day start
                    self.workers[tf_secs].prepare([])
                    continue

                states = self.engine.batch_compute_states(tf_df, use_cuda=True)
                self.workers[tf_secs].prepare(states)
            except Exception as e:
                logger.warning(f"TBN: TF={tf_secs}s state compute failed: {e}")
                self.workers[tf_secs].prepare([])

        # Sub-resolution workers (5s, 1s): cannot resample from df_micro, use external data.
        _sub_res_data = {5: df_5s, 1: df_1s}
        for tf_secs in self.TIMEFRAMES_SECONDS:
            if tf_secs >= self.base_resolution_seconds:
                continue
            lbl = self._TF_LABELS.get(tf_secs, str(tf_secs))
            df_sub = _sub_res_data.get(tf_secs)
            if df_sub is None or (hasattr(df_sub, 'empty') and df_sub.empty):
                print(f"  TBN [{lbl}]: no data supplied — worker inactive")
                self.workers[tf_secs].prepare([])
                continue
            try:
                print(f"  TBN [{lbl}]: computing states for {len(df_sub):,} bars ...", end='', flush=True)
                states = self.engine.batch_compute_states(df_sub, use_cuda=True)
                self.workers[tf_secs].prepare(states)
                print(f" {len(states):,} states ready")
            except Exception as e:
                print(f" FAILED: {e}")
                logger.warning(f"TBN: TF={tf_secs}s sub-res state compute failed: {e}")
                self.workers[tf_secs].prepare([])

    # ------------------------------------------------------------------
    # PER-BAR UPDATE
    # ------------------------------------------------------------------

    def tick_all(self, bar_i: int) -> int:
        """
        Update all workers for current micro bar index (15s or 1s).
        Each worker self-decides (event-driven by TF bar change).
        Returns: number of workers that updated their belief.
        """
        self._current_bar = bar_i
        updated = 0
        for worker in self.workers.values():
            if worker.tick(bar_i, self.pattern_library, self.scaler,
                           self.valid_tids, self.centroids_scaled,
                           scaler_mean=self.scaler_mean,
                           scaler_scale=self.scaler_scale):
                updated += 1
        return updated

    def tick_sub_resolution(self, tf_bar_idx_1s: int, tf_bar_idx_5s: int = -1):
        """
        Tick only sub-resolution workers (1s, 5s) at their native bar index.
        Called from the 1s inner loop when a position is open.
        Does NOT touch workers >= base_resolution_seconds (they already ticked).
        """
        w1 = self.workers.get(1)
        if w1 and w1._states and tf_bar_idx_1s >= 0:
            w1.tick_at(tf_bar_idx_1s, self.pattern_library, self.scaler,
                       self.valid_tids, self.centroids_scaled)
        w5 = self.workers.get(5)
        if w5 and w5._states and tf_bar_idx_5s >= 0:
            w5.tick_at(tf_bar_idx_5s, self.pattern_library, self.scaler,
                       self.valid_tids, self.centroids_scaled)

    def get_belief(self) -> Optional[BeliefState]:
        """
        Collect current beliefs from all active workers and compute
        path conviction (weighted geometric mean of P(direction)).

        Returns None if fewer than MIN_ACTIVE_LEVELS have valid beliefs.
        """
        active = {
            tf: w.current_belief
            for tf, w in self.workers.items()
            if w.current_belief is not None
        }

        if len(active) < self.MIN_ACTIVE_LEVELS:
            return None

        probs_long     = []
        probs_short    = []
        weights        = []
        wave_maturities = []

        for tf, belief in active.items():
            p  = np.clip(belief.dir_prob, 1e-7, 1.0 - 1e-7)
            probs_long.append(np.log(p))
            probs_short.append(np.log(1.0 - p))
            # DNA-weighted: workers with higher DNA agreement contribute more
            dna_w = getattr(belief, 'dna_agreement', 0.5)
            weights.append(self._weight_map.get(tf, 1.0) * (0.5 + 0.5 * dna_w))
            wave_maturities.append(belief.wave_maturity)

        w_arr          = np.array(weights) / sum(weights)
        path_long      = float(np.exp(np.dot(w_arr, probs_long)))
        path_short     = float(np.exp(np.dot(w_arr, probs_short)))
        wave_maturity  = float(np.dot(w_arr, wave_maturities))

        if path_long >= path_short:
            direction  = 'long'
            conviction = path_long
        else:
            direction  = 'short'
            conviction = path_short

        # Predicted MFE from the decision-level worker, or nearest available
        dec_belief = active.get(self.decision_tf)
        if dec_belief is None:
            # Pick closest available TF to decision_tf
            dec_belief = min(active.values(),
                             key=lambda b: abs(b.tf_seconds - self.decision_tf))
        pred_mfe_ticks = max(0.0, dec_belief.pred_mfe / 0.25) if dec_belief else 0.0

        # decision_wave_maturity: maturity at the DECISION TF only.
        # A mature 30s sub-wave is often just a forming 5m wave -- don't conflate them.
        # Only the decision-TF worker's maturity tells us if the TRADEABLE move is exhausted.
        decision_wave_maturity = dec_belief.wave_maturity if dec_belief else wave_maturity

        return BeliefState(
            direction              = direction,
            conviction             = conviction,
            predicted_mfe          = pred_mfe_ticks,
            active_levels          = len(active),
            wave_maturity          = wave_maturity,
            decision_wave_maturity = decision_wave_maturity,
            tf_beliefs             = active,
        )

    # ------------------------------------------------------------------
    # DIRECTION CONSENSUS (multi-signal aggregation)
    # ------------------------------------------------------------------

    def get_direction_consensus(self, proposed_side: str) -> dict:
        """
        Multi-signal direction consensus across all active TF workers.

        Aggregates 4 independent signals:
          1. DMI consensus (30%): weighted by ADX strength
          2. Momentum consensus (20%): velocity agreement across scales
          3. Worker vote (30%): weighted majority of dir_prob values
          4. Trend alignment (20%): higher-TF workers (>= 5m) agreement

        Returns {'direction', 'confidence' [0-1], 'signals': {...}}
        """
        dmi_score = 0.0
        mom_score = 0.0
        vote_score = 0.0
        trend_score = 0.0
        dmi_weight = 0.0
        mom_weight = 0.0
        vote_weight = 0.0
        trend_weight = 0.0

        for tf, worker in self.workers.items():
            state = worker._last_state
            belief = worker.current_belief
            if state is None or belief is None:
                continue

            w = self._weight_map.get(tf, 1.0)

            # Signal 1: DMI consensus (weighted by ADX)
            dmi_diff = getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)
            adx = getattr(state, 'adx_strength', 0.0)
            if adx > 15:
                dmi_agrees = ((proposed_side == 'long' and dmi_diff > 0) or
                              (proposed_side == 'short' and dmi_diff < 0))
                _adx_w = min(1.0, adx / 50.0)
                dmi_score += (1.0 if dmi_agrees else -1.0) * _adx_w * w * worker._dmi_reliability * 2.0
                dmi_weight += _adx_w * w

            # Signal 2: Momentum consensus
            mom = getattr(state, 'momentum_strength', 0.0)
            if abs(mom) > 0.01:
                mom_agrees = ((proposed_side == 'long' and mom > 0) or
                              (proposed_side == 'short' and mom < 0))
                _mom_mag = min(1.0, math.log1p(abs(mom)) / 3.0)
                mom_score += (1.0 if mom_agrees else -1.0) * _mom_mag * w
                mom_weight += _mom_mag * w

            # Signal 3: Worker dir_prob vote
            _dir_agrees = ((proposed_side == 'long' and belief.dir_prob > 0.5) or
                           (proposed_side == 'short' and belief.dir_prob < 0.5))
            _strength = abs(belief.dir_prob - 0.5) * 2.0
            vote_score += (1.0 if _dir_agrees else -1.0) * _strength * w * worker._conviction_scale
            vote_weight += w

            # Signal 4: Trend alignment (only higher TFs >= 5m)
            if tf >= 300:
                _agrees = ((proposed_side == 'long' and belief.dir_prob > 0.55) or
                           (proposed_side == 'short' and belief.dir_prob < 0.45))
                trend_score += (1.0 if _agrees else -0.5) * w
                trend_weight += w

        # Normalize each to [-1, +1]
        dmi_n   = dmi_score / dmi_weight   if dmi_weight   > 0.01 else 0.0
        mom_n   = mom_score / mom_weight   if mom_weight   > 0.01 else 0.0
        vote_n  = vote_score / vote_weight if vote_weight  > 0.01 else 0.0
        trend_n = trend_score / trend_weight if trend_weight > 0.01 else 0.0

        # Weighted composite: DMI 30%, Momentum 20%, Vote 30%, Trend 20%
        composite = 0.30 * dmi_n + 0.20 * mom_n + 0.30 * vote_n + 0.20 * trend_n

        # composite > 0 = agrees with proposed_side
        if composite >= 0:
            direction = proposed_side
            confidence = 0.5 + 0.5 * min(1.0, composite)
        else:
            direction = 'short' if proposed_side == 'long' else 'long'
            confidence = 0.5 + 0.5 * min(1.0, abs(composite))

        return {
            'direction':  direction,
            'confidence': round(confidence, 4),
            'signals': {
                'dmi': round(dmi_n, 3), 'momentum': round(mom_n, 3),
                'vote': round(vote_n, 3), 'trend': round(trend_n, 3),
                'composite': round(composite, 3),
            }
        }

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION (shared by all workers)
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_features(state, tf_secs: int, depth: int = 0) -> list:
        """
        Convert ThreeBodyQuantumState -> 16D feature vector.
        Same order as FractalClusteringEngine.extract_features().
        Ancestry features (parent_z, parent_dmi_diff, root_is_roche, tf_alignment)
        are 0.0 because live TF-aggregated bars have no parent chain context.
        PID features (term_pid, oscillation_coherence) default to 0.0 if the
        engine hasn't computed them yet (safe fallback).

        velocity and momentum use log1p(|x|) compression -- must match
        FractalClusteringEngine.extract_features() exactly.
        """
        z = getattr(state, 'z_score',           0.0)
        v = getattr(state, 'particle_velocity',  0.0)
        m = getattr(state, 'momentum_strength',  0.0)
        c = getattr(state, 'coherence',          0.0)

        tf_scale   = np.log2(max(1, tf_secs))
        self_adx   = getattr(state, 'adx_strength',   0.0) / 100.0
        self_hurst = getattr(state, 'hurst_exponent',  0.5)
        self_dmi   = (getattr(state, 'dmi_plus',  0.0)
                    - getattr(state, 'dmi_minus', 0.0)) / 100.0

        # log1p compression -- must match extract_features()
        v_feat = np.log1p(abs(v))
        m_feat = np.log1p(abs(m))

        # PID / oscillation features (positions 14-15 in the 16D vector)
        self_pid     = getattr(state, 'term_pid',              0.0)
        self_osc_coh = getattr(state, 'oscillation_coherence', 0.0)

        # Ancestry = 0.0 (no parent chain for live aggregated TF bars)
        return [z, v_feat, m_feat, c,
                tf_scale, float(depth), 0.0,
                self_adx, self_hurst, self_dmi,
                0.0, 0.0, 0.0, 0.0,
                self_pid, self_osc_coh]

    # ------------------------------------------------------------------
    # PHYSICS DECAY TRACKING
    # ------------------------------------------------------------------

    def start_trade_tracking(self, side: str, entry_bar: int,
                             pattern_horizon_bars: int):
        """
        Called when a trade opens. Records each worker's current z_score
        so we can track physics decay (drift away from expected trajectory).

        The expected trajectory is LINEAR MEAN REVERSION: z -> 0 over T_k bars.
        If reality diverges (z moves AGAINST trade direction), the decay score
        rises. Fast-TF workers weight more for exit (inverse of entry weighting).
        """
        self._decay_trade_side = side
        self._decay_entry_bar = entry_bar
        self._decay_pattern_horizon = max(1, pattern_horizon_bars)
        self._decay_entry_physics = {}
        for tf, worker in self.workers.items():
            b = worker.current_belief
            if b is not None:
                self._decay_entry_physics[tf] = {
                    'z': b.z_score,
                    'm': b.momentum,
                    'd': b.dir_prob,
                }

    def stop_trade_tracking(self):
        """Called when a trade closes -- clears decay state."""
        self._decay_trade_side = None
        self._decay_entry_bar = 0
        self._decay_entry_physics = {}

    def get_decay_cascade(self, cell_bounds=None, live_16d=None) -> dict:
        """
        Compute physics decay cascade across all workers since trade entry.

        Each worker's z_score trajectory is compared to the EXPECTED trajectory
        (linear mean reversion from entry_z -> 0 over the pattern horizon T_k).
        Deviations AGAINST the trade direction are penalized.

        Also checks if the live 16D feature vector has exited the Hypervolume Cell.

        Tolerance band tau narrows from alpha_max -> alpha_min over T_k.
        Fast-TF workers get MORE weight (inverse of entry weighting) because
        they detect reversals first (bottom-up cascade).

        Returns:
            cascade_score: 0=healthy, >theta=exit signal
            per_worker:    {tf_label: decay_w} for diagnostics
            should_exit:   cascade > theta_exit
            progress:      dt / T_k [0..2]
            tau:           current tolerance band width
        """
        if self._decay_trade_side is None:
            return {'cascade_score': 0.0, 'per_worker': {},
                    'should_exit': False, 'progress': 0.0,
                    'tau': self.DECAY_ALPHA_MAX}

        dt = max(1, self._current_bar - self._decay_entry_bar)
        T_k = self._decay_pattern_horizon
        progress = min(2.0, dt / T_k)

        # Adaptive tolerance: wide at start, narrows to alpha_min at horizon
        tau = self.DECAY_ALPHA_MAX - (self.DECAY_ALPHA_MAX - self.DECAY_ALPHA_MIN) * min(1.0, progress)

        # Sign convention:
        #   LONG: entered z < 0, expect z to rise toward 0. Adverse = z falls further.
        #   SHORT: entered z > 0, expect z to fall toward 0. Adverse = z rises further.
        sign_dir = -1.0 if self._decay_trade_side == 'long' else 1.0

        per_worker = {}
        cascade_num = 0.0
        cascade_den = 0.0
        _max_w = max(self.TF_WEIGHTS)

        for tf, worker in self.workers.items():
            b = worker.current_belief
            entry = self._decay_entry_physics.get(tf)
            if b is None or entry is None:
                continue

            entry_z = entry['z']
            current_z = b.z_score

            # Expected: linear mean reversion toward 0
            expected_z = entry_z * max(0.0, 1.0 - progress)
            residual = current_z - expected_z

            # Bad residual: positive when physics move AGAINST trade direction
            bad_residual = residual * sign_dir
            decay_w = max(0.0, bad_residual / (tau + 1e-9))

            tf_label = self._TF_LABELS.get(tf, str(tf))
            per_worker[tf_label] = round(decay_w, 3)

            # Inverse weight: fast TFs -> high exit influence
            inv_w = _max_w - self._weight_map.get(tf, 1.0) + 0.1
            cascade_num += inv_w * decay_w
            cascade_den += inv_w

        cascade = cascade_num / cascade_den if cascade_den > 0 else 0.0

        # Hypervolume Cell Exit Signal
        cell_exit = False
        cell_breach = 0.0
        if cell_bounds and live_16d is not None:
            cell_min, cell_max = cell_bounds
            # Binary check
            inside = np.all(live_16d >= cell_min) and np.all(live_16d <= cell_max)
            if not inside:
                cell_exit = True
                # Compute breach magnitude (Euclidean distance outside the box)
                breach_vec = np.maximum(cell_min - live_16d, 0) + np.maximum(live_16d - cell_max, 0)
                cell_breach = np.linalg.norm(breach_vec)
                # Boost cascade score
                cascade += cell_breach * 0.5

        return {
            'cascade_score': round(cascade, 4),
            'per_worker': per_worker,
            'should_exit': cascade > self.DECAY_THETA_EXIT,
            'progress': round(progress, 3),
            'tau': round(tau, 3),
            'cell_exit': cell_exit,
            'cell_breach': round(cell_breach, 3)
        }

    # ------------------------------------------------------------------
    # WORKER SNAPSHOT
    # ------------------------------------------------------------------

    def get_worker_snapshot(self) -> dict:
        """
        Freeze every active worker's current belief as a compact dict.

        Called at trade ENTRY and EXIT so we can compare each worker's
        assumption against what actually happened:
          entry snapshot -> what did each TF worker predict?
          exit snapshot  -> who flipped direction during the trade?

        Returns:
            {tf_label: {'d': dir_prob, 'c': conviction,
                        'm': wave_maturity, 'mfe': pred_mfe}}
            e.g. {'1h': {'d': 0.71, 'c': 0.42, 'm': 0.12, 'mfe': 8.2}, ...}
        d > 0.5 = worker leans LONG, d < 0.5 = worker leans SHORT.
        """
        snap = {}
        for tf, worker in self.workers.items():
            b = worker.current_belief
            if b is not None:
                _s = worker._last_state
                _dmi = (getattr(_s, 'dmi_plus', 0.0) - getattr(_s, 'dmi_minus', 0.0)) if _s else 0.0
                snap[self._TF_LABELS.get(tf, str(tf))] = {
                    'd':   round(b.dir_prob,      3),
                    'c':   round(b.conviction,    3),
                    'm':   round(b.wave_maturity, 3),
                    'mfe': round(b.pred_mfe,      1),
                    'dmi': round(_dmi,             2),
                    'lz':  getattr(_s, 'lagrange_zone', 'UNKNOWN') if _s else 'UNKNOWN',
                    'z':   round(getattr(_s, 'z_score', 0.0), 2) if _s else 0.0,
                }
        return snap

    def get_worker_state_counts(self) -> dict:
        """Return {tf_label: n_states} for every worker. 0 = worker has no data."""
        return {
            self._TF_LABELS.get(tf, str(tf)): len(w._states)
            for tf, w in self.workers.items()
        }

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_exit_signal(self, side: str, trade_context: dict = None) -> dict:
        """
        Called every bar while a position is open.
        Returns a dict with exit adjustment recommendations.

        side: 'long' or 'short' — the current position direction.
        trade_context: optional dict with real-time trade metrics:
            profit_ticks, running_mfe_ticks, running_mae_ticks,
            pct_mfe_captured, pct_hold_elapsed

        Returns:
            {
              'tighten_trail': bool,   # shrink trail stop distance
              'widen_trail':   bool,   # grow trail stop (conviction is high)
              'urgent_exit':   bool,   # exit NOW (direction flipped, high conviction)
              'conviction':    float,  # current path conviction
              'wave_maturity': float,  # decision TF wave maturity
              'reason':        str,    # human-readable reason
            }
        """
        belief = self.get_belief()
        if belief is None:
            return {'tighten_trail': False, 'widen_trail': False,
                    'urgent_exit': False, 'conviction': 0.0,
                    'wave_maturity': 0.0, 'reason': 'no_belief'}

        trade_long = (side == 'long')
        belief_long = (belief.direction == 'long')
        direction_aligned = (trade_long == belief_long)
        wave_mature = belief.decision_wave_maturity  # decision TF worker only

        # Urgent exit: high conviction in the OPPOSITE direction
        urgent = belief.is_confident and not direction_aligned and belief.conviction > self.URGENT_EXIT_CONVICTION_THRESHOLD

        # Tighten: ONLY when wave is deeply mature (approaching exhaustion zone, > 0.85).
        # Pre-fix: also tightened when conviction was low — this fired on ~32% of all
        # bars, collapsing the trail to 2 ticks mid-move and causing premature exits.
        # Low conviction during an active trade is normal noise, not a reversal signal.
        tighten = wave_mature > self.TIGHTEN_TRAIL_WAVE_MATURITY_THRESHOLD

        # Widen: strong conviction aligned with trade direction, wave is fresh
        widen = belief.is_confident and direction_aligned and wave_mature < self.WIDEN_TRAIL_WAVE_MATURITY_THRESHOLD

        # Time-exhaustion: template's historical MFE peak window has passed.
        # avg_mfe_bar = 0.0 means unknown (no --fresh run yet) → silent.
        # At 1.5× avg_mfe_bar: tighten trail (move is likely past its peak).
        # At 2.5× p75_mfe_bar: urgent exit (conservative window fully elapsed).
        _time_tighten = False
        _time_urgent  = False
        if self._trade_avg_mfe_bar > 0:
            _p75 = self._trade_p75_mfe_bar if self._trade_p75_mfe_bar > 0 else self._trade_avg_mfe_bar * 1.5
            if self._trade_bars_held > _p75 * 2.5:
                _time_urgent  = True
            elif self._trade_bars_held > self._trade_avg_mfe_bar * 1.5:
                _time_tighten = True

        tighten = tighten or _time_tighten
        urgent  = urgent  or _time_urgent

        # ── Continuous hold/exit pressure ──────────────────────────────────
        # Single scalar that drives all exit decisions.
        # Positive = hold (profitable + early + aligned)
        # Negative = exit (losing + late + diverging)
        net_pressure = 0.0
        _pressure_reason = ''

        if trade_context and self._trade_expected_mfe_ticks > 0:
            _profit   = trade_context.get('profit_ticks', 0.0)
            _pct_cap  = trade_context.get('pct_mfe_captured', 0.0)
            _pct_hold = trade_context.get('pct_hold_elapsed', 0.0)
            _aligned  = 1.0 if direction_aligned else 0.0

            # Live P(profitable): prior (template WR) + live DMI/momentum
            _p_prof = self.compute_p_profitable(
                side, trade_context.get('template_win_rate', 0.5))

            # Hold pressure: profitable + early + workers agree, scaled by P(profitable)
            _profit_factor = min(1.0, max(0.0, _profit) / (self._trade_expected_mfe_ticks + 1))
            _time_factor   = max(0.0, 1.0 - _pct_hold)
            _hold = _profit_factor * _time_factor * (0.5 + 0.5 * _aligned) * _p_prof

            # Exit pressure: losing + late + workers disagree, scaled by P(unprofitable)
            _loss_factor = min(1.0, max(0.0, -_profit) / 20.0)
            _late_factor = max(0.0, _pct_hold - 1.0)
            _exit = (_loss_factor + _late_factor * (1.0 - _aligned)) * (1.0 - _p_prof)

            net_pressure = _hold - _exit

            # Map continuous pressure to discrete trail signals
            if net_pressure > 0.3:
                widen = True; tighten = False
                _pressure_reason = 'pressure_hold'
            elif net_pressure < -0.3:
                tighten = True; widen = False
                _pressure_reason = 'pressure_exit'
            elif _pct_cap >= 0.60:
                tighten = True; widen = False
                _pressure_reason = 'mfe_captured_60pct'
            elif _pct_hold > 1.5 and _pct_cap < 0.10:
                urgent = True
                _pressure_reason = 'time_expired_no_capture'

        reason = (_pressure_reason if _pressure_reason else
                  'time_exhausted' if _time_urgent  else
                  'urgent_flip'    if urgent         else
                  'time_tighten'   if _time_tighten  else
                  'wave_mature'    if wave_mature > self.TIGHTEN_TRAIL_WAVE_MATURITY_THRESHOLD else
                  'aligned_fresh'  if widen           else
                  'low_conviction' if not belief.is_confident else 'neutral')

        return {
            'tighten_trail': tighten and not urgent,
            'widen_trail':   widen and not _time_tighten,
            'urgent_exit':   urgent,
            'conviction':    belief.conviction,
            'wave_maturity': wave_mature,
            'reason':        reason,
            'net_pressure':  net_pressure,
        }

    def summary(self) -> str:
        """One-line status of all workers."""
        parts = []
        for tf in self.TIMEFRAMES_SECONDS:
            w = self.workers[tf]
            b = w.current_belief
            if b is None:
                parts.append(f"{tf}s:---")
            else:
                arrow = "L" if b.dir_prob > 0.5 else "S"
                parts.append(f"{tf}s:{arrow}{b.dir_prob:.2f}")
        return " | ".join(parts)
