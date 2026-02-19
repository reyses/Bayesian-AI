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

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkerBelief:
    """Current belief from one TF worker."""
    tf_seconds:  int
    dir_prob:    float    # P(LONG) from logistic regression [0..1]
    pred_mfe:    float    # OLS predicted MFE in price points
    template_id: int
    tf_bar_idx:  int      # which TF bar produced this belief
    conviction:  float    # |dir_prob - 0.5| * 2  -> how sure this worker is [0..1]


@dataclass
class BeliefState:
    """Aggregated belief across all active TF workers."""
    direction:      str           # 'long' | 'short'
    conviction:     float         # weighted geometric mean across TF levels [0..1]
    predicted_mfe:  float         # MFE prediction in ticks (from decision-level worker)
    active_levels:  int           # how many TF levels contributed
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
    """
    coeff = lib.get('dir_coeff')
    if coeff is not None:
        logit = float(np.dot(feat_s, np.array(coeff)) + lib.get('dir_intercept', 0.0))
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
        return float(np.dot(feat_s, np.array(coeff)) + lib.get('mfe_intercept', 0.0))
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

    LEAF_TOP_K = 3  # analysis threads at the 15s leaf

    def __init__(self, tf_seconds: int, is_leaf: bool = False):
        self.tf_seconds      = tf_seconds
        self.bars_per_update = max(1, tf_seconds // 15)
        self.is_leaf         = is_leaf

        # Filled by prepare()
        self._states: list = []

        # Current belief -- sticky until TF bar changes
        self.current_belief: Optional[WorkerBelief] = None
        self._last_tf_bar_idx: int = -1

    def prepare(self, states: list):
        """Supply Task-1 result: pre-computed states for the day."""
        self._states          = states
        self.current_belief   = None
        self._last_tf_bar_idx = -1

    def tick(self, bar_i: int, pattern_library: dict, scaler,
             valid_tids: list, centroids_scaled: np.ndarray) -> bool:
        """
        Called every 15s bar. Returns True if belief was updated.

        Task 1 check: has my TF bar changed?
        Task 2: if yes, re-run cluster match + regression.
        """
        tf_bar_idx = bar_i // self.bars_per_update

        if tf_bar_idx == self._last_tf_bar_idx:
            return False   # TF bar unchanged -- belief still valid
        if not self._states or tf_bar_idx >= len(self._states):
            return False   # No state yet (warmup period)

        state = self._states[tf_bar_idx]
        self._last_tf_bar_idx = tf_bar_idx

        # --- Task 2: Analysis ---
        feat   = TimeframeBeliefNetwork.state_to_features(state, self.tf_seconds)
        feat_s = scaler.transform([feat])[0]
        dists  = np.linalg.norm(centroids_scaled - feat_s, axis=1)

        if self.is_leaf:
            # Multiple parallel analysis threads: top-K nearest clusters
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
        else:
            best_idx    = int(np.argmin(dists))
            primary_tid = valid_tids[best_idx]
            lib         = pattern_library[primary_tid]
            dir_prob    = _logistic_prob(feat_s, lib)
            pred_mfe    = _ols_mfe(feat_s, lib)

        self.current_belief = WorkerBelief(
            tf_seconds  = self.tf_seconds,
            dir_prob    = dir_prob,
            pred_mfe    = pred_mfe,
            template_id = primary_tid,
            tf_bar_idx  = tf_bar_idx,
            conviction  = abs(dir_prob - 0.5) * 2.0,
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

    TIMEFRAMES_SECONDS = [3600, 1800, 900, 300, 180, 60, 30, 15]
    TF_WEIGHTS         = [4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    MIN_CONVICTION     = 0.52   # skip trade if path conviction below this
    MIN_ACTIVE_LEVELS  = 3      # need >=3 active TF levels for a signal
    DEFAULT_DECISION_TF = 300   # 5m: default scale at which to read predicted_mfe

    def __init__(self, pattern_library: dict, scaler, engine,
                 valid_tids: list, centroids_scaled: np.ndarray,
                 decision_tf: int = DEFAULT_DECISION_TF):
        self.pattern_library   = pattern_library
        self.scaler            = scaler
        self.engine            = engine
        self.valid_tids        = valid_tids
        self.centroids_scaled  = centroids_scaled
        self.decision_tf       = decision_tf

        self._weight_map = dict(zip(self.TIMEFRAMES_SECONDS, self.TF_WEIGHTS))

        self.workers: Dict[int, TimeframeWorker] = {
            tf: TimeframeWorker(tf, is_leaf=(tf == 15))
            for tf in self.TIMEFRAMES_SECONDS
        }

    # ------------------------------------------------------------------
    # DAY SETUP
    # ------------------------------------------------------------------

    def prepare_day(self, df_15s: pd.DataFrame, states_15s: list = None):
        """
        Task 1 for all workers: pre-aggregate the day's 15s bars to each
        TF level and compute quantum states (once per day, fast).

        15s states can be supplied directly (states_15s) if already computed
        by the main forward pass, avoiding redundant work.
        """
        # Ensure DatetimeIndex for pandas resample
        df = df_15s.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df.index = pd.to_datetime(df.index, unit='s')

        # Supply 15s states directly if available
        if states_15s is not None:
            self.workers[15].prepare(states_15s)

        for tf_secs in self.TIMEFRAMES_SECONDS:
            if tf_secs == 15:
                if states_15s is None:
                    # Compute fresh from 15s bars
                    try:
                        s = self.engine.batch_compute_states(df_15s, use_cuda=True)
                        self.workers[15].prepare(s)
                    except Exception as e:
                        logger.warning(f"TBN: 15s state compute failed: {e}")
                        self.workers[15].prepare([])
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

    # ------------------------------------------------------------------
    # PER-BAR UPDATE
    # ------------------------------------------------------------------

    def tick_all(self, bar_i: int) -> int:
        """
        Update all workers for current 15s bar index.
        Each worker self-decides (event-driven by TF bar change).
        Returns: number of workers that updated their belief.
        """
        updated = 0
        for worker in self.workers.values():
            if worker.tick(bar_i, self.pattern_library, self.scaler,
                           self.valid_tids, self.centroids_scaled):
                updated += 1
        return updated

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

        probs_long  = []
        probs_short = []
        weights     = []

        for tf, belief in active.items():
            p  = np.clip(belief.dir_prob, 1e-7, 1.0 - 1e-7)
            probs_long.append(np.log(p))
            probs_short.append(np.log(1.0 - p))
            weights.append(self._weight_map.get(tf, 1.0))

        w_arr       = np.array(weights) / sum(weights)
        path_long   = float(np.exp(np.dot(w_arr, probs_long)))
        path_short  = float(np.exp(np.dot(w_arr, probs_short)))

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

        return BeliefState(
            direction     = direction,
            conviction    = conviction,
            predicted_mfe = pred_mfe_ticks,
            active_levels = len(active),
            tf_beliefs    = active,
        )

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION (shared by all workers)
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_features(state, tf_secs: int, depth: int = 0) -> list:
        """
        Convert ThreeBodyQuantumState -> 14D feature vector.
        Same order as FractalClusteringEngine.extract_features().
        Ancestry features (parent_z, parent_dmi_diff, root_is_roche, tf_alignment)
        are 0.0 because live TF-aggregated bars have no parent chain context.

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

        # Ancestry = 0.0 (no parent chain for live aggregated TF bars)
        return [abs(z), v_feat, m_feat, c,
                tf_scale, float(depth), 0.0,
                self_adx, self_hurst, self_dmi,
                0.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

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
