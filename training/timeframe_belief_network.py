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
    tf_seconds:    int
    dir_prob:      float    # P(LONG) from logistic regression [0..1]
    pred_mfe:      float    # OLS predicted MFE in price points
    template_id:   int
    tf_bar_idx:    int      # which TF bar produced this belief
    conviction:    float    # |dir_prob - 0.5| * 2  -> how sure this worker is [0..1]
    wave_maturity: float = 0.0  # P(wave near completion) [0..1]
    # Composite: 0.4*pattern_maturity + 0.3*min(1,|z|/3) + 0.3*tunnel_probability
    # High value = wave is well-developed/near exhaustion = higher entry risk


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

    LEAF_TOP_K = 3  # analysis threads at the leaf (15s or 1s)

    def __init__(self, tf_seconds: int, is_leaf: bool = False, base_resolution_seconds: int = 15):
        self.tf_seconds              = tf_seconds
        self.base_resolution_seconds = base_resolution_seconds
        self.bars_per_update         = max(1, tf_seconds // base_resolution_seconds)
        self.is_leaf                 = is_leaf

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
        if _any_fitted:
            dir_prob = 0.5 * _phys_dir + 0.5 * dir_prob
        else:
            dir_prob = _phys_dir

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

        self.current_belief = WorkerBelief(
            tf_seconds    = self.tf_seconds,
            dir_prob      = dir_prob,
            pred_mfe      = pred_mfe,
            template_id   = primary_tid,
            tf_bar_idx    = tf_bar_idx,
            conviction    = abs(dir_prob - 0.5) * 2.0,
            wave_maturity = wave_maturity,
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
    _TF_LABELS = {3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
                  180:'3m',  60:'1m',   30:'30s',   15:'15s',
                  5:'5s',    1:'1s'}

    def __init__(self, pattern_library: dict, scaler, engine,
                 valid_tids: list, centroids_scaled: np.ndarray,
                 decision_tf: int = DEFAULT_DECISION_TF,
                 base_resolution_seconds: int = 15):
        self.pattern_library   = pattern_library
        self.scaler            = scaler
        self.engine            = engine
        self.valid_tids        = valid_tids
        self.centroids_scaled  = centroids_scaled
        self.decision_tf       = decision_tf
        self.base_resolution_seconds = base_resolution_seconds

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

        probs_long     = []
        probs_short    = []
        weights        = []
        wave_maturities = []

        for tf, belief in active.items():
            p  = np.clip(belief.dir_prob, 1e-7, 1.0 - 1e-7)
            probs_long.append(np.log(p))
            probs_short.append(np.log(1.0 - p))
            weights.append(self._weight_map.get(tf, 1.0))
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
        return [abs(z), v_feat, m_feat, c,
                tf_scale, float(depth), 0.0,
                self_adx, self_hurst, self_dmi,
                0.0, 0.0, 0.0, 0.0,
                self_pid, self_osc_coh]

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
                snap[self._TF_LABELS.get(tf, str(tf))] = {
                    'd':   round(b.dir_prob,      3),
                    'c':   round(b.conviction,    3),
                    'm':   round(b.wave_maturity, 3),
                    'mfe': round(b.pred_mfe,      1),
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
