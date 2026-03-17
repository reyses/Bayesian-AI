"""
Timeframe Belief Network
========================
N workers, each monitoring a different timeframe level simultaneously.

ARCHITECTURE
------------
Each worker has TWO tasks:
  Task 1 (Aggregation):  accumulate 15s bars -> TF OHLCV bar -> MarketState
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

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class BandContext:
    """Where price sits relative to Standard Error Bands at one TF."""
    tf_seconds: int
    z_score: float
    sigma: float
    center: float
    band: int                   # discrete sigma level: -3..+3
    band_position: float        # continuous [-1.0, +1.0]
    at_support: bool            # z <= -1.0
    at_resistance: bool         # z >= +1.0
    band_label: str             # '-2σ', '+1σ', 'center', etc.


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
    band_context:  Optional[BandContext] = None
    # Composite: 0.4*pattern_maturity + 0.3*min(1,|z|/3) + 0.3*reversion_probability
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
    band_confluence: Optional[dict] = None

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
        Receives pre-computed MarketState list for its TF (light).
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
        # batch_compute_states() wraps each result as {'bar_idx': i, 'state': MarketState, ...}
        # Unwrap so state_to_features() receives the actual physics object, not the dict wrapper.
        state = state_raw['state'] if isinstance(state_raw, dict) and 'state' in state_raw else state_raw
        self._last_tf_bar_idx = tf_bar_idx

        return self._analyze(state, tf_bar_idx, pattern_library, scaler,
                             valid_tids, centroids_scaled)

    def tick_at(self, tf_bar_idx: int, pattern_library: dict, scaler,
                valid_tids: list, centroids_scaled: np.ndarray) -> bool:
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
                             valid_tids, centroids_scaled)

    def _analyze(self, state, tf_bar_idx: int, pattern_library: dict, scaler,
                 valid_tids: list, centroids_scaled: np.ndarray) -> bool:
        """Task 2: feature extraction, cluster matching, physics blend, belief update."""
        feat   = TimeframeBeliefNetwork.state_to_features(state, self.tf_seconds)
        # Pad to match scaler dimensions (22D when --lookback, 16D otherwise)
        _expected = scaler.n_features_in_
        if len(feat) < _expected:
            feat = feat + [0.0] * (_expected - len(feat))
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

        # ── Physics blend: momentum-aware direction from velocity + acceleration ──
        # Instead of mean-reverting z_score (which fights trends), use the
        # particle's velocity (dp/dt) and net force (d²p/dt² ≈ net_force) to
        # determine direction.  Positive momentum → P(LONG) high.
        #
        # Sensitivity scales with log(bars_aggregated) — higher TF workers have
        # more samples so the signal is statistically stronger.
        _n_bars = max(1, self.bars_per_update)
        _phys_sensitivity = 0.5 + 0.5 * (np.log(_n_bars) / np.log(240))  # [0.5, 1.0]
        _velocity = float(getattr(state, 'velocity', 0.0))
        _accel    = float(getattr(state, 'net_force', 0.0))
        _momentum = _velocity + 0.5 * _accel
        _phys_dir = _sigmoid(_momentum * _phys_sensitivity)

        # Blend with ML signal only if a fitted logistic regression exists.
        # Unfitted templates fall back to long_bias ≈ 0.59 (NQ bullish noise).
        # In that case, use pure physics to avoid degenerate uniform dir_prob values.
        if _any_fitted:
            dir_prob = 0.5 * _phys_dir + 0.5 * dir_prob
        else:
            dir_prob = _phys_dir

        # Wave maturity: estimate of how "mature" (near completion) the current wave is.
        # High value = wave is well-developed / near exhaustion = higher entry risk.
        # Composite of the three strongest exhaustion signals from the market state:
        #   pattern_maturity  : engine's L7 development measure (0-1)
        #   |z_score| / 3.0   : approach to band extreme (3 sigma = fully mature)
        #   reversion_probability: P(revert to center) = how close to reversal
        _pm  = getattr(state, 'pattern_maturity',   0.0)
        _tp  = getattr(state, 'reversion_probability', 0.0)
        _z   = abs(getattr(state, 'z_score',        0.0))
        wave_maturity = float(np.clip(
            0.4 * _pm + 0.3 * min(1.0, _z / 3.0) + 0.3 * _tp,
            0.0, 1.0
        ))

        # ── Band Context (Standard Error Bands) ─────────────────────────
        _z_raw = float(getattr(state, 'z_score', 0.0))
        _sigma = float(getattr(state, 'regression_sigma', 0.0))
        _center = float(getattr(state, 'regression_center', 0.0))
        _band_int = int(np.clip(np.round(_z_raw), -3, 3))
        _band_pos = float(np.clip(_z_raw / 3.0, -1.0, 1.0))
        if abs(_z_raw) < 0.5:
            _band_lbl = 'center'
        else:
            _sign = '+' if _z_raw > 0 else '-'
            _band_lbl = f'{_sign}{abs(_band_int)}\u03c3'
        _band_ctx = BandContext(
            tf_seconds=self.tf_seconds, z_score=_z_raw, sigma=_sigma,
            center=_center, band=_band_int, band_position=_band_pos,
            at_support=(_z_raw <= -1.0), at_resistance=(_z_raw >= 1.0),
            band_label=_band_lbl,
        )

        self.current_belief = WorkerBelief(
            tf_seconds    = self.tf_seconds,
            dir_prob      = dir_prob,
            pred_mfe      = pred_mfe,
            template_id   = primary_tid,
            tf_bar_idx    = tf_bar_idx,
            conviction    = abs(dir_prob - 0.5) * 2.0,
            wave_maturity = wave_maturity,
            band_context  = _band_ctx,
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

    TIMEFRAMES_SECONDS = [14400, 3600, 1800, 900, 300, 180, 60, 30, 15, 5, 1]
    TF_WEIGHTS         = [5.0,   4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.1]
    MIN_CONVICTION     = 0.48   # skip trade if path conviction below this (physics at z=0 gives 0.50)
    MIN_ACTIVE_LEVELS  = 3      # need >=3 active TF levels for a signal
    DEFAULT_DECISION_TF = 300   # 5m: default scale at which to read predicted_mfe

    # Dynamic Exit Thresholds
    # belief_flip disabled (threshold > 1.0 = unreachable):
    # Forward pass showed 149 belief_flip exits at avg -$72.96 = -$10,872 total.
    # OOS data confirmed winners flip workers MORE than losers across all TFs,
    # meaning urgent_exit was cutting winners not protecting against losers.
    # Trail stop handles exits naturally; re-enable only with directional evidence.
    URGENT_EXIT_CONVICTION_THRESHOLD = 1.01   # effectively disabled
    # Pre-fix: tighten fired on (not is_confident) OR wave_mature > 0.65 → every bar
    # during a normal move triggered tightening, collapsing the trail to 2 ticks.
    # Now: only tighten on extreme maturity (0.85) — wave is clearly exhausting.
    # Low conviction during a trade is normal and does NOT predict reversal.
    TIGHTEN_TRAIL_WAVE_MATURITY_THRESHOLD = 0.85
    WIDEN_TRAIL_WAVE_MATURITY_THRESHOLD = 0.30
    DECAY_ALPHA_MAX  = 3.0    # initial tolerance band (wide at trade start)
    DECAY_ALPHA_MIN  = 1.0    # final tolerance band (narrow at pattern horizon)
    DECAY_THETA_EXIT = 1.5    # cascade score threshold -> trigger exit

    _TF_LABELS = {14400:'4h', 3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
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

        # ATLAS root for loading pre-built TF parquets (set by caller)
        self._atlas_root = None

        # Active-trade time-scale state (set at entry, cleared at exit)
        self._trade_avg_mfe_bar = 0.0
        self._trade_p75_mfe_bar = 0.0
        self._trade_bars_held   = 0

        # Decay cascade state (trade tracking)
        self._decay_trade_side: Optional[str] = None
        self._decay_entry_bar: int = 0
        self._decay_pattern_horizon: int = 1
        self._decay_entry_physics: Dict[int, dict] = {}  # {tf_secs: {'z','m','d'}}
        self._current_bar: int = 0
        # Trade-aware pace tracking
        self._trade_target_mfe_ticks: float = 0.0
        self._trade_resolve_bars: float = 0.0
        self._trade_entry_price: float = 0.0
        self._trade_pace_cache: Optional[dict] = None
        self._trade_pace_blend: float = 0.0

    # ------------------------------------------------------------------
    # TRADE TIME-SCALE MANAGEMENT
    # ------------------------------------------------------------------

    def set_active_trade_timescale(self, avg_mfe_bar: float, p75_mfe_bar: float):
        """
        Call at trade entry with the matched template's time-scale stats.
        avg_mfe_bar / p75_mfe_bar are in 15s bars (0-based bar index where
        MFE historically peaked).  0.0 means unknown → time signals silent.
        """
        self._trade_avg_mfe_bar = avg_mfe_bar
        self._trade_p75_mfe_bar = p75_mfe_bar
        self._trade_bars_held   = 0

    def tick_trade_bar(self):
        """Increment hold counter — call once per 15s bar while position is open."""
        self._trade_bars_held += 1

    def clear_active_trade_timescale(self):
        """Call at trade exit to reset time-scale state."""
        self._trade_avg_mfe_bar = 0.0
        self._trade_p75_mfe_bar = 0.0
        self._trade_bars_held   = 0

    # ------------------------------------------------------------------
    # DAY SETUP
    # ------------------------------------------------------------------

    def prepare_day(self, df_micro: pd.DataFrame, states_micro: list = None,
                    df_5s: pd.DataFrame = None, df_1s: pd.DataFrame = None,
                    df_4h: pd.DataFrame = None):
        """
        Task 1 for all workers: pre-aggregate the day's micro bars (15s or 1s)
        to each TF level and compute market states (once per day, fast).

        Micro states can be supplied directly (states_micro) if already computed
        by the main forward pass, avoiding redundant work.

        df_4h: external 4h bars (supra-resolution — resampling 15s gives <5 bars).
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

        # Supra-resolution workers (4h): resampling 15s→4h gives <5 bars per day.
        # Use external monthly data, same pattern as sub-resolution workers.
        _supra_res_data = {14400: df_4h}
        for tf_secs, df_ext in _supra_res_data.items():
            if tf_secs not in self.workers:
                continue
            lbl = self._TF_LABELS.get(tf_secs, str(tf_secs))
            if df_ext is None or (hasattr(df_ext, 'empty') and df_ext.empty):
                print(f"  TBN [{lbl}]: no data supplied — worker inactive")
                self.workers[tf_secs].prepare([])
                continue
            try:
                print(f"  TBN [{lbl}]: computing states for {len(df_ext):,} bars ...", end='', flush=True)
                states = self.engine.batch_compute_states(df_ext, use_cuda=True)
                self.workers[tf_secs].prepare(states)
                print(f" {len(states):,} states ready")
            except Exception as e:
                print(f" FAILED: {e}")
                logger.warning(f"TBN: TF={tf_secs}s supra-res state compute failed: {e}")
                self.workers[tf_secs].prepare([])

        for tf_secs in self.active_timeframes:
            lbl = self._TF_LABELS.get(tf_secs, str(tf_secs))
            if tf_secs == base_tf:
                if states_micro is None:
                    # Compute fresh from micro bars
                    try:
                        s = self.engine.batch_compute_states(df_micro, use_cuda=True)
                        self.workers[base_tf].prepare(s)
                        print(f"  TBN [{lbl}]: {len(s):,} states (base, computed)")
                    except Exception as e:
                        logger.warning(f"TBN: {base_tf}s state compute failed: {e}")
                        self.workers[base_tf].prepare([])
                else:
                    print(f"  TBN [{lbl}]: {len(states_micro):,} states (base, pre-computed)")
                continue

            # Skip supra-resolution TFs (already handled above)
            if tf_secs in _supra_res_data:
                continue

            try:
                # Try loading pre-built parquet first (avoids resampling)
                tf_df = None
                if hasattr(self, '_atlas_root') and self._atlas_root:
                    _tf_label = self._TF_LABELS.get(tf_secs, f'{tf_secs}s')
                    _parquet_dir = os.path.join(self._atlas_root, _tf_label)
                    if os.path.isdir(_parquet_dir):
                        import glob as _g
                        _pq_files = sorted(_g.glob(os.path.join(_parquet_dir, '*.parquet')))
                        if _pq_files:
                            # Load all months, filter to current day's time range
                            _ts_min = df_micro['timestamp'].iloc[0] if 'timestamp' in df_micro.columns else 0
                            _ts_max = df_micro['timestamp'].iloc[-1] if 'timestamp' in df_micro.columns else 0
                            _chunks = []
                            for _pf in _pq_files:
                                _ch = pd.read_parquet(_pf)
                                if 'timestamp' in _ch.columns:
                                    _ch = _ch[(_ch['timestamp'] >= _ts_min) & (_ch['timestamp'] <= _ts_max)]
                                if len(_ch) > 0:
                                    _chunks.append(_ch)
                            if _chunks:
                                tf_df = pd.concat(_chunks).sort_values('timestamp').reset_index(drop=True)

                # Fallback: resample from 15s
                if tf_df is None or len(tf_df) < 5:
                    tf_df = df.resample(f'{tf_secs}s').agg({
                        'open': 'first', 'high': 'max',
                        'low': 'min',   'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    _src = 'resampled'
                else:
                    _src = 'parquet'

                if len(tf_df) < 5:
                    self.workers[tf_secs].prepare([])
                    print(f"  TBN [{lbl}]: <5 bars -- worker inactive")
                    continue

                states = self.engine.batch_compute_states(tf_df, use_cuda=True)
                self.workers[tf_secs].prepare(states)
                print(f"  TBN [{lbl}]: {len(states):,} states ({_src} from {len(tf_df):,} bars)")
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
                           self.valid_tids, self.centroids_scaled):
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

    def update_partial(self, tf_seconds: int, bar_data: dict):
        """
        Update a worker with a partial (forming) bar from NT8.
        Computes a fresh state and re-runs analysis without advancing
        the bar index — the completed bar will still trigger normal tick().
        """
        if tf_seconds not in self.workers:
            return
        if not self.pattern_library or not self.scaler:
            return

        try:
            df = pd.DataFrame([bar_data])
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], unit='s')
            states = self.engine.batch_compute_states(df, use_cuda=True)
            if not states:
                return
            state_raw = states[-1]
            state = state_raw['state'] if isinstance(state_raw, dict) and 'state' in state_raw else state_raw
            worker = self.workers[tf_seconds]
            tf_bar_idx = max(worker._last_tf_bar_idx, 0)
            worker._analyze(state, tf_bar_idx, self.pattern_library,
                            self.scaler, self.valid_tids, self.centroids_scaled)
        except Exception as e:
            logger.debug(f"TBN partial update TF={tf_seconds}s failed: {e}")

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

        # ── Band confluence direction blend ─────────────────────────────
        # Multi-TF SE bands capture structural trend that template biases miss.
        # When bands strongly signal a direction, blend it into path probabilities
        # to prevent template SHORT bias from overriding a clear LONG trend.
        band_confluence = self.get_band_confluence()
        if band_confluence is not None and band_confluence.get('direction') is not None:
            _bc_str = band_confluence['strength']  # 0..1
            if _bc_str > 0.3:  # only blend when bands have meaningful signal
                _bc_weight = _bc_str * 0.4  # max 40% influence at full strength
                if band_confluence['direction'] == 'long':
                    path_long  = path_long  * (1 - _bc_weight) + _bc_weight * 0.75
                    path_short = path_short * (1 - _bc_weight) + _bc_weight * 0.25
                else:
                    path_long  = path_long  * (1 - _bc_weight) + _bc_weight * 0.25
                    path_short = path_short * (1 - _bc_weight) + _bc_weight * 0.75

        # ── Trade-aware direction blend ──────────────────────────────
        # When a trade is active and we know the template's expected target,
        # blend actual price progress into direction probabilities.
        # Ahead of pace → reinforce trade direction. Behind → weaken it.
        if (self._decay_trade_side is not None and
                self._trade_target_mfe_ticks > 0 and
                self._trade_entry_price > 0):
            _pace_blend = getattr(self, '_trade_pace_blend', 0.0)
            if abs(_pace_blend) > 0.05:
                _trade_is_long = (self._decay_trade_side == 'long')
                _weight = min(0.3, abs(_pace_blend) * 0.15)  # max 30% influence
                if _pace_blend > 0:  # ahead of schedule → reinforce trade dir
                    if _trade_is_long:
                        path_long  = path_long  * (1 - _weight) + _weight * 0.80
                        path_short = path_short * (1 - _weight) + _weight * 0.20
                    else:
                        path_long  = path_long  * (1 - _weight) + _weight * 0.20
                        path_short = path_short * (1 - _weight) + _weight * 0.80
                else:  # behind schedule → weaken trade dir
                    if _trade_is_long:
                        path_long  = path_long  * (1 - _weight) + _weight * 0.35
                        path_short = path_short * (1 - _weight) + _weight * 0.65
                    else:
                        path_long  = path_long  * (1 - _weight) + _weight * 0.65
                        path_short = path_short * (1 - _weight) + _weight * 0.35

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
            band_confluence        = band_confluence,
        )

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION (shared by all workers)
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_features(state, tf_secs: int, depth: int = 0) -> list:
        """
        Convert MarketState -> 16D feature vector.
        Delegates to core.feature_extraction.extract_feature_vector().
        Ancestry features are 0.0 (no parent chain for live aggregated TF bars).
        """
        from core.feature_extraction import extract_feature_vector
        return extract_feature_vector(
            z_score=getattr(state, 'z_score', 0.0),
            velocity=getattr(state, 'velocity', 0.0),
            momentum=getattr(state, 'momentum_strength', 0.0),
            entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
            tf_seconds=tf_secs,
            depth=float(depth),
            parent_is_band_reversal=0.0,
            adx=getattr(state, 'adx_strength', 0.0) / 100.0,
            hurst=getattr(state, 'hurst_exponent', 0.5),
            dmi_diff=(getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0,
            parent_z=0.0, parent_dmi_diff=0.0,
            root_is_roche=0.0, tf_alignment=0.0,
            pid=getattr(state, 'term_pid', 0.0),
            osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0),
        )

    # ------------------------------------------------------------------
    # DECAY CASCADE (trade tracking)
    # ------------------------------------------------------------------

    def start_trade_tracking(self, side: str, entry_bar: int,
                             pattern_horizon_bars: int,
                             target_mfe_ticks: float = 0.0,
                             resolve_bars: float = 0.0,
                             entry_price: float = 0.0):
        """
        Called when a trade opens. Records each worker's current z_score
        so we can track physics decay (drift away from expected trajectory).

        Trade-aware fields:
          target_mfe_ticks: template p75_mfe_ticks — expected move magnitude
          resolve_bars:     template avg_mfe_bar — expected bars to peak
          entry_price:      trade entry price for tick delta calculation
        """
        self._decay_trade_side = side
        self._decay_entry_bar = entry_bar
        self._decay_pattern_horizon = max(1, pattern_horizon_bars)
        # Trade-aware state
        self._trade_target_mfe_ticks = target_mfe_ticks
        self._trade_resolve_bars = resolve_bars
        self._trade_entry_price = entry_price
        self._decay_entry_physics = {}
        for tf, worker in self.workers.items():
            b = worker.current_belief
            if b is not None:
                self._decay_entry_physics[tf] = {
                    'z': getattr(b, 'z_score', 0.0),
                    'm': getattr(b, 'momentum', 0.0),
                    'd': b.dir_prob,
                }

    def stop_trade_tracking(self):
        """Called when a trade closes -- clears decay state."""
        self._decay_trade_side = None
        self._decay_entry_bar = 0
        self._decay_entry_physics = {}
        self._trade_target_mfe_ticks = 0.0
        self._trade_resolve_bars = 0.0
        self._trade_entry_price = 0.0
        self._trade_pace_cache = None
        self._trade_pace_blend = 0.0

    def get_trade_progress(self, current_price: float, tick_size: float = 0.25) -> dict:
        """
        Trade-aware progress: is price moving toward the template's expected target
        at the expected rate?

        Returns:
          tick_progress: actual ticks moved / target ticks (0=no move, 1=at target)
          time_progress: bars held / resolve bars (0=just entered, 1=should have peaked)
          pace:          tick_progress / max(0.1, time_progress)
                         >1 = ahead of schedule, <1 = behind, ~1 = on track
          direction_ok:  True if price is moving in the trade's direction
        """
        if (self._decay_trade_side is None or
                self._trade_target_mfe_ticks <= 0 or
                self._trade_entry_price <= 0):
            return {'tick_progress': 0.0, 'time_progress': 0.0,
                    'pace': 1.0, 'direction_ok': True}

        # Tick delta from entry
        if self._decay_trade_side == 'long':
            delta_ticks = (current_price - self._trade_entry_price) / tick_size
        else:
            delta_ticks = (self._trade_entry_price - current_price) / tick_size

        tick_progress = delta_ticks / max(1.0, self._trade_target_mfe_ticks)
        bars_held = max(1, self._current_bar - self._decay_entry_bar)
        resolve = max(1.0, self._trade_resolve_bars)
        time_progress = bars_held / resolve
        pace = tick_progress / max(0.1, time_progress)
        direction_ok = delta_ticks >= 0

        return {
            'tick_progress': round(float(tick_progress), 3),
            'time_progress': round(float(time_progress), 3),
            'pace': round(float(pace), 3),
            'direction_ok': direction_ok,
            'delta_ticks': round(float(delta_ticks), 1),
        }

    def get_decay_cascade(self) -> dict:
        """
        Compute physics decay cascade across all workers since trade entry.
        Returns cascade_score, per_worker, should_exit, progress, tau.
        """
        if self._decay_trade_side is None:
            return {'cascade_score': 0.0, 'per_worker': {},
                    'should_exit': False, 'progress': 0.0,
                    'tau': self.DECAY_ALPHA_MAX}

        dt = max(1, self._current_bar - self._decay_entry_bar)
        T_k = self._decay_pattern_horizon
        progress = min(2.0, dt / T_k)

        tau = self.DECAY_ALPHA_MAX - (self.DECAY_ALPHA_MAX - self.DECAY_ALPHA_MIN) * min(1.0, progress)

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
            current_z = getattr(b, 'z_score', 0.0)

            expected_z = entry_z * max(0.0, 1.0 - progress)
            residual = current_z - expected_z

            bad_residual = residual * sign_dir
            decay_w = max(0.0, bad_residual / (tau + 1e-9))

            tf_label = self._TF_LABELS.get(tf, str(tf))
            per_worker[tf_label] = round(decay_w, 3)

            inv_w = _max_w - self._weight_map.get(tf, 1.0) + 0.1
            cascade_num += inv_w * decay_w
            cascade_den += inv_w

        cascade = cascade_num / cascade_den if cascade_den > 0 else 0.0

        return {
            'cascade_score': round(cascade, 4),
            'per_worker': per_worker,
            'should_exit': cascade > self.DECAY_THETA_EXIT,
            'progress': round(progress, 3),
            'tau': round(tau, 3),
        }

    # ------------------------------------------------------------------
    # WORKER SNAPSHOT
    # ------------------------------------------------------------------

    def get_band_confluence(self) -> Optional[dict]:
        """Multi-TF Standard Error Band confluence for structural direction.

        Aggregates band positions across active workers:
        - Majority at support (z <= -1) -> LONG
        - Majority at resistance (z >= +1) -> SHORT
        - Mixed -> None (no signal)
        Higher TFs carry more weight (same as path conviction).
        """
        active_bands = {}
        for tf, worker in self.workers.items():
            b = worker.current_belief
            if b is not None and b.band_context is not None:
                active_bands[tf] = b.band_context

        if len(active_bands) < 3:
            return None

        support_score = 0.0
        resistance_score = 0.0
        total_weight = 0.0
        per_tf = {}
        summary_parts = []

        for tf, ctx in active_bands.items():
            w = self._weight_map.get(tf, 1.0)
            total_weight += w
            tf_label = self._TF_LABELS.get(tf, str(tf))
            per_tf[tf_label] = ctx

            if ctx.at_support:
                support_score += w * abs(ctx.z_score)
                summary_parts.append(f"{tf_label}:{ctx.band_label}")
            elif ctx.at_resistance:
                resistance_score += w * ctx.z_score
                summary_parts.append(f"{tf_label}:{ctx.band_label}")
            else:
                summary_parts.append(f"{tf_label}:center")

        if total_weight > 0:
            support_score /= total_weight
            resistance_score /= total_weight

        if support_score > resistance_score * 2 and support_score > 0.5:
            direction = 'long'
            strength = min(1.0, support_score / 3.0)
        elif resistance_score > support_score * 2 and resistance_score > 0.5:
            direction = 'short'
            strength = min(1.0, resistance_score / 3.0)
        else:
            direction = None
            strength = 0.0

        arrow = ('\u2192 LONG' if direction == 'long'
                 else ('\u2192 SHORT' if direction == 'short' else '\u2192 MIXED'))
        summary = ' | '.join(summary_parts) + f' {arrow}'

        return {
            'direction': direction,
            'strength': strength,
            'support_score': support_score,
            'resistance_score': resistance_score,
            'active_bands': len(active_bands),
            'band_summary': summary,
            'per_tf': per_tf,
        }

    def get_macro_trend(self) -> dict:
        """Macro trend from 30m+ TFs: DMI direction + band position.

        Combines DMI trend signal (60%) with band position (40%) across
        30m, 1h, 4h workers.  Higher TFs weighted more heavily.

        Returns:
            {'direction': 'long'|'short'|None, 'strength': 0.0-1.0,
             'macro_z': float, 'macro_band_pos': float, 'detail': str}
        """
        import numpy as _np
        _MACRO_TFS = {1800: 3.5, 3600: 4.0, 14400: 5.0}  # 30m, 1h, 4h

        dmi_score = 0.0
        band_score = 0.0
        z_weighted = 0.0
        bp_weighted = 0.0
        total_w = 0.0

        for tf_sec, w in _MACRO_TFS.items():
            worker = self.workers.get(tf_sec)
            if worker is None:
                continue

            # DMI component
            if worker._last_tf_bar_idx >= 0 and worker._states:
                idx = min(worker._last_tf_bar_idx, len(worker._states) - 1)
                state_raw = worker._states[idx]
                state = (state_raw['state'] if isinstance(state_raw, dict)
                         and 'state' in state_raw else state_raw)
                dmi_p = getattr(state, 'dmi_plus', 0.0)
                dmi_m = getattr(state, 'dmi_minus', 0.0)
                dmi_diff = dmi_p - dmi_m
                if abs(dmi_diff) > 5.0:  # noise floor
                    dmi_score += w * _np.sign(dmi_diff) * min(1.0, abs(dmi_diff) / 30.0)

            # Band component
            b = worker.current_belief
            if b is not None and b.band_context is not None:
                z_weighted += w * b.band_context.z_score
                bp_weighted += w * b.band_context.band_position
                total_w += w

        if total_w < 3.5:  # need at least 30m
            return {'direction': None, 'strength': 0.0, 'macro_z': 0.0,
                    'macro_band_pos': 0.0, 'detail': 'insufficient_macro'}

        macro_z = z_weighted / total_w
        macro_bp = bp_weighted / total_w

        # DMI says WHERE trend is going; band_position says WHERE price IS
        # High DMI bullish + low band_pos = strong long (price low, trend up)
        combined = (0.6 * _np.sign(dmi_score) * min(1.0, abs(dmi_score) / total_w)
                    + 0.4 * (-macro_bp))

        if combined > 0.15:
            direction = 'long'
        elif combined < -0.15:
            direction = 'short'
        else:
            direction = None

        strength = min(1.0, abs(combined))

        return {
            'direction': direction,
            'strength': strength,
            'macro_z': macro_z,
            'macro_band_pos': macro_bp,
            'detail': f'dmi={dmi_score:.2f} bp={macro_bp:.2f}',
        }

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
                entry = {
                    'd':   round(b.dir_prob,      3),
                    'c':   round(b.conviction,    3),
                    'm':   round(b.wave_maturity, 3),
                    'mfe': round(b.pred_mfe,      1),
                }
                if b.band_context is not None:
                    entry['z'] = round(b.band_context.z_score, 2)
                    entry['band'] = b.band_context.band
                    entry['band_label'] = b.band_context.band_label
                snap[self._TF_LABELS.get(tf, str(tf))] = entry
        return snap

    def get_dmi_trend(self, min_strength: float = 20.0) -> Optional[dict]:
        """Scan TFs bottom-up for a strong DMI trend.

        Returns the first TF where |dmi_plus - dmi_minus| > min_strength
        and the dominant side exceeds the other. If no TF qualifies, returns None.

        Returns:
            {'direction': 'long'|'short', 'strength': float,
             'tf': str, 'dmi_plus': float, 'dmi_minus': float}
        """
        # Workers are keyed by tf_seconds — sort ascending (low TF first)
        for tf_sec in sorted(self.workers.keys()):
            worker = self.workers[tf_sec]
            if worker._last_tf_bar_idx < 0 or not worker._states:
                continue
            idx = min(worker._last_tf_bar_idx, len(worker._states) - 1)
            state_raw = worker._states[idx]
            state = (state_raw['state'] if isinstance(state_raw, dict)
                     and 'state' in state_raw else state_raw)
            dmi_p = getattr(state, 'dmi_plus', 0.0)
            dmi_m = getattr(state, 'dmi_minus', 0.0)
            diff = dmi_p - dmi_m
            if abs(diff) > min_strength and max(dmi_p, dmi_m) > min(dmi_p, dmi_m):
                return {
                    'direction': 'long' if diff > 0 else 'short',
                    'strength': abs(diff),
                    'tf': self._TF_LABELS.get(tf_sec, str(tf_sec)),
                    'dmi_plus': dmi_p,
                    'dmi_minus': dmi_m,
                }
        return None

    def get_dmi_alignment(self) -> dict:
        """Check if DMI direction agrees across timeframes.

        Returns dict with:
            alignment_score: -1.0 to +1.0 (negative=bearish, positive=bullish consensus)
            aligned_tfs: count of TFs agreeing with majority direction
            total_tfs: total TFs with valid DMI data
        """
        scores = []
        for tf_sec in sorted(self.workers.keys()):
            worker = self.workers[tf_sec]
            idx = worker._last_tf_bar_idx
            if idx < 0 or not worker._states or idx >= len(worker._states):
                continue
            raw = worker._states[idx]
            state = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
            dmi_p = getattr(state, 'dmi_plus', 0.0)
            dmi_m = getattr(state, 'dmi_minus', 0.0)
            diff = dmi_p - dmi_m
            if abs(diff) > 1.0:  # ignore noise
                scores.append(1.0 if diff > 0 else -1.0)

        if not scores:
            return {'alignment_score': 0.0, 'aligned_tfs': 0, 'total_tfs': 0}

        avg = sum(scores) / len(scores)
        # Majority direction
        majority_bull = sum(1 for s in scores if s > 0) > len(scores) / 2
        aligned = sum(1 for s in scores if (s > 0) == majority_bull)
        return {
            'alignment_score': avg,
            'aligned_tfs': aligned,
            'total_tfs': len(scores),
        }

    def get_worker_state_counts(self) -> dict:
        """Return {tf_label: n_states} for every worker. 0 = worker has no data."""
        return {
            self._TF_LABELS.get(tf, str(tf)): len(w._states)
            for tf, w in self.workers.items()
        }

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    # TF hierarchy for adjacent-higher lookup
    _TF_HIERARCHY = [1, 5, 15, 30, 60, 120, 180, 300, 900, 1800, 3600, 14400, 86400]

    def get_exit_signal(self, side: str, entry_price: float = 0.0,
                        discovery_tf_seconds: float = 300.0) -> dict:
        """
        Called every bar while a position is open.
        Returns a dict with exit adjustment recommendations.

        side: 'long' or 'short' — the current position direction.
        entry_price: trade entry price (enables band-aware exit logic).

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

        # ── Band-aware exit adjustments ─────────────────────────────────
        # Use multi-TF band confluence to modulate trail behavior:
        # - In profit + approaching resistance (LONG) or support (SHORT) → tighten
        # - In loss + sitting at support (LONG) or resistance (SHORT) → widen (give room)
        # - In loss + support/resistance BROKEN → urgent exit
        _band_tighten = False
        _band_widen = False
        _band_urgent = False
        _bc = belief.band_confluence
        if _bc is not None and entry_price > 0:
            _sup = _bc['support_score']
            _res = _bc['resistance_score']
            _in_profit = True  # approximate from band position vs trade side
            # For LONG: profit when price above entry → resistance bands = approaching TP zone
            # For SHORT: profit when price below entry → support bands = approaching TP zone
            if side == 'long':
                # Approaching resistance while long & in profit → tighten
                if _res > 0.5:
                    _band_tighten = True
                # Sitting at support while long → give room (price at value)
                if _sup > 0.5 and direction_aligned:
                    _band_widen = True
                # Support broken while long (bands say SHORT strongly) → urgent
                if _bc['direction'] == 'short' and _bc['strength'] > 0.5:
                    _band_urgent = True
            else:  # short
                # Approaching support while short & in profit → tighten
                if _sup > 0.5:
                    _band_tighten = True
                # Sitting at resistance while short → give room
                if _res > 0.5 and direction_aligned:
                    _band_widen = True
                # Resistance broken while short (bands say LONG strongly) → urgent
                if _bc['direction'] == 'long' and _bc['strength'] > 0.5:
                    _band_urgent = True

        # Exit trend guard: when band confluence confirms trade direction with
        # strong signal, suppress band_tighten. Fast TFs hitting local resistance
        # shouldn't spook us out of a trade the slow TFs still support.
        # See: reports/findings/2026-03-07_brain_aggregation.md (Fix #2)
        if _band_tighten and _bc is not None and _bc.get('direction') is not None:
            _bc_matches_trade = (
                (_bc['direction'] == 'long' and side == 'long') or
                (_bc['direction'] == 'short' and side == 'short')
            )
            if _bc_matches_trade and _bc['strength'] > 0.5:
                _band_tighten = False  # trend is with us, don't tighten

        tighten = tighten or _band_tighten
        widen   = widen or _band_widen
        urgent  = urgent or _band_urgent

        # ── 30m worker flip detection ─────────────────────────────────────
        # When the 30m (slow) worker flips direction against the trade, the
        # structural trend has changed. Tighten giveback threshold by 15pp.
        _slow_flip_tighten = False
        _30m_belief = belief.tf_beliefs.get(1800)
        if _30m_belief is not None:
            if side == 'long' and _30m_belief.dir_prob < 0.45:
                _slow_flip_tighten = True
            elif side == 'short' and _30m_belief.dir_prob > 0.55:
                _slow_flip_tighten = True

        # ── 15s execution-TF flip detection ──────────────────────────
        # OOS data shows 66% of losses had the 15s worker flip mid-trade
        # vs 44% of wins (+22% edge). When the execution TF flips against
        # the trade, the micro structure has reversed.
        _exec_flip = False
        _15s_belief = belief.tf_beliefs.get(15)
        if _15s_belief is not None:
            if side == 'long' and _15s_belief.dir_prob < 0.40:
                _exec_flip = True
            elif side == 'short' and _15s_belief.dir_prob > 0.60:
                _exec_flip = True

        # ── Trade pace check ──────────────────────────────────────────
        # If we know the template's expected target and resolve time,
        # check if price is on track. Behind pace + past halfway = tighten.
        _pace_tighten = False
        _pace_widen = False
        _tp = getattr(self, '_trade_pace_cache', None)
        if _tp is not None and _tp.get('time_progress', 0) > 0.3:
            if _tp['pace'] < 0.3 and not _tp['direction_ok']:
                # Way behind AND wrong direction → tighten aggressively
                _pace_tighten = True
            elif _tp['pace'] > 1.5 and _tp['direction_ok']:
                # Ahead of schedule + correct direction → widen, let it run
                _pace_widen = True

        tighten = tighten or _pace_tighten
        widen   = widen or _pace_widen

        reason = ('band_broken'    if _band_urgent   else
                  'time_exhausted' if _time_urgent    else
                  'urgent_flip'    if urgent           else
                  'exec_tf_flip'   if _exec_flip      else
                  'pace_behind'    if _pace_tighten   else
                  'slow_flip'      if _slow_flip_tighten else
                  'band_tighten'   if _band_tighten   else
                  'time_tighten'   if _time_tighten   else
                  'wave_mature'    if wave_mature > self.TIGHTEN_TRAIL_WAVE_MATURITY_THRESHOLD else
                  'pace_ahead'     if _pace_widen     else
                  'band_widen'     if _band_widen     else
                  'aligned_fresh'  if widen            else
                  'low_conviction' if not belief.is_confident else 'neutral')

        # ── Trade health: fused pace + structure decay ──────────────────
        _decay = self.get_decay_cascade()
        _cascade = _decay['cascade_score']  # 0..1+ (higher = worse)
        _pace_val = _tp.get('pace', 1.0) if _tp is not None else 1.0
        _pace_health = min(1.0, max(0.0, _pace_val))  # clamp 0-1
        _decay_health = max(0.0, 1.0 - _cascade)      # invert: 0=bad, 1=good
        _trade_health = 0.6 * _pace_health + 0.4 * _decay_health

        # ── TF-aware DMI/ADX for exit modules ─────────────────────────
        # Uses discovery TF for exit signal + adjacent higher TF for hold override.
        # Research: 5m DMI crossover 87% accurate (gap>=5), 3-bar confirmation
        # drops MAE 40%. Each trade should check its OWN TF, not hardcoded 5m.
        _disc_tf = int(discovery_tf_seconds)

        # Find adjacent higher TF for hold override
        _higher_tf = _disc_tf  # fallback to same
        for _tf in self._TF_HIERARCHY:
            if _tf > _disc_tf:
                _higher_tf = _tf
                break

        def _read_worker_dmi(tf_seconds):
            """Read DMI/ADX from a specific TF worker."""
            w = self.workers.get(tf_seconds)
            if w is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            mi = w._last_tf_bar_idx
            if mi < 0 or not w._states or mi >= len(w._states):
                return 0.0, 0.0, 0.0, 0.0, 0.0
            raw = w._states[mi]
            ms = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
            dp = getattr(ms, 'dmi_plus', 0.0)
            dm = getattr(ms, 'dmi_minus', 0.0)
            dp_prev = getattr(ms, 'di_plus_prev', dp)
            dm_prev = getattr(ms, 'di_minus_prev', dm)
            adx_now = getattr(ms, 'adx_strength', 0.0)
            adx_prev = getattr(ms, 'adx_prev', adx_now)
            return dp, dm, dp_prev, dm_prev, adx_now - adx_prev

        # Discovery TF DMI (exit signal source)
        _di_plus, _di_minus, _di_plus_prev, _di_minus_prev, _adx_slope = \
            _read_worker_dmi(_disc_tf)

        # Adjacent higher TF DMI (hold override — is macro trend alive?)
        _h_di_plus, _h_di_minus, _, _, _ = _read_worker_dmi(_higher_tf)
        _higher_tf_agrees = (
            (side == 'long' and _h_di_plus > _h_di_minus) or
            (side == 'short' and _h_di_minus > _h_di_plus)
        )

        # ── Sensor fusion: 1s velocity (fast) + 1m volume (slow/accurate) ──
        # 1s detects inflection instantly, 1m confirms via volume collapse
        _1s_vel = 0.0
        _1s_vel_prev = 0.0
        _1s_w = self.workers.get(1)
        if _1s_w is not None and _1s_w._states:
            _mi = _1s_w._last_tf_bar_idx
            if 0 <= _mi < len(_1s_w._states):
                _raw = _1s_w._states[_mi]
                _ms = _raw['state'] if isinstance(_raw, dict) and 'state' in _raw else _raw
                _1s_vel = getattr(_ms, 'velocity', 0.0)
            if 0 <= _mi - 1 < len(_1s_w._states):
                _raw_p = _1s_w._states[_mi - 1]
                _ms_p = _raw_p['state'] if isinstance(_raw_p, dict) and 'state' in _raw_p else _raw_p
                _1s_vel_prev = getattr(_ms_p, 'velocity', 0.0)

        _1m_vol = 0.0
        _1m_vol_prev = 0.0
        _1m_w = self.workers.get(60)
        if _1m_w is not None and _1m_w._states:
            _mi = _1m_w._last_tf_bar_idx
            if 0 <= _mi < len(_1m_w._states):
                _raw = _1m_w._states[_mi]
                _ms = _raw['state'] if isinstance(_raw, dict) and 'state' in _raw else _raw
                _1m_vol = getattr(_ms, 'volume_delta', 0.0)
            if 0 <= _mi - 1 < len(_1m_w._states):
                _raw_p = _1m_w._states[_mi - 1]
                _ms_p = _raw_p['state'] if isinstance(_raw_p, dict) and 'state' in _raw_p else _raw_p
                _1m_vol_prev = getattr(_ms_p, 'volume_delta', 0.0)

        # Velocity flip: 1s velocity changed sign
        _vel_flipped = (_1s_vel * _1s_vel_prev < 0) and abs(_1s_vel_prev) > 0.5

        # Volume collapse: 1m volume dropped >50% from previous bar
        _vol_collapsing = (_1m_vol_prev > 0 and _1m_vol < _1m_vol_prev * 0.5)

        return {
            'tighten_trail': tighten and not urgent,
            'widen_trail':   widen and not _time_tighten and not _band_tighten,
            'urgent_exit':   urgent,
            'conviction':    belief.conviction,
            'wave_maturity': wave_mature,
            'slow_flip_tighten': _slow_flip_tighten,
            'exec_tf_flip': _exec_flip,
            'reason':        reason,
            'trade_health':  round(_trade_health, 3),
            'pace':          round(_pace_val, 3),
            # TF-aware DMI/ADX for exit modules (discovery TF + adjacent higher TF)
            'higher_tf_agrees': _higher_tf_agrees,
            'di_plus':       _di_plus,
            'di_minus':      _di_minus,
            'di_plus_prev':  _di_plus_prev,
            'di_minus_prev': _di_minus_prev,
            'adx_slope':     _adx_slope,
            # Sensor fusion: fast (1s) + slow (1m)
            'vel_flipped':     _vel_flipped,      # 1s velocity changed sign
            'vol_collapsing':  _vol_collapsing,    # 1m volume dropped >50%
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
