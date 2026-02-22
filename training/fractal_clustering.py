"""
Fractal Clustering Engine
Reduces massive pattern datasets into manageable 'Templates' using Recursive K-Means.
Maps raw physics events into "Archetypal Centroids" that are both Physically Tight and Behaviorally Consistent.

Feature vector per pattern (16D):
  [|z_score|, |velocity|, |momentum|, coherence, log2(tf_seconds), depth, parent_is_roche,
   self_adx, self_hurst, self_dmi_diff, parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
   self_pid, self_osc_coh]
The timeframe scale + depth + parent context + PID regime let the clustering naturally separate
patterns that look similar in physics but live at different fractal scales or regimes.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from training.cuda_kmeans import CUDAKMeans, cuda_silhouette_score

# Local import of timeframe mapping
from training.fractal_discovery_agent import TIMEFRAME_SECONDS
from config.oracle_config import (
    TEMPLATE_MIN_MEMBERS_FOR_STATS,
    MARKER_MEGA_LONG, MARKER_SCALP_LONG,
    MARKER_SCALP_SHORT, MARKER_MEGA_SHORT, MARKER_NOISE,
    TRANSITION_MIN_SEQUENCE_GAP_BARS,
    TRANSITION_MAX_SEQUENCE_GAP_BARS
)

# Clustering Constants
MIN_PATTERNS_FOR_SPLIT  = 30   # hard floor — never split below this
MIN_SAMPLES_PER_CLUSTER = 30   # same floor used in _fit_branch coarse pass
MAX_FISSION_CLUSTERS    = 6
MIN_FISSION_SAMPLES     = 5

# Adj-R² thresholds
R2_STOP_THRESHOLD   = 0.15   # stop recursive splitting when adj-R²(mfe~features) >= this
R2_FISSION_MIN_GAIN = 0.05   # minimum weighted adj-R² gain required to allow fission

@dataclass
class PatternTemplate:
    template_id: int
    centroid: np.ndarray  # [z, vel, mom, coh, tf_scale, depth, parent_ctx, p_z, p_mom, root_z, root_is_roche]
    member_count: int
    patterns: List[Any]   # References to the original PatternEvents
    physics_variance: float # Measure of how "tight" the cluster is

    # NAVIGATION & RISK DATA
    transition_map: Dict[int, float] = field(default_factory=dict)   # {Next_Cluster_ID: Probability}
    transition_probs: Dict[int, float] = field(default_factory=dict) # Alias for backward compatibility

    # REWARD
    expected_value: float = 0.0               # (WinRate * AvgWin) - (LossRate * AvgLoss)

    # RISK
    outcome_variance: float = 0.0             # StdDev of PnL outcomes
    avg_drawdown: float = 0.0                 # Average Maximum Adverse Excursion (MAE)
    risk_score: float = 0.0                   # 0.0 (Safe) to 1.0 (Toxic)
    risk_variance: float = 0.0              # StdDev of member MFE values

    # THE STAR SCHEMA DIMENSION (Aggregated from member oracle markers)
    # 1. Performance Stats
    stats_win_rate: float = 0.0             # Fraction of members with |oracle_marker| >= 1
    stats_expectancy: float = 0.0           # Mean (mfe - mae) across members
    stats_mega_rate: float = 0.0            # Fraction of members with |oracle_marker| == 2

    # 3. Direction Bias
    long_bias: float = 0.0                  # Fraction of positive markers (1,2) vs total non-noise
    short_bias: float = 0.0                 # Fraction of negative markers (-1,-2) vs total non-noise

    parent_cluster_id: int = None             # The "Macro" state this belongs to

    direction: str = ''   # 'LONG' or 'SHORT' — set during fit()

    # ORACLE EXIT CALIBRATION (in ticks; populated by _aggregate_oracle_intelligence)
    # Used by workers to anchor TP/SL to what this pattern historically achieves.
    mean_mfe_ticks: float = 0.0   # Mean max-favorable-excursion seen across members
    mean_mae_ticks: float = 0.0   # Mean max-adverse-excursion seen across members
    p75_mfe_ticks:  float = 0.0   # 75th-pct MFE — conservative TP ceiling
    p25_mae_ticks:  float = 0.0   # 25th-pct MAE — tight SL floor
    regression_sigma_ticks: float = 0.0  # Residual std from per-cluster OLS; trail = this * 1.1

    # TIME-SCALE CALIBRATION (populated by _aggregate_oracle_intelligence, requires --fresh)
    avg_mfe_bar: float = 0.0   # mean bar index (15s bars, 0-based) where MFE peaked
    p75_mfe_bar: float = 0.0   # 75th-pct mfe_bar — conservative "still moving" window

    # PER-CLUSTER REGRESSION MODELS (fitted on 14D scaled feature vectors of members)
    # MFE model: predicted_mfe = live_features @ mfe_coeff + mfe_intercept
    # Dir model: P(LONG) = sigmoid(live_features @ dir_coeff + dir_intercept)
    # Both None when cluster has too few members for reliable fitting.
    mfe_coeff:     Optional[List[float]] = field(default=None)  # 14 OLS weights
    mfe_intercept: float = 0.0
    dir_coeff:     Optional[List[float]] = field(default=None)  # 14 logistic weights
    dir_intercept: float = 0.0

class FractalClusteringEngine:
    def __init__(self, n_clusters=1000, max_variance=0.5):
        self.n_clusters = n_clusters
        self.max_variance = max_variance  # Max allowed std deviation for Z-score in a cluster
        self.scaler = StandardScaler()
        self._long_scaler = StandardScaler()
        self._short_scaler = StandardScaler()

    def _get_kmeans_model(self, n_clusters: int, n_samples: int, random_state: int = 42,
                          n_init: int = 3, use_cuda: bool = True):
        """Returns a KMeans model -- CUDA for main process, sklearn CPU for workers."""
        if use_cuda:
            return CUDAKMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state,
                      n_init=n_init, max_iter=300)

    @staticmethod
    def extract_features(p: Any) -> List[float]:
        """
        Extracts 16D feature vector from a PatternEvent.
        [7 base] + [3 self regime] + [2 PID] + [4 ancestry]

        velocity and momentum use log1p(|x|) compression so that
        high-timeframe bars (where volume * velocity blows up) remain
        comparable to 15s bars.  The scaler then standardizes the
        log-compressed values across the full training set.
        """
        z = getattr(p, 'z_score', 0.0)
        v = getattr(p, 'velocity', 0.0)
        m = getattr(p, 'momentum', 0.0)
        c = getattr(p, 'coherence', 0.0)
        # log1p compression keeps extreme TF values finite
        v_feat = np.log1p(abs(v))
        m_feat = np.log1p(abs(m))

        # Fractal hierarchy features
        tf = getattr(p, 'timeframe', '15s')
        tf_secs = TIMEFRAME_SECONDS.get(tf, 15)
        tf_scale = np.log2(max(1, tf_secs))  # log2 for even spacing

        depth = float(getattr(p, 'depth', 0))
        parent_type = getattr(p, 'parent_type', '')
        parent_ctx = 1.0 if parent_type == 'ROCHE_SNAP' else 0.0

        # Self Regime features
        state = getattr(p, 'state', None)
        if state:
             self_adx = getattr(state, 'adx_strength', 0.0) / 100.0
             self_hurst = getattr(state, 'hurst_exponent', 0.5)
             self_dmi_diff = (getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0
             self_pid       = getattr(state, 'term_pid', 0.0)
             self_osc_coh   = getattr(state, 'oscillation_coherence', 0.0)
        else:
             self_adx = 0.0
             self_hurst = 0.5
             self_dmi_diff = 0.0
             self_pid = 0.0
             self_osc_coh = 0.0

        # Ancestry features
        chain = getattr(p, 'parent_chain', None) or []
        if chain:
            # Immediate parent
            parent_z = abs(chain[0].get('z', 0.0))
            parent_dmi_diff = (chain[0].get('dmi_plus', 0.0) - chain[0].get('dmi_minus', 0.0)) / 100.0

            # Root ancestor
            root = chain[-1]
            root_is_roche = 1.0 if root.get('type') == 'ROCHE_SNAP' else 0.0
            root_dmi_diff = (root.get('dmi_plus', 0.0) - root.get('dmi_minus', 0.0)) / 100.0

            # TF Alignment
            self_dir = 1.0 if self_dmi_diff > 0 else -1.0
            root_dir = 1.0 if root_dmi_diff > 0 else -1.0
            tf_alignment = self_dir * root_dir
        else:
            parent_z = 0.0
            parent_dmi_diff = 0.0
            root_is_roche = 0.0
            tf_alignment = 0.0

        return [abs(z), v_feat, m_feat, c, tf_scale, depth, parent_ctx,
                self_adx, self_hurst, self_dmi_diff,
                parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
                self_pid, self_osc_coh]

    @staticmethod
    def _shape_label(p) -> str:
        """
        Discrete shape taxonomy for initial grouping before any K-means split.

        Encodes fractal hierarchy position + physical regime so that patterns
        sharing the same physics context are grouped before quantitative splitting.
        A depth-5 ROCHE_SNAP at L3_ROCHE trending will never share a cluster
        with a depth-2 STRUCTURAL_DRIVE at L1_STABLE mean-reverting.

        Returns a key like: "d5|ROCHE_SNAP|L3_ROCHE|trend"
        """
        depth = int(getattr(p, 'depth', 0))
        ptype = getattr(p, 'pattern_type', 'UNKNOWN')
        state = getattr(p, 'state', None)
        lzone = getattr(state, 'lagrange_zone', 'UNKNOWN') if state else 'UNKNOWN'
        hurst = getattr(state, 'hurst_exponent', 0.5) if state else 0.5
        hcat  = 'trend' if hurst > 0.6 else ('revert' if hurst < 0.4 else 'random')
        return f"d{depth}|{ptype}|{lzone}|{hcat}"

    def _compute_adj_r2(self, patterns: list, scaler) -> float:
        """
        Adjusted R² of oracle_mfe ~ 16D feature vector for a group of patterns.

        Returns -1.0 when too few patterns to fit reliably (n <= k+2 = 18).
        The adjustment penalty (n-1)/(n-k-1) dominates when n is small relative
        to k=16 features, so small clusters naturally score low and never
        qualify for further splitting — the primary anti-overfitting mechanism.
        Returns 1.0 when MFE variance is near-zero (cluster is perfectly coherent).
        """
        pairs = [
            (self.extract_features(p), p.oracle_meta.get('mfe'))
            for p in patterns
            if getattr(p, 'oracle_meta', None) is not None
        ]
        pairs = [(f, y) for f, y in pairs if y is not None]
        n, k = len(pairs), 16
        if n <= k + 2:
            return -1.0
        X = scaler.transform(np.array([f for f, _ in pairs]))
        y = np.array([m for _, m in pairs])
        if np.std(y) < 1e-9:
            return 1.0
        ols = LinearRegression().fit(X, y)
        ss_res = float(np.sum((y - ols.predict(X)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
        return float(adj_r2)

    def _recursive_split(self, X: np.ndarray, patterns: list, start_id: int, scaler, depth: int = 0) -> list:
        """
        Recursively split a cluster guided by adj-R²(oracle_mfe ~ features).

        Stops when:
          1. Hard floor: fewer than MIN_PATTERNS_FOR_SPLIT patterns
          2. Coherence met: adj-R² >= R2_STOP_THRESHOLD (features explain MFE)
          3. Cannot split: only one unique point in feature space

        The adj-R² penalty is large when n is small relative to k=16 features,
        so small clusters naturally score low and stop splitting without a
        depth limit — no arbitrary MAX_RECURSION_DEPTH needed.
        """
        def _make(tid, X_sub, pats):
            c = np.mean(X_sub, axis=0)
            return PatternTemplate(
                template_id=tid, centroid=scaler.inverse_transform([c])[0],
                member_count=len(pats), patterns=pats,
                physics_variance=float(np.std(X_sub[:, 0]))
            )

        # 1. Hard floor
        if len(patterns) <= MIN_PATTERNS_FOR_SPLIT:
            return [_make(start_id, X, patterns)]

        # 2. Coherence check — stop if adj-R² already good enough
        if self._compute_adj_r2(patterns, scaler) >= R2_STOP_THRESHOLD:
            return [_make(start_id, X, patterns)]

        # 3. Unique-point guard
        n_unique = len(np.unique(X, axis=0))
        k = min(3, max(2, len(patterns) // MIN_PATTERNS_FOR_SPLIT))
        k = min(k, n_unique)
        if k <= 1:
            return [_make(start_id, X, patterns)]

        km = self._get_kmeans_model(n_clusters=k, n_samples=len(X))
        labels = km.fit_predict(X)

        result = []
        nid = start_id
        for lbl in range(k):
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            sub_X = X[mask]
            sub_p = [patterns[i] for i in np.where(mask)[0]]
            children = self._recursive_split(sub_X, sub_p, nid, scaler, depth + 1)
            result.extend(children)
            nid += len(children)
        return result

    def _aggregate_oracle_intelligence(self, template, patterns, scaler: StandardScaler):
        """
        Post-clustering: compute template-level stats from member oracle markers.
        Called AFTER clustering is complete. Does NOT influence cluster assignment.
        """
        markers = [p.oracle_marker for p in patterns if hasattr(p, 'oracle_marker')]

        if len(markers) < TEMPLATE_MIN_MEMBERS_FOR_STATS:
            return  # Not enough data

        # 1. Win Rate (any non-noise outcome)
        wins = sum(1 for m in markers if abs(m) >= 1)
        template.stats_win_rate = wins / len(markers)

        # 2. Mega Rate (home runs)
        megas = sum(1 for m in markers if abs(m) == 2)
        template.stats_mega_rate = megas / len(markers)

        # 3. Expectancy (mean MFE - MAE from oracle_meta)
        mfe_values = []
        mae_values = []
        for p in patterns:
            meta = getattr(p, 'oracle_meta', {})
            if 'mfe' in meta and 'mae' in meta:
                mfe_values.append(meta['mfe'])
                mae_values.append(meta['mae'])

        if mfe_values:
            template.stats_expectancy = np.mean(mfe_values) - np.mean(mae_values)
            template.risk_variance = float(np.std(mfe_values))

            # Oracle exit calibration: convert price-points -> ticks (MNQ: 1 tick = 0.25 pts)
            _tick = 0.25
            mfe_ticks = np.array(mfe_values) / _tick
            mae_ticks = np.array(mae_values) / _tick
            template.mean_mfe_ticks = float(np.mean(mfe_ticks))
            template.mean_mae_ticks = float(np.mean(mae_ticks))
            template.p75_mfe_ticks  = float(np.percentile(mfe_ticks, 75))
            template.p25_mae_ticks  = float(np.percentile(mae_ticks, 25))

            # ---------------------------------------------------------------
            # PER-CLUSTER REGRESSION MODELS (14D feature space)
            # ---------------------------------------------------------------
            # Build aligned (features, mfe) pairs for members that have oracle data
            feat_mfe_pairs = [
                (self.extract_features(p), p.oracle_meta['mfe'])
                for p in patterns
                if getattr(p, 'oracle_meta', {}).get('mfe') is not None
            ]
            _MIN_REG = 15  # need at least 15 members for stable regression

            if len(feat_mfe_pairs) >= _MIN_REG:
                raw_X  = np.array([f for f, _ in feat_mfe_pairs])
                mfe_y  = np.array([m for _, m in feat_mfe_pairs])

                # Scale features using the engine's fitted scaler
                X_scaled = scaler.transform(raw_X)

                # --- OLS: predicted_mfe = X @ mfe_coeff + mfe_intercept ---
                ols = LinearRegression().fit(X_scaled, mfe_y)
                template.mfe_coeff     = ols.coef_.tolist()
                template.mfe_intercept = float(ols.intercept_)
                residuals = mfe_y - ols.predict(X_scaled)
                template.regression_sigma_ticks = float(np.std(residuals) / _tick)

                # --- Logistic: P(LONG) = sigmoid(X @ dir_coeff + dir_intercept) ---
                # Use only non-noise members; label +1=LONG, 0=SHORT
                dir_pairs = [
                    (self.extract_features(p), 1 if p.oracle_marker > 0 else 0)
                    for p in patterns
                    if getattr(p, 'oracle_marker', 0) != 0
                    and getattr(p, 'oracle_meta', {}).get('mfe') is not None
                ]
                if len(dir_pairs) >= _MIN_REG:
                    dir_raw = np.array([self.extract_features(p)
                                        for p in patterns
                                        if getattr(p, 'oracle_marker', 0) != 0
                                        and getattr(p, 'oracle_meta', {}).get('mfe') is not None])
                    dir_labels = np.array([1 if p.oracle_marker > 0 else 0
                                           for p in patterns
                                           if getattr(p, 'oracle_marker', 0) != 0
                                           and getattr(p, 'oracle_meta', {}).get('mfe') is not None])
                    # Only fit if both classes present
                    if len(np.unique(dir_labels)) == 2:
                        dir_X = scaler.transform(dir_raw)
                        lr = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs').fit(dir_X, dir_labels)
                        template.dir_coeff     = lr.coef_[0].tolist()
                        template.dir_intercept = float(lr.intercept_[0])
            else:
                template.regression_sigma_ticks = template.mean_mae_ticks  # fallback

        # 4. Risk Score (0 = safe, 1 = toxic)
        # High variance + low win rate = toxic
        if template.stats_win_rate > 0:
            # Coefficient of variation normalized to [0,1]
            cv = template.risk_variance / (np.mean(mfe_values) + 1e-9) if mfe_values else 1.0
            template.risk_score = min(1.0, cv * (1.0 - template.stats_win_rate))
        else:
            template.risk_score = 1.0

        # 5. Direction Bias
        non_noise = [m for m in markers if m != MARKER_NOISE]
        if non_noise:
            longs = sum(1 for m in non_noise if m > 0)
            shorts = sum(1 for m in non_noise if m < 0)
            total_nn = len(non_noise)
            template.long_bias = longs / total_nn
            template.short_bias = shorts / total_nn

        # 6. Time-scale: bar index where MFE peaked (requires mfe_bar in oracle_meta)
        mfe_bars = [
            p.oracle_meta.get('mfe_bar')
            for p in patterns
            if getattr(p, 'oracle_meta', None) is not None
            and p.oracle_meta.get('mfe_bar', -1) >= 0
        ]
        if len(mfe_bars) >= TEMPLATE_MIN_MEMBERS_FOR_STATS:
            template.avg_mfe_bar = float(np.mean(mfe_bars))
            template.p75_mfe_bar = float(np.percentile(mfe_bars, 75))

    def _build_transition_matrix(self, templates: List[PatternTemplate], all_patterns: List[Any]):
        """
        For each template, count how often its members are followed by
        members of other templates (sorted by time).

        This creates a Markov transition map: P(next_template | current_template).
        """
        # Sort all patterns globally by timestamp
        sorted_patterns = sorted(all_patterns, key=lambda p: p.timestamp)

        # Build pattern -> template_id lookup
        pattern_to_template = {}
        for template in templates:
            for p in template.patterns:
                pattern_to_template[id(p)] = template.template_id

        # Count transitions
        for i in range(len(sorted_patterns) - 1):
            curr = sorted_patterns[i]
            curr_tid = pattern_to_template.get(id(curr))
            if curr_tid is None:
                continue

            # Find next pattern within gap window
            for j in range(i + 1, len(sorted_patterns)):
                nxt = sorted_patterns[j]
                time_diff = nxt.timestamp - curr.timestamp
                gap_bars = time_diff / 15.0 # Assuming 15s base

                if gap_bars < TRANSITION_MIN_SEQUENCE_GAP_BARS:
                    continue
                if gap_bars > TRANSITION_MAX_SEQUENCE_GAP_BARS:
                    break

                nxt_tid = pattern_to_template.get(id(nxt))
                if nxt_tid is not None and nxt_tid != curr_tid:
                    # Record transition
                    template_map = next(t for t in templates if t.template_id == curr_tid)
                    if nxt_tid not in template_map.transition_map:
                        template_map.transition_map[nxt_tid] = 0
                    template_map.transition_map[nxt_tid] += 1
                    break  # Only count first transition

        # Normalize to probabilities
        for template in templates:
            total = sum(template.transition_map.values())
            if total > 0:
                template.transition_map = {
                    k: v / total for k, v in template.transition_map.items()
                }
                # Sync alias
                template.transition_probs = template.transition_map

    def create_templates(self, manifest: List[Any]) -> List[PatternTemplate]:
        """
        Shape-first clustering — no LONG/SHORT pre-split.

        The snowflake split is removed: direction separation emerges naturally
        from the shape taxonomy (depth × pattern_type × lagrange_zone × hurst_cat).
        Templates store long_bias/short_bias from _aggregate_oracle_intelligence
        so the forward pass can still gate direction without needing separate libraries.
        """
        # Exclude pure noise patterns (oracle_marker == 0 have no MFE signal for adj-R²)
        active = [p for p in manifest if getattr(p, 'oracle_marker', 0) != 0]
        print(f"Shape Clustering: {len(active)} active patterns ({len(manifest)-len(active)} noise excluded)")

        self.scaler, templates = self._fit_branch(active, 'ALL')

        # Build Transition Matrix on full merged template set
        valid = [p for p in active if p is not None]
        if valid and templates:
            print(f"  Building Transition Matrix...", end="", flush=True)
            self._build_transition_matrix(templates, valid)
            print(" done.")

        self.templates = templates
        return templates

    def _fit_branch(self, patterns: List[Any], direction: str):
        """
        Shape-first clustering for a set of patterns.

        Stage 1: Group by shape taxonomy (depth × pattern_type × lagrange_zone × hurst_cat).
                 Each group represents a geometrically distinct market situation.
        Stage 2: Within each group, recursively split using adj-R²(mfe ~ features).
                 Only split when features do not yet explain MFE variance well enough.

        Returns (scaler, list[PatternTemplate]).
        """
        import time as _time
        from collections import defaultdict
        from sklearn.preprocessing import StandardScaler

        if not patterns:
            return StandardScaler(), []

        print(f"\n--- Shape Clustering: {len(patterns)} patterns ---")
        t0 = _time.perf_counter()

        # 1. Extract features and fit a single scaler on all patterns
        features, valid_patterns = [], []
        for p in patterns:
            try:
                features.append(self.extract_features(p))
                valid_patterns.append(p)
            except AttributeError:
                continue

        if not features:
            return StandardScaler(), []

        X_all = np.array(features)
        scaler = StandardScaler()
        scaler.fit(X_all)
        print(f"  Feature matrix: {X_all.shape[0]} × {X_all.shape[1]}")

        # 2. Group by shape taxonomy
        shape_groups = defaultdict(list)
        for p in valid_patterns:
            shape_groups[self._shape_label(p)].append(p)

        print(f"  Shape groups: {len(shape_groups)}")
        for key in sorted(shape_groups):
            print(f"    {key}: {len(shape_groups[key])} patterns")

        # 3. Per-group adj-R² recursive split
        t2 = _time.perf_counter()
        next_id, final_templates = 0, []

        for shape_key in sorted(shape_groups.keys()):
            group = shape_groups[shape_key]
            sub_feats, ok_pats = [], []
            for p in group:
                try:
                    sub_feats.append(self.extract_features(p))
                    ok_pats.append(p)
                except AttributeError:
                    continue
            if not sub_feats:
                continue

            sub_X = scaler.transform(np.array(sub_feats))

            if len(ok_pats) < MIN_PATTERNS_FOR_SPLIT:
                # Too small → single template, no split
                centroid = np.mean(sub_X, axis=0)
                final_templates.append(PatternTemplate(
                    template_id=next_id,
                    centroid=scaler.inverse_transform([centroid])[0],
                    member_count=len(ok_pats),
                    patterns=ok_pats,
                    physics_variance=float(np.std(sub_X[:, 0]))
                ))
                next_id += 1
            else:
                refined = self._recursive_split(sub_X, ok_pats, next_id, scaler)
                final_templates.extend(refined)
                next_id += len(refined)

        print(f"  Splitting done ({_time.perf_counter() - t2:.2f}s) → {len(final_templates)} templates")

        # 4. Sort by size, aggregate oracle intelligence
        final_templates.sort(key=lambda x: x.member_count, reverse=True)
        print(f"  Aggregating Oracle Intelligence...", end="", flush=True)
        for template in final_templates:
            self._aggregate_oracle_intelligence(template, template.patterns, scaler)
        print(f" done. ({_time.perf_counter() - t0:.1f}s total)")

        return scaler, final_templates

    def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]],
                        original_patterns: List[Any]) -> List[PatternTemplate]:
        """
        CLUSTER FISSION (Adj-R² Gain):

        Splits a template only when doing so genuinely improves the explanatory
        power of oracle_mfe ~ feature_vector across children vs parent.

        Replaces the previous silhouette-on-exit-params approach, which split based
        on divergent {TP, SL, trail} — a within-sample signal that doesn't measure
        whether the split actually improves out-of-sample predictive coherence.
        """
        if len(original_patterns) < 2 * MIN_PATTERNS_FOR_SPLIT:
            return []

        # Parent adj-R² (global scaler fitted on all patterns in create_templates)
        parent_r2 = self._compute_adj_r2(original_patterns, self.scaler)

        # Extract features once
        feats, ok_pats = [], []
        for p in original_patterns:
            try:
                feats.append(self.extract_features(p))
                ok_pats.append(p)
            except AttributeError:
                continue

        if len(ok_pats) < 2 * MIN_PATTERNS_FOR_SPLIT:
            return []

        X_scaled = self.scaler.transform(np.array(feats))

        best_gain, best_n, best_labels = -np.inf, 1, None

        for n in range(2, MAX_FISSION_CLUSTERS):
            if len(ok_pats) < n * MIN_PATTERNS_FOR_SPLIT:
                break
            km = self._get_kmeans_model(n_clusters=n, n_samples=len(X_scaled), use_cuda=False)
            labels = km.fit(X_scaled).labels_

            # Weighted adj-R² across children
            weighted_r2, total, valid = 0.0, len(ok_pats), True
            for lbl in range(n):
                sub = [ok_pats[i] for i in np.where(labels == lbl)[0]]
                if len(sub) < MIN_PATTERNS_FOR_SPLIT:
                    valid = False
                    break
                weighted_r2 += self._compute_adj_r2(sub, self.scaler) * len(sub) / total

            if not valid:
                continue
            gain = weighted_r2 - parent_r2
            if gain > best_gain:
                best_gain, best_n, best_labels = gain, n, labels

        if best_gain < R2_FISSION_MIN_GAIN or best_labels is None:
            return []

        print(f"Template {template_id}: FISSION! adj-R² gain={best_gain:+.3f} → {best_n} sub-templates")

        new_templates = []
        for lbl in range(best_n):
            sub_pats = [ok_pats[i] for i in np.where(best_labels == lbl)[0]]
            if not sub_pats:
                continue
            sub_feats = [self.extract_features(p) for p in sub_pats]
            raw_centroid = np.mean(sub_feats, axis=0)
            new_tmpl = PatternTemplate(
                template_id=int(f"{template_id}{lbl}"),
                centroid=raw_centroid,
                member_count=len(sub_pats),
                patterns=sub_pats,
                physics_variance=float(np.std([f[0] for f in sub_feats]))
            )
            self._aggregate_oracle_intelligence(new_tmpl, sub_pats, self.scaler)
            new_templates.append(new_tmpl)

        return new_templates
