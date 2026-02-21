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
from sklearn.linear_model import LinearRegression
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
MIN_PATTERNS_FOR_SPLIT = 20
MAX_RECURSION_DEPTH = 5
MIN_SAMPLES_PER_CLUSTER = 20
MAX_FISSION_CLUSTERS = 6
MIN_FISSION_SAMPLES = 5

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
        # Sign-preserving log compression: sign(x)*log1p(|x|) keeps direction
        # encoded in the feature so KMeans naturally separates LONG vs SHORT arcs
        v_feat = np.sign(v) * np.log1p(abs(v))
        m_feat = np.sign(m) * np.log1p(abs(m))

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
            # Immediate parent (signed: preserves which side of the field parent lived on)
            parent_z = chain[0].get('z', 0.0)
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

        return [z, v_feat, m_feat, c, tf_scale, depth, parent_ctx,
                self_adx, self_hurst, self_dmi_diff,
                parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
                self_pid, self_osc_coh]

    def _recursive_split(self, X: np.ndarray, patterns: list, start_id: int, scaler: StandardScaler, depth: int = 0) -> list:
        """Recursively split a cluster until z-variance <= max_variance or too small."""
        z_var = np.std(X[:, 0])
        if z_var <= self.max_variance or len(patterns) <= MIN_PATTERNS_FOR_SPLIT or depth > MAX_RECURSION_DEPTH:
            centroid = np.mean(X, axis=0)
            raw_centroid = scaler.inverse_transform([centroid])[0]
            return [PatternTemplate(
                template_id=start_id,
                centroid=raw_centroid,
                member_count=len(patterns),
                patterns=patterns,
                physics_variance=z_var
            )]

        k = min(3, max(2, len(patterns) // MIN_PATTERNS_FOR_SPLIT))
        # Clamp k to the number of distinct points to avoid sklearn ConvergenceWarning
        n_unique = len(np.unique(X, axis=0))
        k = min(k, n_unique)
        if k <= 1:
            centroid = np.mean(X, axis=0)
            raw_centroid = scaler.inverse_transform([centroid])[0]
            return [PatternTemplate(
                template_id=start_id,
                centroid=raw_centroid,
                member_count=len(patterns),
                patterns=patterns,
                physics_variance=z_var
            )]
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
            # Build aligned (features, signed_mfe) pairs for members that have oracle data.
            # signed_mfe = oracle_mfe * sign(oracle_marker):
            #   positive → LONG move, negative → SHORT move.
            # A single OLS on signed_mfe encodes both direction AND magnitude so that:
            #   prediction > 0  →  LONG  (cluster historically resolved bullish)
            #   prediction < 0  →  SHORT (cluster historically resolved bearish)
            #   abs(prediction) →  TP target in points
            # Noise members (oracle_marker == 0) are excluded — they have no direction.
            feat_mfe_pairs = [
                (self.extract_features(p),
                 p.oracle_meta['mfe'] * (1.0 if getattr(p, 'oracle_marker', 0) > 0 else -1.0))
                for p in patterns
                if getattr(p, 'oracle_meta', {}).get('mfe') is not None
                and getattr(p, 'oracle_marker', 0) != 0
            ]
            _MIN_REG = 15  # need at least 15 members for stable regression

            if len(feat_mfe_pairs) >= _MIN_REG:
                raw_X  = np.array([f for f, _ in feat_mfe_pairs])
                mfe_y  = np.array([m for _, m in feat_mfe_pairs])

                # Scale features using the engine's fitted scaler
                X_scaled = scaler.transform(raw_X)

                # --- OLS: signed_mfe = X @ mfe_coeff + mfe_intercept ---
                # sign(prediction) = direction, abs(prediction) = TP target in points
                ols = LinearRegression().fit(X_scaled, mfe_y)
                template.mfe_coeff     = ols.coef_.tolist()
                template.mfe_intercept = float(ols.intercept_)
                residuals = mfe_y - ols.predict(X_scaled)
                template.regression_sigma_ticks = float(np.std(residuals) / _tick)
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
        Unified fit: cluster all patterns together with a single scaler.
        Direction bias (long_bias / short_bias) is computed per-template by
        _aggregate_oracle_intelligence() from oracle_marker labels — no need
        to split before clustering, which would halve training data per cluster.
        """
        # Exclude noise-only patterns (oracle_marker == 0) to keep clusters meaningful
        valid_patterns = [p for p in manifest if getattr(p, 'oracle_marker', 0) != 0]
        noise_count    = len(manifest) - len(valid_patterns)
        print(f"Unified Clustering: {len(manifest)} patterns ({noise_count} noise excluded → {len(valid_patterns)} clustered)")

        long_count  = sum(1 for p in valid_patterns if getattr(p, 'oracle_marker', 0) > 0)
        short_count = sum(1 for p in valid_patterns if getattr(p, 'oracle_marker', 0) < 0)
        print(f"  LONG members: {long_count}  |  SHORT members: {short_count}  (both in same clusters)")

        self.scaler, templates = self._fit_branch(valid_patterns, 'UNIFIED')

        # Build Transition Matrix
        print(f"  Building Transition Matrix...", end="", flush=True)
        self._build_transition_matrix(templates, valid_patterns)
        print(" done.")

        self.templates = templates
        return templates

    def _fit_branch(self, patterns: List[Any], direction: str):
        """
        Fit scaler + recursive cluster tree for one directional branch.
        Returns (scaler, list[PatternTemplate])
        """
        import time as _time
        from sklearn.preprocessing import StandardScaler

        if not patterns:
            return StandardScaler(), []

        print(f"\n--- Fitting {direction} Branch ({len(patterns)} patterns) ---")
        t0 = _time.perf_counter()

        # 1. Extract Feature Matrix
        features = []
        valid_patterns = []

        for p in patterns:
            try:
                feat = self.extract_features(p)
                features.append(feat)
                valid_patterns.append(p)
            except AttributeError:
                continue

        if not features:
             return StandardScaler(), []

        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"  Feature matrix: {X.shape[0]} patterns x {X.shape[1]} features")

        # 2. Initial Coarse Clustering
        # Start with a conservative K
        target_k = min(self.n_clusters // 2, len(valid_patterns) // MIN_SAMPLES_PER_CLUSTER) # Half clusters per branch
        target_k = max(target_k, 1)

        t1 = _time.perf_counter()
        print(f"  Coarse KMeans: fitting into {target_k} clusters...", end="", flush=True)

        # Flush GPU
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            model = self._get_kmeans_model(n_clusters=target_k, n_samples=len(valid_patterns))
            labels = model.fit_predict(X_scaled)
        except Exception as _cuda_err:
             print(f"\n  [KMeans CUDA fallback: {type(_cuda_err).__name__}]", end="", flush=True)
             from sklearn.cluster import KMeans as _SKMeans
             _fallback = _SKMeans(n_clusters=target_k, random_state=42, n_init=3, max_iter=300)
             labels = _fallback.fit_predict(X_scaled)
        print(f" done ({_time.perf_counter() - t1:.2f}s)")

        # Group indices
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices: cluster_indices[label] = []
            cluster_indices[label].append(idx)

        # 3. Recursive Refinement
        t2 = _time.perf_counter()
        print(f"  Recursive refinement...", end="", flush=True)
        final_templates = []
        next_id = 0 if direction == 'LONG' else 1000000 # Offset short IDs to avoid collision if desired, or let them just be unique in list.
        # Actually better to just use a counter locally, but we need unique IDs across branches if possible?
        # The merge step just appends. If ID is 0 in both branches, we have duplicates.
        # Let's check how _recursive_split assigns IDs. It takes start_id.
        # I should probably pass a start_id or re-index later.
        # To be safe, I'll use 0 for LONG and 100000 for SHORT base?
        # Or just let them overlap and rely on 'direction' field to distinguish?
        # Orchestrator keys them by ID in a dict. Overlap is BAD.
        # I will use start_id=0 for LONG and start_id=50000 for SHORT.

        start_id_offset = 0 if direction == 'LONG' else 50000
        next_id = start_id_offset
        splits_count = 0

        for label, indices in cluster_indices.items():
            sub_X = X_scaled[indices]
            sub_patterns = [valid_patterns[i] for i in indices]

            z_variance = np.std(sub_X[:, 0])

            if z_variance > self.max_variance and len(indices) > MIN_SAMPLES_PER_CLUSTER:
                splits_count += 1
                refined_subsets = self._recursive_split(sub_X, sub_patterns, next_id, scaler)
                final_templates.extend(refined_subsets)
                next_id += len(refined_subsets)
            else:
                centroid = np.mean(sub_X, axis=0)
                raw_centroid = scaler.inverse_transform([centroid])[0]
                final_templates.append(PatternTemplate(
                    template_id=next_id,
                    centroid=raw_centroid,
                    member_count=len(sub_patterns),
                    patterns=sub_patterns,
                    physics_variance=z_variance
                ))
                next_id += 1

        print(f" done ({_time.perf_counter() - t2:.2f}s)")

        # Sort by size
        final_templates.sort(key=lambda x: x.member_count, reverse=True)

        # Aggregation
        print(f"  Aggregating Oracle Intelligence...", end="", flush=True)
        for template in final_templates:
            self._aggregate_oracle_intelligence(template, template.patterns, scaler)
        print(" done.")

        # Transition matrix building requires ALL patterns?
        # The patterns are split by branch. A LONG pattern might be followed by a SHORT pattern.
        # _build_transition_matrix uses the list passed to it.
        # If I build transition matrix per branch, I only see L->L or S->S transitions.
        # L->S transitions are missed.
        # BUT: create_templates calls _build_transition_matrix on the MERGED list at the end?
        # No, create_templates (new) needs to call transition matrix building on the merged list!
        # The legacy code called it at the end of create_templates.
        # So I should remove the call from _fit_branch and do it in create_templates.

        return scaler, final_templates

    def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]], original_patterns: List[Any]) -> List[PatternTemplate]:
        """
        CLUSTER FISSION (Regret-Based):
        Analyzes 'Optimal Parameters' (Outcome) to split clusters that are physically tight but behaviorally different.
        """
        if len(member_params) < 10: return []

        # Vectorize Parameters [Stop, TP, Trail]
        param_vectors = [[p.get('stop_loss_ticks', 0), p.get('take_profit_ticks', 0), p.get('trailing_stop_ticks', 0)] for p in member_params]
        X_params = np.array(param_vectors)

        # Silhouette Check (CPU-only -- called from multiprocessing workers, no CUDA context)
        from sklearn.metrics import silhouette_score as cpu_silhouette_score
        best_n, best_score, best_labels = 1, -1.0, None

        for n in range(2, MAX_FISSION_CLUSTERS):
            if len(X_params) < n * MIN_FISSION_SAMPLES: break
            kmeans = self._get_kmeans_model(n_clusters=n, n_samples=len(X_params),
                                            use_cuda=False).fit(X_params)
            try:
                score = cpu_silhouette_score(X_params, kmeans.labels_)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_n = n
                best_labels = kmeans.labels_

        # Decision Gate (High Silhouette = Distinct Behaviors)
        if best_score < 0.45: return []

        print(f"Template {template_id}: FISSION! (Score: {best_score:.2f}) -> Splitting into {best_n}.")

        new_templates = []
        split_map = {i: [] for i in range(best_n)}
        for idx, label in enumerate(best_labels):
            split_map[label].append(original_patterns[idx])

        # IMPORTANT: We need the scaler corresponding to this template's direction!
        # This method doesn't know direction easily.
        # However, Fission uses `self.extract_features` which doesn't use scaler.
        # But `_aggregate_oracle_intelligence` USES scaler.
        # I should probably pick the scaler based on something?
        # Or just use the global `self.scaler` which I populated as fallback?
        # Or pass scaler in?
        # Refine clusters is called by worker? No, by OrchestratorWorker.
        # It's called in `_process_template_job`.
        # The worker process has a copy of clustering_engine.
        # Ideally I should use the correct scaler.
        # Since I can't easily know which scaler to use without checking pattern direction,
        # I will check the first pattern's oracle_marker/z_score to guess direction?
        # No, patterns passed here are just the subset.
        # Let's use `self.scaler` (fallback) which I ensured is fitted on ALL data in create_templates.
        # This is a compromise but safer than changing method signature which breaks worker.

        for label, sub_patterns in split_map.items():
            # Re-calculate Physics Centroid (11D to match create_templates)
            sub_features = []
            for p in sub_patterns:
                feat = self.extract_features(p)
                sub_features.append(feat)

            if not sub_features: continue
            new_phys_centroid = np.mean(sub_features, axis=0)

            # Create new template
            new_tmpl = PatternTemplate(
                template_id=int(f"{template_id}{label}"),
                centroid=new_phys_centroid,
                member_count=len(sub_patterns),
                patterns=sub_patterns,
                physics_variance=0.0 # Assumed tight since parent was tight
            )

            # Inherit direction from parent?
            # PatternTemplate doesn't store parent ref, but we can check first pattern.
            if sub_patterns:
                 if getattr(sub_patterns[0], 'oracle_marker', 0) > 0:
                     new_tmpl.direction = 'LONG'
                 elif getattr(sub_patterns[0], 'oracle_marker', 0) < 0:
                     new_tmpl.direction = 'SHORT'

            # Recalculate Oracle Stats for new split
            # Use self.scaler as fallback
            self._aggregate_oracle_intelligence(new_tmpl, sub_patterns, self.scaler)

            new_templates.append(new_tmpl)

        return new_templates
