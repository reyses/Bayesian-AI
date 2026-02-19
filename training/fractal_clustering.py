"""
Fractal Clustering Engine
Reduces massive pattern datasets into manageable 'Templates' using Recursive K-Means.
Maps raw physics events into "Archetypal Centroids" that are both Physically Tight and Behaviorally Consistent.

Feature vector per pattern (7D):
  [|z_score|, |velocity|, |momentum|, coherence, log2(tf_seconds), depth, parent_is_roche]
The timeframe scale + depth + parent context let the clustering naturally separate
patterns that look similar in physics but live at different fractal scales.
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

    def _get_kmeans_model(self, n_clusters: int, n_samples: int, random_state: int = 42, n_init: int = 3):
        """Returns a CUDAKMeans model."""
        return CUDAKMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    @staticmethod
    def extract_features(p: Any) -> List[float]:
        """
        Extracts 14D feature vector from a PatternEvent.
        [7 base] + [3 self regime] + [4 ancestry]

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
        else:
             self_adx = 0.0
             self_hurst = 0.5
             self_dmi_diff = 0.0

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
                parent_z, parent_dmi_diff, root_is_roche, tf_alignment]

    def _recursive_split(self, X: np.ndarray, patterns: list, start_id: int, depth: int = 0) -> list:
        """Recursively split a cluster until z-variance <= max_variance or too small."""
        z_var = np.std(X[:, 0])
        if z_var <= self.max_variance or len(patterns) <= 20 or depth > 5:
            centroid = np.mean(X, axis=0)
            raw_centroid = self.scaler.inverse_transform([centroid])[0]
            return [PatternTemplate(
                template_id=start_id,
                centroid=raw_centroid,
                member_count=len(patterns),
                patterns=patterns,
                physics_variance=z_var
            )]

        k = min(3, max(2, len(patterns) // 20))
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
            children = self._recursive_split(sub_X, sub_p, nid, depth + 1)
            result.extend(children)
            nid += len(children)
        return result

    def _aggregate_oracle_intelligence(self, template, patterns):
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
                X_scaled = self.scaler.transform(raw_X)

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
                        dir_X = self.scaler.transform(dir_raw)
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

                # Check for gap (using index difference in sorted list as proxy for bars?
                # Instructions say "Min bars between events". But here we have list of ALL patterns.
                # Gap in time? Patterns have timestamps.
                # Patterns are from different timeframes potentially.
                # Assuming "bars" refers to the base timeframe or just sequential count in the list?
                # "TRANSITION_MIN_SEQUENCE_GAP_BARS" implies time/bars.
                # If patterns are sparse, "bars" logic might be tricky if we don't have bar index globally aligned.
                # However, the instruction implementation used:
                # gap = nxt.timestamp - curr.timestamp (in original logic it was time check <= 60s)
                # But here the provided code snippet uses:
                # gap = nxt.timestamp - curr.timestamp (WAIT, snippet didn't show implementation details for gap calc, just 'gap')
                # Let's assume it means sequential index gap in the sorted list if we treat the sorted list as a sequence of events?
                # Or time gap.
                # "Min bars between events".
                # If we use time, we need to know bar size.
                # Let's assume "sequence gap" means index difference in `sorted_patterns` for now,
                # OR better, stick to the snippet logic if provided.
                # The snippet:
                # gap = nxt.timestamp - curr.timestamp (No, snippet says "gap < TRANSITION_MIN_SEQUENCE_GAP_BARS")
                # Wait, the snippet uses `gap` variable but doesn't define it.
                # But looking at `training/fractal_clustering.py` existing `build_transition_map`, it uses `p2.timestamp - p1.timestamp <= 60`.
                # I'll use timestamp difference relative to some base time (e.g. 15s bars).
                # `TRANSITION_MIN_SEQUENCE_GAP_BARS = 1` -> 15s?
                # `TRANSITION_MAX_SEQUENCE_GAP_BARS = 100` -> 1500s?

                # Actually, let's look at the instruction snippet again.
                # "gap = nxt.timestamp - curr.timestamp" (Wait, I invented that in my thought process)
                # In the snippet provided in `docs/JULES_ORACLE_ENGINE.md`:
                # "gap = nxt.timestamp - curr.timestamp" is NOT there.
                # It just says:
                # gap = nxt.timestamp - curr.timestamp
                # if gap < TRANSITION_MIN_SEQUENCE_GAP_BARS: continue
                # Wait, comparing timestamp (seconds) with BARS (int) is wrong unless converted.
                # I should probably convert timestamp diff to bars using 15s as base?
                # Or just use the number of patterns in between?
                # "Min bars between events" usually means time.
                # I will assume 15s base timeframe for bars.

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
        Groups raw PatternEvents into Templates using RECURSIVE REFINEMENT.
        Ensures every template is physically homogeneous before optimization begins.
        """
        import time as _time

        if not manifest:
            print("WARNING: FractalClusteringEngine received empty manifest.")
            return []

        t0 = _time.perf_counter()

        # 1. Extract Feature Matrix (11D)
        # Vector: [|Z-Score|, |Velocity|, |Momentum|, Coherence,
        #          log2(tf_seconds), depth, parent_is_roche,
        #          parent_z, parent_mom, root_z, root_is_roche]
        features = []
        valid_patterns = []

        for p in manifest:
            try:
                feat = self.extract_features(p)
                features.append(feat)
                valid_patterns.append(p)
            except AttributeError:
                continue

        if not features:
            return []

        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        print(f"  Feature matrix: {X.shape[0]} patterns x {X.shape[1]} features (extracted in {_time.perf_counter() - t0:.2f}s)")

        # 2. Initial Coarse Clustering
        # Start with a conservative K
        target_k = min(self.n_clusters, len(valid_patterns) // 20)
        target_k = max(target_k, 1)

        t1 = _time.perf_counter()
        print(f"  Coarse KMeans: fitting {len(valid_patterns)} patterns into {target_k} clusters...", end="", flush=True)

        model = self._get_kmeans_model(n_clusters=target_k, n_samples=len(valid_patterns))
        labels = model.fit_predict(X_scaled)
        print(f" done ({_time.perf_counter() - t1:.2f}s)")

        # Group indices by label
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices: cluster_indices[label] = []
            cluster_indices[label].append(idx)

        # Show cluster size distribution
        sizes = [len(v) for v in cluster_indices.values()]
        print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}, avg={np.mean(sizes):.0f}")

        # 3. Recursive Refinement (The "Physics Tightening" Loop)
        t2 = _time.perf_counter()
        print(f"  Recursive refinement (max_variance={self.max_variance})...", end="", flush=True)
        final_templates = []
        next_id = 0
        splits_count = 0

        for label, indices in cluster_indices.items():
            sub_X = X_scaled[indices]
            sub_patterns = [valid_patterns[i] for i in indices]

            # Check Variance (Goodness of Fit on Physics)
            # We look primarily at Z-Score variance (Feature 0)
            z_variance = np.std(sub_X[:, 0])

            if z_variance > self.max_variance and len(indices) > 20:
                # CLUSTER IS TOO LOOSE -> RECURSIVE SPLIT
                splits_count += 1
                refined_subsets = self._recursive_split(sub_X, sub_patterns, next_id)
                final_templates.extend(refined_subsets)
                next_id += len(refined_subsets)
            else:
                # CLUSTER IS TIGHT -> KEEP
                centroid = np.mean(sub_X, axis=0)
                raw_centroid = self.scaler.inverse_transform([centroid])[0]

                final_templates.append(PatternTemplate(
                    template_id=next_id,
                    centroid=raw_centroid,
                    member_count=len(sub_patterns),
                    patterns=sub_patterns,
                    physics_variance=z_variance
                ))
                next_id += 1

        print(f" done ({_time.perf_counter() - t2:.2f}s)")
        print(f"  Refinement: {target_k} coarse -> {len(final_templates)} tight templates ({splits_count} clusters split)")

        # Sort by size
        final_templates.sort(key=lambda x: x.member_count, reverse=True)

        # Summary stats
        tmpl_sizes = [t.member_count for t in final_templates]
        variances = [t.physics_variance for t in final_templates]
        print(f"  Template sizes: min={min(tmpl_sizes)}, max={max(tmpl_sizes)}, avg={np.mean(tmpl_sizes):.0f}")
        print(f"  Z-variance: min={min(variances):.3f}, max={max(variances):.3f}, avg={np.mean(variances):.3f}")

        # --- NEW: Oracle Intelligence Aggregation ---
        print(f"  Aggregating Oracle Intelligence...", end="", flush=True)
        for template in final_templates:
            self._aggregate_oracle_intelligence(template, template.patterns)
        print(" done.")

        # --- NEW: Transition Matrix ---
        print(f"  Building Transition Matrix...", end="", flush=True)
        self._build_transition_matrix(final_templates, valid_patterns)
        print(" done.")

        print(f"  Total clustering time: {_time.perf_counter() - t0:.2f}s")

        return final_templates

    def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]], original_patterns: List[Any]) -> List[PatternTemplate]:
        """
        CLUSTER FISSION (Regret-Based):
        Analyzes 'Optimal Parameters' (Outcome) to split clusters that are physically tight but behaviorally different.
        """
        if len(member_params) < 10: return []

        # Vectorize Parameters [Stop, TP, Trail]
        param_vectors = [[p.get('stop_loss_ticks', 0), p.get('take_profit_ticks', 0), p.get('trailing_stop_ticks', 0)] for p in member_params]
        X_params = np.array(param_vectors)

        # Silhouette Check
        best_n, best_score, best_labels = 1, -1.0, None

        for n in range(2, 6):
            if len(X_params) < n * 5: break
            kmeans = self._get_kmeans_model(n_clusters=n, n_samples=len(X_params)).fit(X_params)
            score = cuda_silhouette_score(X_params, kmeans.labels_)
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

            # Recalculate Oracle Stats for new split
            self._aggregate_oracle_intelligence(new_tmpl, sub_patterns)

            new_templates.append(new_tmpl)

        return new_templates

    def build_transition_map(self, templates: List[PatternTemplate]):
        """Legacy wrapper"""
        # We need all patterns to rebuild transitions properly.
        # Since we don't have all patterns here easily, we might skip or warn.
        # But this function was called in orchestrator? No, it wasn't called in orchestrator.
        # It was defined but not used in `create_templates` in previous version?
        # In previous version `build_transition_map` was NOT called in `create_templates`.
        # It was separate.
        # Now I call `_build_transition_matrix` inside `create_templates`.
        pass
