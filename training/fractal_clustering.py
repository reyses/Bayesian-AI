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
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from training.cuda_kmeans import CUDAKMeans, cuda_silhouette_score

# Local import of timeframe mapping
from training.fractal_discovery_agent import TIMEFRAME_SECONDS

@dataclass
class PatternTemplate:
    template_id: int
    centroid: np.ndarray  # [z, vel, mom, coh, tf_scale, depth, parent_ctx, p_z, p_mom, root_z, root_is_roche]
    member_count: int
    patterns: List[Any]   # References to the original PatternEvents
    physics_variance: float # Measure of how "tight" the cluster is

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
        """
        z = getattr(p, 'z_score', 0.0)
        v = getattr(p, 'velocity', 0.0)
        m = getattr(p, 'momentum', 0.0)
        c = getattr(p, 'coherence', 0.0)

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

        return [abs(z), abs(v), abs(m), c, tf_scale, depth, parent_ctx,
                self_adx, self_hurst, self_dmi_diff,
                parent_z, parent_dmi_diff, root_is_roche, tf_alignment]

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
        print(f"  Total clustering time: {_time.perf_counter() - t0:.2f}s")

        return final_templates

    def _recursive_split(self, X_subset, patterns_subset, start_id) -> List[PatternTemplate]:
        """
        Recursively splits a cluster using KMeans(k=2) until variance is low.
        """
        # Base case: Small enough or Tight enough
        z_std = np.std(X_subset[:, 0])
        if z_std <= self.max_variance or len(X_subset) < 20:
            centroid = np.mean(X_subset, axis=0)
            raw_centroid = self.scaler.inverse_transform([centroid])[0]
            return [PatternTemplate(
                template_id=start_id,
                centroid=raw_centroid,
                member_count=len(patterns_subset),
                patterns=patterns_subset,
                physics_variance=z_std
            )]

        # Split
        kmeans = self._get_kmeans_model(n_clusters=2, n_samples=len(X_subset), n_init=5).fit(X_subset)
        labels = kmeans.labels_

        results = []
        current_id = start_id

        for i in [0, 1]:
            mask = labels == i
            if np.sum(mask) == 0: continue

            child_X = X_subset[mask]
            child_patterns = [patterns_subset[j] for j in range(len(mask)) if mask[j]]

            # Recurse
            child_templates = self._recursive_split(child_X, child_patterns, current_id)
            results.extend(child_templates)
            current_id += len(child_templates)

        return results

    def build_transition_map(self, templates: List[PatternTemplate]):
        """
        SEQUENCE ANALYSIS:
        Maps how market states flow into each other (A -> B).
        Populates template.transition_probs based on sequential pattern events.
        """
        print(f"  Building Navigation Map (Sequence Analysis)...", end="", flush=True)

        # 1. Map Pattern -> Cluster ID
        # pattern object identity -> template_id
        pattern_to_cluster = {}
        all_patterns = []

        for tmpl in templates:
            for p in tmpl.patterns:
                pattern_to_cluster[id(p)] = tmpl.template_id
                all_patterns.append(p)

        # 2. Sort by Timestamp
        all_patterns.sort(key=lambda p: p.timestamp)

        # 3. Analyze Transitions
        transitions = {t.template_id: {} for t in templates} # from -> {to: count}

        for i in range(len(all_patterns) - 1):
            p1 = all_patterns[i]
            p2 = all_patterns[i+1]

            # Check time proximity (e.g., within 60 seconds)
            if p2.timestamp - p1.timestamp <= 60:
                c1 = pattern_to_cluster.get(id(p1))
                c2 = pattern_to_cluster.get(id(p2))

                if c1 is not None and c2 is not None:
                    if c2 not in transitions[c1]:
                        transitions[c1][c2] = 0
                    transitions[c1][c2] += 1

        # 4. Calculate Probabilities
        count_edges = 0
        for tmpl in templates:
            t_counts = transitions.get(tmpl.template_id, {})
            total = sum(t_counts.values())

            if total > 0:
                tmpl.transition_probs = {
                    next_id: count / total
                    for next_id, count in t_counts.items()
                }
                count_edges += len(tmpl.transition_probs)
            else:
                tmpl.transition_probs = {}

        print(f" done. Mapped {count_edges} transitions.")

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

            new_templates.append(PatternTemplate(
                template_id=int(f"{template_id}{label}"),
                centroid=new_phys_centroid,
                member_count=len(sub_patterns),
                patterns=sub_patterns,
                physics_variance=0.0 # Assumed tight since parent was tight
            ))

        return new_templates
