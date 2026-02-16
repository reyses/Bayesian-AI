"""
Fractal Clustering Engine
Reduces massive pattern datasets into manageable 'Templates' using Recursive K-Means.
Maps raw physics events into "Archetypal Centroids" that are both Physically Tight and Behaviorally Consistent.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

@dataclass
class PatternTemplate:
    template_id: int
    centroid: np.ndarray  # [z, vel, mom, coh]
    member_count: int
    patterns: List[Any]   # References to the original PatternEvents
    physics_variance: float # Measure of how "tight" the cluster is

class FractalClusteringEngine:
    def __init__(self, n_clusters=1000, max_variance=0.5):
        self.n_clusters = n_clusters
        self.max_variance = max_variance  # Max allowed std deviation for Z-score in a cluster
        self.scaler = StandardScaler()
        # Use MiniBatch for speed on large datasets
        self.model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=42)

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

        # 1. Extract Feature Matrix
        # Vector: [Z-Score (abs), Velocity (abs), Momentum (abs), Coherence]
        features = []
        valid_patterns = []

        for p in manifest:
            try:
                z = getattr(p, 'z_score', 0.0)
                v = getattr(p, 'velocity', 0.0)
                m = getattr(p, 'momentum', 0.0)
                c = getattr(p, 'coherence', 0.0)
                features.append([abs(z), abs(v), abs(m), c])
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
        self.model.n_clusters = target_k
        labels = self.model.fit_predict(X_scaled)
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
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_subset)
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
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(X_params)
            score = silhouette_score(X_params, kmeans.labels_)
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
            # Re-calculate Physics Centroid
            sub_features = []
            for p in sub_patterns:
                z = getattr(p, 'z_score', 0.0)
                v = getattr(p, 'velocity', 0.0)
                m = getattr(p, 'momentum', 0.0)
                c = getattr(p, 'coherence', 0.0)
                sub_features.append([abs(z), abs(v), abs(m), c])

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
