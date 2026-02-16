"""
Fractal Clustering Engine
Reduces massive pattern datasets into manageable 'Templates' using K-Means.
Maps raw physics events into "Archetypal Centroids" for optimization.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class PatternTemplate:
    template_id: int
    centroid: np.ndarray  # [z, vel, mom, coh]
    member_count: int
    patterns: List[Any]   # References to the original PatternEvents

class FractalClusteringEngine:
    def __init__(self, n_clusters=1000):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=42)

    def create_templates(self, manifest: List[Any]) -> List[PatternTemplate]:
        """
        Groups raw PatternEvents into Templates based on physics vectors.
        """
        if not manifest:
            print("WARNING: FractalClusteringEngine received empty manifest.")
            return []

        # 1. Extract Feature Matrix
        # Vector: [Z-Score (abs), Velocity (abs), Momentum (abs), Coherence]
        # We use absolute values to group symmetrical Long/Short structures.
        features = []
        valid_patterns = []

        for p in manifest:
            # Ensure attributes exist (handling potential data gaps)
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

        # 2. Normalize
        X_scaled = self.scaler.fit_transform(X)

        # 3. Fit K-Means
        # Dynamic K adjustment: Ensure we don't ask for more clusters than samples
        # Heuristic: Target avg 10 items per cluster, but capped at self.n_clusters
        target_k = min(self.n_clusters, len(valid_patterns) // 10)
        target_k = max(target_k, 1) # Minimum 1 cluster

        self.model.n_clusters = target_k

        print(f"Clustering: Fitting {len(valid_patterns)} patterns into {target_k} templates...")
        labels = self.model.fit_predict(X_scaled)
        centroids = self.model.cluster_centers_

        # 4. Group into Templates
        template_map = {}
        for idx, label in enumerate(labels):
            if label not in template_map:
                template_map[label] = {
                    'patterns': []
                }
            template_map[label]['patterns'].append(valid_patterns[idx])

        # 5. Convert to Objects
        result = []
        for label, data in template_map.items():
            # Denormalize centroid for human-readable logging
            raw_centroid = self.scaler.inverse_transform([centroids[label]])[0]

            result.append(PatternTemplate(
                template_id=int(label),
                centroid=raw_centroid,
                member_count=len(data['patterns']),
                patterns=data['patterns']
            ))

        # Sort by member count (Optimize biggest/most frequent groups first)
        result.sort(key=lambda x: x.member_count, reverse=True)
        return result
