"""
Fractal Clustering Engine (Hypervolume Tree)
Replaces KMeans with a hierarchical 16D hypervolume tree.
Groups patterns by their regression residuals at each depth level.
"""
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from training.cuda_kmeans import CUDAKMeans

# Core import for vector reconstruction
from core.quantum_field_engine import QuantumFieldEngine

# Local import of timeframe mapping
from training.fractal_discovery_agent import TIMEFRAME_SECONDS
from config.oracle_config import TEMPLATE_MIN_MEMBERS_FOR_STATS

# Constants
MIN_GROUP_SIZE = 30
MAX_FISSION_CLUSTERS = 6
R2_STOP_THRESHOLD = 0.15
R2_FISSION_MIN_GAIN = 0.05

@dataclass
class PatternTemplate:
    template_id: int
    centroid: np.ndarray  # 16D centroid (scaled space)
    member_count: int
    patterns: List[Any]   # References to original PatternEvents
    physics_variance: float

    # NAVIGATION & RISK DATA
    transition_map: Dict[int, float] = field(default_factory=dict)
    transition_probs: Dict[int, float] = field(default_factory=dict) # Alias

    # REWARD
    expected_value: float = 0.0
    outcome_variance: float = 0.0
    avg_drawdown: float = 0.0

    # Stats populated by _aggregate_oracle_intelligence
    stats_win_rate: float = 0.0
    stats_expectancy: float = 0.0
    stats_mega_rate: float = 0.0
    risk_score: float = 0.0
    risk_variance: float = 0.0

    # Direction Bias
    long_bias: float = 0.0
    short_bias: float = 0.0
    direction: str = '' # 'LONG' or 'SHORT' derived from bias

    # Oracle calibration (ticks)
    mean_mfe_ticks: float = 0.0
    mean_mae_ticks: float = 0.0
    p75_mfe_ticks: float = 0.0
    p25_mae_ticks: float = 0.0
    regression_sigma_ticks: float = 0.0

    # Time-scale calibration (bars)
    avg_mfe_bar: float = 0.0
    p75_mfe_bar: float = 0.0

    # Regression models
    mfe_coeff: Optional[List[float]] = None
    mfe_intercept: float = 0.0
    dir_coeff: Optional[List[float]] = None
    dir_intercept: float = 0.0

    # CST Basin (legacy field support - now handled by node cell bounds)
    basin_mean: float = 0.0
    basin_std: float = 0.0

@dataclass
class HypervolumeNode:
    """One node in the hypervolume tree. Represents a group at a specific depth."""
    depth: int                             # which depth level this node covers
    centroid_16d: np.ndarray               # 16D centroid of this group at this depth
    cell_min_16d: np.ndarray               # per-axis minimum bounds
    cell_max_16d: np.ndarray               # per-axis maximum bounds
    member_count: int                      # patterns that pass through this node
    children: Dict[int, 'HypervolumeNode'] # child_id -> child node (next depth)
    template: Optional[PatternTemplate]    # leaf nodes get a template
    node_id: str                           # path string e.g. "0.2.1"

    # Per-depth scaler (fitted on residuals at this depth within this group)
    scaler: Optional[StandardScaler] = None
    regression_r2: float = 0.0             # how well parent predicts this level
    branch_tightness: float = 0.0          # residual variance
    adj_r2_mfe: float = 0.0                # adj-R²(oracle_mfe ~ 16D)

@dataclass
class HypervolumeTree:
    """The full hypervolume tree."""
    roots: Dict[int, HypervolumeNode]      # root_id -> depth-0 node
    max_depth: int
    n_templates: int

class FractalClusteringEngine:
    def __init__(self, n_clusters=1000, max_variance=0.5):
        self.n_clusters = n_clusters
        self.max_variance = max_variance
        # Scaler is now per-node, but we keep a dummy one for legacy compat if needed
        self.scaler = StandardScaler()
        self.templates = []

    @staticmethod
    def extract_features(p: Any) -> List[float]:
        """
        Extracts 16D feature vector from a PatternEvent.
        Optimized for speed.
        """
        z = p.z_score
        v_feat = math.log1p(abs(p.velocity))
        m_feat = math.log1p(abs(p.momentum))
        c = p.coherence

        tf = p.timeframe
        tf_secs = TIMEFRAME_SECONDS.get(tf, 15)
        tf_scale = math.log2(max(1, tf_secs))

        depth = float(p.depth)
        parent_ctx = 1.0 if p.parent_type == 'ROCHE_SNAP' else 0.0

        state = p.state
        if state:
             self_adx = state.adx_strength * 0.01
             self_hurst = state.hurst_exponent
             self_dmi_diff = (state.dmi_plus - state.dmi_minus) * 0.01
             self_pid       = state.term_pid
             self_osc_coh   = state.oscillation_coherence
        else:
             self_adx = 0.0
             self_hurst = 0.5
             self_dmi_diff = 0.0
             self_pid = 0.0
             self_osc_coh = 0.0

        chain = p.parent_chain
        if chain:
            parent = chain[0]
            parent_z = abs(parent.get('z', 0.0))
            parent_dmi_diff = (parent.get('dmi_plus', 0.0) - parent.get('dmi_minus', 0.0)) * 0.01

            root = chain[-1]
            root_is_roche = 1.0 if root.get('type') == 'ROCHE_SNAP' else 0.0
            root_dmi_diff = (root.get('dmi_plus', 0.0) - root.get('dmi_minus', 0.0)) * 0.01

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
    def build_hypervolume_matrix(p: Any) -> Optional[np.ndarray]:
        """
        Builds the (Depth+1) x 16 matrix for a pattern.
        Row 0 = depth 0 (macro), Row D = depth D (leaf).
        Uses features_16d from parent_chain enriched in Phase 1.
        """
        if not hasattr(p, 'parent_chain') or not p.parent_chain:
            # Fallback for patterns without chain (e.g. macro level 0)
            # Level 0 pattern is its own root
            feat = FractalClusteringEngine.extract_features(p)
            return np.array([feat])

        # Chain is ordered: [immediate_parent, ..., root]
        # We want: [root, ..., immediate_parent, self]

        # Self features
        self_feat = FractalClusteringEngine.extract_features(p)

        # Ancestor features
        # Note: Phase 1 enriched parent_chain with 'features_16d'
        ancestors = []
        # parent_chain is [immediate_parent, ..., root]
        # we want [root, ..., immediate_parent]
        for ancestor in reversed(p.parent_chain):
            if 'features_16d' in ancestor:
                ancestors.append(ancestor['features_16d'])
            else:
                # Fallback extraction if missing (should be enriched)
                # But dict doesn't have all attributes of PatternEvent easily accessible
                # If features_16d is missing, we might have partial data
                pass

        matrix = ancestors + [self_feat]
        return np.array(matrix)

    def _get_kmeans_model(self, n_clusters: int, n_samples: int, random_state: int = 42):
        """Returns a KMeans model -- CUDA if available/large enough."""
        import torch
        # Use CUDA if n_samples large enough AND CUDA is available, else CPU sklearn
        if n_samples > 1000 and torch.cuda.is_available():
             return CUDAKMeans(n_clusters=n_clusters, random_state=random_state)
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)

    def fit_hypervolume_tree(self, patterns: List[Any], min_group_size: int = MIN_GROUP_SIZE) -> HypervolumeTree:
        """
        Build hierarchical hypervolume tree by recursive 16D grouping.
        """
        print(f"Hypervolume: Building tree from {len(patterns)} patterns...")

        # 1. Build matrices
        matrices = {}
        valid_patterns = []

        # Indices in matrices must match indices in pattern_indices list passed to _split_at_depth
        # We use the index 'i' from enumerate(patterns) as the stable ID

        for i, p in enumerate(patterns):
            mat = self.build_hypervolume_matrix(p)
            if mat is not None and mat.shape[0] >= 1:
                matrices[i] = mat
                valid_patterns.append(p)

        print(f"  Constructed {len(matrices)} hypervolume matrices.")

        # 2. Recursive grouping
        root_nodes = self._split_at_depth(
            pattern_indices=list(matrices.keys()),
            matrices=matrices,
            patterns=patterns,
            depth=0,
            parent_id="",
            min_group_size=min_group_size
        )

        max_depth = 0
        if matrices:
            max_depth = max(m.shape[0] for m in matrices.values()) - 1

        n_templates = self._count_leaves(root_nodes)
        print(f"  Tree built: {len(root_nodes)} roots, max depth {max_depth}, {n_templates} leaf templates.")

        tree = HypervolumeTree(roots=root_nodes, max_depth=max_depth, n_templates=n_templates)

        # Flatten templates list for orchestrator usage
        self.templates = self._collect_templates(root_nodes)

        return tree

    def _split_at_depth(self, pattern_indices: List[int], matrices: Dict[int, np.ndarray],
                        patterns: List[Any], depth: int, parent_id: str,
                        min_group_size: int) -> Dict[int, HypervolumeNode]:

        # Extract 16D vectors at this depth
        feat_at_depth = []
        valid_indices = []
        for idx in pattern_indices:
            mat = matrices[idx]
            if depth < mat.shape[0]:
                feat_at_depth.append(mat[depth])
                valid_indices.append(idx)

        if len(valid_indices) < min_group_size:
            return {} # Too small

        feat_array = np.array(feat_at_depth)

        # Step 0: Check adj-R2 stopping criterion
        oracle_mfe = np.array([getattr(patterns[i], 'oracle_meta', {}).get('mfe', 0.0)
                               for i in valid_indices])

        # Ensure we have enough variance to check R2
        if len(oracle_mfe) > 20 and np.std(oracle_mfe) > 1e-9:
            adj_r2 = self._compute_adj_r2(feat_array, oracle_mfe)
        else:
            adj_r2 = 0.0

        if adj_r2 >= R2_STOP_THRESHOLD and len(valid_indices) < 200:
            # Good prediction + smallish group = leaf
            return {}

        # Step 1: Fit Regression (if depth > 0)
        regression_r2 = 0.0
        if depth > 0:
            parent_feat = []
            for idx in valid_indices:
                mat = matrices[idx]
                parent_feat.append(mat[depth-1])
            parent_array = np.array(parent_feat)

            # Simple fallback for singular matrix
            try:
                reg = LinearRegression().fit(parent_array, feat_array)
                predicted = reg.predict(parent_array)
                residuals = feat_array - predicted
                regression_r2 = reg.score(parent_array, feat_array)
            except Exception:
                residuals = feat_array - feat_array.mean(axis=0)
                regression_r2 = 0.0
        else:
            regression_centroid = feat_array.mean(axis=0)
            residuals = feat_array - regression_centroid

        # Step 2: Cluster Residuals
        scaler = StandardScaler()
        residuals_scaled = scaler.fit_transform(residuals)

        residual_var = np.mean(np.var(residuals_scaled, axis=0))

        if residual_var < 0.3:
            n_clusters = 1
        else:
            n_clusters = self._choose_k(residuals_scaled, valid_indices, patterns, scaler)

        if n_clusters <= 1:
            labels = np.zeros(len(valid_indices), dtype=int)
        else:
            kmeans = self._get_kmeans_model(n_clusters, len(valid_indices))
            labels = kmeans.fit_predict(residuals_scaled)

        # Step 3: Build Nodes
        nodes = {}
        for cluster_id in range(max(1, n_clusters)):
            member_mask = labels == cluster_id
            member_indices = [valid_indices[i] for i, m in enumerate(member_mask) if m]

            if len(member_indices) < min_group_size:
                continue

            node_id = f"{parent_id}{cluster_id}" if parent_id else str(cluster_id)
            if parent_id: node_id = f"{parent_id}.{cluster_id}"

            cluster_feat = feat_array[member_mask]
            centroid = cluster_feat.mean(axis=0)
            cell_min = cluster_feat.min(axis=0)
            cell_max = cluster_feat.max(axis=0)

            # Widen cell slightly to avoid edge cases
            margin = (cell_max - cell_min) * 0.05
            cell_min -= margin
            cell_max += margin

            cluster_residuals = residuals_scaled[member_mask]
            branch_tightness = np.mean(np.var(cluster_residuals, axis=0))

            cluster_mfe = oracle_mfe[member_mask]
            if len(cluster_feat) > 20 and np.std(cluster_mfe) > 1e-9:
                cluster_adj_r2 = self._compute_adj_r2(cluster_feat, cluster_mfe)
            else:
                cluster_adj_r2 = 0.0

            # Recursive Split
            children = self._split_at_depth(
                member_indices, matrices, patterns,
                depth + 1, node_id, min_group_size
            )

            # Gain check
            if children:
                child_count = sum(c.member_count for c in children.values())
                if child_count > 0:
                    child_r2_weighted = sum(
                        c.adj_r2_mfe * c.member_count for c in children.values()
                    ) / child_count

                    if child_r2_weighted < cluster_adj_r2 + R2_FISSION_MIN_GAIN:
                        children = {} # Collapse to leaf

            template = None
            if not children:
                member_patterns = [patterns[i] for i in member_indices]
                import hashlib
                # Deterministic ID based on path
                tid = int(hashlib.sha256(node_id.encode('utf-8')).hexdigest(), 16) % 1000000

                template = PatternTemplate(
                    template_id=tid,
                    centroid=centroid,
                    member_count=len(member_patterns),
                    patterns=member_patterns,
                    physics_variance=branch_tightness,
                    basin_mean=0.0,
                    basin_std=0.0
                )
                self._aggregate_oracle_intelligence(template)

            nodes[cluster_id] = HypervolumeNode(
                depth=depth,
                centroid_16d=centroid,
                cell_min_16d=cell_min,
                cell_max_16d=cell_max,
                member_count=len(member_indices),
                children=children,
                template=template,
                node_id=node_id,
                scaler=scaler,
                regression_r2=regression_r2,
                branch_tightness=branch_tightness,
                adj_r2_mfe=cluster_adj_r2
            )

        return nodes

    def _choose_k(self, X_scaled: np.ndarray, indices: List[int], patterns: List[Any],
                  scaler: StandardScaler) -> int:
        """Choose best k (2..MAX_FISSION_CLUSTERS) using Silhouette Score."""
        best_k = 1
        best_score = -1.0

        if len(X_scaled) < MIN_GROUP_SIZE * 2:
            return 1

        from sklearn.metrics import silhouette_score

        # Only check up to a reasonable k
        max_k = min(MAX_FISSION_CLUSTERS, len(X_scaled) // MIN_GROUP_SIZE)
        if max_k < 2:
             return 1

        for k in range(2, max_k + 1):
            km = self._get_kmeans_model(k, len(X_scaled))
            labels = km.fit_predict(X_scaled)

            try:
                score = silhouette_score(X_scaled, labels)
            except:
                score = -1.0

            if score > best_score:
                best_score = score
                best_k = k

        return best_k if best_score > 0.1 else 1

    def _compute_adj_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        n, k = X.shape
        if n <= k + 2:
            return 0.0
        try:
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
            return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
        except:
            return 0.0

    def _count_leaves(self, nodes: Dict[int, HypervolumeNode]) -> int:
        count = 0
        for node in nodes.values():
            if node.template:
                count += 1
            else:
                count += self._count_leaves(node.children)
        return count

    def _collect_templates(self, nodes: Dict[int, HypervolumeNode]) -> List[PatternTemplate]:
        templates = []
        for node in nodes.values():
            if node.template:
                templates.append(node.template)
            else:
                templates.extend(self._collect_templates(node.children))
        return templates

    def _aggregate_oracle_intelligence(self, template: PatternTemplate):
        """Populate template stats from member patterns."""
        patterns = template.patterns
        markers = [p.oracle_marker for p in patterns if hasattr(p, 'oracle_marker')]

        if not markers: return

        # Win Rates
        non_noise = [m for m in markers if m != 0] # MARKER_NOISE=0
        if non_noise:
            template.long_bias = sum(1 for m in non_noise if m > 0) / len(non_noise)
            template.short_bias = sum(1 for m in non_noise if m < 0) / len(non_noise)
            template.direction = 'LONG' if template.long_bias > template.short_bias else 'SHORT'

        template.stats_win_rate = sum(1 for m in markers if abs(m) >= 1) / len(markers)
        template.stats_mega_rate = sum(1 for m in markers if abs(m) == 2) / len(markers)

        # MFE/MAE
        mfes = [getattr(p, 'oracle_meta', {}).get('mfe', 0.0) for p in patterns]
        maes = [getattr(p, 'oracle_meta', {}).get('mae', 0.0) for p in patterns]

        template.stats_expectancy = np.mean(mfes) - np.mean(maes)
        template.risk_variance = float(np.std(mfes))

        # Ticks (MNQ: 0.25)
        _tick = 0.25
        template.mean_mfe_ticks = np.mean(mfes) / _tick
        template.mean_mae_ticks = np.mean(maes) / _tick
        template.p75_mfe_ticks = np.percentile(mfes, 75) / _tick
        template.p25_mae_ticks = np.percentile(maes, 25) / _tick

        # Bars
        bars = [getattr(p, 'oracle_meta', {}).get('mfe_bar', 0) for p in patterns]
        if bars:
            template.avg_mfe_bar = np.mean(bars)
            template.p75_mfe_bar = np.percentile(bars, 75)

        # Regressions
        if len(patterns) >= 20:
             feats = np.array([self.extract_features(p) for p in patterns])
             mfe_y = np.array(mfes)

             sc = StandardScaler()
             X_sc = sc.fit_transform(feats)

             try:
                 ols = LinearRegression().fit(X_sc, mfe_y)
                 template.mfe_coeff = ols.coef_.tolist()
                 template.mfe_intercept = float(ols.intercept_)

                 resids = mfe_y - ols.predict(X_sc)
                 template.regression_sigma_ticks = np.std(resids) / _tick
             except:
                 template.regression_sigma_ticks = 0.0

             # Direction
             labels = np.array([1 if m > 0 else 0 for m in markers if m != 0])
             if len(labels) >= 20 and len(np.unique(labels)) == 2:
                 X_dir = np.array([self.extract_features(p) for p in patterns if p.oracle_marker != 0])
                 X_dir_sc = sc.transform(X_dir)
                 try:
                     lr = LogisticRegression(max_iter=300).fit(X_dir_sc, labels)
                     template.dir_coeff = lr.coef_[0].tolist()
                     template.dir_intercept = float(lr.intercept_[0])
                 except:
                     pass

    def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]],
                        original_patterns: List[Any]) -> List[PatternTemplate]:
        """
        Legacy fission support for Phase 3.
        Hypervolume Tree handles splitting during construction, so we disable
        post-hoc fission here by returning empty list (no split).
        """
        return []
