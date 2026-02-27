"""
Fractal Clustering Engine (Hypervolume Tree)
Geometric shape-based partitioning via IMR (Individual-Moving Range) charts.
Patterns are ordered along the principal geometric axis, then split where the
composite signal (position deviation + moving range gap) exceeds the IMR
control threshold. DBSCAN discovers sub-clusters on the differential surface.
"""
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Core import for vector reconstruction
from core.quantum_field_engine import QuantumFieldEngine

# Local import of timeframe mapping
from training.fractal_discovery_agent import TIMEFRAME_SECONDS
from config.oracle_config import TEMPLATE_MIN_MEMBERS_FOR_STATS

# Constants
MIN_GROUP_SIZE = 30
R2_STOP_THRESHOLD = 0.90       # DOE Phase 3: target R2 for "real case usage" (was 0.15)
R2_FISSION_MIN_GAIN = 0.05
DOE_MAX_ITERATIONS = 20         # DOE Phase 3: max passes over the tree (safety valve)
DEFAULT_PID = {'pid_kp': 0.5, 'pid_ki': 0.1, 'pid_kd': 0.2}

# I-MR SPC constants (n=2 subgroups)
IMR_D4 = 3.267          # UCL factor for Moving Range
IMR_D2 = 1.128          # Sigma estimation factor
IMR_Z_THRESHOLD = 1.0   # Composite signal threshold for segment boundary

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

    # DOE Phase 3 validation
    consistency_score: float = 0.0
    consistency_diagnostics: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None

@dataclass
class HypervolumeNode:
    """One node in the hypervolume tree. Represents a group at a specific depth."""
    depth: int                             # which depth level this node covers
    centroid_16d: np.ndarray               # 16D centroid of this group at this depth
    cell_min_16d: np.ndarray               # per-axis minimum bounds
    cell_max_16d: np.ndarray               # per-axis maximum bounds
    member_count: int                      # patterns that pass through this node
    member_indices: List[int]              # pattern indices for DOE re-fission
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

    def _imr_geometric_split(self, residuals_scaled: np.ndarray,
                              min_group_size: int) -> np.ndarray:
        """I-MR + differential DBSCAN splitting (replaces KMeans).

        Three layers:
          1. I-MR: Project onto PC1, detect geometry shifts via Moving Range > UCL.
             Segments data into coherent geometric regions at boundary points.
          2. Differential: velocity (1st diff) + acceleration (2nd diff) on sorted 16D
             residuals within each I-MR segment.
          3. DBSCAN: density-based sub-clustering on normalized differential states
             within each segment (discovers sub-patterns).

        Returns: label array (0-based contiguous int per pattern, original order).
        """
        n = len(residuals_scaled)

        if n < min_group_size * 2:
            return np.zeros(n, dtype=int)

        # === LAYER 1: I-MR Control Charts ===

        # Project onto principal variance axis via truncated SVD
        mean_vec = residuals_scaled.mean(axis=0)
        centered = residuals_scaled - mean_vec
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = Vt[0]
        projections = centered @ pc1

        sort_order = np.argsort(projections)
        projections_sorted = projections[sort_order]
        residuals_ordered = residuals_scaled[sort_order]

        # Moving Range: L2 distance between consecutive ordered patterns in full 16D
        mr_l2 = np.array([np.linalg.norm(residuals_ordered[i] - residuals_ordered[i - 1])
                          for i in range(1, n)])

        if len(mr_l2) == 0 or np.max(mr_l2) < 1e-12:
            return np.zeros(n, dtype=int)

        mr_bar = np.mean(mr_l2)
        ucl_mr = IMR_D4 * mr_bar
        sigma_est = mr_bar / IMR_D2

        # I-chart: deviation from mean along PC1
        x_bar = np.mean(projections_sorted)
        ucl_i_half = 3.0 * sigma_est if sigma_est > 1e-12 else 1.0

        # Composite signal: x (normalized I-deviation) + y (normalized MR)
        x_norm = np.abs(projections_sorted[1:] - x_bar) / ucl_i_half
        y_norm = mr_l2 / ucl_mr if ucl_mr > 1e-12 else np.zeros_like(mr_l2)
        signal = x_norm + y_norm

        # Segment boundaries: where composite signal > threshold
        boundary_indices = set(np.where(signal > IMR_Z_THRESHOLD)[0])

        # Build I-MR segments (runs of consecutive sorted points)
        sorted_segment_ids = np.zeros(n, dtype=int)
        current_seg = 0
        for i in range(1, n):
            if (i - 1) in boundary_indices:
                current_seg += 1
            sorted_segment_ids[i] = current_seg

        # === LAYER 2 + 3: Differential DBSCAN within each segment ===

        sorted_labels = np.zeros(n, dtype=int)
        global_label = 0

        for seg_id in range(current_seg + 1):
            seg_mask = sorted_segment_ids == seg_id
            seg_indices = np.where(seg_mask)[0]
            seg_size = len(seg_indices)

            if seg_size < 2:
                # Single point — assign to this segment's label
                sorted_labels[seg_indices] = global_label
                global_label += 1
                continue

            # Differential surface within this segment
            seg_residuals = residuals_ordered[seg_indices]

            # Velocity: first difference in 16D
            velocity = np.diff(seg_residuals, axis=0)  # (seg_size-1, 16)

            if velocity.shape[0] < 2:
                # Too small for acceleration — keep as one group
                sorted_labels[seg_indices] = global_label
                global_label += 1
                continue

            # Acceleration: second difference
            acceleration = np.diff(velocity, axis=0)  # (seg_size-2, 16)
            n_diff = acceleration.shape[0]

            if n_diff < max(3, min_group_size // 10):
                sorted_labels[seg_indices] = global_label
                global_label += 1
                continue

            # Build differential state: [velocity_16d, acceleration_16d] = 32D
            diff_states = np.hstack([velocity[1:], acceleration])

            # Standardize
            diff_std = diff_states.std(axis=0)
            diff_std[diff_std < 1e-12] = 1.0
            diff_norm = diff_states / diff_std

            # DBSCAN: calibrate eps via k-NN
            k_nn = min(5, n_diff - 1)
            if k_nn < 1:
                sorted_labels[seg_indices] = global_label
                global_label += 1
                continue

            nn = NearestNeighbors(n_neighbors=k_nn).fit(diff_norm)
            distances, _ = nn.kneighbors(diff_norm)
            eps = float(np.median(distances[:, -1]))
            if eps < 1e-12:
                eps = 0.5

            dbscan_min = max(3, min_group_size // 10)
            db_labels = DBSCAN(eps=eps, min_samples=dbscan_min).fit_predict(diff_norm)

            # Extend diff_labels (N-2) back to full segment (N)
            full_seg_labels = np.full(seg_size, -1, dtype=int)
            full_seg_labels[1:n_diff + 1] = db_labels
            full_seg_labels[0] = db_labels[0]
            if n_diff + 1 < seg_size:
                full_seg_labels[n_diff + 1:] = db_labels[-1]

            # Map DBSCAN sub-labels to global labels
            unique_sub = np.unique(full_seg_labels[full_seg_labels >= 0])
            if len(unique_sub) == 0:
                # All noise within segment — single group
                sorted_labels[seg_indices] = global_label
                global_label += 1
            else:
                sub_map = {sub: global_label + i for i, sub in enumerate(unique_sub)}
                for j, idx in enumerate(seg_indices):
                    if full_seg_labels[j] >= 0:
                        sorted_labels[idx] = sub_map[full_seg_labels[j]]
                    else:
                        sorted_labels[idx] = -1  # noise, absorb later
                global_label += len(unique_sub)

        # === Map back to original order ===
        raw_labels = np.empty(n, dtype=int)
        raw_labels[sort_order] = sorted_labels

        # === Post-processing ===

        # Absorb noise into nearest non-noise cluster
        noise_mask = raw_labels == -1
        if noise_mask.all():
            return np.zeros(n, dtype=int)

        if noise_mask.any():
            from scipy.spatial.distance import cdist
            non_noise_idx = np.where(~noise_mask)[0]
            noise_idx = np.where(noise_mask)[0]
            dists = cdist(residuals_scaled[noise_idx],
                          residuals_scaled[non_noise_idx], metric='euclidean')
            nearest = np.argmin(dists, axis=1)
            raw_labels[noise_idx] = raw_labels[non_noise_idx[nearest]]

        # Remap to 0-based contiguous
        unique_labels = np.unique(raw_labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        final = np.array([label_map[l] for l in raw_labels], dtype=int)

        # Merge small clusters into nearest large cluster
        changed = True
        while changed:
            changed = False
            for lbl in np.unique(final):
                if np.sum(final == lbl) >= min_group_size:
                    continue
                small_idx = np.where(final == lbl)[0]
                small_centroid = residuals_scaled[small_idx].mean(axis=0)
                best_dist, best_target = float('inf'), -1
                for other in np.unique(final):
                    if other == lbl or np.sum(final == other) < min_group_size:
                        continue
                    other_idx = np.where(final == other)[0]
                    d = np.linalg.norm(small_centroid - residuals_scaled[other_idx].mean(axis=0))
                    if d < best_dist:
                        best_dist, best_target = d, other
                if best_target >= 0:
                    final[small_idx] = best_target
                    changed = True
                    break  # restart after merge

        # Re-compact
        unique_final = np.unique(final)
        remap = {old: new for new, old in enumerate(unique_final)}
        final = np.array([remap[l] for l in final], dtype=int)

        if len(np.unique(final)) <= 1:
            return np.zeros(n, dtype=int)

        return final

    def fit_hypervolume_tree(self, patterns: List[Any],
                             min_group_size: int = MIN_GROUP_SIZE,
                             r2_target: float = R2_STOP_THRESHOLD) -> HypervolumeTree:
        """
        Build hypervolume tree by iterative DOE fission.

        Pass 1: Initial recursive grouping (same as _split_at_depth)
        Pass 2+: Re-examine each leaf -- if R2 < target, attempt further fission
        Iterate until convergence or max iterations.
        """
        print(f"Hypervolume: Building tree from {len(patterns)} patterns (R2 target={r2_target:.2f})...")

        # 1. Build matrices
        matrices = {}

        for i, p in enumerate(patterns):
            mat = self.build_hypervolume_matrix(p)
            if mat is not None and mat.shape[0] >= 1:
                matrices[i] = mat

        print(f"  Constructed {len(matrices)} hypervolume matrices.")

        # Store matrices for DOE iterations
        self._doe_matrices = matrices
        self._doe_patterns = patterns

        # 2. Initial recursive grouping (Pass 1)
        root_nodes = self._split_at_depth(
            pattern_indices=list(matrices.keys()),
            matrices=matrices,
            patterns=patterns,
            depth=0,
            parent_id="",
            min_group_size=min_group_size
        )

        # 3. DOE Iteration Loop (Pass 2+)
        for doe_iter in range(DOE_MAX_ITERATIONS):
            leaves = self._collect_leaf_nodes(root_nodes)

            n_below_target = sum(1 for leaf in leaves
                                 if leaf.adj_r2_mfe < r2_target
                                 and leaf.member_count > min_group_size)
            n_terminal = sum(1 for leaf in leaves
                             if leaf.member_count <= min_group_size)

            print(f"  DOE iteration {doe_iter + 1}: {len(leaves)} leaves, "
                  f"{n_below_target} below R2={r2_target:.2f}, "
                  f"{n_terminal} terminal (min size)")

            if n_below_target == 0:
                print(f"  All leaves meet R2 target -- DOE converged.")
                break

            splits_made = 0
            for leaf in leaves:
                if leaf.adj_r2_mfe >= r2_target:
                    continue
                if leaf.member_count <= min_group_size:
                    continue

                # Attempt fission on this leaf at next depth
                children = self._split_at_depth(
                    pattern_indices=leaf.member_indices,
                    matrices=matrices,
                    patterns=patterns,
                    depth=leaf.depth + 1,
                    parent_id=leaf.node_id,
                    min_group_size=min_group_size
                )

                if children:
                    # Verify R2 gain
                    child_count = sum(c.member_count for c in children.values())
                    if child_count > 0:
                        child_r2_weighted = sum(
                            c.adj_r2_mfe * c.member_count for c in children.values()
                        ) / child_count
                        if child_r2_weighted >= leaf.adj_r2_mfe + R2_FISSION_MIN_GAIN:
                            leaf.children = children
                            leaf.template = None  # no longer a leaf
                            splits_made += 1

            if splits_made == 0:
                print(f"  No further splits possible -- DOE complete.")
                break

        max_depth = 0
        if matrices:
            max_depth = max(m.shape[0] for m in matrices.values()) - 1

        n_templates = self._count_leaves(root_nodes)
        print(f"  Tree built: {len(root_nodes)} roots, max depth {max_depth}, {n_templates} leaf templates.")

        tree = HypervolumeTree(roots=root_nodes, max_depth=max_depth, n_templates=n_templates)

        # Flatten templates list for orchestrator usage
        self.templates = self._collect_templates(root_nodes)

        # 4. Derive analytical params for all final leaves
        from training.orchestrator_worker import _analytical_exits
        for tmpl in self.templates:
            tmpl.best_params = _analytical_exits(tmpl)
            tmpl.best_params.update(DEFAULT_PID)

        # Clean up DOE state
        self._doe_matrices = None
        self._doe_patterns = None

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

        # Step 2: I-MR + Differential DBSCAN (replaces KMeans)
        scaler = StandardScaler()
        residuals_scaled = scaler.fit_transform(residuals)

        residual_var = np.mean(np.var(residuals_scaled, axis=0))

        if residual_var < 0.3:
            labels = np.zeros(len(valid_indices), dtype=int)
        else:
            labels = self._imr_geometric_split(residuals_scaled, min_group_size)

        # Step 3: Build Nodes
        unique_labels = sorted(set(labels))
        nodes = {}
        for cluster_id in unique_labels:
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
                member_indices=member_indices,
                children=children,
                template=template,
                node_id=node_id,
                scaler=scaler,
                regression_r2=regression_r2,
                branch_tightness=branch_tightness,
                adj_r2_mfe=cluster_adj_r2
            )

        return nodes

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

    def _collect_leaf_nodes(self, nodes: Dict[int, HypervolumeNode]) -> List[HypervolumeNode]:
        """Collect all leaf nodes (nodes with templates) for DOE iteration."""
        leaves = []
        for node in nodes.values():
            if node.template is not None:
                leaves.append(node)
            elif node.children:
                leaves.extend(self._collect_leaf_nodes(node.children))
        return leaves

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

    def build_occurrence_dataframe(self, root_nodes: Dict[int, HypervolumeNode],
                                    patterns: List[Any]) -> pd.DataFrame:
        """Build the occurrence DataFrame — ground-truth audit trail for all patterns.

        One row per pattern assigned to a leaf template, with temporal/spatial markers.
        Saved as checkpoints/template_occurrences.parquet.
        """
        occurrence_records = []
        leaf_nodes = self._collect_leaf_nodes(root_nodes)

        for node in leaf_nodes:
            if node.template is None:
                continue
            for idx in node.member_indices:
                if idx >= len(patterns):
                    continue
                p = patterns[idx]
                oracle_meta = getattr(p, 'oracle_meta', {}) or {}
                occurrence_records.append({
                    'template_id': node.template.template_id,
                    'node_id': node.node_id,
                    'timestamp': getattr(p, 'timestamp', 0.0),
                    'price': getattr(p, 'price', 0.0),
                    'timeframe': getattr(p, 'timeframe', ''),
                    'depth': getattr(p, 'depth', 0),
                    'bar_index': getattr(p, 'idx', 0),
                    'file_source': getattr(p, 'file_source', ''),
                    'oracle_mfe': oracle_meta.get('mfe', 0.0),
                    'oracle_mae': oracle_meta.get('mae', 0.0),
                    'adj_r2': node.adj_r2_mfe,
                })

        if occurrence_records:
            df = pd.DataFrame(occurrence_records)
            return df
        return pd.DataFrame()

    def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]],
                        original_patterns: List[Any]) -> List[PatternTemplate]:
        """
        Legacy fission support for Phase 3.
        Hypervolume Tree handles splitting during construction, so we disable
        post-hoc fission here by returning empty list (no split).
        """
        return []
