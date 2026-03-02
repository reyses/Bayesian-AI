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
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# GPU-accelerated distance computation (replaces scipy.spatial.distance.cdist)
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _gpu_cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared-Euclidean distance on GPU."""
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(_DEVICE)
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to(_DEVICE)
    d = torch.cdist(ta, tb)
    return (d ** 2).cpu().numpy()

# Core import for vector reconstruction
from core.quantum_field_engine import QuantumFieldEngine

# Local import of timeframe mapping
from training.fractal_discovery_agent import TIMEFRAME_SECONDS
from config.oracle_config import TEMPLATE_MIN_MEMBERS_FOR_STATS

# Constants
MIN_GROUP_SIZE = 30
R2_STOP_THRESHOLD = 0.90       # DOE Phase 3: target R2 for "real case usage" (was 0.15)
R2_FISSION_MIN_GAIN = 0.01
DOE_MAX_ITERATIONS = 20         # DOE Phase 3: max passes over the tree (safety valve)
DEFAULT_PID = {'pid_kp': 0.5, 'pid_ki': 0.1, 'pid_kd': 0.2}

# I-MR SPC constants (n=2 subgroups)
IMR_D4 = 3.267          # UCL factor for Moving Range (standard SPC for n=2 subgroups)
IMR_D2 = 1.128          # Sigma estimation factor
IMR_Z_THRESHOLD = 1.0   # Composite signal threshold for segment boundary
IMR_MIN_SEGMENTS = 2    # Force at least this many segments via PC1 median fallback

# Parent DNA Matching: dimensions comparable between live state_to_features() and clustering extract_features()
# Excludes: [5]=depth, [6]=parent_ctx, [10]=parent_z, [11]=parent_dmi_diff, [12]=root_is_roche, [13]=tf_alignment
DNA_LIVE_DIMS = [0, 1, 2, 3, 4, 7, 8, 9, 14, 15]

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
    dir_coeff: Optional[List[float]] = None       # balanced LONG-vs-SHORT (fallback)
    dir_intercept: float = 0.0
    quality_coeff: Optional[List[float]] = None    # within-side good-vs-bad entry
    quality_intercept: float = 0.0
    signed_mfe_coeff: Optional[List[float]] = None  # signed MFE: sign=direction, |val|=magnitude
    signed_mfe_intercept: float = 0.0

    # CST Basin (legacy field support - now handled by node cell bounds)
    basin_mean: float = 0.0
    basin_std: float = 0.0

    # DOE Phase 3 validation
    consistency_score: float = 0.0
    consistency_diagnostics: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None

    # Parent DNA Matching: per-TF DNA centroids and bounds from member parent chains
    tf_depth_map: Dict[str, int] = field(default_factory=dict)
    dna_centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    dna_bounds_min: Dict[str, np.ndarray] = field(default_factory=dict)
    dna_bounds_max: Dict[str, np.ndarray] = field(default_factory=dict)

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

        return [z, v_feat, m_feat, c, tf_scale, depth, parent_ctx,
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

    # Indices into the 16D feature vector used for clustering
    # [1]=log1p(volume), [7]=self_adx — market dynamics, not abstract features
    CLUSTER_DIMS = [1, 7]  # volume + ADX

    def _imr_geometric_split(self, feat_scaled: np.ndarray,
                              min_group_size: int) -> np.ndarray:
        """Directional I-MR segmentation + DBSCAN fission.

        Two phases:
          1. I-MR on DMI differential: Patterns sorted by ADX (trend strength),
             MR computed as signed diff of DMI_diff — captures directional acceleration.
             Boundaries fire on |MR| > UCL (big jump) or MR sign flip (reversal).
          2. DBSCAN on volume + ADX: Within each segment, find density clusters.
             The full 16D is kept as identity fingerprint, not used for clustering.

        Returns: (labels, lineage) where:
          labels: int array (0-based contiguous, original order)
          lineage: dict {global_label: (segment_id, sub_cluster_id)}
        """
        n = len(feat_scaled)

        if n < min_group_size * 2:
            return np.zeros(n, dtype=int), {0: (0, 0)}

        # === PHASE 1: I-MR on directional DMI — shape-aware segmentation ===
        #
        # I chart:  DMI_diff (signed) — captures directional movement shape
        #           "big long, small short, big long, bigger long"
        # Sort by: ADX (trend strength) — orders patterns from calm → trending
        # MR:      Signed diff of DMI_diff — captures acceleration/deceleration
        # Boundary: |MR| > UCL  OR  MR sign flips (trend reversal)
        #
        adx_col = 7   # self_adx (trend strength, unsigned)
        dmi_col = 9   # self_dmi_diff (directional bias, signed)

        adx_values = feat_scaled[:, adx_col]
        dmi_values = feat_scaled[:, dmi_col]

        # Sort by ADX (trend strength) — the process ordering
        sort_order = np.argsort(adx_values)
        feat_ordered = feat_scaled[sort_order]

        # I chart: DMI_diff values in ADX-sorted order
        dmi_sorted = dmi_values[sort_order]

        # MR: signed differential — captures directional acceleration
        mr_signed = np.diff(dmi_sorted)
        mr_abs = np.abs(mr_signed)

        if len(mr_abs) == 0 or np.max(mr_abs) < 1e-12:
            return np.zeros(n, dtype=int), {0: (0, 0)}

        mr_bar = np.mean(mr_abs)
        ucl_mr = IMR_D4 * mr_bar

        # Boundary conditions:
        # 1. |MR| > UCL — large directional jump (standard SPC)
        # 2. Sign flip in MR — trend was accelerating, now decelerating (or vice versa)
        magnitude_break = mr_abs > ucl_mr
        sign_flip = np.zeros(len(mr_signed), dtype=bool)
        if len(mr_signed) > 1:
            sign_flip[1:] = (mr_signed[1:] * mr_signed[:-1]) < 0  # sign changed

        # Combine: magnitude break OR sign flip with significant MR
        significant_flip = sign_flip & (mr_abs > mr_bar * 0.5)
        boundary_flags = (magnitude_break | significant_flip).astype(int)

        sorted_segment_ids = np.zeros(n, dtype=int)
        sorted_segment_ids[1:] = np.cumsum(boundary_flags)
        n_segments = sorted_segment_ids[-1] + 1

        # Fallback: force binary split if I-MR found nothing
        fallback_used = False
        if n_segments < IMR_MIN_SEGMENTS and n >= min_group_size * 3:
            median_idx = n // 2
            sorted_segment_ids[median_idx:] = 1
            n_segments = 2
            fallback_used = True

        # Merge tiny adjacent segments (< min_group_size) with their neighbor
        # This prevents I-MR from shredding the data into thousands of micro-segments
        merge_pass = 0
        while True:
            seg_ids_unique = np.unique(sorted_segment_ids)
            seg_sizes_map = {s: int((sorted_segment_ids == s).sum()) for s in seg_ids_unique}
            tiny = [s for s in seg_ids_unique if seg_sizes_map[s] < min_group_size]
            if not tiny:
                break
            # Merge the smallest segment into its nearest neighbor (by segment ID)
            s = tiny[0]
            s_pos = np.where(seg_ids_unique == s)[0][0]
            if s_pos > 0 and s_pos < len(seg_ids_unique) - 1:
                left, right = seg_ids_unique[s_pos - 1], seg_ids_unique[s_pos + 1]
                target = left if seg_sizes_map[left] >= seg_sizes_map[right] else right
            elif s_pos > 0:
                target = seg_ids_unique[s_pos - 1]
            else:
                target = seg_ids_unique[s_pos + 1] if len(seg_ids_unique) > 1 else s
            if target == s:
                break
            sorted_segment_ids[sorted_segment_ids == s] = target
            merge_pass += 1
            if merge_pass > n:  # safety
                break

        # Re-compact segment IDs to 0-based contiguous
        seg_ids_unique = np.unique(sorted_segment_ids)
        seg_remap = {old: new for new, old in enumerate(seg_ids_unique)}
        sorted_segment_ids = np.array([seg_remap[s] for s in sorted_segment_ids])
        n_segments = len(seg_ids_unique)

        # Report I-MR segments (compact summary)
        seg_sizes = sorted([int((sorted_segment_ids == s).sum()) for s in range(n_segments)], reverse=True)
        n_merged = merge_pass
        print(f"    I-MR (DMI→ADX): {n_segments} segments │ UCL={ucl_mr:.3f}  MR_bar={mr_bar:.3f}"
              f"{'  [median fallback]' if fallback_used else ''}"
              f"{f'  [{n_merged} micro-segs merged]' if n_merged else ''}")
        # Show top 5 largest + tail summary
        top = seg_sizes[:5]
        rest = seg_sizes[5:]
        sizes_str = ", ".join(f"{s:,}" for s in top)
        if rest:
            sizes_str += f"  ... +{len(rest)} more (min={min(rest):,})"
        print(f"      Sizes: [{sizes_str}]")

        # === PHASE 2: DBSCAN on volume + ADX (2D) within each I-MR segment ===

        cluster_dims = self.CLUSTER_DIMS  # [1, 7] = volume + ADX
        sorted_labels = np.zeros(n, dtype=int)
        global_label = 0
        lineage = {}  # global_label → (segment_id, sub_cluster_id)
        dbscan_min = max(3, min_group_size // 10)
        _db_stats = {'segs_scanned': 0, 'segs_too_small': 0, 'segs_all_noise': 0,
                     'total_clusters': 0, 'total_noise': 0, 'eps_values': []}

        for seg_id in range(n_segments):
            seg_mask = sorted_segment_ids == seg_id
            seg_indices = np.where(seg_mask)[0]
            seg_size = len(seg_indices)
            _db_stats['segs_scanned'] += 1

            if seg_size < max(4, dbscan_min + 2):
                sorted_labels[seg_indices] = global_label
                lineage[global_label] = (seg_id, 0)
                global_label += 1
                _db_stats['segs_too_small'] += 1
                continue

            # Pull volume + ADX features only (2D) for this segment
            seg_feat_2d = feat_ordered[seg_indices][:, cluster_dims]

            # Calibrate eps from sampled k-NN distances (O(n log n), not O(n²))
            k_nn = min(5, seg_size - 1)
            if k_nn < 1:
                sorted_labels[seg_indices] = global_label
                lineage[global_label] = (seg_id, 0)
                global_label += 1
                continue

            EPS_SAMPLE_CAP = 1000
            if seg_size > EPS_SAMPLE_CAP:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(seg_size, EPS_SAMPLE_CAP, replace=False)
                sample_feat = seg_feat_2d[sample_idx]
            else:
                sample_feat = seg_feat_2d

            nn = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='ball_tree')
            nn.fit(sample_feat)
            knn_dists, _ = nn.kneighbors(sample_feat)
            eps = float(np.median(knn_dists[:, -1]))
            if eps < 1e-12:
                eps = 0.5
            _db_stats['eps_values'].append(eps)

            db_labels = DBSCAN(eps=eps, min_samples=dbscan_min,
                               metric='euclidean', algorithm='ball_tree'
                               ).fit_predict(seg_feat_2d)

            n_noise = int((db_labels == -1).sum())
            n_clusters = len(set(db_labels[db_labels >= 0]))
            _db_stats['total_noise'] += n_noise
            _db_stats['total_clusters'] += n_clusters

            # Map DBSCAN labels to global labels with lineage tracking
            unique_sub = np.unique(db_labels[db_labels >= 0])
            if len(unique_sub) == 0:
                sorted_labels[seg_indices] = global_label
                lineage[global_label] = (seg_id, 0)
                global_label += 1
                _db_stats['segs_all_noise'] += 1
            else:
                sub_map = {int(s): global_label + i for i, s in enumerate(unique_sub)}
                for sub_idx, s in enumerate(unique_sub):
                    lineage[global_label + sub_idx] = (seg_id, sub_idx)
                for j, idx in enumerate(seg_indices):
                    sl = db_labels[j]
                    sorted_labels[idx] = sub_map.get(sl, -1)
                global_label += len(unique_sub)

        # DBSCAN summary
        eps_arr = _db_stats['eps_values']
        eps_summary = f"eps: median={np.median(eps_arr):.3f}  range=[{min(eps_arr):.3f}, {max(eps_arr):.3f}]" if eps_arr else "eps: n/a"
        print(f"    DBSCAN (vol+ADX): {_db_stats['total_clusters']} clusters from "
              f"{_db_stats['segs_scanned']} segments │ {eps_summary}")
        print(f"      noise={_db_stats['total_noise']:,} absorbed │ "
              f"{_db_stats['segs_too_small']} segs too small │ "
              f"{_db_stats['segs_all_noise']} segs all-noise")

        # === Map back to original order ===
        raw_labels = np.empty(n, dtype=int)
        raw_labels[sort_order] = sorted_labels

        # Absorb noise into nearest cluster (using cluster dims only)
        noise_mask = raw_labels == -1
        if noise_mask.all():
            return np.zeros(n, dtype=int), {0: (0, 0)}

        if noise_mask.any():
            non_noise_idx = np.where(~noise_mask)[0]
            noise_idx = np.where(noise_mask)[0]
            feat_2d = feat_scaled[:, cluster_dims]
            dists = _gpu_cdist(feat_2d[noise_idx], feat_2d[non_noise_idx])
            nearest = np.argmin(dists, axis=1)
            raw_labels[noise_idx] = raw_labels[non_noise_idx[nearest]]

        # Remap to 0-based contiguous
        unique_labels = np.unique(raw_labels)
        remap = {old: new for new, old in enumerate(unique_labels)}
        final = np.array([remap[l] for l in raw_labels])

        # Remap lineage to match new contiguous labels
        new_lineage = {}
        for old_lbl, new_lbl in remap.items():
            if old_lbl in lineage:
                new_lineage[new_lbl] = lineage[old_lbl]
            else:
                new_lineage[new_lbl] = (0, new_lbl)

        # Merge small clusters into nearest large (using cluster dims)
        feat_2d = feat_scaled[:, cluster_dims]
        while True:
            labels_unique, counts = np.unique(final, return_counts=True)
            small_mask = counts < min_group_size
            if not small_mask.any():
                break
            large_mask = ~small_mask
            if not large_mask.any():
                return np.zeros(n, dtype=int), {0: (0, 0)}

            centroids = np.array([feat_2d[final == l].mean(axis=0)
                                  for l in labels_unique])
            large_labels = labels_unique[large_mask]
            large_centroids = centroids[large_mask]

            small_idx_pos = np.where(small_mask)[0][0]
            small_lbl = labels_unique[small_idx_pos]
            small_centroid = centroids[small_idx_pos]

            dists_to_large = np.linalg.norm(large_centroids - small_centroid, axis=1)
            best_target = large_labels[np.argmin(dists_to_large)]
            final[final == small_lbl] = best_target
            # Merged label inherits target's lineage
            if best_target in new_lineage:
                new_lineage[small_lbl] = new_lineage[best_target]

        # Final re-compact
        unique_final = np.unique(final)
        remap2 = {old: new for new, old in enumerate(unique_final)}
        final = np.array([remap2[l] for l in final])
        final_lineage = {remap2[old]: new_lineage.get(old, (0, 0)) for old in unique_final}

        if len(unique_final) <= 1:
            return np.zeros(n, dtype=int), {0: final_lineage.get(0, (0, 0))}

        return final, final_lineage

    def fit_hypervolume_tree(self, patterns: List[Any],
                             min_group_size: int = MIN_GROUP_SIZE) -> HypervolumeTree:
        """
        Build hypervolume tree:
          I-MR on PC1 → coarse geometry segments
          DBSCAN on full 16D within each segment → final templates
          Forward pass handles matching + regression
        """
        import hashlib
        import time

        t0 = time.perf_counter()
        print(f"\n{'='*70}")
        print(f"  HYPERVOLUME TREE  │  {len(patterns):,} patterns")
        print(f"{'='*70}")

        # 1. Build matrices
        matrices = {}
        for i, p in enumerate(patterns):
            mat = self.build_hypervolume_matrix(p)
            if mat is not None and mat.shape[0] >= 1:
                matrices[i] = mat

        t1 = time.perf_counter()
        print(f"  [1] Matrices: {len(matrices):,} built ({len(patterns)-len(matrices):,} skipped) [{t1-t0:.1f}s]")

        # 2. Extract full 16D features
        all_indices = list(matrices.keys())
        feat_all = np.array([self.extract_features(patterns[i]) for i in all_indices])

        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_all)

        t2 = time.perf_counter()
        print(f"  [2] Features: 16D extracted + scaled [{t2-t1:.1f}s]")

        # 3. Pre-split by DMI direction, then I-MR + DBSCAN within each side
        #    DMI_DIFF (index 9): >0 = bullish, <0 = bearish
        #    This guarantees every template has a pure directional signal.
        dmi_col = 9  # self_dmi_diff
        dmi_vals = feat_scaled[:, dmi_col]
        long_mask = dmi_vals >= 0
        short_mask = ~long_mask

        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        print(f"  [3] Direction split: {n_long:,} LONG │ {n_short:,} SHORT (DMI)")

        labels = np.full(len(feat_scaled), -1, dtype=int)
        lineage = {}
        offset = 0

        for side_name, side_mask in [('LONG', long_mask), ('SHORT', short_mask)]:
            side_n = int(side_mask.sum())
            if side_n < min_group_size:
                # Too few — assign all to one group
                labels[side_mask] = offset
                lineage[offset] = (offset, 0)
                offset += 1
                print(f"      {side_name}: {side_n:,} patterns → 1 group (too small)")
                continue

            print(f"      {side_name}: I-MR(ADX) → DBSCAN(vol+ADX) on {side_n:,} patterns...")
            side_labels, side_lineage = self._imr_geometric_split(
                feat_scaled[side_mask], min_group_size)

            # Remap to global label space
            for local_lbl in np.unique(side_labels):
                global_lbl = offset + local_lbl
                labels[np.where(side_mask)[0][side_labels == local_lbl]] = global_lbl
                seg_id, sub_id = side_lineage.get(local_lbl, (local_lbl, 0))
                lineage[global_lbl] = (seg_id + offset, sub_id)

            offset += len(set(side_labels))

        n_groups = len(set(labels) - {-1})
        t3 = time.perf_counter()
        print(f"    → {n_groups} final groups [{t3-t2:.1f}s]")

        # === Build templates ===
        print(f"  [4] Templates:")
        root_nodes = {}
        unique_labels = sorted(set(labels) - {-1})
        _skipped = 0

        for cluster_id in unique_labels:
            mask = labels == cluster_id
            member_indices = [all_indices[i] for i, m in enumerate(mask) if m]
            if len(member_indices) < min_group_size:
                _skipped += 1
                continue

            cluster_feat = feat_all[mask]
            centroid = cluster_feat.mean(axis=0)
            cell_min = cluster_feat.min(axis=0)
            cell_max = cluster_feat.max(axis=0)
            margin = (cell_max - cell_min) * 0.05
            cell_min -= margin
            cell_max += margin

            seg_id, sub_id = lineage.get(cluster_id, (cluster_id, 0))
            seg_letter = chr(ord('A') + seg_id % 26)

            # Direction from mean DMI of cluster members
            _dmi_mean = float(feat_all[mask, 9].mean())  # self_dmi_diff
            _dir_tag = 'L' if _dmi_mean >= 0 else 'S'
            node_id = f"{_dir_tag}{seg_letter}:{sub_id}"
            tid = int(hashlib.sha256(node_id.encode('utf-8')).hexdigest(), 16) % 1000000

            member_patterns = [patterns[i] for i in member_indices]

            # Compute mean ADX and volume for this cluster (for logging)
            _adx_mean = float(feat_all[mask, 7].mean())
            _vol_mean = float(feat_all[mask, 1].mean())

            template = PatternTemplate(
                template_id=tid,
                centroid=centroid,
                member_count=len(member_patterns),
                patterns=member_patterns,
                physics_variance=float(np.mean(np.var(feat_scaled[mask], axis=0))),
                basin_mean=0.0,
                basin_std=0.0
            )
            self._aggregate_oracle_intelligence(template)
            self._build_dna_maps(template)

            print(f"      {node_id:>7s} │ {len(member_patterns):>5,} patterns │ "
                  f"ADX={_adx_mean:.2f}  vol={_vol_mean:.2f}  DMI={_dmi_mean:+.3f} │ tid={tid}")

            root_nodes[cluster_id] = HypervolumeNode(
                depth=0,
                centroid_16d=centroid,
                cell_min_16d=cell_min,
                cell_max_16d=cell_max,
                member_count=len(member_indices),
                member_indices=member_indices,
                children={},
                template=template,
                node_id=node_id,
                scaler=scaler,
                regression_r2=0.0,
                branch_tightness=float(np.mean(np.var(feat_scaled[mask], axis=0))),
                adj_r2_mfe=0.0
            )

        t4 = time.perf_counter()
        max_depth = max((m.shape[0] for m in matrices.values()), default=1) - 1
        n_templates = len(root_nodes)
        print(f"  Tree built: {n_templates} templates. [templates {t4-t3:.1f}s, total {t4-t0:.1f}s]")

        tree = HypervolumeTree(roots=root_nodes, max_depth=max_depth, n_templates=n_templates)
        self.templates = list(t.template for t in root_nodes.values() if t.template)

        # Derive analytical params
        from training.orchestrator_worker import _analytical_exits
        for tmpl in self.templates:
            tmpl.best_params = _analytical_exits(tmpl)
            tmpl.best_params.update(DEFAULT_PID)

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

        if residual_var < 0.10:
            # Regression absorbed all variance — split on raw centered features instead
            raw_centered = feat_array - feat_array.mean(axis=0)
            raw_scaled = StandardScaler().fit_transform(raw_centered)
            labels, _ = self._imr_geometric_split(raw_scaled, min_group_size)
        else:
            labels, _ = self._imr_geometric_split(residuals_scaled, min_group_size)

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
                self._build_dna_maps(template)

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

    def _similarity_sort_and_refine(self, feat_scaled: np.ndarray, labels: np.ndarray,
                                     min_group_size: int) -> np.ndarray:
        """Phase B: Sort groups by centroid proximity, refine boundary patterns.

        1. Compute group centroids
        2. Sort groups by nearest-neighbor chain (similar groups adjacent)
        3. Shift boundary patterns to their nearest centroid
        """
        labels = labels.copy()
        unique_labels = sorted(set(labels))
        if len(unique_labels) <= 1:
            return labels

        # Compute centroids
        centroids = {}
        for lbl in unique_labels:
            centroids[lbl] = feat_scaled[labels == lbl].mean(axis=0)

        # Sort groups by nearest-neighbor chain starting from largest group
        sorted_order = []
        remaining = set(unique_labels)
        # Start from the largest group
        current = max(remaining, key=lambda l: int((labels == l).sum()))
        sorted_order.append(current)
        remaining.remove(current)

        while remaining:
            c = centroids[current]
            best_lbl = min(remaining, key=lambda l: float(np.linalg.norm(centroids[l] - c)))
            sorted_order.append(best_lbl)
            remaining.remove(best_lbl)
            current = best_lbl

        # Remap labels to sorted order (0, 1, 2, ...)
        remap = {old: new for new, old in enumerate(sorted_order)}
        labels = np.array([remap[l] for l in labels])

        # Recompute centroids in new label space
        n_groups = len(sorted_order)
        new_centroids = np.array([feat_scaled[labels == i].mean(axis=0) for i in range(n_groups)])

        # Boundary refinement: for each pattern, check if nearest centroid differs from label
        dists_to_centroids = _gpu_cdist(feat_scaled, new_centroids)  # (N, n_groups), sqeuclidean
        nearest = np.argmin(dists_to_centroids, axis=1)

        # Only shift if the nearest centroid is different AND the target group stays large enough
        shifts = 0
        for i in range(len(labels)):
            if nearest[i] != labels[i]:
                old_label = labels[i]
                old_count = int((labels == old_label).sum())
                if old_count > min_group_size:  # Don't shrink groups below min
                    labels[i] = nearest[i]
                    shifts += 1

        # Remove any groups that fell below min_group_size after shifting
        for lbl in range(n_groups):
            count = int((labels == lbl).sum())
            if 0 < count < min_group_size:
                # Merge into nearest large group
                lbl_centroid = feat_scaled[labels == lbl].mean(axis=0)
                large_labels = [l for l in range(n_groups) if int((labels == l).sum()) >= min_group_size]
                if large_labels:
                    best = min(large_labels,
                               key=lambda l: float(np.linalg.norm(new_centroids[l] - lbl_centroid)))
                    labels[labels == lbl] = best

        # Re-compact
        unique_map = {old: new for new, old in enumerate(sorted(set(labels)))}
        labels = np.array([unique_map[l] for l in labels])

        if shifts > 0:
            print(f"    Shifted {shifts} boundary patterns to nearest centroid")

        return labels

    def _statistical_merge_split(self, feat_scaled: np.ndarray, labels: np.ndarray,
                                  oracle_mfe: np.ndarray, min_group_size: int) -> np.ndarray:
        """Phase C: Statistically validate groups (single-pass, no restarts).

        1. Merge pass: test all adjacent pairs at once, batch-merge non-significant
        2. Split pass: test all large groups at once, batch-split heterogeneous ones
        """
        labels = labels.copy()
        STAT_ALPHA = 0.05

        # === MERGE PASS (single sweep) ===
        unique_labels = sorted(set(labels))
        if len(unique_labels) > 1:
            # Build merge graph: which adjacent pairs should merge?
            merge_into = {}  # lbl_b → lbl_a
            for i in range(len(unique_labels) - 1):
                lbl_a, lbl_b = unique_labels[i], unique_labels[i + 1]
                mfe_a = oracle_mfe[labels == lbl_a]
                mfe_b = oracle_mfe[labels == lbl_b]

                if len(mfe_a) < 10 or len(mfe_b) < 10:
                    continue

                if np.std(mfe_a) < 1e-9 and np.std(mfe_b) < 1e-9:
                    merge_into[lbl_b] = lbl_a
                    continue

                try:
                    _, p_val = f_oneway(mfe_a, mfe_b)
                    if p_val > STAT_ALPHA:
                        merge_into[lbl_b] = lbl_a
                except Exception:
                    pass

            # Apply merges (chase chains: if C→B→A, C should go to A)
            if merge_into:
                for lbl_b in merge_into:
                    target = merge_into[lbl_b]
                    while target in merge_into:
                        target = merge_into[target]
                    labels[labels == lbl_b] = target

        # Re-compact
        umap = {old: new for new, old in enumerate(sorted(set(labels)))}
        labels = np.array([umap[l] for l in labels])
        n_after_merge = len(set(labels))

        # === SPLIT PASS (single sweep) ===
        unique_labels = sorted(set(labels))
        next_label = labels.max() + 1

        for lbl in unique_labels:
            mask = labels == lbl
            count = int(mask.sum())
            if count < min_group_size * 2:
                continue

            group_mfe = oracle_mfe[mask]
            group_feat = feat_scaled[mask]

            # PC1 median split + F-test
            centered = group_feat - group_feat.mean(axis=0)
            cov = centered.T @ centered
            try:
                _, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue

            proj = centered @ eigvecs[:, -1]
            half_a = proj <= np.median(proj)
            half_b = ~half_a

            if half_a.sum() < 10 or half_b.sum() < 10:
                continue

            try:
                _, p_val = f_oneway(group_mfe[half_a], group_mfe[half_b])
            except Exception:
                continue

            if p_val < STAT_ALPHA:
                group_indices = np.where(mask)[0]
                labels[group_indices[half_b]] = next_label
                next_label += 1

        # Re-compact
        umap = {old: new for new, old in enumerate(sorted(set(labels)))}
        labels = np.array([umap[l] for l in labels])
        n_after_split = len(set(labels))

        print(f"    Merge: {n_after_merge} groups, Split: {n_after_split} groups")
        return labels

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

    @staticmethod
    def _stepwise_ols(X: np.ndarray, y: np.ndarray, p_enter: float = 0.05) -> List[int]:
        """Forward stepwise OLS — add features whose F-test p < p_enter."""
        n, p = X.shape
        remaining = list(range(p))
        selected = []
        current_rss = np.sum((y - np.mean(y)) ** 2)

        for _ in range(p):
            best_dim, best_pval = None, 1.0
            for dim in remaining:
                trial = selected + [dim]
                ols = LinearRegression().fit(X[:, trial], y)
                rss_new = np.sum((y - ols.predict(X[:, trial])) ** 2)
                # Partial F-test: does adding this dim significantly reduce RSS?
                df1, df2 = 1, n - len(trial) - 1
                if df2 <= 0 or rss_new <= 0:
                    continue
                f_stat = ((current_rss - rss_new) / df1) / (rss_new / df2)
                if f_stat > 0:
                    from scipy.stats import f as f_dist
                    pval = 1.0 - f_dist.cdf(f_stat, df1, df2)
                    if pval < best_pval:
                        best_pval = pval
                        best_dim = dim
            if best_dim is None or best_pval >= p_enter:
                break
            selected.append(best_dim)
            remaining.remove(best_dim)
            ols = LinearRegression().fit(X[:, selected], y)
            current_rss = np.sum((y - ols.predict(X[:, selected])) ** 2)
        return selected

    @staticmethod
    def _stepwise_logistic(X: np.ndarray, y: np.ndarray, p_enter: float = 0.05) -> List[int]:
        """Forward stepwise logistic — add features that improve log-likelihood (LRT p < p_enter)."""
        from scipy.stats import chi2
        n, p = X.shape
        remaining = list(range(p))
        selected = []

        # Null model log-likelihood
        p_bar = np.mean(y)
        if p_bar <= 0 or p_bar >= 1:
            return []
        ll_current = n * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))

        for _ in range(p):
            best_dim, best_pval = None, 1.0
            for dim in remaining:
                trial = selected + [dim]
                try:
                    lr = LogisticRegression(max_iter=300).fit(X[:, trial], y)
                    probs = lr.predict_proba(X[:, trial])[:, 1]
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    ll_new = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                    # Likelihood ratio test
                    lr_stat = 2 * (ll_new - ll_current)
                    if lr_stat > 0:
                        pval = 1.0 - chi2.cdf(lr_stat, df=1)
                        if pval < best_pval:
                            best_pval = pval
                            best_dim = dim
                except:
                    continue
            if best_dim is None or best_pval >= p_enter:
                break
            selected.append(best_dim)
            remaining.remove(best_dim)
            lr = LogisticRegression(max_iter=300).fit(X[:, selected], y)
            probs = lr.predict_proba(X[:, selected])[:, 1]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            ll_current = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        return selected

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

                 # adj-R²: how well 16D features predict MFE
                 n, k = X_sc.shape
                 r2 = ols.score(X_sc, mfe_y)
                 template.adj_r2_mfe = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)
             except:
                 template.regression_sigma_ticks = 0.0
                 template.adj_r2_mfe = 0.0

             # ── Direction regression (balanced fallback for bypass path) ──
             # LONG=1, SHORT=0 — class_weight='balanced' eliminates training imbalance
             dir_labels = np.array([1 if m > 0 else 0 for m in markers if m != 0])
             if len(dir_labels) >= 20 and len(np.unique(dir_labels)) == 2:
                 X_dir = np.array([self.extract_features(p) for p in patterns if p.oracle_marker != 0])
                 X_dir_sc = sc.transform(X_dir)
                 _imp_w = np.ones(X_dir_sc.shape[1])
                 _imp_w[8]  = 1.50  # hurst (#1)
                 _imp_w[15] = 1.45  # osc_coh (#2)
                 _imp_w[7]  = 1.35  # adx (#4)
                 _imp_w[9]  = 1.30  # dmi_diff (#5)
                 _imp_w[0]  = 1.25  # z_score (#6)
                 X_dir_wt = X_dir_sc * _imp_w
                 try:
                     lr = LogisticRegression(max_iter=300, class_weight='balanced').fit(X_dir_wt, dir_labels)
                     template.dir_coeff = lr.coef_[0].tolist()
                     template.dir_intercept = float(lr.intercept_[0])
                 except:
                     pass

             # ── Quality regression (within-side: good entry vs bad entry) ──
             # Direction is known from DMI pre-split. This model predicts
             # P(profitable) for entries in the template's direction.
             # Label 1 = MFE above median (good entry), 0 = below (bad entry)
             _side_sign = 1 if template.direction == 'LONG' else -1
             _same_side = [(p, m) for p, m in zip(patterns, markers)
                           if m != 0 and ((m > 0) == (_side_sign > 0))]
             if len(_same_side) >= 20:
                 _qs_patterns, _qs_markers = zip(*_same_side)
                 _qs_mfes = np.array([getattr(p, 'oracle_mfe', 0.0) for p in _qs_patterns])
                 _qs_median = float(np.median(_qs_mfes))
                 _qs_labels = np.array([1 if mfe > _qs_median else 0 for mfe in _qs_mfes])
                 if len(np.unique(_qs_labels)) == 2:
                     X_qs = np.array([self.extract_features(p) for p in _qs_patterns])
                     X_qs_sc = sc.transform(X_qs)
                     X_qs_wt = X_qs_sc * _imp_w
                     try:
                         lr_q = LogisticRegression(max_iter=300).fit(X_qs_wt, _qs_labels)
                         template.quality_coeff = lr_q.coef_[0].tolist()
                         template.quality_intercept = float(lr_q.intercept_[0])
                     except:
                         pass

             # ── Signed MFE regression (primary: sign=direction, |val|=magnitude) ──
             # Target: MFE * sign(oracle_marker) — positive=up, negative=down
             # One model replaces direction + quality: sign → side, magnitude → confidence + TP
             _non_noise_idx = [i for i, m in enumerate(markers) if m != 0]
             if len(_non_noise_idx) >= 20:
                 _smfe_patterns = [patterns[i] for i in _non_noise_idx]
                 _smfe_markers  = [markers[i]  for i in _non_noise_idx]
                 _smfe_y = np.array([
                     getattr(p, 'oracle_mfe', 0.0) * (1.0 if m > 0 else -1.0)
                     for p, m in zip(_smfe_patterns, _smfe_markers)
                 ])
                 X_smfe = np.array([self.extract_features(p) for p in _smfe_patterns])
                 X_smfe_sc = sc.transform(X_smfe)
                 X_smfe_wt = X_smfe_sc * _imp_w
                 try:
                     ols_s = LinearRegression().fit(X_smfe_wt, _smfe_y)
                     template.signed_mfe_coeff = ols_s.coef_.tolist()
                     template.signed_mfe_intercept = float(ols_s.intercept_)
                 except:
                     pass

    def _build_dna_maps(self, template: PatternTemplate):
        """Build per-TF DNA centroids and bounds from member parent chains.

        For each TF in the members' ancestry, computes the mean 16D centroid
        and min/max bounds on the 10 live-comparable dimensions. Used by the
        belief network for parent-anchor matching (15m) and DNA verification.
        """
        tf_features = {}  # TF label → list of 16D arrays

        for p in template.patterns:
            chain = getattr(p, 'parent_chain', None) or []
            for entry in chain:
                tf = entry.get('tf')
                feat = entry.get('features_16d')
                if tf and feat is not None:
                    tf_features.setdefault(tf, []).append(np.array(feat))
            # Also add self
            self_feat = self.extract_features(p)
            tf_features.setdefault(p.timeframe, []).append(np.array(self_feat))

        mask = DNA_LIVE_DIMS
        for tf, feats in tf_features.items():
            if len(feats) < 3:
                continue
            arr = np.array(feats)
            template.dna_centroids[tf] = arr.mean(axis=0)

            arr_live = arr[:, mask]
            mins = arr_live.min(axis=0)
            maxs = arr_live.max(axis=0)
            margin = (maxs - mins) * 0.05
            margin = np.maximum(margin, 0.01)
            template.dna_bounds_min[tf] = mins - margin
            template.dna_bounds_max[tf] = maxs + margin

        # Build depth map from first member's chain
        if template.patterns:
            p0 = template.patterns[0]
            chain = getattr(p0, 'parent_chain', None) or []
            rev_chain = list(reversed(chain))
            for depth_idx, entry in enumerate(rev_chain):
                tf = entry.get('tf')
                if tf:
                    template.tf_depth_map[tf] = depth_idx
            template.tf_depth_map[p0.timeframe] = len(rev_chain)

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
