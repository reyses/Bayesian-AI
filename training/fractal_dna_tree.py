from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pickle

@dataclass
class TreeNode:
    """
    One node in the fractal tree.
    At each TF level, this is a cluster of patterns sharing similar context at that TF.
    """
    node_id: str              # Unique ID: "L|1h:3" or "15m:7" (root nodes include direction)
    direction: str            # 'LONG', 'SHORT', or '' (intermediate nodes inherit from root)
    timeframe: str            # '1h', '30m', '15m', '5m', '3m', '1m', '30s', '15s'
    centroid: np.ndarray      # Cluster centroid in scaled feature space at this TF level
    member_count: int         # Number of leaf patterns under this subtree
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)
    leaf_pattern_ids: List[int] = field(default_factory=list)  # leaf-only: pattern indices

    # Oracle stats aggregated from ALL leaf patterns in this subtree
    win_rate: float = 0.0
    mean_mfe_ticks: float = 0.0
    mean_mae_ticks: float = 0.0
    p75_mfe_ticks: float = 0.0
    p25_mae_ticks: float = 0.0
    expectancy: float = 0.0   # mean(mfe - mae)

    # Regression models fitted on leaf pattern features within this subtree
    # More specific nodes have smaller, purer training sets → better fit
    mfe_coeff: Optional[List[float]] = None
    mfe_intercept: float = 0.0


@dataclass
class PatternDNA:
    """The path from root to leaf through the fractal tree."""
    direction: str           # 'LONG' or 'SHORT'
    path: List[str]          # ['1h:3', '15m:7', '5m:2', '15s:9']

    @property
    def key(self) -> str:
        """Deterministic string key for this DNA path."""
        return self.direction[0] + '|' + '|'.join(self.path)

    @property
    def depth(self) -> int:
        return len(self.path)

    def parent_dna(self) -> Optional['PatternDNA']:
        """Return the DNA of the parent node (one level up)."""
        if len(self.path) <= 1:
            return None
        return PatternDNA(direction=self.direction, path=self.path[:-1])

    def __str__(self):
        return self.key


class FractalDNATree:
    """
    Hierarchical fractal cluster tree with LONG/SHORT split at root level.

    Builds by level: for each TF level from macro (1h) to leaf (15s),
    cluster the ancestry context features, then assign each leaf pattern
    a DNA path through the tree.
    """

    # Ordered TF levels from macro to micro
    TF_ORDER = ['1h', '30m', '15m', '5m', '3m', '1m', '30s', '15s']

    def __init__(self, n_clusters_per_level: int = 5):
        """
        n_clusters_per_level: target number of k-means clusters at each TF level.
        Actual count may be less if a node has too few members.
        """
        self.n_clusters_per_level = n_clusters_per_level
        self.long_root:  Optional[TreeNode] = None
        self.short_root: Optional[TreeNode] = None
        self._dna_index: Dict[str, TreeNode] = {}  # dna.key -> leaf TreeNode

    def fit(self, patterns: list) -> None:
        """
        Build the full LONG and SHORT fractal trees from a list of PatternEvents.
        Each PatternEvent must have: timeframe, parent_chain, oracle_marker, oracle_meta.
        """
        # Filter for patterns that have oracle markers (non-noise preferred for tree building?
        # The instruction says: "long_patterns = [p for p in patterns if p.oracle_marker > 0]"
        # This implies we only build tree on VALID moves.
        long_patterns  = [p for p in patterns if getattr(p, 'oracle_marker', 0) > 0]
        short_patterns = [p for p in patterns if getattr(p, 'oracle_marker', 0) < 0]

        self.long_root  = self._build_branch(long_patterns,  'LONG')
        self.short_root = self._build_branch(short_patterns, 'SHORT')
        self._build_dna_index()

    def _build_branch(self, patterns: list, direction: str) -> TreeNode:
        """Recursively build the cluster tree for one direction."""
        root = TreeNode(
            node_id=direction[0] + '|root',
            direction=direction,
            timeframe='root',
            centroid=np.zeros(1),
            member_count=len(patterns),
        )
        self._split_by_tf_level(root, patterns, tf_level_idx=0, direction=direction)
        self._aggregate_oracle_stats(root, patterns)
        return root

    def _split_by_tf_level(self, parent_node: TreeNode, patterns: list,
                           tf_level_idx: int, direction: str) -> None:
        """
        For each pattern, extract features at the current TF level from parent_chain,
        then cluster. Recurse into each cluster at the next lower TF level.
        """
        if tf_level_idx >= len(self.TF_ORDER) or len(patterns) < 5:
            # Leaf: attach patterns directly
            parent_node.leaf_pattern_ids = [id(p) for p in patterns]
            return

        current_tf = self.TF_ORDER[tf_level_idx]
        leaf_tf    = getattr(patterns[0], 'timeframe', '15s')  # the actual TF of these PatternEvents

        if current_tf == leaf_tf:
            # We're at the leaf TF — cluster the leaf features directly
            X = np.array([self._extract_tf_features(p, current_tf, is_leaf=True)
                          for p in patterns])
        else:
            # Extract features at this TF from parent_chain ancestry
            X = np.array([self._extract_tf_features(p, current_tf, is_leaf=False)
                          for p in patterns])
            # Patterns with no ancestor at this TF get zero features — they
            # will cluster together and form a "no-context" node.

        # Cluster at this TF level
        k = min(self.n_clusters_per_level, max(2, len(patterns) // 10))
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Handle empty or uniform X to avoid errors
        if len(X) < k:
             parent_node.leaf_pattern_ids = [id(p) for p in patterns]
             return

        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=200)
        labels = km.fit_predict(X_scaled)
        centroids = km.cluster_centers_

        for lbl in range(k):
            mask = labels == lbl
            subset = [patterns[i] for i in np.where(mask)[0]]
            if not subset:
                continue

            node_id = f"{current_tf}:{lbl}"
            child = TreeNode(
                node_id=node_id,
                direction=direction,
                timeframe=current_tf,
                centroid=scaler.inverse_transform([centroids[lbl]])[0],
                member_count=len(subset),
            )
            parent_node.children[node_id] = child

            # Recurse into next TF level
            if current_tf != leaf_tf:
                self._split_by_tf_level(child, subset, tf_level_idx + 1, direction)
            else:
                child.leaf_pattern_ids = [id(p) for p in subset]

    def _extract_tf_features(self, p, tf: str, is_leaf: bool) -> List[float]:
        """
        Extract a compact feature vector for pattern p at timeframe tf.

        For leaf: uses the pattern's own features (z, velocity, momentum, coherence,
                  adx, hurst, dmi_diff, pid, osc_coh).
        For ancestor: extracts from the matching entry in parent_chain.
        """
        if is_leaf:
            s = getattr(p, 'state', None)
            if s:
                return [
                    abs(getattr(p, 'z_score', 0.0)),
                    np.log1p(abs(getattr(p, 'velocity', 0.0))),
                    np.log1p(abs(getattr(p, 'momentum', 0.0))),
                    getattr(s, 'adx_strength', 0.0) / 100.0,
                    getattr(s, 'hurst_exponent', 0.5),
                    (getattr(s, 'dmi_plus', 0.0) - getattr(s, 'dmi_minus', 0.0)) / 100.0,
                    getattr(s, 'term_pid', 0.0),
                    getattr(s, 'oscillation_coherence', 0.0),
                ]
            else:
                return [0.0] * 8
        else:
            chain = getattr(p, 'parent_chain', []) or []
            # Find the ancestor matching this TF
            ancestor = next((c for c in chain if c.get('tf') == tf), None)
            if ancestor is None:
                return [0.0] * 8  # no ancestor at this TF
            return [
                abs(ancestor.get('z', 0.0)),
                np.log1p(abs(ancestor.get('velocity', 0.0))),
                np.log1p(abs(ancestor.get('momentum', 0.0))),
                ancestor.get('adx', 0.0) / 100.0,
                ancestor.get('hurst', 0.5),
                (ancestor.get('dmi_plus', 0.0) - ancestor.get('dmi_minus', 0.0)) / 100.0,
                ancestor.get('pid', 0.0),
                ancestor.get('osc_coh', 0.0),
            ]

    def _aggregate_oracle_stats(self, node: TreeNode, patterns: list) -> None:
        """Recursively aggregate oracle stats bottom-up through the tree."""
        if not node.children:
            # Leaf node — compute directly from patterns with matching leaf_pattern_ids
            _ids = set(node.leaf_pattern_ids)
            members = [p for p in patterns if id(p) in _ids]
            self._compute_stats(node, members)
            return

        for child in node.children.values():
            self._aggregate_oracle_stats(child, patterns)

        # Aggregate from children up to this node
        all_mfe = []
        all_mae = []
        all_wins = 0
        total = 0
        for child in node.children.values():
            total += child.member_count
            all_wins += int(child.win_rate * child.member_count)
            if child.mean_mfe_ticks > 0:
                all_mfe.extend([child.mean_mfe_ticks] * child.member_count)
                all_mae.extend([child.mean_mae_ticks] * child.member_count)

        if total > 0:
            node.win_rate = all_wins / total
        if all_mfe:
            node.mean_mfe_ticks = float(np.mean(all_mfe))
            node.mean_mae_ticks = float(np.mean(all_mae))
            node.p75_mfe_ticks  = float(np.percentile(all_mfe, 75))
            node.p25_mae_ticks  = float(np.percentile(all_mae, 25))
            node.expectancy     = node.mean_mfe_ticks - node.mean_mae_ticks

    def _compute_stats(self, node: TreeNode, patterns: list) -> None:
        """Compute oracle stats for a leaf node from its member patterns."""
        if not patterns:
            return
        markers = [p.oracle_marker for p in patterns if hasattr(p, 'oracle_marker')]
        mfes = [p.oracle_meta.get('mfe', 0.0) / 0.25 for p in patterns
                if hasattr(p, 'oracle_meta') and p.oracle_meta.get('mfe')]
        maes = [p.oracle_meta.get('mae', 0.0) / 0.25 for p in patterns
                if hasattr(p, 'oracle_meta') and p.oracle_meta.get('mae')]

        if markers:
            node.win_rate = sum(1 for m in markers if abs(m) >= 1) / len(markers)
        if mfes:
            node.mean_mfe_ticks = float(np.mean(mfes))
            node.p75_mfe_ticks  = float(np.percentile(mfes, 75))
        if maes:
            node.mean_mae_ticks = float(np.mean(maes))
            node.p25_mae_ticks  = float(np.percentile(maes, 25))
        if mfes and maes:
            node.expectancy = node.mean_mfe_ticks - node.mean_mae_ticks

    def _build_dna_index(self) -> None:
        """Build flat lookup: dna.key -> leaf TreeNode for O(1) access."""
        def _traverse(node, path_so_far, direction):
            if not node.children:
                dna_key = direction[0] + '|' + '|'.join(path_so_far)
                self._dna_index[dna_key] = node
                return
            for child_id, child in node.children.items():
                _traverse(child, path_so_far + [child_id], direction)

        if self.long_root:
            _traverse(self.long_root, [], 'LONG')
        if self.short_root:
            _traverse(self.short_root, [], 'SHORT')

    def match(self, p) -> tuple:
        """
        Match a new PatternEvent to its DNA node in the tree.

        Returns: (dna: PatternDNA, node: TreeNode, confidence: float)
        confidence = 1.0 / (1.0 + dist_to_centroid)

        Traverses top-down from root: 1h → 15m → 5m → 15s.
        At each level, finds nearest child centroid.
        Falls back to parent node stats if a level has no match.
        """
        direction = 'LONG' if getattr(p, 'z_score', 0) <= 0 else 'SHORT'
        root = self.long_root if direction == 'LONG' else self.short_root
        if root is None:
            return None, None, 0.0

        path = []
        current = root
        total_dist = 0.0

        for tf in self.TF_ORDER:
            if not current.children:
                break  # reached leaf level

            is_leaf = (tf == getattr(p, 'timeframe', '15s'))
            feat = np.array(self._extract_tf_features(p, tf, is_leaf=is_leaf))

            # Find nearest child centroid at this TF
            best_child = None
            best_dist = float('inf')
            best_child_id = None
            for child_id, child in current.children.items():
                if child.timeframe != tf:
                    continue
                dist = float(np.linalg.norm(feat - child.centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_child = child
                    best_child_id = child_id

            if best_child is None:
                break

            path.append(best_child_id)
            total_dist += best_dist
            current = best_child

            if is_leaf:
                break

        if not path:
            return None, None, 0.0

        dna = PatternDNA(direction=direction, path=path)
        avg_dist = total_dist / len(path)
        confidence = 1.0 / (1.0 + avg_dist)
        return dna, current, confidence
