"""Seed primitive library and adaptive splitting for trajectory classification.

Extracted from tools/waveform_standalone.py (lines 1662-1947).
Contains the 20-shape SeedPrimitiveLibrary and helper functions for
inflection detection and adaptive KMeans sub-typing.
"""

import numpy as np


# =============================================================================
#  SEED PRIMITIVE LIBRARY (Analysis I)
#
#  12 orthogonal mathematical shapes, normalized 0-1.
#  Categories 1 & 2 get _UP / _DOWN variants (x2) = 16
#  Category 3 (symmetrical) = 4
#  Total: 20 shapes in the dictionary.
# =============================================================================

class SeedPrimitiveLibrary:
    """Library of 20 normalized seed shapes for trajectory classification."""

    CORR_THRESHOLD = 0.75  # minimum Pearson r to classify (below = NOISE)

    def __init__(self, N=16):
        self.N = N
        self.shapes = {}
        self._build(N)

    def _norm01(self, arr):
        """Normalize array to [0, 1]. Returns zeros if flat."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _build(self, N):
        x = np.linspace(0, 1, N)  # normalized time axis

        # --- Category 1: Directional ---
        cat1 = {
            'LINEAR':      x,
            'EXPONENTIAL': x ** 2,
            'LOGARITHMIC': np.log(x + 1),
            'STEP':        np.where(np.arange(N) < N / 2, 0.0, 1.0),
        }

        # --- Category 2: Reversals ---
        cat2 = {
            'SYMMETRIC_V':  np.abs(np.arange(N) - N / 2),
            'ROUNDED_U':    (np.arange(N) - N / 2) ** 2,
            'FRONT_SKEWED': np.exp(-4 * x) - np.exp(-8 * x),
            'BACK_SKEWED':  np.exp(4 * (x - 1)) - np.exp(8 * (x - 1)),
        }

        # Normalize all Cat1 & Cat2 to 0-1, then create UP/DOWN variants
        for name, arr in {**cat1, **cat2}.items():
            normed = self._norm01(arr.astype(float))
            self.shapes[f'{name}_UP'] = normed
            self.shapes[f'{name}_DOWN'] = 1.0 - normed

        # --- Category 3: Volatility (symmetrical, no inversion) ---
        t_cyc = np.linspace(0, 2 * np.pi, N)
        cat3 = {
            'SINE_WAVE':          np.sin(t_cyc),
            'DAMPED_OSCILLATOR':  np.exp(-2 * x) * np.sin(t_cyc),
            'EXPAND_OSCILLATOR':  np.exp(2 * x) * np.sin(t_cyc),
            'FLATLINE':           np.ones(N),
        }
        for name, arr in cat3.items():
            self.shapes[name] = self._norm01(arr.astype(float))

    def classify_trajectory(self, price_segment):
        """Classify a price segment against the 20 seed primitives.

        Args:
            price_segment: raw prices (not pre-normalized)

        Returns:
            (best_shape_name, correlation) or ('NOISE', best_corr)
        """
        seg = np.asarray(price_segment, dtype=float)
        if len(seg) != self.N:
            return 'NOISE', 0.0

        # Normalize input to 0-1
        mn, mx = seg.min(), seg.max()
        if mx - mn < 1e-12:
            return 'FLATLINE', 1.0  # truly flat -> direct match

        normed = (seg - mn) / (mx - mn)

        # Pearson correlation against all 20 shapes
        best_name = 'NOISE'
        best_corr = -999.0

        for name, template in self.shapes.items():
            # Skip zero-variance templates (FLATLINE -> all zeros after norm)
            if template.std() < 1e-12:
                continue
            r = np.corrcoef(normed, template)[0, 1]
            if np.isnan(r):
                continue
            if r > best_corr:
                best_corr = r
                best_name = name

        if best_corr < self.CORR_THRESHOLD:
            return 'NOISE', best_corr

        return best_name, best_corr


def _detect_inflections(centroid):
    """Detect inflection points on a centroid (raw ticks or normalized).

    An inflection = where the bar-to-bar direction flips sign.
    Returns list of (bar_idx, level) for each inflection point,
    plus segment descriptors between them.
    """
    d = np.diff(centroid)  # bar-to-bar changes
    inflections = [(0, centroid[0])]  # start point always included

    for i in range(1, len(d)):
        # Sign flip: direction changed
        if d[i] * d[i - 1] < 0:
            inflections.append((i, centroid[i]))

    inflections.append((len(centroid) - 1, centroid[-1]))  # end point

    # Build segment descriptors between inflection points
    segments = []
    for k in range(len(inflections) - 1):
        b0, v0 = inflections[k]
        b1, v1 = inflections[k + 1]
        delta = v1 - v0
        if abs(delta) < 1e-6:
            label = 'HOLD'
        elif delta > 0:
            label = 'RISE'
        else:
            label = 'DROP'
        segments.append({'start': b0, 'end': b1, 'v0': v0, 'v1': v1, 'label': label})

    return inflections, segments


def _adaptive_split(deltas, r2_target=0.80, min_n=2, max_k=48):
    """Find optimal k where all sub-types hit shape R2 >= target.

    Clustering uses raw deltas (ticks) so magnitude matters for grouping.
    R2 is computed on shape-normalized segments (0-1) so it measures shape
    consistency, not magnitude consistency.

    Tries k=1,2,3,...,max_k. Keeps the k with highest minimum R2 across
    all clusters. Stops early if all clusters hit the target.

    Returns (labels, centroids, shape_r2s) -- labels[i] = cluster id,
    centroids[k] = raw mean trace, shape_r2s[k] = shape-normalized R2.
    """
    from sklearn.cluster import KMeans

    def _shape_r2(sub):
        """Mean Pearson r2 between each segment (0-1 normed) and centroid.

        More robust than global R2 for small clusters -- each segment's
        shape agreement is measured independently then averaged.
        """
        n = len(sub)
        if n < 2:
            return 1.0
        normed_sub = np.zeros_like(sub)
        for i in range(n):
            mn, mx = sub[i].min(), sub[i].max()
            rng = mx - mn
            normed_sub[i] = (sub[i] - mn) / rng if rng > 1e-12 else 0.0
        centroid = normed_sub.mean(axis=0)
        if centroid.std() < 1e-12:
            return 0.0
        r2_vals = []
        for i in range(n):
            if normed_sub[i].std() < 1e-12:
                continue
            r = np.corrcoef(normed_sub[i], centroid)[0, 1]
            if not np.isnan(r):
                r2_vals.append(r ** 2)
        return np.mean(r2_vals) if r2_vals else 0.0

    n_total = len(deltas)

    # Build shape-normalized version for clustering (0-1 per segment)
    normed = np.zeros_like(deltas)
    for i in range(n_total):
        mn, mx = deltas[i].min(), deltas[i].max()
        rng = mx - mn
        normed[i] = (deltas[i] - mn) / rng if rng > 1e-12 else 0.0

    # k=1 baseline (no splitting)
    base_r2 = _shape_r2(deltas)
    best_labels = np.zeros(n_total, dtype=int)
    best_centroids = np.array([deltas.mean(axis=0)])
    best_r2s = np.array([base_r2])
    best_min_r2 = base_r2

    if base_r2 >= r2_target:
        return best_labels, best_centroids, best_r2s

    # Try increasing k -- cluster on SHAPE (normalized), report in raw ticks
    k_limit = min(max_k, n_total // min_n)
    for k in range(2, k_limit + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(normed)  # cluster by shape, not magnitude

        # Check all clusters have min_n segments
        valid = True
        r2s = []
        raw_centroids = []
        for ci in range(k):
            mask = (labels == ci)
            n_ci = mask.sum()
            if n_ci < min_n:
                valid = False
                break
            r2s.append(_shape_r2(deltas[mask]))
            raw_centroids.append(deltas[mask].mean(axis=0))

        if not valid:
            continue

        min_r2 = min(r2s)
        if min_r2 > best_min_r2:
            best_labels = labels.copy()
            best_centroids = np.array(raw_centroids)
            best_r2s = np.array(r2s)
            best_min_r2 = min_r2

        if min_r2 >= r2_target:
            break  # all clusters meet target

    # --- Phase 2: targeted bisection of worst clusters ---
    # KMeans finds the best global k, but some clusters may still be below
    # target. Bisect those specifically until they meet R2 or hit min_n.
    improved = True
    while improved:
        improved = False
        new_labels = best_labels.copy()
        new_centroids = list(best_centroids)
        new_r2s = list(best_r2s)

        # Find worst cluster that can be split
        worst_ci = -1
        worst_r2 = r2_target
        for ci in range(len(new_centroids)):
            if new_r2s[ci] < worst_r2:
                ci_mask = (new_labels == ci)
                if ci_mask.sum() >= 2 * min_n:
                    worst_ci = ci
                    worst_r2 = new_r2s[ci]

        if worst_ci < 0:
            break  # nothing to split

        ci_mask = (new_labels == worst_ci)
        ci_indices = np.where(ci_mask)[0]
        ci_normed = normed[ci_indices]

        km2 = KMeans(n_clusters=2, random_state=42, n_init=20)
        sub_labels = km2.fit_predict(ci_normed)

        idx_a = ci_indices[sub_labels == 0]
        idx_b = ci_indices[sub_labels == 1]

        if len(idx_a) < min_n or len(idx_b) < min_n:
            # Can't split -- mark as final and stop trying this cluster
            break

        r2_a = _shape_r2(deltas[idx_a])
        r2_b = _shape_r2(deltas[idx_b])

        # Only accept if BOTH halves improve over the original
        if min(r2_a, r2_b) > worst_r2:
            new_id = len(new_centroids)
            new_labels[idx_a] = worst_ci
            new_labels[idx_b] = new_id
            new_centroids[worst_ci] = deltas[idx_a].mean(axis=0)
            new_centroids.append(deltas[idx_b].mean(axis=0))
            new_r2s[worst_ci] = r2_a
            new_r2s.append(r2_b)

            best_labels = new_labels
            best_centroids = np.array(new_centroids)
            best_r2s = np.array(new_r2s)
            best_min_r2 = min(best_r2s)
            improved = True

    return best_labels, best_centroids, best_r2s
