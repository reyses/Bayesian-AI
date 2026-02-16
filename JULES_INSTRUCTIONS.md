# Jules Task: Replace sklearn KMeans with CUDA KMeans (numba)

## Problem
`sklearn.cluster.KMeans` and `MiniBatchKMeans` hang on Windows due to a BLAS threading deadlock. This blocks Phase 2.5 (clustering 2,634 patterns into templates) indefinitely. The sklearn dependency must be completely removed from the clustering hot path.

## Solution
Write a GPU-native KMeans using `numba.cuda` — the same GPU framework already used throughout the project. The RTX 3060 has 3,584 CUDA cores. With 2,634 patterns x 7 features x 26 clusters, each CUDA thread handles one pattern's distance calculations in parallel.

## Architecture

### New file: `core/cuda_kmeans.py`

Implement a `CudaKMeans` class that replaces `sklearn.cluster.KMeans` and `MiniBatchKMeans`:

```python
"""
GPU-native KMeans clustering using numba CUDA.
1 CUDA thread per pattern — computes distances to all K centroids.
"""
import numpy as np
from numba import cuda
import math

# ---- CUDA Kernels ----

@cuda.jit
def _assign_clusters_kernel(X, centroids, labels, n_samples, n_features, n_clusters):
    """
    Each thread handles ONE pattern (row of X).
    Computes squared Euclidean distance to all centroids.
    Assigns pattern to nearest centroid.
    """
    idx = cuda.grid(1)
    if idx >= n_samples:
        return

    best_dist = 1e30
    best_label = 0

    for c in range(n_clusters):
        dist = 0.0
        for f in range(n_features):
            diff = X[idx, f] - centroids[c, f]
            dist += diff * diff
        if dist < best_dist:
            best_dist = dist
            best_label = c

    labels[idx] = best_label


@cuda.jit
def _update_centroids_kernel(X, labels, centroids, counts, n_samples, n_features, n_clusters):
    """
    Each thread handles ONE (cluster, feature) pair.
    Accumulates sum and count for centroid update.
    NOTE: This uses atomic adds — suitable for moderate K.
    """
    idx = cuda.grid(1)
    if idx >= n_samples:
        return

    label = labels[idx]
    cuda.atomic.add(counts, label, 1)
    for f in range(n_features):
        cuda.atomic.add(centroids, (label, f), X[idx, f])


class CudaKMeans:
    """
    Drop-in replacement for sklearn KMeans.
    Uses numba CUDA kernels for parallel assignment + centroid update.

    Usage:
        model = CudaKMeans(n_clusters=26, max_iter=100)
        labels = model.fit_predict(X)  # X is numpy array (n_samples, n_features)
        model.cluster_centers_  # centroids as numpy array
    """

    def __init__(self, n_clusters=8, max_iter=100, n_init=3, random_state=42, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def fit_predict(self, X):
        """
        Run KMeans and return labels.
        Runs n_init times with different random seeds, keeps best (lowest inertia).
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)

        best_labels = None
        best_inertia = float('inf')
        best_centroids = None

        # Transfer data to GPU once (shared across all n_init runs)
        d_X = cuda.to_device(X)

        threads_per_block = 256
        blocks = math.ceil(n_samples / threads_per_block)

        for init_run in range(self.n_init):
            # KMeans++ initialization (on CPU — cheap for small K)
            centroids = self._kmeans_plus_plus_init(X, rng)

            d_labels = cuda.device_array(n_samples, dtype=np.int32)

            for iteration in range(self.max_iter):
                # 1. Assignment step (GPU) — 1 thread per pattern
                d_centroids = cuda.to_device(centroids)
                _assign_clusters_kernel[blocks, threads_per_block](
                    d_X, d_centroids, d_labels, n_samples, n_features, self.n_clusters
                )
                cuda.synchronize()

                # 2. Update step (GPU) — accumulate sums with atomic adds
                new_centroids = np.zeros((self.n_clusters, n_features), dtype=np.float64)
                counts = np.zeros(self.n_clusters, dtype=np.int32)
                d_new_centroids = cuda.to_device(new_centroids)
                d_counts = cuda.to_device(counts)

                _update_centroids_kernel[blocks, threads_per_block](
                    d_X, d_labels, d_new_centroids, d_counts, n_samples, n_features, self.n_clusters
                )
                cuda.synchronize()

                new_centroids = d_new_centroids.copy_to_host()
                counts = d_counts.copy_to_host()

                # Divide by count (handle empty clusters)
                for c in range(self.n_clusters):
                    if counts[c] > 0:
                        new_centroids[c] /= counts[c]
                    else:
                        # Re-seed empty cluster with random data point
                        new_centroids[c] = X[rng.randint(n_samples)]

                # Check convergence
                shift = np.sum((new_centroids - centroids) ** 2)
                centroids = new_centroids

                if shift < self.tol:
                    break

            # Compute inertia (sum of squared distances to assigned centroid)
            labels = d_labels.copy_to_host()
            inertia = 0.0
            for i in range(n_samples):
                diff = X[i] - centroids[labels[i]]
                inertia += np.dot(diff, diff)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        return best_labels

    def fit(self, X):
        """Fit and store results."""
        self.fit_predict(X)
        return self

    def _kmeans_plus_plus_init(self, X, rng):
        """KMeans++ initialization on CPU (only K iterations, cheap)."""
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=np.float64)

        # First centroid: random
        centroids[0] = X[rng.randint(n_samples)]

        for c in range(1, self.n_clusters):
            # Distance from each point to nearest existing centroid
            dists = np.full(n_samples, np.inf)
            for j in range(c):
                d = np.sum((X - centroids[j]) ** 2, axis=1)
                dists = np.minimum(dists, d)

            # Probability proportional to distance
            probs = dists / dists.sum()
            centroids[c] = X[rng.choice(n_samples, p=probs)]

        return centroids
```

### Also implement `cuda_silhouette_score` in the same file

The `refine_clusters()` method in `fractal_clustering.py` uses `silhouette_score` from sklearn. Implement a CUDA version:

```python
@cuda.jit
def _pairwise_distances_kernel(X, labels, a_scores, b_scores, b_counts, n_samples, n_features, n_clusters):
    """
    Each thread computes the a(i) and b(i) terms for one sample i.
    a(i) = mean distance to same-cluster points
    b(i) = min over other clusters of mean distance to that cluster's points
    """
    idx = cuda.grid(1)
    if idx >= n_samples:
        return

    my_label = labels[idx]

    # Accumulators per cluster
    cluster_dist_sum = cuda.local.array(64, dtype=numba.float64)  # max 64 clusters
    cluster_count = cuda.local.array(64, dtype=numba.int32)

    for c in range(n_clusters):
        cluster_dist_sum[c] = 0.0
        cluster_count[c] = 0

    for j in range(n_samples):
        if j == idx:
            continue
        dist = 0.0
        for f in range(n_features):
            diff = X[idx, f] - X[j, f]
            dist += diff * diff
        dist = math.sqrt(dist)

        c = labels[j]
        cluster_dist_sum[c] += dist
        cluster_count[c] += 1

    # a(i) = mean intra-cluster distance
    if cluster_count[my_label] > 0:
        a_scores[idx] = cluster_dist_sum[my_label] / cluster_count[my_label]
    else:
        a_scores[idx] = 0.0

    # b(i) = min inter-cluster mean distance
    min_b = 1e30
    for c in range(n_clusters):
        if c == my_label:
            continue
        if cluster_count[c] > 0:
            mean_d = cluster_dist_sum[c] / cluster_count[c]
            if mean_d < min_b:
                min_b = mean_d

    b_scores[idx] = min_b


def cuda_silhouette_score(X, labels):
    """GPU silhouette score. Returns mean silhouette coefficient."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    n_samples, n_features = X.shape
    n_clusters = int(labels.max()) + 1

    a_scores = np.zeros(n_samples, dtype=np.float64)
    b_scores = np.zeros(n_samples, dtype=np.float64)
    b_counts = np.zeros(n_samples, dtype=np.int32)  # unused but kept for kernel sig

    d_X = cuda.to_device(X)
    d_labels = cuda.to_device(labels)
    d_a = cuda.to_device(a_scores)
    d_b = cuda.to_device(b_scores)
    d_bc = cuda.to_device(b_counts)

    threads = 256
    blocks = math.ceil(n_samples / threads)

    _pairwise_distances_kernel[blocks, threads](
        d_X, d_labels, d_a, d_b, d_bc, n_samples, n_features, n_clusters
    )
    cuda.synchronize()

    a_scores = d_a.copy_to_host()
    b_scores = d_b.copy_to_host()

    # Silhouette: (b - a) / max(a, b)
    sil = np.zeros(n_samples)
    for i in range(n_samples):
        max_ab = max(a_scores[i], b_scores[i])
        if max_ab > 0:
            sil[i] = (b_scores[i] - a_scores[i]) / max_ab

    return float(np.mean(sil))
```

**IMPORTANT**: The `_pairwise_distances_kernel` uses `cuda.local.array(64, ...)` — this limits max clusters to 64. If `n_clusters` could exceed 64, increase this constant. For this project 64 is more than enough.

---

## Modify: `training/fractal_clustering.py`

### Changes needed:

1. **Replace sklearn imports** with CUDA imports:
```python
# Remove:
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

# Add:
from core.cuda_kmeans import CudaKMeans, cuda_silhouette_score
```

2. **Keep** `from sklearn.preprocessing import StandardScaler` — this is CPU-only and works fine.

3. **Replace all KMeans usage in `create_templates()`** (lines 94-99):
```python
# Old:
if len(valid_patterns) < MINIBATCH_KMEANS_SAMPLES_THRESHOLD:
    model = KMeans(n_clusters=target_k, random_state=42, n_init=10)
else:
    model = MiniBatchKMeans(n_clusters=target_k, batch_size=min(DEFAULT_KMEANS_BATCH_SIZE, len(valid_patterns)), random_state=42)
labels = model.fit_predict(X_scaled)

# New:
model = CudaKMeans(n_clusters=target_k, random_state=42, n_init=3, max_iter=100)
labels = model.fit_predict(X_scaled)
```

4. **Replace KMeans in `_recursive_split()`** (line 180):
```python
# Old:
kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_subset)

# New:
kmeans = CudaKMeans(n_clusters=2, random_state=42, n_init=3).fit(X_subset)
```

5. **Replace KMeans and silhouette_score in `refine_clusters()`** (lines 214-221):
```python
# Old:
kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(X_params)
score = silhouette_score(X_params, kmeans.labels_)

# New:
kmeans = CudaKMeans(n_clusters=n, random_state=42, n_init=3).fit(X_params)
score = cuda_silhouette_score(X_params, kmeans.labels_)
```

6. **Remove the constants** `MINIBATCH_KMEANS_SAMPLES_THRESHOLD` and `DEFAULT_KMEANS_BATCH_SIZE` — no longer needed.

---

## Verification

Run:
```bash
python training/orchestrator.py --fresh --no-dashboard --iterations 50
```

Expected:
1. Phase 2 discovery: ~2,634 patterns across 9 timeframe levels (same as before)
2. Phase 2.5 clustering: should complete in **under 5 seconds** (not hang)
3. Phase 3 optimization: should process templates, checkpoint after each batch
4. No sklearn import errors in the clustering path

Also run a unit test:
```bash
python -c "
from core.cuda_kmeans import CudaKMeans, cuda_silhouette_score
import numpy as np
X = np.random.randn(2634, 7)
model = CudaKMeans(n_clusters=26, random_state=42, n_init=3)
labels = model.fit_predict(X)
print(f'Clusters: {len(set(labels))}, Centers shape: {model.cluster_centers_.shape}')
score = cuda_silhouette_score(X, labels)
print(f'Silhouette: {score:.3f}')
print('PASS')
"
```

---

## File Summary

| File | Action |
|------|--------|
| `core/cuda_kmeans.py` | NEW — CudaKMeans class + cuda_silhouette_score |
| `training/fractal_clustering.py` | Replace sklearn KMeans/silhouette with CUDA versions |

## Key Design Decisions
- **1 CUDA thread per pattern** for assignment step (distance to all K centroids)
- **1 CUDA thread per pattern** for update step (atomic add to centroid accumulators)
- **KMeans++ init on CPU** — only K iterations, negligible cost
- **n_init=3** — run 3 random initializations, keep best inertia (same as sklearn default behavior)
- **Convergence check on CPU** — centroid shift < tol after each iteration
- **No new dependencies** — uses existing numba.cuda stack
