import warnings
import torch
import numpy as np

# Silence CUDA underutilization / perf warnings for small datasets
warnings.filterwarnings("ignore", message=".*not.*fully.*utilizing.*")
warnings.filterwarnings("ignore", message=".*CUDA.*performance.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

class CUDAKMeans:
    """
    A GPU-accelerated K-Means implementation using PyTorch.
    Mimics sklearn.cluster.KMeans API.
    """
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, n_init=10, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().cuda()
            X_cpu = X
        elif torch.is_tensor(X):
            X_tensor = X.float().cuda()
            X_cpu = X.cpu().numpy()
        else:
            raise ValueError("X must be numpy array or torch tensor")
        return X_tensor, X_cpu

    def fit(self, X):
        """
        Compute k-means clustering.
        X: numpy array or torch tensor (n_samples, n_features)
        """
        X_tensor, X_cpu = self._to_tensor(X)

        n_samples, n_features = X_tensor.shape

        # Handle edge case: fewer samples than clusters
        if n_samples < self.n_clusters:
            # Fallback to just assigning each sample to its own cluster (if possible)
            # or raising error. Sklearn raises error usually, or warns.
            # We will just warn and cap clusters.
            real_k = n_samples
            if self.verbose:
                print(f"CUDAKMeans Warning: n_samples={n_samples} < n_clusters={self.n_clusters}. Setting k={n_samples}")
            self.n_clusters = real_k
            if real_k == 0:
                self.labels_ = np.array([])
                return self

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)

        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        # Determine seed for numpy (kmeans++)
        rng = np.random.RandomState(self.random_state)

        for init_idx in range(self.n_init):
            # 1. KMeans++ Initialization (CPU — only K iterations, cheap)
            centroids_cpu = self._kmeans_plus_plus_init(X_cpu, rng)
            centroids = torch.from_numpy(centroids_cpu).float().cuda()

            current_centroids = centroids.clone()

            # 2. Iteration
            for i in range(self.max_iter):
                # Compute distances
                # cdist computes L2 distance. We need squared L2 for KMeans objective usually,
                # but argmin of L2 is same as argmin of L2^2.
                dists = torch.cdist(X_tensor, current_centroids)

                # Assign labels
                labels = torch.argmin(dists, dim=1)

                # Update centroids (Vectorized)
                new_centroids = torch.zeros_like(current_centroids)
                counts = torch.bincount(labels, minlength=self.n_clusters).float()

                new_centroids.index_add_(0, labels, X_tensor)

                # Handle empty clusters
                empty_mask = (counts == 0)
                if empty_mask.any():
                    # Re-initialize empty clusters to random points
                    num_empty = empty_mask.sum()
                    rand_indices = torch.randint(0, n_samples, (num_empty,), device=X_tensor.device)
                    new_centroids[empty_mask] = X_tensor[rand_indices]
                    counts[empty_mask] = 1.0 # Avoid division by zero

                # Compute the mean
                new_centroids /= counts.unsqueeze(1)

                # Check convergence
                # Shift: sum of squared distances between old and new centroids
                center_shift = torch.sum((current_centroids - new_centroids) ** 2)
                current_centroids = new_centroids

                if center_shift < self.tol:
                    break

            # 3. Compute Inertia (Sum of squared distances to closest centroid)
            dists = torch.cdist(X_tensor, current_centroids)
            sq_dists = dists.pow(2)
            min_sq_dists, labels = torch.min(sq_dists, dim=1)
            inertia = torch.sum(min_sq_dists).item()

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = current_centroids.clone()
                best_labels = labels.clone()

        self.centroids = best_centroids
        self.cluster_centers_ = best_centroids.cpu().numpy()
        self.labels_ = best_labels.cpu().numpy()
        self.inertia_ = best_inertia

        return self

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model not fitted")

        X_tensor, _ = self._to_tensor(X)

        dists = torch.cdist(X_tensor, self.centroids)
        labels = torch.argmin(dists, dim=1)
        return labels.cpu().numpy()

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _kmeans_plus_plus_init(self, X, rng):
        """KMeans++ initialization on GPU."""
        n_samples, n_features = X.shape
        X_t = torch.from_numpy(X).float().cuda()
        centroids = torch.empty((self.n_clusters, n_features), dtype=torch.float32, device='cuda')

        # First centroid: random
        centroids[0] = X_t[rng.randint(n_samples)]

        for c in range(1, self.n_clusters):
            # Squared distance from each point to nearest existing centroid (all on GPU)
            dists = torch.cdist(X_t, centroids[:c]).pow(2).min(dim=1).values

            # Probability proportional to distance
            total = dists.sum()
            if total < 1e-30:
                centroids[c] = X_t[rng.randint(n_samples)]
            else:
                probs = (dists / total).cpu().numpy()
                centroids[c] = X_t[rng.choice(n_samples, p=probs)]

        return centroids.cpu().numpy()


def cuda_silhouette_score(X, labels):
    """GPU-accelerated silhouette score using PyTorch."""
    X_tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64)).float().cuda()
    labels_tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)).cuda()

    n_samples = X_tensor.shape[0]
    n_clusters = int(labels_tensor.max().item()) + 1

    # Pairwise distance matrix (n_samples x n_samples)
    dist_matrix = torch.cdist(X_tensor, X_tensor)

    a_scores = torch.zeros(n_samples, device=X_tensor.device)
    b_scores = torch.full((n_samples,), float('inf'), device=X_tensor.device)

    for c in range(n_clusters):
        mask = (labels_tensor == c)
        count = mask.sum().float()

        # Mean distance from every point to all points in cluster c
        cluster_dists = dist_matrix[:, mask].sum(dim=1)  # (n_samples,)

        # For points IN this cluster: a(i) = mean intra-cluster distance
        in_cluster = mask
        if count > 1:
            a_scores[in_cluster] = cluster_dists[in_cluster] / (count - 1)

        # For points NOT in this cluster: candidate for b(i)
        out_cluster = ~mask
        if count > 0:
            mean_dist = cluster_dists[out_cluster] / count
            b_scores[out_cluster] = torch.minimum(b_scores[out_cluster], mean_dist)

    # Silhouette: (b - a) / max(a, b)
    max_ab = torch.maximum(a_scores, b_scores)
    sil = torch.where(max_ab > 0, (b_scores - a_scores) / max_ab, torch.zeros_like(a_scores))

    return float(sil.mean().item())
