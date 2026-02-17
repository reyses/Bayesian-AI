import torch
import numpy as np
from sklearn.cluster import kmeans_plusplus

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

    def fit(self, X):
        """
        Compute k-means clustering.
        X: numpy array or torch tensor (n_samples, n_features)
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().cuda()
            X_cpu = X
        elif torch.is_tensor(X):
            X_tensor = X.float().cuda()
            X_cpu = X.cpu().numpy()
        else:
            raise ValueError("X must be numpy array or torch tensor")

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
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

        for init_idx in range(self.n_init):
            # 1. Initialization (CPU for kmeans++)
            try:
                centroids_cpu, _ = kmeans_plusplus(X_cpu, n_clusters=self.n_clusters, random_state=rng)
                centroids = torch.from_numpy(centroids_cpu).float().cuda()
            except Exception as e:
                # Fallback to random choice if kmeans++ fails
                if self.verbose:
                    print(f"CUDAKMeans: kmeans++ failed ({e}), using random init.")
                indices = torch.randperm(n_samples)[:self.n_clusters]
                centroids = X_tensor[indices].clone()

            current_centroids = centroids.clone()

            # 2. Iteration
            for i in range(self.max_iter):
                # Compute distances
                # cdist computes L2 distance. We need squared L2 for KMeans objective usually,
                # but argmin of L2 is same as argmin of L2^2.
                dists = torch.cdist(X_tensor, current_centroids)

                # Assign labels
                labels = torch.argmin(dists, dim=1)

                # Update centroids
                new_centroids = torch.zeros_like(current_centroids)

                # Fast update using scatter_add logic or loop
                # For K < 1000, loop is acceptable on GPU
                # Or use index_add_ (requires expanding dimensions)

                unique_labels, counts = torch.unique(labels, return_counts=True)

                # We need to handle empty clusters
                # A simple way: mask loop
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if mask.any():
                        new_centroids[k] = X_tensor[mask].mean(dim=0)
                    else:
                        # Empty cluster: Re-initialize to a random point (far from others?)
                        # Or just keep old centroid (simplest)
                        # Or pick random point from X
                        rand_idx = torch.randint(0, n_samples, (1,)).item()
                        new_centroids[k] = X_tensor[rand_idx]

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

        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().cuda()
        elif torch.is_tensor(X):
            X_tensor = X.float().cuda()
        else:
            raise ValueError("X must be numpy array or torch tensor")

        dists = torch.cdist(X_tensor, self.centroids)
        labels = torch.argmin(dists, dim=1)
        return labels.cpu().numpy()

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
