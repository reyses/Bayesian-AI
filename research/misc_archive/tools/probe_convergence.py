import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def poly_expand_cpu(X):
    # Matches poly_expand_gpu from stage2_parallel_chaos.py
    N, p = X.shape
    # quad = X_t[:, idx_i] * X_t[:, idx_j] where idx_i, idx_j = torch.triu_indices
    # In PyTorch triu_indices, i corresponds to the row and j to the column (j >= i).
    # Specifically, it returns pairs in order of i first, then j.
    # Let's reproduce the exact order PyTorch's triu_indices uses.
    quad_terms = []
    for i in range(p):
        for j in range(i, p):
            quad_terms.append(X[:, i] * X[:, j])
            
    if len(quad_terms) > 0:
        quad = np.column_stack(quad_terms)
        return np.hstack([X, quad])
    return X

def main():
    print("Loading segments...")
    json_path = 'artifacts/stage2_year_segments.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        segments = json.load(f)
        
    print(f"Loaded {len(segments)} total segments.")
    
    # Filter to only PRISTINE or RECOVERED (exclude pure chaos)
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    print(f"Filtering to {len(valid)} valid segments.")
    
    # Take a random sample to keep it fast
    np.random.seed(42)
    sample = np.random.choice(valid, min(5000, len(valid)), replace=False)
    
    # Find max feature index to know P
    max_feat = 0
    for s in sample:
        if len(s['active_grid_cells']) > 0:
            max_feat = max(max_feat, max(s['active_grid_cells']))
            
    P = max_feat + 1
    print(f"Max feature index is {P-1}, generating probe set of shape (N_PROBE, {P})")
    
    N_PROBE = 1000
    # Generate synthetic Gaussian probe set (standardized features)
    X_probe = np.random.randn(N_PROBE, P)
    
    print("Evaluating segments on probe set...")
    Y_outputs = []
    
    for s in sample:
        active_idx = s['active_grid_cells']
        fixed_terms = s['surviving_polynomial_indices']
        betas = s['beta_coefficients']
        
        if isinstance(active_idx, (int, float)): active_idx = [int(active_idx)]
        if isinstance(fixed_terms, (int, float)): fixed_terms = [int(fixed_terms)]
        if isinstance(betas, (int, float)): betas = [float(betas)]
        
        if len(active_idx) == 0 or len(fixed_terms) == 0 or len(betas) == 0:
            Y_outputs.append(np.zeros(N_PROBE))
            continue
            
        X_sub = X_probe[:, active_idx]
        X_poly = poly_expand_cpu(X_sub)
        
        valid_terms = [t for t in fixed_terms if t < X_poly.shape[1]]
        valid_betas = [betas[i] for i, t in enumerate(fixed_terms) if t < X_poly.shape[1]]
        
        if len(valid_terms) == 0:
            Y_outputs.append(np.zeros(N_PROBE))
            continue
            
        X_poly_fixed = X_poly[:, valid_terms]
        Y_pred = X_poly_fixed @ np.array(valid_betas)
        Y_outputs.append(Y_pred)
        
    Y_outputs = np.array(Y_outputs)
    
    # Drop zero rows
    norms = np.linalg.norm(Y_outputs, axis=1)
    mask = norms > 1e-6
    Y_valid = Y_outputs[mask]
    print(f"Dropped {len(Y_outputs) - len(Y_valid)} empty/invalid outputs. Remaining: {len(Y_valid)}")
    
    # Standardize predictions per segment so we compare SHAPE, not scale
    Y_mean = np.mean(Y_valid, axis=1, keepdims=True)
    Y_std = np.std(Y_valid, axis=1, keepdims=True)
    Y_shape = (Y_valid - Y_mean) / (Y_std + 1e-8)
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(Y_shape)
    
    print("Clustering...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels = kmeans.fit_predict(Y_pca)
    
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Y_pca[:, 0], Y_pca[:, 1], c=labels, cmap='tab10', alpha=0.5, s=15, edgecolors='none')
    plt.title("Phase 1 Convergence Probe: PCA of Curve Outputs on Synthetic Data")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.colorbar(scatter, label="KMeans Cluster")
    plt.grid(True, alpha=0.3)
    
    out_path = "artifacts/probe_convergence_pca.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    
if __name__ == '__main__':
    main()
