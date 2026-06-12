import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import os
import gc

def poly_expand_cpu(X):
    N, p = X.shape
    quad_terms = []
    for i in range(p):
        for j in range(i, p):
            quad_terms.append(X[:, i] * X[:, j])
    if len(quad_terms) > 0:
        quad = np.column_stack(quad_terms)
        return np.hstack([X, quad])
    return X

def main():
    json_path = 'artifacts/stage2_year_segments.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    print("Loading full segment corpus...")
    with open(json_path, 'r') as f:
        segments = json.load(f)
        
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    print(f"Processing {len(valid)} valid segments.")
    
    max_feat = 0
    for s in valid:
        if len(s['active_grid_cells']) > 0:
            max_feat = max(max_feat, max(s['active_grid_cells']))
            
    P = max_feat + 1
    N_PROBE = 1000
    np.random.seed(42)
    X_probe = np.random.randn(N_PROBE, P)
    
    print("Evaluating all segments on probe set (this may take a minute)...")
    Y_outputs = []
    valid_indices = []
    
    # Process in chunks to manage memory
    for i, s in enumerate(valid):
        if i % 10000 == 0:
            print(f" Evaluated {i} segments...")
            
        active_idx = s['active_grid_cells']
        fixed_terms = s['surviving_polynomial_indices']
        betas = s['beta_coefficients']
        
        if isinstance(active_idx, (int, float)): active_idx = [int(active_idx)]
        if isinstance(fixed_terms, (int, float)): fixed_terms = [int(fixed_terms)]
        if isinstance(betas, (int, float)): betas = [float(betas)]
        
        if len(active_idx) == 0 or len(fixed_terms) == 0 or len(betas) == 0:
            continue
            
        X_sub = X_probe[:, active_idx]
        X_poly = poly_expand_cpu(X_sub)
        
        valid_terms = [t for t in fixed_terms if t < X_poly.shape[1]]
        valid_betas = [betas[idx] for idx, t in enumerate(fixed_terms) if t < X_poly.shape[1]]
        
        if len(valid_terms) == 0:
            continue
            
        X_poly_fixed = X_poly[:, valid_terms]
        Y_pred = X_poly_fixed @ np.array(valid_betas)
        
        norm = np.linalg.norm(Y_pred)
        if norm > 1e-6:
            Y_outputs.append(Y_pred)
            valid_indices.append(i)
            
    Y_outputs = np.array(Y_outputs)
    print(f"Finished evaluating. Valid outputs: {len(Y_outputs)}")
    
    # Standardize predictions per segment
    Y_mean = np.mean(Y_outputs, axis=1, keepdims=True)
    Y_std = np.std(Y_outputs, axis=1, keepdims=True)
    Y_shape = (Y_outputs - Y_mean) / (Y_std + 1e-8)
    
    K = 100
    print(f"Running KMeans to extract {K} Candidate Seeds...")
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(Y_shape)
    
    # Find the actual segment that is closest to each cluster centroid (the medoid)
    print("Finding medoids for each candidate bucket...")
    closest_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, Y_shape)
    
    candidate_seeds = []
    for k in range(K):
        medoid_segment_idx = valid_indices[closest_idx[k]]
        medoid_segment = valid[medoid_segment_idx]
        
        # Count how many segments fell into this bucket
        bucket_size = int(np.sum(kmeans.labels_ == k))
        
        seed_data = {
            'bucket_id': k,
            'bucket_size': bucket_size,
            'representative_segment': medoid_segment
        }
        candidate_seeds.append(seed_data)
        
    out_path = 'artifacts/candidate_seeds.json'
    with open(out_path, 'w') as f:
        json.dump(candidate_seeds, f, indent=2)
        
    print(f"Saved {K} candidate seeds to {out_path}!")

if __name__ == '__main__':
    main()
