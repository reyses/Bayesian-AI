import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.features import load_features

def main():
    print("[INFO] Loading feature columns to map indices to names...")
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    df = load_features(['2025_02_05'], root=features_root)
    
    features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    N_FEATURES = len(features_cols)
    print(f"[INFO] Found {N_FEATURES} features.")
    
    files = glob.glob('artifacts/stage2_segments_*.json')
    print(f"[INFO] Found {len(files)} segment JSON files.")
    
    # 2D matrix to count interactions
    interaction_matrix = np.zeros((N_FEATURES, N_FEATURES), dtype=np.int32)
    
    total_segments = 0
    
    for f_path in files:
        with open(f_path, 'r') as f:
            segments = json.load(f)
            
        for seg in segments:
            if seg.get('status') in ['PRISTINE', 'RECOVERED']:
                active = seg.get('active_grid_cells', [])
                surviving = seg.get('surviving_polynomial_indices', [])
                
                p = len(active)
                if p == 0 or len(surviving) == 0:
                    continue
                    
                total_segments += 1
                
                # Reconstruct the triu indices to match poly_expand_gpu
                idx_i, idx_j = torch.triu_indices(p, p, offset=0)
                
                for k in surviving:
                    if k >= p:
                        # Interaction term!
                        offset = k - p
                        if offset < len(idx_i):
                            local_i = idx_i[offset].item()
                            local_j = idx_j[offset].item()
                            
                            global_i = active[local_i]
                            global_j = active[local_j]
                            
                            # Increment symmetric matrix
                            interaction_matrix[global_i, global_j] += 1
                            if global_i != global_j:
                                interaction_matrix[global_j, global_i] += 1
                                
    print(f"[INFO] Processed {total_segments} pristine segments with interaction terms.")
    
    # Extract top interacting features
    # Since matrix is symmetric, we only look at upper triangle for unique pairs
    upper_tri = np.triu(interaction_matrix)
    
    # Flatten and get top N pairs
    flat_indices = np.argsort(upper_tri.flatten())[::-1]
    
    top_n = 20
    top_pairs = []
    
    print("\n[TOP INTERACTIONS]")
    for idx in flat_indices[:top_n]:
        val = upper_tri.flatten()[idx]
        if val == 0:
            break
        i, j = np.unravel_index(idx, interaction_matrix.shape)
        name_i = features_cols[i]
        name_j = features_cols[j]
        top_pairs.append((name_i, name_j, val))
        print(f"[{val:4d} times] {name_i} x {name_j}")
        
    # Plotting the dense sub-matrix of top interacting features
    # Let's collect the unique features involved in the top 50 pairs
    top_50_pairs = []
    for idx in flat_indices[:50]:
        val = upper_tri.flatten()[idx]
        if val > 0:
            i, j = np.unravel_index(idx, interaction_matrix.shape)
            top_50_pairs.append((i, j))
            
    unique_active_features = set()
    for i, j in top_50_pairs:
        unique_active_features.add(i)
        unique_active_features.add(j)
        
    unique_active_features = list(unique_active_features)
    # Sort them by total interaction frequency to make the heatmap look nice
    feature_freqs = [np.sum(interaction_matrix[idx, :]) for idx in unique_active_features]
    unique_active_features = [x for _, x in sorted(zip(feature_freqs, unique_active_features), reverse=True)]
    
    sub_matrix = np.zeros((len(unique_active_features), len(unique_active_features)))
    labels = [features_cols[idx] for idx in unique_active_features]
    
    for row_idx, f_i in enumerate(unique_active_features):
        for col_idx, f_j in enumerate(unique_active_features):
            sub_matrix[row_idx, col_idx] = interaction_matrix[f_i, f_j]
            
    plt.figure(figsize=(14, 12))
    sns.heatmap(sub_matrix, xticklabels=labels, yticklabels=labels, cmap="YlOrRd", annot=False)
    plt.title("Aggregated Polynomial Interaction Matrix (Pristine Segments)", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('artifacts/combined_segment_interactions.png', dpi=150)
    print("\n[INFO] Saved heatmap to artifacts/combined_segment_interactions.png")

if __name__ == "__main__":
    main()
