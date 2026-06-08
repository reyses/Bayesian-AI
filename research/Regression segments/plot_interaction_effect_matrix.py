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
    
    # 2D matrices: one for accumulating effect sums, one for counts
    effect_sum_matrix = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float32)
    effect_count_matrix = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float32)
    
    total_segments = 0
    
    for f_path in files:
        with open(f_path, 'r') as f:
            segments = json.load(f)
            
        for seg in segments:
            if seg.get('status') in ['PRISTINE', 'RECOVERED']:
                active = seg.get('active_grid_cells', [])
                surviving = seg.get('surviving_polynomial_indices', [])
                betas = seg.get('beta_coefficients', [])
                
                p = len(active)
                if not isinstance(betas, list):
                    betas = [betas]
                
                if p == 0 or len(surviving) == 0 or len(betas) != len(surviving):
                    continue
                    
                total_segments += 1
                
                idx_i, idx_j = torch.triu_indices(p, p, offset=0)
                
                for idx, k in enumerate(surviving):
                    beta_val = abs(betas[idx])
                    if k >= p:
                        # Interaction/Quadratic term!
                        offset = k - p
                        if offset < len(idx_i):
                            local_i = idx_i[offset].item()
                            local_j = idx_j[offset].item()
                            
                            global_i = active[local_i]
                            global_j = active[local_j]
                            
                            # Increment sum and count
                            effect_sum_matrix[global_i, global_j] += beta_val
                            effect_count_matrix[global_i, global_j] += 1
                            
                            if global_i != global_j:
                                effect_sum_matrix[global_j, global_i] += beta_val
                                effect_count_matrix[global_j, global_i] += 1
                                
    print(f"[INFO] Processed {total_segments} pristine segments.")
    
    # Calculate average effect sizes where count > 0, otherwise 0
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_effect_matrix = np.nan_to_num(effect_sum_matrix / effect_count_matrix)
        
    # We want to plot the densest sub-matrix of high-effect features
    # Let's find the features with the highest max interaction effects
    # Or top N pairs by total count to keep the matrix focused on statistically robust effects
    upper_tri_counts = np.triu(effect_count_matrix)
    flat_indices = np.argsort(upper_tri_counts.flatten())[::-1]
    
    # Get top 50 pairs by frequency to construct the grid
    top_50_pairs = []
    for idx in flat_indices[:50]:
        val = upper_tri_counts.flatten()[idx]
        if val > 50:  # Require at least 50 observations to plot
            i, j = np.unravel_index(idx, effect_count_matrix.shape)
            top_50_pairs.append((i, j))
            
    unique_active_features = set()
    for i, j in top_50_pairs:
        unique_active_features.add(i)
        unique_active_features.add(j)
        
    unique_active_features = list(unique_active_features)
    # Sort them by their max effect size to order the heatmap nicely
    feature_max_effect = [np.max(avg_effect_matrix[idx, :]) for idx in unique_active_features]
    unique_active_features = [x for _, x in sorted(zip(feature_max_effect, unique_active_features), reverse=True)]
    
    sub_matrix = np.zeros((len(unique_active_features), len(unique_active_features)))
    labels = [features_cols[idx] for idx in unique_active_features]
    
    for row_idx, f_i in enumerate(unique_active_features):
        for col_idx, f_j in enumerate(unique_active_features):
            sub_matrix[row_idx, col_idx] = avg_effect_matrix[f_i, f_j]
            
    plt.figure(figsize=(14, 12))
    # Using a diverging colormap to show effect magnitude
    sns.heatmap(sub_matrix, xticklabels=labels, yticklabels=labels, cmap="magma", annot=False)
    plt.title("Quadratic & Interaction Effect Matrix\n(Average Absolute Beta Magnitude in Pristine Regimes)", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('artifacts/quadratic_interaction_effect_matrix.png', dpi=150)
    print("\n[INFO] Saved heatmap to artifacts/quadratic_interaction_effect_matrix.png")

if __name__ == "__main__":
    main()
