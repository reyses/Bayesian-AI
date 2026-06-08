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
    
    # Dictionary to accumulate sum of absolute betas for each term
    term_betas = {}
    term_counts = {}
    
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
                
                idx_i, idx_j = torch.triu_indices(p, p, offset=0)
                
                for idx, k in enumerate(surviving):
                    beta_val = abs(betas[idx])
                    
                    if k < p:
                        # Linear term
                        global_i = active[k]
                        term_name = features_cols[global_i]
                    else:
                        # Interaction term
                        offset = k - p
                        if offset < len(idx_i):
                            local_i = idx_i[offset].item()
                            local_j = idx_j[offset].item()
                            global_i = active[local_i]
                            global_j = active[local_j]
                            
                            name_i = features_cols[global_i]
                            name_j = features_cols[global_j]
                            
                            if global_i == global_j:
                                term_name = f"{name_i}*"
                            else:
                                term_name = f"{name_i} x {name_j}"
                        else:
                            continue
                            
                    if term_name not in term_betas:
                        term_betas[term_name] = 0.0
                        term_counts[term_name] = 0
                        
                    term_betas[term_name] += beta_val
                    term_counts[term_name] += 1

    # Calculate average standardized effect across all segments where the term appeared
    avg_effects = {k: term_betas[k] / term_counts[k] for k in term_betas}
    
    # Alternatively, calculate total standardized effect (proxy for overall importance)
    # We will plot the top 30 by average magnitude where count > 100 to avoid outliers
    valid_effects = {k: v for k, v in avg_effects.items() if term_counts[k] > 50}
    
    sorted_terms = sorted(valid_effects.items(), key=lambda x: x[1], reverse=True)[:30]
    
    terms = [x[0] for x in sorted_terms]
    effects = [x[1] for x in sorted_terms]
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=effects, y=terms, palette="coolwarm")
    
    # Add a pseudo-significance line (Minitab style)
    mean_effect = np.mean(list(valid_effects.values()))
    std_effect = np.std(list(valid_effects.values()))
    sig_line = mean_effect + 2 * std_effect
    plt.axvline(sig_line, color='red', linestyle='--', label='Pseudo-Significance Threshold (2 Std Dev)')
    
    plt.title("Standardized Pareto Chart of Effects (Minitab Style)", fontsize=16)
    plt.xlabel("Average Absolute Beta Coefficient (Standardized Effect)", fontsize=12)
    plt.ylabel("Feature / Interaction Term", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/minitab_pareto_effect.png', dpi=150)
    print("\n[INFO] Saved Pareto chart to artifacts/minitab_pareto_effect.png")

if __name__ == "__main__":
    main()
