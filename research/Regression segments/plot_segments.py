import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.features import load_features

def poly_expand_gpu(X_t):
    device = X_t.device
    N, p = X_t.shape
    idx_i, idx_j = torch.triu_indices(p, p, offset=0, device=device)
    quad = X_t[:, idx_i] * X_t[:, idx_j]
    return torch.cat([X_t, quad], dim=1)

def main():
    day = '2025_02_05'
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    # 1. Load Data
    print(f"Loading data for {day}...")
    df = load_features([day], root=features_root)
    ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f'{day}.parquet'))
    
    min_len = min(len(df), len(ohlcv))
    df = df.iloc[:min_len]
    ohlcv = ohlcv.iloc[:min_len]
    
    features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    scaler = StandardScaler()
    X_global = scaler.fit_transform(df[features_cols].values)
    
    valid_idx = ~np.isnan(X_global).any(axis=1)
    X_global = X_global[valid_idx]
    close_prices = ohlcv['close'].values[valid_idx]
    
    X_global_t = torch.tensor(X_global, dtype=torch.float32)
    
    # 2. Load Segments
    json_path = f"artifacts/stage2_segments_{day}.json"
    print(f"Loading segments from {json_path}...")
    with open(json_path, 'r') as f:
        segments = json.load(f)
        
    # 3. Setup Plot
    print("Generating plot...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(30, 10), dpi=100)
    
    # Plot base price
    x_axis = np.arange(len(close_prices))
    ax.plot(x_axis, close_prices, color='white', linewidth=0.8, alpha=0.9, label='Close Price')
    
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap('RdYlGn_r') # Green to Red
    
    for seg in segments:
        s_idx = seg['start_idx']
        e_idx = seg['end_idx']
        status = seg['status']
        E = seg['error_band_used']
        tier = seg.get('volatility_tier', 9)
        
        # Make sure indices are within bounds
        if s_idx >= len(close_prices) or s_idx >= e_idx:
            continue
            
        e_idx = min(e_idx, len(close_prices))
        L = e_idx - s_idx
        
        x_range = x_axis[s_idx:e_idx]
        
        # Calculate color based on Tier 1 to 9
        norm = max(0, min(1, (tier - 1) / 8.0))
        color = mcolors.to_hex(cmap(norm))
        
        beta_raw = seg.get('beta_coefficients', [])
        if isinstance(beta_raw, (float, int)):
            beta_raw = [beta_raw]
        
        if status in ['PRISTINE', 'RECOVERED'] and len(beta_raw) > 0:
            active_idx = seg['active_grid_cells']
            fixed_terms = seg['surviving_polynomial_indices']
            beta = torch.tensor(beta_raw, dtype=torch.float32)
            
            # Reconstruct regression
            X_chunk = X_global_t[s_idx:e_idx]
            if len(active_idx) > 0:
                X_sub = X_chunk[:, active_idx]
                X_poly = poly_expand_gpu(X_sub)
                
                # Check if fixed_terms are valid indices
                if len(fixed_terms) > 0 and max(fixed_terms) < X_poly.shape[1]:
                    X_poly_fixed = X_poly[:, fixed_terms]
                    
                    if X_poly_fixed.shape[1] == len(beta):
                        preds = (X_poly_fixed @ beta).cpu().numpy()
                        regression_line = preds + close_prices[s_idx]
                        
                        # Plot regression line
                        ax.plot(x_range, regression_line, color=color, linewidth=1.5, alpha=0.8)
                        
                        # Plot error band
                        ax.fill_between(x_range, regression_line - E, regression_line + E, 
                                        color=color, alpha=0.4)
                    else:
                        # Fallback shading
                        ax.axvspan(s_idx, e_idx, color=color, alpha=0.2)
                else:
                    ax.axvspan(s_idx, e_idx, color=color, alpha=0.2)
            else:
                ax.axvspan(s_idx, e_idx, color=color, alpha=0.2)
                
        elif status == 'PURE_CHAOS':
            # Shade Chaos
            ax.axvspan(s_idx, e_idx, color=color, alpha=0.3)
            
    # Clean up plot aesthetics
    ax.set_title(f'Segment Classification by Volatility Tiers (1-9) - {day}', fontsize=18, fontweight='bold', color='white')
    ax.set_ylabel('Price', fontsize=14, color='white')
    ax.set_xlabel('Bar Index (5s)', fontsize=14, color='white')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='white', lw=1, label='Close Price')]
    
    # Add a representative sample of Tiers to the legend
    legend_elements.append(patches.Patch(facecolor=mcolors.to_hex(cmap(0.0)), alpha=0.4, label='Tier 1 (Pristine)'))
    legend_elements.append(patches.Patch(facecolor=mcolors.to_hex(cmap(3/8.0)), alpha=0.4, label='Tier 4 (Recovered)'))
    legend_elements.append(patches.Patch(facecolor=mcolors.to_hex(cmap(6/8.0)), alpha=0.4, label='Tier 7 (High Volatility)'))
    legend_elements.append(patches.Patch(facecolor=mcolors.to_hex(cmap(1.0)), alpha=0.4, label='Tier 9 (Pure Chaos)'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.5)
    
    plt.tight_layout()
    out_path = f"artifacts/segment_plot_{day}.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved plot to {out_path}")
    
if __name__ == "__main__":
    main()
