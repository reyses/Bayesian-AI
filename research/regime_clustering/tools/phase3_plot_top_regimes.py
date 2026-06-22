import torch
import json
import matplotlib.pyplot as plt

def plot_top_regimes():
    print("[Plot] Loading flat cache and buckets...")
    cache = torch.load('artifacts/sweep_cache_flat.pt', weights_only=False)
    Y_flat = cache['Y_flat']
    boundaries = cache['boundaries']
    
    with open('artifacts/regime_buckets.json', 'r') as f:
        buckets = json.load(f)
        
    for i in range(1, 6):
        b_id = str(i)
        bucket = buckets[b_id]
        root_idx = bucket['root_segment']
        
        plt.figure(figsize=(12, 7))
        
        # Plot up to 300 Tier 1/Tier 2 members in the background to show the exact geometric envelope
        members_to_plot = (bucket['members_tier_1'] + bucket['members_tier_2'])[:300]
        
        for mem_idx in members_to_plot:
            start = boundaries[mem_idx]
            end = boundaries[mem_idx+1]
            y = Y_flat[start:end].numpy()
            plt.plot(y, color='royalblue', alpha=0.05)
            
        # Plot root segment in thick red
        start = boundaries[root_idx]
        end = boundaries[root_idx+1]
        root_y = Y_flat[start:end].numpy()
        plt.plot(root_y, color='crimson', linewidth=3, label=f'Root Segment #{root_idx}')
        
        # Format the beautiful chart
        plt.title(f"Alpha Market Regime {i}\n(Overlay of {len(members_to_plot)} highly correlated Tier 1/2 geometric paths)", fontsize=14, fontweight='bold')
        plt.xlabel("Time (5s ticks over 60 bars)", fontsize=12)
        plt.ylabel("Price Offset (0-anchored)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add a sleek dark background option for premium look
        plt.style.use('dark_background')
        plt.tight_layout()
        
        save_path = f'artifacts/regime_{i}.png'
        plt.savefig(save_path, facecolor='#111111')
        plt.close()
        print(f"  Generated {save_path}")
        
    print("[Plot] All plots generated successfully!")
    
if __name__ == '__main__':
    plot_top_regimes()
