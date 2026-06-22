import sys
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_regime(regime_id=1, output_file='artifacts/regime_1_animated.gif'):
    print(f"[Animate] Loading flat cache and buckets for Regime {regime_id}...")
    cache = torch.load('artifacts/sweep_cache_flat.pt', weights_only=False)
    Y_flat = cache['Y_flat']
    boundaries = cache['boundaries']
    
    with open('artifacts/regime_buckets.json', 'r') as f:
        buckets = json.load(f)
        
    bucket = buckets[str(regime_id)]
    root_idx = bucket['root_segment']
    
    # We will plot the root, then frame-by-frame draw the members
    # Limit to 150 members to keep GIF file size manageable
    members_to_plot = (bucket['members_tier_1'] + bucket['members_tier_2'])[:150] 
    
    # Setup Figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Alpha Market Regime {regime_id}\n(Geometric Channel Formation)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Dynamic 5s Ticks)", fontsize=12)
    ax.set_ylabel("Price Offset (0-anchored)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Pre-calculate axes limits so it doesn't jump around
    all_y = []
    
    # Root
    start = boundaries[root_idx]
    end = boundaries[root_idx+1]
    root_y = Y_flat[start:end].numpy()
    all_y.extend(root_y.flatten())
    
    member_ys = []
    for mem_idx in members_to_plot:
        start = boundaries[mem_idx]
        end = boundaries[mem_idx+1]
        y = Y_flat[start:end].numpy()
        member_ys.append(y)
        all_y.extend(y.flatten())
        
    min_y = min(all_y)
    max_y = max(all_y)
    ax.set_ylim(min_y - 5, max_y + 5)
    
    max_len = max([len(root_y)] + [len(y) for y in member_ys])
    ax.set_xlim(0, max_len + 5)
    
    # Plot root segment in thick red permanently
    ax.plot(root_y, color='crimson', linewidth=3, label=f'Root Segment #{root_idx}')
    ax.legend(loc='upper left', fontsize=10)
    
    lines = []
    
    def init():
        return lines
        
    def update(frame):
        # Every frame, we draw one more member on top of the root
        if frame < len(member_ys):
            y = member_ys[frame]
            line, = ax.plot(y, color='royalblue', alpha=0.2)
            lines.append(line)
        return lines

    print(f"[Animate] Rendering {len(members_to_plot)} frames to {output_file}...")
    ani = animation.FuncAnimation(fig, update, frames=len(members_to_plot), init_func=init, blit=True, interval=50)
    ani.save(output_file, writer='pillow', fps=20)
    plt.close()
    print("[Animate] Done!")

if __name__ == '__main__':
    regime = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = sys.argv[2] if len(sys.argv) > 2 else f'artifacts/regime_{regime}_animated.gif'
    animate_regime(regime, out)
