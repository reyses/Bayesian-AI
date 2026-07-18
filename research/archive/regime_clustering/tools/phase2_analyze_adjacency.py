import torch
import time

def analyze_adjacency():
    print("Loading 6.5GB Adjacency Matrix...")
    t0 = time.time()
    adj = torch.load('artifacts/adjacency_matrix.pt', weights_only=False)
    print(f"Loaded in {time.time()-t0:.2f} seconds!")
    
    print(f"Matrix shape: {adj.shape}, dtype: {adj.dtype}")
    
    # Count occurrences safely without massive RAM spikes
    total_elements = adj.numel()
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 8: 0}
    
    for i in range(0, len(adj), 5000):
        chunk = adj[i:i+5000]
        # Compare with PyTorch uint8 tensor to avoid promotion to 52GB int64 masks
        for t in [1, 2, 3, 4, 8]:
            t_tensor = torch.tensor(t, dtype=torch.uint8)
            tier_counts[t] += torch.sum(chunk == t_tensor).item()
            
    tier_1 = tier_counts[1]
    tier_2 = tier_counts[2]
    tier_3 = tier_counts[3]
    tier_4 = tier_counts[4]
    tier_8 = tier_counts[8]
    
    print(f"\nDistribution across {total_elements:,} total evaluated pairings:")
    print(f"  Tier 1 (Flawless Match <= 1.0x bounds) : {tier_1:>12,}  ({tier_1/total_elements*100:>7.4f}%)")
    print(f"  Tier 2 (Great Match   <= 1.5x bounds) : {tier_2:>12,}  ({tier_2/total_elements*100:>7.4f}%)")
    print(f"  Tier 3 (Good Match    <= 2.0x bounds) : {tier_3:>12,}  ({tier_3/total_elements*100:>7.4f}%)")
    print(f"  Tier 4 (Edge Match    <= 2.5x bounds) : {tier_4:>12,}  ({tier_4/total_elements*100:>7.4f}%)")
    print(f"  Tier 8 (Failed Match   > 2.5x bounds) : {tier_8:>12,}  ({tier_8/total_elements*100:>7.4f}%)")

if __name__ == '__main__':
    analyze_adjacency()
