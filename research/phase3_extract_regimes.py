import torch
import time
import json

def extract_regimes():
    print("[Extract] Loading 6.5GB Adjacency Matrix...")
    t0 = time.time()
    adj = torch.load('artifacts/adjacency_matrix.pt', weights_only=False)
    N = adj.shape[0]
    print(f"[Extract] Loaded in {time.time()-t0:.2f}s")
    
    print("[Extract] Calculating Degree Centrality (Tier 1 & 2) chunk-by-chunk...")
    t1 = time.time()
    degrees = torch.zeros(N, dtype=torch.int32)
    
    # Process in chunks to avoid blowing up memory with boolean masks
    for i in range(0, N, 2000):
        chunk = adj[i:i+2000]
        # Match Tier 1 or 2
        mask = (chunk == 1) | (chunk == 2)
        degrees[i:i+2000] = mask.sum(dim=1).to(torch.int32)
    print(f"[Extract] Degrees calculated in {time.time()-t1:.2f}s")
    
    print("[Extract] Sorting segments by Most Matches (Degree Centrality)...")
    sorted_indices = torch.argsort(degrees, descending=True)
    
    assigned = torch.zeros(N, dtype=torch.bool)
    buckets = {}
    
    print("[Extract] Commencing Bucket Extraction...")
    bucket_id = 0
    t2 = time.time()
    
    for idx in sorted_indices:
        idx = idx.item()
        
        # Stop if we hit segments that have zero high-conviction matches
        if degrees[idx] == 0:
            break
            
        # If this root is already inside another bucket, skip it
        if assigned[idx]:
            continue
            
        # Create new bucket
        bucket_id += 1
        
        # Get the row for this segment (all its evaluations against everything else)
        root_row = adj[idx]
        
        # We accept Tier 1, 2, 3, and 4 matches into the bucket!
        # Break them down into specific tiers so we retain resolution for modeling
        
        mask_t1 = (root_row == 1) & (~assigned)
        mask_t1[idx] = True # Force the root itself to be in Tier 1
        
        mask_t2 = (root_row == 2) & (~assigned)
        mask_t3 = (root_row == 3) & (~assigned)
        mask_t4 = (root_row == 4) & (~assigned)
        
        # Convert boolean masks to list of segment indices
        m_t1 = torch.where(mask_t1)[0].tolist()
        m_t2 = torch.where(mask_t2)[0].tolist()
        m_t3 = torch.where(mask_t3)[0].tolist()
        m_t4 = torch.where(mask_t4)[0].tolist()
        
        total_members = len(m_t1) + len(m_t2) + len(m_t3) + len(m_t4)
        
        buckets[bucket_id] = {
            "root_segment": idx,
            "tier_1_2_degree": degrees[idx].item(),
            "total_members": total_members,
            "members_tier_1": m_t1,
            "members_tier_2": m_t2,
            "members_tier_3": m_t3,
            "members_tier_4": m_t4
        }
        
        # Mark all pulled members as permanently assigned so they can't join another bucket
        assigned[mask_t1 | mask_t2 | mask_t3 | mask_t4] = True
        
        if bucket_id % 500 == 0:
            print(f"  Extracted {bucket_id} buckets... ({assigned.sum().item()}/{N} segments classified)")
            
    print(f"\n[Extract] Extraction Complete in {time.time()-t2:.2f}s!")
    print(f"Total Unique Regimes Found: {bucket_id}")
    print(f"Total Segments Classified: {assigned.sum().item()} / {N}")
    print(f"Noise (Unclassified): {N - assigned.sum().item()} / {N}")
    
    # Save the buckets to disk
    with open('artifacts/regime_buckets.json', 'w') as f:
        json.dump(buckets, f)
    print("[Extract] Saved to artifacts/regime_buckets.json")

if __name__ == '__main__':
    extract_regimes()
