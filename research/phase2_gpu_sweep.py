import os
import sys
import json
import time
import torch
import numpy as np

def map_local_poly_to_global(active_idx, local_poly_idx, P_total):
    p_local = len(active_idx)
    if local_poly_idx < p_local: return active_idx[local_poly_idx]
    
    quad_offset = local_poly_idx - p_local
    local_i = 0
    while quad_offset >= (p_local - local_i):
        quad_offset -= (p_local - local_i)
        local_i += 1
    local_j = local_i + quad_offset
    
    f1, f2 = active_idx[local_i], active_idx[local_j]
    if f1 > f2: f1, f2 = f2, f1
    idx = f1 * P_total - (f1 * (f1 - 1)) // 2 + (f2 - f1)
    return P_total + idx

def poly_expand_global_gpu(X_t):
    device = X_t.device
    idx_i, idx_j = torch.triu_indices(177, 177, offset=0, device=device)
    quad = X_t[:, idx_i] * X_t[:, idx_j]
    return torch.cat([X_t, quad], dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Sweep] Using device: {device}")
    
    with open('artifacts/stage2_year_segments.json', 'r') as f:
        segments = json.load(f)
        
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    N_segments = len(valid)
    P_total = 177
    total_expanded_features = P_total + (P_total * (P_total + 1)) // 2
    
    print(f"[Sweep] Building Dense Beta Matrix ({total_expanded_features} x {N_segments})...")
    t0 = time.time()
    # ALLOCATE ON CPU FIRST to avoid 4.4 million PCIe transfers!
    B_dense_cpu = torch.zeros((total_expanded_features, N_segments), dtype=torch.float32, device='cpu')
    
    for seg_idx, s in enumerate(valid):
        active_idx = s['active_grid_cells']
        fixed_terms = s['surviving_polynomial_indices']
        betas = s['beta_coefficients']
        
        if isinstance(active_idx, (int, float)): active_idx = [int(active_idx)]
        if isinstance(fixed_terms, (int, float)): fixed_terms = [int(fixed_terms)]
        if isinstance(betas, (int, float)): betas = [float(betas)]
        
        for local_poly_idx, beta_val in zip(fixed_terms, betas):
            try:
                global_idx = map_local_poly_to_global(active_idx, local_poly_idx, P_total)
                B_dense_cpu[global_idx, seg_idx] = beta_val
            except: pass
            
    print(f"  Matrix built in {time.time()-t0:.2f}s")
    
    print(f"[Sweep] Moving Dense Matrix to GPU in 4 blocks to bypass WDDM limit & lower VRAM peak...")
    blocks = 4
    chunk_size = N_segments // blocks
    B_chunks_gpu = []
    
    for i in range(blocks):
        start_idx = i * chunk_size
        end_idx = N_segments if i == blocks - 1 else (i + 1) * chunk_size
        B_chunks_gpu.append(B_dense_cpu[:, start_idx:end_idx].contiguous().to(device))
        
    del B_dense_cpu
    import gc; gc.collect()
    
    print(f"[Sweep] Allocating 6.5GB Adjacency Matrix on CPU...")
    # Rows: Target Segment Reality. Columns: Curve Equation.
    # Value: The exact tier (1-8) that Curve j achieved on Segment i's data.
    adj_matrix = torch.zeros((N_segments, N_segments), dtype=torch.uint8, device='cpu')
    
    print(f"\n[Sweep] Loading Flat Tensor Payload...")
    t_load = time.time()
    payload = torch.load('artifacts/sweep_cache_flat.pt', weights_only=False)
    X_flat = payload['X_flat']
    Y_flat = payload['Y_flat']
    boundaries = payload['boundaries'].numpy()
    error_bands = payload['error_bands'].to(device)
    print(f"  Loaded successfully in {time.time()-t_load:.2f}s!")
    
    BATCH_SIZE = 150
    print(f"\n[Sweep] Commencing Batched GPU Matrix Sweep (Batch Size {BATCH_SIZE})...")
    
    t_start = time.time()
    
    for b_start in range(0, N_segments, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, N_segments)
        batch_size = b_end - b_start
        
        row_start = boundaries[b_start]
        row_end = boundaries[b_end]
        
        # Slicing the exact contiguous batch chunk!
        X_batch = X_flat[row_start:row_end].to(device)
        Y_batch = Y_flat[row_start:row_end].to(device)
        batch_error_bands = error_bands[b_start:b_end]
        
        X_batch_poly = poly_expand_global_gpu(X_batch)
        
        max_res_blocks = [ [] for _ in range(blocks) ]
        local_boundaries = boundaries[b_start:b_end+1] - row_start
        
        # Process each block sequentially to keep peak VRAM under 6 GB total
        for i, B_chunk in enumerate(B_chunks_gpu):
            Y_preds = torch.matmul(X_batch_poly, B_chunk)
            Y_preds.sub_(Y_batch).abs_()
            
            for idx_in_batch in range(batch_size):
                r_s = local_boundaries[idx_in_batch]
                r_e = local_boundaries[idx_in_batch+1]
                m, _ = torch.max(Y_preds[r_s:r_e, :], dim=0)
                max_res_blocks[i].append(m)
                
            del Y_preds
        
        # Combine the blocks and calculate tiers
        for idx_in_batch in range(batch_size):
            max_res = torch.cat([ max_res_blocks[i][idx_in_batch] for i in range(blocks) ])
            
            ratios = max_res / batch_error_bands[idx_in_batch]
            
            tiers = torch.ones_like(ratios, dtype=torch.uint8) * 8
            tiers[ratios <= 2.5] = 4
            tiers[ratios <= 2.0] = 3
            tiers[ratios <= 1.5] = 2
            tiers[ratios <= 1.0] = 1
            
            global_row_idx = b_start + idx_in_batch
            adj_matrix[global_row_idx, :] = tiers.cpu()
            
        # Free VRAM before next batch
        del X_batch, Y_batch, X_batch_poly
        for i in range(blocks): max_res_blocks[i].clear()
        torch.cuda.empty_cache()
            
        # Logging progress
        if b_start > 0 and b_start % 5000 == 0:
            elapsed = time.time() - t_start
            rate = b_start / elapsed
            eta = (N_segments - b_start) / rate
            print(f"  Processed {b_start}/{N_segments} segments... ETA: {eta/60:.1f} minutes")
            
    total_time = time.time() - t_start
    print(f"\n[Sweep] ALL-VS-ALL Sweep completed in {total_time/60:.2f} minutes!")
    
    out_path = 'artifacts/adjacency_matrix.pt'
    print(f"[Sweep] Saving final 6.5GB boolean adjacency matrix to {out_path}...")
    torch.save(adj_matrix, out_path)
    print("Done!")

if __name__ == '__main__':
    main()
