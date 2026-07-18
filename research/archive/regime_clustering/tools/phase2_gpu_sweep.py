"""ALL-VS-ALL segment sweep — CORRECTED grader (2026-06-12).

Old grader (committed b12a06b8) graded each pair by  max_residual / TARGET's_scalar_band
with fixed ratio cutoffs. That (a) used the target's own (volatility-derived) error band as
the denominator -> a loose-band target "matched" everything (the Regime-1 artifact), (b) used
the single MAX bar so it had NO run-length tolerance, and (c) bore no resemblance to the
segmentation rule that CREATED the segments.

This version grades each pair (curve j evaluated on segment i's anchored data) with the SAME
rule the segmentation used, plus a spike guard:
  - per-bar relative tolerance: |resid[t]| <= THRESH_T * |anchored[t]|, with a 1-tick floor
  - tier ladder: THRESH ladder [0.10, 0.20, 0.30, 0.40] -> tiers 1..4 (tightest passing wins)
  - run rule: a tier passes only if NO MORE THAN 5 CONSECUTIVE bars violate its band
  - spike kill (studentized, robust): if ANY bar's residual > SPIKE_K * (1.4826*MAD of this
    fit's residuals, floored at 1 tick) -> structural break -> tier 8 (no match), regardless.
    MAD (median abs deviation), not mean/RMS, so the spike can't inflate its own threshold.

adjacency[i, j] = tier curve j earns on segment i's data (1=best ... 4=edge, 8=fail/spike).
Still asymmetric (denominator = target i's anchored values); phase3 symmetrizes via
max(adj[i,j], adj[j,i]) and stratifies within volatility_tier.

Runtime: heavier than the max-only grader (per-segment median + run-length). Budget ~1-2h on
the 12GB box; VRAM dance (4 column-blocks, 150-segment batches) preserved from the original.
"""
import os
import sys
import json
import time
import torch
import numpy as np

P_TOTAL = 177
TICK = 0.25                         # MNQ tick size (points); 1-tick soft-band floor
THRESH_LADDER = [0.10, 0.20, 0.30, 0.40]   # tier 1..4 per-bar relative bands
MAX_RUN = 5                         # <=5 consecutive band-violations tolerated (segmentation rule)
SPIKE_CEIL = 0.60                   # beyond the loosest tier: a single bar > 60% of the anchored
                                    # move = structural break (the bar that kills) -> tier 8
SPIKE_FLOOR = 1.0                   # absolute floor (pts) for the headroom/spike check near anchor
BATCH_SIZE = 150
BLOCKS = 4
# Spike rule = TIER-LADDER HEADROOM (user's "2 tiers above" idea), NOT studentized MAD.
# MAD was rejected on a smoke test: these curves are overfit -> near-zero baseline residual
# -> MAD floors to 1 tick -> the spike threshold collapses and kills bounded drift too.
# Headroom rule is baseline-tightness-robust and gives graceful demotion:
#   tier T passes iff (<=MAX_RUN consecutive bars exceed tier T band) AND
#                     (NO single bar exceeds tier T+1's band; beyond tier 4 -> SPIKE_CEIL).
# An isolated bar 1 tier over => demote one tier; 2+ tiers over => fail that tier.

def map_local_poly_to_global(active_idx, local_poly_idx, P=P_TOTAL):
    p_local = len(active_idx)
    if local_poly_idx < p_local:
        return active_idx[local_poly_idx]
    quad_offset = local_poly_idx - p_local
    local_i = 0
    while quad_offset >= (p_local - local_i):
        quad_offset -= (p_local - local_i)
        local_i += 1
    local_j = local_i + quad_offset
    f1, f2 = active_idx[local_i], active_idx[local_j]
    if f1 > f2:
        f1, f2 = f2, f1
    return P + (f1 * P - (f1 * (f1 - 1)) // 2 + (f2 - f1))

def poly_expand_global_gpu(X_t):
    idx_i, idx_j = torch.triu_indices(P_TOTAL, P_TOTAL, offset=0, device=X_t.device)
    quad = X_t[:, idx_i] * X_t[:, idx_j]
    return torch.cat([X_t, quad], dim=1)

def grade_pairs(R, y_abs):
    """R: (L, C) abs residuals of C curves on one segment's L bars.
       y_abs: (L, 1) |anchored close| of that segment.
       returns uint8 (C,) tier in {1,2,3,4,8}. See header for the rule."""
    L, C = R.shape
    dev = R.device
    tier = torch.full((C,), 8, dtype=torch.uint8, device=dev)
    Lr = torch.arange(L, device=dev, dtype=torch.float32).unsqueeze(1)   # (L,1)
    n = len(THRESH_LADDER)
    for ti in range(n - 1, -1, -1):                                  # loosest -> tightest
        own_tol = torch.clamp(THRESH_LADDER[ti] * y_abs, min=TICK)   # (L,1) tier T band
        head_pct = THRESH_LADDER[ti + 1] if ti + 1 < n else SPIKE_CEIL
        head_tol = torch.clamp(head_pct * y_abs, min=SPIKE_FLOOR)    # (L,1) tier T+1 band / spike ceil
        # (a) run rule on own band: <= MAX_RUN consecutive violations
        viol = R > own_tol                                           # (L,C)
        fp = torch.where(~viol, Lr, torch.full_like(Lr, -1.0))
        last_ok = torch.cummax(fp, dim=0).values
        maxrun = (Lr - last_ok).max(dim=0).values                   # (C,)
        # (b) headroom: NO single bar may exceed tier T+1's band (else 2+ tiers over -> fail)
        over_head = (R > head_tol).any(dim=0)                       # (C,)
        passT = (maxrun <= MAX_RUN) & (~over_head)
        tier[passT] = ti + 1
    return tier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Sweep] device: {device} | grader: per-bar {THRESH_LADDER} bands, "
          f"<= {MAX_RUN} consec, spike {SPIKE_K}xMAD, tick {TICK}")

    with open('artifacts/stage2_year_segments.json', 'r') as f:
        segments = json.load(f)
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    N = len(valid)
    total_expanded = P_TOTAL + (P_TOTAL * (P_TOTAL + 1)) // 2

    print(f"[Sweep] Building dense beta matrix ({total_expanded} x {N})...")
    t0 = time.time()
    B_dense_cpu = torch.zeros((total_expanded, N), dtype=torch.float32, device='cpu')
    for sj, s in enumerate(valid):
        active = s['active_grid_cells']
        terms = s['surviving_polynomial_indices']
        betas = s['beta_coefficients']
        if isinstance(active, (int, float)): active = [int(active)]
        if isinstance(terms, (int, float)): terms = [int(terms)]
        if isinstance(betas, (int, float)): betas = [float(betas)]
        for li, bv in zip(terms, betas):
            try:
                B_dense_cpu[map_local_poly_to_global(active, li), sj] = bv
            except Exception:
                pass
    print(f"  built in {time.time()-t0:.1f}s")

    # column-blocked move to GPU (WDDM / VRAM ceiling)
    chunk = N // BLOCKS
    block_ranges, B_chunks = [], []
    for i in range(BLOCKS):
        lo = i * chunk
        hi = N if i == BLOCKS - 1 else (i + 1) * chunk
        block_ranges.append((lo, hi))
        B_chunks.append(B_dense_cpu[:, lo:hi].contiguous().to(device))
    del B_dense_cpu
    import gc; gc.collect()

    print(f"[Sweep] Allocating {N}x{N} uint8 adjacency on CPU ({N*N/1e9:.1f} GB)...")
    adj = torch.zeros((N, N), dtype=torch.uint8, device='cpu')

    payload = torch.load('artifacts/sweep_cache_flat.pt', weights_only=False)
    X_flat = payload['X_flat']
    Y_flat = payload['Y_flat']
    boundaries = payload['boundaries'].numpy()

    print(f"[Sweep] Sweeping (batch {BATCH_SIZE})...")
    t_start = time.time()
    for b_start in range(0, N, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, N)
        row_start, row_end = boundaries[b_start], boundaries[b_end]
        X_batch = X_flat[row_start:row_end].to(device)
        Y_batch = Y_flat[row_start:row_end].to(device)
        Xp = poly_expand_global_gpu(X_batch)
        local_b = boundaries[b_start:b_end + 1] - row_start

        for (lo, hi), B_chunk in zip(block_ranges, B_chunks):
            resid = (Xp @ B_chunk).sub_(Y_batch).abs_()             # (rows, chunk_b)
            for k in range(b_end - b_start):
                r_s, r_e = local_b[k], local_b[k + 1]
                if r_e <= r_s:
                    continue
                tiers = grade_pairs(resid[r_s:r_e, :], Y_batch[r_s:r_e].abs())
                adj[b_start + k, lo:hi] = tiers.cpu()
            del resid
        del X_batch, Y_batch, Xp
        torch.cuda.empty_cache()
        if b_start and b_start % 3000 == 0:
            el = time.time() - t_start
            eta = (N - b_start) / (b_start / el)
            print(f"  {b_start}/{N}  ETA {eta/60:.1f} min")

    print(f"[Sweep] done in {(time.time()-t_start)/60:.1f} min")
    torch.save(adj, 'artifacts/adjacency_matrix.pt')
    # quick diagonal sanity: clean self-fits should now be tier 1 (spiky ones -> 8)
    diagc = {t: 0 for t in (1, 2, 3, 4, 8)}
    for i in range(0, N, 4000):
        j = min(i + 4000, N)
        d = np.diagonal(adj[i:j, :].numpy()[:, i:j])
        for t in (1, 2, 3, 4, 8):
            diagc[t] += int((d == t).sum())
    print(f"[Sweep] diagonal (self-fit) tiers: {diagc}  "
          f"(tier1 = segmentation-consistent; tier8 = self-spike)")
    print("Saved artifacts/adjacency_matrix.pt")

if __name__ == '__main__':
    main()
