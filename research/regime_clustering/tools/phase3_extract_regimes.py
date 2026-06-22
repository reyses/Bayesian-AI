"""Regime-bucket extraction — CORRECTED (2026-06-12).

Two fixes over the committed version (b12a06b8):
  SYMMETRIC (reciprocal tier): a match is graded by max(adj[A,B], adj[B,A]) -- B joins A's
    bucket at tier T only if B fits A's tolerance AND A fits B's. Kills the loose-band tube:
    a wide-band root passes everyone one-way, but must ALSO pass their tight bands the other
    way -- it won't. This is the core repair for the Regime-1 artifact.
  WITHIN_VOL_TIER: a root only recruits members of the SAME volatility_tier, so buckets are
    structure WITHIN a vol level, not a re-derivation of vol regimes.

Greedy degree-centrality bucketing otherwise unchanged. Prints the pass-gates inline so you
see immediately whether the repair worked (top bucket should be <20%, not 56%).

Toggle the two fixes with the constants below.
"""
import torch
import time
import json
import numpy as np

SYMMETRIC = True          # reciprocal tier via max(adj, adj.T)
WITHIN_VOL_TIER = True     # only match segments of the same volatility_tier

def main():
    print(f"[Extract] SYMMETRIC={SYMMETRIC}  WITHIN_VOL_TIER={WITHIN_VOL_TIER}")
    print("[Extract] Loading adjacency matrix...")
    t0 = time.time()
    adj = torch.load('artifacts/adjacency_matrix.pt', weights_only=False)
    N = adj.shape[0]
    print(f"  loaded {N}x{N} in {time.time()-t0:.1f}s")

    # volatility tiers (same valid order as the sweep)
    with open('artifacts/stage2_year_segments.json') as f:
        segs = json.load(f)
    valid = [s for s in segs if s['status'] in ('PRISTINE', 'RECOVERED')]
    assert len(valid) == N, f"valid {len(valid)} != adj {N}"
    vol = np.array([s.get('volatility_tier', -1) for s in valid])
    bands = np.array([s.get('error_band_used', np.nan) for s in valid])

    def sym_row(i):
        """symmetrized row i: max(adj[i,:], adj[:,i]) as int tier (lower = better)."""
        r = adj[i, :]
        if SYMMETRIC:
            c = adj[:, i]
            r = torch.maximum(r, c)
        return r

    # ---- degree centrality on tier 1|2 (symmetrized, optional within-vol), chunked ----
    print("[Extract] Degree centrality (tier 1|2)...")
    t1 = time.time()
    degrees = torch.zeros(N, dtype=torch.int32)
    vol_t = torch.from_numpy(vol)                      # (N,)
    CH = 2000
    for i in range(0, N, CH):
        j = min(i + CH, N)
        rows = adj[i:j, :]                             # (B, N)
        if SYMMETRIC:
            rows = torch.maximum(rows, adj[:, i:j].t().contiguous())  # (B, N)
        m = (rows == 1) | (rows == 2)
        if WITHIN_VOL_TIER:
            same = (vol_t[i:j].unsqueeze(1) == vol_t.unsqueeze(0))    # (B, N)
            m &= same
        degrees[i:j] = m.sum(dim=1).to(torch.int32)
        if i and i % 20000 == 0:
            print(f"  {i}/{N} ({time.time()-t1:.0f}s)")
    print(f"  degrees in {time.time()-t1:.0f}s")

    order = torch.argsort(degrees, descending=True)
    assigned = torch.zeros(N, dtype=torch.bool)
    buckets = {}
    bid = 0
    for idx_t in order:
        idx = idx_t.item()
        if degrees[idx] == 0:
            break
        if assigned[idx]:
            continue
        bid += 1
        r = sym_row(idx)
        vt_ok = (vol_t == int(vol[idx])) if WITHIN_VOL_TIER else torch.ones(N, dtype=torch.bool)
        masks = {}
        for t in (1, 2, 3, 4):
            m = (r == t) & (~assigned) & vt_ok
            if t == 1:
                m[idx] = True
            masks[t] = m
        members = {t: torch.where(masks[t])[0].tolist() for t in (1, 2, 3, 4)}
        total = sum(len(v) for v in members.values())
        buckets[bid] = {
            "root_segment": idx,
            "root_vol_tier": int(vol[idx]),
            "root_error_band": float(bands[idx]) if not np.isnan(bands[idx]) else None,
            "tier_1_2_degree": int(degrees[idx]),
            "total_members": total,
            "members_tier_1": members[1],
            "members_tier_2": members[2],
            "members_tier_3": members[3],
            "members_tier_4": members[4],
        }
        all_m = masks[1] | masks[2] | masks[3] | masks[4]
        assigned[all_m] = True
        if bid % 500 == 0:
            print(f"  {bid} buckets, {int(assigned.sum())}/{N} classified")

    classified = int(assigned.sum())
    with open('artifacts/regime_buckets.json', 'w') as f:
        json.dump(buckets, f)

    # ---- inline pass-gates ----
    sizes = np.array(sorted((b['total_members'] for b in buckets.values()), reverse=True))
    top_share = 100 * sizes[0] / classified if classified else 0
    med_sz = int(np.median(sizes)) if len(sizes) else 0
    print("\n========== PASS GATES ==========")
    print(f"buckets: {bid:,} | classified {classified:,}/{N:,} ({100*classified/N:.1f}%) | "
          f"NOISE {N-classified:,}")
    print(f"top bucket = {sizes[0]:,} = {top_share:.1f}% of classified   "
          f"[GATE: <20%  ->  {'PASS' if top_share < 20 else 'FAIL (still a tube)'}]")
    print(f"median bucket size = {med_sz}   [GATE: >=10  ->  {'PASS' if med_sz >= 10 else 'FAIL'}]")
    # Spearman(root degree, root band): should collapse toward 0 if the band artifact is gone
    try:
        from scipy.stats import spearmanr
        rd = np.array([b['tier_1_2_degree'] for b in buckets.values()])
        rb = np.array([b['root_error_band'] for b in buckets.values()], dtype=float)
        ok = ~np.isnan(rb)
        rho, p = spearmanr(rd[ok], rb[ok])
        print(f"Spearman(degree, root_band) = {rho:+.3f} (p={p:.1e})   "
              f"[GATE: ~0  ->  {'PASS' if abs(rho) < 0.10 else 'WARN (band still drives degree)'}]")
    except Exception as e:
        print(f"Spearman skipped: {e}")
    print("Run research/audit_regime_findings.py for the full gate report + null.")
    print("[Extract] Saved artifacts/regime_buckets.json")

if __name__ == '__main__':
    main()
