"""Does the F-space follow a MEASURABLE pattern?  (post-hoc / lookahead diagnostic)

Question: across 80,717 overfit segment regressions, do the SAME features keep
getting selected to explain the anchored price path -- or does every segment grab
a random subset (= no structure, pure overfit)?

If concentrated -> the heavily-selected features are "the indicators worth seeing"
(a hypothesis for causal feature work; NOT a live signal -- segments use the
whole-day scaler + completed window = lookahead, acknowledged).

Reads only artifacts/stage2_year_segments.json (the betas). No buckets, no 6.5GB matrix.

Method / built-in rigor:
  - map each surviving LOCAL poly term -> global feature(s) (linear credits 1,
    quadratic credits both parents), using the SAME mapping as phase2_gpu_sweep.
  - selection frequency per feature; concentration (Gini, top-k coverage).
  - NULL: re-draw each segment's surviving-count uniformly from its own local
    poly space; recompute concentration. Real >> null => measurable pattern.
  - beta sign stability for linear terms (0.5 = context-dependent; ->0/1 = monotone).
  - stratify by volatility_tier (does structure survive within a vol level, or is
    it just re-deriving vol regimes?).
  - feature index -> V2 name via the SAME column filter phase2_extract_cache uses.
Output: reports/findings/2026-06-12_fspace_pattern.md
"""
import os, sys, json, glob
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

P_TOTAL = 177
OUT = 'reports/findings/2026-06-12_fspace_pattern.md'
RNG = np.random.RandomState(42)
L = []
def log(s=''):
    print(s); L.append(s)

# ── global poly layout: [0..176 linear] + [177.. quad in triu(177,offset=0) order]
iu = np.triu_indices(P_TOTAL, k=0)          # (i,j) with i<=j, length 15753
QUAD_I, QUAD_J = iu[0], iu[1]
N_QUAD = len(QUAD_I)
def decode_global(g):
    """global poly idx -> (kind, feat_a, feat_b)."""
    if g < P_TOTAL:
        return ('lin', g, None)
    k = g - P_TOTAL
    if 0 <= k < N_QUAD:
        return ('quad', int(QUAD_I[k]), int(QUAD_J[k]))
    return ('oob', None, None)

def map_local_poly_to_global(active_idx, local_poly_idx, P=P_TOTAL):
    p_local = len(active_idx)
    if local_poly_idx < p_local:
        return active_idx[local_poly_idx]
    quad_offset = local_poly_idx - p_local
    local_i = 0
    while quad_offset >= (p_local - local_i):
        quad_offset -= (p_local - local_i); local_i += 1
    local_j = local_i + quad_offset
    f1, f2 = active_idx[local_i], active_idx[local_j]
    if f1 > f2: f1, f2 = f2, f1
    return P + (f1 * P - (f1 * (f1 - 1)) // 2 + (f2 - f1))

def local_poly_count(p_local):
    return p_local + (p_local * (p_local + 1)) // 2

# ── feature names (same filter as phase2_extract_cache) ────────────────────
def feature_names():
    try:
        from core_v2.features import load_features
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DATA', 'ATLAS', 'FEATURES_5s_v2'))
        days = sorted(os.path.basename(f).replace('.parquet','') for f in glob.glob(os.path.join(root, 'L3_1m', '*.parquet')))
        if not days:  # fallback: any subdir
            days = sorted(os.path.basename(f).replace('.parquet','') for f in glob.glob(os.path.join(root, '*', '*.parquet')))
        df = load_features([days[0]], root=root)
        cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
        return cols[:P_TOTAL], days[0]
    except Exception as e:
        return None, str(e)

def main():
    log('# Does the F-space follow a measurable selection pattern? (post-hoc / lookahead)')
    log()
    with open('artifacts/stage2_year_segments.json') as f:
        segs = json.load(f)
    valid = [s for s in segs if s['status'] in ('PRISTINE', 'RECOVERED')]
    N = len(valid)
    log(f'- valid segments: {N:,}  (PRISTINE/RECOVERED). Lookahead acknowledged: whole-day scaler + completed window.')

    names, nmeta = feature_names()
    if names and len(names) >= P_TOTAL:
        log(f'- feature names mapped from FEATURES_5s_v2 (day {nmeta}); first {P_TOTAL} non-(timestamp/price_mean/vwap) columns.')
    else:
        names = [f'feat_{i}' for i in range(P_TOTAL)]
        log(f'- WARNING: feature names unavailable ({nmeta}); using indices.')
    log()

    # ── pass 1: real selection ─────────────────────────────────────────────
    feat_hits = np.zeros(P_TOTAL, dtype=np.int64)      # segments where feature appears in a surviving term
    feat_lin_hits = np.zeros(P_TOTAL, dtype=np.int64)  # via a LINEAR term
    lin_pos = np.zeros(P_TOTAL, dtype=np.int64)        # linear beta > 0 count
    lin_tot = np.zeros(P_TOTAL, dtype=np.int64)
    n_terms, n_lin, n_quad, n_oob = 0, 0, 0, 0
    surv_counts, plocal_counts = [], []
    by_tier_hits = {}                                  # vol_tier -> (count_segs, feat_hits array)

    for s in valid:
        active = s['active_grid_cells']
        terms  = s['surviving_polynomial_indices']
        betas  = s['beta_coefficients']
        if isinstance(active, (int, float)): active = [int(active)]
        if isinstance(terms, (int, float)):  terms = [int(terms)]
        if isinstance(betas, (int, float)):  betas = [float(betas)]
        surv_counts.append(len(terms)); plocal_counts.append(len(active))
        seg_feats = set()
        tier = s.get('volatility_tier', 'NA')
        if tier not in by_tier_hits: by_tier_hits[tier] = [0, np.zeros(P_TOTAL, dtype=np.int64)]
        by_tier_hits[tier][0] += 1
        for li, beta in zip(terms, betas):
            try: g = map_local_poly_to_global(active, li)
            except Exception: n_oob += 1; continue
            kind, a, b = decode_global(g)
            n_terms += 1
            if kind == 'lin':
                n_lin += 1; seg_feats.add(a)
                feat_lin_hits[a] += 1
                lin_tot[a] += 1; lin_pos[a] += (beta > 0)
            elif kind == 'quad':
                n_quad += 1; seg_feats.add(a); seg_feats.add(b)
            else:
                n_oob += 1
        for fi in seg_feats:
            feat_hits[fi] += 1
            by_tier_hits[tier][1][fi] += 1

    surv = np.array(surv_counts); pl = np.array(plocal_counts)
    log('## 1. Selection profile')
    log(f'- surviving terms/segment: median {int(np.median(surv))}, p90 {int(np.percentile(surv,90))}, max {surv.max()}')
    log(f'- active features/segment (p_local): median {int(np.median(pl))}, p90 {int(np.percentile(pl,90))}')
    log(f'- term kinds: linear {n_lin:,} ({100*n_lin/max(n_terms,1):.1f}%), quadratic {n_quad:,} ({100*n_quad/max(n_terms,1):.1f}%), oob {n_oob}')
    log(f'  (linear-dominant => individual indicators explain price; quad-dominant => joint/interaction structure)')
    log()

    # ── concentration ───────────────────────────────────────────────────────
    def gini(x):
        x = np.sort(x.astype(float)); n = len(x); cx = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cx) / cx[-1]) / n if cx[-1] > 0 else 0.0
    freq = feat_hits / N
    order = np.argsort(freq)[::-1]
    g_real = gini(feat_hits)
    top10 = 100 * feat_hits[order[:10]].sum() / feat_hits.sum()
    top20 = 100 * feat_hits[order[:20]].sum() / feat_hits.sum()
    never = int((feat_hits == 0).sum())
    log('## 2. Concentration (is selection structured or uniform?)')
    log(f'- Gini of per-feature selection frequency: {g_real:.3f}  (0 = uniform/no structure, 1 = all in one feature)')
    log(f'- top-10 features account for {top10:.1f}% of all selections; top-20 -> {top20:.1f}%')
    log(f'- features NEVER selected: {never}/{P_TOTAL}')
    log()

    # ── NULL: uniform re-draw within each segment's own poly space ──────────
    NREP = 5
    null_ginis = []
    for _ in range(NREP):
        nf = np.zeros(P_TOTAL, dtype=np.int64)
        for s, sc, plc in zip(valid, surv_counts, plocal_counts):
            active = s['active_grid_cells']
            if isinstance(active, (int, float)): active = [int(active)]
            T = local_poly_count(len(active))
            if T == 0 or sc == 0: continue
            picks = RNG.choice(T, size=min(sc, T), replace=False)
            seg_feats = set()
            for li in picks:
                try: g = map_local_poly_to_global(active, int(li))
                except Exception: continue
                kind, a, b = decode_global(g)
                if kind == 'lin': seg_feats.add(a)
                elif kind == 'quad': seg_feats.add(a); seg_feats.add(b)
            for fi in seg_feats: nf[fi] += 1
        null_ginis.append(gini(nf))
    g_null = float(np.mean(null_ginis))
    log('## 3. Null model (uniform selection from each segment\'s own poly space)')
    log(f'- null Gini (mean of {NREP}): {g_null:.3f}  +/- {np.std(null_ginis):.3f}')
    log(f'- REAL Gini {g_real:.3f} vs NULL {g_null:.3f}  ->  ' +
        ('MEASURABLE PATTERN (real concentration exceeds chance)' if g_real > g_null + 0.05
         else 'NO PATTERN beyond chance (selection ~ uniform)'))
    log(f'  Note: even the null is concentrated if active_grid_cells (stage-1 availability) is itself non-uniform;')
    log(f'  the REAL>NULL gap isolates the elastic-net\'s preference ON TOP of stage-1 availability.')
    log()

    # ── top features (the payoff) ───────────────────────────────────────────
    log('## 4. Indicators worth seeing (top 25 by selection frequency)')
    log('| rank | feature | selected in % of segments | via-linear % | beta sign stability |')
    log('|---|---|---|---|---|')
    for r, fi in enumerate(order[:25], 1):
        sel = 100 * feat_hits[fi] / N
        vlin = 100 * feat_lin_hits[fi] / max(feat_hits[fi], 1)
        if lin_tot[fi] >= 20:
            p = lin_pos[fi] / lin_tot[fi]
            sign = f'{max(p,1-p):.2f} {"(+)" if p>0.5 else "(-)"}'
        else:
            sign = 'n/a (<20 lin)'
        log(f'| {r} | {names[fi]} | {sel:.1f}% | {vlin:.0f}% | {sign} |')
    log()

    # ── volatility-tier stratification ──────────────────────────────────────
    log('## 5. Stratify by volatility_tier (is structure within-vol, or just re-deriving vol regimes?)')
    log('Top-10 feature overlap between vol tiers (Jaccard); if ~1.0 the selected set is vol-invariant (real structure), if low it tracks vol.')
    tiers = sorted([t for t, v in by_tier_hits.items() if v[0] >= 200], key=lambda t: -by_tier_hits[t][0])
    tops = {}
    for t in tiers:
        cnt, arr = by_tier_hits[t]
        tops[t] = set(np.argsort(arr)[::-1][:10].tolist())
        log(f'- vol_tier {t}: {cnt:,} segs; top-5 = {[names[i] for i in np.argsort(arr)[::-1][:5]]}')
    if len(tiers) >= 2:
        import itertools
        for a, b in itertools.combinations(tiers, 2):
            j = len(tops[a] & tops[b]) / len(tops[a] | tops[b])
            log(f'- Jaccard(top10 vol {a}, vol {b}) = {j:.2f}')
    log()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f'\nReport -> {OUT}')

if __name__ == '__main__':
    main()
