"""Standardized Pareto chart of effects — FULL YEAR (2026-06-12).

Re-run of archive/research/Regression segments/plot_minitab_pareto.py, which was a PARTIAL
run: it globbed artifacts/stage2_segments_*.json (partial per-period files) and missed the
consolidated artifacts/stage2_year_segments.json (the full 80,717-valid-segment set).

'Standardized effect' = mean absolute beta coefficient of each polynomial term across the
PRISTINE/RECOVERED segments where it survived (count > FILTER). This is the in-segment
DESCRIPTIVE importance (post-hoc / lookahead) -- which terms the fits lean on -- NOT a
forward-predictive ranking. Dumps the full ranked table to CSV so rankings are exact.
"""
import os, sys, json, glob, csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_v2.features import load_features

FILTER_COUNT = 50   # term must appear in > this many segments to be plotted/ranked

def feature_names():
    root = os.path.abspath('DATA/ATLAS/FEATURES_5s_v2')
    days = sorted(os.path.basename(f).replace('.parquet', '') for f in glob.glob(os.path.join(root, 'L3_1m', '*.parquet')))
    if not days:
        days = sorted(os.path.basename(f).replace('.parquet', '') for f in glob.glob(os.path.join(root, '*', '*.parquet')))
    df = load_features([days[0]], root=root)
    return [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]

def main():
    cols = feature_names(); P = len(cols)
    print(f"[pareto] {P} features mapped")
    segs = json.load(open('artifacts/stage2_year_segments.json'))
    valid = [s for s in segs if s.get('status') in ('PRISTINE', 'RECOVERED')]
    print(f"[pareto] {len(valid):,} valid segments (FULL year)")

    term_sum, term_cnt, triu = {}, {}, {}
    for seg in valid:
        active = seg.get('active_grid_cells', [])
        surv = seg.get('surviving_polynomial_indices', [])
        betas = seg.get('beta_coefficients', [])
        if isinstance(active, (int, float)): active = [int(active)]
        if isinstance(surv, (int, float)): surv = [int(surv)]
        if isinstance(betas, (int, float)): betas = [betas]
        p = len(active)
        if p == 0 or len(surv) == 0 or len(betas) != len(surv):
            continue
        if p not in triu:
            ii, jj = torch.triu_indices(p, p, offset=0)
            triu[p] = (ii.numpy(), jj.numpy())
        ii, jj = triu[p]
        for idx, k in enumerate(surv):
            bv = abs(betas[idx])
            if k < p:
                name = cols[active[k]]
            else:
                off = k - p
                if off >= len(ii):
                    continue
                gi, gj = active[ii[off]], active[jj[off]]
                if gi > gj:
                    gi, gj = gj, gi
                name = f"{cols[gi]}*" if gi == gj else f"{cols[gi]} x {cols[gj]}"
            term_sum[name] = term_sum.get(name, 0.0) + bv
            term_cnt[name] = term_cnt.get(name, 0) + 1

    avg = {k: term_sum[k] / term_cnt[k] for k in term_sum}
    valid_eff = {k: v for k, v in avg.items() if term_cnt[k] > FILTER_COUNT}
    rows = sorted(valid_eff.items(), key=lambda x: -x[1])

    with open('artifacts/pareto_effects_full.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'term', 'avg_abs_beta', 'count', 'total_abs_beta'])
        for r, (k, v) in enumerate(rows, 1):
            w.writerow([r, k, f"{v:.4f}", term_cnt[k], f"{term_sum[k]:.2f}"])
    print(f"[pareto] {len(rows)} terms (count>{FILTER_COUNT}) -> artifacts/pareto_effects_full.csv")

    top = rows[:30]
    terms = [t for t, _ in top]; effs = [e for _, e in top]
    vals = np.array(list(valid_eff.values()))
    sig = vals.mean() + 2 * vals.std()
    fig, ax = plt.subplots(figsize=(12, 10))
    norm = (np.array(effs) - min(effs)) / (max(effs) - min(effs) + 1e-9)
    ax.barh(range(len(terms)), effs, color=plt.get_cmap('coolwarm')(norm))
    ax.set_yticks(range(len(terms))); ax.set_yticklabels(terms, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(sig, color='red', linestyle='--', label='Pseudo-Significance (2 SD)')
    ax.set_title('Standardized Pareto Chart of Effects (FULL year)', fontsize=15)
    ax.set_xlabel('Average Absolute Beta Coefficient (Standardized Effect)')
    ax.legend(); plt.tight_layout()
    plt.savefig('artifacts/minitab_pareto_effect.png', dpi=150)
    print("[pareto] chart -> artifacts/minitab_pareto_effect.png")

    print("\nTOP 15 EFFECTS (full year):")
    for r, (k, v) in enumerate(rows[:15], 1):
        print(f"  {r:2d}. {k:58s} avg|b|={v:8.1f}  n={term_cnt[k]:,}")
    print(f"\nsig threshold (2SD) = {sig:.1f}; terms above it = {int((vals > sig).sum())}/{len(vals)}")
    nlin = sum(1 for k in valid_eff if ' x ' not in k and not k.endswith('*'))
    print(f"linear terms in ranked set: {nlin}/{len(valid_eff)} (rest are interactions)")

if __name__ == '__main__':
    main()
