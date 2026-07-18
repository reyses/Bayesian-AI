"""Hierarchical Standardized Main-Effects Plots (SMEP) over Regression segments.

L1: global standardized main effect (avg |beta|) + frequency of each term across all
    fitted segments  -> the "common features".
L2: stratify by volatility_tier (regime axis) -> heatmap term x tier (how the dominant
    features shift from clean tier-1 to chaotic tier-9).
L3: stratify by tier-band x time-of-session (open/mid/close) -> heatmap term x cell.

FIREWALL: segment betas are NON-CAUSAL (in-sample fits over whole segments). This is a
DIAGNOSTIC of what structure the segmentation leans on — NOT a tradeable signal.

Schema note: betas are indexed against the PRE-L5 feature schema, so names are
reconstructed with layers=['L0','L1','L2','L3'] (NOT the L5-baked default).

Run: python research/smep_hierarchical_segments.py
"""
import os, sys, glob, json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.features import load_features

ARTIFACT_DIR = 'artifacts'   # repo-root: full 2025 + 2026-Q1 daily stage2 segments
OUTDIR = 'reports/findings/smep'
MIN_COUNT_L1 = 20          # a term must appear in >=N segments to be plotted (anti-outlier)
TOP_N = 25
RNG = np.random.RandomState(42)


def feature_cols():
    """Reconstruct the EXACT column set/order the betas were indexed against
    (pre-L5 schema, price_mean/vwap excluded — matches plot_minitab_pareto.py)."""
    df = load_features(['2025_02_05'], layers=['L0', 'L1', 'L2', 'L3'])
    return [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]


_TFS = ('15s', '5s', '1m', '5m', '15m', '1h', '4h', '1D')


def base_features(k, p, active, ti, tj, cols):
    """The base feature(s) a surviving polynomial term touches. Linear/square -> [f];
    interaction f x g -> [f, g]. (Right unit for 'common features': exact terms are
    too sparse — 94% are quadratic/overfit.)"""
    def nm(gi):
        return cols[gi] if gi < len(cols) else f"f{gi}"
    if k < p:
        return [nm(active[k])]
    off = k - p
    if off >= len(ti):
        return []
    gi, gj = active[ti[off]], active[tj[off]]
    return [nm(gi)] if gi == gj else [nm(gi), nm(gj)]


def family(name):
    """Roll a feature name up to its TF-invariant family, e.g. L3_5s_hurst_9 -> L3_hurst."""
    parts = name.split('_')
    parts = [p for p in parts if p not in _TFS and not p.isdigit()]
    return '_'.join(parts)


def tod_bucket(seg, day_max):
    """open / mid / close third of the session from start_idx."""
    frac = seg['start_idx'] / max(day_max, 1)
    return 'open' if frac < 1/3 else ('mid' if frac < 2/3 else 'close')


def tier_band(t):
    return 'low(1-3)' if t <= 3 else ('mid(4-6)' if t <= 6 else 'high(7-9)')


def collect():
    cols = feature_cols()
    files = sorted(glob.glob(f'{ARTIFACT_DIR}/stage2_segments_*.json'))
    rows = []
    for fp in files:
        segs = json.load(open(fp))
        day_max = max((s['raw_end_idx'] for s in segs), default=1)
        for seg in segs:
            if seg.get('status') not in ('PRISTINE', 'RECOVERED'):
                continue
            active = seg.get('active_grid_cells', [])
            surv = seg.get('surviving_polynomial_indices', [])
            betas = seg.get('beta_coefficients', [])
            if not isinstance(surv, list):
                surv = [surv]
            if not isinstance(betas, list):
                betas = [betas]
            p = len(active)
            if p == 0 or len(surv) == 0 or len(betas) != len(surv):
                continue
            ti, tj = np.triu_indices(p, k=0)
            tier = seg.get('volatility_tier', np.nan)
            tod = tod_bucket(seg, day_max)
            for b, k in zip(betas, surv):
                for feat in base_features(k, p, active, ti, tj, cols):   # one row per base feature touched
                    rows.append((feat, family(feat), abs(float(b)), int(tier),
                                 tier_band(tier), tod, seg['status'], seg['day']))
    return pd.DataFrame(rows, columns=['term', 'family', 'abeta', 'tier', 'band', 'tod', 'status', 'day'])


def l1(df):
    g = df.groupby('term')['abeta'].agg(['median', 'count'])   # median = robust to ill-conditioned fits
    g = g[g['count'] >= MIN_COUNT_L1].sort_values('median', ascending=False)
    top = g.head(TOP_N)
    fig, ax = plt.subplots(1, 2, figsize=(17, 9))
    ax[0].barh(top.index[::-1], top['median'][::-1], color='#4575b4')
    sig = g['median'].mean() + 2 * g['median'].std()
    ax[0].axvline(sig, color='red', ls='--', label=f'2σ sig ({sig:.2f})')
    ax[0].set_title('L1 main effect (median |β|, robust)'); ax[0].legend(fontsize=8)
    freq = g.sort_values('count', ascending=False).head(TOP_N)
    ax[1].barh(freq.index[::-1], freq['count'][::-1], color='#91bfdb')
    ax[1].set_title('L1 commonality (segment frequency)')
    fig.suptitle('SMEP Level 1 — common features across all fitted segments (DIAGNOSTIC, non-causal)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{OUTDIR}/smep_L1.png', dpi=130); plt.close(fig)
    return top, g


def heat(piv, title, fname, top_terms):
    piv = piv.reindex(top_terms).dropna(how='all')
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * piv.shape[1] + 4), 0.42 * len(piv) + 2))
    im = ax.imshow(piv.values, aspect='auto', cmap='RdYlBu_r')
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, label='avg |β|'); ax.set_title(title, fontweight='bold')
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = collect()
    if df.empty:
        print("no segment terms collected"); return
    top, g = l1(df)
    top_terms = list(top.index)

    # L2 — term x tier  (median = robust)
    p2 = df.pivot_table(index='term', columns='tier', values='abeta', aggfunc='median')
    heat(p2, 'SMEP Level 2 — main effect by volatility tier (clean→chaotic)', f'{OUTDIR}/smep_L2_tier.png', top_terms)
    # L3 — term x (band x tod)
    df['cell'] = df['band'] + ' | ' + df['tod']
    p3 = df.pivot_table(index='term', columns='cell', values='abeta', aggfunc='median')
    heat(p3, 'SMEP Level 3 — main effect by tier-band × session phase', f'{OUTDIR}/smep_L3_band_tod.png', top_terms)

    # report
    n_terms = df['term'].nunique()
    L = ["# Hierarchical SMEP over Regression segments (DIAGNOSTIC — non-causal)\n",
         f"Source: {df['day'].nunique()} day(s) {sorted(df['day'].unique())}, "
         f"{len(df)} term-instances over fitted segments; {n_terms} distinct terms.",
         f"⚠️ Pre-roll-fix data (a few post-expiry days may be thin); betas are non-causal "
         f"in-sample fits (FIREWALL — diagnostic of structure only, not a live signal).\n",
         "## L1 — common features (top by standardized main effect, count≥%d):" % MIN_COUNT_L1]
    for term, r in top.head(15).iterrows():
        L.append(f"- {term}: median|β| {r['median']:.3f} (n={int(r['count'])})")
    fam = df.groupby('family')['abeta'].agg(['median', 'count']).sort_values('count', ascending=False)
    L += ["", "## L1 — common feature FAMILIES (by commonality / frequency, the robust axis):"]
    for f, r in fam[fam['count'] >= MIN_COUNT_L1].head(12).iterrows():
        L.append(f"- {f}: n={int(r['count'])} segments, median|β| {r['median']:.3f}")
    # which terms shift most across tiers (L2 signal)
    span = (p2.reindex(top_terms).max(axis=1) - p2.reindex(top_terms).min(axis=1)).sort_values(ascending=False)
    L += ["", "## L2 — terms whose effect shifts MOST across tiers (regime-dependent):"]
    for term, v in span.head(8).items():
        L.append(f"- {term}: Δ avg|β| across tiers = {v:.3f}")
    L += ["", "Plots: smep_L1.png, smep_L2_tier.png, smep_L3_band_tod.png",
          "FIREWALL: betas are in-sample fits — diagnostic of structure, NOT a live signal."]
    open(f'{OUTDIR}/smep_hierarchical.md', 'w', encoding='utf-8').write("\n".join(L))
    print("\n".join(L).encode('ascii', 'replace').decode())
    print(f"\n[plots + report -> {OUTDIR}/]")


if __name__ == '__main__':
    main()
