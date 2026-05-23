"""
Peak Signature Clustering — discover MULTIPLE exit signatures for a tier.

Premise (user: 2026-04-19): a tier's winners may peak via DIFFERENT physics
mechanisms. One entry setup → multiple exit signatures. A single peak rule
(p_center + reversion + vr) captures one mechanism; others (exhaustion,
volume cascade, opposite wick, chaos collapse) need their own rules.

Methodology: cluster the ABSOLUTE peak-state feature vectors. Each cluster
centroid IS a candidate exit rule — "when market state looks like THIS,
exit." Skip the entry→peak delta step (delta captures mechanism; we need
state for runtime exit check).

Units: peaks measured in TICKS (1 tick = $TV = $0.50 for MNQ). Dollar
thresholds are shown secondary — ticks are the instrument-native unit.

Peak-tier filtering (2026-04-19): cluster only on REAL+ peaks
(peak_ticks >= 20, = $10+). NOISE/FAKE peaks are bar-oscillation or
small chop — polluting the cluster population with them would wash out
real mechanisms.

Pipeline:
  1. Load trade pickle + filter tier + keep winners with peak_ticks >= 20
  2. Extract peak_feat (91D) at argmax(pnl) bar for each winner
  3. StandardScaler — z-score each feature across trades
  4. PCA to K_PC components (captures >80% variance typically)
  5. GMM with BIC search over K=2-5 (or HDBSCAN with min_cluster_size)
  6. Per-cluster: inverse-transform centroid, rank features by
     |centroid - global_mean| / global_std (distinctiveness)
  7. Propose exit rule per cluster: top 3 distinctive features +
     threshold inferred from cluster centroid value

Output: reports/findings/peak_signatures_{TIER}.md

Usage:
    python tools/peak_signature_cluster.py --tier KILL_SHOT
    python tools/peak_signature_cluster.py --tier KILL_SHOT_INVERSE \
        --trades training_iso/output/trades/iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl
    python tools/peak_signature_cluster.py --tier RIDE_AGAINST --min-peak-ticks 30
"""
import os
import sys
import pickle
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES


TRADES_DIR = 'training_iso/output/trades'
OUT_DIR = 'reports/findings'

# Instrument constants (MNQ). TV = value per tick in dollars.
TICK = 0.25
TV = 0.50

# Peak classification by tick magnitude.
PEAK_TIERS = [
    ('NOISE',    0,   5),     # 0-4 ticks ($0-2)
    ('FAKE',     5,   10),    # 5-9 ticks ($2.50-4.50)
    ('MARGINAL', 10,  20),    # 10-19 ticks ($5-9.50)
    ('REAL',     20,  40),    # 20-39 ticks ($10-19.50)
    ('STRONG',   40,  80),    # 40-79 ticks ($20-39.50)
    ('DOMINANT', 80,  10000), # 80+ ticks ($40+)
]

# Default cluster-population filter: REAL+ peaks only.
DEFAULT_MIN_PEAK_TICKS = 20

# Clustering config
PCA_VARIANCE_TARGET = 0.85    # target cumulative variance
K_RANGE = (2, 6)              # GMM component search range (inclusive-exclusive)
MIN_CLUSTER_N = 15            # a cluster must have >= N samples to matter
TOP_FEATURES_PER_CLUSTER = 5  # features to report per cluster


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def resolve_pickle(tier: str, explicit: str | None) -> str:
    """Find the best pickle for a given tier."""
    if explicit:
        return explicit
    # Look for per-tier or combined pickles first
    candidates = [
        os.path.join(TRADES_DIR, f'iso_is_{tier}.pkl'),
        os.path.join(TRADES_DIR, f'iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl'),
        os.path.join(TRADES_DIR, 'iso_is.pkl'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


def classify_peak(peak_ticks: float) -> str:
    for name, lo, hi in PEAK_TIERS:
        if lo <= peak_ticks < hi:
            return name
    return 'UNKNOWN'


def peak_bar_of(trade: dict) -> int | None:
    path = trade.get('path', [])
    if not path:
        return None
    best_bar, best_pnl = 0, path[0].get('pnl', 0.0)
    for p in path:
        pnl = p.get('pnl', 0.0)
        if pnl > best_pnl:
            best_pnl = pnl
            best_bar = p.get('bar', 0)
    return best_bar


def peak_feat_of(trade: dict):
    """Extract the 91D feature vector at the peak bar."""
    pb = peak_bar_of(trade)
    if pb is None:
        return None
    for p in trade.get('path', []):
        if p.get('bar') == pb:
            feat = p.get('features')
            if feat is not None and len(feat) >= 91:
                return np.asarray(feat[:91], dtype=float)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Peak-tier tally (before clustering, to show population health)
# ═══════════════════════════════════════════════════════════════════════

def peak_tier_tally(trades):
    tallies = Counter()
    stats_by_tier = {name: {'count': 0, 'total_pnl': 0.0, 'winners': 0}
                     for name, _, _ in PEAK_TIERS}
    for t in trades:
        peak_dollars = t.get('peak', 0.0)
        peak_ticks = peak_dollars / TV
        tier = classify_peak(peak_ticks)
        tallies[tier] += 1
        stats_by_tier[tier]['count'] += 1
        stats_by_tier[tier]['total_pnl'] += t.get('pnl', 0.0)
        if t.get('pnl', 0.0) > 0:
            stats_by_tier[tier]['winners'] += 1
    return stats_by_tier


# ═══════════════════════════════════════════════════════════════════════
# Clustering
# ═══════════════════════════════════════════════════════════════════════

def select_k_by_bic(X_pc: np.ndarray, k_range=K_RANGE, seeds=3) -> tuple:
    """Fit GMMs across K range, pick lowest-BIC. Returns (best_K, best_gmm)."""
    best_k = None
    best_bic = np.inf
    best_gmm = None
    bic_scores = {}
    for k in range(k_range[0], k_range[1]):
        # Average BIC across seeds for robustness (small-sample GMM can vary)
        bics = []
        gmms = []
        for seed in range(seeds):
            try:
                g = GaussianMixture(n_components=k, covariance_type='full',
                                    random_state=seed, max_iter=300,
                                    reg_covar=1e-4)
                g.fit(X_pc)
                bics.append(g.bic(X_pc))
                gmms.append(g)
            except Exception:
                continue
        if not bics:
            continue
        avg_bic = float(np.mean(bics))
        bic_scores[k] = avg_bic
        if avg_bic < best_bic:
            best_bic = avg_bic
            best_k = k
            # pick the seed with the best fit
            idx = int(np.argmin(bics))
            best_gmm = gmms[idx]
    return best_k, best_gmm, bic_scores


def cluster_peaks(peak_matrix: np.ndarray):
    """Full clustering pipeline."""
    if len(peak_matrix) < MIN_CLUSTER_N * 2:
        return None
    scaler = StandardScaler()
    X_std = scaler.fit_transform(peak_matrix)
    # Dynamic PCA: pick n_components that hits target variance
    pca_full = PCA()
    pca_full.fit(X_std)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_pc = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET)) + 1
    n_pc = max(3, min(n_pc, 10))
    pca = PCA(n_components=n_pc)
    X_pc = pca.fit_transform(X_std)
    best_k, best_gmm, bic_scores = select_k_by_bic(X_pc)
    if best_gmm is None:
        return None
    labels = best_gmm.predict(X_pc)
    return {
        'scaler': scaler,
        'pca': pca,
        'n_pc': n_pc,
        'cumvar_at_n_pc': float(cumvar[n_pc - 1]),
        'gmm': best_gmm,
        'labels': labels,
        'best_k': best_k,
        'bic_scores': bic_scores,
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-cluster rule synthesis
# ═══════════════════════════════════════════════════════════════════════

def cluster_signature(peak_matrix: np.ndarray, cluster_mask: np.ndarray,
                      global_mean: np.ndarray, global_std: np.ndarray):
    """For a cluster, rank features by distinctiveness + synthesize rule.

    distinctiveness[i] = |cluster_mean[i] - global_mean[i]| / global_std[i]

    Returns list of (feature_name, cluster_mean, global_mean, distinctiveness)
    sorted by distinctiveness desc.
    """
    cluster_data = peak_matrix[cluster_mask]
    cluster_mean = cluster_data.mean(axis=0)
    cluster_std = cluster_data.std(axis=0, ddof=1) if len(cluster_data) > 1 else np.ones_like(cluster_mean)
    out = []
    for i in range(len(FEATURE_NAMES[:91])):
        if global_std[i] == 0:
            continue
        distinct = abs(cluster_mean[i] - global_mean[i]) / global_std[i]
        direction = 'HIGH' if cluster_mean[i] > global_mean[i] else 'LOW'
        out.append({
            'feature': FEATURE_NAMES[i],
            'cluster_mean': float(cluster_mean[i]),
            'cluster_std': float(cluster_std[i]),
            'global_mean': float(global_mean[i]),
            'global_std': float(global_std[i]),
            'distinctiveness': float(distinct),
            'direction': direction,
        })
    out.sort(key=lambda x: x['distinctiveness'], reverse=True)
    return out


def synthesize_rule(top_features, direction_side_1, direction_side_2, direction_side_3):
    """Build a human-readable exit rule from top-3 distinctive features."""
    parts = []
    for feat_info, direction in zip(top_features[:3],
                                    [direction_side_1, direction_side_2, direction_side_3]):
        op = '>' if direction == 'HIGH' else '<'
        thr = feat_info['cluster_mean']
        # Round threshold by feature-value magnitude
        if abs(thr) >= 1:
            thr_str = f'{thr:.2f}'
        else:
            thr_str = f'{thr:.3f}'
        parts.append(f"{feat_info['feature']} {op} {thr_str}")
    return ' AND '.join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Report writer
# ═══════════════════════════════════════════════════════════════════════

def write_report(tier, peak_tally, clustering_result, cluster_signatures,
                 n_filtered, n_total, min_peak_ticks, out_path):
    L = []
    L.append(f'# Peak Signature Clusters — {tier}')
    L.append('')
    L.append('## Population summary')
    L.append('')
    L.append(f'- Total winners: {n_total}')
    L.append(f'- Filter: `peak_ticks >= {min_peak_ticks}` (REAL+ peaks only)')
    L.append(f'- Clustered population: **{n_filtered}** '
             f'({100 * n_filtered / max(n_total, 1):.0f}% of winners)')
    L.append('')

    L.append('## Peak-tier distribution (winners + losers)')
    L.append('')
    L.append('| Tier | Ticks | Dollars | N | WR % | Total PnL |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for name, lo, hi in PEAK_TIERS:
        stats = peak_tally[name]
        if stats['count'] == 0:
            continue
        wr = stats['winners'] / stats['count'] * 100
        dollar_lo = lo * TV
        dollar_hi = hi * TV if hi < 10000 else float('inf')
        dollar_range = f'${dollar_lo:.1f}-${dollar_hi:.1f}' if dollar_hi != float('inf') else f'${dollar_lo:.1f}+'
        tick_range = f'{lo}-{hi-1}' if hi < 10000 else f'{lo}+'
        L.append(f'| {name} | {tick_range} | {dollar_range} | '
                 f'{stats["count"]:,} | {wr:.0f}% | ${stats["total_pnl"]:+,.0f} |')
    L.append('')

    if clustering_result is None:
        L.append(f'## Clustering skipped')
        L.append('')
        L.append(f'Need at least {MIN_CLUSTER_N * 2} winners with peak >= '
                 f'{min_peak_ticks} ticks. Only {n_filtered} available.')
        L.append('')
        L.append('**Recommendation:** either lower `--min-peak-ticks`, or '
                 'run the tier with more days / more samples before clustering.')
        _finalize(L, out_path, tier)
        return

    best_k = clustering_result['best_k']
    cumvar = clustering_result['cumvar_at_n_pc']
    n_pc = clustering_result['n_pc']
    bic_scores = clustering_result['bic_scores']

    L.append('## Clustering diagnostics')
    L.append('')
    L.append(f'- PCA: **{n_pc} components** explain **{cumvar:.1%}** of variance')
    L.append(f'- GMM component search: {K_RANGE[0]}-{K_RANGE[1]-1}')
    L.append('- BIC scores (lower = better):')
    for k in sorted(bic_scores.keys()):
        marker = ' ← selected' if k == best_k else ''
        L.append(f'  - K={k}: BIC={bic_scores[k]:.1f}{marker}')
    L.append(f'- **Selected K: {best_k}** — this is how many distinct peak '
             f'signatures the data supports.')
    L.append('')

    L.append('## Cluster signatures')
    L.append('')
    for cluster_info in cluster_signatures:
        k = cluster_info['k']
        n = cluster_info['n']
        peak_stats = cluster_info['peak_stats']
        time_stats = cluster_info['time_stats']
        L.append(f'### Cluster {k}  —  N={n}')
        L.append('')
        L.append(f'- Peak size (ticks): median **{peak_stats["median_ticks"]:.0f}** '
                 f'(${peak_stats["median_ticks"] * TV:+.1f}), '
                 f'mean {peak_stats["mean_ticks"]:.1f}, '
                 f'p75 {peak_stats["p75_ticks"]:.0f}, '
                 f'p95 {peak_stats["p95_ticks"]:.0f}')
        L.append(f'- Time-to-peak (bars): median **{time_stats["median"]:.0f}**, '
                 f'p75 {time_stats["p75"]:.0f}')
        L.append('')
        L.append('**Distinctive features at peak:**')
        L.append('')
        L.append('| rank | feature | cluster mean | global mean | |Δ/σ| | side |')
        L.append('|---:|---|---:|---:|---:|---|')
        for i, f in enumerate(cluster_info['top_features'], 1):
            L.append(f'| {i} | {f["feature"]} | {f["cluster_mean"]:.3f} | '
                     f'{f["global_mean"]:.3f} | {f["distinctiveness"]:.2f} | '
                     f'{f["direction"]} |')
        L.append('')
        top = cluster_info['top_features']
        if len(top) >= 3:
            rule = synthesize_rule(top, top[0]['direction'],
                                   top[1]['direction'], top[2]['direction'])
            L.append(f'**Proposed exit rule (top-3 distinctive, |Δ/σ| >= 0.5):**')
            L.append(f'```')
            L.append(f'if {rule}:')
            L.append(f"    return 'kill_shot_peak_cluster_{k}'")
            L.append(f'```')
        L.append('')

    L.append('## Interpretation guide')
    L.append('')
    L.append('- Clusters with peak size STRONG/DOMINANT (>= 40 ticks) are '
             'high-value signatures — their rules save the largest moves.')
    L.append('- Clusters with N < 20 are fragile (may not replicate OOS); '
             'skip shipping their rules until validated.')
    L.append('- A cluster where top-3 features overlap with an existing rule '
             '(p_center / reversion / vr) confirms our current peak rule; '
             'new top-3 combinations are new signatures to wire.')
    L.append('- |Δ/σ| >= 1 on a feature = strong signal. < 0.5 = weak, probably '
             'not rule-worthy.')
    L.append('')
    _finalize(L, out_path, tier)


def _finalize(lines, out_path, tier):
    lines.append('---')
    lines.append(f'_Generated by `tools/peak_signature_cluster.py --tier {tier}`_')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True, help='tier name to analyze')
    ap.add_argument('--trades', default=None, help='pickle path override')
    ap.add_argument('--out', default=None, help='output md path')
    ap.add_argument('--min-peak-ticks', type=int, default=DEFAULT_MIN_PEAK_TICKS,
                    help='min peak ticks for clustering population (default 20)')
    args = ap.parse_args()

    pkl_path = resolve_pickle(args.tier, args.trades)
    print(f'Loading {pkl_path}...')
    with open(pkl_path, 'rb') as f:
        trades = pickle.load(f)
    counts = Counter(t.get('entry_tier', '?') for t in trades)
    print(f'  Tiers in file: {dict(counts)}')

    sub = [t for t in trades if t.get('entry_tier') == args.tier]
    if not sub:
        print(f'ERROR: no trades for tier {args.tier!r}')
        return

    winners = [t for t in sub if t.get('pnl', 0) > 0]
    losers = [t for t in sub if t.get('pnl', 0) < 0]
    print(f'  {args.tier}: {len(sub):,} trades '
          f'({len(winners):,} W / {len(losers):,} L)')

    # Peak-tier tally (winners + losers — shows population health)
    peak_tally = peak_tier_tally(sub)
    print()
    print('Peak-tier distribution:')
    for name, lo, hi in PEAK_TIERS:
        st = peak_tally[name]
        if st['count'] == 0:
            continue
        wr = st['winners'] / st['count'] * 100
        print(f'  {name:>10} ({lo:>3}-{hi-1 if hi<10000 else "+":>3} ticks): '
              f'N={st["count"]:>4} WR={wr:>5.1f}% '
              f'PnL=${st["total_pnl"]:>+9,.0f}')

    # Filter winners to REAL+ peaks for clustering
    winners_real = [t for t in winners
                    if (t.get('peak', 0.0) / TV) >= args.min_peak_ticks]
    print()
    print(f'Clustering population: {len(winners_real)} winners with peak >= '
          f'{args.min_peak_ticks} ticks (${args.min_peak_ticks * TV:.1f})')

    # Extract peak features for each winner
    peak_matrix = []
    peak_pnls_keep = []
    times_keep = []
    for t in winners_real:
        pf = peak_feat_of(t)
        if pf is None:
            continue
        peak_matrix.append(pf)
        peak_pnls_keep.append(t.get('peak', 0.0))
        times_keep.append(peak_bar_of(t))
    peak_matrix = np.array(peak_matrix, dtype=float)
    peak_pnls_keep = np.array(peak_pnls_keep)
    times_keep = np.array(times_keep)
    print(f'  With path features intact: {len(peak_matrix)}')

    if len(peak_matrix) < MIN_CLUSTER_N * 2:
        print(f'  INSUFFICIENT SAMPLES for clustering (need >= '
              f'{MIN_CLUSTER_N * 2}). Writing population-only report.')
        out_path = args.out or os.path.join(OUT_DIR, f'peak_signatures_{args.tier}.md')
        write_report(args.tier, peak_tally, None, [], len(peak_matrix),
                     len(winners), args.min_peak_ticks, out_path)
        print(f'Wrote: {out_path}')
        return

    print()
    print('Clustering...')
    result = cluster_peaks(peak_matrix)
    best_k = result['best_k']
    labels = result['labels']
    print(f'  PCA: {result["n_pc"]} components '
          f'({result["cumvar_at_n_pc"]:.1%} var)')
    print(f'  GMM K={best_k} selected '
          f'(BIC: {", ".join(f"K{k}={v:.0f}" for k,v in sorted(result["bic_scores"].items()))})')

    global_mean = peak_matrix.mean(axis=0)
    global_std = peak_matrix.std(axis=0, ddof=1)

    cluster_signatures = []
    for k in range(best_k):
        mask = (labels == k)
        n = int(mask.sum())
        if n < MIN_CLUSTER_N:
            print(f'  Cluster {k}: N={n} — skipping (below MIN_CLUSTER_N)')
            continue
        ranked = cluster_signature(peak_matrix, mask, global_mean, global_std)
        top = ranked[:TOP_FEATURES_PER_CLUSTER]
        cluster_peaks_ticks = peak_pnls_keep[mask] / TV
        cluster_times = times_keep[mask]
        cluster_signatures.append({
            'k': k,
            'n': n,
            'top_features': top,
            'peak_stats': {
                'median_ticks': float(np.median(cluster_peaks_ticks)),
                'mean_ticks':   float(np.mean(cluster_peaks_ticks)),
                'p75_ticks':    float(np.percentile(cluster_peaks_ticks, 75)),
                'p95_ticks':    float(np.percentile(cluster_peaks_ticks, 95)),
            },
            'time_stats': {
                'median': float(np.median(cluster_times)),
                'p75':    float(np.percentile(cluster_times, 75)),
            },
        })
        print(f'  Cluster {k}: N={n} peak median={np.median(cluster_peaks_ticks):.0f}t '
              f'time p50={np.median(cluster_times):.0f} — '
              f'top: {top[0]["feature"]} ({top[0]["direction"]}, '
              f'|d/σ|={top[0]["distinctiveness"]:.2f})')

    out_path = args.out or os.path.join(OUT_DIR, f'peak_signatures_{args.tier}.md')
    write_report(args.tier, peak_tally, result, cluster_signatures,
                 len(peak_matrix), len(winners), args.min_peak_ticks, out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
