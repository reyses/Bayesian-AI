"""
Peak Bucket Lifecycle — track how trades migrate through peak buckets
over their lifetime + cluster peak signatures within each bucket.

Peak buckets (ticks; TV = $0.50 / tick):
    NOISE     0-4    ($0-2)      bar-oscillation noise
    FAKE      5-9    ($2.50-4.50) small move, no conviction
    MARGINAL  10-19  ($5-9.50)   real but small (could be chop)
    REAL      20-39  ($10-19.50) significant directional move
    STRONG    40-79  ($20-39.50) major move, thesis worked
    DOMINANT  80+    ($40+)      captured full range

Two outputs in one report:
  Part 1 — Bar-by-bar bucket heatmap. At each bar N, what % of still-open
           trades sit in each peak bucket? Split by winners/losers to see
           when the trajectories diverge.
  Part 2 — Per-bucket peak clustering. For each of REAL / STRONG /
           DOMINANT populations (winners only), cluster the 91D peak-state
           feature vectors. Each cluster = a distinct exit signature.
           Propose top-3 feature rule per cluster.

Usage:
    python tools/peak_bucket_lifecycle.py --tier KILL_SHOT
    python tools/peak_bucket_lifecycle.py --tier KILL_SHOT_INVERSE \
        --trades training_iso/output/trades/iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl

Output:
    reports/findings/peak_lifecycle_{TIER}.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES


TRADES_DIR = 'training_iso/output/trades'
OUT_DIR = 'reports/findings'

TICK = 0.25
TV = 0.50

PEAK_BUCKETS = [
    ('NOISE',    0,   5),
    ('FAKE',     5,   10),
    ('MARGINAL', 10,  20),
    ('REAL',     20,  40),
    ('STRONG',   40,  80),
    ('DOMINANT', 80,  10000),
]

# Bar checkpoints for the lifecycle heatmap
BARS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45]

# Only cluster peaks in these buckets (losers dominate below)
CLUSTER_BUCKETS = ['REAL', 'STRONG', 'DOMINANT']

# Clustering config
MIN_CLUSTER_N = 15
K_RANGE = (2, 5)
TOP_FEATURES = 5
PCA_VAR_TARGET = 0.85


def classify(peak_ticks):
    for name, lo, hi in PEAK_BUCKETS:
        if lo <= peak_ticks < hi:
            return name
    return 'UNKNOWN'


def resolve_pickle(tier, explicit):
    if explicit:
        return explicit
    for cand in (
        os.path.join(TRADES_DIR, f'iso_is_{tier}.pkl'),
        os.path.join(TRADES_DIR, 'iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl'),
        os.path.join(TRADES_DIR, 'iso_is.pkl'),
    ):
        if os.path.exists(cand):
            return cand
    return cand


def running_peak_at_bar(trade, bar_n):
    """Max pnl (in dollars) seen at bar <= bar_n, or None if trade closed before bar_n."""
    path = trade.get('path', [])
    if not path:
        return None
    if trade.get('held', 0) < bar_n:
        return None
    best = 0.0
    for p in path:
        if p.get('bar', 0) <= bar_n:
            best = max(best, p.get('pnl', 0.0))
    return best


def peak_bar_of(trade):
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


def peak_feat_of(trade):
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
# Part 1: Lifecycle heatmap
# ═══════════════════════════════════════════════════════════════════════

def lifecycle_heatmap(trades):
    """Returns dict: {(cohort, bar_N): {bucket_name: count, 'total': N}}"""
    out = {}
    for cohort_name, cohort_trades in [
        ('winners', [t for t in trades if t.get('pnl', 0) > 0]),
        ('losers',  [t for t in trades if t.get('pnl', 0) < 0]),
    ]:
        for bar_n in BARS:
            counts = Counter()
            total = 0
            for t in cohort_trades:
                peak_d = running_peak_at_bar(t, bar_n)
                if peak_d is None:
                    continue
                peak_ticks = peak_d / TV
                counts[classify(peak_ticks)] += 1
                total += 1
            out[(cohort_name, bar_n)] = {'counts': counts, 'total': total}
    return out


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Per-bucket clustering
# ═══════════════════════════════════════════════════════════════════════

def select_k_by_bic(X_pc, k_range=K_RANGE, seeds=3):
    best_k, best_bic, best_gmm = None, np.inf, None
    bic_scores = {}
    for k in range(k_range[0], k_range[1]):
        bics, gmms = [], []
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
        avg = float(np.mean(bics))
        bic_scores[k] = avg
        if avg < best_bic:
            best_bic = avg
            best_k = k
            idx = int(np.argmin(bics))
            best_gmm = gmms[idx]
    return best_k, best_gmm, bic_scores


def cluster_peaks(peak_matrix):
    if len(peak_matrix) < MIN_CLUSTER_N * 2:
        return None
    scaler = StandardScaler()
    X_std = scaler.fit_transform(peak_matrix)
    pca_full = PCA()
    pca_full.fit(X_std)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_pc = int(np.searchsorted(cumvar, PCA_VAR_TARGET)) + 1
    n_pc = max(3, min(n_pc, 10))
    pca = PCA(n_components=n_pc)
    X_pc = pca.fit_transform(X_std)
    best_k, best_gmm, bic_scores = select_k_by_bic(X_pc)
    if best_gmm is None:
        return None
    return {
        'pca_n': n_pc, 'cumvar': float(cumvar[n_pc - 1]),
        'best_k': best_k, 'labels': best_gmm.predict(X_pc),
        'bic_scores': bic_scores,
    }


def cluster_distinctiveness(peak_matrix, mask, global_mean, global_std):
    cdata = peak_matrix[mask]
    cmean = cdata.mean(axis=0)
    out = []
    for i, name in enumerate(FEATURE_NAMES[:91]):
        if global_std[i] == 0:
            continue
        distinct = abs(cmean[i] - global_mean[i]) / global_std[i]
        out.append({
            'feature': name,
            'cluster_mean': float(cmean[i]),
            'global_mean': float(global_mean[i]),
            'distinct': float(distinct),
            'side': 'HIGH' if cmean[i] > global_mean[i] else 'LOW',
        })
    out.sort(key=lambda x: x['distinct'], reverse=True)
    return out


def cluster_bucket(winners_in_bucket, bucket_name):
    """Cluster peak features for winners in one bucket."""
    peak_mat = []
    for t in winners_in_bucket:
        pf = peak_feat_of(t)
        if pf is not None:
            peak_mat.append(pf)
    peak_mat = np.array(peak_mat, dtype=float)
    n = len(peak_mat)
    if n < MIN_CLUSTER_N * 2:
        return {'n': n, 'status': 'insufficient_samples',
                'n_with_features': n, 'n_total': len(winners_in_bucket)}
    result = cluster_peaks(peak_mat)
    if result is None:
        return {'n': n, 'status': 'clustering_failed'}
    labels = result['labels']
    gmean = peak_mat.mean(axis=0)
    gstd = peak_mat.std(axis=0, ddof=1)
    clusters = []
    for k in range(result['best_k']):
        mask = (labels == k)
        count = int(mask.sum())
        if count < MIN_CLUSTER_N:
            clusters.append({'k': k, 'n': count, 'skip': True})
            continue
        top = cluster_distinctiveness(peak_mat, mask, gmean, gstd)[:TOP_FEATURES]
        clusters.append({'k': k, 'n': count, 'top_features': top, 'skip': False})
    return {'n': n, 'status': 'ok', 'best_k': result['best_k'],
            'pca_n': result['pca_n'], 'cumvar': result['cumvar'],
            'bic_scores': result['bic_scores'], 'clusters': clusters}


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════

def write_report(tier, heatmap, bucket_cluster_results,
                 n_total, n_winners, n_losers, out_path):
    L = []
    L.append(f'# Peak Bucket Lifecycle — {tier}')
    L.append('')
    L.append(f'**{n_total} trades** ({n_winners} winners / {n_losers} losers)')
    L.append('')

    # ── Part 1: Lifecycle heatmap ──────────────────────────────────
    L.append('## Part 1 — Bar-by-bar peak bucket heatmap')
    L.append('')
    L.append('At each bar N of the trade life, what % of still-open trades sit '
             'in each peak bucket? Row = cohort × bar. Cell = % of trades in '
             'that bucket at that bar.')
    L.append('')
    L.append('Buckets (ticks): NOISE 0-4 · FAKE 5-9 · MARGINAL 10-19 · '
             'REAL 20-39 · STRONG 40-79 · DOMINANT 80+')
    L.append('')
    for cohort in ('winners', 'losers'):
        L.append(f'### {cohort.upper()}')
        L.append('')
        hdr_cells = ['bar', 'n_open'] + [n for n, _, _ in PEAK_BUCKETS]
        L.append('| ' + ' | '.join(hdr_cells) + ' |')
        L.append('|' + '|'.join(['---:'] * len(hdr_cells)) + '|')
        for bar_n in BARS:
            entry = heatmap[(cohort, bar_n)]
            total = entry['total']
            if total == 0:
                continue
            cells = [str(bar_n), str(total)]
            for name, _, _ in PEAK_BUCKETS:
                pct = entry['counts'][name] / total * 100
                cells.append(f'{pct:.0f}%')
            L.append('| ' + ' | '.join(cells) + ' |')
        L.append('')

    # ── Part 2: Per-bucket clustering ──────────────────────────────
    L.append('## Part 2 — Per-bucket peak signature clustering')
    L.append('')
    L.append('Clusters the 91D feature vector AT the peak bar for WINNERS in '
             'each bucket. Each cluster = distinct exit signature. Top 3 '
             'distinctive features per cluster -> candidate exit rule.')
    L.append('')
    for bucket_name in CLUSTER_BUCKETS:
        res = bucket_cluster_results.get(bucket_name)
        L.append(f'### Bucket: {bucket_name}')
        L.append('')
        if res is None:
            L.append('_No trades in this bucket._')
            L.append('')
            continue
        L.append(f'- Winners in bucket: {res["n"]}')
        if res['status'] == 'insufficient_samples':
            L.append(f'- **Clustering skipped** — need >= {MIN_CLUSTER_N * 2} '
                     f'samples with path features. Have {res["n_with_features"]}.')
            L.append('  (If features are 0, the pickle has path features '
                     'stripped — rerun single-tier for features-intact data.)')
            L.append('')
            continue
        if res['status'] != 'ok':
            L.append(f'- Clustering failed: {res["status"]}')
            L.append('')
            continue
        L.append(f'- PCA: {res["pca_n"]} components ({res["cumvar"]:.0%} var)')
        best_k = res['best_k']
        bics = res['bic_scores']
        bic_str = ', '.join(f'K{k}={v:.0f}' for k, v in sorted(bics.items()))
        L.append(f'- BIC: {bic_str}  ->  **K={best_k}** selected')
        L.append('')
        for c in res['clusters']:
            if c.get('skip'):
                L.append(f'#### Cluster {c["k"]}  —  N={c["n"]} (skipped, too small)')
                L.append('')
                continue
            L.append(f'#### Cluster {c["k"]}  —  N={c["n"]}')
            L.append('')
            L.append('| rank | feature | cluster mean | global mean | |Δ/σ| | side |')
            L.append('|---:|---|---:|---:|---:|---|')
            for i, f in enumerate(c['top_features'], 1):
                L.append(f'| {i} | {f["feature"]} | {f["cluster_mean"]:.3f} | '
                         f'{f["global_mean"]:.3f} | {f["distinct"]:.2f} | '
                         f'{f["side"]} |')
            L.append('')
            top3 = c['top_features'][:3]
            if len(top3) >= 3 and all(t['distinct'] >= 0.5 for t in top3):
                parts = []
                for t in top3:
                    op = '>' if t['side'] == 'HIGH' else '<'
                    thr = t['cluster_mean']
                    thr_str = f'{thr:.2f}' if abs(thr) >= 1 else f'{thr:.3f}'
                    parts.append(f"{t['feature']} {op} {thr_str}")
                rule = ' AND '.join(parts)
                L.append('**Candidate exit rule:**')
                L.append('```')
                L.append(f'if {rule}:')
                L.append(f"    return '{bucket_name.lower()}_cluster_{c['k']}'")
                L.append('```')
            else:
                L.append('_Signal too weak for rule (top features |Δ/σ| < 0.5)._')
            L.append('')

    L.append('---')
    L.append(f'_Generated by `tools/peak_bucket_lifecycle.py --tier {tier}`_')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--trades', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pkl = resolve_pickle(args.tier, args.trades)
    print(f'Loading {pkl}...')
    with open(pkl, 'rb') as f:
        trades = pickle.load(f)
    sub = [t for t in trades if t.get('entry_tier') == args.tier]
    if not sub:
        print(f'No trades for tier {args.tier}')
        return
    winners = [t for t in sub if t.get('pnl', 0) > 0]
    losers = [t for t in sub if t.get('pnl', 0) < 0]
    print(f'  {args.tier}: {len(sub)} ({len(winners)}W / {len(losers)}L)')

    print('Part 1: lifecycle heatmap...')
    heatmap = lifecycle_heatmap(sub)

    # Bucket winners by FINAL peak bucket, then cluster within each
    winners_by_bucket = {name: [] for name, _, _ in PEAK_BUCKETS}
    for t in winners:
        peak_ticks = t.get('peak', 0.0) / TV
        winners_by_bucket[classify(peak_ticks)].append(t)

    print('Part 2: per-bucket clustering...')
    cluster_results = {}
    for bucket in CLUSTER_BUCKETS:
        pop = winners_by_bucket[bucket]
        print(f'  {bucket}: {len(pop)} winners', end='')
        if pop:
            res = cluster_bucket(pop, bucket)
            cluster_results[bucket] = res
            if res['status'] == 'ok':
                print(f'  -> K={res["best_k"]}')
            else:
                print(f'  -> {res["status"]}')
        else:
            print()
            cluster_results[bucket] = None

    out_path = args.out or os.path.join(OUT_DIR, f'peak_lifecycle_{args.tier}.md')
    write_report(args.tier, heatmap, cluster_results,
                 len(sub), len(winners), len(losers), out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
