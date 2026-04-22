"""
Pareto loss concentration — find the X% of trades that account for Y% of
total losses, then identify which entry features discriminate them.

Research question: is the trading-system bleed 80/20 concentrated, and if
so, what feature(s) at entry time separate the bleed cohort from the rest?

Pipeline:
  1. Per dataset (IS / OOS) and per tier, sort trades by PnL ascending.
  2. Compute cumulative loss curve. Report key Pareto points:
       - % of trades that account for 50% / 70% / 80% / 90% / 95% of losses
  3. Define the "bleed cohort" as the worst --trade-pct (default 20%) of
     trades for each tier. These carry the target for any filter.
  4. For every entry feature (91 canonical features from core.features),
     compute Cohen d between bleed cohort and rest-of-trades.
  5. Rank features by |d|. Walk-forward stability = sign(d_IS) == sign(d_OOS)
     AND min(|d_IS|, |d_OOS|) >= 0.30.
  6. Output: per-tier Pareto table, top-10 discriminators per tier per
     dataset, and a shortlist of walk-forward-stable features.

Also computes a derived feature `1h_z_range = 1h_z_high - 1h_z_low` so the
original hypothesis can be compared to alternatives in the same ranking.

Usage:
    python tools/pareto_loss_concentration.py
    python tools/pareto_loss_concentration.py --trade-pct 15

Output: reports/findings/pareto_loss_concentration.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_NAMES


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH = 'reports/findings/pareto_loss_concentration.md'

MIN_TIER_N = 40
LOSS_MILESTONES = [50, 70, 80, 90, 95]


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_matrix(trades):
    """Return (N x 92) feature matrix, (N,) pnl, (N,) tier_labels.

    Feature order: FEATURE_NAMES[:91] + ['1h_z_range'] (derived).
    """
    names = list(FEATURE_NAMES[:91]) + ['1h_z_range']
    idx_hi = FEATURE_NAMES.index('1h_z_high')
    idx_lo = FEATURE_NAMES.index('1h_z_low')

    rows = []
    pnls = []
    tiers = []
    for t in trades:
        ef = t.get('entry_79d')
        if ef is None or len(ef) < 91:
            continue
        ef = np.asarray(ef, dtype=np.float64)[:91]
        z_range = float(ef[idx_hi] - ef[idx_lo])
        feat = np.concatenate([ef, [z_range]])
        rows.append(feat)
        pnls.append(float(t.get('pnl', 0.0)))
        tiers.append(t.get('entry_tier', '?'))
    if not rows:
        return None, None, None, names
    X = np.vstack(rows)
    y = np.asarray(pnls)
    tiers = np.asarray(tiers)
    return X, y, tiers, names


def pareto_points(pnls):
    """Given an array of PnLs, return (sorted_cum_loss_pct, trade_pct).

    Sorts ascending (worst first), computes cumulative absolute loss
    contribution as % of total absolute losses.
    """
    p = np.asarray(pnls)
    neg = p[p < 0]
    if neg.size == 0:
        return None, None, 0.0
    total_loss = -neg.sum()  # positive number
    order = np.argsort(p)
    sorted_p = p[order]
    # Cumulative loss as we walk from worst to best
    cum = np.cumsum(np.where(sorted_p < 0, -sorted_p, 0.0))
    cum_loss_pct = (cum / total_loss) * 100.0
    trade_pct = (np.arange(1, len(p) + 1) / len(p)) * 100.0
    return cum_loss_pct, trade_pct, total_loss


def pct_trades_for_loss(cum_loss_pct, trade_pct, loss_target):
    """Return % of trades needed to reach loss_target% of total losses."""
    if cum_loss_pct is None:
        return None
    idx = np.searchsorted(cum_loss_pct, loss_target, side='left')
    if idx >= len(trade_pct):
        return 100.0
    return float(trade_pct[idx])


def cohen_d(a, b):
    """Cohen d between two groups."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                     / max(len(a) + len(b) - 2, 1))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def rank_features(X, y, trade_pct, feature_names):
    """Return sorted list of (name, d, bleed_mean, rest_mean, n_bleed).

    Bleed cohort = worst trade_pct percent of trades by PnL.
    """
    n = len(y)
    if n < 10:
        return []
    n_bleed = max(1, int(round(n * trade_pct / 100.0)))
    order = np.argsort(y)
    bleed_idx = order[:n_bleed]
    rest_idx = order[n_bleed:]
    Xb = X[bleed_idx]
    Xr = X[rest_idx]
    results = []
    for i, name in enumerate(feature_names):
        a = Xb[:, i]
        b = Xr[:, i]
        d = cohen_d(a, b)
        if np.isnan(d):
            d = 0.0
        results.append({
            'name': name,
            'd': d,
            'abs_d': abs(d),
            'bleed_mean': float(a.mean()),
            'rest_mean': float(b.mean()),
            'n_bleed': n_bleed,
            'n_rest': len(rest_idx),
        })
    results.sort(key=lambda r: -r['abs_d'])
    return results


def analyze_tier(X, y, feature_names, trade_pct):
    """Return per-tier diagnostics: pareto points + feature ranking."""
    cum_loss, trades_pct, total_loss = pareto_points(y)
    milestones = {}
    for m in LOSS_MILESTONES:
        milestones[m] = pct_trades_for_loss(cum_loss, trades_pct, m)
    features = rank_features(X, y, trade_pct, feature_names)
    return {
        'n': len(y),
        'total_pnl': float(y.sum()),
        'total_loss': total_loss,
        'milestones': milestones,
        'features': features,
    }


def write_report(is_res, oos_res, out_path, trade_pct):
    L = []
    L.append('# Pareto loss concentration — where the bleed lives')
    L.append('')
    L.append(f'Bleed cohort = worst **{trade_pct}%** of trades by PnL '
             '(per tier, independent of overall position in the book).')
    L.append('')
    L.append('Feature separation = Cohen d between bleed cohort and '
             'rest-of-trades at entry time. |d| >= 0.30 is meaningful, '
             '|d| >= 0.50 is strong, |d| >= 0.80 is dominant.')
    L.append('')
    L.append('**Walk-forward stable** = sign of d matches on IS and OOS '
             'AND min(|d_IS|, |d_OOS|) >= 0.30.')
    L.append('')

    # Section 1: pareto concentration per tier
    L.append('## 1. Loss concentration per tier')
    L.append('')
    L.append('% of trades needed to reach each loss milestone. '
             'Lower % = more concentrated bleed = better target for '
             'a filter.')
    L.append('')
    L.append('| Dataset | Tier | N | Total $ | Total loss $ | '
             '% trades -> 50% loss | 70% | 80% | 90% | 95% |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for label, res in [('IS', is_res), ('OOS', oos_res)]:
        for tier in sorted(res.keys()):
            r = res[tier]
            if r['n'] < MIN_TIER_N:
                continue
            m = r['milestones']
            def fmt(x):
                return '—' if x is None else f'{x:.1f}%'
            L.append(f'| {label} | {tier} | {r["n"]:,} | '
                     f'${r["total_pnl"]:+,.0f} | ${r["total_loss"]:,.0f} | '
                     f'{fmt(m[50])} | {fmt(m[70])} | {fmt(m[80])} | '
                     f'{fmt(m[90])} | {fmt(m[95])} |')
    L.append('')

    # Section 2: top discriminators per tier per dataset
    L.append('## 2. Top-10 feature discriminators (bleed vs rest)')
    L.append('')
    L.append(f'Features ranked by |d| at the {trade_pct}% bleed cutoff, '
             'per tier per dataset. Positive d = bleed has a HIGHER value '
             'of that feature; negative d = LOWER value.')
    L.append('')
    for label, res in [('IS', is_res), ('OOS', oos_res)]:
        L.append(f'### {label}')
        L.append('')
        for tier in sorted(res.keys()):
            r = res[tier]
            if r['n'] < MIN_TIER_N:
                continue
            L.append(f'**{tier}** (N={r["n"]:,})')
            L.append('')
            L.append('| Rank | Feature | d | bleed mean | rest mean |')
            L.append('|---:|---|---:|---:|---:|')
            for i, f in enumerate(r['features'][:10]):
                sign = '+' if f['d'] >= 0 else ''
                L.append(f'| {i+1} | `{f["name"]}` | {sign}{f["d"]:.2f} | '
                         f'{f["bleed_mean"]:+.3f} | {f["rest_mean"]:+.3f} |')
            L.append('')

    # Section 3: walk-forward stable shortlist
    L.append('## 3. Walk-forward stable discriminators')
    L.append('')
    L.append('Features where sign of d matches IS and OOS AND '
             '|d| >= 0.30 on both. Per tier.')
    L.append('')
    L.append('| Tier | Feature | d_IS | d_OOS | min |d| | bleed_mean_IS | '
             'bleed_mean_OOS | rest_mean_IS | rest_mean_OOS |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    tiers = sorted(set(is_res.keys()) & set(oos_res.keys()))
    any_rows = False
    for tier in tiers:
        r_is = is_res[tier]
        r_oos = oos_res[tier]
        if r_is['n'] < MIN_TIER_N or r_oos['n'] < MIN_TIER_N:
            continue
        # Match features by name
        is_by_name = {f['name']: f for f in r_is['features']}
        oos_by_name = {f['name']: f for f in r_oos['features']}
        hits = []
        for name, f_is in is_by_name.items():
            f_oos = oos_by_name.get(name)
            if f_oos is None:
                continue
            if f_is['d'] * f_oos['d'] <= 0:
                continue  # sign mismatch
            mn = min(abs(f_is['d']), abs(f_oos['d']))
            if mn < 0.30:
                continue
            hits.append((mn, name, f_is, f_oos))
        hits.sort(key=lambda x: -x[0])
        for mn, name, f_is, f_oos in hits[:10]:
            any_rows = True
            sign_is = '+' if f_is['d'] >= 0 else ''
            sign_oos = '+' if f_oos['d'] >= 0 else ''
            L.append(f'| {tier} | `{name}` | '
                     f'{sign_is}{f_is["d"]:.2f} | '
                     f'{sign_oos}{f_oos["d"]:.2f} | '
                     f'{mn:.2f} | '
                     f'{f_is["bleed_mean"]:+.3f} | '
                     f'{f_oos["bleed_mean"]:+.3f} | '
                     f'{f_is["rest_mean"]:+.3f} | '
                     f'{f_oos["rest_mean"]:+.3f} |')
    if not any_rows:
        L.append('| — | — | — | — | — | — | — | — | — |')
        L.append('')
        L.append('_No features clear the bar. Bleed cohort is not '
                 'statically separable by any single feature with '
                 'walk-forward stability._')
    L.append('')

    # Section 4: where z_range ranks
    L.append('## 4. Original hypothesis check: `1h_z_range`')
    L.append('')
    L.append('Rank of `1h_z_range` among all 92 features, per tier per '
             'dataset. Rank 1 = top discriminator.')
    L.append('')
    L.append('| Dataset | Tier | Rank | d | bleed mean | rest mean |')
    L.append('|---|---|---:|---:|---:|---:|')
    for label, res in [('IS', is_res), ('OOS', oos_res)]:
        for tier in sorted(res.keys()):
            r = res[tier]
            if r['n'] < MIN_TIER_N:
                continue
            for i, f in enumerate(r['features']):
                if f['name'] == '1h_z_range':
                    sign = '+' if f['d'] >= 0 else ''
                    L.append(f'| {label} | {tier} | {i+1} | '
                             f'{sign}{f["d"]:.2f} | '
                             f'{f["bleed_mean"]:+.3f} | '
                             f'{f["rest_mean"]:+.3f} |')
                    break
    L.append('')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def analyze_dataset(path, feature_names, trade_pct):
    """Return {tier: analysis_dict}."""
    print(f'Loading {path}...')
    trades = load(path)
    X, y, tiers, _ = build_matrix(trades)
    if X is None:
        print(f'  no trades with features')
        return {}
    results = {}
    unique_tiers = sorted(set(tiers.tolist()))
    for tier in unique_tiers:
        mask = (tiers == tier)
        if mask.sum() < MIN_TIER_N:
            continue
        Xt = X[mask]
        yt = y[mask]
        results[tier] = analyze_tier(Xt, yt, feature_names, trade_pct)
    # Also overall
    results['ALL'] = analyze_tier(X, y, feature_names, trade_pct)
    return results


def print_summary(label, results):
    print(f'\n=== {label} ===')
    print(f'{"Tier":<18} {"N":>7} {"Total$":>10} {"TotLoss$":>10} '
          f'{"p50":>5} {"p70":>5} {"p80":>5} {"p90":>5}')
    print('-' * 72)
    for tier in sorted(results.keys()):
        r = results[tier]
        if r['n'] < MIN_TIER_N:
            continue
        m = r['milestones']
        def f(x):
            return '  -  ' if x is None else f'{x:>4.1f}%'
        print(f'{tier:<18} {r["n"]:>7,} {r["total_pnl"]:>+10,.0f} '
              f'{r["total_loss"]:>10,.0f} '
              f'{f(m[50])} {f(m[70])} {f(m[80])} {f(m[90])}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trade-pct', type=float, default=20.0,
                    help='Percent of worst-PnL trades to treat as bleed cohort')
    args = ap.parse_args()

    feature_names = list(FEATURE_NAMES[:91]) + ['1h_z_range']

    is_res = analyze_dataset(
        os.path.join(TRADES_DIR, 'blended_is.pkl'),
        feature_names, args.trade_pct)
    oos_res = analyze_dataset(
        os.path.join(TRADES_DIR, 'blended_oos.pkl'),
        feature_names, args.trade_pct)

    print_summary('IS', is_res)
    print_summary('OOS', oos_res)

    write_report(is_res, oos_res, OUT_PATH, args.trade_pct)
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
