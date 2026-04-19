"""
BIG_LOSS Entry Signature — are there features at entry that predict
catastrophic outcomes (pnl < -$50) BEFORE they develop?

Cohen d on 91-feature entry vectors: BIG_LOSS cohort vs everyone else.
If |d| > 0.5 on any feature, we have a universal entry-filter candidate
that could veto BIG_LOSS-prone setups across tiers.

Also does per-tier breakdown — same-tier signatures might be stronger
than cross-tier (since different tiers fire on different market states).

Usage:
    python tools/big_loss_entry_signature.py

Output: reports/findings/big_loss_entry_signature.md
"""
import os
import sys
import pickle
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_NAMES


TRADES_PATH = 'training_iso/output/trades/iso_is.pkl'
OUT_PATH = 'reports/findings/big_loss_entry_signature.md'

BIG_LOSS_THRESHOLD = -50.0


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def extract_feature_matrix(trades):
    X = []
    valid = []
    for t in trades:
        ef = t.get('entry_79d')
        if ef is None:
            continue
        if hasattr(ef, 'tolist'):
            ef = ef.tolist()
        if len(ef) < 91:
            continue
        X.append(list(ef[:91]))
        valid.append(t)
    return np.array(X, dtype=float), valid


def rank_features(X_a, X_b, top_k=15):
    """Cohen d for each feature: a - b. Returns (name, d, mean_a, mean_b)."""
    out = []
    for i, name in enumerate(FEATURE_NAMES[:91]):
        d = cohen_d(X_a[:, i], X_b[:, i])
        out.append((name, d, float(X_a[:, i].mean()), float(X_b[:, i].mean())))
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out[:top_k]


def main():
    print(f'Loading {TRADES_PATH}...')
    with open(TRADES_PATH, 'rb') as f:
        trades = pickle.load(f)
    print(f'  {len(trades):,} trades')

    big_loss = [t for t in trades if t.get('pnl', 0) < BIG_LOSS_THRESHOLD]
    other = [t for t in trades if t.get('pnl', 0) >= BIG_LOSS_THRESHOLD]
    print(f'  BIG_LOSS: {len(big_loss):,}  Other: {len(other):,}')

    X_bl, _ = extract_feature_matrix(big_loss)
    X_ot, _ = extract_feature_matrix(other)
    print(f'  Entry features extracted: BL={len(X_bl)}  Other={len(X_ot)}')

    # ── Cross-tier (all trades) ranking ───────────────────────────
    print()
    print('=' * 80)
    print('CROSS-TIER: Entry-feature Cohen d (BIG_LOSS vs Other)')
    print('=' * 80)
    ranked = rank_features(X_bl, X_ot, top_k=15)
    print(f'{"feature":<26} {"d":>7} {"BL mean":>10} {"Other mean":>11}')
    print('-' * 60)
    strong_found = False
    for name, d, bl_m, ot_m in ranked:
        mark = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
        print(f'{name:<26} {d:>+7.3f} {bl_m:>10.3f} {ot_m:>11.3f}{mark}')
        if abs(d) >= 0.5:
            strong_found = True
    print()
    if strong_found:
        print('  STRONG separator found — entry filter candidate.')
    else:
        top = ranked[0]
        print(f'  No strong cross-tier separator. Top: {top[0]} d={top[1]:+.2f} '
              f'(moderate, not rule-worthy alone).')

    # ── Per-tier breakdown ────────────────────────────────────────
    tier_counts = Counter(t.get('entry_tier', '?') for t in big_loss)
    per_tier_results = {}
    print()
    print('=' * 80)
    print('PER-TIER: Entry-feature Cohen d (BIG_LOSS vs Other within same tier)')
    print('=' * 80)
    for tier, n_bl in tier_counts.most_common():
        tier_bl = [t for t in big_loss if t.get('entry_tier') == tier]
        tier_ot = [t for t in other if t.get('entry_tier') == tier]
        if len(tier_bl) < 20 or len(tier_ot) < 50:
            continue
        X_bl_t, _ = extract_feature_matrix(tier_bl)
        X_ot_t, _ = extract_feature_matrix(tier_ot)
        if len(X_bl_t) < 20 or len(X_ot_t) < 50:
            continue
        ranked = rank_features(X_bl_t, X_ot_t, top_k=5)
        per_tier_results[tier] = {
            'n_bl': len(tier_bl), 'n_ot': len(tier_ot),
            'top': ranked,
        }
        print(f'\n{tier}:  BL={len(tier_bl)}  Other={len(tier_ot)}')
        for name, d, bl_m, ot_m in ranked:
            mark = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
            print(f'  {name:<26} {d:>+7.3f} {bl_m:>10.3f} {ot_m:>11.3f}{mark}')

    # ── Write markdown ────────────────────────────────────────────
    L = []
    L.append('# BIG_LOSS Entry Signature — physics-based filter candidate')
    L.append('')
    L.append(f'BIG_LOSS (pnl < ${BIG_LOSS_THRESHOLD}): **{len(big_loss):,} trades** '
             f'across all tiers. Can features at ENTRY predict them?')
    L.append('')
    L.append(f'Methodology: Cohen d between BIG_LOSS cohort and Other (pnl >= '
             f'${BIG_LOSS_THRESHOLD}) cohort on all 91 entry features. `**` |d| ≥ '
             f'0.5 (strong, ship as filter). `*` |d| ≥ 0.3 (moderate).')
    L.append('')

    L.append('## Cross-tier separators')
    L.append('')
    L.append('| feature | Cohen d | BL mean | Other mean |')
    L.append('|---|---:|---:|---:|')
    ranked = rank_features(X_bl, X_ot, top_k=15)
    for name, d, bl_m, ot_m in ranked:
        mark = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
        L.append(f'| {name} | {d:+.3f}{mark} | {bl_m:.3f} | {ot_m:.3f} |')
    L.append('')
    top = ranked[0]
    if abs(top[1]) >= 0.5:
        L.append(f'**Strong cross-tier separator**: `{top[0]}` (d={top[1]:+.2f}). '
                 f'Consider universal entry filter.')
    else:
        L.append(f'_No strong cross-tier separator. Top: `{top[0]}` d={top[1]:+.2f} '
                 f'— moderate. Cross-tier entry filter unlikely to help; try '
                 f'per-tier._')
    L.append('')

    L.append('## Per-tier separators (top 5 each)')
    L.append('')
    for tier, res in per_tier_results.items():
        L.append(f'### {tier}  (BL={res["n_bl"]}, Other={res["n_ot"]})')
        L.append('')
        L.append('| feature | d | BL mean | Other mean |')
        L.append('|---|---:|---:|---:|')
        for name, d, bl_m, ot_m in res['top']:
            mark = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
            L.append(f'| {name} | {d:+.3f}{mark} | {bl_m:.3f} | {ot_m:.3f} |')
        top_tier = res['top'][0]
        if abs(top_tier[1]) >= 0.5:
            L.append('')
            L.append(f'**Strong separator**: `{top_tier[0]}` d={top_tier[1]:+.2f} — '
                     f'tier-specific entry filter candidate.')
        L.append('')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
