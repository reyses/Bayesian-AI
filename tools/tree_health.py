"""
Tree Health — check if tree leaves are actually distinct.

Compares every leaf pair on three dimensions:
  1. APPROACH: mean 79D path in bars before entry
  2. ENTRY: mean 79D snapshot at entry
  3. TRADE PATH: mean PnL curve during the trade

If two leaves have high similarity on all three → they're duplicates.
The tree split on noise, not on real market conditions.

Also checks:
  - Spread: how tight is each leaf? Wide spread = catch-all, not a specialist.
  - Regret alignment: do similar leaves have similar regret profiles?
  - Singleton leaves: leaves with < N trades are statistically meaningless.

Output: reports/findings/tree_health_YYYYMMDD.md

Usage:
    python tools/tree_health.py
    python tools/tree_health.py --min-trades 10    # minimum trades per leaf
    python tools/tree_health.py --sim-threshold 0.90  # cosine similarity threshold
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features import FEATURE_NAMES

TRADE_LOG = 'DATA/NMP_TRADES/nmp_is.pkl'
REGRET_FILE = 'DATA/NMP_TREE/regret_analysis.csv'
TREE_FILE = 'DATA/NMP_TREE/strategy_tree.pkl'
BOOK_FILE = 'DATA/NMP_TREE/strategy_book.pkl'

# Minimum trades for a leaf to be considered meaningful
MIN_TRADES_DEFAULT = 5
# Cosine similarity threshold for "duplicate" detection
SIM_THRESHOLD_DEFAULT = 0.90


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Tree health — leaf distinctiveness analysis')
    p.add_argument('--min-trades', type=int, default=MIN_TRADES_DEFAULT)
    p.add_argument('--sim-threshold', type=float, default=SIM_THRESHOLD_DEFAULT)
    return p.parse_args()


def cosine_sim(a, b):
    """Cosine similarity between two vectors. Handles zero vectors."""
    a = np.nan_to_num(a.flatten())
    b = np.nan_to_num(b.flatten())
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_data():
    """Load trades, regret, tree, and book."""
    with open(TRADE_LOG, 'rb') as f:
        trades = pickle.load(f)

    regret = pd.read_csv(REGRET_FILE)

    with open(TREE_FILE, 'rb') as f:
        tree_data = pickle.load(f)

    tree = tree_data['tree']

    # Classify trades
    for i, t in enumerate(trades):
        feat = np.array(t['entry_79d']).reshape(1, -1)
        t['leaf_id'] = int(tree.apply(np.nan_to_num(feat))[0])
        t['regret_row'] = regret.iloc[i].to_dict() if i < len(regret) else {}

    # Load book if available
    book = {}
    if os.path.exists(BOOK_FILE):
        with open(BOOK_FILE, 'rb') as f:
            book = pickle.load(f)

    return trades, regret, tree_data, book


def build_leaf_signatures(trades, min_trades):
    """Build approach, entry, and path signatures per leaf."""
    by_leaf = defaultdict(list)
    for t in trades:
        by_leaf[t['leaf_id']].append(t)

    signatures = {}
    for lid, leaf_trades in by_leaf.items():
        if len(leaf_trades) < min_trades:
            continue

        # ENTRY: mean 79D at entry
        entry_feats = np.nan_to_num(np.array([np.array(t['entry_79d']) for t in leaf_trades]))
        entry_mean = entry_feats.mean(axis=0)
        entry_std = entry_feats.std(axis=0)
        entry_spread = float(entry_std.mean())

        # APPROACH: mean 79D in bars before entry (flattened)
        approach_vecs = []
        for t in leaf_trades:
            approach = t.get('approach', [])
            if approach:
                app_feats = [a['features'] for a in approach if 'features' in a]
                if app_feats:
                    approach_vecs.append(np.concatenate(app_feats))
        approach_mean = np.nan_to_num(np.mean(approach_vecs, axis=0)) if approach_vecs else None

        # TRADE PATH: PnL curve (not 79D, to keep comparison fast)
        path_curves = []
        for t in leaf_trades:
            path = t.get('path', [])
            if path:
                pnls = [p['pnl'] for p in path]
                path_curves.append(pnls)

        # Pad paths to same length for averaging
        path_mean = None
        if path_curves:
            max_len = min(30, max(len(p) for p in path_curves))
            padded = []
            for p in path_curves:
                if len(p) >= max_len:
                    padded.append(p[:max_len])
                else:
                    padded.append(p + [p[-1]] * (max_len - len(p)))
            path_mean = np.mean(padded, axis=0)

        # Regret profile
        regret_dist = defaultdict(int)
        for t in leaf_trades:
            action = t.get('regret_row', {}).get('best_action', 'unknown')
            regret_dist[action] += 1
        total = sum(regret_dist.values())
        regret_pcts = {k: v / max(total, 1) for k, v in regret_dist.items()}

        signatures[lid] = {
            'n_trades': len(leaf_trades),
            'entry_mean': entry_mean,
            'entry_spread': entry_spread,
            'approach_mean': approach_mean,
            'path_mean': path_mean,
            'regret_dist': regret_pcts,
            'wr': sum(1 for t in leaf_trades if t['pnl'] > 0) / len(leaf_trades),
            'total_pnl': sum(t['pnl'] for t in leaf_trades),
            'avg_pnl': sum(t['pnl'] for t in leaf_trades) / len(leaf_trades),
        }

    return signatures


def find_duplicates(signatures, sim_threshold):
    """Find leaf pairs that are too similar (potential duplicates)."""
    leaves = sorted(signatures.keys())
    duplicates = []

    for i, lid_a in enumerate(leaves):
        for lid_b in leaves[i + 1:]:
            sig_a = signatures[lid_a]
            sig_b = signatures[lid_b]

            # Entry similarity
            entry_sim = cosine_sim(sig_a['entry_mean'], sig_b['entry_mean'])

            # Approach similarity (if both have it)
            approach_sim = 0.0
            if sig_a['approach_mean'] is not None and sig_b['approach_mean'] is not None:
                # Need same length for comparison
                min_len = min(len(sig_a['approach_mean']), len(sig_b['approach_mean']))
                if min_len > 0:
                    approach_sim = cosine_sim(
                        sig_a['approach_mean'][:min_len],
                        sig_b['approach_mean'][:min_len]
                    )

            # Path similarity
            path_sim = 0.0
            if sig_a['path_mean'] is not None and sig_b['path_mean'] is not None:
                min_len = min(len(sig_a['path_mean']), len(sig_b['path_mean']))
                if min_len > 0:
                    path_sim = cosine_sim(
                        sig_a['path_mean'][:min_len],
                        sig_b['path_mean'][:min_len]
                    )

            # Regret profile similarity (Jensen-Shannon-like)
            all_actions = set(sig_a['regret_dist'].keys()) | set(sig_b['regret_dist'].keys())
            regret_sim = 0.0
            if all_actions:
                vec_a = np.array([sig_a['regret_dist'].get(a, 0) for a in sorted(all_actions)])
                vec_b = np.array([sig_b['regret_dist'].get(a, 0) for a in sorted(all_actions)])
                regret_sim = cosine_sim(vec_a, vec_b)

            # Overall similarity: weighted average
            # Entry is most important, then path, then approach, then regret
            overall = 0.4 * entry_sim + 0.3 * path_sim + 0.15 * approach_sim + 0.15 * regret_sim

            if overall >= sim_threshold:
                duplicates.append({
                    'leaf_a': lid_a,
                    'leaf_b': lid_b,
                    'entry_sim': entry_sim,
                    'approach_sim': approach_sim,
                    'path_sim': path_sim,
                    'regret_sim': regret_sim,
                    'overall': overall,
                })

    return sorted(duplicates, key=lambda d: -d['overall'])


def main():
    args = parse_args()
    print(f'Tree Health Analysis')
    print(f'  Min trades/leaf: {args.min_trades} | Sim threshold: {args.sim_threshold}')

    trades, regret, tree_data, book = load_data()
    print(f'  {len(trades)} trades, {len(tree_data["branches"])} tree branches')

    signatures = build_leaf_signatures(trades, args.min_trades)
    print(f'  {len(signatures)} leaves with >= {args.min_trades} trades')

    # Singleton analysis
    all_leaves = set(b['leaf_id'] for b in tree_data['branches'])
    by_leaf = defaultdict(int)
    for t in trades:
        by_leaf[t['leaf_id']] += 1
    singletons = [lid for lid in all_leaves if by_leaf.get(lid, 0) < args.min_trades]
    print(f'  {len(singletons)} singleton leaves (< {args.min_trades} trades) — statistically meaningless')

    # Spread analysis
    spreads = [(lid, sig['entry_spread'], sig['n_trades']) for lid, sig in signatures.items()]
    spreads.sort(key=lambda x: -x[1])

    # Duplicate detection
    duplicates = find_duplicates(signatures, args.sim_threshold)
    print(f'  {len(duplicates)} potential duplicate pairs (similarity >= {args.sim_threshold})')

    # Build report
    lines = []
    date_str = datetime.now().strftime('%Y-%m-%d')

    lines.append(f'# Tree Health Report — {date_str}')
    lines.append(f'')
    lines.append(f'## Summary')
    lines.append(f'- Total tree branches: {len(all_leaves)}')
    lines.append(f'- Leaves with >= {args.min_trades} trades: {len(signatures)}')
    lines.append(f'- Singleton leaves (< {args.min_trades} trades): {len(singletons)}')
    lines.append(f'- Potential duplicate pairs: {len(duplicates)}')
    lines.append(f'- **Effective distinct strategies: ~{len(signatures) - len(duplicates)}**')
    lines.append(f'')

    # Spread table
    lines.append(f'## Leaf Spread (entry 79D std — lower = tighter specialist)')
    lines.append(f'| Leaf | Trades | Spread | WR | $/trade | Total$ |')
    lines.append(f'|------|--------|--------|-----|---------|--------|')
    for lid, spread, n in spreads:
        sig = signatures[lid]
        lines.append(f'| {lid} | {n} | {spread:.3f} | {sig["wr"]:.0%} | '
                     f'${sig["avg_pnl"]:.0f} | ${sig["total_pnl"]:.0f} |')
    lines.append(f'')

    # Duplicates table
    if duplicates:
        lines.append(f'## Duplicate Pairs (overall similarity >= {args.sim_threshold})')
        lines.append(f'| Leaf A | Leaf B | Entry | Approach | Path | Regret | Overall |')
        lines.append(f'|--------|--------|-------|----------|------|--------|---------|')
        for d in duplicates:
            lines.append(f'| {d["leaf_a"]} | {d["leaf_b"]} | {d["entry_sim"]:.2f} | '
                         f'{d["approach_sim"]:.2f} | {d["path_sim"]:.2f} | '
                         f'{d["regret_sim"]:.2f} | **{d["overall"]:.2f}** |')
        lines.append(f'')

        # Merge recommendations
        lines.append(f'## Merge Recommendations')
        merged_groups = []
        used = set()
        for d in duplicates:
            if d['leaf_a'] not in used and d['leaf_b'] not in used:
                merged_groups.append((d['leaf_a'], d['leaf_b']))
                used.add(d['leaf_a'])
                used.add(d['leaf_b'])
        for a, b in merged_groups:
            sig_a = signatures[a]
            sig_b = signatures[b]
            lines.append(f'- Merge leaf {a} ({sig_a["n_trades"]} trades, ${sig_a["total_pnl"]:.0f}) '
                         f'+ leaf {b} ({sig_b["n_trades"]} trades, ${sig_b["total_pnl"]:.0f})')
        lines.append(f'')

    # Singletons
    if singletons:
        lines.append(f'## Singleton Leaves (< {args.min_trades} trades — drop or merge)')
        singleton_info = []
        for lid in singletons:
            n = by_leaf.get(lid, 0)
            singleton_info.append((lid, n))
        singleton_info.sort(key=lambda x: -x[1])
        for lid, n in singleton_info:
            lines.append(f'- Leaf {lid}: {n} trades')
        lines.append(f'')

    # Regret profile diversity
    lines.append(f'## Regret Profile Diversity (per leaf)')
    lines.append(f'| Leaf | N | Top Action | Top% | 2nd Action | 2nd% |')
    lines.append(f'|------|---|------------|------|------------|------|')
    for lid in sorted(signatures.keys()):
        sig = signatures[lid]
        sorted_actions = sorted(sig['regret_dist'].items(), key=lambda x: -x[1])
        top = sorted_actions[0] if sorted_actions else ('?', 0)
        second = sorted_actions[1] if len(sorted_actions) > 1 else ('?', 0)
        lines.append(f'| {lid} | {sig["n_trades"]} | {top[0]} | {top[1]:.0%} | '
                     f'{second[0]} | {second[1]:.0%} |')

    # Save report
    os.makedirs('reports/findings', exist_ok=True)
    report_path = f'reports/findings/tree_health_{date_str.replace("-", "")}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nReport: {report_path}')

    # Print key findings
    print(f'\n--- KEY FINDINGS ---')
    print(f'  Distinct strategies: ~{len(signatures) - len(duplicates)} '
          f'(from {len(all_leaves)} tree branches)')
    if duplicates:
        print(f'  Top duplicate: leaf {duplicates[0]["leaf_a"]} ↔ {duplicates[0]["leaf_b"]} '
              f'(similarity={duplicates[0]["overall"]:.2f})')
    if spreads:
        widest = spreads[0]
        tightest = spreads[-1]
        print(f'  Widest leaf: {widest[0]} (spread={widest[1]:.3f}, {widest[2]} trades) — catch-all?')
        print(f'  Tightest leaf: {tightest[0]} (spread={tightest[1]:.3f}, {tightest[2]} trades) — specialist')


if __name__ == '__main__':
    main()
