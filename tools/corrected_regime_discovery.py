"""
Corrected-trade regime discovery — CART on best_action labels.

Hypothesis under test: is there a regime (subset of entry 91D feature space)
where counter-direction would have been materially better than vanilla?

Method:
  1. Load corrected_is.pkl (9,882 iso trades with oracle best_action).
  2. Target: binary(best_action starts with 'counter') — flip-at-entry wins.
  3. Features: entry_91d vector (named, not indexed).
  4. Train DecisionTreeClassifier on 80%, validate on 20% (day-stratified).
  5. For each leaf with N >= 100, report:
       - counter share (baseline 47%)
       - mean (pnl - original_pnl) delta
       - feature conditions (interpretable rule)
  6. Verdict per leaf: CANDIDATE (counter > 58% and delta > $10) vs noise.

If no candidate leaf survives: direction at entry is confirmed random on
this feature set.

Writes reports/findings/regime_discovery_<ts>.md with full tree + leaf table.

Usage:
    python tools/corrected_regime_discovery.py
    python tools/corrected_regime_discovery.py --max-depth 5 --min-leaf 150
"""
import os
import sys
import pickle
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.features import FEATURE_NAMES  # 91 named columns


def build_xy(corrected, iso):
    """Build X (n x 91), y (binary counter=1/same=0), delta ($ oracle - actual),
    and days array for stratified split.

    CRITICAL: entry_79d in corrected_is.pkl is feature-leaked — regret.py
    replaces it with approach-buffer features (up to 10 bars before actual
    entry) for 99.7% of trades. We join the corrected labels to the ORIGINAL
    iso trade's entry_79d by (day, timestamp) to get clean at-entry features.
    """
    # Index iso trades by (day, timestamp)
    iso_idx = {}
    for t in iso:
        key = (t.get('day'), float(t.get('timestamp', 0)))
        if len(t.get('entry_79d', [])) >= 91:
            iso_idx[key] = np.asarray(t['entry_79d'], dtype=np.float32)

    X, y, delta, days = [], [], [], []
    n_missing = 0
    for c in corrected:
        key = (c.get('day'), float(c.get('timestamp', 0)))
        entry = iso_idx.get(key)
        if entry is None:
            n_missing += 1
            continue
        ba = c.get('best_action', '')
        is_counter = 1 if ba.startswith('counter') else 0
        X.append(entry)
        y.append(is_counter)
        delta.append(c.get('pnl', 0.0) - c.get('original_pnl', 0.0))
        days.append(c.get('day', ''))
    print(f'  join: matched {len(X)}/{len(corrected)} corrected trades '
          f'({n_missing} missing in iso)')
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int32),
            np.asarray(delta, dtype=np.float32),
            np.asarray(days))


def day_stratified_split(days, valid_frac=0.2, seed=17):
    """Split by day to avoid intra-day leakage."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(days)))
    rng.shuffle(unique)
    n_valid = max(1, int(len(unique) * valid_frac))
    valid_days = set(unique[:n_valid])
    train_mask = np.array([d not in valid_days for d in days])
    valid_mask = ~train_mask
    return train_mask, valid_mask, len(unique) - n_valid, n_valid


def extract_leaf_rules(tree, feature_names):
    """Walk a fitted sklearn tree and return per-leaf rule list."""
    t = tree.tree_
    leaves = []

    def recurse(node, conditions):
        if t.children_left[node] == -1:  # leaf
            leaves.append({
                'node_id': node,
                'conditions': list(conditions),
                'n_samples': int(t.n_node_samples[node]),
                'value': t.value[node][0],  # [n_same, n_counter]
            })
            return
        feat = feature_names[t.feature[node]]
        thr = t.threshold[node]
        recurse(t.children_left[node], conditions + [f'{feat} <= {thr:.4f}'])
        recurse(t.children_right[node], conditions + [f'{feat} > {thr:.4f}'])

    recurse(0, [])
    return leaves


def main():
    from sklearn.tree import DecisionTreeClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--min-leaf', type=int, default=100)
    ap.add_argument('--corrected', default='training_iso/output/trades/corrected_is.pkl')
    ap.add_argument('--iso', default='training_iso/output/trades/iso_is.pkl',
                    help='source for at-entry features (corrected_is has leakage)')
    args = ap.parse_args()

    with open(args.corrected, 'rb') as f:
        corrected = pickle.load(f)
    print(f'Loaded {len(corrected)} corrected trades')
    with open(args.iso, 'rb') as f:
        iso = pickle.load(f)
    print(f'Loaded {len(iso)} iso trades (for clean entry features)')

    X, y, delta, days = build_xy(corrected, iso)
    print(f'Built X={X.shape}, counter share={y.mean():.3f} (baseline)')

    train_mask, valid_mask, n_td, n_vd = day_stratified_split(days)
    print(f'Day-stratified split: {n_td} train days / {n_vd} valid days')
    print(f'  train N={train_mask.sum()}  valid N={valid_mask.sum()}')

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_leaf,
        class_weight=None,
        random_state=17,
    )
    clf.fit(X[train_mask], y[train_mask])

    train_acc = clf.score(X[train_mask], y[train_mask])
    valid_acc = clf.score(X[valid_mask], y[valid_mask])
    baseline_acc = max(1 - y.mean(), y.mean())
    print(f'Train acc={train_acc:.3f}  Valid acc={valid_acc:.3f}  '
          f'Baseline(majority)={baseline_acc:.3f}')

    leaves = extract_leaf_rules(clf, FEATURE_NAMES)
    print(f'Tree has {len(leaves)} leaves\n')

    # Evaluate each leaf on VALID set (OOS measurement)
    leaf_ids_valid = clf.apply(X[valid_mask])
    leaf_ids_train = clf.apply(X[train_mask])
    y_valid = y[valid_mask]
    delta_valid = delta[valid_mask]
    y_train = y[train_mask]
    delta_train = delta[train_mask]

    rows = []
    for leaf in leaves:
        lid = leaf['node_id']
        tr_idx = (leaf_ids_train == lid)
        va_idx = (leaf_ids_valid == lid)
        n_tr = tr_idx.sum()
        n_va = va_idx.sum()
        if n_tr == 0:
            continue
        counter_tr = y_train[tr_idx].mean() if n_tr else 0
        counter_va = y_valid[va_idx].mean() if n_va else 0
        delta_tr = delta_train[tr_idx].mean() if n_tr else 0
        delta_va = delta_valid[va_idx].mean() if n_va else 0
        rule = ' AND '.join(leaf['conditions']) if leaf['conditions'] else '(root)'
        rows.append({
            'leaf': lid,
            'n_train': int(n_tr),
            'n_valid': int(n_va),
            'counter_train': counter_tr,
            'counter_valid': counter_va,
            'delta_train': delta_tr,
            'delta_valid': delta_va,
            'rule': rule,
        })
    rows.sort(key=lambda r: -r['counter_valid'])

    print(f'{"Leaf":>5} {"N_tr":>6} {"N_va":>6} {"Ctr%_tr":>8} {"Ctr%_va":>8} '
          f'{"$D_tr":>8} {"$D_va":>8}  Rule')
    print('-' * 115)
    candidates = []
    for r in rows:
        flag = ''
        if (r['counter_valid'] > 0.58 and r['delta_valid'] > 10 and r['n_valid'] > 30
                and r['counter_train'] > 0.58):
            flag = ' *CAND'
            candidates.append(r)
        elif (r['counter_valid'] < 0.40 and r['delta_valid'] < -10):
            flag = ' *SKIP'
        print(f'{r["leaf"]:>5} {r["n_train"]:>6,} {r["n_valid"]:>6,} '
              f'{r["counter_train"]*100:>7.1f}% {r["counter_valid"]*100:>7.1f}% '
              f'${r["delta_train"]:>+7.2f} ${r["delta_valid"]:>+7.2f}  {r["rule"][:60]}{flag}')

    print()
    print(f'Candidates (>58% counter both folds + >$10 delta valid + n>30): {len(candidates)}')
    for c in candidates:
        print(f'  Leaf {c["leaf"]}: {c["rule"]}')

    # Write report
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md_path = os.path.join(out_dir, f'regime_discovery_{ts}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# Regime discovery from corrected trades — {ts}\n\n')
        f.write(f'Dataset: {len(corrected):,} corrected iso trades.\n')
        f.write(f'Baseline counter share: {y.mean()*100:.1f}% '
                f'(direction flip would\'ve won this often).\n\n')
        f.write(f'Day-stratified split: {n_td} train / {n_vd} valid days.\n')
        f.write(f'Tree max_depth={args.max_depth}, min_samples_leaf={args.min_leaf}.\n\n')
        f.write(f'**Train acc: {train_acc:.3f}**  |  '
                f'**Valid acc: {valid_acc:.3f}**  |  '
                f'Majority baseline: {baseline_acc:.3f}\n\n')
        f.write('## Leaf table (sorted by counter share on valid)\n\n')
        f.write('| Leaf | N_train | N_valid | Counter%_train | Counter%_valid | D$_train | D$_valid | Rule | Flag |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|---|---|\n')
        for r in rows:
            flag = ''
            if (r['counter_valid'] > 0.58 and r['delta_valid'] > 10 and r['n_valid'] > 30
                    and r['counter_train'] > 0.58):
                flag = 'CAND'
            elif (r['counter_valid'] < 0.40 and r['delta_valid'] < -10):
                flag = 'SKIP'
            f.write(f'| {r["leaf"]} | {r["n_train"]:,} | {r["n_valid"]:,} | '
                    f'{r["counter_train"]*100:.1f}% | {r["counter_valid"]*100:.1f}% | '
                    f'${r["delta_train"]:+.2f} | ${r["delta_valid"]:+.2f} | '
                    f'`{r["rule"]}` | {flag} |\n')
        f.write('\n## Verdict\n\n')
        if candidates:
            f.write(f'**{len(candidates)} candidate regime(s) found.** Listed above with rules. '
                    f'These pass the bar of >58% counter on both train and valid, >$10 mean '
                    f'oracle delta on valid, and >30 valid samples. Worth porting as tier '
                    f'rules in training_iso/nightmare_iso.py for end-to-end validation.\n')
        else:
            f.write('**No candidate regime survives.** No leaf achieves >58% counter on both '
                    f'train and valid with >$10 delta. Direction at entry is confirmed random '
                    f'on this feature set. Stop chasing flip tiers; pivot effort to exits or '
                    f'entry filtering.\n')
    print(f'\nWrote report: {md_path}')


if __name__ == '__main__':
    main()
