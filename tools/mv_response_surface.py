"""
Multivariate Response Surface — per-tier feature interaction analysis.

Loads per-tier trade pickles from `training/output/isolated/{TIER}.pkl`
(produced by tools/run_tier_isolated.py) and finds where features INTERACT
to separate winners from losers — not just where they move the needle
alone.

Three analyses per tier:

  1. UNIVARIATE ranking (Cohen-d + GBM importance) — baseline.
  2. 2D INTERACTION HEATMAPS — for every pair of top-K features, bin both
     into quartiles and report cells with >=30 trades and WR that diverges
     from the tier's baseline WR by >=5 points. These are readable rules:
     "feat_A in Q4 AND feat_B in Q1 -> 61% WR over N=180 trades."
  3. DECISION TREE RULES — shallow tree (max_depth=3) on win/loss. Each
     leaf is an interpretable rule chain. Reports the rules sorted by
     |leaf_WR - baseline_WR| * leaf_N so big-effect large-N rules bubble up.

Output:
  reports/findings/mv_response_{TIER}.md  (one per tier)
  reports/findings/mv_response_summary.md (combined rule list)

Usage:
  python tools/mv_response_surface.py                   # all tiers
  python tools/mv_response_surface.py FADE_CALM KILL_SHOT   # specific
  python tools/mv_response_surface.py --topk 10         # more features
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_NAMES, N_FEATURES

ISOLATED_DIR = 'training/output/isolated'
OUT_DIR = 'reports/findings'
MIN_CELL_N = 30        # ignore 2D cells with fewer than this many trades
WR_DIVERGE_MIN = 5.0   # WR must diverge from baseline by at least this (pp)
DEFAULT_TOPK = 8       # number of top univariate features to cross-pair

ALL_TIERS = ['FADE_CALM', 'RIDE_AGAINST', 'KILL_SHOT', 'CASCADE',
             'FADE_AGAINST', 'MTF_BREAKOUT', 'MTF_EXHAUSTION', 'FREIGHT_TRAIN']


def load_trades(tier: str) -> pd.DataFrame:
    """Load per-tier isolated trades → DataFrame with features + outcome."""
    path = os.path.join(ISOLATED_DIR, f'{tier}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        trades = pickle.load(f)
    if not trades:
        return None

    rows = []
    for t in trades:
        # is_chain defaults False; skip chain contracts for tier entry analysis
        if t.get('is_chain', False):
            continue
        ef = t.get('entry_features')
        if ef is None:
            ef = t.get('entry_79d')   # back-compat
        if ef is None:
            continue
        if isinstance(ef, list):
            ef = np.array(ef)
        if len(ef) < 91:
            continue
        row = {f: float(ef[i]) for i, f in enumerate(FEATURE_NAMES[:91])}
        row['_pnl'] = float(t['pnl'])
        row['_win'] = 1 if t['pnl'] > 0 else 0
        row['_held'] = int(t.get('held', 0))
        rows.append(row)
    return pd.DataFrame(rows)


def univariate_rank(df: pd.DataFrame, topk: int) -> list:
    """Rank features by |Cohen d| between winners and losers. Return top K."""
    from scipy import stats
    winners = df[df['_win'] == 1]
    losers = df[df['_win'] == 0]
    if len(winners) < 10 or len(losers) < 10:
        return []
    ranked = []
    for feat in FEATURE_NAMES[:91]:
        wv = winners[feat].values
        lv = losers[feat].values
        if np.std(wv) == 0 and np.std(lv) == 0:
            continue
        pooled = np.sqrt((np.var(wv) + np.var(lv)) / 2)
        d = (np.mean(wv) - np.mean(lv)) / pooled if pooled > 0 else 0.0
        try:
            _, p = stats.ttest_ind(wv, lv, equal_var=False)
        except Exception:
            p = 1.0
        ranked.append((feat, d, p))
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    return ranked[:topk]


def gbm_importance(df: pd.DataFrame, topk: int) -> list:
    """Rank features by gradient-boosted tree importance on win/loss."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception:
        return []
    X = df[FEATURE_NAMES[:91]].values
    y = df['_win'].values
    if len(np.unique(y)) < 2:
        return []
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                        random_state=0)
    model.fit(X, y)
    imp = list(zip(FEATURE_NAMES[:91], model.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)
    return imp[:topk]


def interaction_cells(df: pd.DataFrame, feat_a: str, feat_b: str,
                      baseline_wr: float) -> list:
    """For one feature pair, return cells that diverge from baseline WR."""
    try:
        qa = pd.qcut(df[feat_a], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        qb = pd.qcut(df[feat_b], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except Exception:
        return []
    out = []
    for qav in ['Q1', 'Q2', 'Q3', 'Q4']:
        for qbv in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = (qa == qav) & (qb == qbv)
            n = int(mask.sum())
            if n < MIN_CELL_N:
                continue
            sub = df.loc[mask]
            wr = sub['_win'].mean() * 100
            avg = sub['_pnl'].mean()
            diverge = wr - baseline_wr
            if abs(diverge) >= WR_DIVERGE_MIN:
                out.append({
                    'feat_a': feat_a, 'qa': qav,
                    'feat_b': feat_b, 'qb': qbv,
                    'n': n, 'wr': wr, 'avg_pnl': avg,
                    'diverge_pp': diverge,
                })
    return out


def tree_rules(df: pd.DataFrame, baseline_wr: float, max_depth: int = 3) -> list:
    """Fit shallow tree on win/loss. Extract each leaf as a rule."""
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        return []
    X = df[FEATURE_NAMES[:91]].values
    y = df['_win'].values
    if len(np.unique(y)) < 2:
        return []
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=50,
                                   random_state=0)
    tree.fit(X, y)

    rules = []
    t = tree.tree_

    def walk(node_id, conditions):
        feat_idx = t.feature[node_id]
        if feat_idx < 0:
            # leaf
            n = int(t.n_node_samples[node_id])
            values = t.value[node_id][0]   # [n_loss, n_win]
            if n < 50:
                return
            wr = values[1] / (values[0] + values[1]) * 100
            # approx avg pnl is not available from sklearn tree; skip
            rules.append({
                'conditions': list(conditions),
                'n': n,
                'wr': wr,
                'diverge_pp': wr - baseline_wr,
            })
            return
        feat_name = FEATURE_NAMES[feat_idx]
        thr = t.threshold[node_id]
        left = t.children_left[node_id]
        right = t.children_right[node_id]
        walk(left,  conditions + [(feat_name, '<=', thr)])
        walk(right, conditions + [(feat_name, '>',  thr)])

    walk(0, [])
    rules.sort(key=lambda r: abs(r['diverge_pp']) * np.log1p(r['n']),
               reverse=True)
    return rules


def write_report(tier: str, df: pd.DataFrame, univ, gbm, cells, rules,
                 out_path: str):
    baseline_wr = df['_win'].mean() * 100
    avg_pnl = df['_pnl'].mean()
    lines = []
    lines.append(f'# Multivariate Response Surface — {tier}')
    lines.append('')
    lines.append(f'**Trades:** {len(df):,}  '
                 f'**WR:** {baseline_wr:.1f}%  '
                 f'**Avg $/trade:** ${avg_pnl:+.2f}')
    lines.append('')

    # ── Univariate top features ──
    lines.append('## Top univariate features (Cohen d)')
    lines.append('')
    lines.append('| Feature | Cohen d | p | Win mean | Loss mean |')
    lines.append('|---|---:|---:|---:|---:|')
    winners = df[df['_win'] == 1]
    losers = df[df['_win'] == 0]
    for feat, d, p in univ:
        wm = winners[feat].mean() if len(winners) else 0
        lm = losers[feat].mean() if len(losers) else 0
        lines.append(f'| {feat} | {d:+.3f} | {p:.4f} | '
                     f'{wm:.4f} | {lm:.4f} |')
    lines.append('')

    lines.append('## Top GBM importance')
    lines.append('')
    lines.append('| Feature | Importance |')
    lines.append('|---|---:|')
    for feat, imp in gbm:
        lines.append(f'| {feat} | {imp:.4f} |')
    lines.append('')

    # ── 2D interaction cells ──
    cells_sorted = sorted(cells, key=lambda c: abs(c['diverge_pp']),
                          reverse=True)
    lines.append(f'## 2D interaction cells (|ΔWR| ≥ {WR_DIVERGE_MIN:.0f}pp, '
                 f'N ≥ {MIN_CELL_N})')
    lines.append('')
    lines.append(f'Baseline WR = {baseline_wr:.1f}%. Cells listed with higher '
                 f'WR first; negative diverge_pp = cell is WORSE than baseline.')
    lines.append('')
    lines.append('| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |')
    lines.append('|---|---|---|---|---:|---:|---:|---:|')
    for c in cells_sorted[:50]:   # top 50
        lines.append(f'| {c["feat_a"]} | {c["qa"]} | {c["feat_b"]} | {c["qb"]} | '
                     f'{c["n"]:,} | {c["wr"]:.1f}% | {c["diverge_pp"]:+.1f} | '
                     f'${c["avg_pnl"]:+.2f} |')
    lines.append('')

    # ── Decision tree rules ──
    lines.append(f'## Shallow decision-tree rules (max_depth=3)')
    lines.append('')
    lines.append(f'Top rules by |ΔWR| × log(N). Baseline WR = {baseline_wr:.1f}%.')
    lines.append('')
    lines.append('| Rule | N | WR | ΔWR |')
    lines.append('|---|---:|---:|---:|')
    for r in rules[:15]:
        conds = ' AND '.join(f'{f} {op} {thr:.3f}' for f, op, thr in r['conditions'])
        if not conds:
            conds = '(no splits)'
        lines.append(f'| {conds} | {r["n"]:,} | {r["wr"]:.1f}% | {r["diverge_pp"]:+.1f} |')
    lines.append('')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run_tier(tier: str, topk: int) -> dict:
    df = load_trades(tier)
    if df is None or len(df) < 100:
        print(f'  {tier:<18} SKIP — only {0 if df is None else len(df)} trades')
        return None

    baseline_wr = df['_win'].mean() * 100
    avg_pnl = df['_pnl'].mean()

    # 1. Univariate
    univ = univariate_rank(df, topk)

    # 2. GBM importance
    gbm = gbm_importance(df, topk)

    # 3. 2D interactions across union of top univariate + top GBM
    top_feats = list(dict.fromkeys([f for f, _, _ in univ]
                                   + [f for f, _ in gbm]))[:topk]
    cells = []
    for fa, fb in combinations(top_feats, 2):
        cells.extend(interaction_cells(df, fa, fb, baseline_wr))

    # 4. Decision tree rules
    rules = tree_rules(df, baseline_wr)

    out_path = os.path.join(OUT_DIR, f'mv_response_{tier}.md')
    write_report(tier, df, univ, gbm, cells, rules, out_path)

    n_div_cells = len([c for c in cells if abs(c['diverge_pp']) >= WR_DIVERGE_MIN])
    strong_rules = [r for r in rules if abs(r['diverge_pp']) >= WR_DIVERGE_MIN]

    print(f'  {tier:<18} N={len(df):>6,} WR={baseline_wr:>4.1f}% '
          f'$/t={avg_pnl:+6.2f}  '
          f'2D_cells={n_div_cells:>3}  tree_rules={len(strong_rules):>2}  '
          f'-> {os.path.basename(out_path)}')
    return {
        'tier': tier, 'n': len(df), 'wr': baseline_wr, 'avg_pnl': avg_pnl,
        'n_cells': n_div_cells, 'n_rules': len(strong_rules),
        'univ': univ, 'gbm': gbm, 'cells': cells, 'rules': rules,
    }


def write_summary(results: list, out_path: str):
    lines = []
    lines.append('# Multivariate Response Surface — Summary')
    lines.append('')
    lines.append('| Tier | N | WR | Avg $ | 2D cells | Tree rules |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for r in results:
        lines.append(f'| {r["tier"]} | {r["n"]:,} | {r["wr"]:.1f}% | '
                     f'${r["avg_pnl"]:+.2f} | {r["n_cells"]} | {r["n_rules"]} |')
    lines.append('')

    lines.append('## Strongest candidate gate rules across tiers')
    lines.append('')
    lines.append('Pulled from the top interaction cells and tree rules. '
                 'ΔWR is measured against the tier\'s own baseline.')
    lines.append('')
    lines.append('| Tier | Rule | N | WR | ΔWR | avg $ |')
    lines.append('|---|---|---:|---:|---:|---:|')

    combined = []
    for r in results:
        tier = r['tier']
        # from 2D cells
        for c in r['cells']:
            rule_str = (f'{c["feat_a"]} in {c["qa"]} AND '
                        f'{c["feat_b"]} in {c["qb"]}')
            combined.append({
                'tier': tier, 'rule': rule_str, 'n': c['n'],
                'wr': c['wr'], 'diverge': c['diverge_pp'],
                'avg_pnl': c['avg_pnl'],
            })
        # from tree leaves (no avg pnl available, leave blank)
        for tr in r['rules']:
            conds = ' AND '.join(f'{f} {op} {thr:.2f}' for f, op, thr in tr['conditions'])
            if not conds:
                continue
            combined.append({
                'tier': tier, 'rule': conds, 'n': tr['n'],
                'wr': tr['wr'], 'diverge': tr['diverge_pp'],
                'avg_pnl': None,
            })
    combined.sort(key=lambda r: abs(r['diverge']) * np.log1p(r['n']),
                  reverse=True)
    for r in combined[:40]:
        avg_str = f'${r["avg_pnl"]:+.2f}' if r['avg_pnl'] is not None else '—'
        lines.append(f'| {r["tier"]} | {r["rule"]} | {r["n"]:,} | '
                     f'{r["wr"]:.1f}% | {r["diverge"]:+.1f} | {avg_str} |')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tiers', nargs='*', help='tiers to analyze (default: all)')
    parser.add_argument('--topk', type=int, default=DEFAULT_TOPK,
                        help='top features to cross for interactions')
    args = parser.parse_args()

    tiers = args.tiers or ALL_TIERS
    tiers = [t for t in tiers if t in ALL_TIERS]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Tiers: {tiers}')
    print(f'TopK:  {args.topk}')
    print()
    print(f'{"Tier":<18} {"N":>7} {"WR":>6} {"$/t":>8}  '
          f'{"2D_cells":>10}  {"tree_rules":>12}')
    print('-' * 95)

    results = []
    for tier in tiers:
        r = run_tier(tier, args.topk)
        if r is not None:
            results.append(r)

    summary_path = os.path.join(OUT_DIR, 'mv_response_summary.md')
    write_summary(results, summary_path)
    print()
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
