"""
Derive Physics — extract entry/exit rules from corrected trades.

Reads profitable corrected trades, groups by direction + hold bucket,
extracts 79D feature ranges at entry and exit. Ranks features by
discriminative power (what separates winners in this group from others).

Output: physics-based ExNMP rules as readable if/then conditions.

Usage:
    python tools/derive_physics.py
    python tools/derive_physics.py --min-trades 50     # min group size
    python tools/derive_physics.py --top-features 10   # features per rule
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D, TF_ORDER

CORRECTED_LOG = 'training/output/trades/corrected_is.pkl'

# Hold duration buckets (same as tree.py)
HOLD_BUCKETS = [(0, 3, 'fast'), (3, 8, 'medium'), (8, 16, 'long'), (16, 999, 'extended')]

# Percentile range for rule extraction (inner 80% of profitable trades)
RULE_PERCENTILE_LO = 10
RULE_PERCENTILE_HI = 90

# Minimum absolute effect size to include a feature in a rule
MIN_EFFECT_SIZE = 0.3


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Derive physics-based entry/exit rules')
    p.add_argument('--min-trades', type=int, default=30,
                   help='Minimum trades per group to derive a rule')
    p.add_argument('--top-features', type=int, default=10,
                   help='Max features per rule')
    return p.parse_args()


def load_corrected_trades():
    """Load corrected trades and split into features + metadata."""
    with open(CORRECTED_LOG, 'rb') as f:
        trades = pickle.load(f)

    rows = []
    for t in trades:
        entry_79d = np.array(t.get('entry_79d', []))
        if len(entry_79d) != len(FEATURE_NAMES_79D):
            continue

        row = {
            'pnl': t['pnl'],
            'dir': t['dir'],
            'held': t['held'],
            'day': t.get('day', ''),
            'best_action': t.get('best_action', ''),
            'original_pnl': t.get('original_pnl', 0),
            'early_bars': t.get('early_bars', 0),
        }
        for j, name in enumerate(FEATURE_NAMES_79D):
            row[name] = float(entry_79d[j])
        rows.append(row)

    return pd.DataFrame(rows)


def classify_group(direction, held):
    """Classify into direction + hold bucket."""
    for lo, hi, label in HOLD_BUCKETS:
        if lo <= held < hi:
            return f'{direction}_{label}'
    return f'{direction}_extended'


def compute_discriminative_power(group_df, all_df, feature_cols):
    """Rank features by how well they separate this group from all others.

    Uses standardized mean difference (Cohen's d):
      d = (mean_group - mean_all) / pooled_std

    Higher |d| = more discriminative.
    """
    scores = []
    for feat in feature_cols:
        group_vals = group_df[feat].values
        all_vals = all_df[feat].values

        group_mean = np.nanmean(group_vals)
        all_mean = np.nanmean(all_vals)
        group_std = np.nanstd(group_vals)
        all_std = np.nanstd(all_vals)

        # Pooled std
        pooled = np.sqrt((group_std**2 + all_std**2) / 2)
        if pooled < 1e-10:
            d = 0.0
        else:
            d = (group_mean - all_mean) / pooled

        scores.append({
            'feature': feat,
            'group_mean': group_mean,
            'all_mean': all_mean,
            'cohen_d': d,
            'abs_d': abs(d),
            'group_p10': np.nanpercentile(group_vals, RULE_PERCENTILE_LO),
            'group_p50': np.nanpercentile(group_vals, 50),
            'group_p90': np.nanpercentile(group_vals, RULE_PERCENTILE_HI),
        })

    return pd.DataFrame(scores).sort_values('abs_d', ascending=False)


def format_rule(feat_row):
    """Format one feature condition as a human-readable rule."""
    name = feat_row['feature']
    d = feat_row['cohen_d']
    p10 = feat_row['group_p10']
    p50 = feat_row['group_p50']
    p90 = feat_row['group_p90']

    # Direction of the condition
    if d > 0:
        condition = f'{name} > {p10:.2f}'
        note = f'(typical={p50:.2f}, range=[{p10:.2f}, {p90:.2f}])'
    else:
        condition = f'{name} < {p90:.2f}'
        note = f'(typical={p50:.2f}, range=[{p10:.2f}, {p90:.2f}])'

    strength = 'STRONG' if abs(d) > 0.8 else 'MODERATE' if abs(d) > 0.5 else 'WEAK'

    return condition, note, strength, d


def main():
    args = parse_args()

    print('Derive Physics — extract entry/exit rules from corrected trades')
    df = load_corrected_trades()
    print(f'  Loaded {len(df)} corrected trades')

    # Filter to profitable only
    profitable = df[df['pnl'] > 0]
    losing = df[df['pnl'] <= 0]
    print(f'  Profitable: {len(profitable)} ({len(profitable)/len(df)*100:.0f}%)')
    print(f'  Losing: {len(losing)} ({len(losing)/len(df)*100:.0f}%)')

    # Group by direction + hold bucket
    df['group'] = df.apply(lambda r: classify_group(r['dir'], r['held']), axis=1)
    profitable['group'] = profitable.apply(
        lambda r: classify_group(r['dir'], r['held']), axis=1)

    feature_cols = [c for c in FEATURE_NAMES_79D if c in df.columns]

    lines = []
    date_str = datetime.now().strftime('%Y-%m-%d')
    lines.append(f'# Physics-Based Entry Rules — {date_str}')
    lines.append(f'')
    lines.append(f'Derived from {len(profitable)} profitable corrected trades.')
    lines.append(f'Features ranked by Cohen\'s d (standardized mean difference vs all trades).')
    lines.append(f'')

    # Global: what separates ALL winners from ALL losers?
    lines.append(f'## Global: Winners vs Losers')
    lines.append(f'')
    global_scores = compute_discriminative_power(profitable, df, feature_cols)
    strong_global = global_scores[global_scores['abs_d'] >= MIN_EFFECT_SIZE]
    lines.append(f'Features that distinguish winners ({len(strong_global)} with |d| >= {MIN_EFFECT_SIZE}):')
    lines.append(f'')
    lines.append(f'| Feature | Cohen\'s d | Winner typical | All typical | Direction |')
    lines.append(f'|---------|-----------|---------------|------------|-----------|')
    for _, row in strong_global.head(15).iterrows():
        direction = 'HIGHER' if row['cohen_d'] > 0 else 'LOWER'
        lines.append(f'| {row["feature"]} | {row["cohen_d"]:+.2f} | '
                     f'{row["group_p50"]:.2f} | {row["all_mean"]:.2f} | {direction} |')
    lines.append(f'')

    # Per group rules
    groups = sorted(profitable['group'].unique())
    lines.append(f'## Per-Strategy Rules ({len(groups)} groups)')
    lines.append(f'')

    all_rules = {}
    for group in groups:
        group_df = profitable[profitable['group'] == group]
        if len(group_df) < args.min_trades:
            continue

        group_pnl = group_df['pnl'].sum()
        group_wr = len(group_df) / max(len(df[df['group'] == group]), 1)
        avg_pnl = group_df['pnl'].mean()
        avg_held = group_df['held'].mean()

        lines.append(f'### ExNMP: {group}')
        lines.append(f'  Trades: {len(group_df)} | Avg PnL: ${avg_pnl:.1f} | '
                     f'Total: ${group_pnl:,.0f} | Avg hold: {avg_held:.0f} bars')
        lines.append(f'')

        # Discriminative features for this group
        scores = compute_discriminative_power(group_df, df, feature_cols)
        top = scores[scores['abs_d'] >= MIN_EFFECT_SIZE].head(args.top_features)

        if len(top) == 0:
            lines.append(f'  No strongly discriminative features (all |d| < {MIN_EFFECT_SIZE})')
            lines.append(f'')
            continue

        lines.append(f'  **Entry conditions (if/then):**')
        lines.append(f'  ```')

        rule_parts = []
        for _, row in top.iterrows():
            condition, note, strength, d = format_rule(row)
            lines.append(f'    {condition}  # {strength} d={d:+.2f} {note}')
            rule_parts.append({
                'feature': row['feature'],
                'condition': condition,
                'cohen_d': d,
                'p10': row['group_p10'],
                'p50': row['group_p50'],
                'p90': row['group_p90'],
            })

        lines.append(f'  ```')
        lines.append(f'')

        # TF alignment summary
        tf_features = defaultdict(list)
        for _, row in top.iterrows():
            for tf in TF_ORDER:
                if row['feature'].startswith(tf + '_'):
                    tf_features[tf].append((row['feature'], row['cohen_d']))
                    break

        if tf_features:
            lines.append(f'  **TF alignment:**')
            for tf in TF_ORDER:
                if tf in tf_features:
                    feats = tf_features[tf]
                    feat_str = ', '.join(f'{f.split("_",1)[1]}={d:+.1f}' for f, d in feats)
                    lines.append(f'    {tf:>4}: {feat_str}')
            lines.append(f'')

        all_rules[group] = rule_parts

    # Summary: which features appear most across groups?
    lines.append(f'## Feature Importance Across All Groups')
    lines.append(f'')
    feat_counts = defaultdict(list)
    for group, rules in all_rules.items():
        for r in rules:
            feat_counts[r['feature']].append((group, r['cohen_d']))

    feat_freq = [(feat, len(groups_list), np.mean([abs(d) for _, d in groups_list]))
                 for feat, groups_list in feat_counts.items()]
    feat_freq.sort(key=lambda x: (-x[1], -x[2]))

    lines.append(f'| Feature | Groups | Avg |d| | Physics |')
    lines.append(f'|---------|--------|---------|---------|')
    for feat, n_groups, avg_d in feat_freq[:20]:
        # Infer physics meaning
        physics = ''
        if 'z_se' in feat:
            physics = 'band position'
        elif 'dmi' in feat:
            physics = 'trend direction/strength'
        elif 'vr' in feat or 'variance' in feat:
            physics = 'regime (trending vs reverting)'
        elif 'velocity' in feat:
            physics = 'rate of change'
        elif 'accel' in feat:
            physics = 'momentum change (chop)'
        elif 'vol' in feat:
            physics = 'volume conviction'
        elif 'hurst' in feat:
            physics = 'persistence'
        elif 'reversion' in feat:
            physics = 'reversion probability'
        elif 'range' in feat:
            physics = 'volatility/risk'
        elif 'time' in feat:
            physics = 'time of day'
        elif 'wick' in feat:
            physics = 'rejection (wick = indecision)'
        elif 'dir_vol' in feat:
            physics = 'directional volume'

        lines.append(f'| {feat} | {n_groups} | {avg_d:.2f} | {physics} |')

    # Save
    os.makedirs('reports/findings', exist_ok=True)
    report_path = f'reports/findings/physics_rules_{date_str.replace("-","")}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nReport: {report_path}')

    # Print summary
    print(f'\n--- SUMMARY ---')
    print(f'  Groups with rules: {len(all_rules)}')
    total_features = set()
    for rules in all_rules.values():
        for r in rules:
            total_features.add(r['feature'])
    print(f'  Unique features used: {len(total_features)} of {len(feature_cols)}')
    print(f'  Top 5 features across groups:')
    for feat, n_groups, avg_d in feat_freq[:5]:
        print(f'    {feat:<30} appears in {n_groups} groups, avg |d|={avg_d:.2f}')


if __name__ == '__main__':
    main()
