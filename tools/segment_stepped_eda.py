"""Stepped EDA across all 5 segmentation levels.

For each level (phrase, motif, sub_motif, measure, note) computes:
    1. shape distribution (count + pct)
    2. length distribution (min/p10/median/mean/p90/max in minutes)
    3. skew distribution per shape (BACK/FRONT/SYMMETRIC)
    4. Pearson r distribution per shape
    5. IS vs OOS comparison
    6. Markov transitions: P(next_shape | current_shape) within each day
    7. Parent->child conditional: P(child_shape | parent_shape)

Outputs (one per level):
    reports/findings/segments/stepped_eda/<level>_shape_dist.csv
    reports/findings/segments/stepped_eda/<level>_length_dist.csv
    reports/findings/segments/stepped_eda/<level>_skew_dist.csv
    reports/findings/segments/stepped_eda/<level>_r_dist.csv
    reports/findings/segments/stepped_eda/<level>_split_dist.csv
    reports/findings/segments/stepped_eda/<level>_markov.csv
    reports/findings/segments/stepped_eda/<level>__given__<parent>.csv
    reports/findings/segments/stepped_eda/<level>_overview.png

USAGE
    python tools/segment_stepped_eda.py
    python tools/segment_stepped_eda.py --in-dir reports/findings/segments/simple_bulk_v2 \
                                          --out-dir reports/findings/segments/stepped_eda
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEVELS = ['phrase', 'motif', 'sub_motif', 'measure', 'note']
PARENT = {'phrase': None, 'motif': 'phrase', 'sub_motif': 'motif',
          'measure': 'sub_motif', 'note': 'measure'}


def shape_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df['shape'].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    out = pd.DataFrame({'count': counts, 'pct': pct}).reset_index().rename(
        columns={'index': 'shape'})
    return out


def length_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sh, sub in df.groupby('shape'):
        L = sub['length_min'].dropna().values
        if len(L) == 0:
            continue
        rows.append({
            'shape': sh, 'n': len(L),
            'min':    round(float(np.min(L)), 2),
            'p10':    round(float(np.percentile(L, 10)), 2),
            'median': round(float(np.median(L)), 2),
            'mean':   round(float(np.mean(L)), 2),
            'p90':    round(float(np.percentile(L, 90)), 2),
            'max':    round(float(np.max(L)), 2),
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False)


def skew_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if 'skew' not in df.columns:
        return pd.DataFrame()
    rows = []
    for sh, sub in df.groupby('shape'):
        skew_counts = sub['skew'].value_counts()
        total = len(sub)
        rows.append({
            'shape': sh, 'n': total,
            'BACK_SKEWED':  int(skew_counts.get('BACK_SKEWED', 0)),
            'FRONT_SKEWED': int(skew_counts.get('FRONT_SKEWED', 0)),
            'SYMMETRIC':    int(skew_counts.get('SYMMETRIC', 0)),
            'NONE':         int(skew_counts.get('NONE', 0)),
            'pct_BACK':     round(100 * skew_counts.get('BACK_SKEWED', 0) / total, 1),
            'pct_FRONT':    round(100 * skew_counts.get('FRONT_SKEWED', 0) / total, 1),
            'pct_SYM':      round(100 * skew_counts.get('SYMMETRIC', 0) / total, 1),
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False)


def r_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sh, sub in df.groupby('shape'):
        R = sub['r'].dropna().values
        if len(R) == 0:
            continue
        rows.append({
            'shape': sh, 'n': len(R),
            'min':    round(float(np.min(R)), 3),
            'p10':    round(float(np.percentile(R, 10)), 3),
            'median': round(float(np.median(R)), 3),
            'mean':   round(float(np.mean(R)), 3),
            'p90':    round(float(np.percentile(R, 90)), 3),
            'max':    round(float(np.max(R)), 3),
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False)


def split_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sh in sorted(df['shape'].unique()):
        sub = df[df['shape'] == sh]
        is_n = (sub['split'] == 'IS').sum()
        oos_n = (sub['split'] == 'OOS').sum()
        is_total = (df['split'] == 'IS').sum()
        oos_total = (df['split'] == 'OOS').sum()
        rows.append({
            'shape': sh,
            'IS_n': int(is_n),
            'OOS_n': int(oos_n),
            'IS_pct': round(100 * is_n / max(is_total, 1), 2),
            'OOS_pct': round(100 * oos_n / max(oos_total, 1), 2),
            'IS_minus_OOS_pct': round(100 * is_n / max(is_total, 1)
                                       - 100 * oos_n / max(oos_total, 1), 2),
        })
    return pd.DataFrame(rows)


def markov_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """For each (day, parent_chain ancestors), compute next-shape transitions
    sorted by start_ts. Use idx ordering within parent."""
    df = df.copy()
    df['parent_chain'] = df['parent_chain'].fillna('').astype(str)
    df['idx'] = df['idx'].astype(int)
    df = df.sort_values(['day', 'parent_chain', 'idx']).reset_index(drop=True)
    transitions = defaultdict(int)
    for (day, parent), sub in df.groupby(['day', 'parent_chain']):
        seq = sub['shape'].tolist()
        for i in range(len(seq) - 1):
            transitions[(seq[i], seq[i + 1])] += 1
    rows = []
    row_totals = defaultdict(int)
    for (a, _), c in transitions.items():
        row_totals[a] += c
    for (a, b), c in transitions.items():
        rows.append({
            'from': a, 'to': b, 'count': c,
            'p_to_given_from': round(c / row_totals[a], 4),
        })
    if not rows:
        return pd.DataFrame(columns=['from', 'to', 'count', 'p_to_given_from'])
    return pd.DataFrame(rows).sort_values(['from', 'count'], ascending=[True, False])


def parent_child_conditional(child_df: pd.DataFrame, parent_df: pd.DataFrame,
                              parent_level: str) -> pd.DataFrame:
    """For each parent shape, compute child shape distribution.

    parent_df['parent_chain'] is the path of ancestors (excluding self).
    parent_df['idx'] is index within its own parent (or in the day, for phrase).
    A parent is uniquely identified by (day, parent_chain, idx).

    A child's IMMEDIATE parent has key:
        day == child.day
        parent_chain == child.parent_chain[:-1] (split by '/')
        idx == int(child.parent_chain[-1])
    """
    parent_df = parent_df.copy()
    parent_df['parent_chain'] = parent_df['parent_chain'].fillna('').astype(str)
    parent_df['idx'] = parent_df['idx'].astype(int)
    parent_lookup = parent_df.set_index(
        ['day', 'parent_chain', 'idx']
    )['shape'].to_dict()
    child_df = child_df.copy()
    child_df['parent_chain'] = child_df['parent_chain'].fillna('').astype(str)

    def get_parent_shape(row):
        pc = row['parent_chain']
        if not pc:
            return None
        toks = pc.split('/')
        parent_chain_above = '/'.join(toks[:-1])
        try:
            parent_idx = int(toks[-1])
        except ValueError:
            return None
        return parent_lookup.get((row['day'], parent_chain_above, parent_idx))

    child_df['parent_shape'] = child_df.apply(get_parent_shape, axis=1)
    rows = []
    for parent_shape, sub in child_df.groupby('parent_shape', dropna=True):
        if pd.isna(parent_shape):
            continue
        total = len(sub)
        for child_shape, n in sub['shape'].value_counts().items():
            rows.append({
                'parent_shape': parent_shape,
                'child_shape': child_shape,
                'n': int(n),
                'p_child_given_parent': round(n / total, 4),
                'parent_n_total': total,
            })
    if not rows:
        return pd.DataFrame(columns=['parent_shape', 'child_shape', 'n',
                                      'p_child_given_parent', 'parent_n_total'])
    return (pd.DataFrame(rows)
              .sort_values(['parent_shape', 'n'], ascending=[True, False]))


def make_overview_chart(level: str, shape_dist: pd.DataFrame,
                         length_dist: pd.DataFrame, out_png: str):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    # Bar: shape distribution
    top = shape_dist.head(15)
    axes[0].barh(top['shape'][::-1], top['count'][::-1], color='#1E88E5')
    axes[0].set_xlabel('count')
    axes[0].set_title(f'[{level}] shape distribution (n={shape_dist["count"].sum():,})')
    axes[0].grid(True, axis='x', alpha=0.3)
    # Box: length distribution per top-10 shape
    top_shapes = shape_dist.head(10)['shape'].tolist()
    data = []
    for sh in top_shapes:
        row = length_dist[length_dist['shape'] == sh]
        if not row.empty:
            data.append({
                'shape': sh,
                'p10': float(row['p10'].iloc[0]),
                'median': float(row['median'].iloc[0]),
                'mean': float(row['mean'].iloc[0]),
                'p90': float(row['p90'].iloc[0]),
            })
    if data:
        ddf = pd.DataFrame(data)
        x = np.arange(len(ddf))
        axes[1].fill_between(x, ddf['p10'], ddf['p90'], alpha=0.25, color='#1E88E5',
                              label='p10-p90')
        axes[1].plot(x, ddf['median'], 'o-', color='#0D47A1', label='median')
        axes[1].plot(x, ddf['mean'], 's--', color='#FB8C00', label='mean')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ddf['shape'], rotation=30, ha='right', fontsize=9)
        axes[1].set_ylabel('length (min)')
        axes[1].set_title(f'[{level}] length per shape')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='reports/findings/segments/simple_bulk_v2')
    ap.add_argument('--out-dir', default='reports/findings/segments/stepped_eda')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all level CSVs
    dfs = {}
    for level in LEVELS:
        path = os.path.join(args.in_dir, f'all_{level}s.csv')
        if not os.path.exists(path):
            print(f'  skip [{level}] (no csv at {path})')
            continue
        df = pd.read_csv(path)
        dfs[level] = df
        print(f'  loaded [{level}] {len(df):,} rows from {path}')

    summary_overall = []
    for level, df in dfs.items():
        print(f'\n=== {level} (n={len(df):,}) ===')
        shape_dist = shape_distribution(df)
        length_dist = length_distribution(df)
        skew_dist = skew_distribution(df)
        r_dist = r_distribution(df)
        split_dist = split_distribution(df)

        shape_dist.to_csv(os.path.join(args.out_dir, f'{level}_shape_dist.csv'), index=False)
        length_dist.to_csv(os.path.join(args.out_dir, f'{level}_length_dist.csv'), index=False)
        skew_dist.to_csv(os.path.join(args.out_dir, f'{level}_skew_dist.csv'), index=False)
        r_dist.to_csv(os.path.join(args.out_dir, f'{level}_r_dist.csv'), index=False)
        split_dist.to_csv(os.path.join(args.out_dir, f'{level}_split_dist.csv'), index=False)

        markov = markov_transitions(df)
        markov.to_csv(os.path.join(args.out_dir, f'{level}_markov.csv'), index=False)

        # Parent-child conditional
        parent_lvl = PARENT[level]
        if parent_lvl and parent_lvl in dfs:
            cond = parent_child_conditional(df, dfs[parent_lvl], parent_lvl)
            cond.to_csv(os.path.join(args.out_dir,
                                       f'{level}_given_{parent_lvl}.csv'), index=False)
            print(f'  [{level} | {parent_lvl}] cond table -> {len(cond):,} rows')

        make_overview_chart(level, shape_dist, length_dist,
                              os.path.join(args.out_dir, f'{level}_overview.png'))

        # Top-3 shapes
        top3 = shape_dist.head(3)
        for _, r in top3.iterrows():
            print(f'    {r["shape"]:<24s} n={r["count"]:,} ({r["pct"]}%)')
        summary_overall.append({
            'level': level, 'n': len(df),
            'unique_shapes': df['shape'].nunique(),
            'noise_pct': round(100 * (df['shape'] == 'NOISE').sum() / len(df), 2),
            'flat_pct': round(100 * (df['shape'] == 'FLATLINE').sum() / len(df), 2),
            'mean_length_min': round(df['length_min'].mean(), 2),
            'median_length_min': round(df['length_min'].median(), 2),
        })

    pd.DataFrame(summary_overall).to_csv(
        os.path.join(args.out_dir, 'overall_summary.csv'), index=False)
    print(f'\nOverall summary -> {os.path.join(args.out_dir, "overall_summary.csv")}')


if __name__ == '__main__':
    main()
