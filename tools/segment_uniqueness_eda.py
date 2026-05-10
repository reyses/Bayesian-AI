"""Pre-EDA: how common vs unique are child-segment patterns within a parent bound?

For each parent segment (e.g. a phrase of shape LINEAR_DOWN), the children
(motifs inside it) form a pattern. We measure how often that pattern repeats
across all parents of the same shape.

Three signature definitions per parent:
    multiset    sorted tuple of UNIQUE child shapes (order-independent)
                 ("what was played", set semantics)
    counts      tuple of (shape, count) sorted by shape
                 ("what was played and how many of each", multiset)
    sequence    ordered tuple of child shapes
                 ("what was played and in what order")

Per parent shape, the analyzer reports:
    n_parents              how many parents have that shape
    n_unique_signatures    distinct signatures observed
    top1_frequency_pct     most common signature's share
    top3_frequency_pct     top 3 signatures' combined share
    entropy_bits           Shannon entropy of signature distribution
    common_signatures      list of top-N signatures with counts

Low entropy + high top1% = the data has REPEATING substructure --&gt; Bayesian
table cells will populate well.
High entropy + low top1% = each parent is idiosyncratic --&gt; table cells stay
thin and we need to either coarsen the signature or accept hierarchical
shrinkage to do the heavy lifting.

Inputs (already produced by segment_all_days.py + oracle_label_segments.py):
    reports/findings/segments/all_motifs_labeled.csv     (phrases at 15m)
    reports/findings/segments/all_melodies_labeled.csv   (motifs at 5m)

USAGE:
    python tools/segment_uniqueness_eda.py
    python tools/segment_uniqueness_eda.py --signature counts --top 5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _signature_multiset(shapes: list[str]) -> tuple:
    return tuple(sorted(set(shapes)))


def _signature_counts(shapes: list[str]) -> tuple:
    counter = Counter(shapes)
    return tuple(sorted(counter.items()))


def _signature_sequence(shapes: list[str]) -> tuple:
    return tuple(shapes)


SIG_FN = {
    'multiset': _signature_multiset,
    'counts':   _signature_counts,
    'sequence': _signature_sequence,
}


def _entropy_bits(freqs: list[int]) -> float:
    n = sum(freqs)
    if n == 0:
        return 0.0
    p = np.array(freqs, dtype=float) / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def analyze(parent_df: pd.DataFrame, child_df: pd.DataFrame,
             parent_shape_col: str = 'shape_class',
             child_shape_col: str = 'shape_class',
             signature_kind: str = 'counts',
             min_parents_to_report: int = 30,
             top_n_signatures: int = 5,
             split: str = 'IS') -> pd.DataFrame:
    """For each parent shape_class compute uniqueness stats over child signatures."""
    if split:
        parent_df = parent_df[parent_df['split'] == split].copy()
        child_df = child_df[child_df['split'] == split].copy()

    sig_fn = SIG_FN[signature_kind]
    rows = []

    # Group parents by shape_class
    for parent_shape, parents_in_shape in parent_df.groupby(parent_shape_col):
        n_parents = len(parents_in_shape)
        if n_parents < min_parents_to_report:
            continue

        signatures = []
        for _, parent in parents_in_shape.iterrows():
            kids = child_df[(child_df['day'] == parent['day']) &
                            (child_df['parent_motif_idx'] == parent['seg_idx'])]
            kids = kids.sort_values('start_ts')
            shapes = kids[child_shape_col].tolist()
            sig = sig_fn(shapes)
            signatures.append(sig)

        sig_counter = Counter(signatures)
        n_unique = len(sig_counter)
        top_signatures = sig_counter.most_common(top_n_signatures)
        top1_count = top_signatures[0][1] if top_signatures else 0
        top3_count = sum(c for _, c in top_signatures[:3])
        entropy = _entropy_bits(list(sig_counter.values()))
        max_entropy = np.log2(n_unique) if n_unique > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        rows.append({
            'parent_shape': parent_shape,
            'n_parents': n_parents,
            'n_unique_signatures': n_unique,
            'unique_pct': round(100 * n_unique / n_parents, 1),
            'top1_pct': round(100 * top1_count / n_parents, 1),
            'top3_pct': round(100 * top3_count / n_parents, 1),
            'entropy_bits': round(entropy, 2),
            'max_entropy_bits': round(max_entropy, 2),
            'normalized_entropy': round(normalized_entropy, 3),
            'top_signatures': top_signatures,
        })

    out = pd.DataFrame(rows).sort_values('n_parents', ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
                    default='reports/findings/segments/all_motifs_labeled.csv')
    ap.add_argument('--motif-csv',
                    default='reports/findings/segments/all_melodies_labeled.csv')
    ap.add_argument('--signature', default='counts',
                    choices=['multiset', 'counts', 'sequence'])
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--out',
                    default='reports/findings/segments/uniqueness_eda.md')
    args = ap.parse_args()

    phrases = pd.read_csv(args.phrase_csv)
    motifs = pd.read_csv(args.motif_csv)
    print(f'Loaded {len(phrases)} phrases, {len(motifs)} motifs')
    print(f'Signature kind: {args.signature}    split: {args.split}')

    if args.split == 'BOTH':
        result_is = analyze(phrases, motifs, signature_kind=args.signature,
                            top_n_signatures=args.top, split='IS')
        result_oos = analyze(phrases, motifs, signature_kind=args.signature,
                             top_n_signatures=args.top, split='OOS')
    else:
        result_is = analyze(phrases, motifs, signature_kind=args.signature,
                            top_n_signatures=args.top, split=args.split)
        result_oos = None

    print()
    cols_to_show = ['parent_shape', 'n_parents', 'n_unique_signatures',
                    'unique_pct', 'top1_pct', 'top3_pct',
                    'entropy_bits', 'max_entropy_bits', 'normalized_entropy']
    print(result_is[cols_to_show].to_string(index=False))

    # Markdown report
    md = ['# Segment uniqueness pre-EDA',
          '',
          f'_Generated {datetime.now().isoformat()}_',
          '',
          f'**Hierarchy bound**: phrase (15m macro segment)',
          f'**Children measured**: motifs (5m segments) inside each phrase',
          f'**Signature kind**: `{args.signature}`',
          f'**Split**: {args.split}',
          '',
          '## How to read this',
          '',
          '- `unique_pct` = (n_unique_signatures / n_parents) * 100',
          '   - 100% = every parent has a different motif sequence (idiosyncratic)',
          '   - low % = patterns repeat --&gt; Bayesian table cells will populate',
          '- `top1_pct` = share of the most common signature among that shape\'s parents',
          '- `entropy_bits` = Shannon entropy of signature distribution',
          '- `normalized_entropy` = entropy / log2(n_unique); 1.0 = uniform, 0.0 = single signature',
          '',
          '## Within-shape uniqueness (IS)',
          '',
          '```',
          result_is[cols_to_show].to_string(index=False),
          '```',
          '',
          '## Top signatures per phrase shape',
          '',
          '```']
    for _, r in result_is.iterrows():
        md.append(f'\n### {r["parent_shape"]}  (n_parents={r["n_parents"]}, '
                  f'unique_pct={r["unique_pct"]}%, '
                  f'top1={r["top1_pct"]}%, '
                  f'norm_H={r["normalized_entropy"]})')
        for sig, cnt in r['top_signatures']:
            pct = 100 * cnt / r['n_parents']
            md.append(f'    {pct:>5.1f}% ({cnt:>4d}/{r["n_parents"]}) {sig}')
    md.append('```')
    md.append('')
    md.append('## Implication for Bayesian table')
    md.append('')
    md.append(
        '- LOW unique_pct + HIGH top1_pct + LOW normalized_entropy = patterns ARE')
    md.append(
        '  generalizable --&gt; Bayesian-table cells populate well from IS data')
    md.append(
        '- HIGH unique_pct + LOW top1_pct + HIGH normalized_entropy = each parent')
    md.append(
        '  is idiosyncratic --&gt; cells stay thin; must rely on hierarchical shrinkage')
    md.append(
        '  or use a coarser signature (multiset > counts > sequence)')
    md.append('')
    md.append('Within-shape entropy ranking tells us WHICH shapes have repeating')
    md.append('substructure (good substrate) vs which are essentially random in their')
    md.append('child compositions (need finer keying or external context).')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(md))
    print(f'\nReport -> {args.out}')


if __name__ == '__main__':
    main()
