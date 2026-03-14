#!/usr/bin/env python
"""
Shape Primitive Report — reads shape_primitives.pkl and writes a text report.

Usage:
    python tools/shape_primitive_report.py
    python tools/shape_primitive_report.py --input checkpoints/shape_primitives.pkl
"""

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.shape_primitive_builder import ShapePrimitive, ShapePrimitiveLibrary  # noqa: F401, E402


def generate_report(pkl_path: str) -> str:
    with open(pkl_path, 'rb') as f:
        lib = pickle.load(f)

    lines = []

    def w(line=''):
        lines.append(line)

    w('=' * 70)
    w('  SHAPE PRIMITIVE LIBRARY REPORT')
    w('=' * 70)
    w(f'  Source:          {pkl_path}')
    w(f'  Created:         {lib.created_at}')
    w(f'  Version:         {lib.version}')
    w(f'  Total seeds:     {lib.n_total_seeds:,}')
    w(f'  Clustered:       {lib.n_clustered_seeds:,} ({lib.n_clustered_seeds / max(lib.n_total_seeds, 1) * 100:.1f}%)')
    w(f'  Noise:           {lib.n_noise_seeds:,} ({lib.n_noise_seeds / max(lib.n_total_seeds, 1) * 100:.1f}%)')
    w(f'  Primitives:      {len(lib.primitives)}')

    # UMAP / HDBSCAN params
    w(f'\n  UMAP params:     {lib.umap_params}')
    w(f'  HDBSCAN params:  {lib.hdbscan_params}')

    # ZigZag params per TF
    w(f'\n  ZigZag params:')
    for tf, params in sorted(lib.tf_params.items(), key=lambda x: x[0]):
        w(f'    {tf:5s}: min_rev={params["min_reversal"]:4d}t  min_bars={params["min_bars"]}  max_bars={params["max_bars"]}')

    # ── Aggregate stats ──
    w(f'\n{"=" * 70}')
    w('  AGGREGATE STATISTICS')
    w(f'{"=" * 70}')

    # By TF
    tf_totals = defaultdict(int)
    for p in lib.primitives:
        for tf, cnt in p.tf_distribution.items():
            tf_totals[tf] += cnt
    w(f'\n  Members by TF:')
    for tf, cnt in sorted(tf_totals.items(), key=lambda x: -x[1]):
        w(f'    {tf:5s}: {cnt:6,d}')

    # By shape
    shape_totals = defaultdict(int)
    for p in lib.primitives:
        for sh, cnt in p.shape_distribution.items():
            shape_totals[sh] += cnt
    w(f'\n  Members by shape:')
    for sh, cnt in sorted(shape_totals.items(), key=lambda x: -x[1]):
        w(f'    {sh:22s}: {cnt:6,d}')

    # By quality tier
    tier_counts = Counter(p.quality_tier_label for p in lib.primitives)
    w(f'\n  Primitives by quality tier:')
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'NOISE', '']:
        if tier in tier_counts:
            label = tier if tier else 'UNSCORED'
            w(f'    {label:8s}: {tier_counts[tier]}')

    # Direction bias
    n_long_biased = sum(1 for p in lib.primitives if p.direction_bias > 0.6)
    n_short_biased = sum(1 for p in lib.primitives if p.direction_bias < 0.4)
    n_neutral = len(lib.primitives) - n_long_biased - n_short_biased
    w(f'\n  Direction bias:')
    w(f'    LONG biased (>60%):   {n_long_biased}')
    w(f'    SHORT biased (<40%):  {n_short_biased}')
    w(f'    Neutral (40-60%):     {n_neutral}')

    # ── Per-primitive detail ──
    w(f'\n{"=" * 70}')
    w('  PRIMITIVE DETAIL (sorted by member count)')
    w(f'{"=" * 70}')
    w(f'  {"#":>3s} {"Members":>7s} {"TF":>4s} {"Shape":>16s} {"Tier":>6s} {"Q":>5s} '
      f'{"Dir":>5s} {"Bias":>5s} {"R2":>6s} {"MFE":>6s} {"Dur":>6s} {"TF Dist":>30s}')
    w(f'  {"-"*3} {"-"*7} {"-"*4} {"-"*16} {"-"*6} {"-"*5} '
      f'{"-"*5} {"-"*5} {"-"*6} {"-"*6} {"-"*6} {"-"*30}')

    for p in sorted(lib.primitives, key=lambda x: x.n_members, reverse=True):
        bias_pct = max(p.direction_bias, 1 - p.direction_bias) * 100
        dir_label = 'LONG' if p.direction_bias > 0.5 else 'SHORT'

        # Compact TF distribution
        tf_str = ', '.join(f'{tf}={cnt}' for tf, cnt in
                           sorted(p.tf_distribution.items(), key=lambda x: -x[1])[:4])

        w(f'  {p.primitive_id:3d} {p.n_members:7,d} {p.dominant_tf:>4s} '
          f'{p.dominant_shape:>16s} {p.quality_tier_label:>6s} {p.mean_quality_score:5.2f} '
          f'{dir_label:>5s} {bias_pct:4.0f}% {p.shape_r2:6.3f} '
          f'{p.mean_mfe_ticks:5.0f}t {p.mean_duration_mins:5.1f}m '
          f'{tf_str}')

    # ── Shape distribution per primitive ──
    w(f'\n{"=" * 70}')
    w('  SHAPE BREAKDOWN PER PRIMITIVE')
    w(f'{"=" * 70}')

    for p in sorted(lib.primitives, key=lambda x: x.n_members, reverse=True):
        shape_str = ', '.join(f'{sh}={cnt}' for sh, cnt in
                              sorted(p.shape_distribution.items(), key=lambda x: -x[1]))
        w(f'  #{p.primitive_id:3d} ({p.n_members:,d}): {shape_str}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Shape Primitive Report')
    parser.add_argument('--input', default='checkpoints/shape_primitives.pkl',
                        help='Path to shape_primitives.pkl')
    parser.add_argument('--output', default=None,
                        help='Output report path (default: reports/findings/shape_primitives_report.txt)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"  ERROR: {args.input} not found")
        return

    report = generate_report(args.input)
    print(report)

    out_path = args.output or os.path.join(
        'reports', 'findings',
        f'shape_primitives_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n  Report saved: {out_path}')


if __name__ == '__main__':
    main()
