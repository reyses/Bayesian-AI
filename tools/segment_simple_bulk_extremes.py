"""Bulk-segment all days with simple-shapes-only segmenter, find extremes.

Walks all 345 days, counts phrases per day, identifies the 5 days with
MOST phrases and 5 days with LEAST phrases. Renders charts for those 10
days so we can validate the segmenter behavior at both extremes.

USAGE
    python tools/segment_simple_bulk_extremes.py
    python tools/segment_simple_bulk_extremes.py --threshold 0.85 --top-n 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_simple_shapes import segment_day_simple, render_chart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--top-n', type=int, default=5,
                    help='Top N most + Bottom N least phrase-count days to render')
    ap.add_argument('--out-dir-counts',
                    default='reports/findings/segments/simple_bulk')
    ap.add_argument('--out-dir-charts',
                    default='chart/segments/simple/extremes')
    args = ap.parse_args()

    is_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2025_*.parquet'))
    oos_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2026_*.parquet'))
    all_days = ([(os.path.basename(p).replace('.parquet', ''), 'IS') for p in is_paths]
                + [(os.path.basename(p).replace('.parquet', ''), 'OOS') for p in oos_paths])
    print(f'Total days: {len(all_days)} ({len(is_paths)} IS, {len(oos_paths)} OOS)')

    rows = []
    failed = []
    for day, split in tqdm(all_days, desc='segmenting'):
        try:
            h = segment_day_simple(day, threshold=args.threshold)
        except Exception as e:
            failed.append((day, str(e)))
            continue
        if not h:
            failed.append((day, 'empty hierarchy'))
            continue
        # Count shapes
        from collections import Counter
        shape_counts = Counter(p['shape'] for p in h['phrases'])
        rows.append({
            'day': day, 'split': split,
            'n_phrases': h['n_phrases'],
            'n_motifs': h['n_motifs'],
            **{f'phrase_{s}': n for s, n in shape_counts.items()},
        })

    df = pd.DataFrame(rows).fillna(0)
    print(f'\nProcessed: {len(df)} days; failed: {len(failed)}')

    os.makedirs(args.out_dir_counts, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir_counts, 'per_day_counts.csv'), index=False)
    print(f'Wrote per-day counts CSV')

    # Stats
    print(f'\nPhrases-per-day distribution:')
    print(f'  min:    {df["n_phrases"].min()}')
    print(f'  q10:    {df["n_phrases"].quantile(0.10):.1f}')
    print(f'  median: {df["n_phrases"].median()}')
    print(f'  mean:   {df["n_phrases"].mean():.1f}')
    print(f'  q90:    {df["n_phrases"].quantile(0.90):.1f}')
    print(f'  max:    {df["n_phrases"].max()}')

    # Top N most
    most = df.nlargest(args.top_n, 'n_phrases')
    print(f'\nTop {args.top_n} MOST phrases:')
    print(most[['day', 'split', 'n_phrases', 'n_motifs']].to_string(index=False))

    # Bottom N least
    least = df.nsmallest(args.top_n, 'n_phrases')
    print(f'\nBottom {args.top_n} LEAST phrases:')
    print(least[['day', 'split', 'n_phrases', 'n_motifs']].to_string(index=False))

    # Render charts
    os.makedirs(args.out_dir_charts, exist_ok=True)
    print(f'\nRendering {2 * args.top_n} extreme-case charts...')
    for label, sub in [('most', most), ('least', least)]:
        for _, r in sub.iterrows():
            day = r['day']
            try:
                h = segment_day_simple(day, threshold=args.threshold)
            except Exception:
                continue
            if not h:
                continue
            out = os.path.join(args.out_dir_charts,
                                f'{label}_{int(r["n_phrases"])}p_{day}.png')
            render_chart(day, h, out)
            print(f'  {label}: {day}  ({int(r["n_phrases"])} phrases)  -> {out}')

    if failed:
        print(f'\n[warn] {len(failed)} days failed:')
        for d, e in failed[:10]:
            print(f'  {d}: {e}')


if __name__ == '__main__':
    main()
