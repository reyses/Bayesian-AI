"""Bulk multi-level segmentation: phrases (15m) -> motifs (5m) ->
sub_motifs (1m) -> measures (15s) -> notes (5s).

Parametric pipeline. Outputs one CSV per LEVEL plus per-day JSONs.

Output layout:
    reports/findings/segments/simple_bulk_v2/
        all_phrases.csv     (one row per 15m phrase)
        all_motifs.csv      (one row per 5m motif)
        all_sub_motifs.csv  (one row per 1m sub_motif)
        all_measures.csv    (one row per 15s measure)
        all_notes.csv       (one row per 5s note)
        themes.csv          (one row per day)
        per_day/<day>.json  (full hierarchy)

Each row carries: day, split, level, parent_chain (idx path), shape, skew,
r, mean_sigma, length_min, start_ts, end_ts, start_i, end_i.

USAGE
    python tools/segment_simple_bulk_v2.py
    python tools/segment_simple_bulk_v2.py --threshold 0.85 --max-depth 5
    python tools/segment_simple_bulk_v2.py --max-depth 3   # only down to sub_motifs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_simple_shapes import segment_day_simple, LEVELS


OUT_DIR = 'reports/findings/segments/simple_bulk_v2'


def _seg_to_row(seg: dict, day: str, split: str, level: str,
                 idx: int, parent_chain: tuple) -> dict:
    return {
        'day':              day,
        'split':            split,
        'level':            level,
        'idx':              idx,
        'parent_chain':     '/'.join(str(x) for x in parent_chain) if parent_chain else '',
        'shape':            seg['shape'],
        'shape_class':      seg['shape'],
        'skew':             seg.get('skew', 'NONE'),
        'r':                seg.get('r', float('nan')),
        'mean_sigma':       seg.get('mean_sigma', float('nan')),
        'start_ts':         seg.get('start_ts', None),
        'end_ts':           seg.get('end_ts', None),
        'start_i':          seg.get('start_i', None),
        'end_i':            seg.get('end_i', None),
        'length_min':       seg.get('length_min', float('nan')),
    }


def _walk_hierarchy(node: dict, day: str, split: str,
                    level_idx: int, parent_chain: tuple,
                    rows_per_level: dict) -> None:
    """Recursively walk the hierarchy, appending rows to rows_per_level
    keyed by level name."""
    if level_idx >= len(LEVELS):
        return
    level_name, child_attr, _, _, _ = LEVELS[level_idx]
    children = node.get(child_attr if level_idx > 0 else 'phrases', [])
    for idx, child in enumerate(children):
        rows_per_level[level_name].append(
            _seg_to_row(child, day, split, level_name, idx, parent_chain))
        _walk_hierarchy(child, day, split, level_idx + 1,
                        parent_chain + (idx,), rows_per_level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--max-depth', type=int, default=5,
                    help='1=phrases only, 5=down to 5s notes')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'per_day'), exist_ok=True)

    is_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2025_*.parquet'))
    oos_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2026_*.parquet'))
    all_days = ([(os.path.basename(p).replace('.parquet', ''), 'IS') for p in is_paths]
                + [(os.path.basename(p).replace('.parquet', ''), 'OOS') for p in oos_paths])
    print(f'Bulk segmenting {len(all_days)} days '
          f'(max_depth={args.max_depth}, threshold={args.threshold})')

    # rows_per_level[level_name] = list of row dicts
    levels_used = LEVELS[:args.max_depth]
    rows_per_level = {lvl[0]: [] for lvl in levels_used}
    theme_rows = []
    failed = []

    for day, split in tqdm(all_days, desc='segmenting'):
        try:
            h = segment_day_simple(day, threshold=args.threshold,
                                    max_depth=args.max_depth)
        except Exception as e:
            failed.append((day, str(e)))
            continue
        if not h:
            failed.append((day, 'empty'))
            continue

        with open(os.path.join(args.out_dir, 'per_day', f'{day}.json'), 'w') as f:
            json.dump(h, f, indent=2)

        theme_rows.append({
            'day': day, 'split': split,
            'date_iso': h.get('date_iso', day.replace('_', '-')),
            'threshold': h['threshold'],
            'max_depth': h['max_depth'],
            **{k: v for k, v in h.items()
               if k.startswith('n_')},
        })

        # Top level (phrases): index from h['phrases']
        for p_idx, phrase in enumerate(h['phrases']):
            rows_per_level['phrase'].append(
                _seg_to_row(phrase, day, split, 'phrase', p_idx, ()))
            # Walk sub-levels
            _walk_hierarchy(phrase, day, split, 1, (p_idx,), rows_per_level)

    print(f'\nProcessed {len(theme_rows)} days successfully ({len(failed)} failed)')

    # Write CSVs
    pd.DataFrame(theme_rows).to_csv(
        os.path.join(args.out_dir, 'themes.csv'), index=False)
    for level_name, rows in rows_per_level.items():
        df = pd.DataFrame(rows)
        out_csv = os.path.join(args.out_dir, f'all_{level_name}s.csv')
        df.to_csv(out_csv, index=False)
        print(f'  {level_name:<10s} -> {len(rows):,} rows  ({out_csv})')

    # Per-level shape distribution
    print(f'\nShape distributions per level:')
    for level_name, rows in rows_per_level.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        top = df['shape'].value_counts().head(8)
        print(f'\n  [{level_name}] n={len(rows):,}')
        for sh, n in top.items():
            print(f'    {sh:<25s} {n:>8d}  ({100*n/len(rows):.1f}%)')

    if failed:
        print(f'\n[warn] {len(failed)} days failed:')
        for d, e in failed[:10]:
            print(f'  {d}: {e}')


if __name__ == '__main__':
    main()
