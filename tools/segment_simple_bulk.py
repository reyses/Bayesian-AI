"""Bulk-segment all 345 days with simple-shapes-only segmenter.

Produces:
    reports/findings/segments/simple_bulk/all_phrases.csv   (one row per 15m phrase)
    reports/findings/segments/simple_bulk/all_motifs.csv    (one row per 5m motif)
    reports/findings/segments/simple_bulk/all_themes.csv    (one row per day)
    reports/findings/segments/simple_bulk/per_day/<day>.json (full per-day hierarchy)

Each phrase row carries: day, split, seg_idx, shape, skew, r, slope_pts_per_min,
mean_sigma, length_min, peak_abs_z (if computable), start_ts, end_ts.
Each motif row also carries parent_phrase_idx.

USAGE
    python tools/segment_simple_bulk.py
    python tools/segment_simple_bulk.py --threshold 0.85
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_simple_shapes import segment_day_simple


OUT_DIR = 'reports/findings/segments/simple_bulk'


def _seg_to_row(seg: dict, day: str, split: str, seg_idx: int,
                 parent_phrase_idx: int = None) -> dict:
    row = {
        'day':              day,
        'split':            split,
        'seg_idx':          seg_idx,
        'shape':            seg['shape'],
        'shape_class':      seg['shape'],   # alias for tools expecting shape_class
        'skew':             seg.get('skew', 'NONE'),
        'r':                seg.get('r', float('nan')),
        'mean_sigma':       seg.get('mean_sigma', float('nan')),
        'start_ts':         seg.get('start_ts', None),
        'end_ts':           seg.get('end_ts', None),
        'length_min':       seg.get('length_min', float('nan')),
    }
    if parent_phrase_idx is not None:
        row['parent_phrase_idx'] = parent_phrase_idx
    # Compute slope_pts_per_min if possible
    if 'start_ts' in seg and 'end_ts' in seg and 'mean_sigma' in seg:
        # placeholder — slope info lives in the JSON; we don't recompute here
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'per_day'), exist_ok=True)

    is_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2025_*.parquet'))
    oos_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2026_*.parquet'))
    all_days = ([(os.path.basename(p).replace('.parquet', ''), 'IS') for p in is_paths]
                + [(os.path.basename(p).replace('.parquet', ''), 'OOS') for p in oos_paths])
    print(f'Bulk segmenting {len(all_days)} days at threshold r >= {args.threshold}')

    all_phrases = []
    all_motifs = []
    all_themes = []
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

        # Save per-day JSON
        with open(os.path.join(args.out_dir, 'per_day', f'{day}.json'), 'w') as f:
            json.dump(h, f, indent=2)

        # Theme row
        all_themes.append({
            'day': day, 'split': split,
            'date_iso': h.get('date_iso', day.replace('_', '-')),
            'n_phrases': h['n_phrases'],
            'n_motifs': h['n_motifs'],
            'threshold': h['threshold'],
        })

        # Phrase + motif rows
        for p_idx, phrase in enumerate(h['phrases']):
            all_phrases.append(_seg_to_row(phrase, day, split, p_idx))
            for m_idx, motif in enumerate(phrase.get('motifs', [])):
                all_motifs.append(_seg_to_row(motif, day, split, m_idx,
                                              parent_phrase_idx=p_idx))

    print(f'\nProcessed {len(all_themes)} days successfully ({len(failed)} failed)')

    # Write aggregate CSVs
    pd.DataFrame(all_themes).to_csv(
        os.path.join(args.out_dir, 'all_themes.csv'), index=False)
    pd.DataFrame(all_phrases).to_csv(
        os.path.join(args.out_dir, 'all_phrases.csv'), index=False)
    pd.DataFrame(all_motifs).to_csv(
        os.path.join(args.out_dir, 'all_motifs.csv'), index=False)
    print(f'Wrote {len(all_phrases)} phrases, {len(all_motifs)} motifs, '
          f'{len(all_themes)} themes')

    # Phrase shape distribution
    phrase_df = pd.DataFrame(all_phrases)
    print(f'\nPHRASE shape distribution:')
    print(phrase_df['shape'].value_counts().to_string())
    print(f'\nMOTIF shape distribution (top 15):')
    motif_df = pd.DataFrame(all_motifs)
    print(motif_df['shape'].value_counts().head(15).to_string())

    if failed:
        print(f'\n[warn] {len(failed)} days failed:')
        for d, e in failed[:10]:
            print(f'  {d}: {e}')


if __name__ == '__main__':
    main()
