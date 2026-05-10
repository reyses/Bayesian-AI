"""Bulk-segment all IS+OOS days into motifs + melodies; emit population census.

Walks every day in DATA/ATLAS/FEATURES_5s_v2/L0/, runs segment_day from
segment_day_motif_melody.py, aggregates ALL motifs into one CSV and ALL
melodies into another. Output is the segment population census, analogous
to band_touch_aggregation.py but for hierarchical segments instead of
1h-HL macro events.

Usage:
    python tools/segment_all_days.py
    python tools/segment_all_days.py --days 2025_01_*  --skip-existing
    python tools/segment_all_days.py --min-motif-min 20 --min-melody-min 5
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

from tools.segment_day_motif_melody import segment_day


OUTPUT_DIR = 'reports/findings/segments'


def _resolve_days(filter_glob: str = None) -> list[tuple[str, str]]:
    paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/*.parquet'))
    if filter_glob:
        paths = [p for p in paths if glob.fnmatch.fnmatch(os.path.basename(p), filter_glob + '.parquet')]
    out = []
    for p in paths:
        d = os.path.basename(p).replace('.parquet', '')
        if d.startswith('2025_'):
            out.append((d, 'IS'))
        elif d.startswith('2026_'):
            out.append((d, 'OOS'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', default=None, help='glob filter (e.g. 2025_01_*)')
    ap.add_argument('--motif-tf', default='15m')
    ap.add_argument('--melody-tf', default='5m')
    ap.add_argument('--min-motif-min', type=float, default=30.0)
    ap.add_argument('--min-melody-min', type=float, default=5.0)
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip days that already have a per-day JSON')
    args = ap.parse_args()

    days = _resolve_days(args.days)
    print(f'Bulk segmenting {len(days)} days '
          f'({sum(1 for _,s in days if s=="IS")} IS, '
          f'{sum(1 for _,s in days if s=="OOS")} OOS)')
    print(f'  motif_tf={args.motif_tf} (>={args.min_motif_min}min), '
          f'melody_tf={args.melody_tf} (>={args.min_melody_min}min)')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'per_day'), exist_ok=True)

    all_motifs = []
    all_melodies = []
    themes = []
    skipped = []
    failed = []

    for day, split in tqdm(days, desc='segmenting'):
        per_day_path = os.path.join(OUTPUT_DIR, 'per_day', f'{day}.json')
        if args.skip_existing and os.path.exists(per_day_path):
            try:
                with open(per_day_path) as f:
                    hierarchy = json.load(f)
                skipped.append(day)
            except Exception:
                hierarchy = None
        else:
            try:
                hierarchy = segment_day(day,
                                        motif_tf=args.motif_tf,
                                        melody_tf=args.melody_tf,
                                        min_motif_min=args.min_motif_min,
                                        min_melody_min=args.min_melody_min)
            except Exception as e:
                failed.append((day, str(e)))
                continue
            if not hierarchy:
                failed.append((day, 'empty hierarchy'))
                continue
            with open(per_day_path, 'w') as f:
                json.dump(hierarchy, f, indent=2)

        if not hierarchy:
            continue
        th = hierarchy['theme']
        th['split'] = split
        themes.append(th)

        for motif in hierarchy['motifs']:
            row = {k: v for k, v in motif.items() if k != 'melodies'}
            row['day'] = day
            row['split'] = split
            all_motifs.append(row)
            for mel in motif['melodies']:
                mr = dict(mel)
                mr['day'] = day
                mr['split'] = split
                all_melodies.append(mr)

    # ── Write aggregates ───────────────────────────────────────
    motif_df = pd.DataFrame(all_motifs)
    melody_df = pd.DataFrame(all_melodies)
    theme_df = pd.DataFrame(themes)

    motif_df.to_csv(os.path.join(OUTPUT_DIR, 'all_motifs.csv'), index=False)
    melody_df.to_csv(os.path.join(OUTPUT_DIR, 'all_melodies.csv'), index=False)
    theme_df.to_csv(os.path.join(OUTPUT_DIR, 'all_themes.csv'), index=False)
    print(f'\nWrote {OUTPUT_DIR}/all_motifs.csv  ({len(motif_df)} motifs)')
    print(f'Wrote {OUTPUT_DIR}/all_melodies.csv ({len(melody_df)} melodies)')
    print(f'Wrote {OUTPUT_DIR}/all_themes.csv  ({len(theme_df)} themes)')

    # ── Population summary ──────────────────────────────────────
    md = ['# Segment population census', '',
          f'_Generated {datetime.now().isoformat()}_', '',
          f'- Days processed: {len(days)} ({len(themes)} succeeded, '
          f'{len(failed)} failed, {len(skipped)} skipped from cache)',
          f'- Motifs total:   {len(motif_df)}',
          f'- Melodies total: {len(melody_df)}',
          f'- Motif TF: {args.motif_tf}  min_motif_min: {args.min_motif_min}',
          f'- Melody TF: {args.melody_tf}  min_melody_min: {args.min_melody_min}',
          '']

    if not motif_df.empty:
        md += ['## Motif shape distribution', '',
               '```',
               'shape_class           n_motifs   pct      mean_len   mean_slope   '
               'mean_r2adj   mean_pk_z',
               '-' * 95]
        shape_cnts = motif_df['shape_class'].value_counts()
        for sh, n in shape_cnts.items():
            sub = motif_df[motif_df['shape_class'] == sh]
            md.append(
                f'{sh:<22s} {n:>6d}   {100*n/len(motif_df):>5.1f}%  '
                f'{sub["length_min"].mean():>8.1f}m  '
                f'{sub["slope_pts_per_min"].mean():>+10.3f}  '
                f'{sub["r2adj"].mean():>10.2f}   '
                f'{sub["peak_abs_z"].mean():>8.2f}')
        md.append('```')
        md.append('')

        md += ['## Motif IS vs OOS counts by shape', '',
               '```',
               'shape_class           IS        OOS       IS/d     OOS/d   sign-stable',
               '-' * 80]
        n_is_days = (theme_df['split'] == 'IS').sum() if not theme_df.empty else 1
        n_oos_days = (theme_df['split'] == 'OOS').sum() if not theme_df.empty else 1
        for sh in shape_cnts.index:
            sub = motif_df[motif_df['shape_class'] == sh]
            n_is = (sub['split'] == 'IS').sum()
            n_oos = (sub['split'] == 'OOS').sum()
            r_is = n_is / max(n_is_days, 1)
            r_oos = n_oos / max(n_oos_days, 1)
            stable = abs(r_is - r_oos) / max(r_is, r_oos, 1e-9) < 0.30
            tag = 'YES' if stable else 'no'
            md.append(
                f'{sh:<22s} {n_is:>6d}   {n_oos:>6d}   {r_is:>6.2f}  {r_oos:>6.2f}  '
                f'{tag}')
        md.append('```')
        md.append('')

    if not melody_df.empty:
        md += ['## Melody shape distribution', '',
               '```',
               'shape_class           n_melodies pct      mean_len   mean_slope   '
               'mean_r2adj   mean_pk_z',
               '-' * 95]
        shape_cnts = melody_df['shape_class'].value_counts()
        for sh, n in shape_cnts.items():
            sub = melody_df[melody_df['shape_class'] == sh]
            md.append(
                f'{sh:<22s} {n:>6d}   {100*n/len(melody_df):>5.1f}%  '
                f'{sub["length_min"].mean():>8.1f}m  '
                f'{sub["slope_pts_per_min"].mean():>+10.3f}  '
                f'{sub["r2adj"].mean():>10.2f}   '
                f'{sub["peak_abs_z"].mean():>8.2f}')
        md.append('```')
        md.append('')

    md += ['## Notes', '',
           '- Motifs = 15m-CRM-anchored macro segments (>= min_motif_min)',
           '- Melodies = 5m-CRM-anchored micro sub-segments NESTED inside motifs',
           '- Each segment carries its segment_chord (slope, sigma_rank, r2adj, '
           'shape_class, length, peak_z, ...) computed lookahead-clean over the '
           'segment span',
           '- Sign-stable column flags shapes whose IS rate-per-day is within 30% '
           'of OOS rate-per-day; non-stable shapes may not generalize',
           '']

    md_path = os.path.join(OUTPUT_DIR, 'summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f'Wrote {md_path}')

    if failed:
        print(f'\n[warn] {len(failed)} days failed:')
        for d, e in failed[:10]:
            print(f'  {d}: {e}')


if __name__ == '__main__':
    main()
