"""Absorb small shapes into adjacent large shapes at the 15m level.

User correction 2026-05-10: don't use chord fingerprint, absorb based on
the 15m segment's own direction (sign of slope) and shape neighborhood.

Each small shape's primitive name encodes its direction (UP / DOWN) and
its curve family (LINEAR, EXPONENTIAL, LOGARITHMIC, STEP, BACK_SKEWED,
FRONT_SKEWED, SYMMETRIC_V, ROUNDED_U, oscillators). We absorb based on:

    1. Direction-matching: small UP shapes  -> nearest large UP shape
       small DOWN shapes -> nearest large DOWN shape
    2. Shape-family priority within same direction:
       BACK_SKEWED  -> EXPONENTIAL (similar accelerating-curve)
       FRONT_SKEWED -> LOGARITHMIC (similar exhausting-curve)
       SYMMETRIC_V  -> ROUNDED_U (both reversal patterns; or LINEAR if those small too)
       Oscillators (SINE/DAMPED/EXPAND) -> NOISE (no clear primitive home)
    3. If the preferred-family large shape doesn't exist as a large shape,
       fall back to the largest same-direction shape

Outputs an absorption mapping CSV usable as a preprocessing step.

USAGE
    python tools/small_shape_absorption.py
    python tools/small_shape_absorption.py --min-shape-n 50
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Family-based absorption preferences (no chord fingerprint).
# Each shape maps to a list of fallback shapes in priority order; the first
# shape in the list that ALSO is a "large" shape becomes the absorption target.
ABSORPTION_PREFERENCE = {
    # UP-direction families
    'LINEAR_UP':         ['LINEAR_UP', 'EXPONENTIAL_UP', 'LOGARITHMIC_UP', 'STEP_UP'],
    'EXPONENTIAL_UP':    ['EXPONENTIAL_UP', 'LINEAR_UP', 'LOGARITHMIC_UP', 'STEP_UP'],
    'LOGARITHMIC_UP':    ['LOGARITHMIC_UP', 'LINEAR_UP', 'EXPONENTIAL_UP', 'STEP_UP'],
    'STEP_UP':           ['STEP_UP', 'LINEAR_UP', 'EXPONENTIAL_UP'],
    'BACK_SKEWED_UP':    ['EXPONENTIAL_UP', 'LINEAR_UP', 'STEP_UP', 'LOGARITHMIC_UP'],
    'FRONT_SKEWED_UP':   ['LOGARITHMIC_UP', 'LINEAR_UP', 'EXPONENTIAL_UP'],
    # DOWN-direction families
    'LINEAR_DOWN':       ['LINEAR_DOWN', 'EXPONENTIAL_DOWN', 'LOGARITHMIC_DOWN', 'STEP_DOWN'],
    'EXPONENTIAL_DOWN':  ['EXPONENTIAL_DOWN', 'LINEAR_DOWN', 'LOGARITHMIC_DOWN', 'STEP_DOWN'],
    'LOGARITHMIC_DOWN':  ['LOGARITHMIC_DOWN', 'LINEAR_DOWN', 'EXPONENTIAL_DOWN', 'STEP_DOWN'],
    'STEP_DOWN':         ['STEP_DOWN', 'LINEAR_DOWN', 'EXPONENTIAL_DOWN'],
    'BACK_SKEWED_DOWN':  ['EXPONENTIAL_DOWN', 'LINEAR_DOWN', 'STEP_DOWN', 'LOGARITHMIC_DOWN'],
    'FRONT_SKEWED_DOWN': ['LOGARITHMIC_DOWN', 'LINEAR_DOWN', 'EXPONENTIAL_DOWN'],
    # Reversal families (V <-> U, both pivot patterns)
    'SYMMETRIC_V_UP':    ['ROUNDED_U_UP', 'SYMMETRIC_V_UP', 'NOISE'],
    'SYMMETRIC_V_DOWN':  ['ROUNDED_U_DOWN', 'SYMMETRIC_V_DOWN', 'NOISE'],
    'ROUNDED_U_UP':      ['ROUNDED_U_UP', 'SYMMETRIC_V_UP', 'NOISE'],
    'ROUNDED_U_DOWN':    ['ROUNDED_U_DOWN', 'SYMMETRIC_V_DOWN', 'NOISE'],
    # Oscillators (no direction; absorb to NOISE if no peer is large)
    'SINE_WAVE':         ['NOISE', 'FLATLINE'],
    'DAMPED_OSCILLATOR': ['NOISE', 'FLATLINE'],
    'EXPAND_OSCILLATOR': ['NOISE', 'FLATLINE'],
    # FLATLINE / NOISE (already large; identity)
    'FLATLINE':          ['FLATLINE'],
    'NOISE':             ['NOISE'],
}


def _absorb(small_shape: str, large_shapes: set) -> str:
    """Walk the absorption preference list; return the first large shape found."""
    prefs = ABSORPTION_PREFERENCE.get(small_shape, [small_shape])
    for candidate in prefs:
        if candidate in large_shapes:
            return candidate
    # No preference matches a large shape; fall back to NOISE
    return 'NOISE' if 'NOISE' in large_shapes else list(large_shapes)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/all_motifs_labeled.csv',
        help='Use the labeled CSV (no chord fingerprint needed)')
    ap.add_argument('--min-shape-n', type=int, default=30,
                    help='Shapes with n_phrases below this are SMALL and absorbed')
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--out',
        default='reports/findings/segments/small_shape_absorption.md')
    ap.add_argument('--out-mapping-csv',
        default='reports/findings/segments/small_shape_absorption_map.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.phrase_csv)
    if args.split != 'BOTH':
        df = df[df['split'] == args.split]
    print(f'Phrases: {len(df)} ({args.split})')

    shape_counts = df['shape_class'].value_counts()
    large_shapes = set(shape_counts[shape_counts >= args.min_shape_n].index.tolist())
    small_shapes = shape_counts[shape_counts < args.min_shape_n].index.tolist()
    print(f'\nLarge shapes (>={args.min_shape_n}): {sorted(large_shapes)}')
    print(f'Small shapes (<{args.min_shape_n}):  {small_shapes}')

    # Build per-shape absorption mapping
    rows = []
    for small in small_shapes:
        n_small = int(shape_counts.loc[small])
        prefs = ABSORPTION_PREFERENCE.get(small, [small])
        target = _absorb(small, large_shapes)
        rows.append({
            'small_shape': small,
            'n_small': n_small,
            'preference_list': ' -> '.join(prefs),
            'absorbed_to': target,
        })
    abs_df = pd.DataFrame(rows).sort_values('n_small', ascending=False)
    print('\nAbsorption mapping (family-priority, NO chord/feature-structure):')
    print('=' * 100)
    print(abs_df.to_string(index=False))

    # Build per-phrase mapping
    mapping_rows = []
    for _, r in df.iterrows():
        shape = r['shape_class']
        if shape in large_shapes:
            target = shape
        else:
            target = _absorb(shape, large_shapes)
        mapping_rows.append({
            'day':            r['day'],
            'seg_idx':        r['seg_idx'],
            'original_shape': shape,
            'absorbed_to':    target,
        })
    map_df = pd.DataFrame(mapping_rows)

    # Save mapping CSV
    os.makedirs(os.path.dirname(args.out_mapping_csv), exist_ok=True)
    map_df.to_csv(args.out_mapping_csv, index=False)
    print(f'\nPer-phrase mapping CSV -> {args.out_mapping_csv}  ({len(map_df)} rows)')

    # Markdown report
    md = ['# Small-shape absorption (family-priority, 15m level only)',
          '',
          f'_Generated {datetime.now().isoformat()}_',
          '',
          f'Split: {args.split}',
          f'Min shape n threshold: {args.min_shape_n}',
          f'Large shapes (kept as primitive labels): {sorted(large_shapes)}',
          f'Small shapes (absorbed): {small_shapes}',
          '',
          '## Absorption rules',
          '',
          'No chord or feature-structure used. Each small shape is absorbed into the',
          'first member of its family-preference list that is itself a large shape.',
          'Family logic:',
          '',
          '- **UP-direction families**: BACK_SKEWED_UP -> EXPONENTIAL_UP (accelerating',
          '  curve neighbor); FRONT_SKEWED_UP -> LOGARITHMIC_UP (exhausting curve);',
          '  STEP_UP -> LINEAR_UP if STEP_UP itself too small; etc.',
          '- **DOWN-direction families**: mirror of UP.',
          '- **Reversal families**: SYMMETRIC_V <-> ROUNDED_U (both pivot patterns).',
          '- **Oscillators** (SINE_WAVE, DAMPED, EXPAND): -> NOISE (no clean primitive home).',
          '',
          '## Per-shape absorption mapping',
          '',
          '```',
          abs_df.to_string(index=False),
          '```',
          '',
          '## Resulting absorbed shape distribution',
          '',
          '```',
          map_df['absorbed_to'].value_counts().to_string(),
          '```',
          '',
          '## Per-original-shape destination crosstab',
          '',
          '```',
          pd.crosstab(map_df['original_shape'], map_df['absorbed_to']).to_string(),
          '```',
          '',
          '## Implication',
          '',
          'After this absorption, only the large shapes remain as primitive labels.',
          'Small-shape phrases inherit the prior of the nearest large-shape neighbor',
          'in the family tree (no chord/feature-structure used). The Bayesian-table',
          'cell count reduces accordingly: each small shape no longer needs its own',
          'cell. Per-shape HDBSCAN now runs on this consolidated label set.',
          '']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'Report -> {args.out}')


if __name__ == '__main__':
    main()
