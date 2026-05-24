"""Pre-EDA: walk the conditioning hierarchy step by step, measuring where
uniqueness becomes prohibitive.

The probabilistic-Bayesian framework asks: at what conditioning depth does
the substrate become too thin to extract structure? We walk the hierarchy
from coarse to fine, ADDING one axis at a time, and at each step we
measure:

    n_cells_populated      distinct cells observed
    median_per_cell        median observations per cell
    p25_per_cell, p10      lower-tail of cell populations
    pct_cells_with_n_lt_5  fraction of cells too thin for any inference
    pct_cells_with_n_lt_30 fraction where Bayesian shrinkage would dominate

Heuristic breakdown threshold: median_per_cell < 5 means the typical cell
has too few observations for even a hierarchical posterior to pull useful
signal — the data is gone. The level BEFORE that breakdown is where the
Bayesian table should live.

Per-shape breakdown is reported separately because directional shapes and
FLATLINE behave very differently (per the 2026-05-10 uniqueness EDA: 80-99%
unique for directional shapes; 9% for FLATLINE).

USAGE
    python tools/uniqueness_step_walker.py
    python tools/uniqueness_step_walker.py --axes shape,motif_counts,variation_slope_q4
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


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin values into 0..n_bins-1 (NaN -> -1)."""
    out = np.full(len(values), -1, dtype=int)
    finite = np.isfinite(values)
    if finite.sum() < n_bins * 5:
        return out
    edges = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1))
    edges[0]  = -np.inf
    edges[-1] = np.inf
    out[finite] = np.clip(np.searchsorted(edges, values[finite], side='right') - 1,
                          0, n_bins - 1)
    return out


def _motif_signature_counts(phrase_row, motifs_for_day) -> tuple:
    kids = motifs_for_day[motifs_for_day['parent_motif_idx'] == phrase_row['seg_idx']]
    counter = Counter(kids['shape_class'].tolist())
    return tuple(sorted(counter.items()))


def _motif_signature_multiset(phrase_row, motifs_for_day) -> tuple:
    kids = motifs_for_day[motifs_for_day['parent_motif_idx'] == phrase_row['seg_idx']]
    return tuple(sorted(set(kids['shape_class'].tolist())))


def _build_axis_columns(phrases: pd.DataFrame, motifs: pd.DataFrame,
                        axis_specs: list[str]) -> pd.DataFrame:
    """For each phrase row, attach axis-value columns named by spec."""
    df = phrases.copy()
    motifs_by_day = {d: g for d, g in motifs.groupby('day')}

    for spec in axis_specs:
        if spec == 'shape':
            df['axis_shape'] = df['shape_class']
        elif spec == 'motif_multiset':
            df['axis_motif_multiset'] = df.apply(
                lambda r: _motif_signature_multiset(r, motifs_by_day.get(r['day'],
                                                  motifs.iloc[0:0])), axis=1)
        elif spec == 'motif_counts':
            df['axis_motif_counts'] = df.apply(
                lambda r: _motif_signature_counts(r, motifs_by_day.get(r['day'],
                                                  motifs.iloc[0:0])), axis=1)
        elif spec.startswith('variation_'):
            # e.g. variation_slope_15m__std__q4 = slope_15m__std binned into 4 quantiles
            tokens = spec.split('_')
            # Recover the column name and n_bins
            n_bins = 4
            if tokens[-1].startswith('q') and tokens[-1][1:].isdigit():
                n_bins = int(tokens[-1][1:])
                col_tokens = tokens[1:-1]
            else:
                col_tokens = tokens[1:]
            col = '_'.join(col_tokens)
            if col not in df.columns:
                print(f'  [warn] axis spec {spec}: column {col} not in data; skipping')
                continue
            df[f'axis_{spec}'] = _quantile_bin(df[col].values.astype(float), n_bins)
        elif spec == 'tod_hour':
            df['axis_tod_hour'] = df['tod_start_hour_utc']
        elif spec == 'dow':
            df['axis_dow'] = pd.to_datetime(df['day'].str.replace('_', '-')
                                             ).dt.strftime('%a')
        else:
            print(f'  [warn] unknown axis spec: {spec}; skipping')

    return df


def _level_stats(df: pd.DataFrame, axis_cols: list[str]) -> dict:
    """For a given conditioning (axis_cols), compute per-cell population stats."""
    if not axis_cols:
        return {}
    # Build cell key
    keys = list(zip(*[df[c].astype(str).values for c in axis_cols]))
    cell_counter = Counter(keys)
    cell_sizes = np.array(list(cell_counter.values()))
    n_total = int(cell_sizes.sum())
    n_cells = int(len(cell_sizes))
    return {
        'n_total_obs':     n_total,
        'n_cells':         n_cells,
        'mean_per_cell':   round(float(cell_sizes.mean()), 1),
        'median_per_cell': int(np.median(cell_sizes)),
        'p25_per_cell':    int(np.quantile(cell_sizes, 0.25)),
        'p10_per_cell':    int(np.quantile(cell_sizes, 0.10)),
        'max_per_cell':    int(cell_sizes.max()),
        'pct_cells_n_lt_5':   round(100 * float((cell_sizes < 5).mean()), 1),
        'pct_cells_n_lt_30':  round(100 * float((cell_sizes < 30).mean()), 1),
        'pct_cells_n_eq_1':   round(100 * float((cell_sizes == 1).mean()), 1),
    }


def walk(df: pd.DataFrame, axis_columns: list[str],
         per_shape: bool = True) -> pd.DataFrame:
    """Step through conditioning levels, computing stats at each prefix."""
    rows = []
    for k in range(1, len(axis_columns) + 1):
        prefix = axis_columns[:k]
        global_stats = _level_stats(df, prefix)
        rows.append({
            'step':       k,
            'axis_added': axis_columns[k-1],
            'axes_active': ' x '.join(prefix),
            'scope':      'GLOBAL',
            **global_stats,
        })
        if per_shape and 'axis_shape' in prefix:
            # Per-shape breakdown
            for shape, sub in df.groupby('axis_shape'):
                stats = _level_stats(sub, prefix)
                rows.append({
                    'step':       k,
                    'axis_added': axis_columns[k-1],
                    'axes_active': ' x '.join(prefix),
                    'scope':      f'shape={shape}',
                    **stats,
                })
    return pd.DataFrame(rows)


def _format_table(out_df: pd.DataFrame) -> str:
    cols = ['step', 'axes_active', 'scope', 'n_total_obs', 'n_cells',
            'mean_per_cell', 'median_per_cell', 'p25_per_cell', 'p10_per_cell',
            'pct_cells_n_lt_5', 'pct_cells_n_lt_30', 'pct_cells_n_eq_1']
    return out_df[cols].to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/all_motifs_labeled_with_chord.csv')
    ap.add_argument('--motif-csv',
        default='reports/findings/segments/all_melodies_labeled_with_chord.csv')
    ap.add_argument('--axes',
        default='shape,variation_slope_15m__std__q4,motif_multiset,motif_counts',
        help='comma-sep axis specs to add step-by-step')
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--breakdown-median-threshold', type=int, default=5,
        help='Below this median_per_cell, level is considered "kills us"')
    ap.add_argument('--out',
        default='reports/findings/segments/uniqueness_step_walker.md')
    args = ap.parse_args()

    phrases = pd.read_csv(args.phrase_csv)
    motifs = pd.read_csv(args.motif_csv)
    if args.split != 'BOTH':
        phrases = phrases[phrases['split'] == args.split]
        motifs = motifs[motifs['split'] == args.split]
    print(f'Phrases: {len(phrases)}    Motifs: {len(motifs)}    Split: {args.split}')

    axis_specs = [s.strip() for s in args.axes.split(',') if s.strip()]
    print(f'Axes (step-by-step): {axis_specs}')
    df = _build_axis_columns(phrases, motifs, axis_specs)

    axis_cols = [c for c in df.columns if c.startswith('axis_')]
    # Reorder to match the input spec order
    axis_cols = [f'axis_{s}' for s in axis_specs if f'axis_{s}' in df.columns]

    out = walk(df, axis_cols, per_shape=True)

    print('\n' + '=' * 100)
    print('STEP WALKER: GLOBAL stats per level')
    print('=' * 100)
    global_only = out[out['scope'] == 'GLOBAL']
    print(_format_table(global_only))

    print('\n' + '=' * 100)
    print(f'BREAKDOWN POINT (median_per_cell < {args.breakdown_median_threshold})')
    print('=' * 100)
    breakdown = global_only[global_only['median_per_cell'] < args.breakdown_median_threshold]
    if breakdown.empty:
        print(f'  All conditioning levels have median >= {args.breakdown_median_threshold} (no breakdown observed)')
    else:
        first = breakdown.iloc[0]
        print(f'  GLOBAL breakdown at step {first["step"]} (axis: {first["axis_added"]})')
        print(f'  At that step: {first["axes_active"]}')
        print(f'    n_cells={first["n_cells"]}  median_per_cell={first["median_per_cell"]}  '
              f'pct_n_lt_5={first["pct_cells_n_lt_5"]}%')
        print(f'  -> Use the LEVEL BEFORE this as the Bayesian-table substrate.')

    # Per-shape breakdown points
    print('\n' + '=' * 100)
    print(f'PER-SHAPE breakdown points')
    print('=' * 100)
    per_shape_rows = []
    for shape, sub in out[out['scope'].str.startswith('shape=')].groupby('scope'):
        thin_levels = sub[sub['median_per_cell'] < args.breakdown_median_threshold]
        if thin_levels.empty:
            per_shape_rows.append({
                'shape': shape, 'breakdown_step': None,
                'breakdown_axis': '(never breaks down at these axes)',
                'final_median': sub.iloc[-1]['median_per_cell'],
            })
        else:
            first = thin_levels.iloc[0]
            per_shape_rows.append({
                'shape': shape,
                'breakdown_step': first['step'],
                'breakdown_axis': first['axis_added'],
                'final_median': first['median_per_cell'],
            })
    per_shape_df = pd.DataFrame(per_shape_rows).sort_values('shape')
    print(per_shape_df.to_string(index=False))

    # Markdown report
    md = ['# Uniqueness step walker (probabilistic-depth finder)',
          '',
          f'_Generated {datetime.now().isoformat()}_',
          '',
          f'Split: {args.split}',
          f'Breakdown threshold: median_per_cell < {args.breakdown_median_threshold}',
          f'Axes (step order): {axis_specs}',
          '',
          '## How to read this',
          '',
          '- At each step, one more axis is ADDED to the cell key.',
          '- For each step, we measure how many cells exist and how populated they are.',
          '- The "breakdown point" is the FIRST step where median observations per cell',
          '  drops below the threshold. The level BEFORE that is where the Bayesian',
          '  table substrate has enough data; deeper conditioning starves the cells.',
          '- Per-shape breakdown points may differ: FLATLINE may tolerate deeper',
          '  conditioning than directional shapes (per 2026-05-10 uniqueness EDA).',
          '',
          '## Global step walker',
          '',
          '```',
          _format_table(global_only),
          '```',
          '',
          '## Per-shape breakdown',
          '',
          '```',
          per_shape_df.to_string(index=False),
          '```',
          '',
          '## Implication',
          '',
          'Use each shape\'s LEVEL BEFORE breakdown as the conditioning depth for that',
          'shape\'s Bayesian-table cells. Apply hierarchical shrinkage: cell -- shape -- universal.',
          'The probabilistic model adapts conditioning depth by shape, rather than imposing',
          'a one-size-fits-all hierarchy.',
          '']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(md))
    print(f'\nReport -> {args.out}')


if __name__ == '__main__':
    main()
