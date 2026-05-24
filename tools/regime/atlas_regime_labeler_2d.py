"""
atlas_regime_labeler_2d.py — Extend daily regime labels into a 2D taxonomy:

  direction × variation

Reads `DATA/ATLAS/regime_labels.csv` (produced by tools/atlas_regime_labeler.py)
and adds three columns:

  direction_axis ∈ {UP, DOWN, FLAT}
  variation_axis ∈ {SMOOTH, CHOPPY}
  regime_2d      = "<direction>_<variation>"  (e.g., "UP_CHOPPY")

Plus a `split` column ∈ {IS, VAL, OOS} based on time-ordered fractions
(default 60/20/20) so downstream tools can filter to "IS UP_SMOOTH days only".

DECISION RULES (defaults, all CLI-tunable):

  Direction (uses existing `directional_strength` + signed `net_move`):
    UP    if directional_strength >= dir_threshold (0.5) AND net_move > 0
    DOWN  if directional_strength >= dir_threshold (0.5) AND net_move < 0
    FLAT  otherwise

  Variation (uses `range_expansion` and `efficiency_ratio`):
    SMOOTH  if range_expansion < smooth_range_threshold (0.7)
            OR efficiency_ratio >= smooth_eff_threshold (0.05)
    CHOPPY  otherwise

Why this split:
  - SMOOTH means either (a) day didn't move much (small range) OR (b) day
    moved a lot but with high efficiency (smooth trend).
  - CHOPPY means range expanded AND efficiency was low — lots of
    intra-day churn. This is where counter-trend strategies bleed.

OUTPUTS:

  DATA/ATLAS/regime_labels_2d.csv             -- per-day with all axes + split
  reports/findings/regime_2d/<date>_summary.md  -- distribution tables + per-regime stats
  reports/findings/regime_2d/<date>_distribution.png

USAGE:

    python tools/atlas_regime_labeler_2d.py
    python tools/atlas_regime_labeler_2d.py --is-frac 0.6 --val-frac 0.2
    python tools/atlas_regime_labeler_2d.py --dir-threshold 0.6 --smooth-range 0.8

Programmatic access:

    from tools.atlas_regime_labeler_2d import load_regime_labels
    df = load_regime_labels()
    is_up_smooth = df[(df.split == 'IS') & (df.regime_2d == 'UP_SMOOTH')]
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_LABELS_CSV = 'DATA/ATLAS/regime_labels.csv'
DEFAULT_OUT_CSV = 'DATA/ATLAS/regime_labels_2d.csv'
DEFAULT_FINDINGS_DIR = 'reports/findings/regime_2d'

DIRECTION_ORDER = ['UP', 'DOWN', 'FLAT']
VARIATION_ORDER = ['SMOOTH', 'CHOPPY']
REGIME_2D_ORDER = [f'{d}_{v}' for d in DIRECTION_ORDER for v in VARIATION_ORDER]
SPLIT_ORDER = ['IS', 'VAL', 'OOS']

REGIME_2D_COLORS = {
    'UP_SMOOTH':   '#16a34a',  # solid green
    'UP_CHOPPY':   '#86efac',  # light green
    'DOWN_SMOOTH': '#dc2626',  # solid red
    'DOWN_CHOPPY': '#fca5a5',  # light red
    'FLAT_SMOOTH': '#64748b',  # slate
    'FLAT_CHOPPY': '#f59e0b',  # amber
}


def label_direction(direction_strength: float, net_move: float,
                     dir_threshold: float) -> str:
    if pd.isna(direction_strength) or pd.isna(net_move):
        return 'FLAT'
    if direction_strength >= dir_threshold:
        return 'UP' if net_move > 0 else 'DOWN'
    return 'FLAT'


def label_variation(range_expansion: float, efficiency_ratio: float,
                     smooth_range_threshold: float,
                     smooth_eff_threshold: float) -> str:
    """SMOOTH if either (a) small range OR (b) high efficiency. Else CHOPPY."""
    if pd.isna(range_expansion) or pd.isna(efficiency_ratio):
        return 'SMOOTH'  # default to safer label
    if range_expansion < smooth_range_threshold:
        return 'SMOOTH'
    if efficiency_ratio >= smooth_eff_threshold:
        return 'SMOOTH'
    return 'CHOPPY'


def assign_split(df: pd.DataFrame, is_frac: float, val_frac: float) -> pd.Series:
    """Time-ordered IS / VAL / OOS split (by row order, which matches date order)."""
    n = len(df)
    n_is = int(n * is_frac)
    n_val = int(n * val_frac)
    splits = ['IS'] * n
    splits[n_is:n_is + n_val] = ['VAL'] * n_val
    splits[n_is + n_val:] = ['OOS'] * (n - n_is - n_val)
    return pd.Series(splits, index=df.index)


def label_2d(input_csv: str = DEFAULT_LABELS_CSV,
             output_csv: str = DEFAULT_OUT_CSV,
             dir_threshold: float = 0.5,
             smooth_range_threshold: float = 0.7,
             smooth_eff_threshold: float = 0.05,
             is_frac: float = 0.6,
             val_frac: float = 0.2) -> pd.DataFrame:
    """Read 1D labels, add 2D + split columns, write enriched CSV."""
    df = pd.read_csv(input_csv)
    df = df.sort_values('date').reset_index(drop=True)

    df['direction_axis'] = df.apply(
        lambda r: label_direction(r['directional_strength'], r['net_move'],
                                    dir_threshold), axis=1)
    df['variation_axis'] = df.apply(
        lambda r: label_variation(r['range_expansion'], r['efficiency_ratio'],
                                    smooth_range_threshold, smooth_eff_threshold),
        axis=1)
    df['regime_2d'] = df['direction_axis'] + '_' + df['variation_axis']
    df['split'] = assign_split(df, is_frac, val_frac)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def load_regime_labels(path: str = DEFAULT_OUT_CSV) -> pd.DataFrame:
    """Convenience loader for downstream tools."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run tools/atlas_regime_labeler_2d.py first")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def write_summary(df: pd.DataFrame, findings_dir: str,
                   args: argparse.Namespace) -> tuple[str, str]:
    """Write distribution tables + summary markdown."""
    os.makedirs(findings_dir, exist_ok=True)
    today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')

    # Cross-tab: direction × variation
    cross = pd.crosstab(df['direction_axis'], df['variation_axis'])
    cross = cross.reindex(index=DIRECTION_ORDER, columns=VARIATION_ORDER, fill_value=0)
    cross_pct = (cross / cross.sum().sum() * 100).round(1)

    # Cross-tab: regime_2d × split
    split_cross = pd.crosstab(df['regime_2d'], df['split'])
    split_cross = split_cross.reindex(
        index=[r for r in REGIME_2D_ORDER if r in split_cross.index],
        columns=[s for s in SPLIT_ORDER if s in split_cross.columns],
        fill_value=0,
    )

    # Per-regime stats
    stats_rows = []
    for r2d in REGIME_2D_ORDER:
        sub = df[df['regime_2d'] == r2d]
        if len(sub) == 0:
            continue
        stats_rows.append({
            'regime_2d': r2d,
            'n_days': len(sub),
            'avg_net_move': sub['net_move'].mean(),
            'median_range': sub['range'].median(),
            'avg_dir_strength': sub['directional_strength'].mean(),
            'avg_eff_ratio': sub['efficiency_ratio'].mean(),
            'avg_range_exp': sub['range_expansion'].mean(),
        })
    stats_df = pd.DataFrame(stats_rows)

    md_path = os.path.join(findings_dir, f'{today}_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 2D regime labels — DATA/ATLAS\n\n")
        f.write(f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("## Settings\n\n")
        f.write(f"- Source: `{args.input_csv}`\n")
        f.write(f"- Direction threshold: {args.dir_threshold}\n")
        f.write(f"- Smooth range threshold: {args.smooth_range}\n")
        f.write(f"- Smooth efficiency threshold: {args.smooth_eff}\n")
        f.write(f"- Split fractions: IS={args.is_frac}, VAL={args.val_frac}, "
                f"OOS={1 - args.is_frac - args.val_frac:.2f}\n\n")

        f.write("## Distribution: direction × variation\n\n")
        f.write("Counts:\n\n")
        f.write(cross.to_string())
        f.write("\n\nPercent of all days:\n\n")
        f.write(cross_pct.to_string())
        f.write("\n\n")

        f.write("## Distribution: regime_2d × split\n\n")
        f.write(split_cross.to_string())
        f.write("\n\n")

        f.write("## Per-regime stats\n\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")

        f.write("## How to use\n\n")
        f.write("```python\n")
        f.write("from tools.atlas_regime_labeler_2d import load_regime_labels\n")
        f.write("df = load_regime_labels()\n")
        f.write("# IS days only:\n")
        f.write("df_is = df[df['split'] == 'IS']\n")
        f.write("# OOS UP_SMOOTH days only:\n")
        f.write("df_oos_up_smooth = df[(df['split'] == 'OOS') & "
                "(df['regime_2d'] == 'UP_SMOOTH')]\n")
        f.write("```\n")

    # Distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['regime_2d'].value_counts().reindex(REGIME_2D_ORDER, fill_value=0)
    colors = [REGIME_2D_COLORS.get(r, '#888') for r in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title('Days per 2D regime')
    axes[0].set_ylabel('Days')
    axes[0].tick_params(axis='x', rotation=45)
    for tick in axes[0].get_xticklabels():
        tick.set_ha('right')

    # Stacked split bars
    split_pct = split_cross.div(split_cross.sum(axis=1), axis=0).fillna(0) * 100
    bottoms = np.zeros(len(split_pct))
    split_colors = {'IS': '#1f77b4', 'VAL': '#ff7f0e', 'OOS': '#2ca02c'}
    for s in [s for s in SPLIT_ORDER if s in split_pct.columns]:
        axes[1].barh(split_pct.index, split_pct[s].values, left=bottoms,
                     color=split_colors[s], label=s)
        bottoms += split_pct[s].values
    axes[1].set_title('Split distribution per regime (%)')
    axes[1].set_xlabel('% of regime days')
    axes[1].legend()
    axes[1].set_xlim(0, 100)

    fig.tight_layout()
    chart_path = os.path.join(findings_dir, f'{today}_distribution.png')
    fig.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return md_path, chart_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', default=DEFAULT_LABELS_CSV)
    parser.add_argument('--output-csv', default=DEFAULT_OUT_CSV)
    parser.add_argument('--findings-dir', default=DEFAULT_FINDINGS_DIR)
    parser.add_argument('--dir-threshold', type=float, default=0.5,
                        help='directional_strength threshold for UP/DOWN vs FLAT')
    parser.add_argument('--smooth-range', type=float, default=0.7,
                        help='range_expansion below this → SMOOTH (small range)')
    parser.add_argument('--smooth-eff', type=float, default=0.05,
                        help='efficiency_ratio above this → SMOOTH (clean trend)')
    parser.add_argument('--is-frac', type=float, default=0.6)
    parser.add_argument('--val-frac', type=float, default=0.2)
    args = parser.parse_args()

    print(f"Reading {args.input_csv}")
    df = label_2d(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        dir_threshold=args.dir_threshold,
        smooth_range_threshold=args.smooth_range,
        smooth_eff_threshold=args.smooth_eff,
        is_frac=args.is_frac,
        val_frac=args.val_frac,
    )
    print(f"  [saved] {args.output_csv} ({len(df)} days)")

    print(f"\nDistribution:")
    print(f"  direction × variation:")
    cross = pd.crosstab(df['direction_axis'], df['variation_axis'])
    cross = cross.reindex(index=DIRECTION_ORDER, columns=VARIATION_ORDER, fill_value=0)
    print(cross)

    print(f"\n  regime_2d × split:")
    split_cross = pd.crosstab(df['regime_2d'], df['split'])
    split_cross = split_cross.reindex(
        index=[r for r in REGIME_2D_ORDER if r in split_cross.index],
        columns=[s for s in SPLIT_ORDER if s in split_cross.columns],
        fill_value=0,
    )
    print(split_cross)

    md_path, chart_path = write_summary(df, args.findings_dir, args)
    print(f"\n  [saved] {md_path}")
    print(f"  [saved] {chart_path}")


if __name__ == '__main__':
    main()
