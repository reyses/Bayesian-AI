"""V0 Bayesian Table — joint hierarchical chord lookup.

The meta-router needs a single lookup table keyed by the at-bar primitive
chord vector that returns:
    n           how many IS samples landed in this chord
    n_days      how many distinct days contributed (for day-clustered honesty)
    P_up_<h>    Beta(1,1) posterior P(fwd_return > 0) at multiple horizons
    ci95_<h>    95% credible interval per horizon
    mean_<h>    mean forward return per horizon
    n_oos       OOS sample count
    P_oos_<h>   OOS posterior at the same horizons
    sign_match  whether IS sign of (P-0.5) matches OOS sign per horizon
    trust_tag   {'high' / 'med' / 'low' / 'reject'} based on stability

CHORD VECTOR
    (phrase_shape, motif_shape, sub_motif_shape, measure_shape)
    note (5s) level INTENTIONALLY DROPPED — see
    `memory/feedback_5s_inherently_noise.md` (5s is microstructure noise;
    inner shape adds zero predictive content beyond parent context).

LOOKUP TIMEPOINT
    end_ts of every MEASURE — i.e. ~every 15s, the natural decision boundary
    where a sub-15s shape just resolved and forward outcomes are well-defined.
    Total events ≈ 70,782 measures across 345 days.

HORIZONS
    30s, 2min, 5min, 10min, 30min, 60min
    Picked to span microstructure (30s) → tactical (2-10min) →
    strategic (30-60min). Cells where signal lives only at 30s but dies
    by 2min are tagged microstructure-only (drop for strategic use).

USAGE
    python tools/bayes_table_v0_joint_chord.py
    python tools/bayes_table_v0_joint_chord.py --min-n 30 --min-days 5
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
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s


HORIZONS_S = {
    'h30s':   30,
    'h2m':    120,
    'h5m':    300,
    'h10m':   600,
    'h30m':   1800,
    'h60m':   3600,
}


def attach_parent_chain(child_df: pd.DataFrame, parent_df: pd.DataFrame,
                         col_name: str) -> pd.DataFrame:
    """For each child row, attach parent shape as <col_name>."""
    parent_df = parent_df.copy()
    parent_df['parent_chain'] = parent_df['parent_chain'].fillna('').astype(str)
    parent_df['idx'] = parent_df['idx'].astype(int)
    plk = parent_df.set_index(['day', 'parent_chain', 'idx'])['shape'].to_dict()
    child_df = child_df.copy()
    child_df['parent_chain'] = child_df['parent_chain'].fillna('').astype(str)
    def _gp(row):
        pc = row['parent_chain']
        if not pc:
            return None
        toks = pc.split('/')
        try:
            return plk.get((row['day'], '/'.join(toks[:-1]), int(toks[-1])))
        except ValueError:
            return None
    child_df[col_name] = child_df.apply(_gp, axis=1)
    return child_df


def beta_posterior(k: int, n: int) -> tuple[float, float, float]:
    a, b = k + 1, n - k + 1
    return (float(a / (a + b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def build_chord_per_measure(in_dir: str) -> pd.DataFrame:
    """Walk the hierarchy and emit one row per measure with full chord."""
    print('Loading hierarchy CSVs...')
    phrases = pd.read_csv(os.path.join(in_dir, 'all_phrases.csv'))
    motifs = pd.read_csv(os.path.join(in_dir, 'all_motifs.csv'))
    sub_motifs = pd.read_csv(os.path.join(in_dir, 'all_sub_motifs.csv'))
    measures = pd.read_csv(os.path.join(in_dir, 'all_measures.csv'))

    # Walk up the parent chain at each level
    print('Joining 4-level chord onto each measure...')
    measures = attach_parent_chain(measures, sub_motifs, 'sub_motif_shape')
    sub_motifs = attach_parent_chain(sub_motifs, motifs, 'motif_shape')
    motifs = attach_parent_chain(motifs, phrases, 'phrase_shape')

    # Now build measure → sub_motif (got it) → motif → phrase via stepped lookups
    sm_lk = sub_motifs.set_index(
        ['day', sub_motifs['parent_chain'].fillna('').astype(str),
         sub_motifs['idx'].astype(int)]
    )['motif_shape'].to_dict()
    motif_lk = motifs.set_index(
        ['day', motifs['parent_chain'].fillna('').astype(str),
         motifs['idx'].astype(int)]
    )['phrase_shape'].to_dict()

    # For each measure, walk up
    measures['parent_chain'] = measures['parent_chain'].fillna('').astype(str)
    measures['idx'] = measures['idx'].astype(int)
    def get_motif_shape(row):
        pc = row['parent_chain']
        if not pc:
            return None
        toks = pc.split('/')
        try:
            sm_chain_above = '/'.join(toks[:-1])
            sm_idx = int(toks[-1])
            return sm_lk.get((row['day'], sm_chain_above, sm_idx))
        except ValueError:
            return None
    def get_phrase_shape(row):
        pc = row['parent_chain']
        if not pc:
            return None
        toks = pc.split('/')
        # phrase is two levels up — walk up to motif's parent_chain (which
        # is just the phrase_idx as a single token)
        try:
            sm_chain_above = '/'.join(toks[:-1])
            sm_idx = int(toks[-1])
            # sub_motif lives at (day, sm_chain_above, sm_idx); its parent_chain
            # is sm_chain_above, which is "phrase_idx/motif_idx"
            motif_chain_above = '/'.join(sm_chain_above.split('/')[:-1])
            motif_idx_str = sm_chain_above.split('/')[-1]
            motif_idx = int(motif_idx_str)
            return motif_lk.get((row['day'], motif_chain_above, motif_idx))
        except (ValueError, IndexError):
            return None

    tqdm.pandas(desc='walk-up motif')
    measures['motif_shape'] = measures.progress_apply(get_motif_shape, axis=1)
    tqdm.pandas(desc='walk-up phrase')
    measures['phrase_shape'] = measures.progress_apply(get_phrase_shape, axis=1)

    # Rename measure's own shape for clarity
    measures = measures.rename(columns={'shape': 'measure_shape'})
    return measures[['day', 'split', 'idx', 'parent_chain', 'measure_shape',
                       'sub_motif_shape', 'motif_shape', 'phrase_shape',
                       'mean_sigma', 'length_min', 'start_ts', 'end_ts']].copy()


def attach_multi_horizon_returns(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Add fwd_return_<h> columns for every horizon."""
    df = df.copy()
    for h_name in HORIZONS_S:
        df[f'fwd_return_{h_name}'] = np.nan

    print('Computing forward returns at all horizons...')
    for day, sub in tqdm(df.groupby('day')):
        if day not in cache:
            d = _load_5s(day)
            if d.empty:
                cache[day] = None
            else:
                cache[day] = (d['timestamp'].values.astype(np.int64),
                               d['close'].values.astype(np.float64))
        if cache[day] is None:
            continue
        ts, close = cache[day]
        end_ts = sub['end_ts'].astype(np.int64).values
        i_end = np.searchsorted(ts, end_ts)
        valid_end = i_end < len(ts)
        base = np.where(valid_end, close[np.clip(i_end, 0, len(ts)-1)], np.nan)
        for h_name, h_s in HORIZONS_S.items():
            tgt = end_ts + h_s
            i_t = np.searchsorted(ts, tgt)
            v = (i_t < len(ts)) & valid_end
            target = np.where(v, close[np.clip(i_t, 0, len(ts)-1)], np.nan)
            df.loc[sub.index, f'fwd_return_{h_name}'] = target - base
    return df


def aggregate_chord_table(df: pd.DataFrame, min_n: int,
                           min_days: int) -> pd.DataFrame:
    """Group by 4-level chord, compute IS/OOS Beta posteriors at all horizons."""
    chord_cols = ['phrase_shape', 'motif_shape', 'sub_motif_shape', 'measure_shape']
    df = df.dropna(subset=chord_cols).copy()
    rows = []
    for keys, g in tqdm(df.groupby(chord_cols), desc='chord agg'):
        n_total = len(g)
        if n_total < min_n:
            continue
        n_days = g['day'].nunique()
        if n_days < min_days:
            continue
        is_g = g[g['split'] == 'IS']
        oos_g = g[g['split'] == 'OOS']
        rec = {
            'phrase_shape':    keys[0],
            'motif_shape':     keys[1],
            'sub_motif_shape': keys[2],
            'measure_shape':   keys[3],
            'n':               n_total,
            'n_days':          n_days,
            'n_is':            len(is_g),
            'n_oos':           len(oos_g),
        }
        for h_name in HORIZONS_S:
            col = f'fwd_return_{h_name}'
            yi = is_g[col].dropna()
            yo = oos_g[col].dropna()
            if len(yi) >= 5:
                k = int((yi > 0).sum())
                p, lo, hi = beta_posterior(k, len(yi))
                rec[f'P_is_{h_name}'] = round(p, 4)
                rec[f'cilo_is_{h_name}'] = round(lo, 4)
                rec[f'cihi_is_{h_name}'] = round(hi, 4)
                rec[f'mean_is_{h_name}'] = round(float(yi.mean()), 3)
            else:
                rec[f'P_is_{h_name}'] = np.nan
                rec[f'cilo_is_{h_name}'] = np.nan
                rec[f'cihi_is_{h_name}'] = np.nan
                rec[f'mean_is_{h_name}'] = np.nan
            if len(yo) >= 5:
                k = int((yo > 0).sum())
                p, lo, hi = beta_posterior(k, len(yo))
                rec[f'P_oos_{h_name}'] = round(p, 4)
                rec[f'cilo_oos_{h_name}'] = round(lo, 4)
                rec[f'cihi_oos_{h_name}'] = round(hi, 4)
                rec[f'mean_oos_{h_name}'] = round(float(yo.mean()), 3)
            else:
                rec[f'P_oos_{h_name}'] = np.nan
                rec[f'cilo_oos_{h_name}'] = np.nan
                rec[f'cihi_oos_{h_name}'] = np.nan
                rec[f'mean_oos_{h_name}'] = np.nan
            # Sign match: same side of 0.5
            pi = rec[f'P_is_{h_name}']
            po = rec[f'P_oos_{h_name}']
            if pd.notna(pi) and pd.notna(po):
                rec[f'sign_match_{h_name}'] = int(np.sign(pi - 0.5) ==
                                                    np.sign(po - 0.5))
            else:
                rec[f'sign_match_{h_name}'] = -1  # NA
        # Trust tag based on multi-horizon sign-match
        sm = [rec[f'sign_match_{h}'] for h in HORIZONS_S
              if rec[f'sign_match_{h}'] >= 0]
        if not sm:
            rec['trust_tag'] = 'no_oos'
        else:
            match_rate = sum(sm) / len(sm)
            if match_rate >= 0.85 and len(sm) >= 4:
                rec['trust_tag'] = 'high'
            elif match_rate >= 0.65:
                rec['trust_tag'] = 'med'
            elif match_rate >= 0.40:
                rec['trust_tag'] = 'low'
            else:
                rec['trust_tag'] = 'reject'
        # Strongest IS edge
        max_edge = 0.0
        max_h = None
        for h_name in HORIZONS_S:
            pi = rec[f'P_is_{h_name}']
            if pd.notna(pi) and abs(pi - 0.5) > abs(max_edge):
                max_edge = pi - 0.5
                max_h = h_name
        rec['max_is_edge'] = round(max_edge, 4)
        rec['max_is_edge_horizon'] = max_h
        rows.append(rec)
    return pd.DataFrame(rows)


def render_summary_chart(table: pd.DataFrame, out_dir: str):
    """High-trust cells with edge — show top-25 strongest |max_is_edge| with
    horizon hint + IS/OOS comparison."""
    if table.empty:
        return
    high = table[table['trust_tag'].isin(['high', 'med'])].copy()
    if high.empty:
        print('  no high/med trust cells to render')
        return
    high['abs_edge'] = high['max_is_edge'].abs()
    top = high.sort_values('abs_edge', ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(14, max(5, 0.4 * len(top) + 2)))
    y = np.arange(len(top))[::-1]
    p_is = top['max_is_edge'].values + 0.5
    # find OOS at the strongest IS horizon
    p_oos = []
    for _, row in top.iterrows():
        h = row['max_is_edge_horizon']
        po = row.get(f'P_oos_{h}', np.nan)
        p_oos.append(po if pd.notna(po) else 0.5)
    p_oos = np.array(p_oos)

    ax.barh(y - 0.18, p_is, height=0.36, color='#1E88E5',
              alpha=0.85, label='IS')
    ax.barh(y + 0.18, p_oos, height=0.36, color='#FB8C00',
              alpha=0.85, label='OOS')
    ax.axvline(0.5, color='black', lw=0.6, alpha=0.5)
    labels = top.apply(
        lambda r: f"{r['phrase_shape'][:8]}/{r['motif_shape'][:8]}/"
                  f"{r['sub_motif_shape'][:8]}/{r['measure_shape'][:8]}  "
                  f"@{r['max_is_edge_horizon']}  "
                  f"n={r['n_is']}/{r['n_oos']}  [{r['trust_tag']}]",
        axis=1).values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('P(fwd_return > 0)')
    ax.set_xlim(0, 1)
    ax.set_title('V0 Bayesian Table — top |edge| cells with high/med trust\n'
                  'IS vs OOS at the strongest-IS horizon', fontsize=11)
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'v0_top_edge_cells.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  chart -> {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='reports/findings/segments/simple_bulk_v2')
    ap.add_argument('--out-dir', default='reports/findings/segments/bayes_table_v0')
    ap.add_argument('--min-n', type=int, default=30,
                     help='Minimum total samples to keep a chord cell')
    ap.add_argument('--min-days', type=int, default=5,
                     help='Minimum distinct days for cell honesty')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chord_df = build_chord_per_measure(args.in_dir)
    print(f'Built {len(chord_df):,} measure-rows with full chord')
    print(f'Drops with NaN chord: {chord_df.isna().any(axis=1).sum():,}')

    cache = {}
    chord_df = attach_multi_horizon_returns(chord_df, cache)

    # Save the per-measure substrate
    sub_path = os.path.join(args.out_dir, 'measure_chord_with_returns.parquet')
    chord_df.to_parquet(sub_path, index=False)
    print(f'Per-measure substrate -> {sub_path}')

    # Aggregate
    print('\nAggregating to chord cells...')
    table = aggregate_chord_table(chord_df, args.min_n, args.min_days)
    print(f'\n{len(table):,} chord cells with n>={args.min_n} on >={args.min_days} days')

    # Save the V0 Bayesian table
    table_path = os.path.join(args.out_dir, 'bayes_table_v0.csv')
    table.to_csv(table_path, index=False)
    print(f'Bayesian table V0 -> {table_path}')

    # Trust tag distribution
    print(f'\nTrust tag distribution:')
    for tag, n in table['trust_tag'].value_counts().items():
        print(f'  {tag:<8s} {n:>5d}')

    # Top |edge| cells
    table_sorted = table.copy()
    table_sorted['abs_edge'] = table_sorted['max_is_edge'].abs()
    print(f'\nTop 15 |IS edge| cells (any trust tag):')
    cols = ['phrase_shape', 'motif_shape', 'sub_motif_shape', 'measure_shape',
            'n_is', 'n_oos', 'max_is_edge', 'max_is_edge_horizon', 'trust_tag']
    print(table_sorted.sort_values('abs_edge', ascending=False).head(15)[cols].to_string(index=False))

    # Top high-trust edges
    high = table_sorted[table_sorted['trust_tag'] == 'high']
    print(f'\nTop 15 high-trust cells:')
    print(high.sort_values('abs_edge', ascending=False).head(15)[cols].to_string(index=False))

    render_summary_chart(table, args.out_dir)


if __name__ == '__main__':
    main()
