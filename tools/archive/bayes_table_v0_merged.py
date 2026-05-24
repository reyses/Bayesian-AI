"""V0 Bayesian Table — MERGED location + chord (Table 3).

Joins the two existing V0 tables on the same population:
    Table A (chord-shape):    bayes_table_v0_joint_chord.py
    Table B (trade-location): bayes_table_v0_trade_location.py
    Table C (THIS):           merged location + chord, see if joint adds edge

Population is Table B's: 6,495 macro events at 1h HL k>=3 sigma (>=60s).

For each event, we look up the containing segment at each of 4 hierarchy
levels (phrase/motif/sub_motif/measure) and attach those 4 chord-shape
columns. Then we build joint conditional probability cells across mixed
keys to answer:
    Does joint(location, chord) add edge over location-alone or chord-alone?
    Or does the chord wash out under the strong location prior?

Comparison axes:
    KEY_LOC  = (side, anchor, sigma_rank_q, slope_q)  4 axes (location-only)
    KEY_CHORD= (motif_shape, sub_motif_shape, measure_shape)  3 axes (chord-only)
    KEY_MIX  = (side, anchor, sigma_rank_q, measure_shape)   4 axes (mixed)
    KEY_FULL = (side, anchor, sigma_rank_q, slope_q,
                motif_shape, measure_shape)              6 axes (joint)

For each key cell with n>=10 IS:
    P(fwd_ret_h60m > 0) IS, OOS, sign-match, mean fwd return.

USAGE
    python tools/bayes_table_v0_merged.py
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


def beta_p(k: int, n: int) -> tuple[float, float, float]:
    a, b = k + 1, n - k + 1
    return (float(a / (a + b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def attach_chord_shape(events: pd.DataFrame, level_csv: str,
                        col_name: str) -> pd.DataFrame:
    """For each event, find the segment at this level whose [start_ts, end_ts]
    contains event.start_ts. Attach its shape as col_name."""
    seg = pd.read_csv(level_csv)
    seg['start_ts'] = seg['start_ts'].astype(np.int64)
    seg['end_ts'] = seg['end_ts'].astype(np.int64)
    out = events.copy()
    out[col_name] = None
    for day, evs in tqdm(out.groupby('day'),
                          desc=f'attach {col_name}'):
        day_segs = seg[seg['day'] == day]
        if day_segs.empty:
            continue
        # Sort by start_ts
        day_segs = day_segs.sort_values('start_ts').reset_index(drop=True)
        ev_ts = evs['start_ts'].astype(np.int64).values
        # For each event ts, find the segment whose start <= ts <= end
        seg_starts = day_segs['start_ts'].values
        seg_ends = day_segs['end_ts'].values
        seg_shapes = day_segs['shape'].values
        # Use searchsorted on starts: candidate index = np.searchsorted(starts, ts) - 1
        cand = np.searchsorted(seg_starts, ev_ts, side='right') - 1
        cand = np.clip(cand, 0, len(seg_starts) - 1)
        # Verify ts <= end_ts of candidate
        valid = ev_ts <= seg_ends[cand]
        result = np.where(valid, seg_shapes[cand], None)
        out.loc[evs.index, col_name] = result
    return out


def build_table_for_key(events: pd.DataFrame, key_cols: list[str],
                         min_n: int = 10) -> pd.DataFrame:
    """Compute IS/OOS P(fwd_ret_h60m > 0) per cell defined by key_cols."""
    sub = events.dropna(subset=['fwd_ret_h60m'] + key_cols).copy()
    rows = []
    for keys, g in sub.groupby(key_cols):
        n_total = len(g)
        if n_total < min_n:
            continue
        is_g = g[g['split'] == 'IS']
        oos_g = g[g['split'] == 'OOS']
        rec = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec['n_total'] = n_total
        rec['n_is'] = len(is_g)
        rec['n_oos'] = len(oos_g)
        if len(is_g) >= 5:
            yi = is_g['fwd_ret_h60m'].values
            k = int((yi > 0).sum())
            p, lo, hi = beta_p(k, len(yi))
            rec['P_is'] = round(p, 4)
            rec['ci_is_lo'] = round(lo, 4)
            rec['ci_is_hi'] = round(hi, 4)
            rec['mean_is'] = round(float(yi.mean()), 2)
        if len(oos_g) >= 5:
            yo = oos_g['fwd_ret_h60m'].values
            k = int((yo > 0).sum())
            p, lo, hi = beta_p(k, len(yo))
            rec['P_oos'] = round(p, 4)
            rec['ci_oos_lo'] = round(lo, 4)
            rec['ci_oos_hi'] = round(hi, 4)
            rec['mean_oos'] = round(float(yo.mean()), 2)
        # sign-match
        pi = rec.get('P_is')
        po = rec.get('P_oos')
        if pi is not None and po is not None:
            rec['sign_match'] = int(np.sign(pi - 0.5) == np.sign(po - 0.5))
        else:
            rec['sign_match'] = -1
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_table(table: pd.DataFrame, label: str,
                     min_n_is: int = 10) -> dict:
    """Return summary stats for a built table."""
    valid = table[(table['n_is'] >= min_n_is) & table['P_is'].notna()]
    has_oos = valid[valid['P_oos'].notna()]
    sign_stable = has_oos[has_oos['sign_match'] == 1]
    if not has_oos.empty:
        match_rate = 100 * sign_stable.shape[0] / has_oos.shape[0]
    else:
        match_rate = float('nan')
    if not sign_stable.empty:
        edge = (sign_stable['P_is'] - 0.5).abs()
        mean_edge = round(float(edge.mean()), 4)
        max_edge = round(float(edge.max()), 4)
    else:
        mean_edge = max_edge = float('nan')
    return {
        'key_label': label,
        'n_cells': len(valid),
        'n_with_oos': len(has_oos),
        'n_sign_stable': len(sign_stable),
        'sign_match_pct': round(match_rate, 1),
        'mean_edge_when_stable': mean_edge,
        'max_edge_when_stable': max_edge,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--substrate',
                     default='reports/findings/segments/bayes_table_v0_location/event_substrate.parquet')
    ap.add_argument('--seg-dir', default='reports/findings/segments/simple_bulk_v2')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_v0_merged')
    ap.add_argument('--min-n', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading event substrate from {args.substrate}...')
    events = pd.read_parquet(args.substrate)
    print(f'  {len(events):,} events  '
           f'(IS: {(events["split"]=="IS").sum():,}, '
           f'OOS: {(events["split"]=="OOS").sum():,})')

    # Attach chord shapes at event entry timestamp
    for level, col in [('phrases', 'phrase_shape'),
                        ('motifs', 'motif_shape'),
                        ('sub_motifs', 'sub_motif_shape'),
                        ('measures', 'measure_shape')]:
        path = os.path.join(args.seg_dir, f'all_{level}.csv')
        events = attach_chord_shape(events, path, col)

    # Coverage
    chord_cols = ['phrase_shape', 'motif_shape', 'sub_motif_shape', 'measure_shape']
    n_full = events[chord_cols].notna().all(axis=1).sum()
    print(f'\nEvents with full 4-level chord attached: {n_full:,} of {len(events):,} '
           f'({100*n_full/len(events):.1f}%)')

    # Save merged substrate
    sub_path = os.path.join(args.out_dir, 'event_substrate_with_chord.parquet')
    events.to_parquet(sub_path, index=False)
    print(f'Merged substrate -> {sub_path}')

    # Build tables for each key configuration
    KEYS = {
        'LOC_4axis':   ['side', 'anchor', 'sigma_rank_q', 'slope_q'],
        'LOC_3axis':   ['side', 'anchor', 'sigma_rank_q'],
        'CHORD_3':     ['motif_shape', 'sub_motif_shape', 'measure_shape'],
        'CHORD_2':     ['sub_motif_shape', 'measure_shape'],
        'MIX_4axis':   ['side', 'anchor', 'sigma_rank_q', 'measure_shape'],
        'MIX_5axis':   ['side', 'anchor', 'sigma_rank_q', 'sub_motif_shape',
                        'measure_shape'],
        'FULL_6axis':  ['side', 'anchor', 'sigma_rank_q', 'slope_q',
                        'motif_shape', 'measure_shape'],
    }

    print(f'\n=== KEY CONFIGURATION COMPARISON ===')
    summaries = []
    tables_built = {}
    for label, key_cols in KEYS.items():
        # Drop rows with any NaN in the key cols
        for_key = events.dropna(subset=key_cols).copy()
        if 'sigma_rank_q' in key_cols:
            for_key = for_key[for_key['sigma_rank_q'] >= 0]
        if 'slope_q' in key_cols:
            for_key = for_key[for_key['slope_q'] >= 0]
        t = build_table_for_key(for_key, key_cols, min_n=args.min_n)
        tables_built[label] = t
        out_csv = os.path.join(args.out_dir, f'table_{label}.csv')
        t.to_csv(out_csv, index=False)
        s = summarize_table(t, label)
        s['n_axes'] = len(key_cols)
        s['key_cols'] = ','.join(key_cols)
        summaries.append(s)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(args.out_dir, 'key_comparison_summary.csv'),
                       index=False)
    print(summary_df[['key_label', 'n_axes', 'n_cells', 'n_with_oos',
                       'n_sign_stable', 'sign_match_pct',
                       'mean_edge_when_stable', 'max_edge_when_stable']]
          .to_string(index=False))

    # For the FULL key, show top stable cells
    print(f'\n=== TOP STABLE CELLS — FULL_6axis (joint location + chord) ===')
    t = tables_built.get('FULL_6axis')
    if t is not None and not t.empty:
        stable = t[(t['sign_match'] == 1) & (t['P_is'].notna()) & (t['P_oos'].notna())].copy()
        stable['edge'] = (stable['P_is'] - 0.5).abs()
        top = stable.sort_values('edge', ascending=False).head(20)
        cols = ['side', 'anchor', 'sigma_rank_q', 'slope_q', 'motif_shape',
                 'measure_shape', 'n_is', 'n_oos', 'P_is', 'P_oos',
                 'mean_is', 'mean_oos']
        print(top[cols].to_string(index=False))
        top.to_csv(os.path.join(args.out_dir, 'FULL_6axis_top_stable.csv'),
                    index=False)

    # MIX_4axis top cells (most useful: concise key + chord component)
    print(f'\n=== TOP STABLE CELLS — MIX_4axis (side+anchor+sigma_q+measure_shape) ===')
    t = tables_built.get('MIX_4axis')
    if t is not None and not t.empty:
        stable = t[(t['sign_match'] == 1) & (t['P_is'].notna()) & (t['P_oos'].notna())].copy()
        stable['edge'] = (stable['P_is'] - 0.5).abs()
        top = stable.sort_values('edge', ascending=False).head(20)
        cols = ['side', 'anchor', 'sigma_rank_q', 'measure_shape',
                 'n_is', 'n_oos', 'P_is', 'P_oos', 'mean_is', 'mean_oos']
        print(top[cols].to_string(index=False))
        top.to_csv(os.path.join(args.out_dir, 'MIX_4axis_top_stable.csv'),
                    index=False)

    # Final synth chart: edge distribution per key configuration
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data = []
    plot_labels = []
    for label, table in tables_built.items():
        stable = table[(table['sign_match'] == 1) & (table['P_is'].notna())
                        & (table['P_oos'].notna())]
        if stable.empty:
            continue
        edges = (stable['P_is'] - 0.5).abs().values
        plot_data.append(edges)
        plot_labels.append(f'{label}\nn_stable={len(stable)}')
    if plot_data:
        ax.boxplot(plot_data, labels=plot_labels, showfliers=True,
                    patch_artist=True,
                    boxprops=dict(facecolor='#E3F2FD', edgecolor='#0D47A1'))
        ax.set_ylabel('|P_is − 0.5|  (edge of sign-stable cells)')
        ax.set_title('Edge distribution per key configuration\n'
                      '(only sign-stable cells; higher = more predictive)',
                      fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'key_comparison_edge.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nEdge comparison chart -> {out_png}')


if __name__ == '__main__':
    main()
