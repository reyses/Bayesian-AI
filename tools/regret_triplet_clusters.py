"""3-feature joint quantile clustering — all continuous triplets.

Per user 2026-05-16: extend the sediment process to 3-feature joints.

For each unordered triplet (A, B, C):
    qcut each into 5 quantile bins → up to 5×5×5 = 125 cells
    Per cell (n ≥ MIN_N_PER_CELL): n, mode_$, median_$, mean_$ + 95% CI

Caveat (MEMORY 2026-05-03 hard rule): with 969 triplets × ≤125 cells each =
up to ~121k cells against 7,925 trades, this is a serious multi-comparison
problem. Top cells WILL contain selection bias — single-cell findings here
need OOS validation before any selector uses them.

Output:
    triplet_cluster_table_<name>.csv     one row per (A, B, C, qa, qb, qc)
    triplet_cluster_summary_<name>.csv   per-triplet aggregate stats
"""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR_DEFAULT = Path('reports/findings/regret_oracle')
N_BINS          = 5
N_BOOT          = 4000
BIN_W_MFE       = 2.0
MIN_N_PER_CELL  = 20

CONTINUOUS_FEATS = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'bar_range', 'volume',
    'tod_minutes',
]


def hist_mode(vals: np.ndarray, bin_w: float = BIN_W_MFE) -> float:
    if len(vals) == 0: return float('nan')
    lo = np.floor(vals.min() / bin_w) * bin_w
    hi = np.ceil(vals.max() / bin_w) * bin_w
    if hi <= lo: hi = lo + bin_w
    c, e = np.histogram(vals, bins=np.arange(lo, hi + bin_w, bin_w))
    k = int(np.argmax(c))
    return float((e[k] + e[k + 1]) / 2)


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int = N_BOOT, rng=None):
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(vals) < 2:
        m = float(vals[0]); return m, m, m
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def cluster_triplet(df, fa, fb, fc, n_bins=N_BINS, rng=None):
    needed = [fa, fb, fc, 'mfe_dollars', 'direction']
    sub = df[needed].dropna(subset=[fa, fb, fc, 'mfe_dollars'])
    if len(sub) < 200:
        return []
    sub = sub.copy()
    try:
        sub['_ba'] = pd.qcut(sub[fa].astype(float), n_bins, duplicates='drop')
        sub['_bb'] = pd.qcut(sub[fb].astype(float), n_bins, duplicates='drop')
        sub['_bc'] = pd.qcut(sub[fc].astype(float), n_bins, duplicates='drop')
    except Exception:
        return []
    cats_a = list(sub['_ba'].cat.categories)
    cats_b = list(sub['_bb'].cat.categories)
    cats_c = list(sub['_bc'].cat.categories)
    idx_a = {iv: i + 1 for i, iv in enumerate(cats_a)}
    idx_b = {iv: i + 1 for i, iv in enumerate(cats_b)}
    idx_c = {iv: i + 1 for i, iv in enumerate(cats_c)}

    rows = []
    for (ba, bb, bc), cell in sub.groupby(['_ba', '_bb', '_bc'], observed=True):
        if len(cell) < MIN_N_PER_CELL:
            continue
        mfe = cell['mfe_dollars'].astype(float).values
        mode_d = hist_mode(mfe)
        mean_d, ci_lo, ci_hi = bootstrap_mean_ci(mfe, rng=rng)
        rows.append({
            'feat_a': fa, 'feat_b': fb, 'feat_c': fc,
            'qa': idx_a[ba], 'qb': idx_b[bb], 'qc': idx_c[bc],
            'a_lo': float(ba.left), 'a_hi': float(ba.right),
            'b_lo': float(bb.left), 'b_hi': float(bb.right),
            'c_lo': float(bc.left), 'c_hi': float(bc.right),
            'n':         int(len(cell)),
            'mode_$':    round(mode_d, 2),
            'median_$':  round(float(np.median(mfe)), 2),
            'mean_$':    round(mean_d, 2),
            'ci_lo_$':   round(ci_lo, 2),
            'ci_hi_$':   round(ci_hi, 2),
            'pct_long':  round(100 * (cell['direction'] == 'LONG').mean(), 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--name', default='IS_full_daisy')
    ap.add_argument('--n-bins', type=int, default=N_BINS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} trades')

    feats = [c for c in CONTINUOUS_FEATS if c in df.columns]
    triplets = list(combinations(feats, 3))
    print(f'Features: {len(feats)}    Triplets: {len(triplets)}    '
          f'(max cells per triplet: {args.n_bins**3})')

    rng = np.random.default_rng(42)
    all_rows = []
    tri_count = 0
    for i, (fa, fb, fc) in enumerate(triplets):
        rows = cluster_triplet(df, fa, fb, fc, n_bins=args.n_bins, rng=rng)
        if rows:
            all_rows.extend(rows)
            tri_count += 1
        if (i + 1) % 100 == 0:
            print(f'  ...processed {i+1}/{len(triplets)} triplets '
                  f'({len(all_rows)} cells so far)', flush=True)

    table = pd.DataFrame(all_rows)
    out_path = out_dir / f'triplet_cluster_table_{args.name}.csv'
    table.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'  total cells: {len(table)}    triplets with data: {tri_count}')
    if len(table) > 0:
        print(f'  cell-size distribution: '
              f'p25={table["n"].quantile(.25):.0f}  '
              f'p50={table["n"].quantile(.50):.0f}  '
              f'p75={table["n"].quantile(.75):.0f}  '
              f'max={table["n"].max()}')

    if table.empty:
        return

    # Per-triplet summary
    sig = table[table['ci_lo_$'] > 0]
    tri_summary = sig.groupby(['feat_a', 'feat_b', 'feat_c']).agg(
        n_cells=('mean_$', 'count'),
        best_mean=('mean_$', 'max'),
        worst_mean=('mean_$', 'min'),
        mean_swing=('mean_$', lambda v: round(v.max() - v.min(), 2)),
        avg_n_per_cell=('n', 'mean'),
    ).reset_index().sort_values('best_mean', ascending=False)
    tri_summary = tri_summary.rename(columns={
        'best_mean':  'best_mean_$',
        'worst_mean': 'worst_mean_$',
        'mean_swing': 'mean_swing_$',
    })
    tri_summary.to_csv(out_dir / f'triplet_cluster_summary_{args.name}.csv', index=False)

    print(f'\n=== Top 20 triplets by best-cell mean_$ (CI > 0) ===')
    print(tri_summary.head(20).to_string(index=False))

    print(f'\n=== Top 20 triplets by mean_$ SWING (most discriminative) ===')
    print(tri_summary.sort_values('mean_swing_$', ascending=False).head(20).to_string(index=False))

    print(f'\n=== Top 25 INDIVIDUAL cells by mean_$ (CI > 0) — note small-n risk ===')
    top = sig.sort_values('mean_$', ascending=False).head(25)
    cols = ['feat_a', 'qa', 'feat_b', 'qb', 'feat_c', 'qc',
            'n', 'mode_$', 'mean_$', 'ci_lo_$', 'ci_hi_$', 'pct_long']
    print(top[cols].to_string(index=False))


if __name__ == '__main__':
    main()
