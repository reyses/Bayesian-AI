"""Paired feature stratification — all continuous × continuous pairs.

Per user 2026-05-16: same sediment process as 1D, now JOINT quantile bins
for each unordered pair of continuous features. 5×5 = 25 cells per pair.

For each pair (A, B):
    qcut A into 5 bins, qcut B into 5 bins → up to 25 cells
    Per cell (n ≥ MIN_N_PER_CELL): n, mode_$, median_$, mean_$ + 95% boot CI

Caveat (MEMORY 2026-05-03): quantile-cell selection on this many cells
(~171 pairs × 25 cells = 4,275 cells) is a massive multiple-comparisons
problem. Single-cell findings are NOT trustworthy in isolation — look for
patterns across cells within a pair, and OOS-validate any selection.

Output:
    pair_cluster_table_<name>.csv    one row per (feat_a, feat_b, bin_a, bin_b)
    pair_cluster_summary_<name>.csv  one row per pair with summary stats
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
    """Vectorized bootstrap — ~10× faster than the python-loop version."""
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


def cluster_pair(df: pd.DataFrame, fa: str, fb: str, n_bins: int = N_BINS,
                 rng=None) -> list:
    needed = [fa, fb, 'mfe_dollars', 'direction']
    if 'time_to_mfe_min' in df.columns:
        needed.append('time_to_mfe_min')
    sub = df[needed].dropna(subset=[fa, fb, 'mfe_dollars'])
    if len(sub) < 200:
        return []
    sub = sub.copy()
    try:
        sub['_ba'] = pd.qcut(sub[fa].astype(float), n_bins, duplicates='drop')
        sub['_bb'] = pd.qcut(sub[fb].astype(float), n_bins, duplicates='drop')
    except Exception:
        return []
    cats_a = list(sub['_ba'].cat.categories)
    cats_b = list(sub['_bb'].cat.categories)
    idx_a = {iv: i + 1 for i, iv in enumerate(cats_a)}
    idx_b = {iv: i + 1 for i, iv in enumerate(cats_b)}

    rows = []
    for (ba, bb), cell in sub.groupby(['_ba', '_bb'], observed=True):
        if len(cell) < MIN_N_PER_CELL:
            continue
        mfe = cell['mfe_dollars'].astype(float).values
        mode_d = hist_mode(mfe)
        mean_d, ci_lo, ci_hi = bootstrap_mean_ci(mfe, rng=rng)
        dur = (round(float(cell['time_to_mfe_min'].mean()), 2)
               if 'time_to_mfe_min' in cell.columns else None)
        rows.append({
            'feat_a':       fa,
            'feat_b':       fb,
            'qa':           idx_a[ba],
            'qb':           idx_b[bb],
            'bin_a':        f'Q{idx_a[ba]} [{ba.left:.3g}, {ba.right:.3g}]',
            'bin_b':        f'Q{idx_b[bb]} [{bb.left:.3g}, {bb.right:.3g}]',
            'a_lo':         float(ba.left),
            'a_hi':         float(ba.right),
            'b_lo':         float(bb.left),
            'b_hi':         float(bb.right),
            'n':            int(len(cell)),
            'mode_$':       round(mode_d, 2),
            'median_$':     round(float(np.median(mfe)), 2),
            'mean_$':       round(mean_d, 2),
            'ci_lo_$':      round(ci_lo, 2),
            'ci_hi_$':      round(ci_hi, 2),
            'pct_long':     round(100 * (cell['direction'] == 'LONG').mean(), 1),
            'mean_dur_min': dur,
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
    n_pairs = len(feats) * (len(feats) - 1) // 2
    print(f'Features: {len(feats)}    Pairs to evaluate: {n_pairs}')

    rng = np.random.default_rng(42)
    all_rows = []
    pair_count = 0
    for i, (fa, fb) in enumerate(combinations(feats, 2)):
        rows = cluster_pair(df, fa, fb, n_bins=args.n_bins, rng=rng)
        if rows:
            all_rows.extend(rows)
            pair_count += 1
        if (i + 1) % 25 == 0:
            print(f'  ...processed {i+1}/{n_pairs} pairs', flush=True)

    table = pd.DataFrame(all_rows)
    out_path = out_dir / f'pair_cluster_table_{args.name}.csv'
    table.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'  total cells: {len(table)}    pairs with data: {pair_count}')

    if table.empty:
        return

    # ── Per-pair summary (use rename instead of $-named kwargs) ──
    sig = table[table['ci_lo_$'] > 0]
    pair_summary = sig.groupby(['feat_a', 'feat_b']).agg(
        n_cells=('mean_$', 'count'),
        best_mean=('mean_$', 'max'),
        worst_mean=('mean_$', 'min'),
        mean_swing=('mean_$', lambda v: round(v.max() - v.min(), 2)),
        mode_swing=('mode_$', lambda v: round(v.max() - v.min(), 2)),
        avg_n_per_cell=('n', 'mean'),
    ).reset_index().sort_values('best_mean', ascending=False)
    pair_summary = pair_summary.rename(columns={
        'best_mean':  'best_mean_$',
        'worst_mean': 'worst_mean_$',
        'mean_swing': 'mean_swing_$',
        'mode_swing': 'mode_swing_$',
    })
    pair_summary.to_csv(out_dir / f'pair_cluster_summary_{args.name}.csv', index=False)

    print(f'\n=== Top 20 pairs by best-cell mean_$ (CI > 0) ===')
    print(pair_summary.head(20).to_string(index=False))

    print(f'\n=== Top 20 pairs by mean_$ SWING (max-min across cells) — most discriminative ===')
    swing_sort = pair_summary.sort_values('mean_swing_$', ascending=False)
    print(swing_sort.head(20).to_string(index=False))

    print(f'\n=== Top 25 INDIVIDUAL cells by mean_$ (CI > 0) ===')
    top = sig.sort_values('mean_$', ascending=False).head(25)
    cols = ['feat_a', 'bin_a', 'feat_b', 'bin_b', 'n',
            'mode_$', 'mean_$', 'ci_lo_$', 'ci_hi_$', 'pct_long']
    print(top[cols].to_string(index=False))


if __name__ == '__main__':
    main()
