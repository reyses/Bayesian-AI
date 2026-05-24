"""Generalized k-way feature stratification — DIRECTION-aware analysis.

Per user 2026-05-16: "the most important signal we need is direction."

Target = signed_mfe = mfe_dollars × (+1 if LONG else −1).
    Sign      → direction (positive cells are LONG-skewed)
    Magnitude → trade size
Cells with |mean_signed| high AND pct_long far from 50% are direction-callable.

For each k-tuple:
    Clustering: joint quantile bins (configurable bin count)
        Per cell: n, pct_long + Wilson CI, mean_signed + 95% bootstrap CI,
                   mode_signed, mean_$_magnitude (mfe_dollars for context),
                   long_callable (Wilson lower > 70%) / short_callable flags
    Regression: y = signed_mfe ~ all interactions up to max-order
        Reports R² at each interaction order + marginal gain

Run for each k:
    k=2: --n-bins 5  (5×5 = 25 cells per pair, 171 pairs)
    k=3: --n-bins 5  (125 cells per triplet, 969 triplets)
    k=4: --n-bins 3  (81 cells per quad, 3876 quads — bin reduction for n)
    k=5: --n-bins 2  (32 cells per quint, 11628 quints — bin reduction for n)

Output:
    kway_<k>_clusters_<name>.csv     one row per cell
    kway_<k>_regression_<name>.csv   one row per k-tuple
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR_DEFAULT = Path('reports/findings/regret_oracle')
MIN_N_PER_CELL  = 20
N_BOOT          = 2000           # smaller bootstrap for speed at scale
BIN_W_SIGNED    = 10.0           # signed-mfe ranges roughly ±2500; bin = $10

CONTINUOUS_FEATS = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'bar_range', 'volume',
    'tod_minutes',
]


def hist_mode(vals: np.ndarray, bin_w: float = BIN_W_SIGNED) -> float:
    if len(vals) == 0: return float('nan')
    lo = np.floor(vals.min() / bin_w) * bin_w
    hi = np.ceil(vals.max() / bin_w) * bin_w
    if hi <= lo: hi = lo + bin_w
    c, e = np.histogram(vals, bins=np.arange(lo, hi + bin_w, bin_w))
    k = int(np.argmax(c))
    return float((e[k] + e[k + 1]) / 2)


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int = N_BOOT, rng=None):
    if len(vals) < 2:
        m = float(vals[0]) if len(vals) else float('nan')
        return m, m, m
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n < 1: return float('nan'), float('nan')
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def cluster_ktuple(df: pd.DataFrame, feats: tuple, n_bins: int, rng):
    feats = list(feats)
    needed = feats + ['signed_mfe', 'mfe_dollars', 'direction']
    sub = df[needed].dropna(subset=feats + ['signed_mfe'])
    if len(sub) < 200:
        return []
    sub = sub.copy()
    bin_cols = []
    for f in feats:
        col = f'_b_{f}'
        try:
            sub[col] = pd.qcut(sub[f].astype(float), n_bins, duplicates='drop')
        except Exception:
            return []
        bin_cols.append(col)
    idx_maps = [{iv: i + 1 for i, iv in enumerate(sub[c].cat.categories)} for c in bin_cols]

    rows = []
    for key, cell in sub.groupby(bin_cols, observed=True):
        if len(cell) < MIN_N_PER_CELL:
            continue
        signed = cell['signed_mfe'].astype(float).values
        mfe    = cell['mfe_dollars'].astype(float).values
        pct_long = float((cell['direction'] == 'LONG').mean())
        pl_lo, pl_hi = wilson_ci(pct_long, len(cell))
        sm_mean, sm_lo, sm_hi = bootstrap_mean_ci(signed, rng=rng)
        sm_mode = hist_mode(signed)

        if not isinstance(key, tuple):
            key = (key,)
        row = {f'feat_{i + 1}': feats[i] for i in range(len(feats))}
        for i, (k_val, m) in enumerate(zip(key, idx_maps)):
            row[f'q_{i + 1}'] = m[k_val]
        row.update({
            'n':                  int(len(cell)),
            'pct_long':           round(100 * pct_long, 1),
            'pct_long_ci_lo':     round(100 * pl_lo, 1),
            'pct_long_ci_hi':     round(100 * pl_hi, 1),
            'mode_signed':        round(sm_mode, 1),
            'mean_signed':        round(sm_mean, 2),
            'mean_signed_ci_lo':  round(sm_lo, 2),
            'mean_signed_ci_hi':  round(sm_hi, 2),
            'mean_$_magnitude':   round(float(mfe.mean()), 2),
            'long_callable':      bool(pl_lo > 0.70),
            'short_callable':     bool(pl_hi < 0.30),
        })
        rows.append(row)
    return rows


def regress_ktuple(df: pd.DataFrame, feats: tuple, max_order: int):
    feats = list(feats)
    sub = df[feats + ['signed_mfe']].dropna()
    if len(sub) < 200:
        return None
    y = sub['signed_mfe'].astype(float).values
    arrs = [sub[f].astype(float).values for f in feats]
    stds = [float(np.std(a)) for a in arrs]
    if min(stds) == 0:
        return None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    one = np.ones(len(y))

    cols_by_order = {1: [one] + arrs}
    for order in range(2, max_order + 1):
        prev = cols_by_order[order - 1]
        new = list(prev)
        for indices in itertools.combinations(range(len(feats)), order):
            prod = arrs[indices[0]].copy()
            for i in indices[1:]:
                prod = prod * arrs[i]
            new.append(prod)
        cols_by_order[order] = new

    r2_by_order = {}
    for order, cols in cols_by_order.items():
        X = np.column_stack(cols)
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coefs
        ss_res = float(np.sum((y - y_pred) ** 2))
        r2_by_order[order] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'feats':  '|'.join(feats),
        'n':      int(len(sub)),
        **{f'r2_order_{o}': round(r, 4) for o, r in r2_by_order.items()},
        **{f'gain_order_{o}': round(r2_by_order[o] - r2_by_order[o - 1], 4)
           for o in range(2, max_order + 1)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV or parquet')
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--n-bins', type=int, required=True)
    ap.add_argument('--name', default='IS_full_daisy')
    ap.add_argument('--max-interaction-order', type=int, default=None)
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--features-file', default=None,
                    help='Optional text file with feature names (one per line); '
                         'overrides CONTINUOUS_FEATS default')
    ap.add_argument('--top-n', type=int, default=None,
                    help='If features-file given, take only the first top-n')
    args = ap.parse_args()

    max_order = args.max_interaction_order or args.k
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect CSV vs parquet
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    if 'signed_mfe' not in df.columns:
        df['signed_mfe'] = df['mfe_dollars'] * np.where(df['direction'] == 'LONG', 1, -1)
    print(f'Loaded {len(df)} trades. k={args.k}  n_bins={args.n_bins}  max_order={max_order}')

    # Feature list: from --features-file (one per line) or CONTINUOUS_FEATS default
    if args.features_file:
        with open(args.features_file) as f:
            wanted = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
        if args.top_n:
            wanted = wanted[:args.top_n]
        feats = [c for c in wanted if c in df.columns]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            print(f'  WARN: {len(missing)} features in file not in CSV (e.g., {missing[:3]})')
        print(f'  Using {len(feats)} features from --features-file')
    else:
        feats = [c for c in CONTINUOUS_FEATS if c in df.columns]
    ktuples = list(itertools.combinations(feats, args.k))
    print(f'Features: {len(feats)}    k-tuples: {len(ktuples)}    '
          f'(max cells per tuple: {args.n_bins ** args.k})')

    rng = np.random.default_rng(42)

    # ── Clustering ──
    print('\n--- CLUSTERING ---')
    cluster_rows = []
    chk = max(1, len(ktuples) // 10)
    for i, ft in enumerate(ktuples):
        rows = cluster_ktuple(df, ft, args.n_bins, rng)
        cluster_rows.extend(rows)
        if (i + 1) % chk == 0:
            print(f'  ...{i+1}/{len(ktuples)} '
                  f'({len(cluster_rows)} cells so far)', flush=True)

    cl_table = pd.DataFrame(cluster_rows)
    cl_path = out_dir / f'kway_{args.k}_clusters_{args.name}.csv'
    cl_table.to_csv(cl_path, index=False)
    print(f'  Wrote: {cl_path}    ({len(cl_table)} cells)')

    # ── Regression ──
    print('\n--- REGRESSION ---')
    reg_rows = []
    chk_r = max(1, len(ktuples) // 5)
    for i, ft in enumerate(ktuples):
        r = regress_ktuple(df, ft, max_order)
        if r:
            reg_rows.append(r)
        if (i + 1) % chk_r == 0:
            print(f'  ...{i+1}/{len(ktuples)}', flush=True)

    reg_table = pd.DataFrame(reg_rows)
    reg_path = out_dir / f'kway_{args.k}_regression_{args.name}.csv'
    reg_table.to_csv(reg_path, index=False)
    print(f'  Wrote: {reg_path}    ({len(reg_table)} regressions)')

    # ── Top direction-callable cells ──
    if not cl_table.empty:
        long_callable  = cl_table[cl_table['long_callable']]
        short_callable = cl_table[cl_table['short_callable']]
        print(f'\n=== Direction-callable cells ===')
        print(f'  LONG-callable  (pct_long_ci_lo > 70%): {len(long_callable)} of {len(cl_table)} '
              f'({100*len(long_callable)/max(len(cl_table),1):.1f}%)')
        print(f'  SHORT-callable (pct_long_ci_hi < 30%): {len(short_callable)} of {len(cl_table)} '
              f'({100*len(short_callable)/max(len(cl_table),1):.1f}%)')

        if len(long_callable) > 0:
            print(f'\nTop 10 LONG-callable cells by mean_signed:')
            top_l = long_callable.sort_values('mean_signed', ascending=False).head(10)
            feat_q_cols = sorted([c for c in top_l.columns if c.startswith('feat_') or c.startswith('q_')])
            cols = feat_q_cols + ['n', 'pct_long', 'pct_long_ci_lo',
                                  'mean_signed', 'mean_$_magnitude']
            print(top_l[cols].to_string(index=False))

        if len(short_callable) > 0:
            print(f'\nTop 10 SHORT-callable cells by |mean_signed|:')
            top_s = short_callable.sort_values('mean_signed', ascending=True).head(10)
            feat_q_cols = sorted([c for c in top_s.columns if c.startswith('feat_') or c.startswith('q_')])
            cols = feat_q_cols + ['n', 'pct_long', 'pct_long_ci_hi',
                                  'mean_signed', 'mean_$_magnitude']
            print(top_s[cols].to_string(index=False))

    # ── Top regressions ──
    if not reg_table.empty:
        r2_col = f'r2_order_{max_order}'
        if r2_col in reg_table.columns:
            print(f'\n=== Top 10 k-tuples by R² (max-order={max_order}) ===')
            top_r = reg_table.sort_values(r2_col, ascending=False).head(10)
            print(top_r.to_string(index=False))


if __name__ == '__main__':
    main()
