"""Feature cluster table — 1D stratification of trades, one feature at a time.

Per user 2026-05-16: "sediment trades on each feature." For every continuous
feature, qcut into quantile bins; for every categorical, group by value;
then per bin compute the CLAUDE.md protocol stats on $/trade:

    n, mode_$ (bin $2), median_$, mean_$ + 95% bootstrap CI (4k resamples)

Plus supplemental: mean_duration_min, mean_velocity_$/min, pct_long.

Note: Trade WR and Day WR are degenerate on the daisy-chain ORACLE output
(every trade has MFE ≥ 0 by construction; no losers, no losing days). Those
metrics become meaningful only after a strategy is applied (subset selection
+ direction prediction + costs). For ORACLE feature characterization, mode_$
and mean_$ + CI are the right summary.

Output:
    feature_cluster_table_<prefix><name>.csv  one row per (feature × bin)

Stdout: feature-range summary (which features discriminate most across their
bins) + top-15 cells by mean_$ where 95% CI excludes 0.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR_DEFAULT = Path('reports/findings/regret_oracle')
N_BOOT          = 4000
BIN_W_MFE       = 2.0
N_BINS_DEFAULT  = 5
MIN_N_PER_BIN   = 20

# Entry-time continuous features. Excludes the target (mfe_*) and any
# derived columns that contain lookahead (duration_*, mfe_velocity).
CONTINUOUS_FEATS = [
    # Structural position
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    # Anchor spread / fan
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    # Velocity / slope
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    # Microstructure at entry
    'bar_range', 'volume',
    # Time-of-day (continuous, lookahead-free)
    'tod_minutes',
]
CATEGORICAL_FEATS = [
    'd_stack', 'd_z_15m_bin', 'd_rail_position',
    'd_fan_bin', 'd_slope_15m_sign',
    'direction',  # for completeness — split LONG vs SHORT outcomes
]


def hist_mode(vals: np.ndarray, bin_w: float = BIN_W_MFE) -> float:
    if len(vals) == 0:
        return float('nan')
    lo = np.floor(vals.min() / bin_w) * bin_w
    hi = np.ceil(vals.max() / bin_w) * bin_w
    if hi <= lo:
        hi = lo + bin_w
    c, e = np.histogram(vals, bins=np.arange(lo, hi + bin_w, bin_w))
    k = int(np.argmax(c))
    return float((e[k] + e[k + 1]) / 2)


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int = N_BOOT):
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(vals) < 2:
        m = float(vals[0]); return m, m, m
    rng = np.random.default_rng(42)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(vals, size=len(vals), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def stats_for(sub: pd.DataFrame, bin_label: str, feat: str,
              bin_lo=None, bin_hi=None) -> dict | None:
    if len(sub) < MIN_N_PER_BIN:
        return None
    mfe = sub['mfe_dollars'].astype(float).values
    mode_d = hist_mode(mfe)
    mean_d, ci_lo, ci_hi = bootstrap_mean_ci(mfe)
    median_d = float(np.median(mfe))
    dur = sub['time_to_mfe_min'].astype(float).values if 'time_to_mfe_min' in sub.columns else np.array([])
    vel = sub['mfe_velocity'].astype(float).values if 'mfe_velocity' in sub.columns else np.array([])
    return {
        'feature':              feat,
        'bin':                  bin_label,
        'bin_lo':               bin_lo,
        'bin_hi':               bin_hi,
        'n':                    len(sub),
        'mode_$':               round(mode_d, 2),
        'median_$':             round(median_d, 2),
        'mean_$':               round(mean_d, 2),
        'ci_lo_$':              round(ci_lo, 2),
        'ci_hi_$':              round(ci_hi, 2),
        'mean_duration_min':    round(float(dur.mean()), 2) if len(dur) else None,
        'median_duration_min':  round(float(np.median(dur)), 2) if len(dur) else None,
        'mean_velocity_$/min':  round(float(vel.mean()), 3) if len(vel) else None,
        'pct_long':             round(100 * (sub['direction'] == 'LONG').mean(), 1),
    }


def cluster_continuous(df: pd.DataFrame, feat: str, n_bins: int) -> list:
    rows = []
    if feat not in df.columns:
        return rows
    vals = df[feat].astype(float)
    if vals.isna().all():
        return rows
    try:
        labels = pd.qcut(vals, n_bins, duplicates='drop')
    except Exception:
        return rows
    df_w = df.copy()
    df_w['_bin'] = labels
    # Sort bins in order
    bin_keys = [iv for iv in df_w['_bin'].dropna().cat.categories]
    for i, interval in enumerate(bin_keys):
        sub = df_w[df_w['_bin'] == interval]
        lo, hi = float(interval.left), float(interval.right)
        bin_label = f'Q{i + 1} [{lo:.3g}, {hi:.3g}]'
        row = stats_for(sub, bin_label, feat, bin_lo=lo, bin_hi=hi)
        if row:
            rows.append(row)
    return rows


def cluster_categorical(df: pd.DataFrame, feat: str) -> list:
    rows = []
    if feat not in df.columns:
        return rows
    for val, sub in df.groupby(feat, dropna=False):
        if pd.isna(val):
            continue
        row = stats_for(sub, str(val), feat)
        if row:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--name', default='IS_full_daisy')
    ap.add_argument('--n-bins', type=int, default=N_BINS_DEFAULT,
                    help='Quantile bins per continuous feature (default 5)')
    ap.add_argument('--exit', action='store_true',
                    help='Use exit_* features instead of entry features')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} trades from {args.input}')

    if args.exit:
        cont_feats = [f'exit_{c}' for c in CONTINUOUS_FEATS
                      if f'exit_{c}' in df.columns]
        cat_feats  = [f'exit_{c}' for c in CATEGORICAL_FEATS
                      if f'exit_{c}' in df.columns]
        # Some features are entry-only (tod_minutes, volume, bar_range) —
        # if exit_* isn't there, fall back to the entry version for those
        prefix = 'exit_'
    else:
        cont_feats = [c for c in CONTINUOUS_FEATS if c in df.columns]
        cat_feats  = [c for c in CATEGORICAL_FEATS if c in df.columns]
        prefix = ''

    print(f'  features: {len(cont_feats)} continuous + {len(cat_feats)} categorical')

    all_rows = []
    for feat in cont_feats:
        all_rows.extend(cluster_continuous(df, feat, n_bins=args.n_bins))
    for feat in cat_feats:
        all_rows.extend(cluster_categorical(df, feat))

    table = pd.DataFrame(all_rows)
    out_path = out_dir / f'feature_cluster_table_{prefix}{args.name}.csv'
    table.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'  total rows: {len(table)}  '
          f'(min_n_per_bin={MIN_N_PER_BIN} filtered)')

    if table.empty:
        return

    # ── Feature-discrimination ranking ──
    # For each feature, how much does mean_$ swing across its bins?
    # A feature with wide swing is a strong stratifier.
    print(f'\n=== Per-feature discrimination (mean_$ spread across bins) ===')
    print(f'   Sorted by mean_$ range — features that separate trades most.\n')
    fr = table.groupby('feature').agg(
        n_bins=('bin', 'count'),
        n_total=('n', 'sum'),
        mean_min=('mean_$', 'min'),
        mean_max=('mean_$', 'max'),
        mode_min=('mode_$', 'min'),
        mode_max=('mode_$', 'max'),
    )
    fr['mean_range'] = (fr['mean_max'] - fr['mean_min']).round(2)
    fr['mode_range'] = (fr['mode_max'] - fr['mode_min']).round(2)
    fr = fr.sort_values('mean_range', ascending=False)
    print(fr[['n_bins', 'mean_min', 'mean_max', 'mean_range',
              'mode_min', 'mode_max', 'mode_range']].to_string())

    # ── Top single-bin cells where CI is positive ──
    print(f'\n=== Top 15 cells by mean_$ (ci_lo > 0) ===')
    sig = table[table['ci_lo_$'] > 0].sort_values('mean_$', ascending=False)
    cols = ['feature', 'bin', 'n', 'mode_$', 'median_$',
            'mean_$', 'ci_lo_$', 'ci_hi_$', 'pct_long']
    print(sig[cols].head(15).to_string(index=False))

    # ── Bottom cells (smallest mean_$) — the "weakest" sediment ──
    print(f'\n=== Bottom 10 cells by mean_$ (ci_lo > 0) ===')
    bot = table[table['ci_lo_$'] > 0].sort_values('mean_$', ascending=True)
    print(bot[cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
