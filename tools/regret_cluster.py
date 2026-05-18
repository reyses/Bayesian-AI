"""Exploratory clustering on daisy-chain regret trades — entry then exit.

Per user 2026-05-14 (sleep run):
    Cluster on the continuous state-vector features at entry, then at exit.
    KMeans on z-scored features. Profile each cluster by $/trade outcome,
    duration, direction balance, and the dominant discrete categoricals
    (d_stack, d_rail_position, d_z_15m_bin, d_fan_bin, d_slope_15m_sign).

Outputs to reports/findings/regret_oracle/:
    daisy_clusters_<name>.csv             per-trade cluster_entry + cluster_exit
    cluster_entry_summary_<name>.csv      per-entry-cluster profile
    cluster_exit_summary_<name>.csv       per-exit-cluster profile
    cluster_entry_x_exit_<name>.csv       joint entry × exit cluster counts/$

Caveats per MEMORY 2026-05-03:
    KMeans on multi-D features can false-merge unrelated patterns into the
    same cluster. This is exploratory pattern surfacing only — any cluster
    finding must be validated OOS before being used.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


OUT_DIR = Path('reports/findings/regret_oracle')
BIN_W   = 2.0

# Continuous state-vector features the entry/exit clustering uses.
# Selected: z-scores against each anchor + inter-anchor distances + slopes.
# Excluded: lookahead/target columns (mfe_*, time_to_mfe_*, mfe_velocity, etc.)
CONTINUOUS_FEATS = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m',
    'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
]
DISCRETE_FEATS = ['d_stack', 'd_z_15m_bin', 'd_rail_position',
                  'd_fan_bin', 'd_slope_15m_sign']


def hist_mode(vals: np.ndarray, bin_w: float = BIN_W) -> float:
    if len(vals) == 0:
        return float('nan')
    lo = np.floor(vals.min() / bin_w) * bin_w
    hi = np.ceil(vals.max() / bin_w) * bin_w
    if hi <= lo:
        hi = lo + bin_w
    c, e = np.histogram(vals, bins=np.arange(lo, hi + bin_w, bin_w))
    k = int(np.argmax(c))
    return float((e[k] + e[k + 1]) / 2)


def cluster_and_profile(df: pd.DataFrame, feat_prefix: str, disc_prefix: str,
                         k: int, label: str):
    """Cluster trades on prefix+CONTINUOUS_FEATS; return df_with_labels + summary."""
    feats = [f'{feat_prefix}{c}' for c in CONTINUOUS_FEATS]
    avail = [f for f in feats if f in df.columns]
    if len(avail) < 5:
        print(f'  {label}: only {len(avail)} features available — skipping')
        return None, None

    X = df[avail].astype(float).fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    df = df.copy()
    cluster_col = f'cluster_{label}'
    df[cluster_col] = labels

    # Silhouette on a sample (full computation is O(n²))
    sil = silhouette_score(Xs, labels, sample_size=min(2000, len(labels)),
                           random_state=42)
    print(f'  k={k}  silhouette={sil:.3f}  '
          f'(closer to 1.0 = better-separated clusters; <0.2 = poor)')

    # Per-cluster profile
    rows = []
    for cid in sorted(set(labels)):
        sub = df[df[cluster_col] == cid]
        # Inverse-transformed centroid (original feature units)
        centroid = scaler.inverse_transform(km.cluster_centers_[cid:cid+1])[0]
        mfes = sub['mfe_dollars'].astype(float).values
        durs = sub['time_to_mfe_min'].astype(float).values
        profile = {
            'cluster_id':           int(cid),
            'n':                    len(sub),
            'pct_of_total':         round(100 * len(sub) / len(df), 1),
            'mode_mfe_$':           round(hist_mode(mfes), 2),
            'median_mfe_$':         round(float(np.median(mfes)), 2),
            'mean_mfe_$':           round(float(mfes.mean()), 2),
            'mean_duration_min':    round(float(durs.mean()), 2),
            'pct_long':             round(100 * (sub['direction'] == 'LONG').mean(), 1),
            'mean_velocity_$/min':  round(float(sub['mfe_velocity'].mean())
                                          if 'mfe_velocity' in sub.columns else float('nan'), 2),
        }
        # Dominant discrete categoricals (the most common d_* value per cluster)
        for d in DISCRETE_FEATS:
            d_full = f'{disc_prefix}{d}'
            if d_full in sub.columns:
                vc = sub[d_full].value_counts(dropna=False)
                if len(vc) > 0:
                    profile[f'top_{d}'] = (
                        f'{vc.index[0]} ({100*vc.iloc[0]/len(sub):.0f}%)'
                    )
        # Centroid in original units (compact: just the strongest ±2 features)
        named = list(zip(avail, centroid))
        named_sorted = sorted(named, key=lambda x: abs(x[1]), reverse=True)
        profile['centroid_top_features'] = '; '.join(
            f'{n}={v:+.2f}' for n, v in named_sorted[:5]
        )
        rows.append(profile)

    summary = pd.DataFrame(rows).sort_values('mean_mfe_$', ascending=False).reset_index(drop=True)
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='daisy-chain CSV (with both entry and exit state vectors)')
    ap.add_argument('--name', default='IS_full_daisy')
    ap.add_argument('--k', type=int, default=8,
                    help='Number of clusters (default 8)')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} trades from {args.input}')

    # ── ENTRY clustering ─────────────────────────────────────────────────
    print(f'\n=== Entry-state clustering (k={args.k}) ===')
    df_entry, summary_entry = cluster_and_profile(
        df, feat_prefix='', disc_prefix='', k=args.k, label='entry')
    if summary_entry is not None:
        summary_entry.to_csv(OUT_DIR / f'cluster_entry_summary_{args.name}.csv', index=False)
        cols_show = ['cluster_id', 'n', 'pct_of_total',
                     'mode_mfe_$', 'median_mfe_$', 'mean_mfe_$',
                     'mean_duration_min', 'pct_long']
        cols_show += [c for c in summary_entry.columns if c.startswith('top_')]
        print(summary_entry[cols_show].to_string(index=False))
        print('\nCentroid top features per entry cluster:')
        for _, row in summary_entry.iterrows():
            print(f"  c{int(row.cluster_id)}: {row.centroid_top_features}")

    # ── EXIT clustering ─────────────────────────────────────────────────
    print(f'\n=== Exit-state clustering (k={args.k}) ===')
    df_exit, summary_exit = cluster_and_profile(
        df, feat_prefix='exit_', disc_prefix='exit_', k=args.k, label='exit')
    if summary_exit is not None:
        summary_exit.to_csv(OUT_DIR / f'cluster_exit_summary_{args.name}.csv', index=False)
        cols_show = ['cluster_id', 'n', 'pct_of_total',
                     'mode_mfe_$', 'median_mfe_$', 'mean_mfe_$',
                     'mean_duration_min', 'pct_long']
        cols_show += [c for c in summary_exit.columns if c.startswith('top_')]
        print(summary_exit[cols_show].to_string(index=False))
        print('\nCentroid top features per exit cluster:')
        for _, row in summary_exit.iterrows():
            print(f"  c{int(row.cluster_id)}: {row.centroid_top_features}")

    # ── Combined CSV + joint entry × exit ─────────────────────────────────
    if df_entry is not None and df_exit is not None:
        df_combined = df_exit.copy()
        df_combined['cluster_entry'] = df_entry['cluster_entry']
        df_combined.to_csv(OUT_DIR / f'daisy_clusters_{args.name}.csv', index=False)

        joint = df_combined.groupby(['cluster_entry', 'cluster_exit']).agg(
            n=('mfe_dollars', 'size'),
            mean_mfe=('mfe_dollars', 'mean'),
            mean_dur=('time_to_mfe_min', 'mean'),
            pct_long=('direction', lambda s: 100 * (s == 'LONG').mean()),
        ).round(2).reset_index().sort_values('n', ascending=False)
        joint.to_csv(OUT_DIR / f'cluster_entry_x_exit_{args.name}.csv', index=False)

        print(f'\n=== Joint entry × exit cluster cells ===')
        print(f'  Top 15 cells by trade count:')
        print(joint.head(15).to_string(index=False))
        print(f'\n  Top 15 cells by mean $/trade:')
        print(joint.sort_values('mean_mfe', ascending=False).head(15).to_string(index=False))
        print(f'\n  Most asymmetric (most LONG-skewed or SHORT-skewed):')
        joint_skew = joint.copy()
        joint_skew['skew'] = (joint_skew['pct_long'] - 50).abs()
        print(joint_skew.sort_values('skew', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    main()
