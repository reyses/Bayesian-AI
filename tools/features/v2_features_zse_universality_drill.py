"""
v2_features_zse_universality_drill.py — Drill Surprise #1: why is z_se_w
the universal contextualizer (mean lift 0.28, top of D8 ranking)?

Sub-questions:
  A. Is z_se_w just a regime proxy in disguise?
       Cross-tab z_se_w quintile vs regime_2d — does extreme z_se_w
       overwhelmingly land in one regime?
  B. Does z_se_w contextualize feature -> price relationships too
       (mirror of D8 but with forward-return as target)?
       If yes, z_se_w is a TRADING contextualizer, not just a structural one.
  C. Compare z_se_w to its peer regression metrics (z_high_w, z_low_w,
       SE_high_w, SE_low_w, hurst_w). Why does z_se_w dominate?

Outputs:
  reports/findings/v2_drill_zse_universality/
    A_zse_regime_crosstab.csv     z_se_w quintile vs regime_2d distribution
    B_zse_price_lift.csv          per (X, TF, z_se_w_q) corr(X, fwd_ret)
                                   — analog of D8 but with price as Y
    C_modifier_compare.csv        z_se_w vs z_high/low/SE_high/SE_low/hurst
                                   side-by-side mean lift
    summary.md
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_tf_sweep_eda import feature_column_for


CONCEPTS = [
    'price_velocity_1b', 'price_accel_1b',
    'vol_velocity_1b',   'vol_accel_1b',
    'bar_range', 'body',
    'price_velocity_w', 'price_accel_w',
    'vol_velocity_w',   'vol_accel_w',
    'price_mean_w', 'price_sigma_w',
    'vol_mean_w',   'vol_sigma_w',
    'vwap_w',
    'z_se_w', 'z_high_w', 'z_low_w',
    'SE_high_w', 'SE_low_w',
    'hurst_w', 'reversion_prob_w', 'swing_noise_w',
]
DEFAULT_TFS = ['5s', '1m', '5m', '15m', '1h']
PEER_MODIFIERS = ['z_se_w', 'z_high_w', 'z_low_w',
                    'SE_high_w', 'SE_low_w', 'hurst_w']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_drill_zse_universality')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    Q = args.quantiles
    print(f"{'='*70}")
    print(f"  Drill: z_se_w universality")
    print(f"{'='*70}")

    print(f"\n--- Loading data ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int
    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base_df['date'] = dt_la.dt.date.astype(str)

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    close = full['close'].values.astype(np.float64)
    n_total = len(close)
    fwd = np.full(n_total, np.nan)
    if n_total > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    regimes = full['regime_2d'].values.astype(str)

    # ---- (A) z_se_w quintile x regime cross-tab ----
    print(f"\n--- (A) z_se_w quintile x regime distribution ---")
    a_rows = []
    for tf in args.tfs:
        col = feature_column_for('z_se_w', tf)
        if col not in full.columns:
            continue
        v = full[col].values.astype(np.float64)
        valid = ~np.isnan(v)
        qs = np.quantile(v[valid], np.linspace(0, 1, Q + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        bins = np.digitize(v, qs[1:-1])
        for q in range(Q):
            for regime in REGIME_2D_ORDER:
                cell = (bins == q) & (regimes == regime) & valid
                a_rows.append({
                    'tf': tf,
                    'q_z_se': q,
                    'regime_2d': regime,
                    'n': int(cell.sum()),
                })
    a_df = pd.DataFrame(a_rows)
    a_df.to_csv(os.path.join(args.output_dir, 'A_zse_regime_crosstab.csv'),
                  index=False)
    print(f"  [saved] A_zse_regime_crosstab.csv")

    # Print pivot per TF: rows=q_z_se, cols=regime, values=fraction
    print(f"\n  z_se_w quintile composition by regime (% of bars in quintile):")
    for tf in args.tfs:
        sub = a_df[a_df['tf'] == tf]
        if sub.empty:
            continue
        pv = sub.pivot(index='q_z_se', columns='regime_2d', values='n').fillna(0)
        pv = pv.reindex(columns=[r for r in REGIME_2D_ORDER if r in pv.columns])
        pct = pv.div(pv.sum(axis=1), axis=0) * 100
        print(f"\n  TF={tf}:")
        for q in pct.index:
            row_str = '  '.join(f'{r}={pct.loc[q, r]:.1f}%' for r in pct.columns)
            print(f"    Q{q}: {row_str}")

    # If z_se_w is just a regime proxy, extreme quintiles should heavily
    # concentrate in one regime. If quintiles are roughly uniform across
    # regimes, z_se_w carries info beyond regime label.

    # ---- (B) z_se_w as price contextualizer ----
    # For each (X, TF, z_se_w bin): corr(X, fwd_return)
    # Check: does z_se_w bin shift price corr in the same way it shifted
    # feature×feature corr in D8?
    print(f"\n--- (B) z_se_w as price-target contextualizer ---")
    b_rows = []
    for tf in args.tfs:
        col_z = feature_column_for('z_se_w', tf)
        if col_z not in full.columns:
            continue
        z_arr = full[col_z].values.astype(np.float64)
        valid_z = ~np.isnan(z_arr)
        if valid_z.sum() < Q * 5:
            continue
        qs = np.quantile(z_arr[valid_z], np.linspace(0, 1, Q + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        z_bins = np.digitize(z_arr, qs[1:-1])

        for x_concept in CONCEPTS:
            if x_concept == 'z_se_w':
                continue
            col_x = feature_column_for(x_concept, tf)
            if col_x not in full.columns:
                continue
            x = full[col_x].values.astype(np.float64)
            valid_xy = ~np.isnan(x) & ~np.isnan(fwd) & valid_z
            if valid_xy.sum() < args.min_cell_n * Q:
                continue
            for q in range(Q):
                m = valid_xy & (z_bins == q)
                if m.sum() < args.min_cell_n:
                    continue
                xv, yv = x[m], fwd[m]
                if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                    continue
                r = float(np.corrcoef(xv, yv)[0, 1])
                b_rows.append({
                    'tf': tf,
                    'X': x_concept,
                    'q_z': q,
                    'n': int(m.sum()),
                    'corr_x_fwd': r,
                })
    b_df = pd.DataFrame(b_rows)
    b_df.to_csv(os.path.join(args.output_dir, 'B_zse_price_lift.csv'),
                  index=False)
    print(f"  [saved] B_zse_price_lift.csv ({len(b_df)} rows)")

    # Per (X, TF), compute corr range across z bins
    grouped = b_df.groupby(['tf', 'X'])
    b_lift_rows = []
    for (tf, X), g in grouped:
        rs = g['corr_x_fwd'].values
        if len(rs) < 2:
            continue
        r_min, r_max = float(rs.min()), float(rs.max())
        b_lift_rows.append({
            'tf': tf,
            'X': X,
            'r_min': r_min,
            'r_max': r_max,
            'lift': r_max - r_min,
            'n_bins': len(rs),
            'sign_flip': (rs.min() < -0.05) and (rs.max() > 0.05),
        })
    b_lift = pd.DataFrame(b_lift_rows).sort_values('lift', ascending=False)
    b_lift.to_csv(os.path.join(args.output_dir, 'B_zse_price_lift_summary.csv'),
                    index=False)

    print(f"\n  Top 20 features X whose corr(X, fwd_ret) is most contextualized "
          f"by z_se_w:")
    print(f"    {'tf':>4}  {'X':>22}  {'r_min':>7}  {'r_max':>7}  {'lift':>5}  {'flip':>4}")
    for _, r in b_lift.head(20).iterrows():
        print(f"    {r['tf']:>4}  {r['X']:>22}  "
              f"{r['r_min']:>+7.3f}  {r['r_max']:>+7.3f}  {r['lift']:>5.3f}  "
              f"{'YES' if r['sign_flip'] else 'no':>4}")

    print(f"\n  z_se_w price-contextualization sign-flip rate: "
          f"{int(b_lift['sign_flip'].sum())} / {len(b_lift)} "
          f"({100.0*b_lift['sign_flip'].sum()/max(len(b_lift),1):.1f}%)")

    # ---- (C) Compare z_se_w to peer regression-related modifiers ----
    print(f"\n--- (C) z_se_w vs peer regression-related modifiers (D8 lift)---")
    # Read D8 modifier influence ranking
    d8_path = 'reports/findings/v2_features_within_tf_contextualizer/modifier_influence.csv'
    if os.path.exists(d8_path):
        mi = pd.read_csv(d8_path)
        peer_mi = mi[mi['Z'].isin(PEER_MODIFIERS)].copy()
        peer_mi = peer_mi.sort_values('mean_lift', ascending=False)
        peer_mi.to_csv(os.path.join(args.output_dir, 'C_modifier_compare.csv'),
                          index=False)
        print(f"  Peer comparison (mean lift across triplets they modify):")
        for _, r in peer_mi.iterrows():
            print(f"    {r['Z']:>16}  mean={r['mean_lift']:>5.3f}  "
                  f"max={r['max_lift']:>5.3f}  n={int(r['n_triplets'])}")

        # z_se_w score vs runner-up
        peer_mi_sorted = peer_mi.sort_values('mean_lift', ascending=False).reset_index(drop=True)
        if len(peer_mi_sorted) >= 2:
            top = peer_mi_sorted.iloc[0]
            second = peer_mi_sorted.iloc[1]
            print(f"\n  Top peer: {top['Z']} (lift {top['mean_lift']:.3f})")
            print(f"  Second:   {second['Z']} (lift {second['mean_lift']:.3f})")
            print(f"  Gap: {top['mean_lift']/second['mean_lift']:.2f}x")

    # ---- Markdown ----
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Drill: z_se_w universality - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"## A. z_se_w quintile x regime distribution\n\n")
        f.write(f"If z_se_w is a regime proxy, extreme quintiles concentrate in "
                f"one regime. If uniform, z_se_w carries info beyond regime label.\n\n")
        for tf in args.tfs:
            sub = a_df[a_df['tf'] == tf]
            if sub.empty:
                continue
            pv = sub.pivot(index='q_z_se', columns='regime_2d',
                              values='n').fillna(0)
            pv = pv.reindex(columns=[r for r in REGIME_2D_ORDER if r in pv.columns])
            pct = pv.div(pv.sum(axis=1), axis=0) * 100
            f.write(f"### TF={tf}\n\n")
            f.write(pct.round(1).to_string())
            f.write("\n\n")
        f.write(f"## B. z_se_w as price contextualizer\n\n")
        f.write(f"Top-20 features whose corr(X, fwd_ret) is most modified "
                f"by z_se_w bin:\n\n")
        f.write(b_lift.head(30).to_string(index=False))
        f.write(f"\n\n## C. Peer comparison\n\n")
        if 'peer_mi' in locals():
            f.write(peer_mi.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
