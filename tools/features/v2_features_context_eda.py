"""
v2_features_context_eda.py — Contextualization analysis.

User: "there might be features that do not directly relate to price but
they might contextualize other features".

Volume doesn't predict price direction, but it modifies the meaning of a
velocity move (high vol = conviction; low vol = noise). Hurst doesn't
predict direction but conditions whether mean-reversion or trend-following
is the right framework.

This tool measures, for each (modifier_concept, target_concept) pair at a
chosen TF:
  - target's price-relationship STRENGTH within each modifier-quantile bin
  - delta = max(strength_per_bin) - min(strength_per_bin)
  - "context_lift" = how much the modifier changes the target's signal

Specifically: bin modifier into Q quantiles. Within each modifier-bin,
compute target's |corr_with_forward_return| and |cohen_d UP vs DOWN|.
The pair's context score = how much these vary across modifier bins.

A high context score means the modifier MEANINGFULLY CHANGES the target's
price relationship. A low score means the modifier is irrelevant for that
target.

Modifier candidates (typically don't track price velocity directly):
  vol_mean_w, vol_sigma_w, vol_velocity_w, vol_accel_w
  hurst_w, reversion_prob_w, swing_noise_w
  bar_range, price_sigma_w
  time_of_day (L0)

Target candidates (typically the directional signals):
  price_velocity_1b, price_velocity_w, body, price_accel_w
  z_se_w, z_high_w, z_low_w
  vwap_w, price_mean_w  (autoregressive but treated as targets here)

Output: ranked list of (modifier, target, tf) tuples with their
context_lift score.

Usage:
  python tools/v2_features_context_eda.py
  python tools/v2_features_context_eda.py --tf 5m --quantiles 4
"""

from __future__ import annotations
import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels


# Default modifiers: features that don't directly track price velocity
DEFAULT_MODIFIERS = [
    'vol_mean_w', 'vol_sigma_w', 'vol_velocity_w', 'vol_accel_w',
    'hurst_w', 'reversion_prob_w', 'swing_noise_w',
    'bar_range', 'price_sigma_w',
]

# Default targets: features that may carry direction
DEFAULT_TARGETS = [
    'price_velocity_1b', 'price_velocity_w', 'body',
    'price_accel_w', 'price_accel_1b',
    'z_se_w', 'z_high_w', 'z_low_w',
]


def feature_column_for(concept: str, tf: str) -> str:
    if concept.endswith('_1b') or concept in ('bar_range', 'body'):
        return f'L1_{tf}_{concept}'
    if concept.endswith('_w'):
        l3_set = {'z_se_w', 'z_high_w', 'z_low_w', 'SE_high_w', 'SE_low_w',
                   'hurst_w', 'reversion_prob_w', 'swing_noise_w'}
        if concept in l3_set:
            return f'L3_{tf}_{concept}'
        return f'L2_{tf}_{concept}'
    return concept


def quantile_bins(values: np.ndarray, q: int) -> np.ndarray:
    valid = ~np.isnan(values)
    out = np.full(len(values), -1, dtype=np.int8)
    if valid.sum() < q * 5:
        return out
    qs = np.quantile(values[valid], np.linspace(0, 1, q + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_idx = np.digitize(values[valid], qs[1:-1])
    out[valid] = bin_idx.astype(np.int8)
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 30:
        return float('nan')
    a = a[mask]; b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze_context_pair(modifier: np.ndarray, target: np.ndarray,
                          fwd: np.ndarray, q: int) -> dict:
    """For one (modifier, target) pair: target's correlation w/ fwd return
    PER modifier-quantile bin."""
    bm = quantile_bins(modifier, q)
    if (bm >= 0).sum() < q * 30:
        return {'per_bin': [], 'context_lift_corr': float('nan'),
                'context_lift_cohen': float('nan')}

    bin_rows = []
    for ib in range(q):
        mask = (bm == ib)
        if mask.sum() < 30:
            continue
        sub_target = target[mask]
        sub_fwd = fwd[mask]
        # Target's signed correlation with forward return WITHIN this modifier bin
        r = safe_corr(sub_target, sub_fwd)
        # Target's correlation MAGNITUDE
        bin_rows.append({
            'modifier_q': ib,
            'n': int(mask.sum()),
            'target_mean': float(np.nanmean(sub_target)),
            'target_std': float(np.nanstd(sub_target)),
            'target_corr_fwd': r,
        })
    if len(bin_rows) < 2:
        return {'per_bin': [], 'context_lift_corr': float('nan'),
                'context_lift_cohen': float('nan')}

    corrs = [b['target_corr_fwd'] for b in bin_rows
              if not np.isnan(b['target_corr_fwd'])]
    if len(corrs) < 2:
        return {'per_bin': bin_rows, 'context_lift_corr': float('nan'),
                'context_lift_cohen': float('nan')}

    # Lift in correlation across bins (range of corr values)
    context_lift_corr = float(max(corrs) - min(corrs))
    # Lift in absolute correlation (does modifier amplify or kill the signal?)
    abs_corrs = [abs(c) for c in corrs]
    context_amplification = float(max(abs_corrs) - min(abs_corrs))

    return {
        'per_bin': bin_rows,
        'context_lift_corr': context_lift_corr,
        'context_amplification': context_amplification,
        'n_bins': len(bin_rows),
        'min_corr': float(min(corrs)),
        'max_corr': float(max(corrs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=['5m', '15m', '1h'],
                        help='TFs to consider (modifier and target taken from same TF)')
    parser.add_argument('--modifiers', nargs='+', default=DEFAULT_MODIFIERS)
    parser.add_argument('--targets', nargs='+', default=DEFAULT_TARGETS)
    parser.add_argument('--quantiles', type=int, default=4)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_context')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features Contextualization EDA")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  TFs to analyze: {args.tfs}")
    print(f"  Modifiers: {args.modifiers}")
    print(f"  Targets: {args.targets}")
    print(f"  Quantiles: {args.quantiles}")
    print(f"{'='*70}")

    # Load
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
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]

    print(f"\n--- Analyzing (modifier x target x TF) cells ---")
    pair_rows = []
    bin_rows_all = []

    for tf in args.tfs:
        for mod_concept in args.modifiers:
            mod_col = feature_column_for(mod_concept, tf)
            if mod_col not in full.columns:
                continue
            mod_v = full[mod_col].values.astype(np.float64)
            for tgt_concept in args.targets:
                if mod_concept == tgt_concept:
                    continue
                tgt_col = feature_column_for(tgt_concept, tf)
                if tgt_col not in full.columns:
                    continue
                tgt_v = full[tgt_col].values.astype(np.float64)

                result = analyze_context_pair(mod_v, tgt_v, fwd, args.quantiles)
                if not result['per_bin']:
                    continue

                pair_rows.append({
                    'tf': tf,
                    'modifier': mod_concept,
                    'target': tgt_concept,
                    'n_bins': result['n_bins'],
                    'min_target_corr_fwd': result['min_corr'],
                    'max_target_corr_fwd': result['max_corr'],
                    'context_lift_corr': result['context_lift_corr'],
                    'context_amplification': result['context_amplification'],
                })
                for br in result['per_bin']:
                    bin_rows_all.append({
                        'tf': tf, 'modifier': mod_concept, 'target': tgt_concept,
                        **br,
                    })

    pair_df = pd.DataFrame(pair_rows)
    bin_df = pd.DataFrame(bin_rows_all)

    pair_path = os.path.join(args.output_dir, 'context_summary.csv')
    bin_path = os.path.join(args.output_dir, 'per_bin_detail.csv')
    pair_df.to_csv(pair_path, index=False)
    bin_df.to_csv(bin_path, index=False)
    print(f"\n  [saved] {pair_path}")
    print(f"  [saved] {bin_path}")

    # Top contextualizers by lift
    pair_df = pair_df.sort_values('context_lift_corr', ascending=False)
    print(f"\n  Top 25 (modifier x target x TF) by context_lift_corr "
          f"(span of target's corr_fwd across modifier bins):")
    for _, r in pair_df.head(25).iterrows():
        print(f"    {r['tf']:>3}  {r['modifier']:>20}  modifies  {r['target']:>22}  "
              f"corr_fwd in [{r['min_target_corr_fwd']:+.3f}, "
              f"{r['max_target_corr_fwd']:+.3f}]  lift={r['context_lift_corr']:+.3f}  "
              f"amp={r['context_amplification']:+.3f}")

    print(f"\n  Top 25 by amplification (modifier strengthens or kills target signal):")
    by_amp = pair_df.sort_values('context_amplification', ascending=False).head(25)
    for _, r in by_amp.iterrows():
        print(f"    {r['tf']:>3}  {r['modifier']:>20}  modifies  {r['target']:>22}  "
              f"corr_fwd in [{r['min_target_corr_fwd']:+.3f}, "
              f"{r['max_target_corr_fwd']:+.3f}]  amp={r['context_amplification']:+.3f}")

    # Drilldown: top 5 (modifier, target) — show per-bin detail
    print(f"\n--- Top 5 contextualizer drilldowns ---")
    for _, r in pair_df.head(5).iterrows():
        print(f"\n  {r['tf']} | {r['modifier']} --> {r['target']}  "
              f"(lift={r['context_lift_corr']:+.3f})")
        sub = bin_df[(bin_df['tf'] == r['tf']) &
                       (bin_df['modifier'] == r['modifier']) &
                       (bin_df['target'] == r['target'])].sort_values('modifier_q')
        for _, b in sub.iterrows():
            print(f"      Q{int(b['modifier_q'])}  n={int(b['n']):>5}  "
                  f"target_mean={b['target_mean']:+.3f}  "
                  f"target_corr_fwd={b['target_corr_fwd']:+.4f}")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features contextualization EDA — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"For each (modifier, target, TF) pair: bin modifier into "
                f"{args.quantiles} quantiles; within each bin, compute the "
                f"target's correlation with forward return. The PAIR's score "
                f"= range of target_corr_fwd values across modifier bins.\n\n")
        f.write(f"**TFs:** {args.tfs}  **Quantiles:** {args.quantiles}  "
                f"**Forward N:** {args.forward_n}  **Split:** {args.split}\n\n")
        f.write("## Top 30 (modifier x target x TF) by context_lift_corr\n\n")
        f.write(pair_df.head(30).to_string(index=False))
        f.write("\n\n## Top 30 by amplification\n\n")
        f.write(by_amp.head(30).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
