"""
v2_features_lookback_pair_eda.py — Layer B2 of the regime EDA stack.

Pairwise lookback patterns: "when feature X does pattern A AND
feature Y does pattern B over N bars, how does price react?".

For each (X, Y) pair from top-K shortlist, for each window N, detect
joint patterns at the same time t:

  BOTH_RISING        - X rising mono AND Y rising mono
  BOTH_FALLING       - both falling mono
  DIVERGE_X_UP       - X rising, Y falling (X leading?)
  DIVERGE_X_DOWN     - X falling, Y rising
  SPIKE_BOTH_UP      - both spike up at same bar
  SPIKE_BOTH_DOWN    - both spike down
  REVERSAL_AGREE_UP  - both flipping up after sustained down
  REVERSAL_AGREE_DOWN - both flipping down

For each combo detection: forward return at t+forward_n, WR, MFE/MAE.

Top results identify joint-pattern signals where the pair behavior
predicts price reaction beyond what either feature alone gives.

Outputs:
  reports/findings/v2_features_lookback_pair_eda/
    combo_summary.csv  — per (x, y, window, pattern) row
    top_combos.md      — top 30 by WR_lift, by |mean_fwd|, by interaction lift

Usage:
  python tools/v2_features_lookback_pair_eda.py
  python tools/v2_features_lookback_pair_eda.py --top-k 10 --windows 6 12
"""

from __future__ import annotations
import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels
# Reuse pattern detectors from B1
from tools.v2_features_lookback_eda import (
    detect_rising_mono, detect_falling_mono,
    detect_spike, detect_reversal,
    compute_forward_metrics, load_shortlist,
)


DEFAULT_BASE_TF = '5m'
DEFAULT_FORWARD_N = 12
DEFAULT_WINDOWS = (6, 12)        # tighter than B1 to keep pair runtime manageable
DEFAULT_TOP_K = 12                # 12 choose 2 = 66 pairs


def detect_combo_patterns(x: np.ndarray, y: np.ndarray, n: int) -> dict:
    """For each combo, return boolean detection array."""
    rx = detect_rising_mono(x, n)
    ry = detect_rising_mono(y, n)
    fx = detect_falling_mono(x, n)
    fy = detect_falling_mono(y, n)
    sx_up = detect_spike(x, n, positive=True)
    sx_dn = detect_spike(x, n, positive=False)
    sy_up = detect_spike(y, n, positive=True)
    sy_dn = detect_spike(y, n, positive=False)
    revx_up = detect_reversal(x, n, positive=True)
    revx_dn = detect_reversal(x, n, positive=False)
    revy_up = detect_reversal(y, n, positive=True)
    revy_dn = detect_reversal(y, n, positive=False)
    return {
        'BOTH_RISING':         rx & ry,
        'BOTH_FALLING':        fx & fy,
        'DIVERGE_X_UP':        rx & fy,
        'DIVERGE_X_DOWN':      fx & ry,
        'SPIKE_BOTH_UP':       sx_up & sy_up,
        'SPIKE_BOTH_DOWN':     sx_dn & sy_dn,
        'REVERSAL_AGREE_UP':   revx_up & revy_up,
        'REVERSAL_AGREE_DOWN': revx_dn & revy_dn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--layer1-dir',
                        default='reports/findings/v2_features_regime_eda')
    parser.add_argument('--rank-by', default='lookback_corr',
                        choices=['cohen_d', 'lookback_corr', 'forward_corr'])
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='Top-K features -> C(K,2) pairs to analyze')
    parser.add_argument('--windows', nargs='+', type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-detections', type=int, default=10,
                        help='Minimum detections to include a combo in the report')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_lookback_pair_eda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features x Price — Layer B2 (pairwise lookback)")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  Top-K: {args.top_k} -> {args.top_k * (args.top_k - 1) // 2} pairs")
    print(f"  Windows: {args.windows}  Forward N: {args.forward_n}")
    print(f"  Min detections: {args.min_detections}")
    print(f"{'='*70}")

    # Shortlist
    shortlist = load_shortlist(args.layer1_dir, args.top_k, args.rank_by)
    print(f"\n  Shortlist:")
    for f in shortlist:
        print(f"    {f}")

    # Load + merge
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
    print(f"  v2 features: {len(features_5s):,} 5s rows")
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)
    shortlist = [f for f in shortlist if f in full.columns]

    close = full['close'].values.astype(np.float64)
    high = full['high'].values.astype(np.float64)
    low = full['low'].values.astype(np.float64)

    # Baseline
    n_total = len(close)
    fwd = np.full(n_total, np.nan)
    if n_total > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    valid = ~np.isnan(fwd)
    baseline_wr = float((fwd[valid] > 0).mean()) if valid.sum() else 0.5
    print(f"\n  Baseline WR: {baseline_wr:.1%}")

    # Loop pairs x windows x combos
    pairs = list(combinations(shortlist, 2))
    print(f"\n--- Analyzing {len(pairs)} pairs x {len(args.windows)} windows x 8 patterns ---")
    rows = []
    for (x_name, y_name) in pairs:
        x = full[x_name].values.astype(np.float64)
        y = full[y_name].values.astype(np.float64)
        for n in args.windows:
            combos = detect_combo_patterns(x, y, n)
            for pat_name, det in combos.items():
                if det.sum() < args.min_detections:
                    continue
                metrics = compute_forward_metrics(close, high, low, det,
                                                    args.forward_n)
                if metrics['n'] < args.min_detections:
                    continue
                rows.append({
                    'x_feature': x_name,
                    'y_feature': y_name,
                    'window': n,
                    'pattern': pat_name,
                    'pair': f'{x_name}__x__{y_name}',
                    'n': metrics['n'],
                    'freq_pct': float(det.sum() / len(x) * 100),
                    'win_rate': metrics['win_rate'],
                    'wr_lift': metrics['win_rate'] - baseline_wr,
                    'mean_fwd': metrics['mean_fwd'],
                    'std_fwd': metrics['std_fwd'],
                    'mean_mfe': metrics['mean_mfe'],
                    'mean_mae': metrics['mean_mae'],
                })

    df = pd.DataFrame(rows).sort_values('wr_lift', ascending=False, na_position='last')
    out_path = os.path.join(args.output_dir, 'combo_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path} ({len(df)} combo rows)")

    print(f"\n  Top 25 combos by WR_lift:")
    print(f"    {'pair':>62}  {'win':>4} {'pattern':>20}  "
          f"{'n':>5} {'WR':>6} {'lift':>6} {'fwd':>7}")
    for _, r in df.head(25).iterrows():
        pair_short = f"{r['x_feature'][:28]} x {r['y_feature'][:28]}"
        print(f"    {pair_short[:62]:>62}  {r['window']:>4} {r['pattern']:>20}  "
              f"{r['n']:>5} {r['win_rate']:>6.1%} {r['wr_lift']:>+6.1%} "
              f"{r['mean_fwd']:>+7.2f}")

    print(f"\n  Top 25 combos by |mean_fwd|:")
    by_fwd = df.assign(abs_fwd=df['mean_fwd'].abs()) \
        .sort_values('abs_fwd', ascending=False).head(25)
    for _, r in by_fwd.iterrows():
        pair_short = f"{r['x_feature'][:28]} x {r['y_feature'][:28]}"
        print(f"    {pair_short[:62]:>62}  {r['window']:>4} {r['pattern']:>20}  "
              f"n={r['n']:>5}  WR={r['win_rate']:.1%}  "
              f"fwd={r['mean_fwd']:>+7.2f}")

    md_path = os.path.join(args.output_dir, 'top_combos.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Pairwise lookback combos — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`  **Split:** `{args.split}`  "
                f"**Forward N:** {args.forward_n}\n")
        f.write(f"**Baseline WR:** {baseline_wr:.1%}\n\n")
        f.write("## Top combos by WR lift\n\n")
        f.write(df.head(30).to_string(index=False))
        f.write("\n\n")
        f.write("## Top combos by |mean_fwd|\n\n")
        f.write(by_fwd.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
