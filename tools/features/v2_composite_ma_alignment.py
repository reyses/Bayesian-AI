"""
v2 Composite Directional — MA Alignment Signal
================================================
Per the user's counter-proposal: don't strip the autoregressive features
(vwap_w, price_mean_w) — use them as a SMOOTHING signal for direction.

The idea: at each 5m bar, ask "is current price above or below the
smoothed price (vwap or rolling-mean) at each of the 8 TFs?". Each TF
votes +1 (price above), 0 (within tolerance), or -1 (price below). The
sum across TFs is an "alignment score". If all TFs say price is above,
that's strong long alignment; mixed signals = neutral.

This is the technical-analysis-style multi-TF MA alignment, but
computed directly from the precomputed v2 features (no fitting,
purely deterministic). Compare against actual signed-MFE direction to
score it.

Two smoothing signals tested per TF:
  - L2_<TF>_vwap_w     (volume-weighted price avg over window)
  - L2_<TF>_price_mean_w (simple rolling mean over window)

Decision rules tested:
  - all_above: alignment_score == +N_TFS (every TF bullish) → LONG
  - majority: alignment_score > 0 (more bullish than bearish) → LONG
  - threshold N: alignment_score >= N
  - vwap_only / mean_only: split signal source

Usage:
    python tools/v2_composite_ma_alignment.py
    python tools/v2_composite_ma_alignment.py --tolerance 0.5  # ticks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.imr import compute_price_imr, detect_regimes, compute_regime_oracle
from tools.research.features_v2 import (
    TF_HIERARCHY_V2, load_v2_features, align_v2_to_base_tf,
)
from config.oracle_config import ORACLE_LOOKAHEAD_BARS

# Default common cadence
DEFAULT_COMMON_TF = '5m'
TICK_SIZE = 0.25


def compute_target(common_tf: str, atlas_root: str,
                   context_days: int, analysis_days: int,
                   verbose: bool = True) -> dict:
    """Run common-cadence regime detection + oracle MFE/MAE → signed_mfe target."""
    base_df = load_atlas_tf(atlas_root, common_tf)
    if base_df.empty:
        raise FileNotFoundError(f"no OHLC for {common_tf}")
    if verbose:
        print(f"  Common TF {common_tf}: {len(base_df):,} bars")

    price_imr = compute_price_imr(base_df, context_days, analysis_days)
    regime_ids, regime_meta = detect_regimes(price_imr)
    lookahead = ORACLE_LOOKAHEAD_BARS.get(common_tf, 16)
    bar_indices, mfes, maes, directions = compute_regime_oracle(
        base_df, regime_ids, regime_meta, lookahead=lookahead)

    sign_per = np.array([+1.0 if d == 'LONG' else -1.0 for d in directions])
    signed_mfe = mfes * sign_per

    base_ts = base_df['timestamp'].values.astype(np.int64)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        base_ts = base_ts // 10**9

    return {
        'common_tf': common_tf,
        'base_df': base_df,
        'bar_indices': bar_indices,
        'target_ts': base_ts[bar_indices],
        'target_close': base_df['close'].values[bar_indices].astype(np.float64),
        'signed_mfe': signed_mfe,
        'mfes': mfes,
        'maes': maes,
        'lookahead': lookahead,
    }


def compute_alignment_matrix(target: dict, v2_dir: str, atlas_root: str,
                              tolerance_ticks: float = 0.0,
                              verbose: bool = True) -> dict:
    """For each (target_bar, TF), compute price - smoothed_price votes."""
    target_ts = target['target_ts']
    close = target['target_close']

    ts_min = int(target_ts.min())
    ts_max = int(target_ts.max())
    features_5s = load_v2_features(
        v2_dir=v2_dir, atlas_root=atlas_root, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    if verbose:
        print(f"  v2 features: {len(features_5s):,} 5s rows")

    aligned = align_v2_to_base_tf(features_5s, target_ts)

    n = len(target_ts)
    n_tfs = len(TF_HIERARCHY_V2)

    vwap_diffs = np.zeros((n, n_tfs), dtype=np.float64)
    mean_diffs = np.zeros((n, n_tfs), dtype=np.float64)
    available_vwap = np.zeros(n_tfs, dtype=bool)
    available_mean = np.zeros(n_tfs, dtype=bool)

    for j, tf in enumerate(TF_HIERARCHY_V2):
        vwap_col = f'L2_{tf}_vwap_w'
        mean_col = f'L2_{tf}_price_mean_w'
        if vwap_col in aligned.columns:
            v = aligned[vwap_col].values
            v = np.nan_to_num(v, nan=close.mean())
            vwap_diffs[:, j] = close - v
            available_vwap[j] = True
        if mean_col in aligned.columns:
            m = aligned[mean_col].values
            m = np.nan_to_num(m, nan=close.mean())
            mean_diffs[:, j] = close - m
            available_mean[j] = True

    tol = tolerance_ticks * TICK_SIZE
    vwap_votes = np.where(vwap_diffs > tol, 1, np.where(vwap_diffs < -tol, -1, 0)).astype(np.int8)
    mean_votes = np.where(mean_diffs > tol, 1, np.where(mean_diffs < -tol, -1, 0)).astype(np.int8)

    return {
        'vwap_diffs': vwap_diffs,
        'mean_diffs': mean_diffs,
        'vwap_votes': vwap_votes,
        'mean_votes': mean_votes,
        'available_vwap': available_vwap,
        'available_mean': available_mean,
        'tf_order': list(TF_HIERARCHY_V2),
    }


def evaluate_signal(consensus: np.ndarray, actual_dir: np.ndarray,
                    test_mask: np.ndarray, baseline_acc: float,
                    label: str) -> dict:
    """Score a per-bar consensus vector on the test region."""
    m = (consensus != 0) & (actual_dir != 0) & test_mask
    if m.sum() == 0:
        return {'label': label, 'n': 0, 'accuracy': float('nan'),
                'lift': float('nan'), 'pct_data': 0.0}
    sub_acc = float((consensus[m] == actual_dir[m]).mean())
    test_n_total = (test_mask & (actual_dir != 0)).sum()
    return {
        'label': label,
        'n': int(m.sum()),
        'accuracy': sub_acc,
        'lift': sub_acc - baseline_acc,
        'pct_data': float(m.sum() / test_n_total * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--common-tf', default=DEFAULT_COMMON_TF)
    parser.add_argument('--context-days', type=int, default=21)
    parser.add_argument('--analysis-days', type=int, default=0)
    parser.add_argument('--tolerance-ticks', type=float, default=0.0,
                        help='Distance from smoothed price within which vote is 0 (ticks)')
    parser.add_argument('--test-frac', type=float, default=0.2,
                        help='Last fraction of timeline to use as test set')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_composite_ma_alignment')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Composite — MA Alignment")
    print(f"  Common cadence: {args.common_tf}")
    print(f"  Tolerance: {args.tolerance_ticks} ticks")
    print(f"  Test frac (last): {args.test_frac}")
    print(f"{'='*70}")

    # Build target
    print(f"\n--- Step 1: build target at {args.common_tf} ---")
    target = compute_target(args.common_tf, args.data,
                             args.context_days, args.analysis_days,
                             verbose=True)
    actual_dir = np.sign(target['signed_mfe'])
    n = len(actual_dir)
    test_start = int(n * (1 - args.test_frac))
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_start:] = True
    valid_test = test_mask & (actual_dir != 0)
    baseline_acc = float(max((actual_dir[valid_test] == 1).mean(),
                              (actual_dir[valid_test] == -1).mean())) \
        if valid_test.sum() > 0 else 0.5
    print(f"  Test region: bars {test_start}-{n} ({valid_test.sum()} non-flat bars)")
    print(f"  Baseline (majority class): {baseline_acc:.1%}")

    # Build MA alignment matrices
    print(f"\n--- Step 2: alignment matrices ---")
    alignment = compute_alignment_matrix(target, args.cache, args.data,
                                           tolerance_ticks=args.tolerance_ticks,
                                           verbose=True)
    vwap_votes = alignment['vwap_votes']     # (N, 8) ±1/0
    mean_votes = alignment['mean_votes']
    n_tfs = vwap_votes.shape[1]

    # Per-TF signal evaluation (each TF alone)
    print(f"\n--- Step 3: per-TF signals ---")
    per_tf_rows = []
    for j, tf in enumerate(alignment['tf_order']):
        row = {'tf': tf}
        for src, votes in (('vwap', vwap_votes), ('mean', mean_votes)):
            r = evaluate_signal(votes[:, j], actual_dir, test_mask,
                                  baseline_acc, f'{tf}_{src}')
            row[f'{src}_n'] = r['n']
            row[f'{src}_pct'] = r['pct_data']
            row[f'{src}_acc'] = r['accuracy']
            row[f'{src}_lift'] = r['lift']
        per_tf_rows.append(row)
        print(f"  {tf:>3}: vwap n={row['vwap_n']:>5} acc={row['vwap_acc']:.1%} "
              f"lift={row['vwap_lift']:+.1%} | "
              f"mean n={row['mean_n']:>5} acc={row['mean_acc']:.1%} "
              f"lift={row['mean_lift']:+.1%}")
    pd.DataFrame(per_tf_rows).to_csv(
        os.path.join(args.output_dir, 'per_tf_signals.csv'), index=False)

    # Aggregate scores
    vwap_score = vwap_votes.sum(axis=1)
    mean_score = mean_votes.sum(axis=1)
    combined_score = vwap_score + mean_score   # (range -2*n_tfs to +2*n_tfs)

    print(f"\n--- Step 4: aggregate alignment thresholds ---")

    def stratify(score: np.ndarray, label: str) -> list:
        rows = []
        # Use absolute thresholds 1..max
        max_t = int(np.abs(score).max()) if score.size else 0
        for thr in range(1, max_t + 1):
            consensus = np.zeros(len(score), dtype=np.int8)
            consensus[score >= thr] = 1
            consensus[score <= -thr] = -1
            r = evaluate_signal(consensus, actual_dir, test_mask,
                                  baseline_acc, f'{label}_thr_{thr}')
            if r['n'] >= 50:
                rows.append({'threshold': thr, **{k: v for k, v in r.items()
                                                    if k != 'label'}})
        return rows

    vwap_rows = stratify(vwap_score, 'vwap')
    print(f"  VWAP-alignment (range -{n_tfs} to +{n_tfs}):")
    for r in vwap_rows:
        print(f"    |score|>={r['threshold']:>2}: n={r['n']:>5}, "
              f"acc={r['accuracy']:.1%}, lift={r['lift']:+.1%}, "
              f"%data={r['pct_data']:.1f}%")
    pd.DataFrame(vwap_rows).to_csv(
        os.path.join(args.output_dir, 'vwap_alignment.csv'), index=False)

    mean_rows = stratify(mean_score, 'mean')
    print(f"\n  PriceMean-alignment (range -{n_tfs} to +{n_tfs}):")
    for r in mean_rows:
        print(f"    |score|>={r['threshold']:>2}: n={r['n']:>5}, "
              f"acc={r['accuracy']:.1%}, lift={r['lift']:+.1%}, "
              f"%data={r['pct_data']:.1f}%")
    pd.DataFrame(mean_rows).to_csv(
        os.path.join(args.output_dir, 'mean_alignment.csv'), index=False)

    combined_rows = stratify(combined_score, 'combined')
    print(f"\n  Combined VWAP+PriceMean (range -{2*n_tfs} to +{2*n_tfs}):")
    for r in combined_rows:
        print(f"    |score|>={r['threshold']:>2}: n={r['n']:>5}, "
              f"acc={r['accuracy']:.1%}, lift={r['lift']:+.1%}, "
              f"%data={r['pct_data']:.1f}%")
    pd.DataFrame(combined_rows).to_csv(
        os.path.join(args.output_dir, 'combined_alignment.csv'), index=False)

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 MA Alignment Composite — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Common cadence:** `{args.common_tf}`\n")
        f.write(f"**Tolerance:** {args.tolerance_ticks} ticks\n")
        f.write(f"**Baseline (majority class):** {baseline_acc:.1%}\n\n")

        f.write("## Per-TF (own signal alone)\n\n")
        f.write(pd.DataFrame(per_tf_rows).to_string(index=False))
        f.write("\n\n")

        f.write(f"## VWAP alignment (sum of price>vwap_w votes across "
                f"{n_tfs} TFs)\n\n")
        f.write(pd.DataFrame(vwap_rows).to_string(index=False))
        f.write("\n\n")

        f.write(f"## Price-mean alignment (sum of price>price_mean_w votes)\n\n")
        f.write(pd.DataFrame(mean_rows).to_string(index=False))
        f.write("\n\n")

        f.write(f"## Combined VWAP+PriceMean alignment\n\n")
        f.write(pd.DataFrame(combined_rows).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
