#!/usr/bin/env python
"""
Hurst Exponent Validation — Is Hurst reliable at 15s with 100-bar window?

Computes Hurst exponent at multiple window sizes (50/100/200/400) on 15s ATLAS
data and compares against ADX-confirmed trending/ranging ground truth.

Ground truth (ADX-based):
  - ADX > 25 AND |DMI_diff| > 10 → TRENDING
  - ADX < 20 → RANGING
  - Otherwise → AMBIGUOUS (excluded from confusion matrix)

Hurst classification:
  - H > 0.5 → TRENDING (persistent)
  - H < 0.5 → RANGING (anti-persistent / mean-reverting)

Output: Confusion matrix, accuracy, precision/recall for each window size,
plus recommendation for gate threshold.

Usage:
    python scripts/hurst_validation.py                     # default: ATLAS_1WEEK, 15s
    python scripts/hurst_validation.py --data DATA/ATLAS   # full dataset
    python scripts/hurst_validation.py --tf 1m             # different TF
    python scripts/hurst_validation.py --months 2025_01    # specific month
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.physics_utils import compute_adx_dmi_cpu, ADX_PERIOD, HURST_WINDOW
from tools.research.data import load_atlas_tf


# ── Hurst R/S computation (mirrors statistical_field_engine) ─────────────────

def compute_hurst_rs(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute Hurst exponent via R/S method at given window size.
    Returns array of length len(prices), NaN where insufficient data."""
    from numba import njit

    @njit
    def _rs_single(returns, w):
        n = len(returns)
        if w < 4 or n < w:
            return 0.0
        cumdev = np.empty(w)
        mean_r = 0.0
        for j in range(w):
            mean_r += returns[n - w + j]
        mean_r /= w
        running = 0.0
        mn, mx = 1e30, -1e30
        for j in range(w):
            running += returns[n - w + j] - mean_r
            cumdev[j] = running
            if running < mn:
                mn = running
            if running > mx:
                mx = running
        R = mx - mn
        S = 0.0
        for j in range(w):
            S += (returns[n - w + j] - mean_r) ** 2
        S = (S / w) ** 0.5
        if S < 1e-12:
            return 0.0
        return R / S

    n = len(prices)
    hurst = np.full(n, np.nan)
    if n < window:
        return hurst

    returns = np.diff(prices)
    sizes = sorted(set([max(sz, 4) for sz in [window // 8, window // 4, window // 2, window]]))
    if len(sizes) < 2:
        return hurst

    log_ns = np.log(np.array(sizes, dtype=np.float64))
    A = np.vstack([log_ns, np.ones(len(log_ns))]).T
    pinv = np.linalg.pinv(A)
    pinv_slope = pinv[0, :]

    # Precompute R/S for each size at each position
    for i in tqdm(range(window, n), desc=f'  Hurst(w={window})', leave=False):
        log_rs = np.zeros(len(sizes))
        valid = True
        for si, sz in enumerate(sizes):
            w_ret = sz - 1
            if i < w_ret:
                valid = False
                break
            sub = returns[i - w_ret:i]
            rs = _rs_single(sub, len(sub))
            if rs <= 0:
                valid = False
                break
            log_rs[si] = np.log(rs)
        if valid:
            hurst[i] = np.dot(pinv_slope, log_rs)

    return hurst


# ── ADX ground truth ─────────────────────────────────────────────────────────

def compute_adx_ground_truth(highs, lows, closes):
    """Compute ADX + DMI and classify each bar as TRENDING/RANGING/AMBIGUOUS."""
    n = len(closes)
    tr_raw = np.zeros(n)
    plus_dm_raw = np.zeros(n)
    minus_dm_raw = np.zeros(n)

    for i in range(1, n):
        tr_raw[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm_raw[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm_raw[i] = down_move

    adx, dmi_plus, dmi_minus = compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, ADX_PERIOD)
    dmi_diff = dmi_plus - dmi_minus

    # Classification
    labels = np.full(n, -1, dtype=np.int8)  # -1 = ambiguous
    trending = (adx > 25) & (np.abs(dmi_diff) > 10)
    ranging = adx < 20
    labels[trending] = 1  # TRENDING
    labels[ranging] = 0   # RANGING

    return adx, dmi_diff, labels


# ── Confusion matrix ─────────────────────────────────────────────────────────

def confusion_matrix_report(hurst_arr, truth_labels, threshold=0.5, label=''):
    """Print confusion matrix for Hurst vs ADX ground truth."""
    valid = (~np.isnan(hurst_arr)) & (truth_labels >= 0)
    h = hurst_arr[valid]
    t = truth_labels[valid]

    h_pred = (h > threshold).astype(int)  # 1 = trending, 0 = ranging

    tp = np.sum((h_pred == 1) & (t == 1))
    tn = np.sum((h_pred == 0) & (t == 0))
    fp = np.sum((h_pred == 1) & (t == 0))
    fn = np.sum((h_pred == 0) & (t == 1))
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'─' * 50}")
    print(f"  {label}  (threshold = {threshold:.2f})")
    print(f"{'─' * 50}")
    print(f"  Total valid bars: {total:,} (excluded {np.sum(~valid):,} ambiguous/NaN)")
    print(f"  Ground truth: {np.sum(t == 1):,} trending, {np.sum(t == 0):,} ranging")
    print(f"  Hurst predicts: {np.sum(h_pred == 1):,} trending, {np.sum(h_pred == 0):,} ranging")
    print()
    print(f"                     Predicted")
    print(f"                  TREND    RANGE")
    print(f"  Actual TREND  {tp:>7,}  {fn:>7,}")
    print(f"  Actual RANGE  {fp:>7,}  {tn:>7,}")
    print()
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}  (trending class)")
    print(f"  Recall:    {recall:.1%}  (trending class)")
    print(f"  F1:        {f1:.1%}")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'total': total}


def sweep_thresholds(hurst_arr, truth_labels, label=''):
    """Sweep Hurst thresholds from 0.35 to 0.65 and find optimal."""
    valid = (~np.isnan(hurst_arr)) & (truth_labels >= 0)
    h = hurst_arr[valid]
    t = truth_labels[valid]
    total = len(h)
    if total == 0:
        print(f"  {label}: no valid data")
        return 0.5

    thresholds = np.arange(0.35, 0.66, 0.01)
    best_f1, best_t = 0, 0.5
    print(f"\n  Threshold sweep for {label}:")
    print(f"  {'Thresh':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'FP%':>6}  {'FN%':>6}")

    for thr in thresholds:
        pred = (h > thr).astype(int)
        tp = np.sum((pred == 1) & (t == 1))
        tn = np.sum((pred == 0) & (t == 0))
        fp = np.sum((pred == 1) & (t == 0))
        fn = np.sum((pred == 0) & (t == 1))
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fp_pct = fp / total
        fn_pct = fn / total
        marker = ' ◄' if f1 > best_f1 else ''
        if thr in [0.40, 0.45, 0.50, 0.55, 0.60] or f1 > best_f1:
            print(f"  {thr:>6.2f}  {acc:>5.1%}  {prec:>5.1%}  {rec:>5.1%}  {f1:>5.1%}  {fp_pct:>5.1%}  {fn_pct:>5.1%}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_t = thr

    print(f"\n  → Best threshold: {best_t:.2f} (F1 = {best_f1:.1%})")
    return best_t


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Hurst exponent validation vs ADX ground truth')
    parser.add_argument('--data', default='DATA/ATLAS_1WEEK', help='ATLAS data directory')
    parser.add_argument('--tf', default='15s', help='Timeframe to analyze (default: 15s)')
    parser.add_argument('--months', nargs='+', default=None, help='Specific months (e.g., 2025_01)')
    parser.add_argument('--windows', nargs='+', type=int, default=[50, 100, 200, 400],
                        help='Hurst window sizes to test')
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"HURST EXPONENT VALIDATION")
    print(f"=" * 60)
    print(f"  Data: {args.data}")
    print(f"  TF:   {args.tf}")
    print(f"  Windows: {args.windows}")

    # Load data
    df = load_atlas_tf(args.data, args.tf, months=args.months)
    if df.empty:
        print(f"\nERROR: No data found at {args.data}/{args.tf}")
        sys.exit(1)

    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    print(f"  Loaded {len(df):,} bars")

    # Compute ADX ground truth
    print("\nComputing ADX ground truth...")
    adx, dmi_diff, truth = compute_adx_ground_truth(highs, lows, closes)
    n_trend = np.sum(truth == 1)
    n_range = np.sum(truth == 0)
    n_ambig = np.sum(truth == -1)
    print(f"  Trending: {n_trend:,} bars ({n_trend / len(truth):.1%})")
    print(f"  Ranging:  {n_range:,} bars ({n_range / len(truth):.1%})")
    print(f"  Ambiguous: {n_ambig:,} bars ({n_ambig / len(truth):.1%})")

    # Compute Hurst at each window size
    results = {}
    for w in args.windows:
        print(f"\nComputing Hurst (window={w})...")
        h = compute_hurst_rs(prices, w)
        valid_h = h[~np.isnan(h)]
        print(f"  Valid: {len(valid_h):,} bars, "
              f"mean={np.nanmean(h):.3f}, std={np.nanstd(h):.3f}, "
              f"median={np.nanmedian(valid_h):.3f}")

        # Confusion matrix at default 0.5
        stats = confusion_matrix_report(h, truth, threshold=0.5, label=f'Window={w}')
        results[w] = stats

        # Threshold sweep
        best_t = sweep_thresholds(h, truth, label=f'Window={w}')
        if abs(best_t - 0.5) > 0.02:
            print(f"\n  Re-evaluating at optimal threshold {best_t:.2f}:")
            confusion_matrix_report(h, truth, threshold=best_t, label=f'Window={w} @{best_t:.2f}')

    # Summary comparison
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Window':>6}  {'Accuracy':>8}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}  {'Total':>8}")
    best_w, best_f1 = 0, 0
    for w, s in results.items():
        marker = ''
        if s['f1'] > best_f1:
            best_f1 = s['f1']
            best_w = w
            marker = ' ◄'
        print(f"  {w:>6}  {s['accuracy']:>7.1%}  {s['f1']:>5.1%}  {s['precision']:>5.1%}  "
              f"{s['recall']:>5.1%}  {s['total']:>8,}{marker}")

    print(f"\n  Best window: {best_w} bars (F1 = {best_f1:.1%})")

    # Recommendations
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATIONS")
    print(f"{'=' * 60}")
    best = results[best_w]
    if best['f1'] < 0.55:
        print(f"  A (UNRELIABLE): Hurst at {args.tf} is near random (F1={best['f1']:.1%}).")
        print(f"     → Convert hurst gate from hard block to score penalty")
        print(f"     → In execution_engine.py: don't skip on hurst<0.5, instead")
        print(f"       reduce conviction by (0.5 - hurst) * penalty_weight")
    elif best['f1'] < 0.70:
        print(f"  B (MARGINAL): Hurst is somewhat informative (F1={best['f1']:.1%}).")
        print(f"     → Consider adjusting hurst_min threshold")
        print(f"     → May benefit from larger window (current HURST_WINDOW={HURST_WINDOW})")
    else:
        print(f"  C (RELIABLE): Hurst works well at this scale (F1={best['f1']:.1%}).")
        print(f"     → Current gate is appropriate, consider fine-tuning threshold")


if __name__ == '__main__':
    main()
