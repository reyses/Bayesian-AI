"""
v2_features_lookback_eda.py — Layer B1 of the regime EDA stack.

Single-feature lookback patterns: "when feature X does PATTERN over N
bars, how does price react?".

For each feature (top-K from Layer 1 by user-chosen ranking) and each
lookback window N ∈ {3, 6, 12, 24} bars, detect six patterns:

  RISING_MONO    - X strictly monotonically increasing over [t-N+1, t]
  FALLING_MONO   - X strictly monotonically decreasing over [t-N+1, t]
  SPIKE_UP       - X[t] > μ_local + 2σ_local (local stats over window)
  SPIKE_DOWN     - X[t] < μ_local - 2σ_local
  REVERSAL_UP    - last bar UP after >=N-2 prior bars going DOWN
  REVERSAL_DOWN  - last bar DOWN after >=N-2 prior bars going UP

For each (feature, window, pattern) detection, measure forward price
reaction:
  - n detections
  - mean fwd return at t+forward_n
  - std fwd return
  - win rate (sign(fwd) > 0)
  - mean MFE (max excursion) over forward_n bars
  - mean MAE (drawdown)

Stratify by regime_2d when --by-regime is passed.

Outputs:
  reports/findings/v2_features_lookback_eda/
    pattern_summary.csv     — per (feature, window, pattern) row
    top_patterns.md         — top 30 patterns by |mean_fwd|, by WR, by lift over baseline
    by_regime.csv           — stratified by regime_2d (optional)

Usage:
  python tools/v2_features_lookback_eda.py
  python tools/v2_features_lookback_eda.py --top-k 15 --windows 6 12 24
  python tools/v2_features_lookback_eda.py --by-regime
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
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels


DEFAULT_BASE_TF = '5m'
DEFAULT_FORWARD_N = 12
DEFAULT_WINDOWS = (3, 6, 12, 24)
DEFAULT_TOP_K = 20
SPIKE_SIGMA = 2.0


# ── Pattern detectors (each returns boolean array same length as input) ──

def detect_rising_mono(x: np.ndarray, n: int) -> np.ndarray:
    """Strictly monotonically increasing over [t-n+1, t]."""
    out = np.zeros(len(x), dtype=bool)
    if len(x) < n:
        return out
    diffs = np.diff(x)
    # Need n-1 consecutive positive diffs ending at index i
    for i in range(n - 1, len(x)):
        if np.all(diffs[i - (n - 1): i] > 0):
            out[i] = True
    return out


def detect_falling_mono(x: np.ndarray, n: int) -> np.ndarray:
    """Strictly monotonically decreasing over [t-n+1, t]."""
    out = np.zeros(len(x), dtype=bool)
    if len(x) < n:
        return out
    diffs = np.diff(x)
    for i in range(n - 1, len(x)):
        if np.all(diffs[i - (n - 1): i] < 0):
            out[i] = True
    return out


def detect_spike(x: np.ndarray, n: int, sigma: float = SPIKE_SIGMA,
                  positive: bool = True) -> np.ndarray:
    """X[t] is more than `sigma` × σ_local from μ_local over the window."""
    out = np.zeros(len(x), dtype=bool)
    if len(x) < n + 1:
        return out
    for i in range(n, len(x)):
        window = x[i - n: i]
        valid = window[~np.isnan(window)]
        if len(valid) < n // 2:
            continue
        mu = valid.mean()
        sd = valid.std(ddof=0)
        if sd < 1e-12 or np.isnan(x[i]):
            continue
        z = (x[i] - mu) / sd
        if positive and z > sigma:
            out[i] = True
        elif (not positive) and z < -sigma:
            out[i] = True
    return out


def detect_reversal(x: np.ndarray, n: int, positive: bool = True) -> np.ndarray:
    """Last bar opposite sign after >=n-2 prior bars all same sign.

    `positive=True` → REVERSAL_UP: many DOWN bars, then UP bar.
    """
    out = np.zeros(len(x), dtype=bool)
    if len(x) < n:
        return out
    diffs = np.diff(x)
    target_min = n - 2
    for i in range(n - 1, len(x)):
        last_d = diffs[i - 1]
        prior = diffs[i - (n - 1): i - 1]
        if positive:
            if last_d > 0 and (prior < 0).sum() >= target_min:
                out[i] = True
        else:
            if last_d < 0 and (prior > 0).sum() >= target_min:
                out[i] = True
    return out


PATTERN_FNS = {
    'RISING_MONO':   lambda x, n: detect_rising_mono(x, n),
    'FALLING_MONO':  lambda x, n: detect_falling_mono(x, n),
    'SPIKE_UP':      lambda x, n: detect_spike(x, n, positive=True),
    'SPIKE_DOWN':    lambda x, n: detect_spike(x, n, positive=False),
    'REVERSAL_UP':   lambda x, n: detect_reversal(x, n, positive=True),
    'REVERSAL_DOWN': lambda x, n: detect_reversal(x, n, positive=False),
}


# ── Forward measurements ─────────────────────────────────────────────────

def compute_forward_metrics(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                              detections: np.ndarray, forward_n: int) -> dict:
    """For all bars where detections=True, compute fwd return + MFE/MAE."""
    idx = np.where(detections)[0]
    valid_idx = [i for i in idx if i + forward_n < len(close)]
    n = len(valid_idx)
    if n == 0:
        return {'n': 0, 'mean_fwd': float('nan'), 'std_fwd': float('nan'),
                'win_rate': float('nan'), 'mean_mfe': float('nan'),
                'mean_mae': float('nan')}
    fwds = np.array([close[i + forward_n] - close[i] for i in valid_idx])
    mfes = np.array([float(high[i + 1: i + 1 + forward_n].max() - close[i])
                       if forward_n > 0 else 0.0 for i in valid_idx])
    maes = np.array([float(close[i] - low[i + 1: i + 1 + forward_n].min())
                       if forward_n > 0 else 0.0 for i in valid_idx])
    win = (fwds > 0).mean()
    return {
        'n': int(n),
        'mean_fwd': float(fwds.mean()),
        'std_fwd': float(fwds.std(ddof=1)) if n > 1 else 0.0,
        'win_rate': float(win),
        'mean_mfe': float(mfes.mean()),
        'mean_mae': float(maes.mean()),
    }


# ── Layer 1 shortlist ────────────────────────────────────────────────────

def load_shortlist(layer1_dir: str, top_k: int, rank_by: str) -> list[str]:
    """Same logic as v2_features_pairwise_eda."""
    sep_path = os.path.join(layer1_dir, 'regime_separation.csv')
    corr_path = os.path.join(layer1_dir, 'price_correlations.csv')
    if rank_by == 'cohen_d':
        sep = pd.read_csv(sep_path)
        ranking = sep.groupby('feature')['abs_d'].max().sort_values(ascending=False)
    elif rank_by == 'lookback_corr':
        corr = pd.read_csv(corr_path)
        corr['abs_lb'] = corr['corr_lookback_return'].abs()
        ranking = corr.set_index('feature')['abs_lb'].sort_values(ascending=False)
    elif rank_by == 'forward_corr':
        corr = pd.read_csv(corr_path)
        corr['abs_fw'] = corr['corr_forward_return'].abs()
        ranking = corr.set_index('feature')['abs_fw'].sort_values(ascending=False)
    else:
        raise ValueError(f"unknown rank_by: {rank_by}")
    return ranking.head(top_k).index.tolist()


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
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--windows', nargs='+', type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--by-regime', action='store_true',
                        help='Also stratify pattern reactions by regime_2d')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_lookback_eda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features × Price — Layer B1 (single-feature lookback)")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  Top-K from Layer 1: {args.top_k} (by {args.rank_by})")
    print(f"  Windows: {args.windows}  Forward N: {args.forward_n}")
    print(f"  Patterns: {list(PATTERN_FNS.keys())}")
    print(f"{'='*70}")

    # Load shortlist
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
    print(f"  After split={args.split}: {len(merged):,} bars "
          f"({merged['date'].nunique()} days)")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    print(f"  v2 features: {len(features_5s):,} 5s rows")
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    close = full['close'].values.astype(np.float64)
    high = full['high'].values.astype(np.float64)
    low = full['low'].values.astype(np.float64)

    # Baseline: forward return on all valid bars
    n_total = len(close)
    fwd_all = np.full(n_total, np.nan)
    if n_total > args.forward_n:
        fwd_all[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    valid_fwd = ~np.isnan(fwd_all)
    baseline_wr = float((fwd_all[valid_fwd] > 0).mean()) if valid_fwd.sum() else 0.5
    baseline_mean_fwd = float(np.nanmean(fwd_all))
    print(f"\n  Baseline: WR={baseline_wr:.1%}, mean_fwd={baseline_mean_fwd:+.3f}")

    # Validate features present
    shortlist = [f for f in shortlist if f in full.columns]

    # Detect + measure
    print(f"\n--- Detecting patterns + measuring forward price reaction ---")
    rows = []
    by_regime_rows = []
    regimes = sorted(full['regime_2d'].unique()) if args.by_regime else []

    for feat in shortlist:
        x = full[feat].values.astype(np.float64)
        for n in args.windows:
            for pat_name, fn in PATTERN_FNS.items():
                detections = fn(x, n)
                if detections.sum() < 5:
                    continue
                metrics = compute_forward_metrics(close, high, low, detections,
                                                    args.forward_n)
                metrics.update({
                    'feature': feat,
                    'window': n,
                    'pattern': pat_name,
                    'pattern_freq_pct': float(detections.sum() / len(x) * 100),
                    'wr_lift': metrics['win_rate'] - baseline_wr,
                    'mean_fwd_lift': metrics['mean_fwd'] - baseline_mean_fwd,
                })
                rows.append(metrics)

                if args.by_regime:
                    for r2d in regimes:
                        regime_mask = (full['regime_2d'].values == r2d)
                        sub_det = detections & regime_mask
                        if sub_det.sum() < 5:
                            continue
                        sub_m = compute_forward_metrics(close, high, low, sub_det,
                                                         args.forward_n)
                        by_regime_rows.append({
                            'feature': feat, 'window': n, 'pattern': pat_name,
                            'regime_2d': r2d, **sub_m,
                        })

    pat_df = pd.DataFrame(rows).sort_values('wr_lift', ascending=False, na_position='last')
    out_path = os.path.join(args.output_dir, 'pattern_summary.csv')
    pat_df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path} ({len(pat_df)} rows)")

    # Top patterns by lift
    print(f"\n  Top 20 patterns by WR_lift:")
    print(f"    {'feature':>32}  {'win':>4} {'pat':>15}  "
          f"{'n':>5} {'freq%':>6} {'WR':>6} {'WR_lift':>7} "
          f"{'mean_fwd':>9} {'mean_mfe':>9} {'mean_mae':>9}")
    for _, r in pat_df.head(20).iterrows():
        print(f"    {r['feature'][:32]:>32}  {r['window']:>4} {r['pattern']:>15}  "
              f"{r['n']:>5} {r['pattern_freq_pct']:>5.1f}% "
              f"{r['win_rate']:>6.1%} {r['wr_lift']:>+6.1%} "
              f"{r['mean_fwd']:>+9.2f} {r['mean_mfe']:>+9.2f} "
              f"{r['mean_mae']:>+9.2f}")

    print(f"\n  Top 20 patterns by |mean_fwd|:")
    by_fwd = pat_df.assign(abs_fwd=pat_df['mean_fwd'].abs()) \
        .sort_values('abs_fwd', ascending=False).head(20)
    for _, r in by_fwd.iterrows():
        print(f"    {r['feature'][:32]:>32}  {r['window']:>4} {r['pattern']:>15}  "
              f"n={r['n']:>5}  WR={r['win_rate']:.1%}  "
              f"mean_fwd={r['mean_fwd']:>+9.2f}  freq={r['pattern_freq_pct']:.1f}%")

    if args.by_regime and by_regime_rows:
        br_df = pd.DataFrame(by_regime_rows)
        br_path = os.path.join(args.output_dir, 'by_regime.csv')
        br_df.to_csv(br_path, index=False)
        print(f"\n  [saved] {br_path}")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'top_patterns.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Single-feature lookback patterns — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`  **Split:** `{args.split}`  "
                f"**Forward N:** {args.forward_n} bars\n")
        f.write(f"**Baseline:** WR={baseline_wr:.1%}, "
                f"mean_fwd={baseline_mean_fwd:+.3f}\n\n")
        f.write(f"## Top patterns by WR lift (vs {baseline_wr:.0%} baseline)\n\n")
        f.write(pat_df.head(30).to_string(index=False))
        f.write("\n\n")
        f.write(f"## Top patterns by |mean forward return|\n\n")
        f.write(by_fwd.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
