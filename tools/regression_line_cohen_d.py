"""
Regression-line Cohen d at zigzag pivots — direction prediction on smoothed signal.

The raw 91D feature set showed |d| < 0.01 for LONG-vs-SHORT at any bar
(noise-dominated). The user's hypothesis: use the REGRESSION LINE (smoothed)
instead of raw features. At each zigzag pivot point (a CLEAN decision moment),
measure regression-line properties and test if they predict next leg direction.

Three improvements over the prior analysis:
  1. Events are zigzag pivots — non-overlapping, meaningful decision points.
  2. Features are regression-line properties (slope, residual) — smoothed.
  3. Multiple window sizes capture multi-TF regression physics.

Features computed at each pivot (from 1m bars):
  - beta_W:    OLS slope over last W 1m bars (for W in {10, 20, 60, 180, 720})
  - res_W:  signed residual = price - fitted_value at W
  - res_W_norm: residual / residual_std in window
  - beta_sign_align: fraction of windows where β has same sign
  - beta_60 - beta_720: short-slope minus long-slope (acceleration proxy)

Target: UP-next-leg vs DOWN-next-leg (next zigzag pivot's direction).

Cohen d between UP-next and DOWN-next cohorts per feature. Walk-forward:
IS vs OOS. Feature survives if sign matches and min|d| >= 0.15.

Usage:
    python tools/regression_line_cohen_d.py                    # R=$15 default
    python tools/regression_line_cohen_d.py --threshold 10     # R=$10
    python tools/regression_line_cohen_d.py --min-leg 20       # drop tiny legs

Output: reports/findings/regression_line_cohen_d.md
"""
import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/regression_line_cohen_d.md'

WINDOWS = [10, 20, 60, 180, 720]  # 1m bars
DOLLAR_PER_POINT = 2.0


def zigzag_pivots(closes, threshold_dollars):
    n = len(closes)
    if n < 2:
        return [0]
    threshold_pts = threshold_dollars / DOLLAR_PER_POINT
    pivots = [0]
    last_pivot_price = closes[0]
    last_dir = None
    for i in range(1, n):
        price = closes[i]
        if last_dir is None:
            move = price - last_pivot_price
            if abs(move) >= threshold_pts:
                last_dir = 'up' if move > 0 else 'down'
                pivots.append(i)
                last_pivot_price = price
        elif last_dir == 'up':
            if price > closes[pivots[-1]]:
                pivots[-1] = i
                last_pivot_price = price
            elif (last_pivot_price - price) >= threshold_pts:
                last_dir = 'down'
                pivots.append(i)
                last_pivot_price = price
        elif last_dir == 'down':
            if price < closes[pivots[-1]]:
                pivots[-1] = i
                last_pivot_price = price
            elif (price - last_pivot_price) >= threshold_pts:
                last_dir = 'up'
                pivots.append(i)
                last_pivot_price = price
    return pivots


def compute_regression_features(closes, pivot_idx, windows):
    """At bar pivot_idx, compute β and residual for each window size.
    Returns dict of features. Returns None if any window lacks data."""
    feats = {}
    price_now = closes[pivot_idx]
    for W in windows:
        if pivot_idx < W - 1:
            return None
        y = closes[pivot_idx - W + 1: pivot_idx + 1]
        x = np.arange(W, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            return None
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = float(ym - slope * xm)
        fit_now = intercept + slope * (W - 1)
        residual = price_now - fit_now
        # Residual std in window
        fits = intercept + slope * x
        resid_std = float(np.std(y - fits, ddof=1)) if W > 2 else 1.0
        resid_norm = residual / resid_std if resid_std > 1e-9 else 0.0
        feats[f'beta_{W}'] = slope
        feats[f'res_{W}'] = residual
        feats[f'res_{W}_norm'] = resid_norm

    # Derived: sign alignment (fraction of windows where β has same sign as beta_60)
    ref_sign = np.sign(feats['beta_60']) if feats['beta_60'] != 0 else 0
    aligned = sum(1 for W in windows
                  if np.sign(feats[f'beta_{W}']) == ref_sign and ref_sign != 0)
    feats['beta_sign_align'] = aligned / len(windows)

    # Acceleration proxies
    feats['beta_60_minus_720'] = feats['beta_60'] - feats['beta_720']
    feats['beta_10_minus_60'] = feats['beta_10'] - feats['beta_60']
    feats['beta_20_minus_180'] = feats['beta_20'] - feats['beta_180']

    return feats


def process_day(path, threshold, windows, min_leg_dollars=0):
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].values.astype(np.float64)
    pivots = zigzag_pivots(closes, threshold)
    if len(pivots) < 3:
        return []

    events = []
    for i in range(len(pivots) - 1):
        piv_idx = pivots[i]
        next_piv_idx = pivots[i + 1]
        leg_points = closes[next_piv_idx] - closes[piv_idx]
        leg_dollars = abs(leg_points) * DOLLAR_PER_POINT
        if leg_dollars < min_leg_dollars:
            continue
        feats = compute_regression_features(closes, piv_idx, windows)
        if feats is None:
            continue
        events.append({
            'next_direction': 'UP' if leg_points > 0 else 'DOWN',
            'leg_dollars': leg_dollars,
            'feats': feats,
        })
    return events


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a = np.asarray(a)
    b = np.asarray(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                     / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((ma - mb) / pooled)


def analyze(events, label):
    """Return {feature: (d, mean_up, mean_down, n_up, n_down)}."""
    up = [e for e in events if e['next_direction'] == 'UP']
    down = [e for e in events if e['next_direction'] == 'DOWN']
    n_up, n_down = len(up), len(down)
    print(f'{label}: UP={n_up:,}, DOWN={n_down:,}, total={len(events):,}')
    feat_names = list(events[0]['feats'].keys())
    out = {}
    for name in feat_names:
        up_vals = [e['feats'][name] for e in up]
        down_vals = [e['feats'][name] for e in down]
        d = cohen_d(up_vals, down_vals)
        out[name] = {
            'd': d,
            'abs_d': abs(d),
            'mean_up': float(np.mean(up_vals)),
            'mean_down': float(np.mean(down_vals)),
            'n_up': n_up,
            'n_down': n_down,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=15.0,
                    help='Zigzag threshold $ (default 15)')
    ap.add_argument('--min-leg', type=float, default=0.0,
                    help='Skip legs smaller than this $ (default 0)')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS: {len(is_paths)} days | OOS: {len(oos_paths)} days')
    print(f'Threshold: ${args.threshold}  Min leg: ${args.min_leg}')

    is_events = []
    for p in tqdm(is_paths, desc='IS', unit='day'):
        is_events.extend(process_day(p, args.threshold, WINDOWS, args.min_leg))
    print(f'IS pivot events: {len(is_events):,}')

    oos_events = []
    for p in tqdm(oos_paths, desc='OOS', unit='day'):
        oos_events.extend(process_day(p, args.threshold, WINDOWS, args.min_leg))
    print(f'OOS pivot events: {len(oos_events):,}')

    is_res = analyze(is_events, 'IS')
    oos_res = analyze(oos_events, 'OOS')

    # Rank by IS |d|
    rank = sorted(is_res.keys(), key=lambda k: -is_res[k]['abs_d'])

    # Walk-forward shortlist
    wf_stable = []
    for name in rank:
        dis = is_res[name]['d']
        doos = oos_res[name]['d']
        if dis * doos > 0 and min(abs(dis), abs(doos)) >= 0.15:
            wf_stable.append((name, dis, doos))

    # Console
    print('\n=== Top 20 features by |d_IS| ===')
    print(f'{"Feature":<25} {"d_IS":>8} {"d_OOS":>8} {"WF":>4}')
    for name in rank[:20]:
        dis = is_res[name]['d']
        doos = oos_res[name]['d']
        wf = 'Y' if (dis * doos > 0 and min(abs(dis), abs(doos)) >= 0.15) else '.'
        print(f'{name:<25} {dis:>+8.3f} {doos:>+8.3f} {wf:>4}')

    # MD
    out = [f'# Regression-line Cohen d at zigzag pivots',
           '', f'**Threshold**: ${args.threshold}. **Min leg**: ${args.min_leg}.',
           '']
    out.append(f'IS pivot events: {len(is_events):,} | OOS: {len(oos_events):,}')
    out.append('')
    out.append('**Hypothesis**: at each zigzag pivot, regression-line slopes '
               'and residuals across multiple windows predict the direction '
               'of the next leg. Uses smoothed OLS β and residual — not raw '
               'velocity/z features.')
    out.append('')

    out.append('## Features computed')
    out.append('')
    out.append('For each pivot, at 5 window sizes (W=10, 20, 60, 180, 720 1m bars):')
    out.append('- `beta_W` — OLS slope (points per bar)')
    out.append('- `res_W` — price minus fitted value at pivot')
    out.append('- `res_W_norm` — residual / in-window residual std')
    out.append('- `beta_sign_align` — fraction of windows where β sign matches beta_60')
    out.append('- `beta_60_minus_720`, `beta_10_minus_60`, `beta_20_minus_180` — '
               'acceleration proxies (short slope minus long slope)')
    out.append('')

    out.append('## Top features by IS |d|')
    out.append('')
    out.append('| Rank | Feature | d_IS | d_OOS | UP mean | DOWN mean | Walk-fwd |')
    out.append('|---:|---|---:|---:|---:|---:|---|')
    for i, name in enumerate(rank[:25]):
        r_is = is_res[name]
        r_oos = oos_res[name]
        wf = '✓' if (r_is['d'] * r_oos['d'] > 0
                     and min(abs(r_is['d']), abs(r_oos['d'])) >= 0.15) else '—'
        out.append(f'| {i+1} | `{name}` | {r_is["d"]:+.3f} | {r_oos["d"]:+.3f} | '
                   f'{r_is["mean_up"]:+.3f} | {r_is["mean_down"]:+.3f} | {wf} |')
    out.append('')

    out.append('## Walk-forward stable features')
    out.append('')
    out.append('Sign match IS/OOS AND min(|d_IS|, |d_OOS|) >= 0.15.')
    out.append('')
    if wf_stable:
        out.append('| Feature | d_IS | d_OOS | min\\|d\\| | UP mean IS | DOWN mean IS |')
        out.append('|---|---:|---:|---:|---:|---:|')
        for name, dis, doos in sorted(wf_stable,
                                       key=lambda x: -min(abs(x[1]), abs(x[2]))):
            r_is = is_res[name]
            out.append(f'| `{name}` | {dis:+.3f} | {doos:+.3f} | '
                       f'{min(abs(dis), abs(doos)):.3f} | '
                       f'{r_is["mean_up"]:+.3f} | {r_is["mean_down"]:+.3f} |')
    else:
        out.append('_No features clear the bar. Regression-line slope/residual '
                   'does NOT predict next-leg direction either._')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWalk-forward stable features: {len(wf_stable)}')
    print(f'Wrote: {OUT_MD}')


if __name__ == '__main__':
    main()
