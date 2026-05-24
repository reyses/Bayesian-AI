"""
Regression-line + S/R Cohen d at zigzag pivots — direction prediction
enhanced with support/resistance levels from prior 5 business days.

Extends tools/regression_line_cohen_d.py by adding S/R features:
  - For each current-day pivot, look back 5 prior business days of 1m bars
  - Apply zigzag at $30 threshold on that concatenated history
  - Each pivot price from prior 5 days = one S/R level
  - Compute distance from current price to nearest level above/below

Hypothesis: if current pivot is at resistance (level ABOVE close), next
leg should be DOWN (mean-reversion from resistance).  If at support (level
BELOW close), next leg should be UP (bounce from support).

Adds these features to the existing regression-line feature set:
  - sr_dist_above         : $ distance to nearest level above (resistance)
  - sr_dist_below         : $ distance to nearest level below (support)
  - sr_dist_nearest       : $ to closest level (signed: +above, -below)
  - sr_at_level           : 1 if within $5 of any level, else 0
  - sr_count_within_10    : # levels within $10 of current price
  - sr_count_within_25    : # levels within $25
  - sr_range_compression  : $ width of band (nearest_above - nearest_below)
  - sr_pos_in_range       : price percentile within band [0..1]

Usage:
    python tools/regression_line_cohen_d_sr.py
    python tools/regression_line_cohen_d_sr.py --sr-threshold 20 --prior-days 10

Output: reports/findings/regression_line_cohen_d_sr.md
"""
import os
import sys
import glob
import argparse
import bisect
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.regression_line_cohen_d import (
    zigzag_pivots, compute_regression_features, cohen_d
)


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/regression_line_cohen_d_sr.md'
WINDOWS = [10, 20, 60, 180, 720]
DOLLAR_PER_POINT = 2.0


def build_sr_levels(prior_day_paths, threshold_dollars):
    """Concatenate close series from prior days, apply zigzag, return
    sorted unique pivot prices."""
    if not prior_day_paths:
        return []
    closes = []
    for p in prior_day_paths:
        df = pd.read_parquet(p).sort_values('timestamp')
        closes.extend(df['close'].values.astype(np.float64))
    closes = np.asarray(closes)
    pivots = zigzag_pivots(closes, threshold_dollars)
    levels = sorted(set(float(closes[i]) for i in pivots))
    # Merge near-duplicate levels (within $2 = 1 tick × 4 = 1 pt)
    merged = []
    for lv in levels:
        if merged and abs(lv - merged[-1]) < 1.0:  # 1 pt = $2
            merged[-1] = (merged[-1] + lv) / 2.0
        else:
            merged.append(lv)
    return merged


def sr_features_for_price(current_price, levels):
    """Return S/R feature dict for current_price against sorted levels list."""
    if not levels:
        return {
            'sr_dist_above_pts': 999.0,
            'sr_dist_below_pts': 999.0,
            'sr_dist_nearest_pts': 0.0,
            'sr_at_level': 0.0,
            'sr_count_within_5pts': 0.0,   # 5 pts = $10
            'sr_count_within_12pts': 0.0,  # 12 pts = $25
            'sr_range_compression_pts': 999.0,
            'sr_pos_in_range': 0.5,
        }
    # bisect to find insertion index
    idx = bisect.bisect_left(levels, current_price)
    above = levels[idx] if idx < len(levels) else None
    below = levels[idx - 1] if idx > 0 else None
    dist_above = (above - current_price) if above is not None else 999.0
    dist_below = (current_price - below) if below is not None else 999.0
    # Signed nearest (positive = level above, negative = level below)
    if dist_above < dist_below:
        dist_nearest = dist_above
    else:
        dist_nearest = -dist_below
    at_level = 1.0 if min(dist_above, dist_below) < 2.5 else 0.0  # within $5
    count_within_5pts  = sum(1 for lv in levels if abs(lv - current_price) <= 5.0)
    count_within_12pts = sum(1 for lv in levels if abs(lv - current_price) <= 12.0)
    if above is not None and below is not None:
        range_comp = above - below
        pos_in_range = (current_price - below) / range_comp if range_comp > 1e-6 else 0.5
    else:
        range_comp = 999.0
        pos_in_range = 0.5
    return {
        'sr_dist_above_pts': float(dist_above),
        'sr_dist_below_pts': float(dist_below),
        'sr_dist_nearest_pts': float(dist_nearest),
        'sr_at_level': at_level,
        'sr_count_within_5pts': float(count_within_5pts),
        'sr_count_within_12pts': float(count_within_12pts),
        'sr_range_compression_pts': float(range_comp),
        'sr_pos_in_range': float(pos_in_range),
    }


def process_day_with_sr(path, prior_day_paths, threshold, windows,
                        sr_threshold, min_leg_dollars=0):
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].values.astype(np.float64)
    pivots = zigzag_pivots(closes, threshold)
    if len(pivots) < 3:
        return []

    sr_levels = build_sr_levels(prior_day_paths, sr_threshold)
    events = []
    for i in range(len(pivots) - 1):
        piv_idx = pivots[i]
        next_piv_idx = pivots[i + 1]
        leg_points = closes[next_piv_idx] - closes[piv_idx]
        leg_dollars = abs(leg_points) * DOLLAR_PER_POINT
        if leg_dollars < min_leg_dollars:
            continue
        reg_feats = compute_regression_features(closes, piv_idx, windows)
        if reg_feats is None:
            continue
        sr_feats = sr_features_for_price(closes[piv_idx], sr_levels)
        feats = {**reg_feats, **sr_feats}
        events.append({
            'next_direction': 'UP' if leg_points > 0 else 'DOWN',
            'leg_dollars': leg_dollars,
            'feats': feats,
        })
    return events


def analyze(events, label):
    up = [e for e in events if e['next_direction'] == 'UP']
    down = [e for e in events if e['next_direction'] == 'DOWN']
    n_up, n_down = len(up), len(down)
    print(f'{label}: UP={n_up:,}, DOWN={n_down:,}')
    if n_up == 0 or n_down == 0:
        return {}
    feat_names = list(events[0]['feats'].keys())
    out = {}
    for name in feat_names:
        up_vals = np.array([e['feats'][name] for e in up])
        down_vals = np.array([e['feats'][name] for e in down])
        d = cohen_d(up_vals, down_vals)
        out[name] = {
            'd': d, 'abs_d': abs(d),
            'mean_up': float(up_vals.mean()),
            'mean_down': float(down_vals.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=15.0,
                    help='Current-day zigzag $ (default 15)')
    ap.add_argument('--sr-threshold', type=float, default=30.0,
                    help='Prior-days zigzag $ for S/R detection (default 30)')
    ap.add_argument('--prior-days', type=int, default=5,
                    help='Number of prior business days for S/R (default 5)')
    ap.add_argument('--min-leg', type=float, default=0.0)
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    # Combined list ordered chronologically for prior-day lookup
    all_paths = is_paths + oos_paths
    print(f'IS {len(is_paths)} days | OOS {len(oos_paths)} | '
          f'total {len(all_paths)}')
    print(f'Current zigzag: ${args.threshold} | S/R zigzag: ${args.sr_threshold} '
          f'| Prior days: {args.prior_days}')

    is_events = []
    for i, p in enumerate(tqdm(is_paths, desc='IS', unit='day')):
        # Find this path's index in all_paths and take prior N
        idx = all_paths.index(p)
        prior = all_paths[max(0, idx - args.prior_days):idx]
        is_events.extend(process_day_with_sr(
            p, prior, args.threshold, WINDOWS, args.sr_threshold,
            args.min_leg))
    print(f'IS pivot events: {len(is_events):,}')

    oos_events = []
    for i, p in enumerate(tqdm(oos_paths, desc='OOS', unit='day')):
        idx = all_paths.index(p)
        prior = all_paths[max(0, idx - args.prior_days):idx]
        oos_events.extend(process_day_with_sr(
            p, prior, args.threshold, WINDOWS, args.sr_threshold,
            args.min_leg))
    print(f'OOS pivot events: {len(oos_events):,}')

    is_res = analyze(is_events, 'IS')
    oos_res = analyze(oos_events, 'OOS')

    rank = sorted(is_res.keys(), key=lambda k: -is_res[k]['abs_d'])
    wf_stable = []
    for name in rank:
        dis = is_res[name]['d']
        doos = oos_res[name]['d']
        if dis * doos > 0 and min(abs(dis), abs(doos)) >= 0.15:
            wf_stable.append((name, dis, doos))

    # Console
    print('\n=== Top 30 features by |d_IS| ===')
    print(f'{"Feature":<28} {"d_IS":>8} {"d_OOS":>8} {"WF":>4}')
    for name in rank[:30]:
        dis = is_res[name]['d']
        doos = oos_res[name]['d']
        wf = 'Y' if (dis * doos > 0
                     and min(abs(dis), abs(doos)) >= 0.15) else '.'
        print(f'{name:<28} {dis:>+8.3f} {doos:>+8.3f} {wf:>4}')

    # MD
    sr_feature_names = [n for n in is_res if n.startswith('sr_')]
    out = [f'# Regression-line + S/R Cohen d at zigzag pivots', '']
    out.append(f'**Current-day zigzag**: ${args.threshold}. '
               f'**S/R zigzag (prior {args.prior_days} days)**: ${args.sr_threshold}.')
    out.append('')
    out.append(f'IS pivot events: {len(is_events):,} | OOS: {len(oos_events):,}')
    out.append('')

    out.append('## S/R features only (new in this run)')
    out.append('')
    out.append('| Feature | d_IS | d_OOS | UP mean | DOWN mean | Walk-fwd |')
    out.append('|---|---:|---:|---:|---:|---|')
    sr_ranked = sorted(sr_feature_names, key=lambda n: -is_res[n]['abs_d'])
    for name in sr_ranked:
        r_is = is_res[name]
        r_oos = oos_res[name]
        wf = 'Y' if (r_is['d'] * r_oos['d'] > 0
                     and min(abs(r_is['d']), abs(r_oos['d'])) >= 0.15) else '.'
        out.append(f'| `{name}` | {r_is["d"]:+.3f} | {r_oos["d"]:+.3f} | '
                   f'{r_is["mean_up"]:+.3f} | {r_is["mean_down"]:+.3f} | {wf} |')
    out.append('')

    out.append('## All features (regression + S/R) — top 30 by |d_IS|')
    out.append('')
    out.append('| Rank | Feature | d_IS | d_OOS | UP mean | DOWN mean | Walk-fwd |')
    out.append('|---:|---|---:|---:|---:|---:|---|')
    for i, name in enumerate(rank[:30]):
        r_is = is_res[name]
        r_oos = oos_res[name]
        wf = 'Y' if (r_is['d'] * r_oos['d'] > 0
                     and min(abs(r_is['d']), abs(r_oos['d'])) >= 0.15) else '.'
        out.append(f'| {i+1} | `{name}` | {r_is["d"]:+.3f} | {r_oos["d"]:+.3f} | '
                   f'{r_is["mean_up"]:+.3f} | {r_is["mean_down"]:+.3f} | {wf} |')
    out.append('')

    out.append('## Walk-forward stable features')
    out.append('')
    out.append(f'Total: {len(wf_stable)}. Sorted by min(|d_IS|, |d_OOS|).')
    out.append('')
    out.append('| Feature | d_IS | d_OOS | min\\|d\\| |')
    out.append('|---|---:|---:|---:|')
    for name, dis, doos in sorted(wf_stable,
                                   key=lambda x: -min(abs(x[1]), abs(x[2]))):
        out.append(f'| `{name}` | {dis:+.3f} | {doos:+.3f} | '
                   f'{min(abs(dis), abs(doos)):.3f} |')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWalk-forward stable features: {len(wf_stable)}')
    print(f'Wrote: {OUT_MD}')


if __name__ == '__main__':
    main()
