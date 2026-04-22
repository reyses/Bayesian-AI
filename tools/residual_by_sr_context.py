"""
Residual direction signal STRATIFIED by S/R context.

Question: does being near a support/resistance level AMPLIFY or WEAKEN the
regression-residual direction signal? S/R as context, not predictor.

For each zigzag pivot, we already know:
  - res_10_norm (strong direction signal: |d|=2.46 unstratified)

Now we also compute S/R context:
  - sr_near_support: price is near a level BELOW (distance to nearest < X pts)
  - sr_near_resistance: price is near a level ABOVE
  - sr_at_level: either direction, within $3
  - sr_far: > $25 from all levels

Then we re-measure res_10_norm's Cohen d WITHIN each context stratum.

Physics hypothesis:
  - At support: both residual (below mean) AND support agree → stronger UP bias
  - At resistance: both residual (above mean) AND resistance agree → stronger DOWN bias
  - Far from levels: only residual is active → weaker signal

If this holds, we'd see Cohen d for res_10_norm increase (more negative)
in at-S/R strata vs mid-range.

Usage:
    python tools/residual_by_sr_context.py

Output: reports/findings/residual_by_sr_context.md
"""
import os
import sys
import glob
import argparse
import bisect
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.regression_line_cohen_d import (
    zigzag_pivots, compute_regression_features, cohen_d
)
from tools.regression_line_cohen_d_sr import build_sr_levels, sr_features_for_price


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/residual_by_sr_context.md'
WINDOWS = [10, 20, 60, 180, 720]
DOLLAR_PER_POINT = 2.0

# S/R context strata (in POINTS; $ = pts × 2)
CONTEXT_STRATA = [
    ('AT_LEVEL',       0.0,    2.5),    # within $5 of any level
    ('NEAR_LEVEL',     2.5,    7.5),    # $5-$15 from nearest
    ('MEDIUM_DIST',    7.5,   15.0),    # $15-$30
    ('FAR_FROM_SR',   15.0,   50.0),    # $30-$100
    ('VERY_FAR',      50.0, 9999.0),    # >$100
]


def classify_sr_context(sr_feats):
    """Return (context_name, direction_of_nearest).
    direction_of_nearest: 'ABOVE' (near resistance) or 'BELOW' (near support).
    """
    dist_above = sr_feats.get('sr_dist_above_pts', 999.0)
    dist_below = sr_feats.get('sr_dist_below_pts', 999.0)
    nearest = min(dist_above, dist_below)
    direction = 'ABOVE' if dist_above < dist_below else 'BELOW'
    for name, lo, hi in CONTEXT_STRATA:
        if lo <= nearest < hi:
            return name, direction, nearest
    return 'OUT_OF_RANGE', direction, nearest


def process_day(path, prior_day_paths, threshold, sr_threshold, min_leg=0):
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
        if leg_dollars < min_leg:
            continue
        reg_feats = compute_regression_features(closes, piv_idx, WINDOWS)
        if reg_feats is None:
            continue
        sr_feats = sr_features_for_price(closes[piv_idx], sr_levels)
        context, direction, nearest_pts = classify_sr_context(sr_feats)
        events.append({
            'next_direction': 'UP' if leg_points > 0 else 'DOWN',
            'res_10_norm': reg_feats['res_10_norm'],
            'res_20_norm': reg_feats['res_20_norm'],
            'res_60_norm': reg_feats['res_60_norm'],
            'res_180_norm': reg_feats['res_180_norm'],
            'context': context,
            'sr_direction': direction,
            'sr_nearest_pts': nearest_pts,
        })
    return events


def analyze_stratum(events, feature_name='res_10_norm'):
    up = [e for e in events if e['next_direction'] == 'UP']
    down = [e for e in events if e['next_direction'] == 'DOWN']
    if not up or not down:
        return None
    up_vals = np.array([e[feature_name] for e in up])
    down_vals = np.array([e[feature_name] for e in down])
    d = cohen_d(up_vals, down_vals)
    # Direction prediction accuracy: predict UP if residual < 0, DOWN if residual > 0
    preds_correct = 0
    for e in events:
        pred = 'UP' if e[feature_name] < 0 else 'DOWN'
        if pred == e['next_direction']:
            preds_correct += 1
    acc = preds_correct / len(events) * 100
    return {
        'n': len(events),
        'n_up': len(up),
        'n_down': len(down),
        'd': d,
        'acc': acc,
        'mean_up': float(up_vals.mean()),
        'mean_down': float(down_vals.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=15.0)
    ap.add_argument('--sr-threshold', type=float, default=30.0)
    ap.add_argument('--prior-days', type=int, default=5)
    ap.add_argument('--min-leg', type=float, default=0.0)
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    all_paths = is_paths + oos_paths

    print(f'IS {len(is_paths)} | OOS {len(oos_paths)}')

    is_events = []
    for p in tqdm(is_paths, desc='IS', unit='day'):
        idx = all_paths.index(p)
        prior = all_paths[max(0, idx - args.prior_days):idx]
        is_events.extend(process_day(p, prior, args.threshold,
                                      args.sr_threshold, args.min_leg))
    print(f'IS pivot events: {len(is_events):,}')

    oos_events = []
    for p in tqdm(oos_paths, desc='OOS', unit='day'):
        idx = all_paths.index(p)
        prior = all_paths[max(0, idx - args.prior_days):idx]
        oos_events.extend(process_day(p, prior, args.threshold,
                                       args.sr_threshold, args.min_leg))
    print(f'OOS pivot events: {len(oos_events):,}')

    # Overall baselines
    is_all = analyze_stratum(is_events, 'res_10_norm')
    oos_all = analyze_stratum(oos_events, 'res_10_norm')
    print(f'\nBaseline (all pivots) res_10_norm:')
    print(f'  IS : d={is_all["d"]:+.3f}, acc={is_all["acc"]:.1f}%, n={is_all["n"]:,}')
    print(f'  OOS: d={oos_all["d"]:+.3f}, acc={oos_all["acc"]:.1f}%, n={oos_all["n"]:,}')

    # Per-context
    print(f'\n{"Context":<18} {"dir":<6} {"IS n":>7} {"IS d":>8} {"IS acc":>7} '
          f'{"OOS n":>7} {"OOS d":>8} {"OOS acc":>7}')
    print('-' * 90)

    strata_results = {}
    for ctx_name, _, _ in CONTEXT_STRATA + [('OUT_OF_RANGE', None, None)]:
        # Further split by direction (above/below nearest level)
        for sr_dir in ('ABOVE', 'BELOW'):
            is_sub = [e for e in is_events
                       if e['context'] == ctx_name and e['sr_direction'] == sr_dir]
            oos_sub = [e for e in oos_events
                        if e['context'] == ctx_name and e['sr_direction'] == sr_dir]
            if len(is_sub) < 50 or len(oos_sub) < 20:
                continue
            is_r = analyze_stratum(is_sub, 'res_10_norm')
            oos_r = analyze_stratum(oos_sub, 'res_10_norm')
            if is_r is None or oos_r is None:
                continue
            strata_results[(ctx_name, sr_dir)] = (is_r, oos_r)
            print(f'{ctx_name:<18} {sr_dir:<6} {is_r["n"]:>7,} '
                  f'{is_r["d"]:>+8.3f} {is_r["acc"]:>6.1f}% '
                  f'{oos_r["n"]:>7,} {oos_r["d"]:>+8.3f} {oos_r["acc"]:>6.1f}%')

    # MD
    out = [f'# Residual direction signal stratified by S/R context', '']
    out.append(f'**Current zigzag**: ${args.threshold}. '
               f'**S/R zigzag**: ${args.sr_threshold} on prior {args.prior_days} days.')
    out.append('')
    out.append('**Question**: does residual signal strength depend on '
               'distance-to-nearest-S/R?')
    out.append('')
    out.append(f'IS pivot events: {len(is_events):,} | OOS: {len(oos_events):,}')
    out.append('')

    out.append('## Baseline (all pivots, unstratified)')
    out.append('')
    out.append('| Dataset | N | d (res_10_norm) | Direction accuracy |')
    out.append('|---|---:|---:|---:|')
    out.append(f'| IS  | {is_all["n"]:,} | {is_all["d"]:+.3f} | {is_all["acc"]:.1f}% |')
    out.append(f'| OOS | {oos_all["n"]:,} | {oos_all["d"]:+.3f} | {oos_all["acc"]:.1f}% |')
    out.append('')
    out.append('Decision rule: predict UP if res_10_norm < 0, DOWN if > 0.')
    out.append('')

    out.append('## Per-stratum results')
    out.append('')
    out.append('Context strata by nearest-level-distance (points, $=pts×2):')
    out.append('')
    for name, lo, hi in CONTEXT_STRATA:
        out.append(f'- `{name}`: {lo}–{hi} pts (${lo*2:.0f}–${hi*2:.0f})')
    out.append('')
    out.append('Each row further splits by whether nearest level is ABOVE (resistance) '
               'or BELOW (support).')
    out.append('')
    out.append('| Context | Side | IS N | IS d | IS acc | OOS N | OOS d | OOS acc | vs baseline |')
    out.append('|---|---|---:|---:|---:|---:|---:|---:|---|')
    for (ctx_name, sr_dir), (is_r, oos_r) in strata_results.items():
        is_delta = is_r['acc'] - is_all['acc']
        oos_delta = oos_r['acc'] - oos_all['acc']
        label = f'{is_delta:+.1f}pp IS, {oos_delta:+.1f}pp OOS'
        out.append(f'| {ctx_name} | {sr_dir} | '
                   f'{is_r["n"]:,} | {is_r["d"]:+.3f} | {is_r["acc"]:.1f}% | '
                   f'{oos_r["n"]:,} | {oos_r["d"]:+.3f} | {oos_r["acc"]:.1f}% | '
                   f'{label} |')
    out.append('')

    out.append('## Interpretation')
    out.append('')
    out.append('- If S/R **amplifies** residual: AT_LEVEL accuracy > baseline '
               'accuracy.')
    out.append('- If S/R **weakens** residual: FAR_FROM_SR accuracy > AT_LEVEL.')
    out.append('- If S/R **irrelevant**: all strata ≈ baseline.')
    out.append('')

    # Compute best/worst stratum
    best = max(strata_results.items(),
               key=lambda kv: kv[1][0]['acc'] + kv[1][1]['acc'])
    worst = min(strata_results.items(),
                key=lambda kv: kv[1][0]['acc'] + kv[1][1]['acc'])
    (bc, bd), (bis, boos) = best
    (wc, wd), (wis, woos) = worst
    out.append(f'**Best stratum**: `{bc} {bd}` — IS {bis["acc"]:.1f}% / OOS {boos["acc"]:.1f}%')
    out.append(f'**Worst stratum**: `{wc} {wd}` — IS {wis["acc"]:.1f}% / OOS {woos["acc"]:.1f}%')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
