"""
Z-stratified direction analysis — anchor on z_se before asking direction.

Hypothesis: at z_high, price is at upper band → mean-reversion → SHORT-first.
At z_low, lower band → LONG-first. The unconditional Cohen d cancels these
out (deep-low LONG bias averages with deep-high SHORT bias = net zero).
Stratifying by z reveals the directional bias WITHIN each regime.

Done at multiple TFs: 15s, 1m, 5m, 15m, 1h. The user's intuition is that
5s noise obscures signal — 1m or 5m anchoring may clean it up.

Additionally offers "--sample-1m" to only use events at 1m boundaries
(every 12th 5s bar) to reduce overlapping-state noise.

Usage:
    python tools/movement_z_stratified.py                # z_se at all TFs
    python tools/movement_z_stratified.py --tf 5m        # single TF only
    python tools/movement_z_stratified.py --sample-1m    # 1m boundaries only
    python tools/movement_z_stratified.py --target 15 --timeout 8

Output: reports/findings/movement_z_stratified_${target}_{timeout}m.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
PRICE_5S_DIR    = 'DATA/ATLAS/5s'
EVENTS_PKL_TEMPLATE = 'training_iso/output/trades/movements_${target}_{timeout}m.pkl'
EVENTS_OOS_PKL_TEMPLATE = 'training_iso/output/trades/movements_oos_${target}_{timeout}m.pkl'
OUT_MD_TEMPLATE = 'reports/findings/movement_z_stratified_${target}_{timeout}m.md'

TF_LIST = ['15s', '1m', '5m', '15m', '1h']
Z_BINS_LABEL = [
    ('z < -2.5',         -np.inf, -2.5),
    ('-2.5 ≤ z < -1.5',  -2.5,    -1.5),
    ('-1.5 ≤ z < -0.5',  -1.5,    -0.5),
    ('-0.5 ≤ z ≤ 0.5',   -0.5,     0.5),
    ('0.5 < z ≤ 1.5',     0.5,     1.5),
    ('1.5 < z ≤ 2.5',     1.5,     2.5),
    ('z > 2.5',           2.5,     np.inf),
]


def load_events(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['events']


def bin_z(z_value):
    for i, (_, lo, hi) in enumerate(Z_BINS_LABEL):
        if lo <= z_value < hi:
            return i
    return len(Z_BINS_LABEL) - 1


def accumulate(events, features_dir, tf, sample_1m=False):
    """Walk events day-by-day. For each event, bin by z_se at requested TF
    and accumulate LONG/SHORT/BOTH counts per bin.

    Returns:
        bins: list of dicts, one per z-bin:
          {'long': int, 'short': int, 'both': int, 'z_mean': float}
    """
    feat_name = f'{tf}_z_se'
    n_bins = len(Z_BINS_LABEL)
    counts = [{'long': 0, 'short': 0, 'both': 0, 'z_sum': 0.0, 'z_n': 0}
              for _ in range(n_bins)]

    by_day = defaultdict(list)
    for e in events:
        by_day[e['day']].append(e)

    for day in tqdm(sorted(by_day.keys()), desc=f'{tf} {"1m" if sample_1m else "5s"}',
                     unit='day'):
        p = os.path.join(features_dir, f'{day}.parquet')
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        if feat_name not in df.columns:
            continue
        df = df.sort_values('timestamp').reset_index(drop=True)
        ts_arr = df['timestamp'].values.astype(np.int64)
        z_arr = df[feat_name].values.astype(np.float64)

        for e in by_day[day]:
            dir_ = e['first_direction']
            if dir_ == 'NEITHER':
                continue
            ts = int(e['timestamp'])
            if sample_1m and (ts % 60) != 0:
                continue
            idx = np.searchsorted(ts_arr, ts)
            if idx >= len(ts_arr) or ts_arr[idx] != ts:
                continue
            z = z_arr[idx]
            if np.isnan(z):
                continue
            bin_i = bin_z(z)
            counts[bin_i]['z_sum'] += z
            counts[bin_i]['z_n'] += 1
            if dir_ == 'LONG':
                counts[bin_i]['long'] += 1
            elif dir_ == 'SHORT':
                counts[bin_i]['short'] += 1
            elif dir_ == 'BOTH':
                counts[bin_i]['both'] += 1

    # Finalize
    out = []
    for i, c in enumerate(counts):
        total = c['long'] + c['short'] + c['both']
        c['total'] = total
        c['z_mean'] = c['z_sum'] / c['z_n'] if c['z_n'] else 0.0
        if total > 0:
            c['p_long'] = c['long'] / total
            c['p_short'] = c['short'] / total
            c['p_both'] = c['both'] / total
            # Directional bias: P(LONG | z) - P(SHORT | z)
            c['bias'] = c['p_long'] - c['p_short']
        else:
            c['p_long'] = c['p_short'] = c['p_both'] = c['bias'] = 0.0
        out.append(c)
    return out


def render_tf_table(tf, bins_is, bins_oos, out, sample_1m=False):
    sample_note = ' (1m boundaries only)' if sample_1m else ''
    out.append(f'## {tf}_z_se stratified{sample_note}')
    out.append('')
    out.append('Directional bias = P(LONG first) - P(SHORT first) per z-bin. '
               '**Mean-reversion hypothesis**: negative bias at z_high, '
               'positive bias at z_low.')
    out.append('')
    out.append('| z bin | z mean | IS N | IS P(L) | IS P(S) | IS bias | '
               'OOS N | OOS P(L) | OOS P(S) | OOS bias | Walk-fwd |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for i, (label, lo, hi) in enumerate(Z_BINS_LABEL):
        b_is = bins_is[i]
        b_oos = bins_oos[i]
        # Walk-forward: same sign of bias AND min |bias| >= 0.03
        wf = '✓' if (b_is['bias'] * b_oos['bias'] > 0
                     and abs(b_is['bias']) >= 0.03
                     and abs(b_oos['bias']) >= 0.03) else '—'
        if b_is['total'] > 0:
            out.append(f'| {label} | {b_is["z_mean"]:+.2f} | '
                       f'{b_is["total"]:,} | '
                       f'{b_is["p_long"]*100:.1f}% | '
                       f'{b_is["p_short"]*100:.1f}% | '
                       f'{b_is["bias"]*100:+.1f}pp | '
                       f'{b_oos["total"]:,} | '
                       f'{b_oos["p_long"]*100:.1f}% | '
                       f'{b_oos["p_short"]*100:.1f}% | '
                       f'{b_oos["bias"]*100:+.1f}pp | '
                       f'{wf} |')
        else:
            out.append(f'| {label} | — | 0 | — | — | — | — | — | — | — | — |')
    out.append('')


def summarize_tf(tf, bins_is, bins_oos):
    """Compute headline: max |bias| and walk-forward-stable bins."""
    max_is = max(abs(b['bias']) for b in bins_is if b['total'] > 0) if any(b['total'] > 0 for b in bins_is) else 0
    max_oos = max(abs(b['bias']) for b in bins_oos if b['total'] > 0) if any(b['total'] > 0 for b in bins_oos) else 0
    wf_bins = 0
    for b_is, b_oos in zip(bins_is, bins_oos):
        if (b_is['bias'] * b_oos['bias'] > 0 and
                abs(b_is['bias']) >= 0.03 and
                abs(b_oos['bias']) >= 0.03):
            wf_bins += 1
    return max_is, max_oos, wf_bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=15)
    ap.add_argument('--timeout', type=int, default=8)
    ap.add_argument('--tf', default=None, help='Single TF only (15s/1m/5m/15m/1h)')
    ap.add_argument('--sample-1m', action='store_true',
                    help='Only use events at 1m boundaries (every 12th 5s bar)')
    args = ap.parse_args()

    tfs = [args.tf] if args.tf else TF_LIST

    is_pkl = EVENTS_PKL_TEMPLATE.replace('${target}', str(args.target)).replace(
        '{timeout}', str(args.timeout))
    oos_pkl = EVENTS_OOS_PKL_TEMPLATE.replace('${target}', str(args.target)).replace(
        '{timeout}', str(args.timeout))

    print(f'Loading IS: {is_pkl}')
    is_events = load_events(is_pkl)
    print(f'Loading OOS: {oos_pkl}')
    oos_events = load_events(oos_pkl)
    print(f'IS events: {len(is_events):,} · OOS events: {len(oos_events):,}')

    out = [f'# Z-stratified direction EDA — target ${args.target}, {args.timeout} min', '']
    sample_note = '1m-boundary sampling only' if args.sample_1m else '5s (all bars)'
    out.append(f'Sampling: {sample_note}')
    out.append('')
    out.append(f'IS events: {len(is_events):,} · OOS events: {len(oos_events):,}')
    out.append('')

    summary_rows = []
    for tf in tfs:
        print(f'\n--- TF: {tf} ---')
        bins_is  = accumulate(is_events,  FEATURES_5S_DIR, tf, args.sample_1m)
        bins_oos = accumulate(oos_events, FEATURES_5S_DIR, tf, args.sample_1m)
        render_tf_table(tf, bins_is, bins_oos, out, args.sample_1m)
        max_is, max_oos, wf_bins = summarize_tf(tf, bins_is, bins_oos)
        summary_rows.append({
            'tf': tf, 'max_is_bias': max_is, 'max_oos_bias': max_oos,
            'wf_bins': wf_bins,
        })

    # Summary section
    out.insert(4, '## TF summary')
    out.insert(5, '')
    out.insert(6, '| TF | Max IS bias | Max OOS bias | Walk-forward-stable bins |')
    out.insert(7, '|---|---:|---:|---:|')
    for i, r in enumerate(summary_rows):
        out.insert(8 + i, f'| {r["tf"]} | {r["max_is_bias"]*100:+.1f}pp | '
                   f'{r["max_oos_bias"]*100:+.1f}pp | {r["wf_bins"]}/'
                   f'{len(Z_BINS_LABEL)} |')
    out.insert(8 + len(summary_rows), '')

    out_path = OUT_MD_TEMPLATE.replace('${target}', str(args.target)).replace(
        '{timeout}', str(args.timeout))
    if args.sample_1m:
        out_path = out_path.replace('.md', '_1mboundary.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {out_path}')

    # Console summary
    print()
    print('=== Z-STRATIFIED SUMMARY ===')
    print(f'{"TF":<6} {"Max IS bias":>12} {"Max OOS bias":>13} {"WF stable bins":>15}')
    for r in summary_rows:
        print(f'{r["tf"]:<6} {r["max_is_bias"]*100:>+10.1f}pp '
              f'{r["max_oos_bias"]*100:>+10.1f}pp {r["wf_bins"]:>9}/{len(Z_BINS_LABEL)}')


if __name__ == '__main__':
    main()
