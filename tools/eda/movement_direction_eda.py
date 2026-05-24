"""
Direction-prediction EDA — Phase 1 (single feature) and Phase 2 (polynomial).

Given the tagged $15/8min movements (from tools/tag_15_movements.py), answer:
  Q1: does any single feature discriminate LONG-first vs SHORT-first?
  Q2: does any feature discriminate OPPORTUNITY vs DEAD ZONE?

Phase 1 — single feature:
  For each of 91 features, compute Cohen d between LONG-first and
  SHORT-first cohorts. Rank by |d|. Walk-forward stable = sign matches
  IS/OOS and min |d| >= 0.20.

Phase 2 — polynomial (run after Phase 1 if signal exists):
  Take top-K features, expand to polynomial features (pairs x_i*x_j,
  squares x_i^2). Compute Cohen d on expanded set. See if combinations
  beat single features.

Uses memory-efficient Welford's running stats — handles 2.87M events
without loading all features into memory at once.

Usage:
    python tools/movement_direction_eda.py                     # Phase 1 on IS+OOS
    python tools/movement_direction_eda.py --phase 2 --top-k 6 # Phase 2 after
    python tools/movement_direction_eda.py --target 15 --timeout 8

Output: reports/findings/movement_direction_eda_${target}_{timeout}m.md
"""
import os
import sys
import glob
import pickle
import argparse
from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FEATURES_5S_DIR_IS  = 'DATA/ATLAS/FEATURES_5s'  # 2025_* files
FEATURES_5S_DIR_OOS = 'DATA/ATLAS/FEATURES_5s'  # same dir, 2026_* files
PRICE_5S_DIR        = 'DATA/ATLAS/5s'
EVENTS_PKL_TEMPLATE = 'training_iso/output/trades/movements_${target}_{timeout}m.pkl'
EVENTS_OOS_PKL_TEMPLATE = 'training_iso/output/trades/movements_oos_${target}_{timeout}m.pkl'
OUT_MD_TEMPLATE = 'reports/findings/movement_direction_eda_${target}_{timeout}m.md'


class WelfordAccumulator:
    """Running mean/var per feature per cohort — memory-efficient."""
    def __init__(self, n_features):
        self.n_features = n_features
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)

    def update_batch(self, X):
        """X: (batch_n, n_features)"""
        if X.size == 0:
            return
        batch_n = X.shape[0]
        batch_mean = X.mean(axis=0)
        delta = batch_mean - self.mean
        new_count = self.count + batch_n
        self.mean = self.mean + delta * (batch_n / new_count)
        batch_M2 = ((X - batch_mean) ** 2).sum(axis=0)
        self.M2 = self.M2 + batch_M2 + (delta ** 2) * (self.count * batch_n / new_count)
        self.count = new_count

    def var(self):
        return self.M2 / max(self.count - 1, 1)

    def std(self):
        return np.sqrt(self.var())


def cohen_d_vec(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """Per-feature Cohen d."""
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b)
                     / max(n_a + n_b - 2, 1))
    pooled = np.where(pooled < 1e-12, 1.0, pooled)
    return (mean_a - mean_b) / pooled


def load_events(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['events']


def accumulate_cohorts(events, features_dir, feature_cols, desc):
    """Return dict of {cohort: Welford accumulator}. Iterates day-by-day
    to keep memory bounded."""
    cohorts = {
        'LONG':    WelfordAccumulator(len(feature_cols)),
        'SHORT':   WelfordAccumulator(len(feature_cols)),
        'NEITHER': WelfordAccumulator(len(feature_cols)),
        'ANY_HIT': WelfordAccumulator(len(feature_cols)),
    }
    # Group events by day
    by_day = defaultdict(list)
    for e in events:
        by_day[e['day']].append(e)

    for day in tqdm(sorted(by_day.keys()), desc=desc, unit='day'):
        p = os.path.join(features_dir, f'{day}.parquet')
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
        # Index by timestamp for fast lookup
        ts_array = df['timestamp'].values.astype(np.int64)
        # Build matrix once
        feat_mat = df[feature_cols].values.astype(np.float64)

        day_events = by_day[day]
        # Group events per cohort for this day
        by_cohort = {'LONG': [], 'SHORT': [], 'NEITHER': [], 'ANY_HIT': []}
        for e in day_events:
            dir_ = e['first_direction']
            if dir_ in ('LONG', 'SHORT'):
                by_cohort[dir_].append(e['timestamp'])
                by_cohort['ANY_HIT'].append(e['timestamp'])
            elif dir_ == 'BOTH':
                # BOTH are tied — add to ANY_HIT but not LONG/SHORT
                by_cohort['ANY_HIT'].append(e['timestamp'])
        # Events with first_direction == 'NEITHER' are not in the tagged
        # events pickle (we only stored hits) — we need to infer NEITHER
        # from the full-day bar set.
        event_ts_set = set()
        for e in day_events:
            event_ts_set.add(e['timestamp'])
        all_ts = set(int(t) for t in ts_array)
        # NEITHER = all_ts \ event_ts where event was tagged at all
        neither_ts = all_ts - event_ts_set
        by_cohort['NEITHER'] = sorted(neither_ts)

        # Lookup feature rows and update cohorts
        for cohort, ts_list in by_cohort.items():
            if not ts_list:
                continue
            ts_arr = np.asarray(ts_list, dtype=np.int64)
            # Binary search for indices
            idx = np.searchsorted(ts_array, ts_arr)
            # Filter to valid (exact matches only)
            valid = (idx < len(ts_array)) & (ts_array[np.minimum(idx, len(ts_array)-1)] == ts_arr)
            valid_idx = idx[valid]
            if valid_idx.size == 0:
                continue
            batch = feat_mat[valid_idx]
            cohorts[cohort].update_batch(batch)

    return cohorts


def phase1_single_feature(is_cohorts, oos_cohorts, feature_cols, out,
                          target, timeout):
    out.append(f'# Movement direction EDA — target ${target}, {timeout} min')
    out.append('')
    out.append('## Phase 1 — single feature Cohen d')
    out.append('')
    out.append('**Binary question**: at a bar with a $15 hit, does any single '
               'feature discriminate LONG-first from SHORT-first?')
    out.append('')

    # ─── IS: LONG vs SHORT ─────────────────────────────────────
    is_long = is_cohorts['LONG']
    is_short = is_cohorts['SHORT']
    is_neither = is_cohorts['NEITHER']
    is_any = is_cohorts['ANY_HIT']
    oos_long = oos_cohorts['LONG']
    oos_short = oos_cohorts['SHORT']
    oos_neither = oos_cohorts['NEITHER']
    oos_any = oos_cohorts['ANY_HIT']

    out.append(f'IS cohort sizes: LONG={is_long.count:,}, '
               f'SHORT={is_short.count:,}, NEITHER={is_neither.count:,}')
    out.append(f'OOS cohort sizes: LONG={oos_long.count:,}, '
               f'SHORT={oos_short.count:,}, NEITHER={oos_neither.count:,}')
    out.append('')

    # LONG vs SHORT Cohen d per feature (IS and OOS)
    d_is_ls = cohen_d_vec(is_long.mean, is_long.var(), is_long.count,
                          is_short.mean, is_short.var(), is_short.count)
    d_oos_ls = cohen_d_vec(oos_long.mean, oos_long.var(), oos_long.count,
                           oos_short.mean, oos_short.var(), oos_short.count)
    # ANY vs NEITHER (dead-zone signal)
    d_is_dz = cohen_d_vec(is_any.mean, is_any.var(), is_any.count,
                          is_neither.mean, is_neither.var(), is_neither.count)
    d_oos_dz = cohen_d_vec(oos_any.mean, oos_any.var(), oos_any.count,
                           oos_neither.mean, oos_neither.var(), oos_neither.count)

    # Rank by IS |d| for LONG-vs-SHORT
    rank_ls = np.argsort(-np.abs(d_is_ls))
    out.append('### Top 30 features — LONG first vs SHORT first (IS)')
    out.append('')
    out.append('Positive d = feature is HIGHER on LONG-first bars.')
    out.append('')
    out.append('| Rank | Feature | d_IS | d_OOS | Walk-forward |')
    out.append('|---:|---|---:|---:|---|')
    wf_stable_ls = []
    for i, idx in enumerate(rank_ls[:30]):
        name = feature_cols[idx]
        dis = d_is_ls[idx]
        doos = d_oos_ls[idx]
        sig_is = '+' if dis >= 0 else ''
        sig_oos = '+' if doos >= 0 else ''
        wf = '✓' if (dis * doos > 0 and abs(dis) >= 0.05 and abs(doos) >= 0.05) else '—'
        if wf == '✓':
            wf_stable_ls.append((name, dis, doos))
        out.append(f'| {i+1} | `{name}` | {sig_is}{dis:.3f} | '
                   f'{sig_oos}{doos:.3f} | {wf} |')
    out.append('')

    # Walk-forward stable features
    out.append(f'### Walk-forward stable LONG-vs-SHORT features '
               f'({len(wf_stable_ls)} total)')
    out.append('')
    if wf_stable_ls:
        out.append('Sign match IS/OOS AND |d| >= 0.05 on both. Sorted by min|d|.')
        out.append('')
        out.append('| Feature | d_IS | d_OOS | min |d| |')
        out.append('|---|---:|---:|---:|')
        wf_stable_ls.sort(key=lambda x: -min(abs(x[1]), abs(x[2])))
        for name, dis, doos in wf_stable_ls[:20]:
            sig_is = '+' if dis >= 0 else ''
            sig_oos = '+' if doos >= 0 else ''
            out.append(f'| `{name}` | {sig_is}{dis:.3f} | '
                       f'{sig_oos}{doos:.3f} | {min(abs(dis), abs(doos)):.3f} |')
    else:
        out.append('_No features clear |d| >= 0.05 on both sides. Pure direction '
                   'prediction from single features is near-zero._')
    out.append('')

    # DEAD ZONE discrimination
    rank_dz = np.argsort(-np.abs(d_is_dz))
    out.append('### Top 20 features — ANY hit vs NEITHER (dead-zone signal)')
    out.append('')
    out.append('Positive d = feature is HIGHER on "opportunity" bars than dead zones.')
    out.append('')
    out.append('| Rank | Feature | d_IS | d_OOS | Walk-forward |')
    out.append('|---:|---|---:|---:|---|')
    wf_stable_dz = []
    for i, idx in enumerate(rank_dz[:20]):
        name = feature_cols[idx]
        dis = d_is_dz[idx]
        doos = d_oos_dz[idx]
        sig_is = '+' if dis >= 0 else ''
        sig_oos = '+' if doos >= 0 else ''
        wf = '✓' if (dis * doos > 0 and abs(dis) >= 0.05 and abs(doos) >= 0.05) else '—'
        if wf == '✓':
            wf_stable_dz.append((name, dis, doos))
        out.append(f'| {i+1} | `{name}` | {sig_is}{dis:.3f} | '
                   f'{sig_oos}{doos:.3f} | {wf} |')
    out.append('')

    return wf_stable_ls, wf_stable_dz, d_is_ls, d_oos_ls


def phase2_polynomial(is_events, oos_events, feature_cols, top_features, out):
    """Expand top features to polynomial space and measure Cohen d.

    Features expanded: x_i (linear), x_i^2 (squared), x_i * x_j (pair products).
    Only uses the top-K features from Phase 1 to keep dimensionality tractable.
    """
    out.append('## Phase 2 — polynomial feature expansion')
    out.append('')
    if not top_features:
        out.append('_No stable single features from Phase 1. Polynomial expansion '
                   'skipped._')
        out.append('')
        return

    k = len(top_features)
    top_names = [f[0] for f in top_features]
    out.append(f'Expanding {k} single features: ' + ', '.join(f'`{n}`' for n in top_names))
    out.append('')

    # Build expanded feature names
    expanded_names = list(top_names)
    for i in range(k):
        expanded_names.append(f'{top_names[i]}^2')
    for i in range(k):
        for j in range(i + 1, k):
            expanded_names.append(f'{top_names[i]} * {top_names[j]}')
    n_expanded = len(expanded_names)
    out.append(f'Expanded to {n_expanded} polynomial features '
               f'({k} linear + {k} squared + {k*(k-1)//2} pair products).')
    out.append('')

    # We need to recompute cohorts with the ORIGINAL 91 features restricted
    # to the top-K, then build polynomial terms. Do this inline with Welford.
    top_indices = [feature_cols.index(n) for n in top_names]

    # Group events by day
    def run_pass(events, features_dir, desc):
        long_acc = WelfordAccumulator(n_expanded)
        short_acc = WelfordAccumulator(n_expanded)
        by_day = defaultdict(list)
        for e in events:
            by_day[e['day']].append(e)
        for day in tqdm(sorted(by_day.keys()), desc=desc, unit='day'):
            p = os.path.join(features_dir, f'{day}.parquet')
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
            ts_array = df['timestamp'].values.astype(np.int64)
            # Pull only top-K columns
            X_raw = df[top_names].values.astype(np.float64)
            # Build expanded features for this day
            X_exp = np.zeros((len(df), n_expanded), dtype=np.float64)
            X_exp[:, :k] = X_raw
            col = k
            for i in range(k):
                X_exp[:, col] = X_raw[:, i] ** 2
                col += 1
            for i in range(k):
                for j in range(i + 1, k):
                    X_exp[:, col] = X_raw[:, i] * X_raw[:, j]
                    col += 1

            for e in by_day[day]:
                dir_ = e['first_direction']
                if dir_ not in ('LONG', 'SHORT'):
                    continue
                ts = e['timestamp']
                idx = np.searchsorted(ts_array, ts)
                if idx >= len(ts_array) or ts_array[idx] != ts:
                    continue
                row = X_exp[idx:idx+1]
                if dir_ == 'LONG':
                    long_acc.update_batch(row)
                else:
                    short_acc.update_batch(row)
        return long_acc, short_acc

    is_long_exp, is_short_exp = run_pass(is_events, FEATURES_5S_DIR_IS, 'IS poly')
    oos_long_exp, oos_short_exp = run_pass(oos_events, FEATURES_5S_DIR_OOS, 'OOS poly')

    d_is  = cohen_d_vec(is_long_exp.mean, is_long_exp.var(), is_long_exp.count,
                        is_short_exp.mean, is_short_exp.var(), is_short_exp.count)
    d_oos = cohen_d_vec(oos_long_exp.mean, oos_long_exp.var(), oos_long_exp.count,
                        oos_short_exp.mean, oos_short_exp.var(), oos_short_exp.count)

    rank = np.argsort(-np.abs(d_is))
    out.append(f'IS: LONG={is_long_exp.count:,}, SHORT={is_short_exp.count:,}')
    out.append(f'OOS: LONG={oos_long_exp.count:,}, SHORT={oos_short_exp.count:,}')
    out.append('')
    out.append('### Top 30 polynomial features — LONG vs SHORT')
    out.append('')
    out.append('| Rank | Feature | d_IS | d_OOS | Walk-forward |')
    out.append('|---:|---|---:|---:|---|')
    wf_count = 0
    for i, idx in enumerate(rank[:30]):
        name = expanded_names[idx]
        dis = d_is[idx]
        doos = d_oos[idx]
        sig_is = '+' if dis >= 0 else ''
        sig_oos = '+' if doos >= 0 else ''
        wf = '✓' if (dis * doos > 0 and abs(dis) >= 0.05
                     and abs(doos) >= 0.05) else '—'
        if wf == '✓':
            wf_count += 1
        out.append(f'| {i+1} | `{name}` | {sig_is}{dis:.3f} | '
                   f'{sig_oos}{doos:.3f} | {wf} |')
    out.append('')
    out.append(f'Walk-forward stable polynomial features: **{wf_count}** of {n_expanded}')
    out.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=15)
    ap.add_argument('--timeout', type=int, default=8)
    ap.add_argument('--phase', type=int, default=1,
                    help='1 (single feature) or 2 (polynomial; requires Phase 1 top-K)')
    ap.add_argument('--top-k', type=int, default=6,
                    help='Number of top Phase-1 features to polynomial-expand')
    ap.add_argument('--oos-events', default=None,
                    help='OOS events pkl (auto-generated via tag_15_movements.py)')
    args = ap.parse_args()

    # Load events for IS
    is_pkl = EVENTS_PKL_TEMPLATE.replace('${target}', str(args.target)).replace(
        '{timeout}', str(args.timeout))
    if not os.path.exists(is_pkl):
        print(f'IS events pkl not found: {is_pkl}')
        print('Run: python tools/tag_15_movements.py --target N --timeout M')
        sys.exit(1)

    print(f'Loading IS events from {is_pkl}...')
    is_events = load_events(is_pkl)

    # OOS events — may need to generate
    oos_pkl = args.oos_events or EVENTS_OOS_PKL_TEMPLATE.replace(
        '${target}', str(args.target)).replace('{timeout}', str(args.timeout))
    if not os.path.exists(oos_pkl):
        print(f'OOS events pkl not found: {oos_pkl}')
        print('Generating via inline OOS scan...')
        oos_events = tag_oos_inline(args.target, args.timeout)
        with open(oos_pkl, 'wb') as f:
            pickle.dump({'events': oos_events}, f)
    else:
        print(f'Loading OOS events from {oos_pkl}...')
        oos_events = load_events(oos_pkl)

    # Features (any day works to extract column names)
    sample_day = os.listdir(FEATURES_5S_DIR_IS)[0]
    sample_df = pd.read_parquet(os.path.join(FEATURES_5S_DIR_IS, sample_day))
    feature_cols = [c for c in sample_df.columns if c != 'timestamp']
    print(f'Feature count: {len(feature_cols)}')

    out = []

    # Phase 1
    print('\nPhase 1: single-feature cohort accumulation...')
    is_cohorts = accumulate_cohorts(is_events, FEATURES_5S_DIR_IS,
                                     feature_cols, 'IS P1')
    oos_cohorts = accumulate_cohorts(oos_events, FEATURES_5S_DIR_OOS,
                                      feature_cols, 'OOS P1')
    wf_stable_ls, wf_stable_dz, d_is_ls, d_oos_ls = phase1_single_feature(
        is_cohorts, oos_cohorts, feature_cols, out, args.target, args.timeout)

    if args.phase >= 2:
        # Use top-K features by IS |d| (not just walk-forward stable;
        # we want to probe combinations)
        top_k_names = [feature_cols[i] for i in
                       np.argsort(-np.abs(d_is_ls))[:args.top_k]]
        top_features = [(n, d_is_ls[feature_cols.index(n)],
                         d_oos_ls[feature_cols.index(n)]) for n in top_k_names]
        phase2_polynomial(is_events, oos_events, feature_cols,
                          top_features, out)

    # Write MD
    out_path = OUT_MD_TEMPLATE.replace('${target}', str(args.target)).replace(
        '{timeout}', str(args.timeout))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {out_path}')

    # Console summary
    print()
    print('=== PHASE 1 SUMMARY ===')
    print(f'Walk-forward stable LONG-vs-SHORT features: {len(wf_stable_ls)}')
    for name, dis, doos in wf_stable_ls[:10]:
        print(f'  {name:<30} d_IS={dis:+.3f}  d_OOS={doos:+.3f}')


def tag_oos_inline(target, timeout_min):
    """Quick inline OOS tagger reusing logic from tag_15_movements."""
    from tools.tag_15_movements import tag_day
    timeout_bars = int(round(timeout_min * 12))
    # 2026 files live in the same ATLAS/5s dir as 2025
    paths = sorted(glob.glob(os.path.join(PRICE_5S_DIR, '2026_*.parquet')))
    events = []
    print(f'Tagging OOS ({len(paths)} days)...')
    for p in tqdm(paths):
        day_name = os.path.basename(p).replace('.parquet', '')
        df = pd.read_parquet(p)
        day_events = tag_day(df, target, timeout_bars)
        for e in day_events:
            if e['long_hit_bar'] is not None or e['short_hit_bar'] is not None:
                e['day'] = day_name
                events.append(e)
    return events


if __name__ == '__main__':
    main()
