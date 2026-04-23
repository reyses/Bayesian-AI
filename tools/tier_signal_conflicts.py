"""
Tier Signal Conflict Analysis — which tiers fire on the same bar,
in what directions, and in what order?

Runs isolated classification (no trades, no chains, no position state) —
just evaluates each tier's fire function against every bar. Logs which
tiers would have fired and their directions. Aggregates to surface:

  1. Fire frequency per tier (bars where it fires alone vs co-fires)
  2. Co-fire matrix: for each pair of tiers, how often do they fire
     together? What fraction of co-fires are DIRECTION CONFLICTS?
  3. Cascade-order analysis: given current TIER_PRIORITY, which tier
     would actually take each bar (first-match-wins)?
  4. Direction conflicts: bars where tier A fires long while tier B
     fires short — these are the "waterfall" contested bars.

Why isolated mode: chains introduce bias (can't open counter-direction
chain). Bare classifier removes that.

Usage:
    python tools/tier_signal_conflicts.py                    # all IS days
    python tools/tier_signal_conflicts.py --days 30          # last 30 days
    python tools/tier_signal_conflicts.py --target oos       # OOS subset

Output:
    reports/findings/tier_signal_conflicts.md
"""
import os
import sys
import glob
import argparse
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso.nightmare_iso import IsoEngine, TIER_PRIORITY
from training.sfe_ticker import FeatureTicker


FEATURES_DIR = 'DATA/ATLAS/FEATURES_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUT_PATH = 'reports/findings/tier_signal_conflicts.md'


def classify_all_tiers(engine, feat, z, vr):
    """For a given bar, evaluate every tier's fire function.

    Returns dict {tier_name: direction or None}.
    """
    return {t: engine._tier_fires(t, feat, z, vr) for t in TIER_PRIORITY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='is', choices=['is', 'oos', 'all'])
    ap.add_argument('--days', type=int, default=None)
    args = ap.parse_args()

    # Select files
    all_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    if args.target == 'is':
        files = [f for f in all_files if '2025_' in os.path.basename(f)]
    elif args.target == 'oos':
        files = [f for f in all_files if '2026_' in os.path.basename(f)]
    else:
        files = all_files
    if args.days:
        files = files[:args.days]
    print(f'Analyzing {len(files)} day(s)')

    # Bare engine just for fire-function access
    engine = IsoEngine(only_tier='NMP_FADE')  # any tier — we use _tier_fires directly

    # Accumulators
    fire_counts = Counter()            # bars where each tier fires (any dir)
    fire_dir_counts = defaultdict(Counter)  # tier -> {long: N, short: N}
    co_fire = defaultdict(int)         # (tier_a, tier_b) -> count of co-fires
    co_fire_conflict = defaultdict(int)  # same pair, but opposite directions
    solo_fire = Counter()              # tier fires alone (no other tier fires)
    cascade_winner = Counter()         # TIER_PRIORITY first-match-wins per bar
    n_bars_considered = 0
    n_bars_with_fire = 0
    firing_tiers_dist = Counter()      # how many tiers fire at once (0/1/2/...)

    for fpath in tqdm(files, desc='Days'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            feat = state['features']
            # Only classify on 1m boundaries (matches engine entry cadence)
            ts = state['timestamp']
            if (int(ts) % 60) >= 5:
                continue
            z = feat[12]   # _1M_Z_IDX
            vr = feat[14]  # _1M_VR_IDX
            # Only bars where |z| > ROCHE trigger classification in engine
            # (so also filter here for apples-to-apples)
            if abs(z) <= 2.0:  # ROCHE
                continue
            n_bars_considered += 1
            dirs = classify_all_tiers(engine, feat, z, vr)
            fired = {t: d for t, d in dirs.items() if d is not None}
            firing_tiers_dist[len(fired)] += 1
            if not fired:
                continue
            n_bars_with_fire += 1

            # Fire counts
            for t, d in fired.items():
                fire_counts[t] += 1
                fire_dir_counts[t][d] += 1
            # Solo fires
            if len(fired) == 1:
                solo_fire[next(iter(fired))] += 1
            # Co-fires: all ordered-pairs
            items = list(fired.items())
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a_name, a_dir = items[i]
                    b_name, b_dir = items[j]
                    pair = tuple(sorted([a_name, b_name]))
                    co_fire[pair] += 1
                    if a_dir != b_dir:
                        co_fire_conflict[pair] += 1
            # Cascade winner
            for t in TIER_PRIORITY:
                if t in fired:
                    cascade_winner[t] += 1
                    break

    # ── Write report ─────────────────────────────────────────────
    L = []
    L.append('# Tier Signal Conflict Analysis')
    L.append('')
    L.append(f'**Bars evaluated (|z|>2 on 1m boundary):** {n_bars_considered:,}  '
             f'**Bars where a tier fired:** {n_bars_with_fire:,} '
             f'({100*n_bars_with_fire/max(n_bars_considered,1):.1f}%)')
    L.append('')
    L.append('## Simultaneous-fire distribution')
    L.append('')
    L.append('How many tiers fire at once?')
    L.append('')
    L.append('| # tiers firing | Bar count | % |')
    L.append('|---:|---:|---:|')
    for k in sorted(firing_tiers_dist.keys()):
        pct = firing_tiers_dist[k] / max(n_bars_considered, 1) * 100
        L.append(f'| {k} | {firing_tiers_dist[k]:,} | {pct:.1f}% |')
    L.append('')

    L.append('## Per-tier fire stats')
    L.append('')
    L.append('| Tier | Total fires | Solo fires | Long | Short | Cascade wins |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for t in TIER_PRIORITY:
        total = fire_counts[t]
        solo = solo_fire[t]
        long_n = fire_dir_counts[t].get('long', 0)
        short_n = fire_dir_counts[t].get('short', 0)
        cw = cascade_winner[t]
        L.append(f'| {t} | {total:,} | {solo:,} | {long_n:,} | {short_n:,} | {cw:,} |')
    L.append('')
    L.append('_**Cascade wins** = bars where this tier fires AND no earlier '
             'tier in TIER_PRIORITY order also fires. If cascade wins is far '
             'smaller than total fires, this tier often loses to an earlier '
             'tier._')
    L.append('')

    L.append('## Co-fire pairs (top 20 by frequency)')
    L.append('')
    L.append('Pairs of tiers that fire on the same bar. `Conflict %` = same-bar '
             'co-fires that point in opposite directions.')
    L.append('')
    L.append('| Tier A | Tier B | Co-fires | Conflict | Conflict % |')
    L.append('|---|---|---:|---:|---:|')
    sorted_pairs = sorted(co_fire.items(), key=lambda kv: -kv[1])[:20]
    for pair, n in sorted_pairs:
        conf = co_fire_conflict[pair]
        conf_pct = conf / max(n, 1) * 100
        L.append(f'| {pair[0]} | {pair[1]} | {n:,} | {conf:,} | {conf_pct:.0f}% |')
    L.append('')

    L.append('## Priority-order efficiency')
    L.append('')
    L.append('For each tier, `cascade_wins / total_fires` = fraction of its '
             'firings where it\'s the first in TIER_PRIORITY to fire. Low value '
             '= tier is being consistently overridden by an earlier tier.')
    L.append('')
    L.append('| Tier | Fires | Cascade wins | Win % |')
    L.append('|---|---:|---:|---:|')
    for t in TIER_PRIORITY:
        total = fire_counts[t]
        cw = cascade_winner[t]
        pct = cw / max(total, 1) * 100
        L.append(f'| {t} | {total:,} | {cw:,} | {pct:.0f}% |')
    L.append('')
    L.append('---')
    L.append(f'_Generated by `tools/tier_signal_conflicts.py` on {args.target} days._')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    # Console summary
    print()
    print(f'Bars considered: {n_bars_considered:,}')
    print(f'Bars with a fire: {n_bars_with_fire:,}')
    print()
    print(f'{"Tier":<20} {"Fires":>8} {"Solo":>8} {"Long":>8} {"Short":>8} {"Cascade":>8}')
    print('-' * 68)
    for t in TIER_PRIORITY:
        print(f'{t:<20} {fire_counts[t]:>8,} {solo_fire[t]:>8,} '
              f'{fire_dir_counts[t].get("long",0):>8,} '
              f'{fire_dir_counts[t].get("short",0):>8,} '
              f'{cascade_winner[t]:>8,}')
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
