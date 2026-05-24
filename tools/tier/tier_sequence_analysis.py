"""
Tier Sequence Analysis — do some tier firings predict others? Is the
joint signal (A then B within N bars) stronger than A alone?

Given the isolated trade dump, for each pair of tiers (A, B):
  - Count A-alone (A fires, no B follow within window_bars)
  - Count A-then-B (B fires same direction within window after A entry)
  - Compare WR, $/trade, total PnL for each subset
  - Flag amplifiers: A-then-B has materially better WR or $/trade than A-alone

Usage:
    python tools/tier_sequence_analysis.py                   # default 15 bar window
    python tools/tier_sequence_analysis.py --window 10

Output: reports/findings/tier_sequence_analysis.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.nightmare_iso import TIER_PRIORITY


TRADES_PATH = 'training_iso/output/trades/iso_is.pkl'
OUT_PATH = 'reports/findings/tier_sequence_analysis.md'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=TRADES_PATH)
    ap.add_argument('--window', type=int, default=15, help='bars window (1-bar = 60s)')
    args = ap.parse_args()

    print(f'Loading {args.trades}...')
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    print(f'  {len(trades):,} trades')

    # Keep only chain_idx=0 primary entries
    primary = [t for t in trades if t.get('chain_idx', 0) == 0
               and t.get('entry_tier') and t.get('timestamp')]
    # Sort by timestamp
    primary.sort(key=lambda t: t['timestamp'])
    print(f'  {len(primary):,} primary entries (chain_idx=0)')

    window_sec = args.window * 60  # bars → seconds

    # Index by timestamp for fast lookup of followers
    # For each A entry, scan forward up to window_sec, collect B entries
    # with SAME direction as A.
    pair_counts = defaultdict(int)  # (A, B) -> count A-then-B pairs
    pair_pnl_A = defaultdict(list)  # (A, B) -> A-trade pnls when followed by B
    pair_pnl_B = defaultdict(list)  # (A, B) -> B-trade pnls
    tier_solo_pnl = defaultdict(list)  # A -> pnls of A when NOT followed by anything
    tier_any_pnl = defaultdict(list)   # A -> all A pnls

    # Naïve O(N*W) — N=29K, W~dozens of followers = fast enough
    n = len(primary)
    for i, a in enumerate(primary):
        tier_any_pnl[a['entry_tier']].append(a.get('pnl', 0.0))
        a_ts = a['timestamp']
        a_dir = a.get('dir')
        has_follower = False
        for j in range(i + 1, n):
            b = primary[j]
            if b['timestamp'] - a_ts > window_sec:
                break
            if b.get('dir') != a_dir:
                continue
            if b['entry_tier'] == a['entry_tier']:
                continue  # same tier = chain / re-entry, skip for pair analysis
            pair = (a['entry_tier'], b['entry_tier'])
            pair_counts[pair] += 1
            pair_pnl_A[pair].append(a.get('pnl', 0.0))
            pair_pnl_B[pair].append(b.get('pnl', 0.0))
            has_follower = True
        if not has_follower:
            tier_solo_pnl[a['entry_tier']].append(a.get('pnl', 0.0))

    # ── Console summary + markdown output ─────────────────────────
    L = []
    L.append(f'# Tier Sequence Analysis (window: {args.window} bars)')
    L.append('')
    L.append(f'For each A→B pair where B fires same direction within '
             f'{args.window} bars of A (and different tier), compare '
             f'A-trade PnL when followed vs A-alone (no follower).')
    L.append('')

    # Per-tier baseline
    L.append('## Per-tier baseline (all primaries, any follow-up)')
    L.append('')
    L.append('| Tier | N total | N solo (no follower) | $/tr total | $/tr solo |')
    L.append('|---|---:|---:|---:|---:|')
    for t in TIER_PRIORITY:
        any_ = tier_any_pnl.get(t, [])
        solo = tier_solo_pnl.get(t, [])
        if not any_:
            continue
        any_avg = np.mean(any_)
        solo_avg = np.mean(solo) if solo else 0.0
        L.append(f'| {t} | {len(any_)} | {len(solo)} | ${any_avg:+.2f} | ${solo_avg:+.2f} |')
    L.append('')

    # Ranked amplifier pairs (A → B): A's $/tr when followed is substantially
    # better than A-solo $/tr. Measures whether the FOLLOWER tier confirms
    # the setup (predictive of better A outcome).
    L.append('## Amplifier pairs (A-follows-B boosts A\'s WR / $/tr)')
    L.append('')
    L.append('Top pairs ranked by **A-trade $/tr lift when B follows**. '
             'If A-then-B $/tr > A-solo $/tr by >= $2, B firing is a '
             'positive confirmation signal for A.')
    L.append('')
    L.append('| A | B | Pairs | A-solo $/tr | A-then-B $/tr | Lift | '
             'A WR solo | A WR then-B |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|')

    pair_rows = []
    for pair, n_pair in pair_counts.items():
        if n_pair < 20:
            continue
        a_tier, b_tier = pair
        solo_a = tier_solo_pnl.get(a_tier, [])
        if not solo_a:
            continue
        solo_avg = np.mean(solo_a)
        then_b_avg = np.mean(pair_pnl_A[pair])
        lift = then_b_avg - solo_avg
        wr_solo = sum(1 for p in solo_a if p > 0) / max(len(solo_a), 1) * 100
        wr_then = sum(1 for p in pair_pnl_A[pair] if p > 0) / max(n_pair, 1) * 100
        pair_rows.append({
            'A': a_tier, 'B': b_tier, 'n': n_pair,
            'solo_avg': solo_avg, 'then_avg': then_b_avg,
            'lift': lift, 'wr_solo': wr_solo, 'wr_then': wr_then,
        })
    pair_rows.sort(key=lambda r: -r['lift'])

    for r in pair_rows[:20]:
        amp = ' **' if r['lift'] >= 5 else ('  *' if r['lift'] >= 2 else '')
        L.append(f'| {r["A"]} | {r["B"]} | {r["n"]} | '
                 f'${r["solo_avg"]:+.2f} | ${r["then_avg"]:+.2f} | '
                 f'${r["lift"]:+.2f}{amp} | {r["wr_solo"]:.0f}% | {r["wr_then"]:.0f}% |')
    L.append('')

    # Dampener pairs (lift < -$2)
    dampeners = [r for r in pair_rows if r['lift'] < -2]
    dampeners.sort(key=lambda r: r['lift'])
    if dampeners:
        L.append('## Dampener pairs (B following A HURTS A\'s outcome)')
        L.append('')
        L.append('| A | B | Pairs | A-solo $/tr | A-then-B $/tr | Drop |')
        L.append('|---|---|---:|---:|---:|---:|')
        for r in dampeners[:15]:
            L.append(f'| {r["A"]} | {r["B"]} | {r["n"]} | '
                     f'${r["solo_avg"]:+.2f} | ${r["then_avg"]:+.2f} | '
                     f'${r["lift"]:+.2f} |')
        L.append('')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    # Console summary
    print()
    print(f'Pairs analyzed: {len(pair_counts):,} (min 20 occurrences)')
    print()
    print('Top amplifiers (A-follows-B lifts A $/tr):')
    print(f'{"A":<20} {"B":<20} {"n":>5} {"solo":>8} {"then":>8} {"lift":>7}')
    print('-' * 72)
    for r in pair_rows[:10]:
        print(f'{r["A"]:<20} {r["B"]:<20} {r["n"]:>5} '
              f'${r["solo_avg"]:>+6.2f} ${r["then_avg"]:>+6.2f} ${r["lift"]:>+5.2f}')

    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
