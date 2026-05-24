"""
Cascade Order Optimizer — find the TIER_PRIORITY order that maximizes
total PnL under forward-pass cascade rules.

Approach: use the isolated trade dump (iso_is.pkl) where every tier's
trades are already recorded. For each entry timestamp, we know which
tiers fired and what pnl each got. Under a cascade ordering, only the
FIRST tier in priority order (that fired at that timestamp) "wins" the
bar. Summing winning pnls for a proposed order gives its cascade total —
no pipeline rerun needed.

Three optimization strategies:
  1. ORACLE — per bar, pick the tier with best pnl (upper bound, non-consistent)
  2. EV-RANK — static by each tier's mean pnl (current TIER_PRIORITY)
  3. HILL-CLIMB — adjacent-swap search starting from EV-rank, keeps swaps that improve total

Runs all three, reports totals, and prints the best order.

Usage:
    python tools/cascade_order_optimizer.py
    python tools/cascade_order_optimizer.py --trades <path>
    python tools/cascade_order_optimizer.py --iterations 3   # hill-climb rounds
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


def group_by_bar(trades):
    """{timestamp: {tier: pnl}} — one entry per tier per bar timestamp.
    If a tier chained (multiple entries on same bar with chain_idx>0),
    we only use chain_idx=0 (primary entry)."""
    by_bar = defaultdict(dict)
    for t in trades:
        if t.get('chain_idx', 0) != 0:
            continue
        ts = int(t.get('timestamp', 0))
        tier = t.get('entry_tier')
        if not tier:
            continue
        # If the same tier has multiple primaries at the same timestamp
        # (shouldn't, but safe), keep the first one.
        if tier not in by_bar[ts]:
            by_bar[ts][tier] = float(t.get('pnl', 0.0))
    return by_bar


def simulate_cascade(by_bar, order):
    """For each bar, first tier in `order` that fired wins. Returns
    (total_pnl, per_tier_stats)."""
    rank = {t: i for i, t in enumerate(order)}
    per_tier = defaultdict(lambda: {'n': 0, 'pnl': 0.0})
    total = 0.0
    for ts, fires in by_bar.items():
        winner = None
        winner_rank = 10_000
        for tier in fires:
            r = rank.get(tier, 10_000)
            if r < winner_rank:
                winner_rank = r
                winner = tier
        if winner is None:
            continue
        p = fires[winner]
        per_tier[winner]['n'] += 1
        per_tier[winner]['pnl'] += p
        total += p
    return total, per_tier


def simulate_oracle(by_bar):
    """Upper bound: pick whichever tier had best pnl on each bar."""
    total = 0.0
    per_tier = defaultdict(lambda: {'n': 0, 'pnl': 0.0})
    for ts, fires in by_bar.items():
        if not fires:
            continue
        winner = max(fires.items(), key=lambda kv: kv[1])
        per_tier[winner[0]]['n'] += 1
        per_tier[winner[0]]['pnl'] += winner[1]
        total += winner[1]
    return total, per_tier


def hill_climb(by_bar, start_order, max_iterations=5, mode='all_pairs'):
    """Hill climb via swaps. `mode='adjacent'` only tries i,i+1 swaps.
    `mode='all_pairs'` tries every pair (i,j) — stronger neighborhood,
    escapes more local minima."""
    order = list(start_order)
    best_total, _ = simulate_cascade(by_bar, order)
    print(f'  Start: ${best_total:+,.0f}  order={" > ".join(order)}')
    n = len(order)
    for it in range(max_iterations):
        improved = False
        best_swap = None
        best_gain = 0.0
        # Enumerate candidate swaps
        if mode == 'adjacent':
            pairs = [(i, i+1) for i in range(n-1)]
        else:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        for i, j in pairs:
            swapped = list(order)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            total, _ = simulate_cascade(by_bar, swapped)
            gain = total - best_total
            if gain > best_gain:
                best_gain = gain
                best_swap = (i, j, swapped, total)
        if best_swap and best_gain > 0:
            i, j, swapped, total = best_swap
            order = swapped
            best_total = total
            improved = True
            print(f'  Iter {it+1} swap ({order[j]}<->{order[i]}) pos {i},{j}: '
                  f'+${best_gain:,.0f} -> ${best_total:+,.0f}')
        if not improved:
            print(f'  Iter {it+1}: no improving swap. Converged.')
            break
    return order, best_total


def print_breakdown(label, order, total, per_tier):
    print()
    print(f'=== {label} ===')
    print(f'Total: ${total:+,.0f}')
    print(f'Order: {" > ".join(order)}')
    print(f'{"Rank":>4} {"Tier":<20} {"N":>8} {"Total":>12} {"$/tr":>8}')
    print('-' * 60)
    for i, t in enumerate(order, 1):
        s = per_tier.get(t, {'n': 0, 'pnl': 0.0})
        avg = s['pnl'] / max(s['n'], 1)
        print(f'{i:>4} {t:<20} {s["n"]:>8,} ${s["pnl"]:>+10,.0f} ${avg:>+7.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=TRADES_PATH)
    ap.add_argument('--iterations', type=int, default=5)
    args = ap.parse_args()

    print(f'Loading {args.trades}...')
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    print(f'  {len(trades):,} trades')

    by_bar = group_by_bar(trades)
    print(f'  {len(by_bar):,} unique entry timestamps')
    multi_fire_bars = sum(1 for fires in by_bar.values() if len(fires) > 1)
    print(f'  {multi_fire_bars:,} bars with multi-tier fires (contested)')

    # Strategy 1: oracle upper bound
    oracle_total, oracle_stats = simulate_oracle(by_bar)
    print()
    print(f'ORACLE upper bound (pick best-pnl tier per bar): ${oracle_total:+,.0f}')

    # Strategy 2: current TIER_PRIORITY (already EV-rank)
    cur_total, cur_stats = simulate_cascade(by_bar, TIER_PRIORITY)
    print_breakdown('CURRENT TIER_PRIORITY (EV-rank)', list(TIER_PRIORITY), cur_total, cur_stats)

    # Strategy 3: hill-climb (all-pairs swap)
    print()
    print('=== HILL-CLIMB (all-pairs swap from current) ===')
    best_order, best_total = hill_climb(by_bar, TIER_PRIORITY, args.iterations,
                                        mode='all_pairs')
    _, best_stats = simulate_cascade(by_bar, best_order)
    print_breakdown('HILL-CLIMB RESULT', best_order, best_total, best_stats)

    # Gap to oracle
    print()
    print(f'Summary:')
    print(f'  Current total:   ${cur_total:+,.0f}')
    print(f'  Hill-climb best: ${best_total:+,.0f}')
    print(f'  Oracle upper:    ${oracle_total:+,.0f}')
    gap_pct = (best_total / oracle_total * 100) if oracle_total else 0
    print(f'  Hill-climb captures {gap_pct:.1f}% of oracle')


if __name__ == '__main__':
    main()
