"""
Hourly OOS report — shows PnL by hour for live trading comparison.

Reads OOS trades pickle and breaks down by:
  - Hour of day (UTC)
  - Per-tier per-hour
  - Best/worst hours
  - Session windows (Asia, London, NY)

Usage:
    python tools/hourly_oos_report.py                          # from OOS trades
    python tools/hourly_oos_report.py nn_v2/output/blended/oos_trades.pkl  # specific file
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hourly_report(trades_path='nn_v2/output/blended/oos_trades.pkl'):
    if not os.path.exists(trades_path):
        # Try physics trades
        alt = trades_path.replace('oos_trades', 'physics_oos_trades')
        if os.path.exists(alt):
            trades_path = alt
        else:
            print(f'No trades file: {trades_path}')
            return

    with open(trades_path, 'rb') as f:
        trades = pickle.load(f)

    print(f'Hourly OOS Report — {len(trades)} trades')
    print(f'{"="*70}')

    # Bucket by hour
    hourly = defaultdict(list)
    for t in trades:
        ts = t.get('timestamp', 0)
        if ts > 0:
            hour = datetime.utcfromtimestamp(ts).hour
            hourly[hour].append(t)

    # Session windows (UTC)
    sessions = {
        'Asia (22-06 UTC)': list(range(22, 24)) + list(range(0, 6)),
        'London (06-14 UTC)': list(range(6, 14)),
        'NY (14-22 UTC)': list(range(14, 22)),
    }

    # Per-hour stats
    print(f'\n  {"Hour":>4} {"Trades":>7} {"WR%":>5} {"PnL":>10} {"$/trade":>8} {"Best":>8} {"Worst":>8}')
    print(f'  {"-"*55}')

    hour_stats = []
    for h in range(24):
        t_list = hourly.get(h, [])
        if not t_list:
            continue
        n = len(t_list)
        wins = sum(1 for t in t_list if t['pnl'] > 0)
        total = sum(t['pnl'] for t in t_list)
        pnls = [t['pnl'] for t in t_list]
        best = max(pnls)
        worst = min(pnls)
        wr = wins / n * 100
        avg = total / n

        flag = ''
        if total > 1000: flag = '  <<<'
        elif total < -500: flag = '  !!!'

        print(f'  {h:>4}h {n:>7} {wr:>4.0f}% ${total:>9,.0f} ${avg:>7.1f} ${best:>7.0f} ${worst:>7.0f}{flag}')
        hour_stats.append({'hour': h, 'trades': n, 'wr': wr, 'pnl': total, 'avg': avg})

    # Session summary
    print(f'\n  Session Summary:')
    print(f'  {"Session":<25} {"Trades":>7} {"WR%":>5} {"PnL":>10} {"$/trade":>8}')
    print(f'  {"-"*55}')
    for session, hours in sessions.items():
        s_trades = [t for h in hours for t in hourly.get(h, [])]
        if not s_trades:
            continue
        n = len(s_trades)
        wins = sum(1 for t in s_trades if t['pnl'] > 0)
        total = sum(t['pnl'] for t in s_trades)
        print(f'  {session:<25} {n:>7} {wins/n*100:>4.0f}% ${total:>9,.0f} ${total/n:>7.1f}')

    # Per-tier per-hour heatmap (top 5 tiers)
    from collections import Counter
    tier_counts = Counter(t.get('entry_tier', '?') for t in trades)
    top_tiers = [t for t, _ in tier_counts.most_common(5)]

    print(f'\n  Per-Tier Hourly PnL (top 5 tiers):')
    print(f'  {"Hour":>4}', end='')
    for tier in top_tiers:
        print(f'  {tier:>14}', end='')
    print()
    print(f'  {"-"*(4 + 16*len(top_tiers))}')

    for h in range(24):
        t_list = hourly.get(h, [])
        if not t_list:
            continue
        print(f'  {h:>4}h', end='')
        for tier in top_tiers:
            tier_trades = [t for t in t_list if t.get('entry_tier') == tier]
            if tier_trades:
                total = sum(t['pnl'] for t in tier_trades)
                print(f'  ${total:>12,.0f}', end='')
            else:
                print(f'  {"---":>14}', end='')
        print()

    # Save
    os.makedirs('reports/findings', exist_ok=True)
    if hour_stats:
        pd.DataFrame(hour_stats).to_csv('reports/findings/hourly_oos.csv', index=False)
        print(f'\n  Saved: reports/findings/hourly_oos.csv')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'nn_v2/output/blended/oos_trades.pkl'
    hourly_report(path)
