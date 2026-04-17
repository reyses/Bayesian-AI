"""Compare new (honest) vs old (inflated) blended IS results.

Reads:
    training/output/trades/blended_is.pkl                         <- NEW (honest)
    training/output/archive_pre_lookahead_fix/blended_is.pkl      <- OLD (inflated)

Filters both to the overlapping day set (new set will usually be smaller
if only a subset of IS features was built). Reports side-by-side.

Usage:
    python tools/compare_lookahead_impact.py
"""
import os
import pickle
import numpy as np
from collections import Counter, defaultdict

NEW_PKL = 'training/output/trades/blended_is.pkl'
OLD_PKL = 'training/output/archive_pre_lookahead_fix/blended_is.pkl'


def load(path):
    if not os.path.exists(path):
        print(f'MISSING: {path}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def day_pnl(trades):
    by_day = defaultdict(float)
    for t in trades:
        by_day[t.get('day', '')] += t['pnl']
    return by_day


def tier_stats(trades):
    primaries = [t for t in trades if not t.get('is_chain', False)]
    chains = [t for t in trades if t.get('is_chain', False)]
    counts = Counter(t.get('entry_tier', '?') for t in primaries)
    stats = {}
    for tier, n in counts.items():
        sub = [t for t in primaries if t.get('entry_tier') == tier]
        wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
        pnl = sum(t['pnl'] for t in sub)
        stats[tier] = {'n': n, 'wr': wr, 'pnl': pnl, 'avg': pnl / n}
    return stats, len(primaries), len(chains)


def summarize(trades, label, days):
    primaries = [t for t in trades if not t.get('is_chain', False)]
    chains = [t for t in trades if t.get('is_chain', False)]
    total_pnl = sum(t['pnl'] for t in trades)
    prim_pnl = sum(t['pnl'] for t in primaries)
    chain_pnl = sum(t['pnl'] for t in chains)
    wins = sum(1 for t in primaries if t['pnl'] > 0)
    win_days = sum(1 for d, p in day_pnl(trades).items() if p > 0)
    n_days = len(days)
    return {
        'label': label,
        'days': n_days,
        'total_trades': len(trades),
        'primaries': len(primaries),
        'chains': len(chains),
        'total_pnl': total_pnl,
        'prim_pnl': prim_pnl,
        'chain_pnl': chain_pnl,
        'per_day': total_pnl / max(n_days, 1),
        'wr': wins / max(len(primaries), 1) * 100,
        'win_days': win_days,
    }


def print_header():
    print('=' * 80)
    print('LOOKAHEAD IMPACT: new (honest features) vs old (inflated features)')
    print('=' * 80)


def print_summary(new_s, old_s):
    print()
    print(f'{"Metric":<20} {"OLD":>18} {"NEW":>18} {"Delta":>14}')
    print('-' * 74)
    def row(label, o, n, fmt='+.0f', suf=''):
        d = n - o
        print(f'{label:<20} {format(o, fmt):>16}{suf:<2} {format(n, fmt):>16}{suf:<2} {format(d, fmt):>12}{suf}')
    row('Days',             old_s['days'],        new_s['days'],        '+d')
    row('Total trades',     old_s['total_trades'], new_s['total_trades'], '+,d')
    row('  Primary',        old_s['primaries'],   new_s['primaries'],   '+,d')
    row('  Chain',          old_s['chains'],      new_s['chains'],      '+,d')
    row('$/day',            old_s['per_day'],     new_s['per_day'],     '+,.0f')
    row('Total PnL',        old_s['total_pnl'],   new_s['total_pnl'],   '+,.0f')
    row('Primary PnL',      old_s['prim_pnl'],    new_s['prim_pnl'],    '+,.0f')
    row('Chain PnL',        old_s['chain_pnl'],   new_s['chain_pnl'],   '+,.0f')
    row('Win rate',         old_s['wr'],          new_s['wr'],          '+.0f', '%')
    row('Win days',         old_s['win_days'],    new_s['win_days'],    '+d')


def print_tiers(new_trades, old_trades):
    new_stats, new_p, _ = tier_stats(new_trades)
    old_stats, old_p, _ = tier_stats(old_trades)
    print()
    print('Tier breakdown (primary trades only):')
    print(f'  {"Tier":<16} {"OLD n":>6} {"NEW n":>6} {"OLD WR":>7} {"NEW WR":>7} '
          f'{"OLD $":>10} {"NEW $":>10} {"Delta":>10}')
    print(f'  {"-"*86}')
    all_tiers = sorted(set(list(new_stats.keys()) + list(old_stats.keys())))
    # Sort by absolute delta PnL
    def abs_delta(tier):
        o = old_stats.get(tier, {'pnl': 0})['pnl']
        n = new_stats.get(tier, {'pnl': 0})['pnl']
        return abs(n - o)
    all_tiers.sort(key=abs_delta, reverse=True)
    for tier in all_tiers:
        o = old_stats.get(tier, {'n': 0, 'wr': 0, 'pnl': 0})
        n = new_stats.get(tier, {'n': 0, 'wr': 0, 'pnl': 0})
        delta = n['pnl'] - o['pnl']
        print(f'  {tier:<16} {o["n"]:>6} {n["n"]:>6} '
              f'{o["wr"]:>6.0f}% {n["wr"]:>6.0f}% '
              f'${o["pnl"]:>+9,.0f} ${n["pnl"]:>+9,.0f} ${delta:>+9,.0f}')


def print_daily_delta(new_trades, old_trades, shared_days):
    new_by_day = day_pnl(new_trades)
    old_by_day = day_pnl(old_trades)
    deltas = [(d, new_by_day.get(d, 0) - old_by_day.get(d, 0)) for d in shared_days]
    deltas_sorted_up = sorted(deltas, key=lambda x: x[1], reverse=True)[:5]
    deltas_sorted_dn = sorted(deltas, key=lambda x: x[1])[:5]
    print()
    print('Biggest daily improvements (new - old):')
    for d, delta in deltas_sorted_up:
        print(f'  {d}: old=${old_by_day.get(d,0):>+8,.0f}  new=${new_by_day.get(d,0):>+8,.0f}  delta=${delta:>+8,.0f}')
    print()
    print('Biggest daily degradations (new - old):')
    for d, delta in deltas_sorted_dn:
        print(f'  {d}: old=${old_by_day.get(d,0):>+8,.0f}  new=${new_by_day.get(d,0):>+8,.0f}  delta=${delta:>+8,.0f}')


def main():
    new = load(NEW_PKL)
    old = load(OLD_PKL)
    if not new or not old:
        return

    # Filter both to shared days (new pkl usually has fewer days if partial build)
    new_days = set(t.get('day', '') for t in new)
    old_days = set(t.get('day', '') for t in old)
    shared = sorted(new_days & old_days)
    print_header()
    print(f'NEW days: {len(new_days)}')
    print(f'OLD days: {len(old_days)}')
    print(f'Shared:   {len(shared)}')

    new_f = [t for t in new if t.get('day', '') in shared]
    old_f = [t for t in old if t.get('day', '') in shared]

    new_s = summarize(new_f, 'NEW', shared)
    old_s = summarize(old_f, 'OLD', shared)

    print_summary(new_s, old_s)
    print_tiers(new_f, old_f)
    print_daily_delta(new_f, old_f, shared)

    # Big-picture take
    print()
    print('=' * 80)
    delta_pct = (new_s['per_day'] - old_s['per_day']) / max(abs(old_s['per_day']), 1) * 100
    print(f'HEADLINE: ${old_s["per_day"]:,.0f}/day -> ${new_s["per_day"]:,.0f}/day '
          f'({delta_pct:+.1f}%)')
    print(f'          Lookahead was worth ${old_s["per_day"] - new_s["per_day"]:+,.0f}/day')
    print('=' * 80)


if __name__ == '__main__':
    main()
