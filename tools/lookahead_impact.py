"""Compare new (honest) features against old (inflated) trades as feature builder produces days.

Runs the blended engine on each new FEATURES_5s/ day as it lands and compares
the tier distribution against the archived blended_is.pkl for the same day.
This gives an early read on how much lookahead was inflating training numbers.

Usage:
    python tools/lookahead_impact.py
"""
import os
import sys
import glob
import time
import pickle
import pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine
from core.ledger import Ledger
from core import sim_executor

FEATURES_DIR = 'DATA/FEATURES_5s'
OLD_TRADES = 'training/output/archive_pre_lookahead_fix/blended_is.pkl'
ATLAS_1M = 'DATA/ATLAS/1m'


def load_old_baseline():
    """Load archived inflated trades, indexed by day."""
    with open(OLD_TRADES, 'rb') as f:
        old = pickle.load(f)
    by_day = defaultdict(list)
    for t in old:
        by_day[t.get('day', '')].append(t)
    return by_day


def run_one_day(fpath):
    """Run blended engine on one day's features, return trade list."""
    day_name = os.path.basename(fpath).replace('.parquet', '')
    price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
    if not os.path.exists(price_file):
        price_file = None

    engine = BlendedEngine(use_cnn=False)
    ledger = Ledger()
    ft = FeatureTicker(fpath, price_file=price_file)
    trades = sim_executor.run(ledger, engine, ft, eod_close=True)
    return day_name, sim_executor.adapt_trades(trades)


def tier_stats(trades, label):
    """Compute tier distribution for a list of trades."""
    primaries = [t for t in trades if not t.get('is_chain', False)]
    if not primaries:
        return None
    tiers = Counter(t.get('entry_tier', '?') for t in primaries)
    stats = {}
    for tier, n in tiers.most_common():
        sub = [t for t in primaries if t.get('entry_tier') == tier]
        wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
        pnl = sum(t['pnl'] for t in sub)
        stats[tier] = {'n': n, 'wr': wr, 'pnl': pnl, 'avg': pnl / n}
    total_pnl = sum(t['pnl'] for t in trades)
    return {'label': label, 'total': len(trades), 'primaries': len(primaries),
            'total_pnl': total_pnl, 'tiers': stats}


def compare_day(day_name, new_trades, old_trades):
    """Side-by-side comparison of new vs old for one day."""
    new_s = tier_stats(new_trades, 'NEW')
    old_s = tier_stats(old_trades, 'OLD')

    print(f'\n{"="*80}')
    print(f'DAY: {day_name}')
    print(f'{"="*80}')
    if old_s:
        print(f'OLD (inflated): {old_s["primaries"]} primaries, ${old_s["total_pnl"]:+,.0f}')
    if new_s:
        print(f'NEW (honest):   {new_s["primaries"]} primaries, ${new_s["total_pnl"]:+,.0f}')

    if old_s and new_s:
        delta = new_s['total_pnl'] - old_s['total_pnl']
        pct = delta / max(abs(old_s['total_pnl']), 1) * 100
        flag = 'INFLATION' if delta < 0 else 'UNCHANGED' if abs(delta) < 10 else 'IMPROVED'
        print(f'Delta:          ${delta:+,.0f} ({pct:+.1f}%)  {flag}')

    # Tier comparison
    if new_s and old_s:
        all_tiers = sorted(set(list(new_s['tiers'].keys()) + list(old_s['tiers'].keys())))
        print(f'\n  {"Tier":<18} {"OLD n":>6} {"NEW n":>6} {"OLD $":>10} {"NEW $":>10} {"Diff":>10}')
        print(f'  {"-"*70}')
        for tier in all_tiers:
            o = old_s['tiers'].get(tier, {'n': 0, 'pnl': 0})
            n = new_s['tiers'].get(tier, {'n': 0, 'pnl': 0})
            diff = n['pnl'] - o['pnl']
            print(f'  {tier:<18} {o["n"]:>6} {n["n"]:>6} '
                  f'${o["pnl"]:>+9,.0f} ${n["pnl"]:>+9,.0f} ${diff:>+9,.0f}')


def main():
    print('Watching DATA/FEATURES_5s/ for new feature files...')
    print(f'Old baseline: {OLD_TRADES}')
    print()

    old_by_day = load_old_baseline()
    seen = set()
    new_totals = defaultdict(float)
    old_totals_seen = 0
    new_totals_seen = 0
    days_processed = 0

    while True:
        files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
        new_files = [f for f in files if os.path.basename(f) not in seen]

        for fpath in new_files:
            day_name = os.path.basename(fpath).replace('.parquet', '')
            seen.add(os.path.basename(fpath))

            # Wait a moment for file to be fully written
            time.sleep(0.5)

            try:
                _, new_trades = run_one_day(fpath)
            except Exception as e:
                print(f'[{day_name}] skipped: {e}')
                continue

            old_trades = old_by_day.get(day_name, [])
            compare_day(day_name, new_trades, old_trades)

            # Running totals
            days_processed += 1
            new_totals_seen += sum(t['pnl'] for t in new_trades)
            old_totals_seen += sum(t['pnl'] for t in old_trades)
            print(f'\n  Running ({days_processed} days): '
                  f'OLD=${old_totals_seen:+,.0f} ({old_totals_seen/days_processed:+,.0f}/day)  '
                  f'NEW=${new_totals_seen:+,.0f} ({new_totals_seen/days_processed:+,.0f}/day)')

        if not new_files:
            time.sleep(10)
            print(f'  ...waiting ({len(seen)} days processed)', end='\r')


if __name__ == '__main__':
    main()
