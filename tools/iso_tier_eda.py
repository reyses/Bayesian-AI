"""
Isolated Tier Forward Pass + EDA — run specific tiers without waterfall competition.

Disables all other tiers so the target tiers capture every eligible bar.
Then runs EDA on the captured trades.

Usage:
    python tools/iso_tier_eda.py FREIGHT_TRAIN FADE_MOMENTUM
    python tools/iso_tier_eda.py FREIGHT_TRAIN --days 50
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine

FEATURES_DIR_5S = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'


def parse_args():
    p = argparse.ArgumentParser(description='Isolated tier forward pass + EDA')
    p.add_argument('tiers', nargs='+', help='Tier names to isolate (e.g. FREIGHT_TRAIN FADE_MOMENTUM)')
    p.add_argument('--days', type=int, default=None, help='Limit number of days')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'])
    return p.parse_args()


def run_isolated(target_tiers, target='is', max_days=None):
    """Run forward pass with only target tiers active in the waterfall."""
    import glob

    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR_5S, '*.parquet')))
    if target == 'is':
        feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        feat_files = [f for f in feat_files if '2026_' in os.path.basename(f)]

    if max_days:
        feat_files = feat_files[:max_days]

    if not feat_files:
        print(f'No feature files for {target}')
        return []

    # Create engine with CNN disabled (physics only)
    engine = BlendedEngine(use_cnn=False)

    # Monkey-patch _classify_full_tier to ONLY return target tiers
    # Everything else returns None (skipped)
    original_classify = engine._classify_full_tier

    def isolated_classify(feat, z):
        direction, tier, flipped = original_classify(feat, z)
        if tier in target_tiers:
            return direction, tier, flipped
        # Force the tier to one of our targets based on velocity
        abs_vel = abs(feat[13])  # _1M_VELOCITY_IDX = 13
        if 'FREIGHT_TRAIN' in target_tiers and abs_vel >= 100.0:
            direction_ft = 'long' if feat[13] > 0 else 'short'
            return direction_ft, 'FREIGHT_TRAIN', True
        if 'FADE_MOMENTUM' in target_tiers and abs_vel >= 50.0:
            return direction, 'FADE_MOMENTUM', False
        # Below velocity threshold — skip this entry
        return None, None, False

    engine._classify_full_tier = isolated_classify

    # Patch on_state to skip entries where tier is None
    original_on_state = engine.on_state

    all_trades = []
    all_results = []

    print(f'ISO FORWARD PASS — {len(feat_files)} days, tiers: {target_tiers}')

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)

        for state in ft:
            # Feed state — the patched classify will only accept target tiers
            try:
                engine.on_state(state)
            except (TypeError, AttributeError):
                # If classify returned None, on_state may fail on _open_trade
                pass

        engine.force_close()

        for t in engine.trades:
            t['day'] = day_name
        all_trades.extend(engine.get_full_trades())

        day_pnl = engine.daily_pnl
        all_results.append({
            'day': day_name,
            'trades': len(engine.trades),
            'pnl': day_pnl,
        })

    return all_trades, all_results


def run_eda(trades, target_tiers):
    """Print EDA summary for isolated trades."""
    if not trades:
        print('\nNo trades captured.')
        return

    print(f'\n{"="*60}')
    print(f'EDA — {len(trades)} trades from tiers: {target_tiers}')
    print(f'{"="*60}')

    # Basic stats
    pnls = [t['pnl'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    held = [t.get('held', 0) for t in trades]

    print(f'\n  Trades: {len(trades)}')
    print(f'  Winners: {len(winners)} ({len(winners)/len(trades)*100:.0f}%)')
    print(f'  Losers:  {len(losers)} ({len(losers)/len(trades)*100:.0f}%)')
    print(f'  Total PnL: ${sum(pnls):,.0f}')
    print(f'  Avg PnL:   ${np.mean(pnls):.1f}')
    print(f'  Median PnL: ${np.median(pnls):.1f}')
    print(f'  Avg held:  {np.mean(held):.0f} bars')

    # Per-tier breakdown
    tiers = Counter(t.get('entry_tier', '?') for t in trades)
    print(f'\n  Per tier:')
    for tier, count in tiers.most_common():
        tier_pnls = [t['pnl'] for t in trades if t.get('entry_tier') == tier]
        tier_held = [t.get('held', 0) for t in trades if t.get('entry_tier') == tier]
        wr = sum(1 for p in tier_pnls if p > 0) / len(tier_pnls) * 100
        print(f'    {tier:<20} {count:>5} trades  WR={wr:4.0f}%  '
              f'avg=${np.mean(tier_pnls):>6.1f}  held={np.mean(tier_held):>4.0f}')

    # Direction breakdown
    print(f'\n  Direction:')
    for d in ['long', 'short']:
        d_trades = [t for t in trades if t.get('dir') == d]
        if d_trades:
            d_pnls = [t['pnl'] for t in d_trades]
            wr = sum(1 for p in d_pnls if p > 0) / len(d_pnls) * 100
            print(f'    {d:<10} {len(d_trades):>5} trades  WR={wr:4.0f}%  avg=${np.mean(d_pnls):.1f}')

    # Exit reasons
    exits = Counter(t.get('exit_reason', '?') for t in trades)
    print(f'\n  Exit reasons:')
    for reason, count in exits.most_common(10):
        sub = [t['pnl'] for t in trades if t.get('exit_reason') == reason]
        print(f'    {reason:<25} {count:>5}  avg=${np.mean(sub):>6.1f}')

    # PnL distribution
    print(f'\n  PnL distribution:')
    bins = [(-9999, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 9999)]
    for lo, hi in bins:
        n = sum(1 for p in pnls if lo <= p < hi)
        label = f'${lo}' if lo > -9999 else '<-$100'
        if hi == 9999:
            label = f'>$100'
        else:
            label = f'${lo} to ${hi}'
        print(f'    {label:<20} {n:>5} ({n/len(pnls)*100:4.0f}%)')

    # Per-day stats
    day_pnls = {}
    for t in trades:
        day = t.get('day', '?')
        day_pnls[day] = day_pnls.get(day, 0) + t['pnl']
    daily = list(day_pnls.values())
    if daily:
        print(f'\n  Daily:')
        print(f'    Days with trades: {len(daily)}')
        print(f'    $/day avg: ${np.mean(daily):.0f}')
        print(f'    $/day median: ${np.median(daily):.0f}')
        print(f'    Win days: {sum(1 for d in daily if d > 0)}/{len(daily)}')

    # Save trades
    out_dir = 'training/output/trades'
    os.makedirs(out_dir, exist_ok=True)
    tier_label = '_'.join(sorted(set(t.get('entry_tier', '?') for t in trades))).lower()
    pkl_path = os.path.join(out_dir, f'iso_{tier_label}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(trades, f)

    csv_rows = [{k: v for k, v in t.items()
                 if not isinstance(v, (list, dict, np.ndarray))} for t in trades]
    csv_path = os.path.join(out_dir, f'iso_{tier_label}.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f'\n  Saved: {pkl_path}')
    print(f'  Saved: {csv_path}')


def main():
    args = parse_args()
    target_tiers = set(args.tiers)
    print(f'Isolating tiers: {target_tiers}')

    trades, results = run_isolated(target_tiers, args.target, args.days)
    run_eda(trades, target_tiers)


if __name__ == '__main__':
    main()
