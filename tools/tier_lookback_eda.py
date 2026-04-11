"""
Tier Lookback EDA — what happened BEFORE entry for winners vs losers?

For each tier, looks at 10 min of pre-entry physics to find signals
that predict whether the trade will win or lose.

Usage:
    python tools/tier_lookback_eda.py
    python tools/tier_lookback_eda.py --tier KILL_SHOT
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker

TRADES_DIR = 'training/output/trades'
FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'

_1M = 12
_5M = 24
_15M = 36
_1H = 48
_Z = 0
_DMI = 1
_VR = 2
_VEL = 3
_ACCEL = 4
_VOL = 5
_HURST = 7

LOOKBACK_BARS = 120  # 10 min at 5s
SAMPLE_PER_TIER = 600


def run_lookback(tier_name, trades, max_trades=SAMPLE_PER_TIER):
    """Compute lookback + forward physics for winners vs losers."""
    sample = trades[:max_trades]
    by_day = defaultdict(list)
    for t in sample:
        by_day[t['day']].append(t)

    winner_lb, loser_lb = [], []
    winner_fwd, loser_fwd = [], []

    for day, dt in tqdm(by_day.items(), desc=f'  {tier_name}', unit='day', leave=False):
        fpath = os.path.join(FEATURES_DIR, f'{day}.parquet')
        if not os.path.exists(fpath):
            continue
        ft = FeatureTicker(fpath, price_file=os.path.join(ATLAS_1M, f'{day}.parquet'))
        bars = list(ft)

        for t in dt:
            idx = None
            for i, b in enumerate(bars):
                if abs(b['timestamp'] - t['timestamp']) < 3:
                    idx = i
                    break
            if idx is None:
                continue

            ep = t['entry_price']
            d = t['dir']
            is_winner = t['pnl'] > 0

            # Lookback
            lb = []
            for j in range(max(0, idx - LOOKBACK_BARS), idx + 1):
                f = bars[j]['features_79d']
                lb.append({
                    'bar': j - idx,
                    'z': f[_1M + _Z], 'vr': f[_1M + _VR],
                    'vel': f[_1M + _VEL], 'accel': f[_1M + _ACCEL],
                    'vol': f[_1M + _VOL], 'dmi': f[_1M + _DMI],
                    'hurst': f[_1M + _HURST],
                    '5m_z': f[_5M + _Z], '5m_vel': f[_5M + _VEL],
                    '1h_z': f[_1H + _Z] if len(f) > _1H + _Z else 0,
                    '1h_vel': f[_1H + _VEL] if len(f) > _1H + _VEL else 0,
                })

            # Forward
            fwd = []
            for j in range(idx, min(idx + 180, len(bars))):
                b = bars[j]
                f = b['features_79d']
                p = b['price']
                if p < 100:
                    continue
                pnl = ((p - ep) if d == 'long' else (ep - p)) / 0.25 * 0.50
                fwd.append({
                    'bar': j - idx, 'pnl': pnl,
                    'z': abs(f[_1M + _Z]), 'vr': f[_1M + _VR],
                    'vel': abs(f[_1M + _VEL]),
                    '1h_vel': f[_1H + _VEL] if len(f) > _1H + _VEL else 0,
                })

            if is_winner:
                winner_lb.append(lb)
                winner_fwd.append(fwd)
            else:
                loser_lb.append(lb)
                loser_fwd.append(fwd)

    return winner_lb, loser_lb, winner_fwd, loser_fwd


def avg_at_bar(paths, bar_num, field):
    vals = []
    for path in paths:
        for p in path:
            if p['bar'] == bar_num:
                vals.append(p[field])
    return np.mean(vals) if vals else 0


def print_tier_report(tier_name, w_lb, l_lb, w_fwd, l_fwd):
    """Print lookback + forward comparison for one tier."""
    print(f'\n{"="*80}')
    print(f'{tier_name} — Winners: {len(w_lb)} | Losers: {len(l_lb)}')
    print(f'{"="*80}')

    # Lookback
    print(f'\nLOOKBACK (before entry):')
    fields = ['z', 'vr', 'vel', 'dmi', 'vol', 'hurst', '5m_vel', '1h_z', '1h_vel']
    header = f'{"Bar":>5} |'
    for f in fields:
        header += f' W_{f[:5]:>5} L_{f[:5]:>5} |'
    print(header)
    print('-' * len(header))

    lookback_bars = [-120, -72, -48, -24, -12, -6, 0]
    for cb in lookback_bars:
        line = f'{cb:>5} |'
        for field in fields:
            w_val = avg_at_bar(w_lb, cb, field)
            l_val = avg_at_bar(l_lb, cb, field)
            line += f' {w_val:>+5.1f} {l_val:>+5.1f} |'
        print(line)

    # Key differences at entry (bar 0)
    print(f'\nENTRY DIFF (bar 0, W - L):')
    for field in fields:
        w_val = avg_at_bar(w_lb, 0, field)
        l_val = avg_at_bar(l_lb, 0, field)
        diff = w_val - l_val
        if abs(diff) > 0.5:
            print(f'  {field:<10} W={w_val:>+6.1f}  L={l_val:>+6.1f}  diff={diff:>+6.1f}  ***')
        else:
            print(f'  {field:<10} W={w_val:>+6.1f}  L={l_val:>+6.1f}  diff={diff:>+6.1f}')

    # Lookback momentum: was the signal building or sudden?
    for field in ['z', 'vr', '1h_vel']:
        w_early = avg_at_bar(w_lb, -72, field)
        w_late = avg_at_bar(w_lb, 0, field)
        l_early = avg_at_bar(l_lb, -72, field)
        l_late = avg_at_bar(l_lb, 0, field)
        w_delta = w_late - w_early
        l_delta = l_late - l_early
        print(f'  {field:<10} buildup: W={w_delta:>+6.1f}  L={l_delta:>+6.1f}')

    # Forward conviction
    print(f'\nFORWARD (bar 12 PnL gap):')
    w_pnl12 = avg_at_bar(w_fwd, 12, 'pnl')
    l_pnl12 = avg_at_bar(l_fwd, 12, 'pnl')
    print(f'  W={w_pnl12:>+.1f}  L={l_pnl12:>+.1f}  gap={w_pnl12 - l_pnl12:>+.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=str, nargs='*', default=None)
    args = parser.parse_args()

    tiers = ['kill_shot', 'fade_against', 'cascade', 'ride_against',
             'fade_calm', 'mtf_breakout', 'mtf_exhaustion',
             'freight_train', 'fade_momentum',
             'exhaustion_bar', 'absorption', 'regime_flip']

    if args.tier:
        tiers = [t.lower() for t in args.tier]

    report_lines = []

    for tier_name in tiers:
        path = os.path.join(TRADES_DIR, f'maxfill_{tier_name}.pkl')
        if not os.path.exists(path):
            print(f'  {tier_name}: no trades file, skipping')
            continue

        with open(path, 'rb') as f:
            trades = pickle.load(f)

        if len(trades) < 20:
            print(f'  {tier_name}: {len(trades)} trades, too few')
            continue

        w_lb, l_lb, w_fwd, l_fwd = run_lookback(tier_name.upper(), trades)
        if w_lb and l_lb:
            print_tier_report(tier_name.upper(), w_lb, l_lb, w_fwd, l_fwd)

    # Save report
    import time
    report_path = f'reports/findings/tier_lookback_{time.strftime("%Y-%m-%d")}.md'
    print(f'\nDone. Console output above is the report.')
    print(f'(Pipe to file: python tools/tier_lookback_eda.py > {report_path})')


if __name__ == '__main__':
    main()
