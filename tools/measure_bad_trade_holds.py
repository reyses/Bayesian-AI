"""
Measure hold times broken down by win vs loss cohort from pivot_physics_exit.
Verify the claim that losing trades ride longer than winners.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pivot_physics_exit import (
    load_day, simulate_day, REG_WINDOW, EOD_UTC_SECONDS, DOLLAR_PER_POINT
)

ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'


def main():
    r_entry_pts = 2.0 / DOLLAR_PER_POINT
    r_reg_pts = 8.0 / DOLLAR_PER_POINT
    min_res = 0.5
    sniper_sec = 30

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    def collect(paths, label):
        all_trades = []
        for p in tqdm(paths, desc=label, unit='day'):
            day = os.path.basename(p).replace('.parquet', '')
            sec_path = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
            feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
            if not os.path.exists(sec_path) or not os.path.exists(feat_path):
                continue
            loaded = load_day(sec_path, p, feat_path)
            if loaded is None:
                continue
            sec, closes_1m, ts_1m, residuals_1s, residuals_1m = loaded
            trades = simulate_day(sec, closes_1m, ts_1m, residuals_1s,
                                   residuals_1m, r_entry_pts, r_reg_pts,
                                   min_res, sniper_sec)
            for t in trades:
                t['day'] = day
            all_trades.extend(trades)
        return all_trades

    def analyze(trades, label):
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        print(f'\n=== {label} ===')
        print(f'Total: {len(trades)}  W: {len(wins)}  L: {len(losses)}')
        if wins:
            w_hold = np.array([t['held_sec'] for t in wins])
            w_pnl = np.array([t['pnl'] for t in wins])
            print(f'Winners — hold sec: mean={w_hold.mean():.0f} '
                  f'median={np.median(w_hold):.0f} p90={np.percentile(w_hold, 90):.0f} '
                  f'max={w_hold.max()}')
            print(f'          pnl: mean=${w_pnl.mean():+.2f} median=${np.median(w_pnl):+.2f}')
        if losses:
            l_hold = np.array([t['held_sec'] for t in losses])
            l_pnl = np.array([t['pnl'] for t in losses])
            print(f'Losers  — hold sec: mean={l_hold.mean():.0f} '
                  f'median={np.median(l_hold):.0f} p90={np.percentile(l_hold, 90):.0f} '
                  f'max={l_hold.max()}')
            print(f'          pnl: mean=${l_pnl.mean():+.2f} median=${np.median(l_pnl):+.2f}')
        # Exit reason distribution by outcome
        from collections import Counter
        w_exits = Counter(t['exit_reason'] for t in wins)
        l_exits = Counter(t['exit_reason'] for t in losses)
        print(f'Winner exits: {dict(w_exits)}')
        print(f'Loser exits:  {dict(l_exits)}')

        # Hold-time buckets for losers
        if losses:
            buckets = [(0, 300, '<5min'),
                       (300, 900, '5-15min'),
                       (900, 1800, '15-30min'),
                       (1800, 3600, '30-60min'),
                       (3600, 7200, '1-2h'),
                       (7200, float('inf'), '>2h')]
            l_hold = [t['held_sec'] for t in losses]
            print(f'Loser hold-time distribution:')
            for lo, hi, name in buckets:
                n = sum(1 for h in l_hold if lo <= h < hi)
                pct = n / len(losses) * 100 if losses else 0
                bar = '#' * int(pct / 2)
                print(f'  {name:<10} {n:>5,} ({pct:>5.1f}%) {bar}')

    is_trades = collect(is_paths, 'IS')
    oos_trades = collect(oos_paths, 'OOS')
    analyze(is_trades, 'IS')
    analyze(oos_trades, 'OOS')


if __name__ == '__main__':
    main()
