"""Strategy run wrapper — drives a Strategy through V2 ForwardPass bars, records trades.

Pipeline:
    ForwardPassSystem (V2 cache)  -- per-day BarState stream
              |
              v
    Strategy.evaluate(state) -> EntrySignal (or None)
              |
              v
    Position tracker (1 contract, reverse-on-flip)
              |
              v
    Trades CSV (hardened-legs schema: day, entry_ts, leg_dir, entry_price,
                exit_ts, exit_price, pnl_pts, pnl_usd, r_price, atr_pts)
              |
              v
    Trade-outcome probability suite (optional, --analyze)

Strategy registry: currently ZigzagStrategy only. Add to STRATEGIES dict to register more.

ATR(14 of 1m) is computed once per day from the 1m parquet, multiplied by --atr-mult,
and passed to the strategy as min_reversal_ticks. Caller responsibility — the strategy
does not compute ATR itself.

Usage:
    python -m training.run_strategy --strategy zigzag --target oos
    python -m training.run_strategy --strategy zigzag --target is --atr-mult 4
    python -m training.run_strategy --strategy zigzag --target oos --days 2026_05_15 --analyze
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.FPS.forward_pass_system import ForwardPassSystem
from training.strategies.base import Strategy
from training.strategies.zigzag import (
    ZigzagStrategy, ATR_MULT_DEFAULT, MIN_BARS_5S_DEFAULT, TICK_SIZE)


TICK_VALUE = 0.50            # $/tick/contract (MNQ)
DOLLAR_PER_POINT = 2.0       # $/pt/contract (MNQ)
COMMISSION_PER_SIDE = 1.0    # $1 entry + $1 exit = $2 round trip


# ── Strategy registry ──────────────────────────────────────────────────
def make_strategy(name: str, min_reversal_ticks: int) -> Strategy:
    if name == 'zigzag':
        return ZigzagStrategy(min_reversal_ticks=min_reversal_ticks,
                              min_bars_5s=MIN_BARS_5S_DEFAULT)
    raise ValueError(f'Unknown strategy: {name!r}. Registered: zigzag')


# ── Atlas / output paths ──────────────────────────────────────────────
ATLAS_BY_TARGET = {
    'is':  ('DATA/ATLAS',     'DATA/ATLAS/FEATURES_5s_v2'),
    'oos': ('DATA/ATLAS_NT8', 'DATA/ATLAS_NT8/FEATURES_5s_v2'),
}
LABELS_CSV = 'DATA/ATLAS/regime_labels_2d.csv'

OUT_DIR = Path('reports/findings/strategy_runs')


def compute_atr_pts(day_1m_path: str, period: int = 14) -> float:
    """ATR(14) on 1m bars — same convention as build_zigzag_pivot_dataset.py."""
    df = pd.read_parquet(day_1m_path)
    h = df['high'].values; l = df['low'].values; c = df['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())


def resolve_days(features_root: str, target: str,
                 days_arg: Optional[list],
                 start: Optional[str], end: Optional[str]) -> list:
    if days_arg:
        return list(days_arg)
    all_p = sorted(glob.glob(os.path.join(features_root, 'L1_5s', '*.parquet')))
    all_days = [Path(p).stem for p in all_p]
    if target == 'is':
        days = [d for d in all_days if d.startswith('2025_')]
    elif target == 'oos':
        days = [d for d in all_days if d.startswith('2026_')]
    else:
        days = all_days
    if start: days = [d for d in days if d >= start]
    if end:   days = [d for d in days if d <= end]
    return days


def run_day(day: str, atlas_root: str, features_root: str,
            strategy_name: str, atr_mult: float) -> list:
    """Drive the strategy across one day. Returns list of closed-trade dicts in
    hardened-legs schema."""
    # 1. Pre-compute ATR -> min_reversal_ticks
    one_min_path = os.path.join(atlas_root, '1m', f'{day}.parquet')
    if not os.path.exists(one_min_path):
        return []
    atr_pts = compute_atr_pts(one_min_path, period=14)
    min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * atr_mult)))
    r_price = min_rev_ticks * TICK_SIZE   # R in points

    # 2. Build strategy
    strategy = make_strategy(strategy_name, min_rev_ticks)

    # 3. Iterate ForwardPassSystem
    try:
        fps = ForwardPassSystem(day=day, atlas_root=atlas_root,
                                features_root=features_root,
                                labels_csv=LABELS_CSV)
    except FileNotFoundError:
        return []

    trades = []
    # Single-position tracker (one contract, reverse-on-flip).
    pos_dir: Optional[str] = None   # 'long' | 'short' | None
    pos_entry_ts: Optional[int] = None
    pos_entry_price: Optional[float] = None
    last_state = None

    for state in fps:
        last_state = state
        sig = strategy.evaluate(state)
        if sig is None:
            continue

        close_5s = state.ohlcv_5s.get('close', state.price) if state.ohlcv_5s else state.price

        # Close existing position (if any) at the 5s close where the flip fired
        if pos_dir is not None:
            sign = 1 if pos_dir == 'long' else -1
            pnl_pts = (close_5s - pos_entry_price) * sign
            pnl_usd = pnl_pts * DOLLAR_PER_POINT - 2 * COMMISSION_PER_SIDE
            trades.append({
                'day': day,
                'entry_ts': pos_entry_ts,
                'leg_dir': 'LONG' if pos_dir == 'long' else 'SHORT',
                'entry_price': pos_entry_price,
                'exit_ts': int(state.timestamp),
                'exit_price': close_5s,
                'pnl_pts': pnl_pts,
                'pnl_usd': pnl_usd,
                'r_price': r_price,
                'atr_pts': atr_pts,
            })

        # Open new position in the flipped direction
        pos_dir = sig.direction
        pos_entry_ts = int(state.timestamp)
        pos_entry_price = close_5s

    # 4. Day-end: force-close any open position at the last bar's 5s close
    if pos_dir is not None and last_state is not None:
        close_5s = last_state.ohlcv_5s.get('close', last_state.price) \
                   if last_state.ohlcv_5s else last_state.price
        sign = 1 if pos_dir == 'long' else -1
        pnl_pts = (close_5s - pos_entry_price) * sign
        pnl_usd = pnl_pts * DOLLAR_PER_POINT - 2 * COMMISSION_PER_SIDE
        trades.append({
            'day': day,
            'entry_ts': pos_entry_ts,
            'leg_dir': 'LONG' if pos_dir == 'long' else 'SHORT',
            'entry_price': pos_entry_price,
            'exit_ts': int(last_state.timestamp),
            'exit_price': close_5s,
            'pnl_pts': pnl_pts,
            'pnl_usd': pnl_usd,
            'r_price': r_price,
            'atr_pts': atr_pts,
        })

    return trades


def main():
    ap = argparse.ArgumentParser(
        description='Run a registered Strategy through V2 ForwardPass; write trades CSV.')
    ap.add_argument('--strategy', default='zigzag', choices=['zigzag'])
    ap.add_argument('--target', choices=['is', 'oos'], required=True)
    ap.add_argument('--atr-mult', type=float, default=ATR_MULT_DEFAULT)
    ap.add_argument('--days', nargs='*', default=None,
                    help='Specific day list YYYY_MM_DD (smoke testing).')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--out', default=None,
                    help='Output CSV path. Defaults to '
                         'reports/findings/strategy_runs/<strategy>_<target>.csv')
    ap.add_argument('--analyze', action='store_true',
                    help='After write, invoke tools/suites/trade_outcome_suite/run_all.py.')
    args = ap.parse_args()

    atlas_root, features_root = ATLAS_BY_TARGET[args.target]
    days = resolve_days(features_root, args.target, args.days, args.start, args.end)
    if not days:
        print(f'No days resolved for target={args.target}.')
        return

    out_path = args.out or str(OUT_DIR / f'{args.strategy}_{args.target}_atr{int(args.atr_mult)}.csv')
    out_p = Path(out_path); out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f'Strategy : {args.strategy}')
    print(f'Target   : {args.target}  ({len(days)} days)')
    print(f'ATR mult : {args.atr_mult}')
    print(f'Atlas    : {atlas_root}')
    print(f'Output   : {out_path}')
    print('')

    all_trades = []
    for day in tqdm(days, desc='days'):
        trades = run_day(day, atlas_root, features_root, args.strategy, args.atr_mult)
        all_trades.extend(trades)

    if not all_trades:
        print('No trades produced.')
        return

    df = pd.DataFrame(all_trades)
    df.to_csv(out_path, index=False)
    tot = df['pnl_usd'].sum()
    win_n = int((df['pnl_usd'] > 0).sum())
    print(f'\nWrote {len(df)} trades to {out_path}')
    print(f'  Total PnL : ${tot:,.0f}')
    print(f'  Win count : {win_n}/{len(df)} ({100.0 * win_n / len(df):.1f}%)')
    print(f'  Days      : {df["day"].nunique()}  ({tot / df["day"].nunique():.0f} $/day mean)')

    if args.analyze:
        # The 'strategy_run' source in trade_outcome_suite/excursions.py points at
        # the runner's canonical output paths. If you customized --out, the suite
        # will read from the canonical paths, not yours — pass the canonical path
        # via --out (or symlink) if you want analysis on a custom run.
        canonical_is = OUT_DIR / f'{args.strategy}_is_atr{int(args.atr_mult)}.csv'
        canonical_oos = OUT_DIR / f'{args.strategy}_oos_atr{int(args.atr_mult)}.csv'
        if Path(out_path).resolve() != (canonical_is if args.target == 'is' else canonical_oos).resolve():
            print(f'\n[warning] --analyze reads the canonical strategy_run paths '
                  f'({canonical_is}, {canonical_oos}), not your custom --out. '
                  f'Skip --analyze or write to the canonical path.')
        else:
            print('\nRunning trade-outcome probability suite (--source strategy_run)...')
            suite_script = 'tools/suites/trade_outcome_suite/trade_outcome_suite/run_all.py'
            subprocess.run([sys.executable, suite_script,
                            '--source', 'strategy_run', '--rebuild'],
                           check=False)


if __name__ == '__main__':
    main()
