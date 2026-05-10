"""CLI runner — V2-native pipeline.

    python -m training_v2.run --is                       # IS only
    python -m training_v2.run --oos                      # OOS only
    python -m training_v2.run --is --oos                 # both
    python -m training_v2.run --days 2025_06_15,2025_06_16
    python -m training_v2.run --start 2025-06-01 --end 2025-06-30
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.ticker import V2Ticker, MultiDayV2Ticker
from training_iso_v2.engine import Engine
from training_iso_v2.strategies import (MAAlignTrendFollow, ReversionFromExtreme,
                                              VelocityBodyChord, RegimeAwareReversion,
                                              FilteredRegimeAwareReversion)
from training_iso_v2.exits import default_exit_suite
from training_iso_v2.regime_router import RegimeRouter
from training_iso_v2.v2_cols import swing_noise_w


OUTPUT_DIR = 'training_iso_v2/output'


def _entry_extras(state):
    """Capture entry-time signal values that exits depend on."""
    return {
        'entry_swing_noise': state.get(swing_noise_w('1m')),
    }


_STRAT_FACTORIES = {
    'MA_ALIGN': lambda: MAAlignTrendFollow(n_align=7, fire_on='5m'),
    'REVERSION': lambda: ReversionFromExtreme(tf='1m', fire_on='1m'),
    'VEL_BODY_CHORD': lambda: VelocityBodyChord(tf='5m', fire_on='5m'),
    # Regime-aware variant: same NMP trigger, applies (regime, direction) flip rule.
    # Tier names emitted: NMP_KEEP (no flip) and NMP_FLIP (direction inverted).
    'NMP_REGIME': lambda: RegimeAwareReversion(tf='1m', fire_on='1m'),
    # Regime-aware + per-cell quality filters (skip trades in loser tail of top discriminator).
    'NMP_FILTERED': lambda: FilteredRegimeAwareReversion(tf='1m', fire_on='1m'),
}


def _build_engine(strategies=None, exits=None,
                      cnn_filter=None, cnn_entry=None,
                      threshold_map=None) -> Engine:
    if strategies is None:
        # Order matters — first signal wins. MA_ALIGN fires on 5m closes,
        # CHORD on 5m, REVERSION on 1m. MA_ALIGN listed first to win 5m ties.
        strategies = [
            _STRAT_FACTORIES['MA_ALIGN'](),
            _STRAT_FACTORIES['VEL_BODY_CHORD'](),
            _STRAT_FACTORIES['REVERSION'](),
        ]
    if exits is None:
        exits = default_exit_suite()
    return Engine(strategies=strategies, exits=exits,
                       cnn_filter=cnn_filter, cnn_entry=cnn_entry,
                       entry_extras_hook=_entry_extras,
                       threshold_map=threshold_map)


def _strategies_from_csv(spec: str):
    if not spec:
        return None
    names = [s.strip() for s in spec.split(',') if s.strip()]
    out = []
    for n in names:
        if n not in _STRAT_FACTORIES:
            raise ValueError(f'Unknown strategy "{n}". Choices: {list(_STRAT_FACTORIES)}')
        out.append(_STRAT_FACTORIES[n]())
    return out


def _resolve_days(target: str = 'is', days_csv: str = None,
                       start: str = None, end: str = None) -> List[str]:
    """Resolve target spec to a list of YYYY_MM_DD day strings."""
    if days_csv:
        return [d.strip().replace('-', '_') for d in days_csv.split(',')]

    import glob
    l0_dir = 'DATA/ATLAS/FEATURES_5s_v2/L0'
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    if start:
        days = [d for d in days if d.replace('_', '-') >= start]
    if end:
        days = [d for d in days if d.replace('_', '-') <= end]
    if target == 'is':
        days = [d for d in days if d.startswith('2025_')]
    elif target == 'oos':
        days = [d for d in days if d.startswith('2026_')]
    return days


def run_target(target: str = 'is', days_csv: str = None,
                  start: str = None, end: str = None,
                  out_pkl: str = None,
                  strategies=None, exits=None,
                  cnn_filter=None, cnn_entry=None,
                  threshold_map=None) -> dict:
    days = _resolve_days(target, days_csv, start, end)
    if not days:
        print(f'No days resolved for target={target} '
                  f'(days_csv={days_csv}, start={start}, end={end})')
        return {'days': 0, 'trades': [], 'pnl': 0.0}

    engine = _build_engine(strategies, exits, cnn_filter, cnn_entry,
                                  threshold_map=threshold_map)
    multi = MultiDayV2Ticker(days=days)

    trades = engine.run(tqdm(multi, total=sum(1 for _ in days),
                                  desc=f'V2 {target.upper()}'))

    # Convert to DataFrame for reporting
    if trades:
        df = pd.DataFrame([{
            'day': t.entry_day,
            'tier': t.entry_tier,
            'dir': t.direction,
            'pnl': t.pnl,
            'peak': t.peak_pnl,
            'held': t.bars_held,
            'reason': t.exit_reason,
            'regime_idx': t.entry_regime_idx,
            'cnn_filtered': t.cnn_filtered,
            'cnn_generated': t.cnn_generated,
        } for t in trades])
    else:
        df = pd.DataFrame()

    n_days = len(days)
    n_trades = len(trades)
    total = float(df['pnl'].sum()) if n_trades else 0.0
    daily = df.groupby('day')['pnl'].sum() if n_trades else pd.Series(dtype=float)
    win_days = int((daily > 0).sum())
    print()
    print('=' * 60)
    print(f'V2 {target.upper()} — {n_days} days | {n_trades} trades | ${total:.2f}')
    print(f'  $/day: ${total / max(n_days, 1):.2f}')
    print(f'  $/trade: ${total / max(n_trades, 1):.2f}')
    print(f'  Winning days: {win_days}/{n_days}')
    if n_trades:
        win_rate_count = int((df['pnl'] > 0).sum()) / n_trades
        gp = float(df.loc[df['pnl'] > 0, 'pnl'].sum())
        gl = float(df.loc[df['pnl'] < 0, 'pnl'].sum())
        pf = gp / abs(gl) if gl < 0 else float('inf')
        # PF-based Trade WR per CLAUDE.md
        trade_wr = (pf - 1.0) if gl < 0 else float('inf')
        print(f'  Count win-rate: {win_rate_count:.1%}')
        print(f'  Profit factor: {pf:.2f}  (PF-Trade WR: {trade_wr:+.2f})')
        print(f'  Tier breakdown:')
        for tier, sub in df.groupby('tier'):
            print(f'    {tier:>20}: {len(sub):>5} trades  ${sub["pnl"].sum():>10.2f}  '
                      f'(${sub["pnl"].mean():>+6.2f}/trade)')
    print('=' * 60)

    if out_pkl is not None:
        os.makedirs(os.path.dirname(out_pkl) or '.', exist_ok=True)
        with open(out_pkl, 'wb') as f:
            pickle.dump(trades, f)
        if not df.empty:
            df.to_csv(out_pkl.replace('.pkl', '_summary.csv'), index=False)
        print(f'Saved: {out_pkl}')

    return {
        'days': n_days, 'trades': trades,
        'pnl': total, 'df': df,
    }


def parse_args():
    p = argparse.ArgumentParser(description='V2-native training run')
    p.add_argument('--is', dest='run_is', action='store_true', help='Run IS')
    p.add_argument('--oos', action='store_true', help='Run OOS')
    p.add_argument('--days', type=str, default=None,
                       help='comma-sep YYYY-MM-DD or YYYY_MM_DD list')
    p.add_argument('--start', type=str, default=None, help='YYYY-MM-DD')
    p.add_argument('--end', type=str, default=None, help='YYYY-MM-DD')
    p.add_argument('--smoke', action='store_true', help='Run on a single test day')
    p.add_argument('--cnn', type=str, default=None,
                       help='Path to trained CNN .pt; enables CNN filter+entry')
    p.add_argument('--cnn-filter-thr', type=float, default=0.5,
                       help='CNN softmax threshold for keeping deterministic entries')
    p.add_argument('--cnn-entry-thr', type=float, default=0.65,
                       help='CNN softmax threshold for spawning CNN-originated entries')
    p.add_argument('--cnn-no-entry', action='store_true',
                       help='Disable CNN-originated entries; CNN acts as filter only')
    p.add_argument('--cnn-no-filter', action='store_true',
                       help='Disable CNN filter; CNN acts as entry-only generator')
    p.add_argument('--cnn-device', type=str, default='cpu',
                       help='Device for CNN inference (cpu | cuda)')
    p.add_argument('--thresholds', type=str, default=None,
                       help='Path to thresholds.json from threshold_optimizer; '
                              'enables per-(regime, tier) adaptive exit thresholds')
    p.add_argument('--strategies', type=str, default=None,
                       help='Comma-sep subset: MA_ALIGN, REVERSION, VEL_BODY_CHORD. '
                              'Default = all three.')
    return p.parse_args()


def _build_cnn_hooks(args):
    """Construct CNN filter + entry callables based on CLI flags."""
    if not args.cnn:
        return None, None
    from training_iso_v2.cnn.inference import CNNFilter, CNNEntry
    cnn_filter = None if args.cnn_no_filter else CNNFilter(
        args.cnn, take_threshold=args.cnn_filter_thr, device=args.cnn_device)
    cnn_entry = None if args.cnn_no_entry else CNNEntry(
        args.cnn, entry_threshold=args.cnn_entry_thr, fire_on='5m',
        device=args.cnn_device)
    flags = []
    if cnn_filter is not None:
        flags.append(f'filter@{args.cnn_filter_thr}')
    if cnn_entry is not None:
        flags.append(f'entry@{args.cnn_entry_thr}')
    print(f'CNN active: {args.cnn}  [{", ".join(flags)}]')
    return cnn_filter, cnn_entry


def _load_thresholds(path):
    if not path:
        return None
    import json as _json
    with open(path, 'r') as f:
        thr = _json.load(f)
    n = len(thr.get('cells', {}))
    print(f'Adaptive thresholds loaded: {n} cells | universal={thr.get("universal")}')
    return thr


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cnn_filter, cnn_entry = _build_cnn_hooks(args)
    threshold_map = _load_thresholds(args.thresholds)
    strategies = _strategies_from_csv(args.strategies)
    if strategies is not None:
        print(f'Strategies: {[s.name for s in strategies]}')

    if args.smoke:
        run_target('is', days_csv='2025_06_15',
                       out_pkl=os.path.join(OUTPUT_DIR, 'smoke.pkl'),
                       strategies=strategies,
                       cnn_filter=cnn_filter, cnn_entry=cnn_entry,
                       threshold_map=threshold_map)
        return

    if args.days or args.start or args.end:
        run_target('is', days_csv=args.days,
                       start=args.start, end=args.end,
                       out_pkl=os.path.join(OUTPUT_DIR, 'custom.pkl'),
                       strategies=strategies,
                       cnn_filter=cnn_filter, cnn_entry=cnn_entry,
                       threshold_map=threshold_map)
        return

    if args.run_is or (not args.oos):
        run_target('is', out_pkl=os.path.join(OUTPUT_DIR, 'is.pkl'),
                       strategies=strategies,
                       cnn_filter=cnn_filter, cnn_entry=cnn_entry,
                       threshold_map=threshold_map)
    if args.oos:
        run_target('oos', out_pkl=os.path.join(OUTPUT_DIR, 'oos.pkl'),
                       strategies=strategies,
                       cnn_filter=cnn_filter, cnn_entry=cnn_entry,
                       threshold_map=threshold_map)


if __name__ == '__main__':
    main()
