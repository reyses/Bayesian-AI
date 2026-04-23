"""
Stepwise sweep of phantom-entry + direction variants.

Tests combinations on IS only (quick), summarizes results in one table.
User picks the winner, we OOS-confirm separately.

Configs tested:
  1. baseline (phantom wait=1, min_fav=0) — reference
  2. wait=2, min_fav=0
  3. wait=3, min_fav=0
  4. wait=1, min_fav=$2 (1.0 pts)
  5. wait=1, min_fav=$5 (2.5 pts)
  6. wait=2, min_fav=$2
  7. direction=INVERSE + wait=1, min_fav=0

Usage:
    python tools/sweep_phantom.py
Output:
    reports/findings/sweep_phantom.md
"""
import os
import sys
import glob
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_RM_physics.ticker_1s import OneSecondTicker
from training_RM_physics.rm_physics_engine import RMPhysicsEngine


ATLAS_1M_DIR = 'DATA/ATLAS/1m'


def run_config(label, days, engine_params, direction_mode='mean_rev'):
    """Run one config across a list of day names. Returns stats dict."""
    all_trades = []
    for day in tqdm(days, desc=label, unit='day', leave=False):
        eng = RMPhysicsEngine(**engine_params)
        # Optional direction override: patch the _open direction at signal time.
        # We do this by monkey-patching the residual sign → direction map.
        # Simpler: flip via a lambda at the instance level.
        if direction_mode == 'inverse':
            # Swap: residual<0 → SHORT instead of LONG
            # Easiest monkey-patch: wrap on_state to flip direction after open
            orig_open = eng._open
            def flipped_open(direction, price, ts, feat, entry_residual):
                flipped = 'short' if direction == 'long' else 'long'
                return orig_open(flipped, price, ts, feat, entry_residual)
            eng._open = flipped_open
        try:
            ft = OneSecondTicker(day)
        except FileNotFoundError:
            continue
        for state in ft:
            eng.on_state(state)
        eng.force_close()
        for t in eng.trades:
            t['day'] = day
        all_trades.extend(eng.trades)
    return summarize(all_trades)


def summarize(trades):
    if not trades:
        return None
    pnls = np.array([x['pnl'] for x in trades])
    profit = float(pnls[pnls > 0].sum())
    loss = float(abs(pnls[pnls < 0].sum()))
    trade_wr = profit / loss - 1 if loss > 0 else float('inf')
    by_day = defaultdict(float)
    for x in trades:
        by_day[x['day']] += x['pnl']
    day_pnls = np.array(list(by_day.values()))
    day_wr = (day_pnls > 0).sum() / len(day_pnls) * 100
    return {
        'n_trades': len(trades),
        'n_days': len(day_pnls),
        'trades_per_day': len(trades) / max(len(day_pnls), 1),
        'net': float(pnls.sum()),
        'daily_mean': float(day_pnls.mean()),
        'trade_wr': trade_wr,
        'day_wr': day_wr,
        'mean_per_trade': float(pnls.mean()),
        'median_per_trade': float(np.median(pnls)),
    }


def main():
    is_files = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    is_days = [os.path.basename(f).replace('.parquet', '') for f in is_files]
    print(f'IS: {len(is_days)} days')

    # Sweep v2: wait + SL variations. sl_pts is in POINTS (÷2 for dollars).
    # SL$5 = 2.5 pt, SL$10 = 5.0 pt (default), SL$15 = 7.5 pt
    base = lambda w, sl: {
        'phantom_enabled': True,
        'phantom_wait_bars': w,
        'phantom_min_favorable_pts': 0.0,
        'sl_pts': sl,
    }
    configs = [
        ('C3 w=3 SL=$10 (prev winner)',  base(3, 5.0), 'mean_rev'),
        ('C8 w=3 SL=$5',                 base(3, 2.5), 'mean_rev'),
        ('C9 w=3 SL=$15',                base(3, 7.5), 'mean_rev'),
        ('C10 w=4 SL=$10',               base(4, 5.0), 'mean_rev'),
        ('C11 w=5 SL=$10',               base(5, 5.0), 'mean_rev'),
        ('C12 w=4 SL=$5',                base(4, 2.5), 'mean_rev'),
        ('C13 w=5 SL=$5',                base(5, 2.5), 'mean_rev'),
    ]

    rows = []
    for label, params, dir_mode in configs:
        print(f'\n=== {label} ({dir_mode}) ===')
        s = run_config(label, is_days, params, dir_mode)
        if s is None:
            print('  no trades')
            continue
        print(f'  trades={s["n_trades"]} ({s["trades_per_day"]:.0f}/d)  net=${s["net"]:+,.0f}  '
              f'$/day=${s["daily_mean"]:+.0f}  TradeWR={s["trade_wr"]:+.2f}  DayWR={s["day_wr"]:.0f}%')
        rows.append({'label': label, **s})

    # Write report
    out_lines = []
    out_lines.append('# Phantom-entry sweep')
    out_lines.append('')
    out_lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    out_lines.append('')
    out_lines.append('IS only (2025). Engine: TP$50/SL$10, 1s ticker with slippage.')
    out_lines.append('')
    out_lines.append('| Config | Trades | /day | $/day | TradeWR | DayWR | $/trade | Net |')
    out_lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        out_lines.append(
            f'| {r["label"]} | {r["n_trades"]} | {r["trades_per_day"]:.0f} '
            f'| ${r["daily_mean"]:+,.0f} | {r["trade_wr"]:+.2f} | {r["day_wr"]:.0f}% '
            f'| ${r["mean_per_trade"]:+.2f} | ${r["net"]:+,.0f} |'
        )
    out_lines.append('')

    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sweep_phantom.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()
