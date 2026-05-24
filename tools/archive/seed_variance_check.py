"""
Seed variance check — how much do results move just by changing the
slippage RNG seed?

Runs C3 (phantom w=3, SL=$10) on IS with seeds 1..5.
If the spread is wide, SL/wait sweeps might be in the noise band.

Usage:
    python tools/seed_variance_check.py
"""
import os
import sys
import glob
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_RM_physics.ticker_1s import OneSecondTicker
from training_RM_physics.rm_physics_engine import RMPhysicsEngine

ATLAS_1M_DIR = 'DATA/ATLAS/1m'


def run_once(days, seed, params):
    all_trades = []
    for day in tqdm(days, desc=f'seed={seed}', unit='day', leave=False):
        eng = RMPhysicsEngine(slippage_seed=seed, **params)
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
    return all_trades


def summarize(trades):
    if not trades:
        return None
    pnls = np.array([x['pnl'] for x in trades])
    by_day = defaultdict(float)
    for x in trades:
        by_day[x['day']] += x['pnl']
    day_pnls = np.array(list(by_day.values()))
    return {
        'n_trades': len(trades),
        'net': float(pnls.sum()),
        'daily_mean': float(day_pnls.mean()),
        'trade_wr': (float(pnls[pnls > 0].sum()) / max(float(abs(pnls[pnls < 0].sum())), 1e-9)) - 1,
        'day_wr': (day_pnls > 0).sum() / len(day_pnls) * 100,
    }


def main():
    files = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    print(f'IS days: {len(days)}')

    C3_PARAMS = {
        'phantom_enabled': True,
        'phantom_wait_bars': 3,
        'phantom_min_favorable_pts': 0.0,
        'sl_pts': 5.0,        # $10
        'tp_pts': 25.0,       # $50
    }

    results = []
    for seed in [1, 2, 3, 4, 5]:
        print(f'\n=== seed={seed} ===')
        trades = run_once(days, seed, C3_PARAMS)
        s = summarize(trades)
        if s:
            print(f'  net=${s["net"]:+,.0f}  $/day=${s["daily_mean"]:+.0f}  '
                  f'TradeWR={s["trade_wr"]:+.2f}  DayWR={s["day_wr"]:.0f}%')
            s['seed'] = seed
            results.append(s)

    if not results:
        print('No data.')
        return

    nets = np.array([r['net'] for r in results])
    daily = np.array([r['daily_mean'] for r in results])
    trade_wrs = np.array([r['trade_wr'] for r in results])
    print('\n--- seed spread ---')
    print(f'  net:     min ${nets.min():+,.0f}  max ${nets.max():+,.0f}  '
          f'range ${nets.max()-nets.min():,.0f}  std ${nets.std():,.0f}')
    print(f'  $/day:   min ${daily.min():+.0f}  max ${daily.max():+.0f}  '
          f'range ${daily.max()-daily.min():.0f}  std ${daily.std():.0f}')
    print(f'  tradeWR: min {trade_wrs.min():+.3f}  max {trade_wrs.max():+.3f}  '
          f'range {trade_wrs.max()-trade_wrs.min():.3f}')

    # Write report
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'seed_variance.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Seed-variance check (C3: phantom w=3, SL=$10, 1s+slippage)\n\n')
        f.write(f'Generated: {datetime.now().isoformat(timespec="seconds")}\n\n')
        f.write('| seed | trades | net | $/day | TradeWR | DayWR |\n')
        f.write('|---:|---:|---:|---:|---:|---:|\n')
        for r in results:
            f.write(f'| {r["seed"]} | {r["n_trades"]} | ${r["net"]:+,.0f} '
                    f'| ${r["daily_mean"]:+.0f} | {r["trade_wr"]:+.2f} '
                    f'| {r["day_wr"]:.0f}% |\n')
        f.write('\n')
        f.write(f'**Range of $/day across seeds: ${daily.max()-daily.min():.0f}**  \n')
        f.write(f'**Std ${daily.std():.0f}**\n\n')
        f.write('If differences in the SL/wait sweep are within this range, they are noise.\n')
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()
