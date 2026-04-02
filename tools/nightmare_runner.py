"""
Nightmare Runner — lightweight wrapper that runs nightmare_ticker per day.

Loads shared data once, loops over days, collects results.

Usage:
  python tools/nightmare_runner.py                     # all OOS days
  python tools/nightmare_runner.py 2026-03-20          # single day
  python tools/nightmare_runner.py 2026-03-20,2026-03-24  # specific days
"""
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'all'

def get_available_days():
    """Get all trading days from 1s data."""
    import glob
    files = sorted(glob.glob('DATA/ATLAS/1s/*.parquet'))
    all_ts = []
    for f in files:
        df = pd.read_parquet(f, columns=['timestamp'])
        all_ts.extend(df['timestamp'].values[::5000])
    dates = sorted(set(datetime.utcfromtimestamp(t).strftime('%Y-%m-%d') for t in all_ts))
    return [d for d in dates if d >= '2026-03-02']


def run_single_day(date_str):
    """Run nightmare_ticker.py for a single day, capture output."""
    result = subprocess.run(
        [sys.executable, 'tools/nightmare_ticker.py', date_str],
        capture_output=True, text=True, timeout=600,
        cwd=os.getcwd()
    )
    return result.stdout, result.stderr


def parse_trades_csv(date_str):
    """Read the trades CSV output from a single-day run."""
    path = f'reports/findings/nightmare_{date_str}_trades.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    if TARGET == 'all':
        days = get_available_days()
    elif ',' in TARGET:
        days = TARGET.split(',')
    else:
        days = [TARGET]

    print(f'\nNIGHTMARE RUNNER — {len(days)} day(s)')
    print(f'  Running nightmare_ticker.py per day as subprocess')
    print()

    all_trades = []
    daily_summary = []

    for i, day in enumerate(days):
        print(f'  [{i+1}/{len(days)}] {day}...', end=' ', flush=True)
        stdout, stderr = run_single_day(day)

        # Parse the result line from stdout
        pnl_line = [l for l in stdout.split('\n') if 'PnL=$' in l and 'trades' in l]
        if pnl_line:
            print(pnl_line[0].strip())
        else:
            print('NO OUTPUT')
            if stderr:
                err_lines = [l for l in stderr.split('\n') if 'Error' in l or 'Traceback' in l]
                for l in err_lines[:3]:
                    print(f'    {l}')
            continue

        # Read trades
        trades = parse_trades_csv(day)
        if len(trades) > 0:
            trades['day'] = day
            all_trades.append(trades)
            day_pnl = trades['pnl'].sum()
            day_wins = (trades['pnl'] > 0).sum()
            daily_summary.append({
                'day': day, 'trades': len(trades),
                'pnl': day_pnl, 'wins': day_wins
            })

    if not all_trades:
        print('\nNo trades collected.')
        return

    # Aggregate
    t = pd.concat(all_trades, ignore_index=True)
    total = t['pnl'].sum()
    n_days = len(daily_summary)
    wr = (t['pnl'] > 0).mean() * 100

    print(f'\n{"="*60}')
    print(f'FULL OOS — {n_days} days — ZERO LOOKAHEAD')
    print(f'{"="*60}')
    print(f'  {len(t)} trades | WR={wr:.1f}% | PnL=${total:,.2f} | $/day=${total/n_days:,.2f}')
    print()

    for ex in sorted(t['exit'].unique()):
        et = t[t['exit'] == ex]
        print(f'  {ex:<22} {len(et):>5}  WR={(et["pnl"]>0).mean()*100:>5.1f}%  ${et["pnl"].sum():>10,.2f}  ${et["pnl"].mean():>7,.2f}/tr')

    print()
    print('DAILY:')
    cumul = 0
    for d in daily_summary:
        cumul += d['pnl']
        wr_d = d['wins'] / d['trades'] * 100 if d['trades'] else 0
        marker = ' <<<' if d['pnl'] > 200 else ' !!!' if d['pnl'] < -200 else ''
        print(f'  {d["day"]} {d["trades"]:>5} {wr_d:>4.0f}% ${d["pnl"]:>9,.2f} ${cumul:>9,.2f}{marker}')

    winning = sum(1 for d in daily_summary if d['pnl'] > 0)
    print(f'\n  Winning days: {winning}/{n_days}')

    # Save aggregate
    os.makedirs('reports/findings', exist_ok=True)
    t.to_csv('reports/findings/nightmare_full_oos_trades.csv', index=False)
    print(f'\n  Saved: reports/findings/nightmare_full_oos_trades.csv')


if __name__ == '__main__':
    main()
