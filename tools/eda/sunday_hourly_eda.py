"""
Sunday Hourly EDA — which hours are profitable on Sundays?

Runs forward pass on Sunday-only days, buckets trades by entry hour.
Output: reports/findings/sunday_hourly_eda.txt

Usage:
    python tools/sunday_hourly_eda.py
"""
import os
import sys
import glob
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'


def is_sunday(day_name: str) -> bool:
    try:
        return datetime.strptime(day_name, '%Y_%m_%d').weekday() == 6
    except ValueError:
        return False


def main():
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    sunday_files = [f for f in feat_files if is_sunday(
        os.path.basename(f).replace('.parquet', ''))]

    print(f'Total days: {len(feat_files)}')
    print(f'Sunday days: {len(sunday_files)}')

    engine = BlendedEngine(use_cnn=False)
    all_trades = []

    for fpath in tqdm(sunday_files, desc='Sundays', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            t['day'] = day_name
        all_trades.extend(engine.get_full_trades())

    if not all_trades:
        print('No Sunday trades found.')
        return

    # Bucket by entry hour (UTC)
    hourly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pnls': []})

    for t in all_trades:
        entry_ts = t.get('entry_ts', 0)
        if entry_ts == 0:
            continue
        hour = datetime.utcfromtimestamp(entry_ts).hour
        h = hourly[hour]
        h['trades'] += 1
        h['pnl'] += t['pnl']
        h['pnls'].append(t['pnl'])
        if t['pnl'] > 0:
            h['wins'] += 1

    # Report
    lines = []
    lines.append('=' * 65)
    lines.append('SUNDAY HOURLY EDA')
    lines.append(f'Days: {len(sunday_files)} | Trades: {len(all_trades)}')
    lines.append('=' * 65)
    lines.append('')
    lines.append(f'{"Hour":>6} {"Trades":>7} {"WR":>5} {"$/tr":>8} {"$/hr":>8} {"Total":>10}')
    lines.append('-' * 50)

    total_pnl = 0
    for hour in sorted(hourly.keys()):
        h = hourly[hour]
        wr = h['wins'] / h['trades'] * 100 if h['trades'] else 0
        avg = h['pnl'] / h['trades'] if h['trades'] else 0
        per_day = h['pnl'] / len(sunday_files)
        total_pnl += h['pnl']
        marker = ' ***' if h['pnl'] < 0 and h['trades'] >= 5 else ''
        lines.append(f'{hour:>4}:00 {h["trades"]:>7} {wr:>4.0f}% {avg:>+8.1f} '
                      f'{per_day:>+8.1f} {h["pnl"]:>+10.1f}{marker}')

    lines.append('-' * 50)
    lines.append(f'{"TOTAL":>6} {len(all_trades):>7} '
                 f'{sum(1 for t in all_trades if t["pnl"]>0)/len(all_trades)*100:>4.0f}% '
                 f'{total_pnl/len(all_trades):>+8.1f} '
                 f'{total_pnl/len(sunday_files):>+8.1f} {total_pnl:>+10.1f}')
    lines.append('')

    # Losing hours summary
    losing_hours = [(h, hourly[h]) for h in sorted(hourly.keys())
                    if hourly[h]['pnl'] < 0 and hourly[h]['trades'] >= 5]
    if losing_hours:
        lines.append('LOSING HOURS (>5 trades):')
        loss_total = sum(d['pnl'] for _, d in losing_hours)
        for hour, h in losing_hours:
            lines.append(f'  {hour:>2}:00 — {h["trades"]} trades, '
                         f'WR={h["wins"]/h["trades"]*100:.0f}%, '
                         f'total=${h["pnl"]:+.0f}')
        lines.append(f'  Skip these hours → save ${abs(loss_total):.0f}')
    else:
        lines.append('No consistently losing hours found.')

    lines.append('')
    lines.append('=' * 65)

    report = '\n'.join(lines)
    print(report)

    # Save
    os.makedirs('reports/findings', exist_ok=True)
    out_path = 'reports/findings/sunday_hourly_eda.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
