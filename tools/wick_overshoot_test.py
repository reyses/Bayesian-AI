"""
Wick Overshoot Test — kill shot entry + overshoot exit + breakeven regret.

After each trade, scans forward from entry to find:
  - Peak PnL and bars to peak
  - Bars until price returns to entry (breakeven = move's lifespan)
  - If never returns → permanent move (trend)

Usage:
    python tools/wick_overshoot_test.py                    # IS
    python tools/wick_overshoot_test.py --target oos       # OOS
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_wick_overshoot import WickOvershootEngine

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_5S = 'DATA/ATLAS/5s'
ATLAS_1M = 'DATA/ATLAS/1m'
TICK = 0.25
TV = 0.50


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Wick overshoot + breakeven regret')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'])
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def get_day_files(target='is'):
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    if target == 'is':
        files = [f for f in files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        files = [f for f in files if '2026_' in os.path.basename(f)]
    return files


def compute_breakeven_regret(trade, prices, timestamps):
    """Scan from entry forward: peak PnL, bars to peak, bars to breakeven.

    Uses 5s price data for fine granularity. No lookahead — this runs
    AFTER the trade is closed, as post-trade analysis only.
    """
    entry_price = trade['entry_price']
    direction = trade['dir']
    entry_ts = trade.get('timestamp', 0)

    # Find entry bar in price data
    entry_idx = int(np.searchsorted(timestamps, entry_ts, side='left'))
    entry_idx = min(entry_idx, len(prices) - 1)

    # Scan forward from entry
    peak_pnl = 0.0
    peak_bar = 0
    breakeven_bar = None

    for i in range(entry_idx, len(prices)):
        p = prices[i]
        if direction == 'long':
            pnl = (p - entry_price) / TICK * TV
        else:
            pnl = (entry_price - p) / TICK * TV

        if pnl > peak_pnl:
            peak_pnl = pnl
            peak_bar = i - entry_idx

        # Breakeven: after reaching peak, when does PnL return to 0?
        if peak_pnl > 2.0 and pnl <= 0 and breakeven_bar is None:
            breakeven_bar = i - entry_idx
            break

    return {
        'regret_peak_pnl': peak_pnl,
        'regret_peak_bar': peak_bar,
        'regret_breakeven_bar': breakeven_bar if breakeven_bar else -1,
        'regret_peak_time_min': peak_bar * 5 / 60,  # bars * 5s / 60
        'regret_lifespan_min': (breakeven_bar * 5 / 60) if breakeven_bar else -1,
    }


def main():
    args = parse_args()
    files = get_day_files(args.target)

    if not files:
        print(f'No files for target={args.target}')
        return

    print(f'WICK OVERSHOOT TEST + BREAKEVEN REGRET')
    print(f'  Target: {args.target.upper()} | Days: {len(files)}')
    print()

    engine = WickOvershootEngine()
    all_results = []
    all_trades_with_regret = []
    cumul = 0

    for fpath in tqdm(files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file_1m = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        price_file_5s = os.path.join(ATLAS_5S, f'{day_name}.parquet')

        # Use 5s prices for regret (finer), 1m for FeatureTicker
        if not os.path.exists(price_file_1m):
            price_file_1m = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file_1m)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        day_pnl = engine.daily_pnl
        day_n = len(engine.trades)
        cumul += day_pnl

        # Breakeven regret for each trade (using 5s prices)
        if os.path.exists(price_file_5s) and day_n > 0:
            pdf = pd.read_parquet(price_file_5s).sort_values('timestamp')
            prices_5s = pdf['close'].values
            ts_5s = pdf['timestamp'].values

            for t in engine.trades:
                regret = compute_breakeven_regret(t, prices_5s, ts_5s)
                t.update(regret)
                t['day'] = day_name
                all_trades_with_regret.append(t)

        all_results.append({
            'day': day_name, 'trades': day_n, 'pnl': day_pnl,
            'wr': sum(1 for t in engine.trades if t['pnl'] > 0) / max(day_n, 1) * 100,
        })

        if args.verbose and day_n > 0:
            tqdm.write(f'  {day_name}: {engine.summary()}  cumul=${cumul:.0f}')

    # Summary
    n_days = len(all_results)
    total_trades = len(all_trades_with_regret)
    total_pnl = sum(t['pnl'] for t in all_trades_with_regret)
    active_days = sum(1 for r in all_results if r['trades'] > 0)
    winning_days = sum(1 for r in all_results if r['pnl'] > 0)
    wins = sum(1 for t in all_trades_with_regret if t['pnl'] > 0)

    print(f'\n{"="*60}')
    print(f'WICK OVERSHOOT RESULTS — {args.target.upper()}')
    print(f'{"="*60}')
    print(f'  Days: {n_days} (active: {active_days})')
    print(f'  Winning days: {winning_days}/{active_days} ({winning_days/max(active_days,1)*100:.0f}%)')
    print(f'  Trades: {total_trades} | WR: {wins}/{total_trades} ({wins/max(total_trades,1)*100:.0f}%)')
    print(f'  Total PnL: ${total_pnl:,.0f}')
    print(f'  $/day (active): ${total_pnl/max(active_days,1):.0f}')
    print(f'  $/trade: ${total_pnl/max(total_trades,1):.1f}')

    # Exit breakdown
    if total_trades > 0:
        exits = Counter(t.get('exit', '?') for t in all_trades_with_regret)
        print(f'\n  Exit reasons:')
        for reason, count in exits.most_common():
            sub = [t for t in all_trades_with_regret if t.get('exit') == reason]
            wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            avg_pnl = np.mean([t['pnl'] for t in sub])
            print(f'    {reason:<25} {count:>4} ({wr:.0f}% WR)  avg ${avg_pnl:.1f}')

    # Breakeven regret analysis
    if total_trades > 0:
        peaks = [t['regret_peak_pnl'] for t in all_trades_with_regret]
        peak_bars = [t['regret_peak_bar'] for t in all_trades_with_regret]
        be_bars = [t['regret_breakeven_bar'] for t in all_trades_with_regret if t['regret_breakeven_bar'] > 0]
        no_be = sum(1 for t in all_trades_with_regret if t['regret_breakeven_bar'] < 0)

        print(f'\n  BREAKEVEN REGRET:')
        print(f'    Avg peak PnL (full extent): ${np.mean(peaks):.1f}')
        print(f'    Avg bars to peak: {np.mean(peak_bars):.0f} ({np.mean(peak_bars)*5/60:.1f} min)')
        print(f'    Actual avg PnL captured: ${total_pnl/max(total_trades,1):.1f}')
        print(f'    Capture rate: {total_pnl/max(sum(peaks),1)*100:.0f}%')
        print(f'    Trades that break even: {len(be_bars)} ({len(be_bars)/max(total_trades,1)*100:.0f}%)')
        print(f'    Trades that NEVER break even: {no_be} ({no_be/max(total_trades,1)*100:.0f}%) — permanent moves')
        if be_bars:
            print(f'    Avg bars to breakeven: {np.mean(be_bars):.0f} ({np.mean(be_bars)*5/60:.1f} min)')

        # Peak PnL distribution
        print(f'\n  PEAK PNL DISTRIBUTION (what the move could reach):')
        for thresh in [5, 10, 20, 30, 50, 75, 100]:
            n = sum(1 for p in peaks if p >= thresh)
            if n > 0:
                sub = [t for t in all_trades_with_regret if t['regret_peak_pnl'] >= thresh]
                avg_actual = np.mean([t['pnl'] for t in sub])
                avg_peak_bar = np.mean([t['regret_peak_bar'] for t in sub])
                print(f'    Peak >= ${thresh:>4}: {n:>4} trades | actual avg ${avg_actual:.1f} | peak at {avg_peak_bar:.0f} bars ({avg_peak_bar*5/60:.1f} min)')

    # Save
    os.makedirs('training/output/reports', exist_ok=True)
    csv_path = f'training/output/reports/wick_overshoot_{args.target}.csv'
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f'\nCSV: {csv_path}')

    if all_trades_with_regret:
        trade_path = f'training/output/reports/wick_overshoot_{args.target}_trades.csv'
        flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
                for t in all_trades_with_regret]
        pd.DataFrame(flat).to_csv(trade_path, index=False)
        print(f'Trades: {trade_path}')


if __name__ == '__main__':
    main()
