"""
recent_volatility_regime.py -- characterize last-N-days market volatility
                                 vs full ATLAS history.

Use case: SL calibration. If recent days are wider/narrower than the historical
mean, the SL distance that worked in backtest may be miscalibrated for current
conditions. Compute key MNQ daily-volatility statistics and bootstrap CIs.

Metrics per day (from 1m parquet):
  session_range_pts   = high_of_day - low_of_day
  p90_bar_range_pts   = 90th percentile of (high-low) across the day's 1m bars
  mean_bar_range_pts  = mean 1m bar range
  realized_vol_pts    = std of 1m close-to-close returns
  max_1m_move_pts     = largest absolute 1m return
  max_drawdown_pts    = largest peak-to-current drop within day

Last 20 days = approximately 4 weeks of recent data.

Usage:
    python tools/recent_volatility_regime.py
    python tools/recent_volatility_regime.py --window 30
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def boot_ci(arr, n_boot=2000):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, len(a), len(a))
        boots[i] = a[idx].mean()
    return float(a.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', default='DATA/ATLAS')
    ap.add_argument('--window', type=int, default=20, help='Recent days window')
    ap.add_argument('--out', default='reports/findings/daily_volatility_regime.csv')
    args = ap.parse_args()

    np.random.seed(42)

    files = sorted(glob.glob(os.path.join(args.atlas, '1m', '*.parquet')))
    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f)
        if len(df) < 2:
            continue
        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        rets   = np.diff(closes)
        bar_ranges = highs - lows
        rows.append({
            'day': day,
            'n_1m_bars': len(df),
            'session_range_pts': float(highs.max() - lows.min()),
            'p90_bar_range_pts': float(np.quantile(bar_ranges, 0.90)),
            'mean_bar_range_pts': float(bar_ranges.mean()),
            'median_bar_range_pts': float(np.median(bar_ranges)),
            'realized_vol_pts': float(np.std(rets)),
            'max_1m_move_pts': float(np.abs(rets).max()) if len(rets) else 0.0,
            'max_drawdown_pts': float(np.max(np.maximum.accumulate(closes) - closes)),
        })
    agg = pd.DataFrame(rows)
    agg['ts']  = agg['day'].apply(lambda d: pd.Timestamp(d.replace('_', '-')).timestamp())
    agg['dow'] = agg['day'].apply(lambda d: pd.Timestamp(d.replace('_', '-')).day_name())
    agg = agg.sort_values('ts').reset_index(drop=True)

    last  = agg.tail(args.window).copy()
    prior = agg.iloc[:-args.window].copy()

    print('=' * 102)
    print(f'  RECENT VOLATILITY REGIME  '
          f'(last {args.window} days vs prior {len(prior)} days)')
    print(f'  Latest day  : {agg.iloc[-1]["day"]}')
    print(f'  Window start: {last.iloc[0]["day"]}')
    print('=' * 102)

    metrics = ['session_range_pts', 'p90_bar_range_pts', 'mean_bar_range_pts',
               'realized_vol_pts', 'max_1m_move_pts', 'max_drawdown_pts']
    print(f'{"Metric":<22} {"Last "+str(args.window)+" mean[95%CI]":>30} '
          f'{"Prior mean[95%CI]":>28} {"delta_pct":>10}')
    print('-' * 102)
    for m in metrics:
        l_m, l_lo, l_hi = boot_ci(last[m].values)
        p_m, p_lo, p_hi = boot_ci(prior[m].values)
        pct = (l_m / p_m - 1) * 100 if p_m else 0
        print(f'{m:<22} '
              f'{l_m:>+8.2f} [{l_lo:>+6.2f},{l_hi:>+6.2f}]  '
              f'{p_m:>+8.2f} [{p_lo:>+6.2f},{p_hi:>+6.2f}]  '
              f'{pct:>+8.1f}%')

    print(f'\nRECENT {args.window} ACTIVE DAYS DETAIL')
    print(f'{"day":<14} {"session_rng":>11} {"mean_bar":>10} {"vol_1m":>9} '
          f'{"max_1m_move":>12} {"max_DD":>9} {"DOW":>10}')
    for _, r in last.iterrows():
        print(f'{r["day"]:<14} '
              f'{r["session_range_pts"]:>+10.1f} '
              f'{r["mean_bar_range_pts"]:>+9.2f} '
              f'{r["realized_vol_pts"]:>+8.2f} '
              f'{r["max_1m_move_pts"]:>+11.2f} '
              f'{r["max_drawdown_pts"]:>+8.1f}  '
              f'{r["dow"]:>10}')

    # SL calibration heuristic
    p90 = last['p90_bar_range_pts'].mean()
    mx  = last['max_1m_move_pts'].mean()
    print(f'\nSL CALIBRATION (using last {args.window} days)')
    print(f'  Mean 1m bar range          : {last["mean_bar_range_pts"].mean():>5.2f} pts')
    print(f'  p90 1m bar range           : {p90:>5.2f} pts')
    print(f'  Mean max-1m-move           : {mx:>5.2f} pts (largest single-bar shock)')
    print(f'  Mean realized vol (1m)     : {last["realized_vol_pts"].mean():>5.2f} pts')
    print()
    print(f'  SL = p90 bar range          : {p90:>5.1f} pts (cuts at typical noise — too tight)')
    print(f'  SL = 1.5x p90 bar range     : {p90*1.5:>5.1f} pts (recommended floor)')
    print(f'  SL = 2.0x p90 bar range     : {p90*2.0:>5.1f} pts (loose, lets bars develop)')
    print(f'  SL = max-1m-move            : {mx:>5.1f} pts (catastrophic-only backstop)')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
