"""
Path-feature EDA — measure 5s intra-minute path metrics at entry.

For each trade in a tier, look at the 12 5s bars immediately preceding entry
(= the 1m window ending at entry bar). Compute path-derived metrics:

  path_efficiency    |close - open| / sum(|delta_i|)       0=chop, 1=pure trend
  reversal_count     # sign flips in delta_i               low=trend, high=chop
  longest_streak     biggest run of same-direction bars    big=persistent force
  autocorr_lag1      corr(delta_i, delta_{i-1})            >0=trending, <0=reverting
  r_squared          R^2 of closes fit to a line           high=structured trend
  jump_ratio         max|delta_i| / sum(|delta_i|)         high=spike, low=sustained
  up_down_ratio      count(delta>0) / count(delta<0)       direction bias

Then Cohen d between winners and tail_losers for each. If any metric separates
them with |d| > 0.3, it's a real signal worth adding to the engine.

Usage:
    python tools/path_features_eda.py --tier TREND_FOLLOWER
    python tools/path_features_eda.py --tier NMP_FADE
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_5S = 'DATA/ATLAS/5s'
BARS_PER_MINUTE = 12   # 5s bars in a 1m window


def compute_path_features(closes: np.ndarray) -> dict:
    """Compute path metrics on an array of 5s closes (12 points = 1m window)."""
    if len(closes) < 3:
        return {}
    deltas = np.diff(closes)
    abs_deltas = np.abs(deltas)
    abs_sum = abs_deltas.sum()
    net = closes[-1] - closes[0]

    # Path efficiency
    efficiency = abs(net) / (abs_sum + 1e-9)

    # Reversal count
    signs = np.sign(deltas)
    sign_changes = np.sum(signs[1:] != signs[:-1]) if len(signs) > 1 else 0

    # Longest same-direction streak
    if len(signs) == 0:
        streak = 0
    else:
        streak = 1
        max_streak = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1] and signs[i] != 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        streak = max_streak

    # Lag-1 autocorrelation of deltas
    if len(deltas) >= 2 and deltas.std() > 0:
        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1])
    else:
        autocorr = 0.0
    if np.isnan(autocorr):
        autocorr = 0.0

    # R² of closes vs time (linear fit)
    t = np.arange(len(closes), dtype=np.float64)
    if closes.std() > 0:
        corr = np.corrcoef(t, closes)[0, 1]
        r2 = float(corr ** 2) if not np.isnan(corr) else 0.0
    else:
        r2 = 0.0

    # Jump ratio
    jump = abs_deltas.max() / (abs_sum + 1e-9)

    # Up/down tick ratio
    up = int((deltas > 0).sum())
    dn = int((deltas < 0).sum())
    udr = up / max(dn, 1)

    return {
        'path_efficiency': float(efficiency),
        'reversal_count': int(sign_changes),
        'longest_streak': int(streak),
        'autocorr_lag1': autocorr,
        'r_squared': r2,
        'jump_ratio': float(jump),
        'up_down_ratio': float(udr),
        'net_move': float(net),
        'abs_range': float(abs_sum),
    }


def attach_path_features(trades, atlas_5s_dir=ATLAS_5S):
    """For each trade, load the 5s parquet for its day, find the 12 bars
    ending at entry timestamp, compute path features."""
    # Group trades by day to load each parquet once
    by_day = defaultdict(list)
    for t in trades:
        day = t.get('day')
        if day:
            by_day[day].append(t)

    n_attached = 0
    for day, day_trades in by_day.items():
        path = os.path.join(atlas_5s_dir, f'{day}.parquet')
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values('timestamp')
        ts_arr = df['timestamp'].values.astype(np.int64)
        close_arr = df['close'].values.astype(np.float64)

        for t in day_trades:
            entry_ts = int(t.get('timestamp', 0))
            # Find the index at/just-before entry
            idx = np.searchsorted(ts_arr, entry_ts, side='right') - 1
            if idx < BARS_PER_MINUTE - 1:
                continue  # not enough history
            window = close_arr[idx - BARS_PER_MINUTE + 1: idx + 1]
            if len(window) != BARS_PER_MINUTE:
                continue
            feats = compute_path_features(window)
            t['_path_features'] = feats
            n_attached += 1
    return n_attached


def cohen_d(arr_a, arr_b):
    if len(arr_a) < 2 or len(arr_b) < 2:
        return 0.0
    pooled = np.sqrt((arr_a.std() ** 2 + arr_b.std() ** 2) / 2) + 1e-9
    return (arr_a.mean() - arr_b.mean()) / pooled


def segment(t):
    p = t['pnl']
    if p >= 5:    return 'winner'
    if p < -15:   return 'tail_loser'
    if p <= -5:   return 'mid_loser'
    return 'small_loser'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--trades', default='training_iso/output/trades/iso_is.pkl')
    args = ap.parse_args()

    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    sub = [t for t in trades if t.get('entry_tier') == args.tier]
    print(f'Loaded {len(sub)} {args.tier} trades')

    n = attach_path_features(sub)
    print(f'Attached path features to {n} trades')
    sub = [t for t in sub if '_path_features' in t]

    # Segment
    buckets = defaultdict(list)
    for t in sub:
        buckets[segment(t)].append(t)

    metric_names = ['path_efficiency', 'reversal_count', 'longest_streak',
                    'autocorr_lag1', 'r_squared', 'jump_ratio', 'up_down_ratio',
                    'net_move', 'abs_range']

    print()
    print(f'Segment counts:')
    for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
        print(f'  {seg:<13}: {len(buckets[seg])}')

    winners = buckets.get('winner', [])
    tails = buckets.get('tail_loser', [])
    if not winners or not tails:
        print('Not enough trades in winner or tail segment.')
        return

    print()
    print(f'Path metric comparison: winner (n={len(winners)}) vs tail_loser (n={len(tails)})')
    print(f'{"metric":<20} {"win_med":>10} {"tail_med":>10} {"win_mean":>10} '
          f'{"tail_mean":>10} {"Cohen d":>10}')
    print('-' * 85)
    rows = []
    for m in metric_names:
        w_arr = np.array([t['_path_features'][m] for t in winners])
        t_arr = np.array([t['_path_features'][m] for t in tails])
        d = cohen_d(w_arr, t_arr)
        rows.append((m, w_arr, t_arr, d))
    rows.sort(key=lambda r: -abs(r[3]))
    for m, w, t, d in rows:
        flag = '  *' if abs(d) > 0.3 else ''
        print(f'{m:<20} {np.median(w):>10.4f} {np.median(t):>10.4f} '
              f'{w.mean():>10.4f} {t.mean():>10.4f} {d:>+10.3f}{flag}')

    print()
    print('Metrics with |d| > 0.3 are real separators worth considering as engine filters.')

    # Also: bucket analysis for top metric
    top_metric, top_w, top_t, top_d = rows[0]
    print()
    print(f'Distribution of top metric ({top_metric}) by segment:')
    print(f'  {"segment":<13} {"n":>5} {"p25":>9} {"p50":>9} {"p75":>9} {"p90":>9} {"p95":>9}')
    for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
        if not buckets.get(seg):
            continue
        arr = np.array([t['_path_features'][top_metric] for t in buckets[seg]])
        print(f'  {seg:<13} {len(arr):>5} {np.percentile(arr, 25):>9.4f} '
              f'{np.percentile(arr, 50):>9.4f} {np.percentile(arr, 75):>9.4f} '
              f'{np.percentile(arr, 90):>9.4f} {np.percentile(arr, 95):>9.4f}')


if __name__ == '__main__':
    main()
