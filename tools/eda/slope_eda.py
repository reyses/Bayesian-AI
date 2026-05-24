"""
Regression-mean kinematics EDA — multi-TF β and γ via 30-sample OLS.

Physics framing:
  price(t) = mean(t) + noise(t)
  mean(t)  = α + β·t + γ·t²

where α, β, γ come from OLS on N=30 closes at the chosen TF.

  β = first derivative  = velocity of the regression mean = trend drift
  γ = second derivative = acceleration of the regression mean

γ distinguishes exhaustion from continuation:
  γ opposing β  → trend DECELERATING (exhaustion — fade candidate)
  γ aligned β   → trend ACCELERATING (continuation — ride candidate)

We measure this at each TF's native cadence:
  1m   (30 bars = 30 min  of data)
  5m   (30 bars = 2.5 hr  of data)
  15m  (30 bars = 7.5 hr  of data)
  1h   (30 bars = 30 hr   of data)

Per trade we compute 4 TFs × (β, γ, |β|, |γ|, aligned_β, aligned_γ) =
24 metrics. Aligned = metric × (+1 if long, -1 if short). Aligned > 0
means slope/accel is WITH our trade direction.

Usage:
    python tools/slope_eda.py --tier TREND_FOLLOWER
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BARS_N = 30
TF_PARQUET_DIRS = {
    '1m':  'DATA/ATLAS/1m',
    '5m':  'DATA/ATLAS/5m',
    '15m': 'DATA/ATLAS/15m',
    '1h':  'DATA/ATLAS/1h',
}


def fit_linear_quadratic(y: np.ndarray):
    """OLS fit y = α + β·x + γ·x² on bar index x. Return (β, γ).

    β in price/bar, γ in price/bar². Uses numpy.polyfit (stable).
    """
    if len(y) < 5:
        return 0.0, 0.0
    x = np.arange(len(y), dtype=np.float64)
    try:
        coeffs = np.polyfit(x, y, 2)   # returns [γ, β, α]
        gamma, beta, _ = coeffs
        return float(beta), float(gamma)
    except np.linalg.LinAlgError:
        return 0.0, 0.0


class TFLoader:
    """Per-day cache of TF parquets."""
    def __init__(self, day: str):
        self.data = {}
        for tf, dir_ in TF_PARQUET_DIRS.items():
            path = os.path.join(dir_, f'{day}.parquet')
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path).sort_values('timestamp')
            self.data[tf] = (df['timestamp'].values.astype(np.int64),
                             df['close'].values.astype(np.float64))

    def slopes_at(self, entry_ts: int) -> dict:
        """For each TF, return (β, γ) from N=30 closes BEFORE entry_ts.

        Strict <: exclude the bar containing entry to avoid lookahead.
        Per-TF: skip if that TF lacks history (but still return other TFs).
        """
        out = {}
        for tf, (ts_arr, close_arr) in self.data.items():
            idx = np.searchsorted(ts_arr, entry_ts, side='left') - 1
            if idx < BARS_N - 1:
                continue
            window = close_arr[idx - BARS_N + 1: idx + 1]
            beta, gamma = fit_linear_quadratic(window)
            out[f'beta_{tf}']      = beta
            out[f'gamma_{tf}']     = gamma
            out[f'abs_beta_{tf}']  = abs(beta)
            out[f'abs_gamma_{tf}'] = abs(gamma)
        return out


def attach_slopes(trades):
    by_day = defaultdict(list)
    for t in trades:
        day = t.get('day')
        if day:
            by_day[day].append(t)

    n_attached = 0
    for day, day_trades in by_day.items():
        loader = TFLoader(day)
        if not loader.data:
            continue
        for t in day_trades:
            entry_ts = int(t.get('timestamp', 0))
            slopes = loader.slopes_at(entry_ts)
            if not slopes:
                continue
            dir_sign = 1 if t['dir'] == 'long' else -1
            for tf in TF_PARQUET_DIRS:
                beta = slopes.get(f'beta_{tf}', 0.0)
                gamma = slopes.get(f'gamma_{tf}', 0.0)
                slopes[f'aligned_beta_{tf}']  = beta * dir_sign
                slopes[f'aligned_gamma_{tf}'] = gamma * dir_sign
                # Exhaustion indicator: γ opposing β (trend decelerating).
                # β * γ < 0 = decelerating (exhaustion). > 0 = accelerating.
                slopes[f'decel_{tf}'] = -1.0 * (beta * gamma)
            t['_slopes'] = slopes
            n_attached += 1
    return n_attached


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2) + 1e-9
    return float((a.mean() - b.mean()) / pooled)


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

    n = attach_slopes(sub)
    print(f'Attached 30-sample multi-TF slopes to {n} trades')
    sub = [t for t in sub if '_slopes' in t]

    buckets = defaultdict(list)
    for t in sub:
        buckets[segment(t)].append(t)

    print()
    print(f'Segment counts:')
    for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
        print(f'  {seg:<13}: {len(buckets[seg])}')

    winners = buckets.get('winner', [])
    tails = buckets.get('tail_loser', [])
    if not winners or not tails:
        print('Not enough trades.')
        return

    # Assemble metric names
    metric_names = []
    for tf in TF_PARQUET_DIRS:
        for prefix in ('beta', 'gamma', 'abs_beta', 'abs_gamma',
                       'aligned_beta', 'aligned_gamma', 'decel'):
            metric_names.append(f'{prefix}_{tf}' if prefix != 'decel' else f'decel_{tf}')

    print()
    print(f'30-sample multi-TF slope comparison: winner (n={len(winners)}) vs '
          f'tail_loser (n={len(tails)})')
    print(f'{"metric":<22} {"win_med":>14} {"tail_med":>14} {"Cohen d":>10}')
    print('-' * 70)

    rows = []
    for m in metric_names:
        w = np.array([t['_slopes'][m] for t in winners if m in t['_slopes']])
        tt = np.array([t['_slopes'][m] for t in tails if m in t['_slopes']])
        if len(w) == 0 or len(tt) == 0:
            continue
        d = cohen_d(w, tt)
        rows.append((m, float(np.median(w)), float(np.median(tt)), d))
    rows.sort(key=lambda r: -abs(r[3]))

    for m, wm, tm, d in rows:
        flag = '  *' if abs(d) > 0.3 else ''
        print(f'{m:<22} {wm:>14.6f} {tm:>14.6f} {d:>+10.3f}{flag}')

    # Breakdown of top metric
    if rows:
        top = rows[0]
        print()
        print(f'Distribution of top metric ({top[0]}, d={top[3]:+.3f}) by segment:')
        print(f'  {"segment":<13} {"n":>5} {"p10":>10} {"p25":>10} {"p50":>10} '
              f'{"p75":>10} {"p90":>10}')
        for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
            bucket = buckets.get(seg, [])
            if not bucket:
                continue
            arr = np.array([t['_slopes'][top[0]] for t in bucket
                            if top[0] in t['_slopes']])
            print(f'  {seg:<13} {len(arr):>5} {np.percentile(arr, 10):>10.5f} '
                  f'{np.percentile(arr, 25):>10.5f} {np.percentile(arr, 50):>10.5f} '
                  f'{np.percentile(arr, 75):>10.5f} {np.percentile(arr, 90):>10.5f}')

    print()
    print('Interpretation:')
    print('  abs_beta_X      = trend power at TF X (direction-free)')
    print('  aligned_beta_X  = trend power * our direction (positive = with us)')
    print('  aligned_gamma_X = trend acceleration * our direction')
    print('  decel_X         = +1 signals decelerating trend (exhaustion)')
    print('  Separators |d|>0.3 = real signal for engine integration')


if __name__ == '__main__':
    main()
