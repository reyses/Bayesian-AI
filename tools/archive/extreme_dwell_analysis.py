"""How often and how long is price in 3+ sigma extreme territory?

For each anchor (M_high, M_close, M_low) at each TF (5m, 15m, 1h),
walks IS data at 5s cadence and answers:

    1. What fraction of bars sit in |σ-position| >= 3 territory?
    2. When in 3+σ territory, how long does the run last before
       price returns inside the band?
    3. Distribution of run lengths (median, q90, max) — tells us if
       the extreme window is long enough to actually trade.
    4. Cumulative bar-count and per-day frequency.

USAGE:
    python tools/extreme_dwell_analysis.py
    python tools/extreme_dwell_analysis.py --tf 1h --sigma-threshold 2
    python tools/extreme_dwell_analysis.py --n-days 50
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TF_CONFIG = {
    '1m':  ('DATA/ATLAS/1m',  15),
    '5m':  ('DATA/ATLAS/5m',   9),
    '15m': ('DATA/ATLAS/15m', 12),
    '1h':  ('DATA/ATLAS/1h',  12),
}


def _load_ohlcv(tf: str, day: str) -> pd.DataFrame:
    base, _ = TF_CONFIG[tf]
    path = os.path.join(base, f'{day}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _per_day_runs(day: str, tf: str, atlas_5s: str,
                          sigma_thresh: float) -> dict:
    base, N = TF_CONFIG[tf]
    period_s = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}[tf]

    ohlcv_5s = pd.read_parquet(os.path.join(atlas_5s, f'{day}.parquet'))
    if pd.api.types.is_datetime64_any_dtype(ohlcv_5s['timestamp']):
        ohlcv_5s = ohlcv_5s.copy()
        ohlcv_5s['timestamp'] = (ohlcv_5s['timestamp'].astype('int64') // 10**9)
    ohlcv_5s = ohlcv_5s.sort_values('timestamp').reset_index(drop=True)
    if ohlcv_5s.empty:
        return {}

    tf_oh = _load_ohlcv(tf, day)
    if tf_oh.empty:
        return {}

    tf_oh['close_mean']  = tf_oh['close'].rolling(N, min_periods=2).mean()
    tf_oh['close_sigma'] = tf_oh['close'].rolling(N, min_periods=2).std()
    tf_oh['high_mean']   = tf_oh['high'].rolling(N, min_periods=2).mean()
    tf_oh['high_sigma']  = tf_oh['high'].rolling(N, min_periods=2).std()
    tf_oh['low_mean']    = tf_oh['low'].rolling(N, min_periods=2).mean()
    tf_oh['low_sigma']   = tf_oh['low'].rolling(N, min_periods=2).std()

    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)
    tf_ts = tf_oh['timestamp'].values.astype(np.int64)
    target = oh_ts - period_s
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)

    p = ohlcv_5s['close'].values
    Mc = tf_oh['close_mean'].values[idx]
    Sc = tf_oh['close_sigma'].values[idx]
    Mh = tf_oh['high_mean'].values[idx]
    Sh = tf_oh['high_sigma'].values[idx]
    Ml = tf_oh['low_mean'].values[idx]
    Sl = tf_oh['low_sigma'].values[idx]

    d_close = (p - Mc) / Sc
    d_high  = (p - Mh) / Sh
    d_low   = (p - Ml) / Sl

    out = {'n_bars': len(p)}

    # Per anchor + side, run-length analysis
    for label, d in [('close+', d_close), ('close-', -d_close),
                              ('high+',  d_high),  ('high-',  -d_high),
                              ('low+',   d_low),   ('low-',   -d_low)]:
        # We use SIGNED distance — "+ side" means above (>= +threshold)
        flag = (d >= sigma_thresh) & np.isfinite(d)
        if flag.sum() == 0:
            out[label] = {'n_extreme': 0, 'frac': 0.0,
                                'run_lengths': [], 'n_events': 0}
            continue
        # Run-length encoding of consecutive True
        runs = []
        cur = 0
        for v in flag:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                    cur = 0
        if cur > 0:
            runs.append(cur)
        out[label] = {
            'n_extreme': int(flag.sum()),
            'frac': float(flag.mean()),
            'run_lengths': runs,
            'n_events': len(runs),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='15m', choices=list(TF_CONFIG.keys()))
    ap.add_argument('--sigma-threshold', type=float, default=3.0)
    ap.add_argument('--n-days', type=int, default=50)
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='reports/findings/extreme_dwell')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.features_root, 'L0', '*.parquet')))
    is_days = [os.path.basename(f).replace('.parquet', '') for f in files
                    if os.path.basename(f).startswith('2025_')]
    if len(is_days) > args.n_days:
        idx = np.linspace(0, len(is_days) - 1, args.n_days, dtype=int)
        sample_days = [is_days[i] for i in idx]
    else:
        sample_days = is_days
    print(f'Sampling {len(sample_days)} IS days, TF={args.tf}, '
              f'σ-threshold={args.sigma_threshold}')

    aggregated = defaultdict(lambda: {'n_extreme': 0, 'n_bars_total': 0,
                                                              'all_runs': [], 'n_events': 0})
    for day in tqdm(sample_days, desc='days'):
        result = _per_day_runs(day, args.tf, args.atlas_5s,
                                            args.sigma_threshold)
        if not result:
            continue
        n_bars = result['n_bars']
        for k in ('close+', 'close-', 'high+', 'high-', 'low+', 'low-'):
            agg = aggregated[k]
            agg['n_extreme']    += result[k]['n_extreme']
            agg['n_bars_total'] += n_bars
            agg['all_runs'].extend(result[k]['run_lengths'])
            agg['n_events']     += result[k]['n_events']

    # Print summary
    print(f'\n{"=" * 95}')
    print(f'EXTREME DWELL ANALYSIS  TF={args.tf}  σ-threshold={args.sigma_threshold}')
    print(f'  Frequency: % of 5s bars where price is at OR beyond ±{args.sigma_threshold}σ')
    print(f'  Run length: consecutive 5s bars in extreme state (1 bar = 5s)')
    print(f'{"=" * 95}')
    print(f'{"anchor":<10} {"n_extreme":>10} {"% bars":>8} {"n_events":>9} '
              f'{"med run":>9} {"mean":>8} {"q90":>8} {"max":>8} '
              f'{"med (s)":>9} {"q90 (s)":>9}')
    print('-' * 95)
    for label in ('close+', 'close-', 'high+', 'high-', 'low+', 'low-'):
        agg = aggregated[label]
        runs = np.asarray(agg['all_runs'], dtype=np.float64)
        if len(runs) == 0:
            print(f'{label:<10} {agg["n_extreme"]:>10}'); continue
        med = float(np.median(runs))
        mean_r = float(runs.mean())
        q90 = float(np.quantile(runs, 0.90))
        mx = float(runs.max())
        pct = agg['n_extreme'] / max(agg['n_bars_total'], 1) * 100
        print(f'{label:<10} {agg["n_extreme"]:>10,} {pct:>7.3f}% '
                  f'{agg["n_events"]:>9} '
                  f'{med:>9.0f} {mean_r:>8.1f} {q90:>8.0f} {mx:>8.0f} '
                  f'{med*5:>8.0f}s {q90*5:>8.0f}s')

    # Run-length distribution chart
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, label in zip(axes.flat,
                                   ['close+', 'close-', 'high+', 'high-', 'low+', 'low-']):
        runs = aggregated[label]['all_runs']
        if not runs:
            ax.set_title(f'{label}: no runs')
            continue
        runs_arr = np.asarray(runs)
        ax.hist(runs_arr.clip(0, 240), bins=50, color='tab:purple', alpha=0.75)
        med = float(np.median(runs_arr))
        ax.axvline(med, color='red', lw=1.0, linestyle='--',
                          label=f'median = {med:.0f} bars ({med*5:.0f}s)')
        frac = (aggregated[label]['n_extreme']
                       / max(aggregated[label]['n_bars_total'], 1))
        ax.set_title(f'{label} run-length distribution\n'
                              f'({len(runs)} events, {frac*100:.2f}% of bars)')
        ax.set_xlabel('consecutive 5s bars in extreme state')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out,
                                       f'extreme_dwell_tf{args.tf}_sigma{args.sigma_threshold}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nDistribution chart -> {out_png}')


if __name__ == '__main__':
    main()
