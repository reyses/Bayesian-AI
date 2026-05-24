"""Test the "compression precedes extreme" hypothesis.

Hypothesis: before a 3σ extreme event begins, σ at the same TF is
in the BOTTOM decile (compressed) of its recent values. Bands narrow,
then expand explosively as the move develops.

Method:
    1. Walk IS days at 5s.
    2. Compute σ at chosen TF + percentile rank over rolling window
       (default 60 min = 720 5s bars).
    3. Detect entry into ±3σ extreme state (transition).
    4. At each entry, look BACKWARDS at σ-percentile-rank in the
       N bars (default 60s = 12 bars) BEFORE the entry.
    5. Compare to baseline σ-percentile distribution.

Output: histogram of σ-percentile at extreme entry; compared to overall.
        If hypothesis holds, distribution shifts heavily toward LOW values.

Usage:
    python tools/compression_before_extreme.py --tf 1h
    python tools/compression_before_extreme.py --tf 15m --lookback-pre 60
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

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


def _per_day_ranks(day: str, tf: str, atlas_5s: str,
                              sigma_thresh: float, sigma_window: int,
                              lookback_pre: int) -> dict:
    """Return σ-percentiles at extreme-entry vs σ-percentiles overall."""
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

    # σ at TF
    tf_oh['high_mean']  = tf_oh['high'].rolling(N, min_periods=2).mean()
    tf_oh['high_sigma'] = tf_oh['high'].rolling(N, min_periods=2).std()
    tf_oh['low_mean']   = tf_oh['low'].rolling(N, min_periods=2).mean()
    tf_oh['low_sigma']  = tf_oh['low'].rolling(N, min_periods=2).std()

    # Forward-fill to 5s
    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)
    tf_ts = tf_oh['timestamp'].values.astype(np.int64)
    target = oh_ts - period_s
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)

    p = ohlcv_5s['close'].values
    Mh = tf_oh['high_mean'].values[idx]
    Sh = tf_oh['high_sigma'].values[idx]
    Ml = tf_oh['low_mean'].values[idx]
    Sl = tf_oh['low_sigma'].values[idx]

    d_high = (p - Mh) / Sh
    d_low  = (p - Ml) / Sl

    # σ-rank: rolling percentile of σ_high over `sigma_window` 5s bars
    Sh_series = pd.Series(Sh)
    Sl_series = pd.Series(Sl)
    Sh_rank = Sh_series.rolling(sigma_window, min_periods=10).rank(pct=True).values
    Sl_rank = Sl_series.rolling(sigma_window, min_periods=10).rank(pct=True).values

    # Find ENTRIES into 3σ extreme: prev bar < threshold, current >= threshold
    out = {'high_entry_ranks': [], 'low_entry_ranks': [],
              'all_high_ranks': Sh_rank[np.isfinite(Sh_rank)],
              'all_low_ranks': Sl_rank[np.isfinite(Sl_rank)]}

    in_high_extreme = (d_high >= sigma_thresh)
    in_low_extreme  = (d_low  <= -sigma_thresh)
    for i in range(1, len(d_high)):
        # Entry into HIGH extreme
        if in_high_extreme[i] and not in_high_extreme[i-1]:
            j = max(0, i - lookback_pre)
            if j < i and np.isfinite(Sh_rank[j]):
                out['high_entry_ranks'].append(float(Sh_rank[j]))
        # Entry into LOW extreme
        if in_low_extreme[i] and not in_low_extreme[i-1]:
            j = max(0, i - lookback_pre)
            if j < i and np.isfinite(Sl_rank[j]):
                out['low_entry_ranks'].append(float(Sl_rank[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='1h', choices=list(TF_CONFIG.keys()))
    ap.add_argument('--sigma-threshold', type=float, default=3.0)
    ap.add_argument('--sigma-window', type=int, default=720,
                          help='Rolling window for σ-rank (5s bars; 720 = 60 min)')
    ap.add_argument('--lookback-pre', type=int, default=12,
                          help='Bars BEFORE entry where we sample σ-rank '
                                  '(default 12 = 60s before entry)')
    ap.add_argument('--n-days', type=int, default=50)
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='reports/findings/compression')
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
              f'σ-threshold={args.sigma_threshold}, '
              f'lookback {args.lookback_pre} bars ({args.lookback_pre*5}s) before entry')

    high_entries, low_entries = [], []
    all_high_ranks, all_low_ranks = [], []
    for day in tqdm(sample_days, desc='days'):
        result = _per_day_ranks(day, args.tf, args.atlas_5s,
                                          args.sigma_threshold,
                                          args.sigma_window, args.lookback_pre)
        if not result: continue
        high_entries.extend(result['high_entry_ranks'])
        low_entries.extend(result['low_entry_ranks'])
        all_high_ranks.extend(result['all_high_ranks'].tolist())
        all_low_ranks.extend(result['all_low_ranks'].tolist())

    print(f'\n{"=" * 80}')
    print(f'COMPRESSION-PRECEDES-EXTREME TEST  TF={args.tf}  threshold=±{args.sigma_threshold}σ')
    print(f'  σ-rank: rolling percentile of σ_high (or σ_low) over {args.sigma_window} bars')
    print(f'  At entry into 3σ extreme, sample σ-rank from {args.lookback_pre} bars BEFORE')
    print(f'{"=" * 80}')

    def stats(label, entries, base):
        e = np.asarray(entries)
        b = np.asarray(base)
        if len(e) == 0:
            print(f'{label}: no entries'); return
        print(f'\n{label}  n_events = {len(e)}')
        print(f'  σ-rank at entry-{args.lookback_pre}bars:  '
                  f'mean={e.mean():.3f}  median={np.median(e):.3f}  '
                  f'q10={np.quantile(e, 0.10):.3f}  q90={np.quantile(e, 0.90):.3f}')
        print(f'  σ-rank baseline (all bars):    '
                  f'mean={b.mean():.3f}  median={np.median(b):.3f}')
        below_q33 = float((e < 0.33).mean())
        below_q20 = float((e < 0.20).mean())
        print(f'  Compression evidence:')
        print(f'    % entries with σ-rank < 0.33 (bottom third): {below_q33*100:.1f}%')
        print(f'    % entries with σ-rank < 0.20 (bottom fifth):  {below_q20*100:.1f}%')
        print(f'    Random expectation if no compression:         33% / 20%')

    stats('HIGH+ extreme entries', high_entries, all_high_ranks)
    stats('LOW− extreme entries',  low_entries,  all_low_ranks)

    # Histogram
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, entries, base) in zip(axes, [
            ('HIGH+', high_entries, all_high_ranks),
            ('LOW−',  low_entries,  all_low_ranks)]):
        if not entries: continue
        ax.hist(np.asarray(entries), bins=20, alpha=0.7, color='tab:purple',
                       density=True, label='at extreme entry')
        ax.hist(np.asarray(base), bins=20, alpha=0.3, color='gray',
                       density=True, label='baseline')
        ax.axvline(0.33, color='red', lw=0.8, linestyle='--')
        ax.axvline(0.20, color='red', lw=1.0, linestyle='--')
        ax.set_title(f'{label}: σ-rank {args.lookback_pre*5}s before extreme\n'
                              f'(left = compressed, right = expanded)')
        ax.set_xlabel('σ-rank percentile')
        ax.set_ylabel('density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    out_png = os.path.join(args.out,
                                       f'compression_tf{args.tf}_sigma{args.sigma_threshold}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
