"""Detect natural intraday regime segments across all days.

Each day has natural breakpoints where the character of price action
changes (rally → top → crash, calm → impulse, etc.). Treating a day as
ONE regime label hides this structure. This tool segments each day
into runs of consistent regime, defined by the 15m regression-mean slope.

ALGORITHM (per day):
    1. Compute slope of 15m regression mean over a 1h lookback.
    2. Threshold |slope| at Q60 of |slope| within the day.
    3. Sign of slope (where above threshold) → UP / DOWN segments.
       Below threshold → FLAT.
    4. Coalesce adjacent same-class bins into segments.
    5. Each segment gets:
        - regime          : UP / DOWN / FLAT
        - intra-segment swing_noise mean → SMOOTH / CHOPPY tag
        - start_ts, end_ts, duration_min
        - net_move_pts (close at end − start)
        - max_excursion_pts

OUTPUT:
    DATA/ATLAS/segments.parquet — one row per (day, segment)
        columns: day, segment_idx, start_ts, end_ts, duration_min,
                 regime_2d, direction, variation, mean_slope,
                 net_move_pts, max_excursion_pts, mean_swing_noise

Each segment is forward-pass-honest (slope uses past-only window).

USAGE:
    python tools/segment_days.py
    python tools/segment_days.py --day 2026_02_12        # single day, prints + chart
    python tools/segment_days.py --slope-thr-quantile 0.7
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import load_features


MEAN_15M = 'L2_15m_price_mean_12'
SWING_1M = 'L3_1m_swing_noise_15'
SLOPE_LB = 720    # 1h lookback (5s bars × 720 = 3600s)
BIN_BARS = 60     # 5-min bins (60 × 5s)
MIN_SEGMENT_BINS = 3   # minimum 15 min before a segment counts


def _segment_one_day(day: str, features_root: str,
                              atlas_5s: str,
                              slope_thr_quantile: float,
                              sn_smooth_quantile: float) -> pd.DataFrame:
    feats = load_features(days=[day], root=features_root)
    if feats.empty or MEAN_15M not in feats.columns:
        return pd.DataFrame()
    feats = feats.sort_values('timestamp').reset_index(drop=True)
    ts_arr = feats['timestamp'].astype('int64').values
    m15 = feats[MEAN_15M].values.astype(np.float64)
    sn = (feats[SWING_1M].values.astype(np.float64)
              if SWING_1M in feats.columns else np.zeros_like(m15))

    n = len(m15)
    slope = np.full(n, np.nan)
    if n > SLOPE_LB:
        slope[SLOPE_LB:] = (m15[SLOPE_LB:] - m15[:-SLOPE_LB]) / SLOPE_LB

    finite = np.abs(slope[np.isfinite(slope)])
    if len(finite) == 0:
        return pd.DataFrame()
    slope_thr = float(np.quantile(finite, slope_thr_quantile))
    sn_thr = float(np.quantile(sn[np.isfinite(sn)], sn_smooth_quantile))

    # Bin into 5-min segments; pick majority direction per bin
    n_bins = (n + BIN_BARS - 1) // BIN_BARS
    bin_dirs = []
    bin_slopes = []
    bin_sns = []
    bin_starts = []
    bin_ends = []
    for b in range(n_bins):
        a = b * BIN_BARS
        z = min(a + BIN_BARS, n)
        seg_slope = slope[a:z]
        seg_slope = seg_slope[np.isfinite(seg_slope)]
        if len(seg_slope) == 0:
            continue
        ms = float(seg_slope.mean())
        if abs(ms) < slope_thr:
            d = 'FLAT'
        elif ms > 0:
            d = 'UP'
        else:
            d = 'DOWN'
        seg_sn = sn[a:z]
        seg_sn = seg_sn[np.isfinite(seg_sn)]
        sn_mean = float(seg_sn.mean()) if len(seg_sn) > 0 else float('nan')
        bin_dirs.append(d)
        bin_slopes.append(ms)
        bin_sns.append(sn_mean)
        bin_starts.append(int(ts_arr[a]))
        bin_ends.append(int(ts_arr[z - 1]))

    # Coalesce adjacent bins with same direction
    segments = []
    cur = None
    for i in range(len(bin_dirs)):
        if cur is None or bin_dirs[i] != cur['direction']:
            if cur is not None:
                segments.append(cur)
            cur = {
                'direction': bin_dirs[i],
                'start_ts': bin_starts[i],
                'end_ts': bin_ends[i],
                'slopes': [bin_slopes[i]],
                'sns': [bin_sns[i]],
            }
        else:
            cur['end_ts'] = bin_ends[i]
            cur['slopes'].append(bin_slopes[i])
            cur['sns'].append(bin_sns[i])
    if cur is not None:
        segments.append(cur)

    # Drop too-short segments by merging into neighbors (keep stability)
    cleaned = []
    for s in segments:
        n_bins_in_seg = max(
            1, (s['end_ts'] - s['start_ts']) // (BIN_BARS * 5))
        if n_bins_in_seg < MIN_SEGMENT_BINS and cleaned:
            # Merge into previous
            prev = cleaned[-1]
            prev['end_ts'] = s['end_ts']
            prev['slopes'].extend(s['slopes'])
            prev['sns'].extend(s['sns'])
        else:
            cleaned.append(s)
    segments = cleaned

    # Get OHLCV close for net-move computation
    ohlcv_path = os.path.join(atlas_5s, f'{day}.parquet')
    if not os.path.exists(ohlcv_path):
        return pd.DataFrame()
    ohlcv = pd.read_parquet(ohlcv_path)
    if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
        ohlcv = ohlcv.copy()
        ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
    ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
    oh_ts = ohlcv['timestamp'].values.astype(np.int64)
    oh_close = ohlcv['close'].values

    rows = []
    for idx, s in enumerate(segments):
        # Find OHLCV closes at segment boundaries
        i_start = int(np.searchsorted(oh_ts, s['start_ts'], side='left'))
        i_end = int(np.searchsorted(oh_ts, s['end_ts'], side='right')) - 1
        i_start = max(0, min(i_start, len(oh_close) - 1))
        i_end = max(0, min(i_end, len(oh_close) - 1))
        net_move = float(oh_close[i_end] - oh_close[i_start])
        seg_close = oh_close[i_start:i_end + 1]
        max_exc = (float(seg_close.max() - seg_close.min())
                          if len(seg_close) > 0 else 0.0)
        mean_sn = float(np.nanmean(s['sns'])) if s['sns'] else float('nan')
        variation = ('SMOOTH' if (mean_sn != mean_sn or mean_sn < sn_thr)
                          else 'CHOPPY')
        regime = f'{s["direction"]}_{variation}'
        duration_s = s['end_ts'] - s['start_ts']
        rows.append({
            'day': day,
            'segment_idx': idx,
            'start_ts': s['start_ts'],
            'end_ts': s['end_ts'],
            'duration_min': duration_s / 60.0,
            'direction': s['direction'],
            'variation': variation,
            'regime_2d': regime,
            'mean_slope': float(np.mean(s['slopes'])) if s['slopes'] else 0.0,
            'mean_swing_noise': mean_sn,
            'net_move_pts': net_move,
            'max_excursion_pts': max_exc,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None,
                          help='Single day (YYYY_MM_DD) — prints + plots; otherwise scan all IS+OOS')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out-parquet', default='DATA/ATLAS/segments.parquet')
    ap.add_argument('--slope-thr-quantile', type=float, default=0.60,
                          help='Q of |slope| above which a bin is "directional"')
    ap.add_argument('--sn-quantile', type=float, default=0.60,
                          help='Q of swing_noise above which segment is CHOPPY')
    ap.add_argument('--out-chart', default='chart')
    args = ap.parse_args()

    if args.day:
        df = _segment_one_day(args.day, args.features_root, args.atlas_5s,
                                          args.slope_thr_quantile, args.sn_quantile)
        if df.empty:
            print(f'No segments for {args.day}'); return
        print(f'\n{"=" * 110}')
        print(f'SEGMENTS for {args.day}  ({len(df)} segments)')
        print(f'{"=" * 110}')
        print(f'{"#":>3} {"start":>8} {"end":>8} {"dur(min)":>8} {"regime":<13} '
                  f'{"net$":>+8} {"max_exc":>+8} {"swing_noise":>+12}')
        for _, r in df.iterrows():
            t0 = datetime.fromtimestamp(int(r['start_ts']), tz=timezone.utc).strftime('%H:%M')
            t1 = datetime.fromtimestamp(int(r['end_ts']), tz=timezone.utc).strftime('%H:%M')
            print(f'{int(r["segment_idx"]):>3} {t0:>8} {t1:>8} '
                      f'{r["duration_min"]:>8.1f} {r["regime_2d"]:<13} '
                      f'{r["net_move_pts"]:>+8.2f} {r["max_excursion_pts"]:>+8.2f} '
                      f'{r["mean_swing_noise"]:>+12.1f}')
        # Single-day chart
        os.makedirs(args.out_chart, exist_ok=True)
        ohlcv = pd.read_parquet(os.path.join(args.atlas_5s, f'{args.day}.parquet'))
        if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
            ohlcv = ohlcv.copy()
            ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
        ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
        oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc)
                       for t in ohlcv['timestamp'].values]
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(oh_dt, ohlcv['close'], color='black', lw=0.7)
        regime_colors = {
            'UP_SMOOTH':   ('lightgreen',  0.30),
            'UP_CHOPPY':   ('palegreen',   0.20),
            'DOWN_SMOOTH': ('lightcoral',  0.30),
            'DOWN_CHOPPY': ('mistyrose',   0.30),
            'FLAT_SMOOTH': ('lightyellow', 0.20),
            'FLAT_CHOPPY': ('khaki',       0.18),
        }
        for _, r in df.iterrows():
            t0 = datetime.fromtimestamp(int(r['start_ts']), tz=timezone.utc)
            t1 = datetime.fromtimestamp(int(r['end_ts']), tz=timezone.utc)
            color, alpha = regime_colors.get(r['regime_2d'],
                                                              ('lightgray', 0.15))
            ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
            mid_ts = (int(r['start_ts']) + int(r['end_ts'])) // 2
            mid_dt = datetime.fromtimestamp(mid_ts, tz=timezone.utc)
            ax.text(mid_dt, ax.get_ylim()[1] - 5, r['regime_2d'],
                         ha='center', fontsize=8, alpha=0.9,
                         rotation=45)
        ax.set_title(f'{args.day} — natural day segments  '
                          f'(slope Q{args.slope_thr_quantile:.0%}, '
                          f'sn Q{args.sn_quantile:.0%})')
        ax.set_ylabel('price')
        ax.set_xlabel('time (UTC)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(True, alpha=0.3)
        out_png = os.path.join(args.out_chart, f'{args.day}_segments.png')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'\nChart -> {out_png}')
        return

    # Scan all days
    files = sorted(glob.glob(os.path.join(args.features_root, 'L0', '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    print(f'Segmenting {len(days)} days...')
    all_rows = []
    for day in tqdm(days, desc='segment'):
        df = _segment_one_day(day, args.features_root, args.atlas_5s,
                                          args.slope_thr_quantile, args.sn_quantile)
        if not df.empty:
            all_rows.append(df)
    if not all_rows:
        print('No segments produced'); return
    out_df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f'\nWrote {len(out_df)} segments → {args.out_parquet}')

    # Distribution print
    print(f'\n{"=" * 80}')
    print('SEGMENT REGIME DISTRIBUTION')
    print(f'{"=" * 80}')
    counts = out_df['regime_2d'].value_counts()
    total = len(out_df)
    for r, n in counts.items():
        avg_dur = out_df[out_df['regime_2d'] == r]['duration_min'].mean()
        avg_net = out_df[out_df['regime_2d'] == r]['net_move_pts'].mean()
        print(f'  {r:<14} {n:>5} ({n/total*100:>4.1f}%)  '
                  f'avg dur {avg_dur:>5.1f} min   avg net {avg_net:>+7.2f} pts')


if __name__ == '__main__':
    main()
