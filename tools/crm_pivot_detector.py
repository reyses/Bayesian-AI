"""CRM flatten-then-pivot detector — early-warning for macro impulse onset.

Hypothesis: before a multi-hour impulse, the close regression mean (CRM)
goes through this progression:
    UPTREND →  FLATTEN  →  PIVOT  →  IMPULSE

Detection:
    1. CRM slope was strongly directional (|slope| in top quartile)
    2. CRM slope drops to near-zero (|slope| in bottom quartile)  → FLATTEN
    3. CRM slope flips sign WITH significant curvature             → PIVOT
    4. Sigma starts expanding                                       → IMPULSE

Marks each detection event on the chart so we can audit visually.

USAGE:
    python tools/crm_pivot_detector.py --day 2026_02_12 --crm-tf 5m
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2026_02_12')
    ap.add_argument('--crm-tf', default='5m', choices=list(TF_CONFIG.keys()))
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='chart')
    ap.add_argument('--dpi', type=int, default=240)
    ap.add_argument('--figwidth', type=float, default=22)
    ap.add_argument('--figheight', type=float, default=14)
    # Detection parameters
    ap.add_argument('--slope-window', type=int, default=60,
                          help='5s bars over which CRM slope is measured (60 = 5min)')
    ap.add_argument('--rank-window', type=int, default=720,
                          help='Rolling window for slope-magnitude rank (720 = 60min)')
    ap.add_argument('--directional-quantile', type=float, default=0.70,
                          help='|slope| above this rank is "directional"')
    ap.add_argument('--flatten-quantile', type=float, default=0.30,
                          help='|slope| below this rank is "flattened"')
    ap.add_argument('--curvature-quantile', type=float, default=0.70,
                          help='|curvature| above this rank confirms a pivot')
    args = ap.parse_args()

    # Load 5s OHLCV
    ohlcv_5s = pd.read_parquet(os.path.join(args.atlas_5s, f'{args.day}.parquet'))
    if pd.api.types.is_datetime64_any_dtype(ohlcv_5s['timestamp']):
        ohlcv_5s = ohlcv_5s.copy()
        ohlcv_5s['timestamp'] = (ohlcv_5s['timestamp'].astype('int64') // 10**9)
    ohlcv_5s = ohlcv_5s.sort_values('timestamp').reset_index(drop=True)
    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)
    oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in oh_ts]

    # Load CRM TF + compute close mean
    base, N = TF_CONFIG[args.crm_tf]
    period_s = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}[args.crm_tf]
    crm_oh = _load_ohlcv(args.crm_tf, args.day)
    if crm_oh.empty:
        print(f'!!! No {args.crm_tf} OHLCV for {args.day}'); sys.exit(1)
    crm_oh['close_mean']  = crm_oh['close'].rolling(N, min_periods=2).mean()
    crm_oh['close_sigma'] = crm_oh['close'].rolling(N, min_periods=2).std()

    # Forward-fill to 5s
    crm_ts = crm_oh['timestamp'].values.astype(np.int64)
    target = oh_ts - period_s
    idx = np.searchsorted(crm_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(crm_ts) - 1)
    M = crm_oh['close_mean'].values[idx]
    S = crm_oh['close_sigma'].values[idx]

    # Compute CRM slope at 5s cadence over slope_window
    n = len(M)
    slope = np.full(n, np.nan)
    if n > args.slope_window:
        slope[args.slope_window:] = (M[args.slope_window:]
                                                       - M[:-args.slope_window]) / args.slope_window
    # Curvature = derivative of slope
    curv = np.full(n, np.nan)
    if n > args.slope_window:
        curv[args.slope_window:] = (slope[args.slope_window:]
                                                    - slope[:-args.slope_window]) / args.slope_window

    # σ-rank, slope-magnitude rank, curvature-magnitude rank (rolling percentile)
    s_rank = pd.Series(np.abs(slope)).rolling(args.rank_window,
                                                                       min_periods=10).rank(pct=True).values
    c_rank = pd.Series(np.abs(curv)).rolling(args.rank_window,
                                                                      min_periods=10).rank(pct=True).values
    sigma_rank = pd.Series(S).rolling(args.rank_window,
                                                              min_periods=10).rank(pct=True).values

    # State machine: detect flatten-then-pivot
    state = 'NORMAL'
    flatten_events = []   # bar indices when flatten was detected
    pivot_events = []     # bar indices when pivot was detected
    impulse_events = []   # bar indices when impulse onset confirmed

    for i in range(args.slope_window + 1, n):
        if not np.isfinite(slope[i]) or not np.isfinite(s_rank[i]):
            continue
        # State: DIRECTIONAL → FLATTEN
        if state == 'NORMAL' and s_rank[i] >= args.directional_quantile:
            state = 'DIRECTIONAL'
            state_start_slope_sign = np.sign(slope[i])
        elif state == 'DIRECTIONAL' and s_rank[i] <= args.flatten_quantile:
            state = 'FLATTENED'
            flatten_events.append((i, slope[i]))
        elif state == 'FLATTENED':
            # Look for pivot: slope sign-flip with significant curvature
            if (np.sign(slope[i]) != state_start_slope_sign
                      and np.isfinite(c_rank[i])
                      and c_rank[i] >= args.curvature_quantile):
                pivot_events.append((i, slope[i], state_start_slope_sign))
                state = 'PIVOTED'
        elif state == 'PIVOTED':
            # Confirm impulse: σ rank rising
            if np.isfinite(sigma_rank[i]) and sigma_rank[i] >= 0.6:
                impulse_events.append((i,))
                state = 'NORMAL'

    print(f'\n{"=" * 80}')
    print(f'CRM PIVOT DETECTOR — {args.day}, CRM={args.crm_tf}')
    print(f'  slope_window {args.slope_window*5}s   rank_window {args.rank_window*5}s')
    print(f'  directional≥{args.directional_quantile}  flatten≤{args.flatten_quantile}  '
              f'curvature≥{args.curvature_quantile}')
    print(f'{"=" * 80}')
    print(f'  FLATTEN events: {len(flatten_events)}')
    for i, sl in flatten_events:
        dt = datetime.fromtimestamp(int(oh_ts[i]), tz=timezone.utc).strftime('%H:%M:%S')
        print(f'    {dt}  CRM slope={sl:+.3f}')
    print(f'  PIVOT events:   {len(pivot_events)}')
    for i, sl, prior_sign in pivot_events:
        dt = datetime.fromtimestamp(int(oh_ts[i]), tz=timezone.utc).strftime('%H:%M:%S')
        print(f'    {dt}  slope flipped from {"+" if prior_sign>0 else "-"} '
                  f'to {"+" if sl>0 else "-"}  curv_rank={c_rank[i]:.2f}')
    print(f'  IMPULSE confirmations: {len(impulse_events)}')
    for (i,) in impulse_events:
        dt = datetime.fromtimestamp(int(oh_ts[i]), tz=timezone.utc).strftime('%H:%M:%S')
        print(f'    {dt}  σ_rank={sigma_rank[i]:.2f}')

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(args.figwidth, args.figheight),
                                                              sharex=True,
                                                              gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1.plot(oh_dt, ohlcv_5s['close'], color='black', lw=0.6, alpha=0.85,
                  label='5s close')
    ax1.plot(oh_dt, M, color='royalblue', lw=1.4, label=f'{args.crm_tf} M_close (CRM)')

    # Mark events
    for i, _ in flatten_events:
        ax1.axvline(oh_dt[i], color='gold', lw=0.8, alpha=0.5)
    for i, _, _ in pivot_events:
        ax1.axvline(oh_dt[i], color='magenta', lw=1.5, alpha=0.85,
                          label='PIVOT' if i == pivot_events[0][0] else None)
    for (i,) in impulse_events:
        ax1.axvline(oh_dt[i], color='red', lw=1.5, alpha=0.85, linestyle='--',
                          label='IMPULSE confirmed' if i == impulse_events[0][0] else None)

    ax1.set_title(f'{args.day}  —  CRM ({args.crm_tf}) flatten/pivot/impulse detection\n'
                       f'gold=flatten  magenta=pivot  dashed-red=impulse confirmed')
    ax1.set_ylabel('price')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(oh_dt, slope, color='royalblue', lw=0.8)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_ylabel(f'CRM slope\n({args.slope_window*5}s lookback)')
    ax2.grid(True, alpha=0.3)

    ax3.plot(oh_dt, sigma_rank, color='tab:purple', lw=0.8, label='σ-rank')
    ax3.plot(oh_dt, s_rank, color='royalblue', lw=0.6, alpha=0.6, label='|slope|-rank')
    ax3.axhline(args.directional_quantile, color='gray', lw=0.4, linestyle=':')
    ax3.axhline(args.flatten_quantile, color='gray', lw=0.4, linestyle=':')
    ax3.set_ylabel('rank')
    ax3.set_xlabel('time (UTC)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, f'{args.day}_crm_pivot.png')
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
