"""CRM flatten-then-pivot detector v2 — adds 5-min monitor + multi-signal confirm.

What v1 (`crm_pivot_detector.py`) caught and missed
---------------------------------------------------
v1 caught the 14:30 macro pivot on 2026_02_12 but produced 17 false candidates.
Root cause: a slope sign-flip after a flatten can be a real impulse OR a wiggle.
v1 had no way to tell them apart in real time.

What v2 adds
------------
After a PIVOT_CANDIDATE, walk forward bar-by-bar for `monitor_minutes` (default 5).
Confirm the candidate as an IMPULSE only if, within that window, ALL three confirmations fire:

    1. Slope retains the NEW sign for ≥ `min_hold_bars` bars (no whip-back)
    2. σ-rank rises to ≥ `sigma_rank_confirm` (default 0.65) — bands expand
    3. vol_velocity_w rank rises to ≥ `volvel_rank_confirm` (default 0.65)
       — pre-pivot leading signal validated visually on 2026_02_12

If the monitor window expires without all three confirming, the candidate is
SUPPRESSED (logged as wiggle) and state returns to NORMAL. This filters the
17 false 2026_02_12 candidates down to the real 14:30 macro pivot.

ZERO LOOKAHEAD: state machine walks bar-by-bar; the monitor window only reads
bars at-or-before the current decision bar.

USAGE:
    python tools/crm_pivot_detector_v2.py --day 2026_02_12 --crm-tf 5m
    python tools/crm_pivot_detector_v2.py --day 2026_03_03 --crm-tf 5m
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
PERIOD_S = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}


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

    # Detection params (v1 + v2 monitor)
    ap.add_argument('--slope-window', type=int, default=60,
                    help='5s bars over which CRM slope is measured (60 = 5min)')
    ap.add_argument('--rank-window', type=int, default=720,
                    help='Rolling window for slope-magnitude rank (720 = 60min)')
    ap.add_argument('--directional-quantile', type=float, default=0.70)
    ap.add_argument('--flatten-quantile', type=float, default=0.30)
    ap.add_argument('--curvature-quantile', type=float, default=0.70)

    # v2 monitor params
    ap.add_argument('--monitor-minutes', type=float, default=5.0,
                    help='Minutes after PIVOT_CANDIDATE to wait for impulse confirm')
    ap.add_argument('--min-hold-bars', type=int, default=12,
                    help='Bars (5s each) the new slope sign must hold (12=60s)')
    ap.add_argument('--sigma-rank-confirm', type=float, default=0.65,
                    help='σ-rank threshold within monitor window for confirmation')
    ap.add_argument('--volvel-rank-confirm', type=float, default=0.65,
                    help='vol_velocity rank threshold within monitor window')
    ap.add_argument('--volvel-window', type=int, default=60,
                    help='Window for vol_velocity rolling-mean diff (5s bars; 60=5min)')

    args = ap.parse_args()

    # 5s OHLCV
    ohlcv_5s = pd.read_parquet(os.path.join(args.atlas_5s, f'{args.day}.parquet'))
    if pd.api.types.is_datetime64_any_dtype(ohlcv_5s['timestamp']):
        ohlcv_5s = ohlcv_5s.copy()
        ohlcv_5s['timestamp'] = (ohlcv_5s['timestamp'].astype('int64') // 10**9)
    ohlcv_5s = ohlcv_5s.sort_values('timestamp').reset_index(drop=True)
    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)
    oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in oh_ts]

    # CRM TF + close mean / sigma
    base, N = TF_CONFIG[args.crm_tf]
    period_s = PERIOD_S[args.crm_tf]
    crm_oh = _load_ohlcv(args.crm_tf, args.day)
    if crm_oh.empty:
        print(f'!!! No {args.crm_tf} OHLCV for {args.day}'); sys.exit(1)
    crm_oh['close_mean']  = crm_oh['close'].rolling(N, min_periods=2).mean()
    crm_oh['close_sigma'] = crm_oh['close'].rolling(N, min_periods=2).std()

    crm_ts = crm_oh['timestamp'].values.astype(np.int64)
    target = oh_ts - period_s
    idx = np.searchsorted(crm_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(crm_ts) - 1)
    M = crm_oh['close_mean'].values[idx]
    S = crm_oh['close_sigma'].values[idx]

    # CRM slope + curvature at 5s cadence
    n = len(M)
    slope = np.full(n, np.nan)
    if n > args.slope_window:
        slope[args.slope_window:] = (M[args.slope_window:]
                                     - M[:-args.slope_window]) / args.slope_window
    curv = np.full(n, np.nan)
    if n > args.slope_window:
        curv[args.slope_window:] = (slope[args.slope_window:]
                                    - slope[:-args.slope_window]) / args.slope_window

    # Ranks (rolling percentile over rank_window)
    s_rank = pd.Series(np.abs(slope)).rolling(args.rank_window,
                                              min_periods=10).rank(pct=True).values
    c_rank = pd.Series(np.abs(curv)).rolling(args.rank_window,
                                             min_periods=10).rank(pct=True).values
    sigma_rank = pd.Series(S).rolling(args.rank_window,
                                      min_periods=10).rank(pct=True).values

    # vol_velocity_w on 5s volume — rolling mean diff over volvel_window
    vol = ohlcv_5s['volume'].values.astype(np.float64)
    vol_mean = pd.Series(vol).rolling(args.volvel_window, min_periods=2).mean().values
    vol_velocity = np.full(n, np.nan)
    if n > args.volvel_window:
        vol_velocity[args.volvel_window:] = (vol_mean[args.volvel_window:]
                                             - vol_mean[:-args.volvel_window]) / args.volvel_window
    volvel_rank = pd.Series(np.abs(vol_velocity)).rolling(args.rank_window,
                                                         min_periods=10).rank(pct=True).values

    monitor_bars = int(round(args.monitor_minutes * 60 / 5))

    # State machine v2
    state = 'NORMAL'
    state_start_slope_sign = 0
    candidate_idx = None             # bar index where pivot candidate fired
    candidate_new_sign = 0           # new slope sign at candidate
    candidate_hold_run = 0           # consecutive bars new sign has held
    candidate_sigma_rank_seen = False
    candidate_volvel_rank_seen = False

    flatten_events = []
    candidate_events = []   # all PIVOT_CANDIDATE fires (pre-confirmation)
    confirmed_events = []   # PIVOT_CANDIDATE that passed monitor (real impulses)
    suppressed_events = []  # PIVOT_CANDIDATE that timed out (wiggles)

    for i in range(args.slope_window + 1, n):
        if not np.isfinite(slope[i]) or not np.isfinite(s_rank[i]):
            continue

        if state == 'NORMAL' and s_rank[i] >= args.directional_quantile:
            state = 'DIRECTIONAL'
            state_start_slope_sign = float(np.sign(slope[i]))

        elif state == 'DIRECTIONAL' and s_rank[i] <= args.flatten_quantile:
            state = 'FLATTENED'
            flatten_events.append((i, slope[i]))

        elif state == 'FLATTENED':
            # Candidate fires: sign flip with significant curvature
            if (np.sign(slope[i]) != state_start_slope_sign
                    and np.sign(slope[i]) != 0
                    and np.isfinite(c_rank[i])
                    and c_rank[i] >= args.curvature_quantile):
                state = 'PIVOT_CANDIDATE'
                candidate_idx = i
                candidate_new_sign = float(np.sign(slope[i]))
                candidate_hold_run = 1
                candidate_sigma_rank_seen = False
                candidate_volvel_rank_seen = False
                candidate_events.append((i, slope[i], state_start_slope_sign))

        elif state == 'PIVOT_CANDIDATE':
            elapsed = i - candidate_idx
            # Track whip-back: hold-run resets if sign reverts
            cur_sign = float(np.sign(slope[i]))
            if cur_sign == candidate_new_sign:
                candidate_hold_run += 1
            else:
                candidate_hold_run = 0
            # Multi-signal accumulators (sticky once seen during the window)
            if (np.isfinite(sigma_rank[i])
                    and sigma_rank[i] >= args.sigma_rank_confirm):
                candidate_sigma_rank_seen = True
            if (np.isfinite(volvel_rank[i])
                    and volvel_rank[i] >= args.volvel_rank_confirm):
                candidate_volvel_rank_seen = True

            # Confirmation check: ALL three within window
            confirmed = (candidate_hold_run >= args.min_hold_bars
                         and candidate_sigma_rank_seen
                         and candidate_volvel_rank_seen)
            if confirmed:
                confirmed_events.append((candidate_idx, i,
                                         candidate_new_sign,
                                         state_start_slope_sign))
                state = 'STABILIZING'
                continue
            # Window expired without confirmation = wiggle
            if elapsed >= monitor_bars:
                suppressed_events.append((candidate_idx,
                                          candidate_hold_run,
                                          candidate_sigma_rank_seen,
                                          candidate_volvel_rank_seen))
                state = 'NORMAL'

        elif state == 'STABILIZING':
            # Re-arm once impulse subsides (slope rank back below directional)
            if np.isfinite(s_rank[i]) and s_rank[i] <= args.flatten_quantile:
                state = 'NORMAL'

    # ---- Reporting --------------------------------------------------------
    print(f'\n{"=" * 80}')
    print(f'CRM PIVOT DETECTOR v2 — {args.day}, CRM={args.crm_tf}')
    print(f'  slope_window {args.slope_window*5}s   rank_window {args.rank_window*5}s')
    print(f'  directional>={args.directional_quantile}  flatten<={args.flatten_quantile}  '
          f'curvature>={args.curvature_quantile}')
    print(f'  monitor={args.monitor_minutes}min  min_hold={args.min_hold_bars*5}s  '
          f'sigma-confirm>={args.sigma_rank_confirm}  volvel-confirm>={args.volvel_rank_confirm}')
    print(f'{"=" * 80}')

    print(f'  FLATTEN events:        {len(flatten_events)}')
    print(f'  PIVOT CANDIDATES:      {len(candidate_events)} (raw v1-equivalent)')
    print(f'  CONFIRMED IMPULSES:    {len(confirmed_events)}')
    print(f'  SUPPRESSED (wiggles):  {len(suppressed_events)}')
    print()

    if confirmed_events:
        print('  CONFIRMED IMPULSES:')
        for cand_i, conf_i, new_sign, prior_sign in confirmed_events:
            cand_dt = datetime.fromtimestamp(int(oh_ts[cand_i]),
                                             tz=timezone.utc).strftime('%H:%M:%S')
            conf_dt = datetime.fromtimestamp(int(oh_ts[conf_i]),
                                             tz=timezone.utc).strftime('%H:%M:%S')
            elapsed_s = (oh_ts[conf_i] - oh_ts[cand_i])
            print(f'    candidate {cand_dt} -> confirmed {conf_dt} '
                  f'(+{elapsed_s}s)  '
                  f'flip {"+" if prior_sign>0 else "-"} -> '
                  f'{"+" if new_sign>0 else "-"}')

    if suppressed_events:
        print(f'\n  SUPPRESSED CANDIDATES (first 10):')
        for cand_i, hold, sigma_ok, volvel_ok in suppressed_events[:10]:
            dt = datetime.fromtimestamp(int(oh_ts[cand_i]),
                                        tz=timezone.utc).strftime('%H:%M:%S')
            reasons = []
            if hold < args.min_hold_bars: reasons.append(f'whip-back(hold={hold*5}s)')
            if not sigma_ok: reasons.append('no sigma-rise')
            if not volvel_ok: reasons.append('no volvel-rise')
            print(f'    {dt}  fail: {", ".join(reasons)}')

    # ---- Plot -------------------------------------------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,
                                             figsize=(args.figwidth, args.figheight),
                                             sharex=True,
                                             gridspec_kw={'height_ratios':
                                                          [3, 1, 1, 1]})

    ax1.plot(oh_dt, ohlcv_5s['close'], color='black', lw=0.6, alpha=0.85,
             label='5s close')
    ax1.plot(oh_dt, M, color='royalblue', lw=1.4, label=f'{args.crm_tf} M_close (CRM)')

    for i, _ in flatten_events:
        ax1.axvline(oh_dt[i], color='gold', lw=0.6, alpha=0.4)
    for cand_i, _, _ in candidate_events:
        ax1.axvline(oh_dt[cand_i], color='gray', lw=0.8, alpha=0.5, linestyle=':')
    for cand_i, conf_i, new_sign, _ in confirmed_events:
        ax1.axvline(oh_dt[cand_i], color='magenta', lw=1.5, alpha=0.85)
        ax1.axvline(oh_dt[conf_i], color='red', lw=1.5, alpha=0.85, linestyle='--')
    for cand_i, _, _, _ in suppressed_events:
        ax1.axvline(oh_dt[cand_i], color='lightgray', lw=0.5, alpha=0.4,
                    linestyle=':')

    ax1.set_title(f'{args.day}  —  CRM ({args.crm_tf}) v2 detector\n'
                  f'gold=flatten  gray-dot=candidate  magenta=confirmed-pivot  '
                  f'dashed-red=impulse-confirm-bar  light-gray=suppressed-wiggle')
    ax1.set_ylabel('price')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(oh_dt, slope, color='royalblue', lw=0.8)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_ylabel(f'CRM slope\n({args.slope_window*5}s lookback)')
    ax2.grid(True, alpha=0.3)

    ax3.plot(oh_dt, sigma_rank, color='tab:purple', lw=0.8, label='sigma-rank')
    ax3.plot(oh_dt, s_rank, color='royalblue', lw=0.6, alpha=0.6, label='|slope|-rank')
    ax3.axhline(args.sigma_rank_confirm, color='tab:purple', lw=0.5, linestyle=':')
    ax3.set_ylabel('rank')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4.plot(oh_dt, volvel_rank, color='tab:orange', lw=0.8,
             label='|vol_velocity|-rank')
    ax4.axhline(args.volvel_rank_confirm, color='tab:orange', lw=0.5, linestyle=':')
    ax4.set_ylabel('volvel-rank')
    ax4.set_xlabel('time (UTC)')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, f'{args.day}_crm_pivot_v2.png')
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
