"""Visualize all four Bayesian-filter overlays on a single day.

Top panel: 5s close + 1h HL bands + 15m M_close
   - TOD risk zones shaded by P_cat color (high=red, low=green)
   - CAT_HARVEST windows marked with vertical dashed lines + 'PRE_SHORT' label
   - COMPRESSION_BOUNCE active periods marked with green tint + 'LONG_BIAS' label

Bottom panel: P_cat(t) curve + size multiplier curve

USAGE
    python tools/chart_augmented_filters_on_day.py --day 2025_10_29
    python tools/chart_augmented_filters_on_day.py --day 2025_10_29 --start-hour 12 --end-hour 24
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, time, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv
from training_iso_v2.filters.bayes_filters import (
    tod_risk_size_multiplier, cat_harvest_signal,
    _load_tod_dow_p_cat, CompressionBounce,
)


def compute_anchor(tf: str, day: str, ts_5s: np.ndarray, window: int, column: str):
    period_s_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    if tf not in period_s_map: return None, None
    period_s = period_s_map[tf]
    oh = _load_tf_ohlcv(tf, day)
    if oh.empty: return None, None
    M = oh[column].rolling(window, min_periods=2).mean().values
    S = oh[column].rolling(window, min_periods=2).std().values
    tf_ts = oh['timestamp'].values.astype(np.int64)
    target = ts_5s - period_s
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)
    return M[idx], S[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_10_29')
    ap.add_argument('--start-hour', type=int, default=None)
    ap.add_argument('--end-hour', type=int, default=None)
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_5s = _load_5s(args.day)
    if df_5s.empty: print(f'No 5s data for {args.day}'); return
    ts = df_5s['timestamp'].values.astype(np.int64)
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    close = df_5s['close'].values.astype(np.float64)

    M_15m, _   = compute_anchor('15m', args.day, ts, 12, 'close')
    Mh_1h, Sh_1h = compute_anchor('1h', args.day, ts, 12, 'high')
    Ml_1h, Sl_1h = compute_anchor('1h', args.day, ts, 12, 'low')

    # Load L2_15m_vol_sigma_12 for compression detection
    vol_sigma_15m = None
    path_l215m = f'DATA/ATLAS/FEATURES_5s_v2/L2_15m/{args.day}.parquet'
    if os.path.exists(path_l215m):
        f_df = pd.read_parquet(path_l215m)
        if 'L2_15m_vol_sigma_12' in f_df.columns:
            f_df['timestamp'] = f_df['timestamp'].astype(np.int64)
            close_df = df_5s.copy()
            close_df['timestamp'] = close_df['timestamp'].astype(np.int64)
            merged = pd.merge_asof(close_df.sort_values('timestamp'),
                                     f_df[['timestamp', 'L2_15m_vol_sigma_12']]
                                       .sort_values('timestamp'),
                                     on='timestamp', direction='backward')
            vol_sigma_15m = merged['L2_15m_vol_sigma_12'].values

    # Compute filter states bar-by-bar
    cb = CompressionBounce()
    cb.reset(args.day)
    p_cat_arr = np.full(len(ts), np.nan)
    size_mult_arr = np.full(len(ts), 1.0)
    cat_harvest_active = np.zeros(len(ts), dtype=bool)
    comp_active = np.zeros(len(ts), dtype=bool)

    table = _load_tod_dow_p_cat()
    for i, t in enumerate(ts):
        dt_i = datetime.fromtimestamp(int(t), tz=timezone.utc)
        p = table.get((dt_i.strftime('%a'), dt_i.hour), 0.034)
        p_cat_arr[i] = p
        size_mult_arr[i] = tod_risk_size_multiplier(int(t))
        cat_harvest_active[i] = cat_harvest_signal(int(t)) is not None
        if vol_sigma_15m is not None and i < len(vol_sigma_15m):
            comp_state = cb.update(vol_sigma_15m[i])
            comp_active[i] = (comp_state == 'LONG_BIAS')

    # ===== RENDER =====
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ---- TOP: price + bands + filter overlays ----
    ax1.plot(dt, Mh_1h, color='#43A047', lw=1.4, alpha=0.85, label='1h M_high')
    ax1.fill_between(dt, Mh_1h+2*Sh_1h, Mh_1h+3*Sh_1h, color='#43A047', alpha=0.13)
    ax1.plot(dt, Mh_1h+3*Sh_1h, color='#1B5E20', lw=0.8, ls='--', alpha=0.7)
    ax1.plot(dt, Ml_1h, color='#E53935', lw=1.4, alpha=0.85, label='1h M_low')
    ax1.fill_between(dt, Ml_1h-3*Sl_1h, Ml_1h-2*Sl_1h, color='#E53935', alpha=0.13)
    ax1.plot(dt, Ml_1h-3*Sl_1h, color='#B71C1C', lw=0.8, ls='--', alpha=0.7)
    ax1.plot(dt, M_15m, color='#1E88E5', lw=1.6, alpha=0.95, label='15m M_close')
    ax1.plot(dt, close, color='black', lw=0.5, alpha=0.85, label='5s close')

    # Shade compression-bounce zones (green tint)
    in_comp_segments = []
    i = 0
    while i < len(comp_active):
        if comp_active[i]:
            j = i
            while j < len(comp_active) and comp_active[j]: j += 1
            in_comp_segments.append((i, j-1))
            i = j
        else:
            i += 1
    for s, e in in_comp_segments:
        ax1.axvspan(dt[s], dt[e], color='#00C853', alpha=0.18,
                      label='COMPRESSION LONG_BIAS' if s == in_comp_segments[0][0] else None,
                      zorder=0)

    # Shade cat-harvest pre-position windows (orange-red striped)
    in_harvest_segments = []
    i = 0
    while i < len(cat_harvest_active):
        if cat_harvest_active[i]:
            j = i
            while j < len(cat_harvest_active) and cat_harvest_active[j]: j += 1
            in_harvest_segments.append((i, j-1))
            i = j
        else:
            i += 1
    for s, e in in_harvest_segments:
        ax1.axvspan(dt[s], dt[e], color='#FF1744', alpha=0.20, hatch='///',
                      label='CAT HARVEST window (PRE_SHORT)' if s == in_harvest_segments[0][0] else None,
                      zorder=0)

    ax1.set_title(f'{args.day} — AUGMENTED FILTERS overlay\n'
                   f'green tint = COMPRESSION LONG_BIAS active  /  '
                   f'red hatched = CAT HARVEST PRE_SHORT window',
                   fontsize=12)
    ax1.set_ylabel('price'); ax1.legend(loc='best', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.20)

    # ---- MIDDLE: P_cat(t) ----
    ax2.fill_between(dt, 0, p_cat_arr * 100, color='#E53935', alpha=0.65,
                       label='P(cat in next 60m) %')
    ax2.axhline(10, color='black', ls='--', lw=0.5, alpha=0.5, label='10% threshold')
    ax2.axhline(20, color='red', ls='--', lw=0.5, alpha=0.5, label='20% peak danger')
    ax2.set_ylabel('P(cat) %'); ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 45)

    # ---- BOTTOM: size multiplier ----
    ax3.fill_between(dt, 0, size_mult_arr, color='#1E88E5', alpha=0.55,
                       label='TOD/DOW size multiplier')
    ax3.set_ylabel('size mult'); ax3.set_xlabel('time (UTC)')
    ax3.set_ylim(0, 1.1); ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    if args.start_hour is not None or args.end_hour is not None:
        sh = args.start_hour if args.start_hour is not None else 0
        eh = args.end_hour if args.end_hour is not None else 24
        date_part = dt[0].date()
        x_start = datetime.combine(date_part, time(sh, 0), tzinfo=timezone.utc)
        x_end = datetime.combine(date_part, time(min(eh, 23), 59), tzinfo=timezone.utc) \
                 if eh < 24 else dt[-1]
        for ax in (ax1, ax2, ax3): ax.set_xlim(x_start, x_end)
        suffix = f'_h{sh:02d}-h{eh:02d}'
    else:
        suffix = ''

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    out_path = os.path.join(args.out_dir,
                              f'augmented_filters_{args.day}{suffix}.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Chart -> {out_path}')

    # Summary
    n_total = len(ts)
    print(f'\nDay summary for {args.day}:')
    print(f'  Total 5s bars: {n_total:,}')
    print(f'  Bars in CAT_HARVEST window:    {int(cat_harvest_active.sum()):,}  '
           f'({100*cat_harvest_active.mean():.1f}%)')
    print(f'  Bars in COMPRESSION LONG_BIAS: {int(comp_active.sum()):,}  '
           f'({100*comp_active.mean():.1f}%)')
    print(f'  Mean P(cat): {100*np.nanmean(p_cat_arr):.2f}%')
    print(f'  Max P(cat):  {100*np.nanmax(p_cat_arr):.2f}%')
    print(f'  Mean size multiplier: {np.mean(size_mult_arr):.3f}')


if __name__ == '__main__':
    main()
