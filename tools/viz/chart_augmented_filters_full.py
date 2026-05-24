"""Augmented filters chart — FULL version with multiple CRMs + multiple HL bands.

Improvements over chart_augmented_filters_on_day.py:
    - Multiple CRM lines: 5s, 1m, 5m, 15m (multi-scale oscillators)
    - Multiple HL RM envelopes: 15m, 1h (multi-scale slow rails)
    - Fixed Y-axis range based on session statistics (prevents auto-scale
      from making small moves look catastrophic)
    - Optional Y segmentation: split top panel into 2 sub-panels (slow rails
      view + fast oscillators view) for clearer reading

USAGE
    python tools/chart_augmented_filters_full.py --day 2025_10_29
    python tools/chart_augmented_filters_full.py --day 2025_10_29 --y-pad 30
    python tools/chart_augmented_filters_full.py --day 2025_10_29 --split-panels
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv
from training.filters.bayes_filters import (
    tod_risk_size_multiplier, cat_harvest_signal,
    _load_tod_dow_p_cat, CompressionBounce,
)


TIER_COLORS = {
    'FADE_CALM':       '#1976D2',
    'FADE_MOMENTUM':   '#0288D1',
    'FADE_AGAINST':    '#039BE5',
    'FADE_AT_BAND':    '#00ACC1',
    'RIDE_CALM':       '#E64A19',
    'RIDE_MOMENTUM':   '#F4511E',
    'RIDE_AGAINST':    '#FF7043',
    'KILL_SHOT':       '#D32F2F',
    'CASCADE':         '#7B1FA2',
    'FREIGHT_TRAIN':   '#5D4037',
    'NMP_FADE_RAW':    '#1565C0',
    'NMP_RIDE_RAW':    '#BF360C',
}


def load_tier_trades_on_day(tier: str, day: str):
    """Load trades for this tier on this day from iso pickles."""
    rows = []
    for split in ('IS', 'OOS'):
        path = f'training_iso_v2/output/{split.lower()}_{tier}.pkl'
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            trades = pickle.load(f)
        for t in trades:
            if t.entry_day == day:
                rows.append({
                    'tier': tier, 'direction': t.direction,
                    'entry_ts': int(t.entry_ts), 'exit_ts': int(t.exit_ts),
                    'entry_price': t.entry_price, 'exit_price': t.exit_price,
                    'pnl': t.pnl, 'exit_reason': t.exit_reason,
                })
    return rows


def compute_anchor(tf: str, day: str, ts_5s: np.ndarray, window: int, column: str):
    period_s_map = {'15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600}
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
    ap.add_argument('--y-pad', type=float, default=20.0,
                     help='Extra Y padding beyond session min/max in points')
    ap.add_argument('--y-mode', default='session',
                     choices=['session', 'auto', 'fixed_atr'])
    ap.add_argument('--split-panels', action='store_true',
                     help='Split price panel into slow-rails + fast-oscillators')
    ap.add_argument('--tiers', default='ALL',
                     help='Comma-separated tier names or ALL/NONE (default ALL)')
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_5s = _load_5s(args.day)
    if df_5s.empty: print('no data'); return
    ts = df_5s['timestamp'].values.astype(np.int64)
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    close = df_5s['close'].values.astype(np.float64)

    # MULTIPLE CRMs — slow to fast
    M_15m, _      = compute_anchor('15m', args.day, ts, 12, 'close')
    M_5m, _       = compute_anchor('5m',  args.day, ts, 9,  'close')
    M_1m, _       = compute_anchor('1m',  args.day, ts, 15, 'close')
    M_15s, _      = compute_anchor('15s', args.day, ts, 12, 'close')  # very fast
    # MULTIPLE HL bands — slow to medium
    Mh_1h, Sh_1h  = compute_anchor('1h',  args.day, ts, 12, 'high')
    Ml_1h, Sl_1h  = compute_anchor('1h',  args.day, ts, 12, 'low')
    Mh_15m, Sh_15m = compute_anchor('15m', args.day, ts, 12, 'high')
    Ml_15m, Sl_15m = compute_anchor('15m', args.day, ts, 12, 'low')

    # Load 15m vol_sigma for compression
    path_l215m = f'DATA/ATLAS/FEATURES_5s_v2/L2_15m/{args.day}.parquet'
    vol_sigma_15m = None
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

    cb = CompressionBounce(); cb.reset(args.day)
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
            comp_active[i] = (cb.update(vol_sigma_15m[i]) == 'LONG_BIAS')

    # Y range: session-based — daily min/max +/- pad
    if args.y_mode == 'session':
        y_lo = float(np.nanmin(close)) - args.y_pad
        y_hi = float(np.nanmax(close)) + args.y_pad
    elif args.y_mode == 'fixed_atr':
        midpoint = (float(np.nanmin(close)) + float(np.nanmax(close))) / 2.0
        atr = float(np.nanmax(close)) - float(np.nanmin(close))
        y_lo = midpoint - atr * 0.8
        y_hi = midpoint + atr * 0.8
    else:
        y_lo = None; y_hi = None

    # ===== RENDER =====
    if args.split_panels:
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 2.5, 1, 1], hspace=0.12)
        ax_slow = fig.add_subplot(gs[0])
        ax_fast = fig.add_subplot(gs[1], sharex=ax_slow)
        ax_pcat = fig.add_subplot(gs[2], sharex=ax_slow)
        ax_size = fig.add_subplot(gs[3], sharex=ax_slow)
        price_axes = [ax_slow, ax_fast]
    else:
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.12)
        ax_slow = fig.add_subplot(gs[0])
        ax_pcat = fig.add_subplot(gs[1], sharex=ax_slow)
        ax_size = fig.add_subplot(gs[2], sharex=ax_slow)
        price_axes = [ax_slow]

    # ─── SLOW PANEL ─── 1h HL bands + 15m HL bands + 15m M_close + slow context
    ax = ax_slow
    # 1h HL with ±2sigma and ±3sigma
    ax.plot(dt, Mh_1h, color='#43A047', lw=1.4, alpha=0.9, label='1h M_high')
    ax.fill_between(dt, Mh_1h, Mh_1h+2*Sh_1h, color='#43A047', alpha=0.05)
    ax.fill_between(dt, Mh_1h+2*Sh_1h, Mh_1h+3*Sh_1h, color='#43A047', alpha=0.15,
                      label='1h M_high +2sigmato+3sigma rally zone')
    ax.plot(dt, Mh_1h+3*Sh_1h, color='#1B5E20', lw=0.7, ls='--', alpha=0.7)
    ax.plot(dt, Ml_1h, color='#E53935', lw=1.4, alpha=0.9, label='1h M_low')
    ax.fill_between(dt, Ml_1h-2*Sl_1h, Ml_1h, color='#E53935', alpha=0.05)
    ax.fill_between(dt, Ml_1h-3*Sl_1h, Ml_1h-2*Sl_1h, color='#E53935', alpha=0.15,
                      label='1h M_low −2sigmato−3sigma crash zone')
    ax.plot(dt, Ml_1h-3*Sl_1h, color='#B71C1C', lw=0.7, ls='--', alpha=0.7)
    # 15m HL ±2sigma (lighter)
    ax.plot(dt, Mh_15m, color='#66BB6A', lw=0.7, alpha=0.6, ls=':', label='15m M_high')
    ax.plot(dt, Mh_15m+2*Sh_15m, color='#66BB6A', lw=0.5, alpha=0.4, ls='--')
    ax.plot(dt, Ml_15m, color='#EF5350', lw=0.7, alpha=0.6, ls=':', label='15m M_low')
    ax.plot(dt, Ml_15m-2*Sl_15m, color='#EF5350', lw=0.5, alpha=0.4, ls='--')
    # 15m M_close
    ax.plot(dt, M_15m, color='#1E88E5', lw=1.6, alpha=0.95, label='15m M_close')
    # 5s close
    ax.plot(dt, close, color='black', lw=0.45, alpha=0.85, label='5s close')

    # Filter overlays
    in_comp = []; i = 0
    while i < len(comp_active):
        if comp_active[i]:
            j = i
            while j < len(comp_active) and comp_active[j]: j += 1
            in_comp.append((i, j-1)); i = j
        else: i += 1
    for s, e in in_comp:
        ax.axvspan(dt[s], dt[e], color='#00C853', alpha=0.16,
                    label='COMPRESSION LONG_BIAS' if s == in_comp[0][0] else None,
                    zorder=0)
    in_harv = []; i = 0
    while i < len(cat_harvest_active):
        if cat_harvest_active[i]:
            j = i
            while j < len(cat_harvest_active) and cat_harvest_active[j]: j += 1
            in_harv.append((i, j-1)); i = j
        else: i += 1
    for s, e in in_harv:
        ax.axvspan(dt[s], dt[e], color='#FF1744', alpha=0.18, hatch='///',
                    label='CAT HARVEST PRE_SHORT' if s == in_harv[0][0] else None,
                    zorder=0)

    # ─── TRADE OVERLAY ───
    if args.tiers.upper() == 'ALL':
        tier_list = list(TIER_COLORS.keys())
    elif args.tiers.upper() == 'NONE':
        tier_list = []
    else:
        tier_list = [t.strip() for t in args.tiers.split(',')]
    all_trades = []
    for tier in tier_list:
        all_trades.extend(load_tier_trades_on_day(tier, args.day))
    tier_summary = {}
    if all_trades:
        for t in all_trades:
            entry_dt = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc)
            col = TIER_COLORS.get(t['tier'], '#666')
            marker = '^' if t['direction'] == 'long' else 'v'
            ax.scatter([entry_dt], [t['entry_price']], color=col, s=70,
                         marker=marker, edgecolor='black', linewidth=0.4, zorder=6)
            # Filled green if win, hollow red if loss
            pnl = t['pnl']
            exit_col = '#43A047' if pnl > 0 else '#E53935' if pnl < 0 else '#9E9E9E'
            ax.scatter([exit_dt], [t['exit_price']], color=exit_col, s=55,
                         marker='X', edgecolor='black', linewidth=0.4, zorder=6)
            # Faint line entry→exit colored by direction
            line_col = '#43A047' if t['direction'] == 'long' else '#E53935'
            ax.plot([entry_dt, exit_dt], [t['entry_price'], t['exit_price']],
                      color=line_col, lw=0.6, alpha=0.45, zorder=5)
            tier_summary.setdefault(t['tier'], {'n': 0, 'pnl': 0, 'wins': 0})
            tier_summary[t['tier']]['n'] += 1
            tier_summary[t['tier']]['pnl'] += pnl
            if pnl > 0: tier_summary[t['tier']]['wins'] += 1

    title_extra = ''
    if all_trades:
        total_pnl = sum(t['pnl'] for t in all_trades)
        total_n = len(all_trades)
        total_wins = sum(1 for t in all_trades if t['pnl'] > 0)
        title_extra = (f'\nTRADES: {total_n} ({total_wins} wins, '
                        f'{100*total_wins/total_n:.0f}% WR)  '
                        f'P&L: ${total_pnl:+.0f}')

    ax.set_title(f'{args.day} — SLOW RAILS (1h HL bands + 15m HL + 15m M_close + 5s close)\n'
                  f'Y range: {y_lo:.0f} to {y_hi:.0f} (session-based, prevents auto-scale distortion)'
                  + title_extra,
                  fontsize=11)
    ax.set_ylabel('price')
    ax.legend(loc='upper left', fontsize=7, ncol=3)
    ax.grid(True, alpha=0.20)
    if y_lo is not None: ax.set_ylim(y_lo, y_hi)

    # ─── FAST PANEL ─── (if split) — 5s close + 15s/1m/5m/15m CRMs
    if args.split_panels:
        ax = ax_fast
        ax.plot(dt, close, color='black', lw=0.55, alpha=0.95, label='5s close')
        if M_15s is not None:
            ax.plot(dt, M_15s, color='#FFEB3B', lw=0.8, alpha=0.85,
                      label='15s M_close (very fast)')
        ax.plot(dt, M_1m,  color='#FFB300', lw=1.2, alpha=0.85,
                  label='1m M_close (fast)')
        ax.plot(dt, M_5m,  color='#FB8C00', lw=1.6, alpha=0.85,
                  label='5m M_close (medium-fast)')
        ax.plot(dt, M_15m, color='#1E88E5', lw=1.4, alpha=0.7,
                  label='15m M_close (medium reference)')
        # Same filter overlays
        for s, e in in_comp:
            ax.axvspan(dt[s], dt[e], color='#00C853', alpha=0.13, zorder=0)
        for s, e in in_harv:
            ax.axvspan(dt[s], dt[e], color='#FF1744', alpha=0.13, hatch='///',
                        zorder=0)
        ax.set_title(f'FAST OSCILLATORS (5s close + 1m / 5m / 15m M_close)',
                       fontsize=11)
        ax.set_ylabel('price')
        ax.legend(loc='upper left', fontsize=8, ncol=4)
        ax.grid(True, alpha=0.20)
        if y_lo is not None: ax.set_ylim(y_lo, y_hi)

    # ─── P_CAT PANEL ───
    ax = ax_pcat
    ax.fill_between(dt, 0, p_cat_arr * 100, color='#E53935', alpha=0.6)
    ax.axhline(10, color='black', ls='--', lw=0.5, alpha=0.5, label='10% threshold')
    ax.axhline(20, color='red', ls='--', lw=0.5, alpha=0.5, label='20% peak')
    ax.set_ylabel('P(cat) %'); ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 45)

    # ─── SIZE MULT PANEL ───
    ax = ax_size
    ax.fill_between(dt, 0, size_mult_arr, color='#1E88E5', alpha=0.55)
    ax.set_ylabel('size mult'); ax.set_xlabel('time (UTC)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    if args.start_hour is not None or args.end_hour is not None:
        sh = args.start_hour if args.start_hour is not None else 0
        eh = args.end_hour if args.end_hour is not None else 24
        date_part = dt[0].date()
        x_start = datetime.combine(date_part, time(sh, 0), tzinfo=timezone.utc)
        x_end = datetime.combine(date_part, time(min(eh, 23), 59), tzinfo=timezone.utc) \
                 if eh < 24 else dt[-1]
        for a in [ax_slow, ax_pcat, ax_size]:
            a.set_xlim(x_start, x_end)
        if args.split_panels: ax_fast.set_xlim(x_start, x_end)
        suffix = f'_h{sh:02d}-h{eh:02d}'
    else:
        suffix = ''

    ax_size.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_size.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    split_tag = '_split' if args.split_panels else ''
    out_path = os.path.join(args.out_dir,
                              f'augmented_FULL_{args.day}{split_tag}{suffix}.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Chart -> {out_path}')

    print(f'\nDay summary for {args.day}:')
    print(f'  Total 5s bars: {len(ts):,}')
    print(f'  Y range: {y_lo:.2f} to {y_hi:.2f}  (range = {y_hi-y_lo:.0f} pts)')
    print(f'  CAT_HARVEST bars: {int(cat_harvest_active.sum()):,} '
           f'({100*cat_harvest_active.mean():.1f}%)')
    print(f'  COMPRESSION bars: {int(comp_active.sum()):,} '
           f'({100*comp_active.mean():.1f}%)')
    print(f'  Mean P(cat): {100*np.nanmean(p_cat_arr):.2f}%   '
           f'Max P(cat): {100*np.nanmax(p_cat_arr):.2f}%')


if __name__ == '__main__':
    main()
