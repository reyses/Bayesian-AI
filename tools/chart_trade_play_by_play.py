"""Bar-by-bar play-by-play of a single trade's entry trigger + progression.

For NMP-style fade tiers (FADE_CALM, FADE_MOMENTUM, ...):
    Entry conditions:
        |L3_1m_z_se_15|        >= z_thr (default 1.8)
        L3_1m_reversion_prob_15 >= r_thr (default 0.55)
        |L2_1m_price_velocity_15| < calm_vel (FADE_CALM only)

Tracks WHICH bars qualified vs WHICH bar actually fired, plus the full
progression of features through the trade.

Panels:
    1. Price + close-TF CRMs (15s/1m/5m/15m)
    2. 1m z_se (entry trigger #1) with +/-1.8 thresholds + entry-bar marker
    3. 1m reversion_prob (entry trigger #2) with 0.55 threshold
    4. 1m price_velocity (calm filter) with +/-5.0 thresholds
    5. PnL through the trade

Annotations:
    - Where each condition is met (colored zones)
    - The exact bar where the trade fired (vertical line)
    - The exact bar where the trade exited (vertical line)
    - Peak P&L marker

USAGE
    python tools/chart_trade_play_by_play.py --day 2025_10_29 --tier FADE_CALM --pick best
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv


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


def load_v2(day: str, ts_5s: np.ndarray):
    layer_dir = 'DATA/ATLAS/FEATURES_5s_v2'
    base = pd.DataFrame({'timestamp': ts_5s})
    base['timestamp'] = base['timestamp'].astype(np.int64)
    for layer in ('L2_1m', 'L3_1m', 'L2_5m', 'L3_5m', 'L2_15m', 'L3_15m'):
        path = f'{layer_dir}/{layer}/{day}.parquet'
        if not os.path.exists(path): continue
        f_df = pd.read_parquet(path)
        f_df['timestamp'] = f_df['timestamp'].astype(np.int64)
        base = pd.merge_asof(base.sort_values('timestamp'),
                                f_df.sort_values('timestamp'),
                                on='timestamp', direction='backward')
    return base


def pick_trade(trades, mode: str):
    if mode == 'best':
        return max(trades, key=lambda t: t.pnl)
    if mode == 'worst':
        return min(trades, key=lambda t: t.pnl)
    if mode == 'giveback':
        valid = [t for t in trades if t.peak_pnl > 10]
        if not valid: valid = trades
        return max(valid, key=lambda t: t.peak_pnl - t.pnl)
    return trades[int(mode)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_10_29')
    ap.add_argument('--tier', default='FADE_CALM')
    ap.add_argument('--pick', default='best')
    ap.add_argument('--pre-window-min', type=float, default=10.0,
                     help='Minutes BEFORE entry to show')
    ap.add_argument('--post-window-min', type=float, default=5.0,
                     help='Minutes AFTER exit to show')
    ap.add_argument('--z-thr', type=float, default=1.8)
    ap.add_argument('--r-thr', type=float, default=0.55)
    ap.add_argument('--calm-vel', type=float, default=5.0)
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades_on_day = []
    for split in ('IS', 'OOS'):
        path = f'training_iso_v2/output/{split.lower()}_{args.tier}.pkl'
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            trades = pickle.load(f)
        for t in trades:
            if t.entry_day == args.day:
                trades_on_day.append(t)
    if not trades_on_day:
        print('No trades'); return
    trade = pick_trade(trades_on_day, args.pick)
    entry_dt = datetime.fromtimestamp(int(trade.entry_ts), tz=timezone.utc)
    exit_dt = datetime.fromtimestamp(int(trade.exit_ts), tz=timezone.utc)
    print(f'Trade: {entry_dt:%H:%M:%S}  {trade.direction}  '
           f'pnl ${trade.pnl:+.0f}  peak ${trade.peak_pnl:+.0f}  '
           f'reason {trade.exit_reason}')

    df_5s = _load_5s(args.day)
    ts = df_5s['timestamp'].values.astype(np.int64)
    close = df_5s['close'].values
    feats = load_v2(args.day, ts)

    # CRMs
    M_15s, _ = compute_anchor('15s', args.day, ts, 12, 'close')
    M_1m, _  = compute_anchor('1m',  args.day, ts, 15, 'close')
    M_5m, _  = compute_anchor('5m',  args.day, ts, 9, 'close')
    M_15m, _ = compute_anchor('15m', args.day, ts, 12, 'close')

    # Window
    pre_s  = int(args.pre_window_min * 60)
    post_s = int(args.post_window_min * 60)
    win_start = int(trade.entry_ts) - pre_s
    win_end   = int(trade.exit_ts)  + post_s
    i_start = max(0, int(np.searchsorted(ts, win_start)))
    i_end   = min(len(ts)-1, int(np.searchsorted(ts, win_end)))
    sl = slice(i_start, i_end + 1)
    seg_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts[sl]]
    seg_close = close[sl]

    # Entry-trigger features
    z_1m  = feats['L3_1m_z_se_15'].values[sl] if 'L3_1m_z_se_15' in feats.columns else None
    rp_1m = feats['L3_1m_reversion_prob_15'].values[sl] if 'L3_1m_reversion_prob_15' in feats.columns else None
    v_1m  = feats['L2_1m_price_velocity_15'].values[sl] if 'L2_1m_price_velocity_15' in feats.columns else None

    # Qualified bars: where ALL conditions are met
    qualified = np.zeros(len(seg_dt), dtype=bool)
    if z_1m is not None and rp_1m is not None and v_1m is not None:
        qualified = ((np.abs(z_1m) >= args.z_thr) &
                       (rp_1m >= args.r_thr) &
                       (np.abs(v_1m) < args.calm_vel))

    # PnL path
    if trade.direction == 'long':
        pnl_path = (seg_close - trade.entry_price) * 2.0
    else:
        pnl_path = (trade.entry_price - seg_close) * 2.0

    # ===== RENDER 5-panel =====
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.5, 1, 1, 1, 1], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])   # price + CRMs
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # 1m z_se
    ax3 = fig.add_subplot(gs[2], sharex=ax1)   # 1m reversion_prob
    ax4 = fig.add_subplot(gs[3], sharex=ax1)   # 1m velocity (calm filter)
    ax5 = fig.add_subplot(gs[4], sharex=ax1)   # PnL

    # ─── PANEL 1: Price + CRMs ───
    ax = ax1
    if M_15m is not None: ax.plot(seg_dt, M_15m[sl], color='#1E88E5', lw=1.5, alpha=0.85, label='15m M_close')
    if M_5m is not None:  ax.plot(seg_dt, M_5m[sl],  color='#FB8C00', lw=1.4, alpha=0.85, label='5m M_close')
    if M_1m is not None:  ax.plot(seg_dt, M_1m[sl],  color='#FFB300', lw=1.0, alpha=0.85, label='1m M_close')
    if M_15s is not None: ax.plot(seg_dt, M_15s[sl], color='#FFEB3B', lw=0.7, alpha=0.85, label='15s M_close')
    ax.plot(seg_dt, seg_close, color='black', lw=0.7, alpha=0.95, label='5s close')

    # Shade qualified-entry bars
    if qualified.any():
        in_seg = []
        i = 0
        while i < len(qualified):
            if qualified[i]:
                j = i
                while j < len(qualified) and qualified[j]: j += 1
                in_seg.append((i, j-1)); i = j
            else: i += 1
        for s, e in in_seg:
            ax.axvspan(seg_dt[s], seg_dt[e], color='#FFEB3B', alpha=0.18, zorder=0,
                          label='ALL entry conds met' if s == in_seg[0][0] else None)

    ax.axvline(entry_dt, color='#1976D2', lw=2.2, alpha=0.95, label='ENTRY')
    ax.axvline(exit_dt, color='#7B1FA2', lw=2.2, alpha=0.95, label='EXIT')
    ax.scatter([entry_dt], [trade.entry_price],
                 color='#43A047' if trade.direction=='long' else '#E53935',
                 s=200, marker='^' if trade.direction=='long' else 'v',
                 edgecolor='black', linewidth=1.5, zorder=10)
    ax.scatter([exit_dt], [trade.exit_price],
                 color='#3F51B5', s=160, marker='X',
                 edgecolor='black', linewidth=1.5, zorder=10)
    # peak marker
    peak_i = int(np.argmax(pnl_path))
    ax.scatter([seg_dt[peak_i]], [seg_close[peak_i]],
                 color='#FFB300', s=180, marker='*',
                 edgecolor='black', zorder=11)

    ax.set_title(
        f'{args.day}  {args.tier}  TRADE PLAY-BY-PLAY\n'
        f'entry {entry_dt:%H:%M:%S} {trade.direction}  '
        f'exit {exit_dt:%H:%M:%S} ({trade.exit_reason})  '
        f'P&L ${trade.pnl:+.0f}  peak ${trade.peak_pnl:+.0f}  '
        f'window: {args.pre_window_min}min before to {args.post_window_min}min after exit',
        fontsize=11)
    ax.set_ylabel('price'); ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.20)

    # ─── PANEL 2: 1m z_se (entry trigger 1) ───
    ax = ax2
    if z_1m is not None:
        ax.plot(seg_dt, z_1m, color='#1976D2', lw=1.2, alpha=0.9)
        ax.axhline(+args.z_thr, color='red', lw=0.7, ls='--',
                     label=f'+{args.z_thr} (SHORT trigger)')
        ax.axhline(-args.z_thr, color='red', lw=0.7, ls='--',
                     label=f'-{args.z_thr} (LONG trigger)')
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.fill_between(seg_dt, z_1m, +args.z_thr,
                          where=(z_1m >= +args.z_thr), color='#E53935', alpha=0.25)
        ax.fill_between(seg_dt, z_1m, -args.z_thr,
                          where=(z_1m <= -args.z_thr), color='#43A047', alpha=0.25)
    ax.axvline(entry_dt, color='#1976D2', lw=1.5)
    ax.axvline(exit_dt, color='#7B1FA2', lw=1.5)
    ax.set_title('Entry trigger #1: L3_1m_z_se_15 (|z| >= 1.8 required)', fontsize=10)
    ax.set_ylabel('1m z_se'); ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    # ─── PANEL 3: reversion_prob (entry trigger 2) ───
    ax = ax3
    if rp_1m is not None:
        ax.plot(seg_dt, rp_1m, color='#7E57C2', lw=1.2, alpha=0.9)
        ax.axhline(args.r_thr, color='red', lw=0.7, ls='--',
                     label=f'{args.r_thr} threshold')
        ax.fill_between(seg_dt, rp_1m, args.r_thr,
                          where=(rp_1m >= args.r_thr), color='#43A047', alpha=0.25)
    ax.axvline(entry_dt, color='#1976D2', lw=1.5)
    ax.axvline(exit_dt, color='#7B1FA2', lw=1.5)
    ax.set_title('Entry trigger #2: L3_1m_reversion_prob_15 (>= 0.55 required)',
                   fontsize=10)
    ax.set_ylabel('rprob'); ax.legend(loc='best', fontsize=7); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ─── PANEL 4: velocity (calm filter) ───
    ax = ax4
    if v_1m is not None:
        ax.plot(seg_dt, v_1m, color='#00897B', lw=1.2, alpha=0.9)
        ax.axhline(+args.calm_vel, color='red', lw=0.7, ls='--',
                     label=f'+{args.calm_vel} calm cap')
        ax.axhline(-args.calm_vel, color='red', lw=0.7, ls='--',
                     label=f'-{args.calm_vel} calm cap')
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.fill_between(seg_dt, v_1m, args.calm_vel,
                          where=(np.abs(v_1m) >= args.calm_vel), color='#E53935',
                          alpha=0.25, label='too fast (rejected)')
    ax.axvline(entry_dt, color='#1976D2', lw=1.5)
    ax.axvline(exit_dt, color='#7B1FA2', lw=1.5)
    ax.set_title(f'Calm filter: L2_1m_price_velocity_15 (|v| < {args.calm_vel} required)',
                   fontsize=10)
    ax.set_ylabel('1m vel'); ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

    # ─── PANEL 5: PnL ───
    ax = ax5
    ax.plot(seg_dt, pnl_path, color='#1E88E5', lw=1.3, alpha=0.9)
    ax.fill_between(seg_dt, 0, pnl_path,
                       where=(pnl_path >= 0), color='#43A047', alpha=0.25)
    ax.fill_between(seg_dt, 0, pnl_path,
                       where=(pnl_path < 0), color='#E53935', alpha=0.25)
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.axvline(entry_dt, color='#1976D2', lw=1.5)
    ax.axvline(exit_dt, color='#7B1FA2', lw=1.5)
    ax.scatter([seg_dt[peak_i]], [trade.peak_pnl], color='#FFB300', s=150, marker='*',
                 edgecolor='black', zorder=10)
    ax.set_title('Trade P&L bar-by-bar', fontsize=10)
    ax.set_ylabel('P&L ($)'); ax.set_xlabel('time (UTC)')
    ax.grid(True, alpha=0.3)

    # X-axis formatting (5-sec resolution if zoomed enough)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    span_s = (seg_dt[-1] - seg_dt[0]).total_seconds()
    if span_s <= 1200:
        ax5.xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0, 60, 30)))
    elif span_s <= 3600:
        ax5.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 2)))
    else:
        ax5.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))

    plt.tight_layout()
    out = os.path.join(args.out_dir,
                          f'play_by_play_{args.tier}_{args.day}_{args.pick}_{int(trade.entry_ts)}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out}')

    # Detailed bar-by-bar log around entry
    print(f'\n=== BAR-BY-BAR LOG (around entry) ===')
    i_entry = int(np.searchsorted(ts, trade.entry_ts))
    i_show_start = max(0, i_entry - 12)  # 60s before entry
    i_show_end   = min(len(ts), i_entry + 6)  # 30s after entry
    print(f"  bar# t            z_1m   r_1m   v_1m   close   qualif  status")
    for i in range(i_show_start, i_show_end):
        dt_i = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc)
        z = z_1m[i - i_start] if i - i_start < len(z_1m) and z_1m is not None else np.nan
        r = rp_1m[i - i_start] if i - i_start < len(rp_1m) and rp_1m is not None else np.nan
        v = v_1m[i - i_start] if i - i_start < len(v_1m) and v_1m is not None else np.nan
        q = (abs(z) >= args.z_thr) and (r >= args.r_thr) and (abs(v) < args.calm_vel) \
              if all(pd.notna([z,r,v])) else False
        status = '<- ENTRY' if i == i_entry else ('Q' if q else '')
        print(f"  {i:>5d} {dt_i:%H:%M:%S}  {z:+5.2f}  {r:.3f}  {v:+6.2f}  {close[i]:.2f}  "
               f"{'YES' if q else ''} {status}")


if __name__ == '__main__':
    main()
