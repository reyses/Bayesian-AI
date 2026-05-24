"""Full multi-TF dissection of a single trade.

Plots ALL anchors / CRMs / HL bands around a chosen trade so we can see
the complete context. Designed to surface patterns/relations we're
missing in aggregate analysis.

Panels:
    1. Price + ALL CRMs (15s/1m/5m/15m M_close) + HL bands (15m/1h M_high/low)
       + entry/exit markers + trade P&L path overlaid
    2. Volume context: vol_mean + vol_sigma at multiple TFs
    3. z-position: z_se at 1m/5m/15m/1h all overlaid
    4. PnL curve through the trade with peak/exit annotations + filter flags

Plus a context table panel showing key V2 features AT ENTRY.

USAGE
    python tools/chart_trade_dissection.py --day 2025_10_29 --tier FADE_CALM --trade-idx 0
    python tools/chart_trade_dissection.py --day 2025_10_29 --tier FADE_CALM --pick best
    python tools/chart_trade_dissection.py --day 2025_10_29 --tier FADE_CALM --pick worst
    python tools/chart_trade_dissection.py --day 2025_10_29 --tier FADE_CALM --pick giveback
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv


def compute_anchor(tf: str, day: str, ts_5s: np.ndarray, window: int,
                    column: str):
    period_s_map = {'15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}
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


def load_v2_features(day: str, ts_5s: np.ndarray):
    """Merge V2 layer features at the 5s grid."""
    layer_dir = 'DATA/ATLAS/FEATURES_5s_v2'
    base = pd.DataFrame({'timestamp': ts_5s})
    base['timestamp'] = base['timestamp'].astype(np.int64)
    for layer in ('L2_1m', 'L2_5m', 'L2_15m', 'L2_1h',
                    'L3_1m', 'L3_5m', 'L3_15m', 'L3_1h',
                    'L1_5m', 'L1_5s'):
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
        return max(trades, key=lambda t: t.peak_pnl - t.pnl)
    return trades[int(mode)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_10_29')
    ap.add_argument('--tier', default='FADE_CALM')
    ap.add_argument('--pick', default='giveback',
                     help='best | worst | giveback | <trade_idx>')
    ap.add_argument('--window-min', type=float, default=10.0,
                     help='minutes around the trade to render')
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load trades for this tier on this day
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
        print(f'No trades for {args.tier} on {args.day}'); return
    trade = pick_trade(trades_on_day, args.pick)
    print(f'Trade selected: tier={args.tier}  '
           f'entry={datetime.fromtimestamp(int(trade.entry_ts), tz=timezone.utc):%H:%M:%S}  '
           f'exit={datetime.fromtimestamp(int(trade.exit_ts), tz=timezone.utc):%H:%M:%S}  '
           f'dir={trade.direction}  '
           f'pnl=${trade.pnl:+.0f}  peak=${trade.peak_pnl:+.0f}  '
           f'reason={trade.exit_reason}')

    df_5s = _load_5s(args.day)
    ts = df_5s['timestamp'].values.astype(np.int64)
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    close = df_5s['close'].values

    # CRMs
    M_15s, _ = compute_anchor('15s', args.day, ts, 12, 'close')
    M_1m, _  = compute_anchor('1m',  args.day, ts, 15, 'close')
    M_5m, _  = compute_anchor('5m',  args.day, ts, 9, 'close')
    M_15m, _ = compute_anchor('15m', args.day, ts, 12, 'close')
    M_1h, _  = compute_anchor('1h',  args.day, ts, 12, 'close')

    # HL bands
    Mh_15m, Sh_15m = compute_anchor('15m', args.day, ts, 12, 'high')
    Ml_15m, Sl_15m = compute_anchor('15m', args.day, ts, 12, 'low')
    Mh_1h,  Sh_1h  = compute_anchor('1h',  args.day, ts, 12, 'high')
    Ml_1h,  Sl_1h  = compute_anchor('1h',  args.day, ts, 12, 'low')

    # V2 features
    feats = load_v2_features(args.day, ts)

    # Determine window
    window_s = int(args.window_min * 60)
    win_start_ts = int(trade.entry_ts) - window_s
    win_end_ts   = int(trade.exit_ts)  + window_s
    i_start = max(0, int(np.searchsorted(ts, win_start_ts)))
    i_end   = min(len(ts)-1, int(np.searchsorted(ts, win_end_ts)))
    sl = slice(i_start, i_end + 1)

    # PnL path
    seg_close = close[sl]
    if trade.direction == 'long':
        pnl_path = (seg_close - trade.entry_price) * 2.0
    else:
        pnl_path = (trade.entry_price - seg_close) * 2.0
    seg_dt = [dt[i] for i in range(i_start, i_end + 1)]

    entry_dt = datetime.fromtimestamp(int(trade.entry_ts), tz=timezone.utc)
    exit_dt = datetime.fromtimestamp(int(trade.exit_ts), tz=timezone.utc)

    # ===== RENDER 4-panel =====
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.5, 1.5, 1.5], hspace=0.18)
    ax1 = fig.add_subplot(gs[0])   # price + all CRMs + HL bands
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # volume context
    ax3 = fig.add_subplot(gs[2], sharex=ax1)   # z-positions
    ax4 = fig.add_subplot(gs[3], sharex=ax1)   # PnL path

    # ─── PANEL 1: price + all CRMs + HL bands ───
    ax = ax1
    # HL bands 1h
    ax.plot(seg_dt, Mh_1h[sl],  color='#43A047', lw=1.2, alpha=0.85,
              label='1h M_high')
    ax.plot(seg_dt, Mh_1h[sl] + 2*Sh_1h[sl],  color='#1B5E20', lw=0.7, ls='--', alpha=0.7,
              label='1h M_high +2sigma')
    ax.plot(seg_dt, Mh_1h[sl] + 3*Sh_1h[sl],  color='#1B5E20', lw=0.7, ls=':', alpha=0.7)
    ax.plot(seg_dt, Ml_1h[sl],  color='#E53935', lw=1.2, alpha=0.85,
              label='1h M_low')
    ax.plot(seg_dt, Ml_1h[sl] - 2*Sl_1h[sl],  color='#B71C1C', lw=0.7, ls='--', alpha=0.7,
              label='1h M_low -2sigma')
    ax.plot(seg_dt, Ml_1h[sl] - 3*Sl_1h[sl],  color='#B71C1C', lw=0.7, ls=':', alpha=0.7)
    # HL bands 15m (light)
    ax.plot(seg_dt, Mh_15m[sl], color='#66BB6A', lw=0.6, alpha=0.5, ls=':')
    ax.plot(seg_dt, Mh_15m[sl] + 2*Sh_15m[sl], color='#66BB6A', lw=0.4, alpha=0.4, ls='--')
    ax.plot(seg_dt, Ml_15m[sl], color='#EF5350', lw=0.6, alpha=0.5, ls=':')
    ax.plot(seg_dt, Ml_15m[sl] - 2*Sl_15m[sl], color='#EF5350', lw=0.4, alpha=0.4, ls='--')
    # CRMs slow to fast
    if M_1h  is not None: ax.plot(seg_dt, M_1h[sl],  color='#7E57C2', lw=1.0, alpha=0.7, label='1h M_close')
    if M_15m is not None: ax.plot(seg_dt, M_15m[sl], color='#1E88E5', lw=1.4, alpha=0.85, label='15m M_close')
    if M_5m  is not None: ax.plot(seg_dt, M_5m[sl],  color='#FB8C00', lw=1.4, alpha=0.85, label='5m M_close')
    if M_1m  is not None: ax.plot(seg_dt, M_1m[sl],  color='#FFB300', lw=1.0, alpha=0.85, label='1m M_close')
    if M_15s is not None: ax.plot(seg_dt, M_15s[sl], color='#FFEB3B', lw=0.7, alpha=0.85, label='15s M_close')
    ax.plot(seg_dt, seg_close, color='black', lw=0.6, alpha=0.95, label='5s close')

    # entry/exit markers + peak marker
    ax.axvline(entry_dt, color='#1976D2', lw=1.5, alpha=0.85)
    ax.axvline(exit_dt,  color='#7B1FA2', lw=1.5, alpha=0.85)
    ax.scatter([entry_dt], [trade.entry_price],
                color='#43A047' if trade.direction=='long' else '#E53935',
                s=180, marker='^' if trade.direction=='long' else 'v',
                edgecolor='black', linewidth=1.5, zorder=10,
                label=f'ENTRY {trade.direction}')
    ax.scatter([exit_dt],  [trade.exit_price],
                color='#3F51B5', s=140, marker='X',
                edgecolor='black', linewidth=1.5, zorder=10,
                label=f'EXIT ({trade.exit_reason})')
    # peak P&L bar marker
    peak_idx_in_seg = int(np.argmax(pnl_path))
    peak_dt = seg_dt[peak_idx_in_seg]
    peak_price = seg_close[peak_idx_in_seg]
    ax.scatter([peak_dt], [peak_price],
                color='#FFB300', s=160, marker='*',
                edgecolor='black', linewidth=1.0, zorder=11,
                label=f'PEAK P&L ${trade.peak_pnl:+.0f}')

    ax.set_title(
        f'{args.day}  {args.tier}  TRADE DISSECTION\n'
        f'entry {entry_dt:%H:%M:%S} @ ${trade.entry_price:.2f} {trade.direction}    '
        f'exit {exit_dt:%H:%M:%S} @ ${trade.exit_price:.2f} ({trade.exit_reason})    '
        f'P&L ${trade.pnl:+.0f}    peak ${trade.peak_pnl:+.0f}    '
        f'giveback ${trade.peak_pnl - trade.pnl:+.0f}    '
        f'held {(trade.exit_ts - trade.entry_ts)/60:.1f}min',
        fontsize=10)
    ax.set_ylabel('price'); ax.legend(loc='best', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.20)

    # ─── PANEL 2: VOLUME context ───
    ax = ax2
    for col, color, lbl in [
        ('L2_1m_vol_mean_15',   '#9C27B0', '1m vol_mean'),
        ('L2_5m_vol_mean_9',    '#673AB7', '5m vol_mean'),
        ('L2_15m_vol_mean_12',  '#3F51B5', '15m vol_mean'),
        ('L2_1h_vol_mean_12',   '#1976D2', '1h vol_mean'),
    ]:
        if col in feats.columns:
            ax.plot(seg_dt, feats[col].values[sl], color=color, lw=1.0, alpha=0.85, label=lbl)
    ax.axvline(entry_dt, color='#1976D2', lw=1.0, alpha=0.6)
    ax.axvline(exit_dt,  color='#7B1FA2', lw=1.0, alpha=0.6)
    ax.set_ylabel('vol_mean'); ax.legend(loc='best', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.2)
    ax.set_title('VOLUME CONTEXT — vol_mean at multiple TFs', fontsize=10)

    # ─── PANEL 3: Z-POSITIONS ───
    ax = ax3
    for col, color, lbl in [
        ('L3_1m_z_se_15',   '#FFB300', '1m z_se'),
        ('L3_5m_z_se_9',    '#FB8C00', '5m z_se'),
        ('L3_15m_z_se_12',  '#1E88E5', '15m z_se'),
        ('L3_1h_z_se_12',   '#7E57C2', '1h z_se'),
    ]:
        if col in feats.columns:
            ax.plot(seg_dt, feats[col].values[sl], color=color, lw=1.2, alpha=0.85, label=lbl)
    ax.axhline(0,   color='black', lw=0.5, alpha=0.5)
    ax.axhline(2,   color='gray',  lw=0.4, alpha=0.4, ls='--')
    ax.axhline(-2,  color='gray',  lw=0.4, alpha=0.4, ls='--')
    ax.axhline(3,   color='red',   lw=0.4, alpha=0.4, ls='--')
    ax.axhline(-3,  color='red',   lw=0.4, alpha=0.4, ls='--')
    ax.axvline(entry_dt, color='#1976D2', lw=1.0, alpha=0.6)
    ax.axvline(exit_dt,  color='#7B1FA2', lw=1.0, alpha=0.6)
    ax.set_ylabel('z_se'); ax.legend(loc='best', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.2)
    ax.set_title('Z-POSITION — z_se at multiple TFs (price vs CRM, sigma units)',
                  fontsize=10)

    # ─── PANEL 4: PnL path ───
    ax = ax4
    ax.plot(seg_dt, pnl_path, color='#1E88E5', lw=1.3, alpha=0.9)
    ax.fill_between(seg_dt, 0, pnl_path,
                      where=(pnl_path >= 0), color='#43A047', alpha=0.25)
    ax.fill_between(seg_dt, 0, pnl_path,
                      where=(pnl_path < 0), color='#E53935', alpha=0.25)
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.axvline(entry_dt, color='#1976D2', lw=1.0, alpha=0.6)
    ax.axvline(exit_dt,  color='#7B1FA2', lw=1.0, alpha=0.6)
    ax.scatter([peak_dt], [trade.peak_pnl], color='#FFB300', s=120, marker='*',
                 edgecolor='black', zorder=10)
    ax.text(peak_dt, trade.peak_pnl, f' peak ${trade.peak_pnl:+.0f}',
              fontsize=8, va='bottom')
    ax.text(exit_dt, trade.pnl, f' exit ${trade.pnl:+.0f}',
              fontsize=8, va='top')
    ax.set_ylabel('P&L ($)'); ax.set_xlabel('time (UTC)')
    ax.grid(True, alpha=0.2)
    ax.set_title('TRADE P&L PATH bar-by-bar', fontsize=10)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))

    plt.tight_layout()
    out_path = os.path.join(args.out_dir,
                              f'trade_dissection_{args.tier}_{args.day}_{args.pick}_{int(trade.entry_ts)}.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_path}')

    # Print context table
    i_entry = int(np.searchsorted(ts, trade.entry_ts))
    print(f'\n=== CONTEXT AT ENTRY (5s bar {i_entry}) ===')
    for col in ['L2_15m_price_velocity_12','L2_15m_vol_sigma_12',
                 'L2_15m_vol_velocity_12','L3_15m_hurst_12',
                 'L3_15m_swing_noise_12','L3_15m_z_se_12',
                 'L3_1h_z_se_12','L2_5m_price_velocity_9',
                 'L2_1m_price_velocity_15','L2_5m_vol_sigma_9']:
        if col in feats.columns:
            v = feats[col].iloc[i_entry]
            if pd.notna(v): print(f'  {col:<32s}: {float(v):+.4f}')


if __name__ == '__main__':
    main()
