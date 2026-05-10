"""High/Low REGRESSION CENTERS with σ bands in price space.

Two regression lines per TF:
    HIGH regression mean = rolling mean of bar HIGHs over N bars
    LOW  regression mean = rolling mean of bar LOWs  over N bars
Each gets its own ±1σ, ±2σ, ±3σ price bands.

The 5s close (jittery) sits between them. When close pierces past
the HIGH-regression+kσ band → the bar topped past statistical reach;
similarly when it pierces past LOW-regression−kσ.

Cleaner than plotting z-scores directly — these are price levels you
can act on.

Usage:
    python tools/day_chart_z_bands.py --day 2026_02_12 --split-time auto
    python tools/day_chart_z_bands.py --day 2026_02_12 --tf 5m
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

# TF → (window, parquet path)
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


def _fit_y_to_xlim(ax, padding=0.05):
    xmin, xmax = ax.get_xlim()
    y_min, y_max = float('inf'), float('-inf')
    for line in ax.get_lines():
        xs = line.get_xdata()
        ys = np.asarray(line.get_ydata(), dtype=np.float64)
        if len(xs) == 0: continue
        try: xs_num = mdates.date2num(xs)
        except Exception: continue
        mask = (xs_num >= xmin) & (xs_num <= xmax)
        if mask.any():
            ys_in = ys[mask]
            ys_in = ys_in[np.isfinite(ys_in)]
            if len(ys_in):
                y_min = min(y_min, float(ys_in.min()))
                y_max = max(y_max, float(ys_in.max()))
    if y_min < y_max and np.isfinite(y_min) and np.isfinite(y_max):
        pad = (y_max - y_min) * padding if y_max > y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2026_02_12')
    ap.add_argument('--tf', default='15m', choices=list(TF_CONFIG.keys()),
                          help='TF whose bar-HIGH and bar-LOW seed the two '
                                  'regression centers (default 15m). When '
                                  '--crm-tf is provided, close mean uses the '
                                  'separate TF for asymmetric mixed-regime mode.')
    ap.add_argument('--crm-tf', default=None, choices=list(TF_CONFIG.keys()),
                          help='TF for the close regression mean (M_close). '
                                  'If different from --tf, mixes regimes: '
                                  'fast close target + slow high/low boundary.')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='chart')
    ap.add_argument('--dpi', type=int, default=320)
    ap.add_argument('--figwidth', type=float, default=30)
    ap.add_argument('--figheight', type=float, default=14)
    ap.add_argument('--split-time', default=None)
    args = ap.parse_args()

    # Load 5s OHLCV (the jittery price)
    ohlcv_5s = pd.read_parquet(os.path.join(args.atlas_5s, f'{args.day}.parquet'))
    if pd.api.types.is_datetime64_any_dtype(ohlcv_5s['timestamp']):
        ohlcv_5s = ohlcv_5s.copy()
        ohlcv_5s['timestamp'] = (ohlcv_5s['timestamp'].astype('int64') // 10**9)
    ohlcv_5s = ohlcv_5s.sort_values('timestamp').reset_index(drop=True)
    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)
    oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in oh_ts]

    # Load HL-TF OHLCV (the slower trigger-boundary TF)
    hl_tf = args.tf
    base, N_hl = TF_CONFIG[hl_tf]
    tf_ohlcv = _load_ohlcv(hl_tf, args.day)
    if tf_ohlcv.empty:
        print(f'!!! No {hl_tf} OHLCV for {args.day} at {base}/'); sys.exit(1)
    tf_ohlcv['high_mean'] = tf_ohlcv['high'].rolling(N_hl, min_periods=2).mean()
    tf_ohlcv['high_sigma'] = tf_ohlcv['high'].rolling(N_hl, min_periods=2).std()
    tf_ohlcv['low_mean'] = tf_ohlcv['low'].rolling(N_hl, min_periods=2).mean()
    tf_ohlcv['low_sigma'] = tf_ohlcv['low'].rolling(N_hl, min_periods=2).std()

    # Close regression mean — same TF or a SEPARATE faster TF
    crm_tf = args.crm_tf or hl_tf
    if crm_tf == hl_tf:
        # Same TF: reuse already-loaded df
        tf_ohlcv['close_mean']  = tf_ohlcv['close'].rolling(N_hl, min_periods=2).mean()
        tf_ohlcv['close_sigma'] = tf_ohlcv['close'].rolling(N_hl, min_periods=2).std()
        crm_ohlcv = tf_ohlcv[['timestamp', 'close_mean', 'close_sigma']].copy()
    else:
        # Different TF for close mean — load it separately
        _, N_crm = TF_CONFIG[crm_tf]
        crm_ohlcv = _load_ohlcv(crm_tf, args.day)
        if crm_ohlcv.empty:
            print(f'!!! No {crm_tf} OHLCV; falling back to hl_tf for close mean')
            crm_tf = hl_tf
            tf_ohlcv['close_mean']  = tf_ohlcv['close'].rolling(N_hl, min_periods=2).mean()
            tf_ohlcv['close_sigma'] = tf_ohlcv['close'].rolling(N_hl, min_periods=2).std()
            crm_ohlcv = tf_ohlcv[['timestamp', 'close_mean', 'close_sigma']].copy()
        else:
            crm_ohlcv['close_mean']  = crm_ohlcv['close'].rolling(N_crm, min_periods=2).mean()
            crm_ohlcv['close_sigma'] = crm_ohlcv['close'].rolling(N_crm, min_periods=2).std()

    # Forward-fill HL-TF (high/low) to 5s timeline
    period_s_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    tf_ts = tf_ohlcv['timestamp'].values.astype(np.int64)
    period_hl = period_s_map[hl_tf]
    target_ts = oh_ts - period_hl
    idx_hl = np.searchsorted(tf_ts, target_ts, side='right') - 1
    idx_hl = np.clip(idx_hl, 0, len(tf_ts) - 1)
    high_mean_5s = tf_ohlcv['high_mean'].values[idx_hl]
    high_sigma_5s = tf_ohlcv['high_sigma'].values[idx_hl]
    low_mean_5s = tf_ohlcv['low_mean'].values[idx_hl]
    low_sigma_5s = tf_ohlcv['low_sigma'].values[idx_hl]

    # Forward-fill CRM-TF (close mean) to 5s timeline (separate map if mixed)
    crm_ts = crm_ohlcv['timestamp'].values.astype(np.int64)
    period_crm = period_s_map[crm_tf]
    target_crm = oh_ts - period_crm
    idx_crm = np.searchsorted(crm_ts, target_crm, side='right') - 1
    idx_crm = np.clip(idx_crm, 0, len(crm_ts) - 1)
    close_mean_5s = crm_ohlcv['close_mean'].values[idx_crm]
    close_sigma_5s = crm_ohlcv['close_sigma'].values[idx_crm]

    # Resolve split
    split_dt = None
    if args.split_time:
        if args.split_time == 'auto':
            i_max = int(np.argmax(ohlcv_5s['close'].values))
            split_dt = oh_dt[i_max]
        else:
            day_iso = args.day.replace('_', '-')
            split_dt = datetime.strptime(
                f'{day_iso} {args.split_time}:00',
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

    fig = plt.figure(figsize=(args.figwidth, args.figheight))
    if split_dt is not None:
        gs = fig.add_gridspec(1, 2, wspace=0.04)
        ax_L = fig.add_subplot(gs[0, 0])
        ax_R = fig.add_subplot(gs[0, 1])
        price_axes = [ax_L, ax_R]
    else:
        ax_L = fig.add_subplot(111)
        price_axes = [ax_L]

    # ── PLOTTING ───────────────────────────────────────────────────────
    for ax in price_axes:
        # 5s jittery price
        ax.plot(oh_dt, ohlcv_5s['close'], color='black', lw=0.6, alpha=0.85,
                     label='5s close', zorder=4)

        # Close regression mean — the center reference (TF: crm_tf)
        ax.plot(oh_dt, close_mean_5s, color='royalblue', lw=1.6,
                     linestyle='-', alpha=0.9,
                     label=f'{crm_tf} M_close (target)', zorder=6)
        # Blue ±2σ_close — primary reversion trigger
        blue_up_2 = close_mean_5s + 2 * close_sigma_5s
        blue_dn_2 = close_mean_5s - 2 * close_sigma_5s
        ax.plot(oh_dt, blue_up_2, color='royalblue', lw=1.2,
                     linestyle='--', alpha=0.95,
                     label=f'{crm_tf} M_close + 2σ_close', zorder=6)
        ax.plot(oh_dt, blue_dn_2, color='royalblue', lw=1.2,
                     linestyle='--', alpha=0.95,
                     label=f'{crm_tf} M_close − 2σ_close', zorder=6)
        # Blue ±3σ_close — outer wall
        blue_up_3 = close_mean_5s + 3 * close_sigma_5s
        blue_dn_3 = close_mean_5s - 3 * close_sigma_5s
        ax.plot(oh_dt, blue_up_3, color='royalblue', lw=1.0,
                     linestyle=':', alpha=0.75,
                     label=f'{crm_tf} M_close + 3σ_close', zorder=6)
        ax.plot(oh_dt, blue_dn_3, color='royalblue', lw=1.0,
                     linestyle=':', alpha=0.75,
                     label=f'{crm_tf} M_close − 3σ_close', zorder=6)

        # HIGH regression center — bands ONLY on the extreme (UP) side
        hm = high_mean_5s; hs = high_sigma_5s
        ax.plot(oh_dt, hm, color='tab:green', lw=1.6,
                     label=f'{args.tf} HIGH regression mean', zorder=5)
        prev_up = hm
        for k, ls, line_alpha, fill_alpha in [
                (1, '-',  0.80, 0.16),
                (2, '--', 0.65, 0.10),
                (3, ':',  0.50, 0.06)]:
            up = hm + k * hs
            ax.plot(oh_dt, up, color='tab:green', lw=0.6,
                         linestyle=ls, alpha=line_alpha,
                         label=f'HIGH +{k}σ' if ax is price_axes[0] else None,
                         zorder=4)
            ax.fill_between(oh_dt, prev_up, up, color='tab:green',
                                      alpha=fill_alpha, zorder=2)
            prev_up = up

        # LOW regression center — bands ONLY on the extreme (DOWN) side
        lm = low_mean_5s; ls_arr = low_sigma_5s
        ax.plot(oh_dt, lm, color='tab:red', lw=1.6,
                     label=f'{args.tf} LOW regression mean', zorder=5)
        prev_dn = lm
        for k, ls_style, line_alpha, fill_alpha in [
                (1, '-',  0.80, 0.16),
                (2, '--', 0.65, 0.10),
                (3, ':',  0.50, 0.06)]:
            dn = lm - k * ls_arr
            ax.plot(oh_dt, dn, color='tab:red', lw=0.6,
                         linestyle=ls_style, alpha=line_alpha,
                         label=f'LOW −{k}σ' if ax is price_axes[0] else None,
                         zorder=4)
            ax.fill_between(oh_dt, dn, prev_dn, color='tab:red',
                                      alpha=fill_alpha, zorder=2)
            prev_dn = dn

        ax.set_ylabel('price')
        ax.set_xlabel('time (UTC)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    price_axes[0].set_title(
        f'{args.day}  —  HIGH/LOW regression {hl_tf} (N={N_hl})  '
        f'+ M_close from {crm_tf}  '
        f'(mixed-regime mode' + (' ON' if crm_tf != hl_tf else ' off') + ')')
    price_axes[0].legend(loc='upper left', fontsize=9, ncol=4)

    # If split, set xlim and refit per-segment y
    if split_dt is not None:
        day_start = oh_dt[0]; day_end = oh_dt[-1]
        price_axes[0].set_xlim(day_start, split_dt)
        price_axes[1].set_xlim(split_dt, day_end)
        _fit_y_to_xlim(price_axes[0])
        _fit_y_to_xlim(price_axes[1])
        price_axes[1].yaxis.tick_right()
        price_axes[1].yaxis.set_label_position('right')
        leg = price_axes[1].get_legend()
        if leg is not None: leg.remove()

    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, f'{args.day}_z_bands.png')
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_png}')


if __name__ == '__main__':
    main()
