"""1-day chart of the OTHER V2 features (beyond regression mean/slope).

Companion to day_chart_regression_means.py. Visualizes the V2 features
we haven't seen yet, grouped by category:

    Panel 1  price reference (5s close + 5m regression mean)
    Panel 2  volume regime         (1m/5m vol_mean, vol_sigma)
    Panel 3  volatility / chop     (1m/5m price_sigma, 1m swing_noise)
    Panel 4  z-scores              (1m z_se, z_high, z_low)
    Panel 5  OU character          (1m hurst, reversion_prob)
    Panel 6  acceleration          (1m/5m price_accel_w)

Usage:
    python tools/day_chart_other_features.py --day 2026_02_12
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

from core_v2.features import load_features


# Feature columns grouped by what they measure. Multi-TF where available.

# L2 — rolling-window aggregates
VOL_MEAN = {
    '1m':   'L2_1m_vol_mean_15',  '5m':   'L2_5m_vol_mean_9',
    '15m':  'L2_15m_vol_mean_12', '1h':   'L2_1h_vol_mean_12',
}
VOL_SIGMA = {
    '1m':   'L2_1m_vol_sigma_15',  '5m':   'L2_5m_vol_sigma_9',
    '15m':  'L2_15m_vol_sigma_12', '1h':   'L2_1h_vol_sigma_12',
}
PRICE_SIGMA = {
    '1m':   'L2_1m_price_sigma_15',  '5m':   'L2_5m_price_sigma_9',
    '15m':  'L2_15m_price_sigma_12', '1h':   'L2_1h_price_sigma_12',
}
PRICE_VEL = {
    '1m':  'L2_1m_price_velocity_15',  '5m':  'L2_5m_price_velocity_9',
    '15m': 'L2_15m_price_velocity_12', '1h':  'L2_1h_price_velocity_12',
}
PRICE_ACCEL = {
    '1m':  'L2_1m_price_accel_15',  '5m':  'L2_5m_price_accel_9',
    '15m': 'L2_15m_price_accel_12', '1h':  'L2_1h_price_accel_12',
}
VOL_VEL = {
    '1m':  'L2_1m_vol_velocity_15',  '5m':  'L2_5m_vol_velocity_9',
    '15m': 'L2_15m_vol_velocity_12', '1h':  'L2_1h_vol_velocity_12',
}
VOL_ACCEL = {
    '1m':  'L2_1m_vol_accel_15',  '5m':  'L2_5m_vol_accel_9',
    '15m': 'L2_15m_vol_accel_12', '1h':  'L2_1h_vol_accel_12',
}

# L3 — per-TF derived stats
SWING_NOISE = {
    '1m':   'L3_1m_swing_noise_15',  '5m':   'L3_5m_swing_noise_9',
    '15m':  'L3_15m_swing_noise_12', '1h':   'L3_1h_swing_noise_12',
}
Z_SE = {
    '1m':   'L3_1m_z_se_15',  '5m':   'L3_5m_z_se_9',
    '15m':  'L3_15m_z_se_12', '1h':   'L3_1h_z_se_12',
}
Z_HIGH = {
    '1m':   'L3_1m_z_high_15',  '5m':   'L3_5m_z_high_9',
    '15m':  'L3_15m_z_high_12',
}
Z_LOW = {
    '1m':   'L3_1m_z_low_15',  '5m':   'L3_5m_z_low_9',
    '15m':  'L3_15m_z_low_12',
}
HURST = {
    '1m':   'L3_1m_hurst_15',  '5m':   'L3_5m_hurst_9',
    '15m':  'L3_15m_hurst_12', '1h':   'L3_1h_hurst_12',
}
RPROB = {
    '1m':   'L3_1m_reversion_prob_15',  '5m':   'L3_5m_reversion_prob_9',
    '15m':  'L3_15m_reversion_prob_12', '1h':   'L3_1h_reversion_prob_12',
}

# L1 — per-bar primitives (TF-anchored single-bar values)
BODY = {
    '1m':  'L1_1m_body',  '5m':  'L1_5m_body',  '15m': 'L1_15m_body',
    '1h':  'L1_1h_body',
}
BAR_RANGE = {
    '1m':  'L1_1m_bar_range',  '5m':  'L1_5m_bar_range',
    '15m': 'L1_15m_bar_range', '1h':  'L1_1h_bar_range',
}
PRICE_VEL_1B = {
    '1m':  'L1_1m_price_velocity_1b',  '5m':  'L1_5m_price_velocity_1b',
    '15m': 'L1_15m_price_velocity_1b',
}
PRICE_ACCEL_1B = {
    '1m':  'L1_1m_price_accel_1b',  '5m':  'L1_5m_price_accel_1b',
    '15m': 'L1_15m_price_accel_1b',
}

# Per-TF colors (consistent across panels)
TF_COLORS = {
    '1m':  'tab:red', '5m':  'tab:orange',
    '15m': 'tab:green', '1h':  'tab:blue',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2026_02_12')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='chart')
    ap.add_argument('--dpi', type=int, default=320)
    ap.add_argument('--figwidth', type=float, default=30)
    ap.add_argument('--figheight', type=float, default=32)
    ap.add_argument('--split-time', default=None,
                          help='HH:MM (UTC) — split each panel into two columns; '
                                  '"auto" splits at the day high.')
    args = ap.parse_args()

    feats = load_features(days=[args.day], root=args.features_root)
    if feats.empty:
        print(f'No features for {args.day}'); sys.exit(1)
    feats = feats.sort_values('timestamp').reset_index(drop=True)

    ohlcv = pd.read_parquet(os.path.join(args.atlas_5s, f'{args.day}.parquet'))
    if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
        ohlcv = ohlcv.copy()
        ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
    ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

    feats_ts = feats['timestamp'].astype('int64').values
    feats_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in feats_ts]
    oh_ts = ohlcv['timestamp'].values
    oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in oh_ts]

    # Resolve split time
    split_dt = None
    if args.split_time:
        if args.split_time == 'auto':
            i_max = int(np.argmax(ohlcv['close'].values))
            split_dt = oh_dt[i_max]
        else:
            day_iso = args.day.replace('_', '-')
            split_dt = datetime.strptime(
                f'{day_iso} {args.split_time}:00',
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

    # 14 panels total: 1 price + 13 feature groups
    N_PANELS = 14
    height_ratios = [2.0] + [0.9] * (N_PANELS - 1)

    fig = plt.figure(figsize=(args.figwidth, args.figheight))
    if split_dt is not None:
        gs = fig.add_gridspec(N_PANELS, 2, height_ratios=height_ratios,
                                          hspace=0.10, wspace=0.06)
        axes_L = [fig.add_subplot(gs[i, 0]) for i in range(N_PANELS)]
        axes_R = [fig.add_subplot(gs[i, 1]) for i in range(N_PANELS)]
        class _AxesGroup:
            def __init__(self, axes_list): self._axes = axes_list
            def __getattr__(self, name):
                def _fanout(*a, **kw):
                    res = [getattr(ax, name)(*a, **kw) for ax in self._axes]
                    return res[0] if res else None
                return _fanout
        axes = [_AxesGroup([axes_L[i], axes_R[i]]) for i in range(N_PANELS)]
    else:
        gs = fig.add_gridspec(N_PANELS, 1, height_ratios=height_ratios, hspace=0.10)
        axes = [fig.add_subplot(gs[i]) for i in range(N_PANELS)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        axes_L = axes
        axes_R = None

    def _plot_multi_tf(ax, feat_dict, label_suffix, lw=0.8, ls='-'):
        """Plot a feature dict across TFs on one panel."""
        for tf, col in feat_dict.items():
            if col not in feats.columns: continue
            ax.plot(feats_dt, feats[col].values, lw=lw, linestyle=ls,
                          color=TF_COLORS.get(tf), label=f'{tf} {label_suffix}')

    # ── Panel 1: price + 5m mean ────────────────────────────────────────
    ax = axes[0]
    ax.plot(oh_dt, ohlcv['close'], color='black', lw=0.6, alpha=0.85,
                 label='5s close')
    if 'L2_5m_price_mean_9' in feats.columns:
        ax.plot(feats_dt, feats['L2_5m_price_mean_9'].values,
                     color='tab:orange', lw=1.4, label='5m mean')
    if 'L2_15m_price_mean_12' in feats.columns:
        ax.plot(feats_dt, feats['L2_15m_price_mean_12'].values,
                     color='tab:green', lw=1.4, label='15m mean')
    ax.set_title(f'{args.day}  —  V2 feature panorama')
    ax.set_ylabel('price')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: VOLUME mean (multi-TF) ────────────────────────────────
    ax = axes[1]
    _plot_multi_tf(ax, VOL_MEAN, 'vol_mean', lw=0.9)
    ax.set_ylabel('vol_mean')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: VOLUME sigma (multi-TF) ───────────────────────────────
    ax = axes[2]
    _plot_multi_tf(ax, VOL_SIGMA, 'vol_sigma', lw=0.9, ls='--')
    ax.set_ylabel('vol_sigma')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: PRICE sigma (volatility, multi-TF) ────────────────────
    ax = axes[3]
    _plot_multi_tf(ax, PRICE_SIGMA, 'price_sigma', lw=0.9)
    ax.set_ylabel('price_sigma')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: SWING_NOISE (chop gauge, multi-TF) ────────────────────
    ax = axes[4]
    _plot_multi_tf(ax, SWING_NOISE, 'swing_noise', lw=0.9)
    ax.set_ylabel('swing_noise')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: PRICE VELOCITY (multi-TF, L2) ─────────────────────────
    ax = axes[5]
    _plot_multi_tf(ax, PRICE_VEL, 'price_vel', lw=0.9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('price velocity (L2)')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 7: PRICE ACCEL (multi-TF, L2) ────────────────────────────
    ax = axes[6]
    _plot_multi_tf(ax, PRICE_ACCEL, 'price_accel', lw=0.9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('price accel (L2)')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 8: VOL VELOCITY (multi-TF) ───────────────────────────────
    ax = axes[7]
    _plot_multi_tf(ax, VOL_VEL, 'vol_vel', lw=0.9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('vol velocity')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 9: VOL ACCEL (multi-TF) ──────────────────────────────────
    ax = axes[8]
    _plot_multi_tf(ax, VOL_ACCEL, 'vol_accel', lw=0.9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('vol accel')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 10: Z_SE multi-TF ────────────────────────────────────────
    ax = axes[9]
    _plot_multi_tf(ax, Z_SE, 'z_se', lw=0.9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(1.8, color='red', lw=0.4, linestyle=':', alpha=0.5)
    ax.axhline(-1.8, color='red', lw=0.4, linestyle=':', alpha=0.5)
    ax.set_ylabel('z_se')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 11: z_high & z_low as I-chart with ±1σ, ±2σ, ±3σ bands ───
    # z_high and z_low oscillate around zero. The horizontal reference
    # lines at ±1, ±2, ±3 are the SPC significance thresholds
    # (the "sigma bands" of the standard normal). Penetration past ±2
    # means the bar's extreme is statistically significant; ±3 is rare.
    ax = axes[10]
    for tf in ('1m', '5m', '15m'):
        zhi_col = Z_HIGH.get(tf)
        zlo_col = Z_LOW.get(tf)
        c = TF_COLORS.get(tf)
        if zhi_col in feats.columns:
            ax.plot(feats_dt, feats[zhi_col].values, lw=0.7, color=c,
                         alpha=0.85, label=f'{tf} z_high')
        if zlo_col in feats.columns:
            ax.plot(feats_dt, feats[zlo_col].values, lw=0.7, color=c,
                         alpha=0.6, linestyle='--', label=f'{tf} z_low')
    # Constant ±1σ ±2σ ±3σ reference lines (the "sigma bands")
    for k, ls, alpha in [(1, '-', 0.40), (2, '--', 0.55), (3, ':', 0.60)]:
        ax.axhline(+k, color='gray', lw=0.6, linestyle=ls, alpha=alpha,
                          label=f'±{k}σ' if k == 1 else None)
        ax.axhline(-k, color='gray', lw=0.6, linestyle=ls, alpha=alpha)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel('z_high / z_low\n(sigma bands)')
    ax.legend(loc='upper left', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 12: HURST multi-TF ───────────────────────────────────────
    ax = axes[11]
    _plot_multi_tf(ax, HURST, 'hurst', lw=0.9)
    ax.axhline(0.5, color='gray', lw=0.5, linestyle=':',
                     label='hurst=0.5 (mean-rev/trend boundary)')
    ax.set_ylabel('hurst')
    ax.legend(loc='upper left', fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    # ── Panel 13: REVERSION_PROB multi-TF ──────────────────────────────
    ax = axes[12]
    _plot_multi_tf(ax, RPROB, 'rprob', lw=0.9)
    ax.set_ylabel('reversion_prob')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 14: PER-BAR body + bar_range (combined) ──────────────────
    ax = axes[13]
    _plot_multi_tf(ax, BODY, 'body', lw=0.7, ls='-')
    _plot_multi_tf(ax, BAR_RANGE, 'bar_range', lw=0.7, ls=':')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('L1 body / range')
    ax.set_xlabel('time (UTC)')
    ax.legend(loc='upper left', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # Apply bottom-row x-formatter to the actual underlying axes
    bottom_axes = [axes_L[-1]] + ([axes_R[-1]] if axes_R is not None else [])
    for a in bottom_axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        a.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    def _fit_y_to_xlim(ax, padding=0.05):
        """Refit y-axis to the data visible within current xlim."""
        xmin, xmax = ax.get_xlim()
        y_min, y_max = float('inf'), float('-inf')
        for line in ax.get_lines():
            xs = line.get_xdata()
            ys = np.asarray(line.get_ydata(), dtype=np.float64)
            if len(xs) == 0:
                continue
            try:
                xs_num = mdates.date2num(xs)
            except Exception:
                continue
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

    if split_dt is None:
        for a in axes_L[:-1]:
            plt.setp(a.get_xticklabels(), visible=False)
    else:
        # Apply split: xlim per column + auto y-axis fit per segment
        day_start = oh_dt[0]
        day_end = oh_dt[-1]
        feats_ts_arr = feats['timestamp'].astype('int64').values
        oh_ts_arr = oh_ts.astype('int64')
        split_ts = int(split_dt.timestamp())
        for i in range(N_PANELS):
            axL = axes_L[i]
            axR = axes_R[i]
            axL.set_xlim(day_start, split_dt)
            axR.set_xlim(split_dt, day_end)
            # Hide x labels except bottom row
            if i < N_PANELS - 1:
                plt.setp(axL.get_xticklabels(), visible=False)
                plt.setp(axR.get_xticklabels(), visible=False)
            # Right column: tick on right, no ylabel duplicate
            axR.yaxis.tick_right()
            axR.yaxis.set_label_position('right')
            # Drop legend on right (keep on left only)
            leg = axR.get_legend()
            if leg is not None:
                leg.remove()
            # Refit y-axis to data WITHIN this column's x-range
            _fit_y_to_xlim(axL)
            _fit_y_to_xlim(axR)
        # Bottom row: tick formatter
        for a in (axes_L[-1], axes_R[-1]):
            a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            a.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, f'{args.day}_other_features.png')
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_png}')


if __name__ == '__main__':
    main()
