"""Visualize the regression mean and SE bands on top of price.

Shows the rolling regression-mean track (price_mean_w) at 1h and 4h with
± SE bands, the corresponding z_se path, and the regression-mean
velocity. Lets you SEE fair value rather than abstractly reason about it.

Panels (top-down):
    1. 5s price overlaid with 1h regression mean + ±1σ + ±2σ SE bands
    2. Same with 4h regression mean + bands
    3. Distance from mean (price - mean) at 1h and 4h, in points
    4. z_se at 1h and 4h with ±2 markers (entry/exit thresholds)
    5. Regression mean velocity at 1h and 4h

Usage:
    python tools/inspect_regression_mean.py --day 2026_02_12
    python tools/inspect_regression_mean.py --day 2026_03_03
    python tools/inspect_regression_mean.py --random
"""
from __future__ import annotations

import argparse
import glob
import os
import random
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


# ── Feature names (V2 canonical) ─────────────────────────────────────
FEAT_1H = {
    'mean':  'L2_1h_price_mean_12',
    'sigma': 'L2_1h_price_sigma_12',
    'se_hi': 'L3_1h_SE_high_12',
    'se_lo': 'L3_1h_SE_low_12',
    'z_se':  'L3_1h_z_se_12',
    'vel':   'L2_1h_price_velocity_12',
}
FEAT_4H = {
    'mean':  'L2_4h_price_mean_18',
    'sigma': 'L2_4h_price_sigma_18',
    'se_hi': 'L3_4h_SE_high_18',
    'se_lo': 'L3_4h_SE_low_18',
    'z_se':  'L3_4h_z_se_18',
    'vel':   'L2_4h_price_velocity_18',
}


def _resolve_day(args) -> str:
    if args.day:
        return args.day
    if args.random:
        files = sorted(glob.glob(os.path.join(args.features_root, 'L0', '*.parquet')))
        days = [os.path.basename(f).replace('.parquet', '') for f in files]
        if not days:
            print('No days found'); sys.exit(1)
        return random.choice(days)
    print('Specify --day YYYY_MM_DD or --random'); sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None)
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='reports/findings/regression_mean')
    ap.add_argument('--tier-overlay', action='store_true',
                          help='If iso pickles exist, overlay trade entries on price')
    args = ap.parse_args()

    day = _resolve_day(args)
    print(f'Day: {day}')

    feats = load_features(days=[day], root=args.features_root)
    if feats.empty:
        print(f'No features for {day}'); sys.exit(1)
    feats = feats.sort_values('timestamp').reset_index(drop=True)

    ohlcv_path = os.path.join(args.atlas_5s, f'{day}.parquet')
    if not os.path.exists(ohlcv_path):
        print(f'No 5s OHLCV at {ohlcv_path}'); sys.exit(1)
    ohlcv = pd.read_parquet(ohlcv_path)
    if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
        ohlcv = ohlcv.copy()
        ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
    ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

    ts_arr = feats['timestamp'].astype('int64').values
    dt_arr = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_arr]
    oh_ts = ohlcv['timestamp'].values
    oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in oh_ts]
    oh_close = ohlcv['close'].values

    def fcol(c):
        return feats[c].values if c in feats.columns else np.full(len(feats), np.nan)

    mean_1h = fcol(FEAT_1H['mean'])
    sigma_1h = fcol(FEAT_1H['sigma'])
    sehi_1h = fcol(FEAT_1H['se_hi'])
    selo_1h = fcol(FEAT_1H['se_lo'])
    zse_1h = fcol(FEAT_1H['z_se'])
    vel_1h = fcol(FEAT_1H['vel'])

    mean_4h = fcol(FEAT_4H['mean'])
    sigma_4h = fcol(FEAT_4H['sigma'])
    sehi_4h = fcol(FEAT_4H['se_hi'])
    selo_4h = fcol(FEAT_4H['se_lo'])
    zse_4h = fcol(FEAT_4H['z_se'])
    vel_4h = fcol(FEAT_4H['vel'])

    # ── Console summary ─────────────────────────────────────────────────
    print(f'\n{"=" * 80}')
    print(f'REGRESSION MEAN AUDIT  day={day}')
    print(f'{"=" * 80}')
    open_p = float(ohlcv.iloc[0]['close']) if len(ohlcv) else np.nan
    close_p = float(ohlcv.iloc[-1]['close']) if len(ohlcv) else np.nan
    print(f'  Price open: {open_p:.2f}    close: {close_p:.2f}    net: {close_p-open_p:+.2f} pts')
    print(f'\n  1h regression mean track:')
    print(f'    first valid:  {np.nanmin(np.where(np.isfinite(mean_1h), mean_1h, np.nan)):.2f}')
    print(f'    last valid:   {mean_1h[~np.isnan(mean_1h)][-1] if (~np.isnan(mean_1h)).any() else float("nan"):.2f}')
    print(f'    drift over day: {mean_1h[~np.isnan(mean_1h)][-1] - mean_1h[~np.isnan(mean_1h)][0] if (~np.isnan(mean_1h)).any() else float("nan"):+.2f} pts')
    print(f'    avg sigma (SE):  {np.nanmean(sigma_1h):.2f} pts')
    print(f'\n  4h regression mean track:')
    print(f'    first valid:  {np.nanmin(np.where(np.isfinite(mean_4h), mean_4h, np.nan)):.2f}')
    print(f'    last valid:   {mean_4h[~np.isnan(mean_4h)][-1] if (~np.isnan(mean_4h)).any() else float("nan"):.2f}')
    print(f'    drift over day: {mean_4h[~np.isnan(mean_4h)][-1] - mean_4h[~np.isnan(mean_4h)][0] if (~np.isnan(mean_4h)).any() else float("nan"):+.2f} pts')
    print(f'    avg sigma (SE):  {np.nanmean(sigma_4h):.2f} pts')

    # Distance from mean — what fraction of bars is price > 1 SE away from mean?
    dist_1h = oh_close[:len(mean_1h)] - mean_1h[:len(oh_close)] if len(oh_close) >= len(mean_1h) else None
    if dist_1h is not None:
        valid = dist_1h[np.isfinite(dist_1h)]
        if len(valid):
            print(f'\n  Price - 1h mean distribution: '
                      f'mean={valid.mean():+.2f}  q10={np.quantile(valid,0.10):+.2f}  '
                      f'q90={np.quantile(valid,0.90):+.2f}')
    print(f'\n  z_se 1h: q10={np.nanquantile(zse_1h, 0.10):+.2f}  q50={np.nanquantile(zse_1h, 0.50):+.2f}  q90={np.nanquantile(zse_1h, 0.90):+.2f}')
    print(f'  z_se 4h: q10={np.nanquantile(zse_4h, 0.10):+.2f}  q50={np.nanquantile(zse_4h, 0.50):+.2f}  q90={np.nanquantile(zse_4h, 0.90):+.2f}')

    # ── Plot ───────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 1, 1, 1], hspace=0.35)

    # Panel 1: price + 1h regression mean + SE bands
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(oh_dt, oh_close, color='black', lw=0.8, label='5s price')
    ax1.plot(dt_arr, mean_1h, color='blue', lw=1.2, label='1h regression mean')
    # ±1 σ
    ax1.fill_between(dt_arr, mean_1h - sigma_1h, mean_1h + sigma_1h,
                              color='blue', alpha=0.10, label='±1σ (1h)')
    # ±2 σ (the SE bands used for entry signals)
    ax1.plot(dt_arr, sehi_1h, color='blue', lw=0.6, linestyle='--',
                  alpha=0.6, label='SE_high (1h)')
    ax1.plot(dt_arr, selo_1h, color='blue', lw=0.6, linestyle='--',
                  alpha=0.6, label='SE_low (1h)')

    if args.tier_overlay:
        # Overlay iso trade entries from any iso run if present
        import pickle
        for tier_path in sorted(glob.glob('training_iso_v2/output/*.pkl')):
            if 'regret' in tier_path: continue
            try:
                with open(tier_path, 'rb') as f:
                    tt = pickle.load(f)
                day_tt = [t for t in tt if t.entry_day == day]
                for t in day_tt:
                    edt = datetime.fromtimestamp(int(t.entry_ts), tz=timezone.utc)
                    marker = '^' if t.direction == 'long' else 'v'
                    color = 'green' if t.pnl > 0 else 'red'
                    ax1.scatter([edt], [t.entry_price], marker=marker,
                                    color=color, s=30, alpha=0.4, zorder=5)
            except Exception:
                continue

    ax1.set_title(f'{day}  —  price + 1h regression mean + SE bands')
    ax1.set_ylabel('price')
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Panel 2: price + 4h regression mean + SE bands
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(oh_dt, oh_close, color='black', lw=0.8, label='5s price')
    ax2.plot(dt_arr, mean_4h, color='purple', lw=1.4, label='4h regression mean')
    ax2.fill_between(dt_arr, mean_4h - sigma_4h, mean_4h + sigma_4h,
                              color='purple', alpha=0.10, label='±1σ (4h)')
    ax2.plot(dt_arr, sehi_4h, color='purple', lw=0.6, linestyle='--',
                  alpha=0.6, label='SE_high (4h)')
    ax2.plot(dt_arr, selo_4h, color='purple', lw=0.6, linestyle='--',
                  alpha=0.6, label='SE_low (4h)')
    ax2.set_title('price + 4h regression mean + SE bands')
    ax2.set_ylabel('price')
    ax2.legend(loc='upper left', fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    # Panel 3: distance from mean (in pts) at 1h vs 4h
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    # Align mean to oh_close index by ts
    feats_ts = ts_arr.astype(np.int64)
    oh_ts_int = oh_ts.astype(np.int64)
    # Walk: at each oh_ts find nearest feats_ts
    idx_map = np.searchsorted(feats_ts, oh_ts_int, side='right') - 1
    idx_map = np.clip(idx_map, 0, len(feats_ts) - 1)
    aligned_mean_1h = mean_1h[idx_map]
    aligned_mean_4h = mean_4h[idx_map]
    dist_1h = oh_close - aligned_mean_1h
    dist_4h = oh_close - aligned_mean_4h
    ax3.plot(oh_dt, dist_1h, color='blue', lw=0.8, label='price - 1h mean')
    ax3.plot(oh_dt, dist_4h, color='purple', lw=0.8, label='price - 4h mean')
    ax3.axhline(0, color='gray', lw=0.5)
    ax3.set_ylabel('dist (pts)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: z_se 1h + 4h
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(dt_arr, zse_1h, color='blue', lw=0.8, label='z_se 1h')
    ax4.plot(dt_arr, zse_4h, color='purple', lw=0.8, label='z_se 4h')
    ax4.axhline(0, color='gray', lw=0.5)
    ax4.axhline(2, color='red', lw=0.4, alpha=0.5, linestyle='--')
    ax4.axhline(-2, color='red', lw=0.4, alpha=0.5, linestyle='--')
    ax4.axhline(1.8, color='orange', lw=0.4, alpha=0.5, linestyle=':')
    ax4.axhline(-1.8, color='orange', lw=0.4, alpha=0.5, linestyle=':')
    ax4.set_ylabel('z_se')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: regression mean velocity
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(dt_arr, vel_1h, color='blue', lw=0.8, label='1h mean velocity')
    ax5.plot(dt_arr, vel_4h, color='purple', lw=0.8, label='4h mean velocity')
    ax5.axhline(0, color='gray', lw=0.5)
    ax5.fill_between(dt_arr, 0, vel_1h, where=(vel_1h > 0),
                              alpha=0.15, color='green', step='mid')
    ax5.fill_between(dt_arr, 0, vel_1h, where=(vel_1h < 0),
                              alpha=0.15, color='red', step='mid')
    ax5.set_ylabel('mean velocity')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax5.set_xlabel('time (UTC)')

    out_png = os.path.join(args.out, f'{day}_regression_mean.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved -> {out_png}')


if __name__ == '__main__':
    main()
