"""Compare candidate DIRECTION signals on a single day.

For one day, extracts multiple candidate "macro direction" signals and:
    1. Plots them as time series over the day
    2. Quantifies agreement (correlation, sign-agreement %) between pairs
    3. Identifies the periods where they diverge
    4. Overlays raw price for visual sanity check

Candidates compared:
    A. L2 price_velocity_w at multiple TFs (1m, 5m, 15m, 1h, 4h, 1D)
    B. L2 price_mean_w slope (true "regression mean velocity") at 1h, 4h
    C. L2 vwap_w slope at 1h, 4h (volume-weighted variant)
    D. L3 z_se_w sign at 1h, 4h (band-relative direction)
    E. Composite: sign agreement across TFs (e.g., "all four longer TFs agree")

The goal: pick the signal that best represents the macro trend you want
to act on, with eyes-on evidence rather than picking blind.

Usage:
    python tools/inspect_direction_signals.py --day 2026_02_12
    python tools/inspect_direction_signals.py --random
    python tools/inspect_direction_signals.py --day 2026_03_03 --out reports/findings/direction_audit/
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


# Candidate direction features, grouped by family
VELOCITY_FEATS = {
    '1m_vel_w':  'L2_1m_price_velocity_5',
    '5m_vel_w':  'L2_5m_price_velocity_9',
    '15m_vel_w': 'L2_15m_price_velocity_12',
    '1h_vel_w':  'L2_1h_price_velocity_12',
    '4h_vel_w':  'L2_4h_price_velocity_18',
    '1D_vel_w':  'L2_1D_price_velocity_5',
}
MEAN_FEATS = {       # regression mean = price_mean_w; slope = its derivative
    '1h_mean':   'L2_1h_price_mean_12',
    '4h_mean':   'L2_4h_price_mean_18',
}
VWAP_FEATS = {
    '1h_vwap':   'L2_1h_vwap_12',
    '4h_vwap':   'L2_4h_vwap_18',
}
ZSE_FEATS = {
    '1h_z_se':   'L3_1h_z_se_12',
    '4h_z_se':   'L3_4h_z_se_18',
}


def _slope(series: np.ndarray, window_bars: int) -> np.ndarray:
    """Per-bar slope of `series` measured over `window_bars` lookback.

    slope[t] = (series[t] - series[t - window_bars]) / window_bars
    """
    out = np.full_like(series, fill_value=np.nan, dtype=np.float64)
    if len(series) > window_bars:
        out[window_bars:] = (series[window_bars:] - series[:-window_bars]) / window_bars
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None,
                          help='Day to inspect (YYYY_MM_DD). Mutually exclusive with --random')
    ap.add_argument('--random', action='store_true',
                          help='Pick a random day from available data')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out',
                          default='reports/findings/direction_signals')
    ap.add_argument('--mean-slope-bars', type=int, default=720,
                          help='Bars to look back for regression-mean slope (720 = 1h)')
    args = ap.parse_args()

    if not args.day and not args.random:
        print('Specify either --day YYYY_MM_DD or --random')
        sys.exit(2)

    # Resolve day
    if args.random:
        files = sorted(glob.glob(os.path.join(args.features_root, 'L0', '*.parquet')))
        days = [os.path.basename(f).replace('.parquet', '') for f in files]
        if not days:
            print('No days found'); sys.exit(1)
        args.day = random.choice(days)
        print(f'Random day picked: {args.day}')

    # Load features + price
    feats = load_features(days=[args.day], root=args.features_root)
    if feats.empty:
        print(f'No features for {args.day}'); sys.exit(1)
    feats = feats.sort_values('timestamp').reset_index(drop=True)

    ohlcv_path = os.path.join(args.atlas_5s, f'{args.day}.parquet')
    if os.path.exists(ohlcv_path):
        ohlcv = pd.read_parquet(ohlcv_path)
        if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
            ohlcv = ohlcv.copy()
            ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
        ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
    else:
        ohlcv = None

    ts_arr = feats['timestamp'].astype('int64').values
    dt_arr = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_arr]

    # Build a combined dataframe of all candidate direction signals
    sig_df = pd.DataFrame({'ts': ts_arr})
    for label, col in VELOCITY_FEATS.items():
        if col in feats.columns:
            sig_df[label] = feats[col].values

    # Mean slopes (price_mean_w derivative)
    for label, col in MEAN_FEATS.items():
        if col in feats.columns:
            mean_arr = feats[col].values.astype(np.float64)
            slope = _slope(mean_arr, args.mean_slope_bars)
            sig_df[f'{label}_slope'] = slope

    # VWAP slopes
    for label, col in VWAP_FEATS.items():
        if col in feats.columns:
            vwap_arr = feats[col].values.astype(np.float64)
            slope = _slope(vwap_arr, args.mean_slope_bars)
            sig_df[f'{label}_slope'] = slope

    # z_se raw (sign carries direction)
    for label, col in ZSE_FEATS.items():
        if col in feats.columns:
            sig_df[label] = feats[col].values

    # Drop the timestamp before describing
    candidate_cols = [c for c in sig_df.columns if c != 'ts']

    # ── Pearson + sign-agreement matrices ──────────────────────────────
    print(f'\n{"=" * 95}')
    print(f'DIRECTION-SIGNAL AUDIT  day={args.day}')
    print(f'  signals available: {len(candidate_cols)}')
    print(f'{"=" * 95}')

    valid_df = sig_df[candidate_cols].dropna(how='any')
    if len(valid_df) < 10:
        print(f'!!! Too few rows ({len(valid_df)}) with all signals; check feature coverage')
    else:
        corr = valid_df.corr()
        print(f'\nPEARSON CORRELATION  (n_valid={len(valid_df)})')
        print(corr.round(2).to_string())

        # Sign-agreement %
        signs = np.sign(valid_df.values)
        agree = pd.DataFrame(index=candidate_cols, columns=candidate_cols, dtype=float)
        for i, ci in enumerate(candidate_cols):
            for j, cj in enumerate(candidate_cols):
                a = float((signs[:, i] == signs[:, j]).mean())
                agree.loc[ci, cj] = a
        print(f'\nSIGN-AGREEMENT %  (fraction of bars where signal_i.sign == signal_j.sign)')
        print(agree.applymap(lambda x: f'{x*100:>5.1f}%').to_string())

    # ── Distribution per signal ────────────────────────────────────────
    print(f'\nPER-SIGNAL DISTRIBUTION  (n={len(sig_df)})')
    for c in candidate_cols:
        v = sig_df[c].dropna().values
        if len(v) == 0: continue
        pos = (v > 0).mean()
        neg = (v < 0).mean()
        print(f'  {c:<20}  mean={v.mean():>+9.3f}  q50={np.median(v):>+9.3f}  '
                  f'q10={np.quantile(v, 0.10):>+9.3f}  q90={np.quantile(v, 0.90):>+9.3f}  '
                  f'%pos={pos*100:>4.1f}%  %neg={neg*100:>4.1f}%')

    # ── Plot ───────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.5, 1, 1, 1, 1], hspace=0.35)

    # 0: price
    ax0 = fig.add_subplot(gs[0])
    if ohlcv is not None:
        oh_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc)
                       for t in ohlcv['timestamp'].values]
        ax0.plot(oh_dt, ohlcv['close'], color='black', lw=0.7)
        ax0.set_title(f'{args.day}  —  price (5s close)')
    else:
        ax0.text(0.5, 0.5, 'no 5s OHLCV', ha='center', transform=ax0.transAxes)
    ax0.set_ylabel('price')
    ax0.grid(True, alpha=0.3)

    # 1: velocity at multiple TFs (different scales)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(VELOCITY_FEATS)))
    for (label, col), c in zip(VELOCITY_FEATS.items(), cmap):
        if label in sig_df.columns:
            ax1.plot(dt_arr, sig_df[label], label=label, color=c, lw=0.8)
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.set_ylabel('price_velocity_w')
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2: regression-mean slope vs vwap slope
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    for label in MEAN_FEATS:
        col = f'{label}_slope'
        if col in sig_df.columns:
            ax2.plot(dt_arr, sig_df[col], label=col, lw=0.8)
    for label in VWAP_FEATS:
        col = f'{label}_slope'
        if col in sig_df.columns:
            ax2.plot(dt_arr, sig_df[col], label=col, lw=0.8, linestyle='--')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_ylabel('mean / vwap slope')
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3: z_se across TFs
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    for label in ZSE_FEATS:
        if label in sig_df.columns:
            ax3.plot(dt_arr, sig_df[label], label=label, lw=0.8)
    ax3.axhline(0, color='gray', lw=0.5)
    ax3.axhline(2, color='red', lw=0.4, alpha=0.5)
    ax3.axhline(-2, color='red', lw=0.4, alpha=0.5)
    ax3.set_ylabel('z_se (band proximity)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4: sign-agreement across velocities (composite)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    vel_cols = [c for c in candidate_cols
                       if c in VELOCITY_FEATS and c != '1m_vel_w']  # exclude 1m noise
    if vel_cols:
        signs_arr = np.sign(sig_df[vel_cols].fillna(0).values)
        agree_count = (signs_arr > 0).sum(axis=1) - (signs_arr < 0).sum(axis=1)
        # Range: -len to +len. Positive = more bullish; negative = more bearish.
        ax4.fill_between(dt_arr, 0, agree_count, where=(agree_count >= 0),
                                  alpha=0.5, color='green', step='mid', label='net bullish')
        ax4.fill_between(dt_arr, 0, agree_count, where=(agree_count < 0),
                                  alpha=0.5, color='red', step='mid', label='net bearish')
        ax4.axhline(0, color='gray', lw=0.5)
        ax4.set_ylabel('sign agreement\n(velocity TFs)')
        ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax4.set_xlabel('time (UTC)')

    out_png = os.path.join(args.out, f'{args.day}_direction_signals.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved -> {out_png}')

    out_csv = os.path.join(args.out, f'{args.day}_direction_signals.csv')
    sig_df.to_csv(out_csv, index=False)
    print(f'CSV saved  -> {out_csv}')


if __name__ == '__main__':
    main()
