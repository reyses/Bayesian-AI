"""1-day chart: 5s price with 5m / 15m / 1h / 4h regression mean overlays.

Usage:
    python tools/day_chart_regression_means.py --day 2026_02_12
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


# Regression means worth showing on the chart:
#   1m / 5m / 15m: smooth, track price intraday — KEEP
#   1h: step function with sizable lag — keep only its slope (useful direction)
#   4h: essentially horizontal intraday — DROP
# 1m mean dropped from chart overlays (too redundant with 5s close).
# Kept in MEAN_COLS for the leg detector which still needs it.
MEAN_COLS = {
    '5s':  'L2_5s_price_mean_9',
    '15s': 'L2_15s_price_mean_12',
    '1m':  'L2_1m_price_mean_15',
    '5m':  'L2_5m_price_mean_9',
    '15m': 'L2_15m_price_mean_12',
}
# Means actually drawn on the price panel
DRAW_MEANS = ['5m', '15m']
# VWAPs are visually identical to regression means at the same TF — DROP all.

# Slopes (bottom panel) — direction signals
SLOPE_COLS = {
    '5m':  'L2_5m_price_mean_9',
    '15m': 'L2_15m_price_mean_12',
    '1h':  'L2_1h_price_mean_12',
}
# Slope of regression mean = (mean[t] - mean[t-lookback]) / lookback
# Slope is always measured FROM a reference point — the regression mean
# is the cleanest reference (smoothed price). Lookback in 5s bars.
SLOPE_LOOKBACK_BARS = 60      # 5 minutes (60 × 5s)
# Means → grayscale (background reference). Reserve red/green ONLY for
# trade legs (primary signal). Strategic shading uses a neutral palette.
COLORS = {'1m': '#bbbbbb', '5m': '#777777', '15m': '#222222',
                '1h': '#888888'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2026_02_12')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--atlas-5s', default='DATA/ATLAS/5s')
    ap.add_argument('--out', default='chart')
    ap.add_argument('--dpi', type=int, default=320)
    ap.add_argument('--figwidth', type=float, default=30)
    ap.add_argument('--figheight', type=float, default=15)
    ap.add_argument('--primary-tf', default='1m',
                          help='Faster TF for divergence/SE bands (1m or 5m)')
    ap.add_argument('--secondary-tf', default='5m',
                          help='Slower TF; divergence is primary - secondary')
    ap.add_argument('--split-time', default=None,
                          help='HH:MM (UTC) — split the price panel into two '
                                  'columns, each y-axis-fitted to its segment. '
                                  '"auto" splits at the day high.')
    ap.add_argument('--confirmed-fade', action='store_true', default=True,
                          help='Mark confirmed-fade entries: 5s price outside '
                                  '15m ±2σ for N consecutive bars, target = 15m mean.')
    ap.add_argument('--cf-band-tf', default='15m',
                          help='TF for the band used by confirmed-fade trigger')
    ap.add_argument('--cf-target-tf', default='5m',
                          help='TF whose mean is the FADE TARGET (default 5m — '
                                  'price actually reaches it, unlike 15m mean)')
    ap.add_argument('--cf-band-k', type=float, default=2.0,
                          help='σ multiplier for the band')
    ap.add_argument('--cf-confirm-bars', type=int, default=6,
                          help='Bars (5s) price must stay outside band')
    ap.add_argument('--leg-min-pts', type=float, default=12.0,
                          help='Minimum leg amplitude in points (ZigZag '
                                  'reversal threshold). Smaller legs merge into '
                                  'continuation. Default 12 pts ≈ $24/contract.')
    ap.add_argument('--leg-tf', default='5m',
                          help='Mean TF used for swing-pivot detection (1m, 5m, 15m)')
    args = ap.parse_args()

    pri_tf = args.primary_tf
    sec_tf = args.secondary_tf
    band_color = COLORS.get(sec_tf, 'tab:gray')

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

    fig = plt.figure(figsize=(args.figwidth, args.figheight))

    # Resolve split time (if any)
    split_dt = None
    if args.split_time:
        if args.split_time == 'auto':
            # Auto-split at the day's HIGH
            i_max = int(np.argmax(ohlcv['close'].values))
            split_dt = oh_dt[i_max]
        else:
            day_iso = args.day.replace('_', '-')
            split_dt = datetime.strptime(
                f'{day_iso} {args.split_time}:00',
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

    # Layout: 2-column price panel when split, full-width slope+divergence below
    if split_dt is not None:
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1],
                                          hspace=0.10, wspace=0.04)
        ax_L = fig.add_subplot(gs[0, 0])
        ax_R = fig.add_subplot(gs[0, 1])
        price_axes = [ax_L, ax_R]
        ax_vel = fig.add_subplot(gs[1, :])
        ax_curv = fig.add_subplot(gs[2, :], sharex=ax_vel)
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)
        ax_L = fig.add_subplot(gs[0])
        price_axes = [ax_L]
        ax_vel = fig.add_subplot(gs[1], sharex=ax_L)
        ax_curv = fig.add_subplot(gs[2], sharex=ax_L)

    # Proxy that fans method calls across all price-panel axes.
    class _AxesGroup:
        def __init__(self, axes): self._axes = axes
        def __getattr__(self, name):
            def _fanout(*a, **kw):
                results = [getattr(ax, name)(*a, **kw) for ax in self._axes]
                return results[0] if results else None
            return _fanout
    ax = _AxesGroup(price_axes)

    # ── Strategic-direction shading (slope of 15m mean over 1h lookback) ──
    # The 15m regression mean LINE on the price panel rises smoothly
    # from morning into afternoon then drops — that's the eyeball "macro
    # direction". To reproduce that, slope must be measured over a
    # window long enough to ignore intra-hour noise: use 1h lookback.
    # Threshold is Q75 of |slope| so neutral periods stay un-shaded.
    LB_strategic = 720    # 1 hour × 12 bars/min @ 5s
    BIN_BARS = 180        # 15-min bins
    if MEAN_COLS.get('15m') in feats.columns:
        m15 = feats[MEAN_COLS['15m']].values.astype(np.float64)
        s15_strat = np.full_like(m15, np.nan)
        if len(m15) > LB_strategic:
            s15_strat[LB_strategic:] = (m15[LB_strategic:]
                                                       - m15[:-LB_strategic]) / LB_strategic
        thr = float(np.nanquantile(np.abs(s15_strat), 0.75))
        n = len(s15_strat)
        for start in range(0, n, BIN_BARS):
            end = min(start + BIN_BARS, n)
            seg = s15_strat[start:end]
            seg = seg[np.isfinite(seg)]
            if len(seg) == 0:
                continue
            mean_slope = float(seg.mean())
            if abs(mean_slope) < thr:
                continue
            # Strategic regime shading — use NEUTRAL palette so green/red
            # remain reserved for trade-leg coloring.
            color = 'lightsteelblue' if mean_slope > 0 else 'wheat'
            ax.axvspan(feats_dt[start], feats_dt[end-1], color=color,
                              alpha=0.30, zorder=0)

    ax.plot(oh_dt, ohlcv['close'], color='black', lw=0.6,
                 alpha=0.85, label='5s close', zorder=2)

    # z_high / z_low bands DROPPED for cleanliness — single ±2σ band below.


    for tf in DRAW_MEANS:
        col = MEAN_COLS.get(tf)
        if not col or col not in feats.columns:
            continue
        lw = 1.6 if tf == '15m' else 1.0
        ax.plot(feats_dt, feats[col].values, color=COLORS[tf], lw=lw,
                     label=f'{tf} regression mean', zorder=3)

    # SE bands at the SECONDARY (slower) TF — uses its mean and sigma
    SIGMA_COLS = {
        '1m':  'L2_1m_price_sigma_15',
        '5m':  'L2_5m_price_sigma_9',
        '15m': 'L2_15m_price_sigma_12',
        '1h':  'L2_1h_price_sigma_12',
    }
    mean_col = MEAN_COLS.get(sec_tf) or {
        '1h':  'L2_1h_price_mean_12',
    }.get(sec_tf)
    sigma_col = SIGMA_COLS.get(sec_tf)
    band_color = COLORS.get(sec_tf, 'tab:gray')
    # SE band — ONLY ±2σ (the trigger band), faint, no inner rings.
    if mean_col in feats.columns and sigma_col in feats.columns:
        m = feats[mean_col].values
        s = feats[sigma_col].values
        hi = m + 2 * s
        lo = m - 2 * s
        ax.plot(feats_dt, hi, color='#aaaaaa', lw=0.7,
                     linestyle='--', alpha=0.7,
                     label=f'{sec_tf} ±2σ', zorder=3)
        ax.plot(feats_dt, lo, color='#aaaaaa', lw=0.7,
                     linestyle='--', alpha=0.7, zorder=3)
        ax.fill_between(feats_dt, lo, hi, color='#aaaaaa',
                                  alpha=0.05, zorder=1)

    ax.set_title(f'{args.day}  —  5s price + regression mean overlays',
                       fontsize=14)
    ax.set_ylabel('price')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    # Slope panel — slope of each regression mean computed from a fixed
    # lookback. Slope is always measured FROM a reference point; the
    # regression mean is the cleanest reference (smoothed price).
    #
    # slope[t] = (mean[t] - mean[t - LOOKBACK]) / LOOKBACK
    #
    # Compared to L2_*_price_velocity_w (a step-function lagged by TF
    # cadence), these slopes update PER 5s BAR because t advances every
    # bar — even when the underlying mean is a step function, the slope
    # changes smoothly as the lookback window shifts.
    LB = SLOPE_LOOKBACK_BARS
    slopes_by_tf = {}
    for tf, col in SLOPE_COLS.items():
        if col not in feats.columns:
            continue
        m = feats[col].values.astype(np.float64)
        slope = np.full_like(m, np.nan)
        if len(m) > LB:
            slope[LB:] = (m[LB:] - m[:-LB]) / LB
        slopes_by_tf[tf] = slope
        ax_vel.plot(feats_dt, slope, color=COLORS[tf], lw=1.2,
                            label=f'{tf} mean slope')
    ax_vel.axhline(0, color='gray', lw=0.6)
    ax_vel.set_ylabel(f'slope (lb={LB*5}s)')
    ax_vel.legend(loc='upper left', fontsize=9, ncol=5)
    ax_vel.grid(True, alpha=0.3)
    plt.setp(ax_vel.get_xticklabels(), visible=False)

    # Primary − Secondary divergence panel. Tradable swings: when the
    # FASTER (primary) mean stretches away from the SLOWER (secondary)
    # mean and snaps back. div[t] = mean_primary[t] − mean_secondary[t].
    pri_mean_col = MEAN_COLS.get(pri_tf) or {
        '1h': 'L2_1h_price_mean_12',
    }.get(pri_tf)
    sec_mean_col = MEAN_COLS.get(sec_tf) or {
        '1h': 'L2_1h_price_mean_12',
    }.get(sec_tf)
    div_signal = None
    if pri_mean_col in feats.columns and sec_mean_col in feats.columns:
        m_pri = feats[pri_mean_col].values.astype(np.float64)
        m_sec = feats[sec_mean_col].values.astype(np.float64)
        div_signal = m_pri - m_sec
        ax_curv.plot(feats_dt, div_signal, color='black', lw=1.1,
                              label=f'{pri_tf} − {sec_tf} mean divergence (pts)')
        # Threshold lines at ±Q75 of |divergence|
        thr75 = float(np.nanquantile(np.abs(div_signal), 0.75))
        ax_curv.axhline(+thr75, color='red', lw=0.5, linestyle='--',
                                alpha=0.6, label=f'±Q75 = ±{thr75:.2f} pts')
        ax_curv.axhline(-thr75, color='red', lw=0.5, linestyle='--',
                                alpha=0.6)
        ax_curv.fill_between(feats_dt, 0, div_signal,
                                          where=(div_signal > 0),
                                          color='red', alpha=0.12, step='mid')
        ax_curv.fill_between(feats_dt, 0, div_signal,
                                          where=(div_signal < 0),
                                          color='green', alpha=0.12, step='mid')
    ax_curv.axhline(0, color='gray', lw=0.6)
    ax_curv.set_ylabel(f'{pri_tf} − {sec_tf} mean\n(divergence in pts)')
    ax_curv.set_xlabel('time (UTC)')
    ax_curv.legend(loc='upper left', fontsize=9, ncol=3)
    ax_curv.grid(True, alpha=0.3)
    ax_curv.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_curv.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # Mark "extreme divergence" entries + snapback exits on price panel.
    # Entry: |divergence| crosses ABOVE Q75 threshold
    # Exit:  divergence crosses back through zero (snapback complete)
    if False and div_signal is not None and pri_mean_col in feats.columns:
        m1 = feats[pri_mean_col].values
        thr_entry = thr75
        n_ent_short, n_ent_long, n_exit = 0, 0, 0
        in_long, in_short = False, False
        for i in range(1, len(div_signal)):
            d = div_signal[i]
            d_prev = div_signal[i-1]
            if not (np.isfinite(d) and np.isfinite(d_prev)):
                continue
            # Entry: cross from below threshold up through it
            if d_prev <= thr_entry and d > thr_entry and not in_short:
                ax.scatter([feats_dt[i]], [m1[i]], marker='v',
                                color='red', s=60, zorder=7,
                                edgecolors='black', linewidths=0.8)
                n_ent_short += 1
                in_short = True
            elif d_prev >= -thr_entry and d < -thr_entry and not in_long:
                ax.scatter([feats_dt[i]], [m1[i]], marker='^',
                                color='green', s=60, zorder=7,
                                edgecolors='black', linewidths=0.8)
                n_ent_long += 1
                in_long = True
            # Exit on snapback through zero
            if in_short and d <= 0:
                ax.scatter([feats_dt[i]], [m1[i]], marker='o',
                                color='white', s=30, zorder=7,
                                edgecolors='red', linewidths=1.0)
                n_exit += 1
                in_short = False
            elif in_long and d >= 0:
                ax.scatter([feats_dt[i]], [m1[i]], marker='o',
                                color='white', s=30, zorder=7,
                                edgecolors='green', linewidths=1.0)
                n_exit += 1
                in_long = False
        print(f'1m-5m divergence trades: {n_ent_short} shorts(▼ red), '
                  f'{n_ent_long} longs(▲ green), {n_exit} snapback exits(○)  '
                  f'thr=Q75=±{thr_entry:.2f} pts')

    # ── Swing pivots → TRADABLE LEGS (ZigZag-filtered) ────────────────
    # 1m-mean slope sign-changes give micro-pivots; ZigZag filtering
    # collapses chains of micro-pivots into MACRO legs of at least
    # `leg-min-pts` amplitude. Captures the moves the user actually
    # wants to trade (multi-bar trends), not the per-bar wiggles.
    pivots = []
    # PIVOTS from chosen TF mean (or 5s raw close); ENTRY sniped on 5s
    # price intra-leg. '5s' uses raw OHLCV close — most granular, no smoothing.
    leg_source_tf = args.leg_tf
    feats_for_legs = None
    pivot_dt_arr = None
    if MEAN_COLS.get(leg_source_tf) in feats.columns:
        m1 = feats[MEAN_COLS[leg_source_tf]].values.astype(np.float64)
        pivot_dt_arr = feats_dt
        # Slope lookback scaled to TF
        LB_swing = {'5s': 6, '15s': 8, '1m': 24, '5m': 12, '15m': 12}.get(
            leg_source_tf, 12)
    else:
        m1 = None

    if m1 is not None:
        s1 = np.full_like(m1, np.nan)
        if len(m1) > LB_swing:
            s1[LB_swing:] = (m1[LB_swing:] - m1[:-LB_swing]) / LB_swing
        # Raw pivots from 5m-slope sign change
        raw = []
        for i in range(1, len(s1)):
            if not (np.isfinite(s1[i]) and np.isfinite(s1[i-1])):
                continue
            if s1[i-1] > 0 and s1[i] <= 0:
                raw.append((i, m1[i], 'high'))
            elif s1[i-1] < 0 and s1[i] >= 0:
                raw.append((i, m1[i], 'low'))

        # ZigZag filter: walk raw pivots, accept a NEW pivot only when
        # the move from the last accepted pivot exceeds leg-min-pts.
        # In a continuation (same direction), update the last accepted
        # pivot if the new candidate is more extreme.
        threshold = float(args.leg_min_pts)
        for r in raw:
            if not pivots:
                pivots.append(r)
                continue
            last = pivots[-1]
            if r[2] == last[2]:
                # Same type — keep the more extreme one as the running last
                if (r[2] == 'high' and r[1] > last[1]) or (
                        r[2] == 'low' and r[1] < last[1]):
                    pivots[-1] = r
            else:
                # Opposite type — accept only if move ≥ threshold
                if abs(r[1] - last[1]) >= threshold:
                    pivots.append(r)
                # else: ignore, awaiting a bigger reversal
        # Draw the legs colored by DIRECTION:
        # green = up-leg (LONG opportunity), red = down-leg (SHORT opportunity)
        for j in range(1, len(pivots)):
            i_a, m_a, t_a = pivots[j-1]
            i_b, m_b, t_b = pivots[j]
            if t_a == t_b:
                continue
            up = (m_b > m_a)
            leg_color = 'limegreen' if up else 'red'
            ax.plot([pivot_dt_arr[i_a], pivot_dt_arr[i_b]], [m_a, m_b],
                         color=leg_color, lw=3.2, alpha=0.85, zorder=10,
                         solid_capstyle='round')
            # Pivot dot
            ax.scatter([pivot_dt_arr[i_b]], [m_b], color=leg_color, s=22,
                            zorder=11, edgecolors='white', linewidths=0.6)

        # Wedge connectors — keep subtle so they don't clash with the legs.
        highs = [(pivot_dt_arr[i], m) for (i, m, t) in pivots if t == 'high']
        lows  = [(pivot_dt_arr[i], m) for (i, m, t) in pivots if t == 'low']
        if len(highs) >= 2:
            ax.plot([h[0] for h in highs], [h[1] for h in highs],
                         color='dimgray', lw=1.0, linestyle=':',
                         alpha=0.7, zorder=8, label='swing-highs')
        if len(lows) >= 2:
            ax.plot([l[0] for l in lows], [l[1] for l in lows],
                         color='dimgray', lw=1.0, linestyle=':',
                         alpha=0.7, zorder=8, label='swing-lows')
        print(f'ZigZag swing pivots ({leg_source_tf}-mean source): {len(pivots)} '
                  f'({sum(1 for p in pivots if p[2]=="high")} highs, '
                  f'{sum(1 for p in pivots if p[2]=="low")} lows)  '
                  f'leg_min={args.leg_min_pts:.1f} pts')

        # Leg amplitude distribution (the "mode of capture")
        amps = []
        green_amps = []
        red_amps = []
        for j in range(1, len(pivots)):
            a = abs(pivots[j][1] - pivots[j-1][1])
            amps.append(a)
            if pivots[j][1] > pivots[j-1][1]:
                green_amps.append(a)
            else:
                red_amps.append(a)
        if amps:
            v = np.asarray(amps)
            # Histogram mode with 4-pt bins
            bin_width = 4.0
            lo, hi = float(v.min()), float(v.max())
            n_bins = max(1, int(np.ceil((hi - lo) / bin_width)))
            edges = np.linspace(lo, hi, n_bins + 1)
            counts, _ = np.histogram(v, bins=edges)
            j_max = int(np.argmax(counts))
            mode_amp = (edges[j_max] + edges[j_max + 1]) / 2
            print(f'Leg amplitude (pts): mode={mode_amp:.1f}  '
                      f'median={float(np.median(v)):.1f}  '
                      f'mean={float(v.mean()):.1f}  '
                      f'q90={float(np.quantile(v, 0.90)):.1f}  '
                      f'max={float(v.max()):.1f}')
            print(f'  GREEN legs (long ops, {len(green_amps)}): '
                      f'mean={float(np.mean(green_amps)) if green_amps else 0:.1f}  '
                      f'max={float(np.max(green_amps)) if green_amps else 0:.1f} pts')
            print(f'  RED legs   (short ops, {len(red_amps)}): '
                      f'mean={float(np.mean(red_amps)) if red_amps else 0:.1f}  '
                      f'max={float(np.max(red_amps)) if red_amps else 0:.1f} pts')

    # 15m-mean inflection markers DROPPED — leg pivot dots cover this.
    if False and '15m' in slopes_by_tf:
        s15 = slopes_by_tf['15m']
        if MEAN_COLS.get('15m') in feats.columns:
            mean15 = feats[MEAN_COLS['15m']].values
            curv15 = np.full_like(s15, np.nan)
            if len(s15) > LB:
                curv15[LB:] = (s15[LB:] - s15[:-LB]) / LB
            # Threshold = Q75 of |curvature|
            curv_thr = float(np.nanquantile(np.abs(curv15), 0.75))
            n_max, n_min = 0, 0
            for i in range(1, len(s15)):
                if not (np.isfinite(s15[i]) and np.isfinite(s15[i-1])
                              and np.isfinite(curv15[i])):
                    continue
                if abs(curv15[i]) < curv_thr:
                    continue
                if s15[i-1] > 0 and s15[i] <= 0:
                    ax.scatter([feats_dt[i]], [mean15[i]], marker='v',
                                    color='black', s=80, zorder=8,
                                    edgecolors='black', linewidths=1.4)
                    n_max += 1
                elif s15[i-1] < 0 and s15[i] >= 0:
                    ax.scatter([feats_dt[i]], [mean15[i]], marker='^',
                                    color='black', s=80, zorder=8,
                                    edgecolors='black', linewidths=1.4)
                    n_min += 1
            print(f'15m-mean inflections marked: {n_max} max(▼ red), '
                      f'{n_min} min(▲ green)  curv_thr=Q75={curv_thr:.6f}')

    # ── Confirmed-fade entry detection ─────────────────────────────────
    # WATCH:   5s close crosses OUTSIDE band_tf ±k·σ
    # CONFIRM: price stays outside for cf_confirm_bars consecutive bars
    # ENTRY:   trigger fade at confirmation bar, target = band_tf mean
    if False and args.confirmed_fade:
        SIGMA_LOOKUP = {
            '1m':  'L2_1m_price_sigma_15',
            '5m':  'L2_5m_price_sigma_9',
            '15m': 'L2_15m_price_sigma_12',
            '1h':  'L2_1h_price_sigma_12',
        }
        cf_mean_col = MEAN_COLS.get(args.cf_band_tf) or {
            '1h': 'L2_1h_price_mean_12'}.get(args.cf_band_tf)
        cf_sigma_col = SIGMA_LOOKUP.get(args.cf_band_tf)
        target_mean_col = MEAN_COLS.get(args.cf_target_tf) or {
            '1h': 'L2_1h_price_mean_12'}.get(args.cf_target_tf)
        if (cf_mean_col in feats.columns and cf_sigma_col in feats.columns
                  and target_mean_col in feats.columns):
            # Align band to OHLCV timestamps via searchsorted
            f_ts = feats['timestamp'].astype('int64').values
            band_mean = feats[cf_mean_col].values
            band_sigma = feats[cf_sigma_col].values
            oh_ts_int = oh_ts.astype(np.int64)
            idx_map = np.searchsorted(f_ts, oh_ts_int, side='right') - 1
            idx_map = np.clip(idx_map, 0, len(f_ts) - 1)
            mean_at_oh = band_mean[idx_map]
            sigma_at_oh = band_sigma[idx_map]
            target_at_oh = feats[target_mean_col].values[idx_map]
            band_hi_oh = mean_at_oh + args.cf_band_k * sigma_at_oh
            band_lo_oh = mean_at_oh - args.cf_band_k * sigma_at_oh

            close = ohlcv['close'].values
            above = close > band_hi_oh
            below = close < band_lo_oh
            # Run-length detection
            n_short = 0
            n_long = 0
            n_target_short = 0
            n_target_long = 0
            consec_above = 0
            consec_below = 0
            in_short = False
            in_long = False
            short_target_pending = None
            long_target_pending = None
            for i in range(len(close)):
                if not (np.isfinite(band_hi_oh[i]) and np.isfinite(band_lo_oh[i])):
                    continue
                # Update consecutive-bar counters
                consec_above = consec_above + 1 if above[i] else 0
                consec_below = consec_below + 1 if below[i] else 0
                # Confirm short: above for N bars
                if (consec_above >= args.cf_confirm_bars and not in_short
                          and not in_long):
                    ax.scatter([oh_dt[i]], [close[i]], marker='v',
                                    color='red', s=100, zorder=9,
                                    edgecolors='black', linewidths=1.2)
                    n_short += 1
                    in_short = True
                    short_target_pending = mean_at_oh[i]
                # Confirm long: below for N bars
                elif (consec_below >= args.cf_confirm_bars and not in_long
                            and not in_short):
                    ax.scatter([oh_dt[i]], [close[i]], marker='^',
                                    color='lime', s=100, zorder=9,
                                    edgecolors='black', linewidths=1.2)
                    n_long += 1
                    in_long = True
                    long_target_pending = mean_at_oh[i]
                # Target hit: short trade closes when price reaches mean
                if in_short and close[i] <= mean_at_oh[i]:
                    ax.scatter([oh_dt[i]], [close[i]], marker='o',
                                    color='white', s=70, zorder=9,
                                    edgecolors='red', linewidths=1.4)
                    n_target_short += 1
                    in_short = False
                    short_target_pending = None
                # Target hit: long trade closes when price reaches mean
                elif in_long and close[i] >= mean_at_oh[i]:
                    ax.scatter([oh_dt[i]], [close[i]], marker='o',
                                    color='white', s=70, zorder=9,
                                    edgecolors='lime', linewidths=1.4)
                    n_target_long += 1
                    in_long = False
                    long_target_pending = None
            print(f'Confirmed-fade ({args.cf_band_tf} ±{args.cf_band_k}σ, '
                      f'{args.cf_confirm_bars}-bar confirm): '
                      f'{n_short} shorts(▼), {n_long} longs(▲), '
                      f'{n_target_short + n_target_long} target-hits(○)')

    # If split, set xlim per column and refit y-axis per visible data range.
    def _fit_y_to_xlim(ax, padding=0.05):
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

    if split_dt is not None:
        day_start = oh_dt[0]
        day_end = oh_dt[-1]
        price_axes[0].set_xlim(day_start, split_dt)
        price_axes[1].set_xlim(split_dt, day_end)
        _fit_y_to_xlim(price_axes[0])
        _fit_y_to_xlim(price_axes[1])
        price_axes[1].yaxis.tick_right()
        price_axes[1].yaxis.set_label_position('right')
        # Drop legend on right (already on left)
        leg = price_axes[1].get_legend()
        if leg is not None:
            leg.remove()
        price_axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        price_axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        price_axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        price_axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=1))

    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, f'{args.day}_regression_means.png')
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_png}')


if __name__ == '__main__':
    main()
