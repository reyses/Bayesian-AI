"""Plot the three-level anchor framework for a single day:
    1. 5s close (the price, the FAST oscillator we trade)
    2. 15m M_close (the MEDIUM CRM context anchor)
    3. 1h M_high + 1h M_low with +/-2sigma and +/-3sigma envelopes (SLOW HL RM)

Used to visualize the multi-scale anchor structure under which the
Bayesian probability table operates.

USAGE
    python tools/chart_anchors_3level.py --day 2025_10_29
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

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv


def compute_anchor(tf: str, day: str, ts_5s: np.ndarray, window: int,
                    column: str = 'close') -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (M, S) ffilled to 5s grid using rolling-window regression on the
    chosen OHLCV column at the given TF."""
    period_s_map = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400}
    if tf not in period_s_map:
        return None
    period_s = period_s_map[tf]
    oh = _load_tf_ohlcv(tf, day)
    if oh.empty:
        return None
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
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_5s = _load_5s(args.day)
    if df_5s.empty:
        print(f'No 5s data for {args.day}')
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    dt_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]
    close_5s = df_5s['close'].values.astype(np.float64)

    # Compute anchors
    M_15m, S_15m = compute_anchor('15m', args.day, ts_5s, window=12, column='close')
    Mh_1h, Sh_1h = compute_anchor('1h', args.day, ts_5s, window=12, column='high')
    Ml_1h, Sl_1h = compute_anchor('1h', args.day, ts_5s, window=12, column='low')
    Mc_1h, Sc_1h = compute_anchor('1h', args.day, ts_5s, window=12, column='close')

    # 3sigma band-touch events (where price crossed 1h HL +/-3sigma)
    z_high = (close_5s - Mh_1h) / Sh_1h
    z_low  = (close_5s - Ml_1h) / Sl_1h
    above_high_3 = z_high >= 3.0
    below_low_3  = z_low <= -3.0

    # Render
    fig, ax = plt.subplots(figsize=(22, 10))

    # 1h SLOW envelope: upper rail (M_high)
    ax.plot(dt_5s, Mh_1h, color='#43A047', lw=1.4, alpha=0.85,
              label='1h M_high (slow upper rail)')
    ax.fill_between(dt_5s, Mh_1h - 2*Sh_1h, Mh_1h + 2*Sh_1h,
                      color='#43A047', alpha=0.07,
                      label='1h M_high ± 2σ')
    ax.fill_between(dt_5s, Mh_1h + 2*Sh_1h, Mh_1h + 3*Sh_1h,
                      color='#43A047', alpha=0.13,
                      label='1h M_high +2σ to +3σ (rally trigger zone)')
    ax.plot(dt_5s, Mh_1h + 3*Sh_1h, color='#1B5E20', lw=0.8, ls='--', alpha=0.7,
              label='1h M_high + 3σ (rally trigger)')

    # 1h SLOW envelope: lower rail (M_low)
    ax.plot(dt_5s, Ml_1h, color='#E53935', lw=1.4, alpha=0.85,
              label='1h M_low (slow lower rail)')
    ax.fill_between(dt_5s, Ml_1h - 2*Sl_1h, Ml_1h + 2*Sl_1h,
                      color='#E53935', alpha=0.07,
                      label='1h M_low ± 2σ')
    ax.fill_between(dt_5s, Ml_1h - 3*Sl_1h, Ml_1h - 2*Sl_1h,
                      color='#E53935', alpha=0.13,
                      label='1h M_low -2σ to -3σ (crash trigger zone)')
    ax.plot(dt_5s, Ml_1h - 3*Sl_1h, color='#B71C1C', lw=0.8, ls='--', alpha=0.7,
              label='1h M_low - 3σ (crash trigger)')

    # 15m MEDIUM anchor: M_close
    ax.plot(dt_5s, M_15m, color='#1E88E5', lw=1.6, alpha=0.95,
              label='15m M_close (medium CRM context)')

    # Optional: show 1h M_close as reference
    ax.plot(dt_5s, Mc_1h, color='#7E57C2', lw=0.8, ls=':', alpha=0.6,
              label='1h M_close (reference)')

    # 5s FAST oscillator: raw close
    ax.plot(dt_5s, close_5s, color='black', lw=0.45, alpha=0.85,
              label='5s close (fast oscillator)')

    # Mark band-touch events
    if above_high_3.any():
        ax.scatter(np.array(dt_5s)[above_high_3], close_5s[above_high_3],
                    color='#43A047', s=4, alpha=0.5, marker='^',
                    label=f'5s close > 1h M_high+3σ  (n={int(above_high_3.sum())} ticks)',
                    zorder=5)
    if below_low_3.any():
        ax.scatter(np.array(dt_5s)[below_low_3], close_5s[below_low_3],
                    color='#E53935', s=4, alpha=0.5, marker='v',
                    label=f'5s close < 1h M_low-3σ  (n={int(below_low_3.sum())} ticks)',
                    zorder=5)

    n_above = int(above_high_3.sum())
    n_below = int(below_low_3.sum())
    n_total = len(close_5s)
    pct_above = 100 * n_above / n_total
    pct_below = 100 * n_below / n_total

    ax.set_title(
        f'{args.day} — 3-LEVEL ANCHOR FRAMEWORK\n'
        f'FAST (black 5s close)  vs  MEDIUM (blue 15m M_close)  vs  '
        f'SLOW (green/red 1h M_high/low ± σ envelopes)\n'
        f'Time spent past 1h HL ±3σ:  '
        f'above_high={pct_above:.2f}%  below_low={pct_below:.2f}%  '
        f'(total bars: {n_total:,})',
        fontsize=12)
    ax.set_xlabel('time (UTC)'); ax.set_ylabel('price')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f'anchors_3level_{args.day}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Chart -> {out_path}')

    # Also print summary stats
    print(f'\nDay summary for {args.day}:')
    print(f'  5s bars:           {n_total:,}')
    print(f'  Price range:       {close_5s.min():.2f} to {close_5s.max():.2f}  '
           f'({close_5s.max() - close_5s.min():.2f} pts)')
    print(f'  Net day move:      {close_5s[-1] - close_5s[0]:+.2f} pts')
    if M_15m is not None:
        print(f'  15m M_close drift: {M_15m[-1] - M_15m[~np.isnan(M_15m)][0]:+.2f} pts')
    if Mh_1h is not None and Ml_1h is not None:
        print(f'  1h M_high range:   {np.nanmin(Mh_1h):.2f} to {np.nanmax(Mh_1h):.2f}')
        print(f'  1h M_low range:    {np.nanmin(Ml_1h):.2f} to {np.nanmax(Ml_1h):.2f}')
        avg_band_width = float(np.nanmean(Mh_1h - Ml_1h))
        print(f'  Avg HL band width: {avg_band_width:.2f} pts  (1h M_high - 1h M_low)')
        avg_se = float(np.nanmean(Sh_1h + Sl_1h) / 2)
        print(f'  Avg HL sigma:          {avg_se:.2f} pts')
    print(f'  Ticks past +3sigma rally trigger:  {n_above:,} ({pct_above:.2f}%)')
    print(f'  Ticks past -3sigma crash trigger:  {n_below:,} ({pct_below:.2f}%)')


if __name__ == '__main__':
    main()
