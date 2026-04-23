"""
Visual comparison: CURRENT system (reg-flip + residual-flip + sniper) vs
IDEAL (cusp → mean cross) on a single day's 1m chart.

Chart layout (3 panels, shared x-axis):
  Panel 1: price + regression + ±1σ/±2σ/±3σ + CURRENT strategy trades
  Panel 2: same price+bands + CUSP strategy trades (cusp-to-mean-cross)
  Panel 3: 1m_z_se residual with ±1/±2/±3 thresholds

Each panel's trades drawn as:
  - Entry marker (green ▲ for LONG, red ▼ for SHORT)
  - Exit marker (×)
  - Line from entry→exit colored green (win) or red (loss)
  - Dollar value annotation at exit

Usage:
    python tools/chart_strategy_comparison.py --day 2025_06_09

Output: charts/strategy_compare_<day>.png
"""
import os
import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_1M_DIR = 'DATA/ATLAS/1m'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
DOLLAR_PER_POINT = 2.0
REG_WINDOW = 60


def rolling_fit_with_se(closes, window):
    """Return (fitted, se, slope) arrays.
    se = stderr of residuals in window.
    slope = OLS beta (points per 1m bar).
    """
    n = len(closes)
    fitted = np.full(n, np.nan)
    se = np.full(n, np.nan)
    slopes = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        x = np.arange(window, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            continue
        slope = (dx * (y - ym)).sum() / denom
        intercept = ym - slope * xm
        fit_last = intercept + slope * (window - 1)
        fits = intercept + slope * x
        resid = y - fits
        sigma = np.sqrt((resid ** 2).sum() / max(window - 2, 1))
        fitted[i] = fit_last
        se[i] = sigma
        slopes[i] = slope
    return fitted, se, slopes


def sim_current_1m(closes, fitted, residuals, r_reg_pts=4.0):
    """Simplified version of current physics-exit strategy using 1m bars.
    Entry: at each 1m close, if |residual|>=0.5 AND residual sign changed
    from previous bar (mini-cusp), try to enter. (Matches the spirit of
    '1s pivot + residual direction' condensed to 1m for visualization.)
    Exit: reg direction flipped AND residual sign flipped from entry sign.
    """
    n = len(closes)
    # Compute reg zigzag pivots
    from tools.pivot_physics_exit import zigzag_pivots_realtime
    reg_pivots = zigzag_pivots_realtime(fitted, r_reg_pts)

    trades = []
    in_pos = False
    entry_bar = 0
    entry_price = 0
    direction = None
    entry_res_sign = 0
    last_reg_pivot_bar = -1

    last_res = 0
    for t in range(n):
        r = residuals[t]
        if in_pos:
            # Check exit
            res_sign_flipped = (np.sign(r) != entry_res_sign and np.sign(r) != 0)
            reg_flip = False
            for k in range(last_reg_pivot_bar + 1, t + 1):
                if k in reg_pivots:
                    pivot_type = reg_pivots[k]
                    if (direction == 'LONG' and pivot_type == 'HIGH') or \
                       (direction == 'SHORT' and pivot_type == 'LOW'):
                        reg_flip = True
                        break
            if res_sign_flipped and reg_flip:
                exit_price = closes[t]
                pnl = ((exit_price - entry_price) if direction == 'LONG'
                       else (entry_price - exit_price)) * DOLLAR_PER_POINT
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': t,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'direction': direction, 'pnl': pnl,
                })
                in_pos = False
        else:
            # Entry check: residual sign CHANGED vs prior bar AND |r| >= 0.5
            if abs(r) >= 0.5 and np.sign(r) != np.sign(last_res) and np.sign(r) != 0:
                direction = 'LONG' if r < 0 else 'SHORT'
                entry_price = closes[t]
                entry_bar = t
                entry_res_sign = np.sign(r)
                last_reg_pivot_bar = -1
                for k in range(t, -1, -1):
                    if k in reg_pivots:
                        last_reg_pivot_bar = k
                        break
                in_pos = True
        last_res = r

    if in_pos:
        exit_price = closes[-1]
        pnl = ((exit_price - entry_price) if direction == 'LONG'
               else (entry_price - exit_price)) * DOLLAR_PER_POINT
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': n - 1,
            'entry_price': entry_price, 'exit_price': exit_price,
            'direction': direction, 'pnl': pnl,
        })
    return trades


def sim_cusp(closes, residuals, arm_threshold=1.5):
    """Ideal cusp strategy on 1m bars.
    State machine:
      UNARMED: waiting for |r| >= arm_threshold
      ARMED:   |r| >= arm_threshold, waiting for cusp (magnitude decrease)
      IN_POS:  entered on cusp, waiting for residual to cross 0
    """
    n = len(closes)
    trades = []
    state = 'UNARMED'
    peak_abs = 0
    entry_bar = 0
    entry_price = 0
    direction = None
    entry_res_sign = 0

    for t in range(n):
        r = residuals[t]
        if np.isnan(r):
            continue
        abs_r = abs(r)

        if state == 'UNARMED':
            if abs_r >= arm_threshold:
                state = 'ARMED'
                peak_abs = abs_r
        elif state == 'ARMED':
            if abs_r > peak_abs:
                peak_abs = abs_r
            elif abs_r < peak_abs - 0.05:  # small buffer to avoid noise
                # Cusp detected — enter
                direction = 'LONG' if r < 0 else 'SHORT'
                entry_price = closes[t]
                entry_bar = t
                entry_res_sign = np.sign(r)
                state = 'IN_POS'
        elif state == 'IN_POS':
            if np.sign(r) != entry_res_sign or abs_r < 0.05:
                # Residual crossed zero (or effectively zero) — exit
                exit_price = closes[t]
                pnl = ((exit_price - entry_price) if direction == 'LONG'
                       else (entry_price - exit_price)) * DOLLAR_PER_POINT
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': t,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'direction': direction, 'pnl': pnl,
                })
                state = 'UNARMED'
                peak_abs = 0

    if state == 'IN_POS':
        exit_price = closes[-1]
        pnl = ((exit_price - entry_price) if direction == 'LONG'
               else (entry_price - exit_price)) * DOLLAR_PER_POINT
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': n - 1,
            'entry_price': entry_price, 'exit_price': exit_price,
            'direction': direction, 'pnl': pnl,
        })
    return trades


def sim_trend_rm(closes, fitted, slopes, rm_retrace_pts=1.0,
                 min_vel=0.10, decel_frac=0.5):
    """Trend-follow on regression-mean cusps.

    Entry  — confirmed zigzag pivot on the RM (fitted) series.
             LOW pivot → RM just troughed → enter LONG
             HIGH pivot → RM just peaked → enter SHORT
             The pivot itself is only confirmed AFTER the RM retraces
             by rm_retrace_pts (small: RM is smooth so $0.50-$1 is enough).

    Exit   — velocity slowing signal:
             (a) |β| dropped below peak_|β| * decel_frac AND |β| < min_vel
                 → next pivot imminent, take it now
             (b) RM zigzag fired a new pivot against us (already too late
                 but exit)

    No stop-loss — rides the trend until decel.
    """
    from tools.pivot_physics_exit import zigzag_pivots_realtime

    # Zigzag pivots on the RM itself (dict: bar_idx -> 'HIGH'/'LOW')
    # Feed only the non-NaN portion of fitted
    n = len(closes)
    fit_clean = np.where(np.isnan(fitted), closes, fitted)  # pad NaN region
    rm_pivots = zigzag_pivots_realtime(fit_clean, rm_retrace_pts)

    trades = []
    in_pos = False
    direction = None
    entry_bar = 0
    entry_price = 0
    entry_sign = 0
    peak_abs_slope = 0

    for t in range(n):
        s = slopes[t]
        if np.isnan(s):
            # Still need to check pivots? Pivots require fitted — skip until valid
            continue
        sgn = 1 if s > 0 else (-1 if s < 0 else 0)
        abs_s = abs(s)

        if in_pos:
            if abs_s > peak_abs_slope:
                peak_abs_slope = abs_s
            decelerated = (peak_abs_slope > 0 and
                           abs_s < peak_abs_slope * decel_frac and
                           abs_s < min_vel)
            # Adverse RM pivot at/before bar t (missed exit — take it)
            adverse_pivot = False
            if t in rm_pivots:
                ptype = rm_pivots[t]
                if (direction == 'LONG' and ptype == 'HIGH') or \
                   (direction == 'SHORT' and ptype == 'LOW'):
                    adverse_pivot = True

            if decelerated or adverse_pivot:
                exit_price = closes[t]
                pnl = ((exit_price - entry_price) if direction == 'LONG'
                       else (entry_price - exit_price)) * DOLLAR_PER_POINT
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': t,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'direction': direction, 'pnl': pnl,
                    'exit_reason': 'DECEL' if decelerated else 'ADV_PIVOT',
                })
                in_pos = False
                peak_abs_slope = 0
        else:
            # Entry: fresh RM pivot at this bar, trade in direction of new trend
            if t in rm_pivots:
                ptype = rm_pivots[t]
                direction = 'LONG' if ptype == 'LOW' else 'SHORT'
                entry_price = closes[t]
                entry_bar = t
                entry_sign = 1 if direction == 'LONG' else -1
                peak_abs_slope = abs_s
                in_pos = True

    if in_pos:
        exit_price = closes[-1]
        pnl = ((exit_price - entry_price) if direction == 'LONG'
               else (entry_price - exit_price)) * DOLLAR_PER_POINT
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': n - 1,
            'entry_price': entry_price, 'exit_price': exit_price,
            'direction': direction, 'pnl': pnl,
            'exit_reason': 'EOD',
        })
    return trades


def draw_trades_on_panel(ax, dts, trades, label):
    """Draw entry/exit markers and connecting lines on a price axis."""
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total = sum(t['pnl'] for t in trades)
    wr = len(wins) / max(len(trades), 1) * 100
    ax.text(0.01, 0.97, f'{label}: {len(trades)} trades  WR {wr:.0f}%  '
            f'Total \\${total:+,.0f}',
            transform=ax.transAxes, fontsize=11, weight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    for t in trades:
        entry_dt = dts[t['entry_bar']]
        exit_dt = dts[t['exit_bar']]
        color = 'tab:green' if t['pnl'] > 0 else 'tab:red'
        alpha = 0.85 if t['pnl'] > 0 else 0.4
        # Connecting line
        ax.plot([entry_dt, exit_dt], [t['entry_price'], t['exit_price']],
                color=color, linewidth=1.4 if t['pnl'] > 0 else 0.8,
                alpha=alpha, zorder=2)
        # Entry marker
        marker = '^' if t['direction'] == 'LONG' else 'v'
        ax.scatter([entry_dt], [t['entry_price']],
                   marker=marker, s=90 if t['pnl'] > 0 else 35,
                   c=color, edgecolors='black', linewidths=0.6,
                   alpha=alpha, zorder=4)
        # Exit marker
        ax.scatter([exit_dt], [t['exit_price']],
                   marker='x', s=50, c=color, linewidths=1.3,
                   alpha=alpha, zorder=4)


def plot_regression_background(ax, dts, closes, fitted, se, title):
    ax.plot(dts, closes, color='black', linewidth=0.9, alpha=0.85,
            label='Price (1m)')
    ax.plot(dts, fitted, color='tab:blue', linewidth=2.2,
            label='Regression mean (60-bar)')
    for k, (color, ls, alpha) in enumerate([
        ('tab:orange', ':', 0.55),      # ±1σ
        ('tab:red', '--', 0.7),         # ±2σ
        ('darkred', '-', 0.75),         # ±3σ
    ]):
        k_sigma = k + 1
        ax.plot(dts, fitted + k_sigma * se, color=color, linewidth=0.8,
                linestyle=ls, alpha=alpha)
        ax.plot(dts, fitted - k_sigma * se, color=color, linewidth=0.8,
                linestyle=ls, alpha=alpha)
    ax.fill_between(dts, fitted - se, fitted + se, color='tab:orange', alpha=0.03)
    ax.fill_between(dts, fitted - 2 * se, fitted + 2 * se, color='tab:red', alpha=0.02)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('MNQ price', fontsize=11)
    ax.grid(True, alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--r-reg', type=float, default=4.0,
                    help='Regression zigzag $ for current strategy (default 4)')
    ap.add_argument('--arm', type=float, default=1.5,
                    help='Cusp arm threshold |z| (default 1.5)')
    ap.add_argument('--rm-retrace', type=float, default=1.0,
                    help='TREND_RM: RM zigzag retracement for pivot '
                         'confirmation, in points (default 1.0 pt = $2)')
    ap.add_argument('--min-vel', type=float, default=0.10,
                    help='TREND_RM: |slope| exit-arm threshold '
                         '(pts/1m bar, default 0.10)')
    ap.add_argument('--decel-frac', type=float, default=0.5,
                    help='TREND_RM: exit when |slope| < peak * decel_frac '
                         'AND |slope| < min_vel (default 0.5)')
    args = ap.parse_args()

    min_path = os.path.join(ATLAS_1M_DIR, f'{args.day}.parquet')
    feat_path = os.path.join(FEATURES_5S_DIR, f'{args.day}.parquet')

    df_min = pd.read_parquet(min_path).sort_values('timestamp').reset_index(drop=True)
    closes = df_min['close'].values.astype(np.float64)
    ts = df_min['timestamp'].values.astype(np.int64)
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]

    df_feat = pd.read_parquet(feat_path).sort_values('timestamp').reset_index(drop=True)
    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    res_feat = df_feat['1m_z_se'].values.astype(np.float64)
    idx = np.searchsorted(ts_feat, ts, side='right') - 1
    idx = np.clip(idx, 0, len(ts_feat) - 1)
    residuals = res_feat[idx]

    fitted, se, slopes = rolling_fit_with_se(closes, REG_WINDOW)

    # Run all three strategies
    current_trades = sim_current_1m(closes, fitted, residuals, args.r_reg)
    cusp_trades = sim_cusp(closes, residuals, args.arm)
    trend_trades = sim_trend_rm(closes, fitted, slopes,
                                rm_retrace_pts=args.rm_retrace,
                                min_vel=args.min_vel,
                                decel_frac=args.decel_frac)

    print(f'Day: {args.day}')
    print(f'CURRENT (reg zigzag ${args.r_reg}): {len(current_trades)} trades, '
          f'total ${sum(t["pnl"] for t in current_trades):+,.0f}')
    print(f'CUSP (arm |z|>={args.arm}, exit at mean cross): '
          f'{len(cusp_trades)} trades, '
          f'total ${sum(t["pnl"] for t in cusp_trades):+,.0f}')
    print(f'TREND_RM (RM cusp entry, decel exit; min_vel={args.min_vel}): '
          f'{len(trend_trades)} trades, '
          f'total ${sum(t["pnl"] for t in trend_trades):+,.0f}')

    # ── CHART ──
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(22, 18),
        gridspec_kw={'height_ratios': [2.6, 2.6, 2.6, 1.0, 1.0]}, sharex=True)

    plot_regression_background(
        ax1, dts, closes, fitted, se,
        f'{args.day} — CURRENT strategy '
        f'(reg-zigzag \\${args.r_reg} flip + residual flip + 30s sniper)')
    draw_trades_on_panel(ax1, dts, current_trades, 'CURRENT')

    plot_regression_background(
        ax2, dts, closes, fitted, se,
        f'{args.day} — CUSP strategy (ideal: arm at |z|>={args.arm}, '
        f'enter at cusp, exit at mean-cross)')
    draw_trades_on_panel(ax2, dts, cusp_trades, 'CUSP')

    plot_regression_background(
        ax3, dts, closes, fitted, se,
        f'{args.day} — TREND_RM (RM cusp entry on slope flip, '
        f'exit when velocity slows; min_vel={args.min_vel} pts/bar)')
    draw_trades_on_panel(ax3, dts, trend_trades, 'TREND_RM')

    # Residual panel
    ax4.plot(dts, residuals, color='tab:purple', linewidth=1.0,
             label='1m_z_se (residual)')
    ax4.axhline(0, color='black', linewidth=0.7)
    for sigma, color in [(1, 'tab:orange'), (2, 'tab:red'), (3, 'darkred')]:
        ax4.axhline(sigma, color=color, linestyle=':' if sigma == 1 else '--',
                    linewidth=0.6, alpha=0.7)
        ax4.axhline(-sigma, color=color, linestyle=':' if sigma == 1 else '--',
                     linewidth=0.6, alpha=0.7)
    ax4.axhline(args.arm, color='tab:green', linestyle='-', linewidth=0.8, alpha=0.5,
                label=f'Cusp arm (±{args.arm})')
    ax4.axhline(-args.arm, color='tab:green', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.fill_between(dts, 0, residuals, where=(residuals > 0),
                      color='tab:red', alpha=0.12)
    ax4.fill_between(dts, 0, residuals, where=(residuals < 0),
                      color='tab:green', alpha=0.12)
    ax4.set_ylabel('residual (z)', fontsize=11)
    ax4.set_ylim(-4.5, 4.5)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.25)

    # Slope (β) panel
    ax5.plot(dts, slopes, color='tab:brown', linewidth=1.0,
             label='RM slope β (pts/bar)')
    ax5.axhline(0, color='black', linewidth=0.7)
    ax5.axhline(args.min_vel, color='tab:green', linestyle='--',
                linewidth=0.7, alpha=0.7, label=f'±min_vel ({args.min_vel})')
    ax5.axhline(-args.min_vel, color='tab:green', linestyle='--',
                linewidth=0.7, alpha=0.7)
    ax5.fill_between(dts, 0, slopes, where=(np.asarray(slopes) > 0),
                      color='tab:green', alpha=0.15)
    ax5.fill_between(dts, 0, slopes, where=(np.asarray(slopes) < 0),
                      color='tab:red', alpha=0.15)
    ax5.set_ylabel('RM β (pts/bar)', fontsize=11)
    ax5.set_xlabel('Time (UTC)', fontsize=11)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.25)

    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    out_path = f'charts/strategy_compare_{args.day}.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
