"""
Chart trades from the 1s-pivot forward pass on a single day.

Three-panel chart:
  (1) Price (1m close) + regression mean + trade entries/exits
  (2) Cumulative PnL equity curve throughout the day
  (3) 1m_z_se residual (to see the signal behind each entry)

Markers:
  - Green ▲ = LONG entry that won (TP hit)
  - Red ▼ = SHORT entry that won
  - Gray ▲/▼ = LOSING entries
  - Line from entry to exit, colored by outcome

Usage:
    python tools/chart_1s_trades.py --day 2025_06_09
    python tools/chart_1s_trades.py --day 2025_04_09 --tp 30 --sl 3

Output: charts/trades_1s_<day>_r{r}_tp{tp}_sl{sl}.png
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
from tools.pivot_1s_forward import load_day, simulate_day


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
DOLLAR_PER_POINT = 2.0


def rolling_fit(closes, window=60):
    n = len(closes)
    fitted = np.full(n, np.nan)
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
        fitted[i] = intercept + slope * (window - 1)
    return fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--r-confirm', type=float, default=2.0)
    ap.add_argument('--tp', type=float, default=20.0)
    ap.add_argument('--sl', type=float, default=3.0)
    ap.add_argument('--min-res-strength', type=float, default=0.5)
    args = ap.parse_args()

    sec_path = os.path.join(ATLAS_1S_DIR, f'{args.day}.parquet')
    min_path = os.path.join(ATLAS_1M_DIR, f'{args.day}.parquet')
    feat_path = os.path.join(FEATURES_5S_DIR, f'{args.day}.parquet')
    for p in (sec_path, min_path, feat_path):
        if not os.path.exists(p):
            print(f'Missing: {p}')
            return

    # Load 1s for sim, 1m for background chart
    closes_1s, highs_1s, lows_1s, ts_1s, residuals_1s = load_day(sec_path, feat_path)
    df_1m = pd.read_parquet(min_path).sort_values('timestamp').reset_index(drop=True)
    closes_1m = df_1m['close'].values.astype(np.float64)
    ts_1m = df_1m['timestamp'].values.astype(np.int64)
    dts_1m = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts_1m]
    fitted_1m = rolling_fit(closes_1m, 60)

    # Also get 1m residual for bottom panel
    df_feat = pd.read_parquet(feat_path).sort_values('timestamp').reset_index(drop=True)
    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    res_feat = df_feat['1m_z_se'].values.astype(np.float64) if '1m_z_se' in df_feat.columns else np.zeros(len(df_feat))
    idx_1m_res = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx_1m_res = np.clip(idx_1m_res, 0, len(ts_feat) - 1)
    res_1m = res_feat[idx_1m_res]

    # Run simulation
    r_pts = args.r_confirm / DOLLAR_PER_POINT
    tp_pts = args.tp / DOLLAR_PER_POINT
    sl_pts = args.sl / DOLLAR_PER_POINT
    trades, n_pivots = simulate_day(closes_1s, highs_1s, lows_1s, ts_1s,
                                     residuals_1s, r_pts, tp_pts, sl_pts,
                                     args.min_res_strength)

    if not trades:
        print('No trades generated — try different params.')
        return

    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] < 0)
    wr = wins / max(wins + losses, 1) * 100
    print(f'{args.day}: {len(trades)} trades, {n_pivots} pivots, '
          f'WR {wr:.1f}%, total ${total_pnl:+.2f}')

    # Build chart
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(20, 12),
        gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    # ── PANEL 1: Price + regression + trades ──
    ax1.plot(dts_1m, closes_1m, color='black', linewidth=0.9,
             label='Price (1m close)', alpha=0.85)
    ax1.plot(dts_1m, fitted_1m, color='tab:blue', linewidth=1.8,
             label='Regression mean (60-bar)', alpha=0.75)

    # Trades
    for t in trades:
        entry_dt = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc)
        exit_dt = datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc)
        win = t['pnl'] > 0
        color = 'tab:green' if win else 'tab:red'
        alpha = 0.8 if win else 0.3
        # Connection line
        ax1.plot([entry_dt, exit_dt],
                 [t['entry_price'], t['exit_price']],
                 color=color, linewidth=0.7 if win else 0.5,
                 alpha=alpha, zorder=2)
        # Entry marker
        marker = '^' if t['direction'] == 'LONG' else 'v'
        edge_color = 'black' if win else 'gray'
        ax1.scatter([entry_dt], [t['entry_price']],
                    marker=marker, s=40 if win else 15,
                    c=color, edgecolors=edge_color,
                    linewidths=0.5, alpha=alpha, zorder=3)

    ax1.set_ylabel('MNQ price', fontsize=12)
    # Escape $ for matplotlib (avoid mathtext parsing)
    title = (f'{args.day} - 1s-pivot trades '
             f'(r=\\${args.r_confirm}, TP=\\${args.tp}, SL=\\${args.sl})  |  '
             f'{len(trades)} trades  WR {wr:.1f}%  '
             f'Total \\${total_pnl:+,.0f}')
    ax1.set_title(title, fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.25)

    # Legend for trade markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:green',
               markeredgecolor='black', markersize=10, label='LONG win'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='tab:red',
               markeredgecolor='black', markersize=10, label='SHORT win'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:green',
               markeredgecolor='gray', markersize=5, alpha=0.4, label='LONG loss'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='tab:red',
               markeredgecolor='gray', markersize=5, alpha=0.4, label='SHORT loss'),
    ]
    ax1.legend(handles=legend_elements + [
        Line2D([0], [0], color='black', linewidth=0.9, label='Price (1m)'),
        Line2D([0], [0], color='tab:blue', linewidth=1.8, label='Regression mean'),
    ], loc='upper left', fontsize=9, ncol=2)

    # ── PANEL 2: Cumulative PnL equity curve ──
    # Sort trades by exit timestamp
    sorted_trades = sorted(trades, key=lambda t: t['exit_ts'])
    equity_ts = [datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc)
                 for t in sorted_trades]
    equity = np.cumsum([t['pnl'] for t in sorted_trades])

    ax2.plot(equity_ts, equity, color='tab:purple', linewidth=1.2,
             label='Cumulative $ PnL')
    ax2.fill_between(equity_ts, 0, equity,
                      where=(equity >= 0), color='tab:green', alpha=0.15)
    ax2.fill_between(equity_ts, 0, equity,
                      where=(equity < 0), color='tab:red', alpha=0.15)
    ax2.axhline(0, color='black', linewidth=0.6)
    ax2.set_ylabel('Cumulative PnL ($)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── PANEL 3: 1m residual z_se ──
    dts_feat = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts_1m]
    ax3.plot(dts_feat, res_1m, color='tab:orange', linewidth=1.0,
             label='1m_z_se (residual)')
    ax3.axhline(0, color='gray', linewidth=0.5)
    for lvl, col in [(0.5, 'tab:red'), (-0.5, 'tab:green'),
                     (2.0, 'darkred'), (-2.0, 'darkgreen')]:
        ax3.axhline(lvl, color=col, linestyle=':', linewidth=0.6, alpha=0.6)
    ax3.fill_between(dts_feat, 0, res_1m, where=(res_1m > 0),
                      color='tab:red', alpha=0.10)
    ax3.fill_between(dts_feat, 0, res_1m, where=(res_1m < 0),
                      color='tab:green', alpha=0.10)
    ax3.set_ylabel('residual (z)', fontsize=11)
    ax3.set_xlabel('Time (UTC)', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.25)
    ax3.set_ylim(-4, 4)

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    out_path = (f'charts/trades_1s_{args.day}_r{int(args.r_confirm)}'
                f'_tp{int(args.tp)}_sl{int(args.sl)}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
