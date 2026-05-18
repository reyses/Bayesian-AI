"""
Trade Visualizer -- Plots price waveform with entry/exit markers.

Usage:
    python -m tools.trade_visualizer --month 2025_01
    python -m tools.trade_visualizer --month 2025_04 --trade-log checkpoints/oracle_trade_log.csv
"""
import argparse
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Visualize trades on price waveform')
    parser.add_argument('--month', default='2025_01', help='Month file stem (e.g. 2025_04)')
    parser.add_argument('--trade-log',
                        default='runs/2026-02-22_pre-depth-gate/oracle_trade_log.csv',
                        help='Path to oracle_trade_log.csv')
    parser.add_argument('--atlas-dir', default='DATA/ATLAS/15s', help='ATLAS parquet directory')
    parser.add_argument('--resample', default='5min', help='Resample freq for price line')
    parser.add_argument('--output', default=None, help='Output image path')
    parser.add_argument('--dpi', type=int, default=300, help='Image DPI')
    args = parser.parse_args()

    # ── Load price data ──────────────────────────────────────────────────
    parquet_path = os.path.join(args.atlas_dir, f'{args.month}.parquet')
    print(f'Loading price: {parquet_path}')
    df = pd.read_parquet(parquet_path)

    # timestamp is epoch seconds (int64)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.set_index('dt').sort_index()

    # Resample to reduce density (5min default)
    ohlc = df['close'].resample(args.resample).agg(['first', 'last', 'min', 'max'])
    ohlc = ohlc.dropna()
    price_line = df['close'].resample(args.resample).last().dropna()

    # ── Load trade log ───────────────────────────────────────────────────
    print(f'Loading trades: {args.trade_log}')
    trades = pd.read_csv(args.trade_log)
    trades['entry_dt'] = pd.to_datetime(trades['entry_time'], unit='s', utc=True)
    trades['exit_dt'] = pd.to_datetime(trades['exit_time'], unit='s', utc=True)

    # Filter to month
    month_start = price_line.index.min()
    month_end = price_line.index.max()
    mask = (trades['entry_dt'] >= month_start) & (trades['entry_dt'] <= month_end)
    mt = trades[mask].copy()

    n = len(mt)
    total_pnl = mt['actual_pnl'].sum()
    wins = (mt['actual_pnl'] > 0).sum()
    wr = wins / n * 100 if n > 0 else 0

    # Separate wins/losses for layering
    mt_win = mt[mt['actual_pnl'] > 0]
    mt_loss = mt[mt['actual_pnl'] <= 0]

    print(f'  {n} trades | {wr:.1f}% WR | ${total_pnl:,.2f} PnL')

    # ── Plot ─────────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(28, 10))

    # Price waveform -- visible white line
    ax.plot(price_line.index, price_line.values,
            color='#cccccc', linewidth=0.6, alpha=0.95, zorder=2)

    # Shade each trading day's range as separate rectangles (no cross-gap interpolation)
    daily_hi = df['high'].resample('1D').max().dropna()
    daily_lo = df['low'].resample('1D').min().dropna()
    common_idx = daily_hi.index.intersection(daily_lo.index)
    for i, day in enumerate(common_idx):
        lo, hi = daily_lo[day], daily_hi[day]
        # Each day gets its own rectangle with alternating subtle shading
        c = '#222244' if i % 2 == 0 else '#1a1a33'
        # Use day boundaries: current day to next day
        day_end = common_idx[i + 1] if i + 1 < len(common_idx) else day + pd.Timedelta(hours=23)
        ax.axvspan(day, day_end, ymin=0, ymax=1, color=c, alpha=0.15, zorder=0)

    # ── Draw trade connecting lines (entry->exit) ──────────────────────
    for _, t in mt.iterrows():
        is_win = t['actual_pnl'] > 0
        entry_dt = t['entry_dt']
        exit_dt = t['exit_dt']
        entry_px = t['entry_price']
        exit_px = t['exit_price']

        # Connecting line only (no axvspan -- too noisy)
        lc = '#00cc66' if is_win else '#ff4444'
        ax.plot([entry_dt, exit_dt], [entry_px, exit_px],
                color=lc, linewidth=0.8, alpha=0.4, zorder=3)

    # ── Entry markers ────────────────────────────────────────────────────
    for subset, label in [(mt_loss, 'loss'), (mt_win, 'win')]:
        if subset.empty:
            continue
        longs = subset[subset['direction'] == 'LONG']
        shorts = subset[subset['direction'] == 'SHORT']

        if not longs.empty:
            c = '#00ff88' if label == 'win' else '#ff6666'
            ax.scatter(longs['entry_dt'], longs['entry_price'],
                       marker='^', color=c, s=50, zorder=6,
                       alpha=0.9, edgecolors='white', linewidths=0.4)
        if not shorts.empty:
            c = '#00ff88' if label == 'win' else '#ff6666'
            ax.scatter(shorts['entry_dt'], shorts['entry_price'],
                       marker='v', color=c, s=50, zorder=6,
                       alpha=0.9, edgecolors='white', linewidths=0.4)

    # ── Exit markers ─────────────────────────────────────────────────────
    for subset, label in [(mt_loss, 'loss'), (mt_win, 'win')]:
        if subset.empty:
            continue
        c = '#00ff88' if label == 'win' else '#ff4444'
        ax.scatter(subset['exit_dt'], subset['exit_price'],
                   marker='x', color=c, s=30, zorder=5, alpha=0.7, linewidths=1.0)

    # ── MFE peak markers (where the trade SHOULD have exited) ────────────
    for _, t in mt.iterrows():
        mfe = t.get('oracle_mfe', 0)
        if mfe and float(mfe) > 0:
            is_long = t['direction'] == 'LONG'
            peak_px = t['entry_price'] + float(mfe) if is_long else t['entry_price'] - float(mfe)
            # Place diamond at midpoint of trade duration
            mid_dt = t['entry_dt'] + (t['exit_dt'] - t['entry_dt']) / 2
            ax.scatter(mid_dt, peak_px, marker='d', color='#ffaa00',
                       s=16, zorder=7, alpha=0.55, edgecolors='#aa7700', linewidths=0.3)

    # ── Formatting ───────────────────────────────────────────────────────
    month_label = args.month.replace('_', '-')
    ax.set_title(
        f'MNQ {month_label}  |  {n} trades  |  {wr:.1f}% WR  |  '
        f'${total_pnl:,.2f} PnL  |  W:{wins} L:{n - wins}',
        fontsize=14, color='white', pad=15, fontweight='bold')
    ax.set_ylabel('Price', fontsize=11)
    ax.set_xlabel('Date (UTC)', fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # Mondays
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.xticks(rotation=45, fontsize=9)
    ax.grid(True, alpha=0.12, which='major')
    ax.grid(True, alpha=0.05, which='minor')

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color='#888888', linewidth=1, label='Price (5min)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#00ff88',
               markersize=9, linestyle='None', label='LONG entry (win)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#00ff88',
               markersize=9, linestyle='None', label='SHORT entry (win)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff6666',
               markersize=9, linestyle='None', label='LONG entry (loss)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#ff6666',
               markersize=9, linestyle='None', label='SHORT entry (loss)'),
        Line2D([0], [0], marker='x', color='#00ff88', markersize=8,
               linestyle='None', label='WIN exit', markeredgewidth=2),
        Line2D([0], [0], marker='x', color='#ff4444', markersize=8,
               linestyle='None', label='LOSS exit', markeredgewidth=2),
        Line2D([0], [0], marker='d', color='#ffaa00', markersize=6,
               linestyle='None', label='Oracle MFE peak'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.4, ncol=2)

    # ── Summary stats text box ───────────────────────────────────────────
    long_n = (mt['direction'] == 'LONG').sum()
    short_n = n - long_n
    avg_hold = mt['hold_bars'].mean() * 15 / 60 if n > 0 else 0  # minutes
    avg_pnl = total_pnl / n if n > 0 else 0
    mh_pct = (mt['exit_reason'] == 'MAX_HOLD').sum() / n * 100 if n > 0 else 0
    ts_pct = (mt['exit_reason'] == 'trail_stop').sum() / n * 100 if n > 0 else 0
    reversed_pct = (mt['capture_rate'].astype(float) < 0).sum() / n * 100 if n > 0 else 0

    stats_text = (
        f"Avg PnL: ${avg_pnl:.2f}/trade\n"
        f"Avg hold: {avg_hold:.0f} min\n"
        f"L/S: {long_n}/{short_n}\n"
        f"MAX_HOLD: {mh_pct:.0f}%  trail: {ts_pct:.0f}%\n"
        f"Reversed: {reversed_pct:.0f}%"
    )
    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222', alpha=0.8),
            color='#cccccc', family='monospace')

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────
    output = args.output or f'reports/trades_{args.month}.png'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {output}  ({args.dpi} DPI, {28 * args.dpi}x{10 * args.dpi} px)')


if __name__ == '__main__':
    main()
