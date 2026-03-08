"""
Capture Comparison — Individual high-def trade charts for best vs worst trades.

Saves each trade as its own image with:
  - Top: Price action with entry/exit/MFE markers, trade MFE peak line
  - Bottom-left: Stats box (PnL, capture, hold, exit, depth, template, physics)
  - Bottom-right: Intra-trade PnL curve showing peak and giveback

Usage:
    python tools/research/capture_comparison.py --week 2026-01-05
    python tools/research/capture_comparison.py --week 2026-02-24 --n 10
    python tools/research/capture_comparison.py --mode is --week 2025-04-07
"""
import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_trade_log(mode='oos'):
    if mode == 'oos':
        candidates = [
            PROJECT_ROOT / 'checkpoints' / 'oos_trade_log.csv',
            PROJECT_ROOT / 'reports' / 'oos' / 'oracle_trade_log.csv',
        ]
    else:
        candidates = [
            PROJECT_ROOT / 'checkpoints' / 'oracle_trade_log.csv',
            PROJECT_ROOT / 'reports' / 'is' / 'oracle_trade_log.csv',
        ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f"  Loaded {len(df)} trades from {p}")
            return df
    print(f"ERROR: No {mode} trade log found.")
    sys.exit(1)


def load_price_data(atlas_dir, ts_min, ts_max):
    import glob
    files = sorted(glob.glob(os.path.join(atlas_dir, '15s', '*.parquet')))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        if df['timestamp'].max() < ts_min - 86400:
            continue
        if df['timestamp'].min() > ts_max + 86400:
            continue
        dfs.append(df)
    if not dfs:
        print("ERROR: No price data found for the time range.")
        sys.exit(1)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates(subset='timestamp')
    return combined


def plot_individual_trade(trade, price_df, out_path, rank_label, tick_size=0.25, dpi=250):
    """One high-def chart per trade: price + intra-trade PnL + stats."""
    entry_ts = float(trade['entry_time'])
    exit_ts = float(trade['exit_time'])
    entry_px = float(trade['entry_price'])
    exit_px = float(trade['exit_price'])
    direction = trade['direction']
    pnl = float(trade['actual_pnl'])
    capture = float(trade.get('capture_rate', 0) or 0)
    hold_bars = int(trade.get('hold_bars', 0) or 0)
    exit_reason = str(trade.get('exit_reason', '?'))
    oracle_mfe = float(trade.get('oracle_mfe', 0) or 0)
    oracle_mae = float(trade.get('oracle_mae', 0) or 0)
    trade_mfe_ticks = float(trade.get('trade_mfe_ticks', 0) or 0)
    depth = int(trade.get('entry_depth', 0) or 0)
    template_id = trade.get('template_id', '?')
    hurst = float(trade.get('hurst', 0) or 0)
    velocity = float(trade.get('velocity', 0) or 0)
    f_momentum = float(trade.get('F_momentum', 0) or 0)
    f_reversion = float(trade.get('F_reversion', 0) or 0)
    conviction = float(trade.get('belief_conviction', 0) or 0)
    wave_mat = float(trade.get('wave_maturity', 0) or 0)
    tunnel_prob = float(trade.get('tunnel_prob', 0) or 0)
    sigma = float(trade.get('sigma', 0) or 0)
    trade_class = str(trade.get('trade_class', '?'))
    result = str(trade.get('result', '?'))
    tp_ticks = float(trade.get('tp_ticks', 0) or 0)
    sl_ticks = float(trade.get('sl_ticks', 0) or 0)
    target_price = float(trade.get('target_price', 0) or 0)
    stop_price = float(trade.get('stop_price', 0) or 0)

    # Context: 10min before entry, 5min after exit (more context for HD)
    ctx_before = 600
    ctx_after = 300
    mask = (price_df['timestamp'] >= entry_ts - ctx_before) & \
           (price_df['timestamp'] <= exit_ts + ctx_after)
    ctx = price_df[mask].copy()

    if ctx.empty:
        print(f"    SKIP: no price data for trade at {entry_ts}")
        return

    ctx['dt'] = pd.to_datetime(ctx['timestamp'], unit='s', utc=True)
    entry_dt = pd.to_datetime(entry_ts, unit='s', utc=True)
    exit_dt = pd.to_datetime(exit_ts, unit='s', utc=True)

    # Intra-trade bars
    trade_mask = (ctx['timestamp'] >= entry_ts) & (ctx['timestamp'] <= exit_ts)
    trade_bars = ctx[trade_mask].copy()

    # Compute intra-trade PnL curve
    if not trade_bars.empty:
        if direction == 'LONG':
            trade_bars['pnl_ticks'] = (trade_bars['close'] - entry_px) / tick_size
        else:
            trade_bars['pnl_ticks'] = (entry_px - trade_bars['close']) / tick_size
        trade_bars['pnl_dollars'] = trade_bars['pnl_ticks'] * tick_size * 2.0  # tick_value

    # ── Figure: 2 rows ──
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))

    # Top: price action (60% height)
    ax_price = fig.add_axes([0.06, 0.38, 0.88, 0.55])
    # Bottom-left: stats (40% height, 45% width)
    ax_stats = fig.add_axes([0.06, 0.04, 0.40, 0.30])
    # Bottom-right: intra-trade PnL curve (40% height, 45% width)
    ax_pnl = fig.add_axes([0.54, 0.04, 0.40, 0.30])

    # ── Price panel ──
    ax_price.plot(ctx['dt'], ctx['close'], color='#cccccc', linewidth=1.0, zorder=2)

    # Shade trade region
    if not trade_bars.empty:
        shade_color = '#00ff8818' if pnl > 0 else '#ff444418'
        ax_price.axvspan(entry_dt, exit_dt, color=shade_color, zorder=1)

    # Entry price horizontal line (dashed)
    ax_price.axhline(entry_px, color='#00aaff', linewidth=0.5, linestyle='--', alpha=0.5, zorder=3)

    # Oracle MFE line (where optimal exit was)
    if oracle_mfe > 0:
        if direction == 'LONG':
            mfe_px = entry_px + oracle_mfe
        else:
            mfe_px = entry_px - oracle_mfe
        ax_price.axhline(mfe_px, color='#ffaa00', linewidth=0.8, linestyle=':', alpha=0.7, zorder=3)
        ax_price.text(ctx['dt'].iloc[0], mfe_px, f'  MFE ${oracle_mfe:.1f}',
                      fontsize=7, color='#ffaa00', va='bottom')

    # Trade MFE line (where price actually peaked during trade)
    if trade_mfe_ticks > 0:
        if direction == 'LONG':
            tmfe_px = entry_px + trade_mfe_ticks * tick_size
        else:
            tmfe_px = entry_px - trade_mfe_ticks * tick_size
        ax_price.axhline(tmfe_px, color='#ff88ff', linewidth=0.6, linestyle='-.', alpha=0.5, zorder=3)

    # Oracle MAE line (worst adverse)
    if oracle_mae > 0:
        if direction == 'LONG':
            mae_px = entry_px - oracle_mae
        else:
            mae_px = entry_px + oracle_mae
        ax_price.axhline(mae_px, color='#ff4444', linewidth=0.6, linestyle=':', alpha=0.4, zorder=3)

    # Target price line (system's expectation at entry)
    if target_price > 0:
        ax_price.axhline(target_price, color='#00ff88', linewidth=1.0, linestyle='--', alpha=0.7, zorder=3)
        ax_price.text(ctx['dt'].iloc[-1], target_price, f'  TP {tp_ticks:.0f}t',
                      fontsize=7, color='#00ff88', va='bottom', ha='right')

    # Stop price line
    if stop_price > 0:
        ax_price.axhline(stop_price, color='#ff4444', linewidth=1.0, linestyle='--', alpha=0.7, zorder=3)
        ax_price.text(ctx['dt'].iloc[-1], stop_price, f'  SL {sl_ticks:.0f}t',
                      fontsize=7, color='#ff4444', va='top', ha='right')

    # Entry marker
    marker = '^' if direction == 'LONG' else 'v'
    ax_price.scatter([entry_dt], [entry_px], marker=marker, color='#00aaff',
                     s=200, zorder=6, edgecolors='white', linewidths=1.0)

    # Exit marker
    exit_color = '#00ff88' if pnl > 0 else '#ff4444'
    ax_price.scatter([exit_dt], [exit_px], marker='X', color=exit_color,
                     s=180, zorder=6, edgecolors='white', linewidths=0.8)

    # Title
    hold_min = hold_bars * 15 / 60
    cap_str = f"{capture * 100:.0f}%" if capture else "N/A"
    title_color = '#00ff88' if pnl > 0 else '#ff4444'
    ax_price.set_title(
        f"{rank_label}  |  {direction} @ ${entry_px:.2f}  |  "
        f"PnL: ${pnl:+.1f}  |  Capture: {cap_str}  |  "
        f"Hold: {hold_min:.1f}min  |  Exit: {exit_reason}",
        fontsize=10, color=title_color, pad=8, fontweight='bold')

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_price.tick_params(axis='both', labelsize=8, colors='#888888')
    ax_price.set_facecolor('#0a0a1a')
    ax_price.set_ylabel('Price', fontsize=8, color='#888888')
    for spine in ax_price.spines.values():
        spine.set_color('#333333')

    # ── Stats panel ──
    ax_stats.axis('off')
    ax_stats.set_facecolor('#0a0a1a')

    stats_lines = [
        f"Template: {template_id}    Depth: {depth}    Class: {trade_class}",
        f"Result: {result}    Conviction: {conviction:.3f}    Wave Maturity: {wave_mat:.3f}",
        f"",
        f"Oracle MFE: ${oracle_mfe:.1f}    Oracle MAE: ${oracle_mae:.1f}",
        f"Target: {tp_ticks:.0f}t (${target_price:.2f})    Stop: {sl_ticks:.0f}t (${stop_price:.2f})" if tp_ticks else "",
        f"Trade MFE: {trade_mfe_ticks:.0f} ticks    Capture: {cap_str}",
        f"",
        f"Hurst: {hurst:.3f}    Velocity: {velocity:.1f}    Sigma: {sigma:.2f}",
        f"F_momentum: {f_momentum:.1f}    F_reversion: {f_reversion:.1f}",
        f"Tunnel Prob: {tunnel_prob:.3f}",
        f"",
        f"Entry Workers:",
    ]

    # Parse entry_workers JSON if available
    try:
        workers_raw = trade.get('entry_workers', '{}')
        if isinstance(workers_raw, str) and workers_raw.startswith('{'):
            workers = json.loads(workers_raw)
            for tf, w in sorted(workers.items(), key=lambda x: -float(x[1].get('mfe', 0))):
                d = float(w.get('d', 0.5))
                dir_label = 'LONG' if d > 0.5 else 'SHORT' if d < 0.5 else 'FLAT'
                c = float(w.get('c', 0))
                z = float(w.get('z', 0))
                stats_lines.append(f"  {tf:>4}: {dir_label} d={d:.2f} c={c:.2f} z={z:+.1f}")
    except Exception:
        pass

    stats_text = '\n'.join(stats_lines)
    ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
                  fontsize=7, color='#cccccc', va='top', ha='left',
                  fontfamily='monospace')

    # ── Intra-trade PnL curve ──
    ax_pnl.set_facecolor('#0a0a1a')
    for spine in ax_pnl.spines.values():
        spine.set_color('#333333')

    if not trade_bars.empty and 'pnl_dollars' in trade_bars.columns:
        bars_x = range(len(trade_bars))
        pnl_vals = trade_bars['pnl_dollars'].values

        # Color by positive/negative
        colors = ['#00ff88' if v >= 0 else '#ff4444' for v in pnl_vals]
        ax_pnl.bar(bars_x, pnl_vals, color=colors, alpha=0.6, width=0.8)

        # Peak PnL line
        peak_idx = np.argmax(pnl_vals)
        peak_val = pnl_vals[peak_idx]
        ax_pnl.axhline(peak_val, color='#ffaa00', linewidth=0.8, linestyle='--', alpha=0.7)
        ax_pnl.text(len(bars_x) * 0.02, peak_val, f' Peak ${peak_val:+.1f}',
                     fontsize=7, color='#ffaa00', va='bottom')

        # Final PnL line
        ax_pnl.axhline(pnl_vals[-1], color=exit_color, linewidth=0.8, linestyle='-', alpha=0.5)

        # Zero line
        ax_pnl.axhline(0, color='#555555', linewidth=0.5, zorder=1)

        # Giveback annotation
        if peak_val > 0:
            giveback = (peak_val - pnl_vals[-1]) / peak_val * 100
            ax_pnl.text(0.98, 0.02, f'Giveback: {giveback:.0f}%',
                        transform=ax_pnl.transAxes, fontsize=9, color='#ffaa00',
                        ha='right', va='bottom', fontweight='bold')

    ax_pnl.set_title('Intra-Trade PnL (per bar)', fontsize=9, color='#888888')
    ax_pnl.set_xlabel('Bars (15s each)', fontsize=7, color='#888888')
    ax_pnl.set_ylabel('PnL ($)', fontsize=7, color='#888888')
    ax_pnl.tick_params(axis='both', labelsize=7, colors='#888888')

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Capture comparison: individual trade charts')
    parser.add_argument('--week', required=True,
                        help='Week start date (YYYY-MM-DD)')
    parser.add_argument('--mode', default='oos', choices=['is', 'oos'])
    parser.add_argument('--n', type=int, default=5, help='Number of best/worst trades')
    parser.add_argument('--sort', default='capture_rate',
                        choices=['capture_rate', 'actual_pnl'])
    parser.add_argument('--dpi', type=int, default=250)
    args = parser.parse_args()

    # Parse week
    if '_' in args.week:
        week_start = pd.Timestamp(args.week.replace('_', '-') + '-01', tz='UTC')
    else:
        week_start = pd.Timestamp(args.week, tz='UTC')
    week_end = week_start + pd.Timedelta(days=7)

    print(f"Capture Comparison: {week_start.date()} to {week_end.date()}")
    print(f"  Mode: {args.mode}, Sort: {args.sort}, N: {args.n}")

    # Load trades
    trades = load_trade_log(args.mode)
    trades['entry_dt'] = pd.to_datetime(trades['entry_time'], unit='s', utc=True)

    # Filter to week
    mask = (trades['entry_dt'] >= week_start) & (trades['entry_dt'] < week_end)
    week_trades = trades[mask].copy()
    print(f"  {len(week_trades)} trades in this week")

    if len(week_trades) < 2:
        print("Not enough trades for comparison.")
        return

    # Sort
    if args.sort == 'capture_rate' and 'capture_rate' in week_trades.columns:
        has_cap = week_trades['capture_rate'].notna()
        week_trades_sorted = week_trades[has_cap].sort_values(args.sort)
    else:
        week_trades_sorted = week_trades.sort_values(args.sort)

    n = min(args.n, len(week_trades_sorted) // 2)
    if n < 1:
        print("Not enough trades after filtering.")
        return

    worst = week_trades_sorted.head(n)
    best = week_trades_sorted.tail(n).iloc[::-1]  # reverse so #1 is best

    print(f"\n  TOP {n} BEST ({args.sort}):")
    for i, (_, t) in enumerate(best.iterrows()):
        cap = t.get('capture_rate', 0)
        print(f"    #{i+1}  ${t['actual_pnl']:+8.1f}  cap={cap*100:5.1f}%  "
              f"exit={t.get('exit_reason', '?'):15s}  hold={t.get('hold_bars', 0):>4} bars")

    print(f"\n  TOP {n} WORST ({args.sort}):")
    for i, (_, t) in enumerate(worst.iterrows()):
        cap = t.get('capture_rate', 0)
        print(f"    #{i+1}  ${t['actual_pnl']:+8.1f}  cap={cap*100:5.1f}%  "
              f"exit={t.get('exit_reason', '?'):15s}  hold={t.get('hold_bars', 0):>4} bars")

    # Load price data
    all_trades = pd.concat([worst, best])
    atlas_dir = 'DATA/ATLAS_OOS' if args.mode == 'oos' else 'DATA/ATLAS'
    ts_min = all_trades['entry_time'].min() - 600
    ts_max = all_trades['exit_time'].max() + 300
    print(f"\n  Loading price data...")
    price_df = load_price_data(atlas_dir, ts_min, ts_max)
    print(f"  {len(price_df)} bars loaded")

    # Output directory
    week_str = week_start.strftime('%Y%m%d')
    out_dir = os.path.join('reports', 'findings', f'capture_{week_str}')
    os.makedirs(out_dir, exist_ok=True)

    # Save individual charts
    print(f"\n  Saving individual trade charts to {out_dir}/")
    for i, (_, t) in enumerate(worst.iterrows()):
        label = f"WORST #{i+1}"
        out_path = os.path.join(out_dir, f'worst_{i+1:02d}.png')
        plot_individual_trade(t, price_df, out_path, label, dpi=args.dpi)
        print(f"    {out_path}")

    for i, (_, t) in enumerate(best.iterrows()):
        label = f"BEST #{i+1}"
        out_path = os.path.join(out_dir, f'best_{i+1:02d}.png')
        plot_individual_trade(t, price_df, out_path, label, dpi=args.dpi)
        print(f"    {out_path}")

    print(f"\n  Done! {2 * n} charts saved to {out_dir}/")


if __name__ == '__main__':
    main()
