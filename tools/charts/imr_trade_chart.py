"""I-MR Trade Chart — Visualize individual trade dynamics.

Shows price action with entry/exit, MFE/MAE, and key state variables
(F_momentum, z_score, volume, DMI) overlaid as subplots.

Usage:
    python tools/imr_trade_chart.py                          # top 10 by |PnL|
    python tools/imr_trade_chart.py --n 20                   # top 20
    python tools/imr_trade_chart.py --filter losers          # only losses
    python tools/imr_trade_chart.py --filter winners         # only wins
    python tools/imr_trade_chart.py --filter giveback        # only giveback exits
    python tools/imr_trade_chart.py --id 42                  # specific trade
    python tools/imr_trade_chart.py --source oos             # OOS replays
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone


def load_replays(source: str = 'is') -> list:
    path = os.path.join('reports', 'trade_replays', f'{source}_replays.json')
    if not os.path.isfile(path):
        print(f"No replay file: {path}")
        return []
    with open(path) as f:
        return json.load(f)


def plot_trade(trade: dict, output_dir: str, idx: int = 0):
    """Plot a single trade with price + state overlays."""
    bars = trade['bars']
    states = trade['states']
    side = trade['side']
    entry_bar = trade['entry_bar']
    exit_bar = trade['exit_bar']
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    pnl = trade['actual_pnl']
    mfe = trade.get('trade_mfe_ticks', 0)
    exit_reason = trade['exit_reason']
    template_id = trade['template_id']
    hold = trade['hold_bars']

    # Parse bars: [ts, open, high, low, close, volume]
    timestamps = [datetime.fromtimestamp(b[0], tz=timezone.utc) for b in bars]
    opens = [b[1] for b in bars]
    highs = [b[2] for b in bars]
    lows = [b[3] for b in bars]
    closes = [b[4] for b in bars]
    volumes = [b[5] for b in bars]

    # Parse states
    z_scores = [s.get('z', 0) for s in states]
    f_mom = [s.get('f_mom', 0) for s in states]
    dmi_p = [s.get('dmi_p', 0) for s in states]
    dmi_m = [s.get('dmi_m', 0) for s in states]
    adx = [s.get('adx', 0) for s in states]
    p_center = [s.get('P_center', 0) for s in states]
    coherence = [s.get('coherence', 0) for s in states]
    vel = [s.get('vel', 0) for s in states]

    # X axis
    x = list(range(len(bars)))
    entry_x = entry_bar
    exit_x = exit_bar

    # Clamp to array bounds
    entry_x = max(0, min(entry_x, len(x) - 1))
    exit_x = max(0, min(exit_x, len(x) - 1))

    # Color based on outcome
    win = pnl > 0
    title_color = '#2ecc71' if win else '#e74c3c'
    result_str = f"WIN ${pnl:.1f}" if win else f"LOSS ${pnl:.1f}"

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
    fig.suptitle(
        f'Trade #{trade["trade_id"]}  {side.upper()}  {result_str}  '
        f'MFE={mfe:.0f}t  Hold={hold}bars  Exit={exit_reason}  TID={template_id}',
        fontsize=13, fontweight='bold', color=title_color
    )

    # ── Panel 1: Price ──
    ax = axes[0]
    ax.plot(x, closes, color='#333', linewidth=1.2, label='Close')
    ax.fill_between(x, lows, highs, alpha=0.15, color='#666')

    # Entry/exit markers
    ax.axvline(entry_x, color='blue', linestyle='--', alpha=0.7, label='Entry')
    ax.axvline(exit_x, color='red', linestyle='--', alpha=0.7, label='Exit')
    ax.axhline(entry_price, color='blue', linestyle=':', alpha=0.4)

    # Shade trade region
    trade_color = '#2ecc71' if win else '#e74c3c'
    ax.axvspan(entry_x, exit_x, alpha=0.08, color=trade_color)

    # MFE line
    if side == 'long':
        mfe_price = entry_price + mfe * 0.25
    else:
        mfe_price = entry_price - mfe * 0.25
    ax.axhline(mfe_price, color='green', linestyle=':', alpha=0.4, label=f'MFE ({mfe:.0f}t)')

    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: F_momentum + Velocity ──
    ax = axes[1]
    ax.plot(x, f_mom, color='purple', linewidth=1, label='F_momentum')
    ax.plot(x, vel, color='orange', linewidth=0.8, alpha=0.7, label='Velocity')
    ax.axvline(entry_x, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(exit_x, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Momentum')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Z-score ──
    ax = axes[2]
    ax.plot(x, z_scores, color='teal', linewidth=1, label='Z-score')
    ax.axvline(entry_x, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(exit_x, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(2, color='red', linestyle=':', alpha=0.3)
    ax.axhline(-2, color='red', linestyle=':', alpha=0.3)
    ax.set_ylabel('Z-score')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: DMI ──
    ax = axes[3]
    ax.plot(x, dmi_p, color='green', linewidth=1, label='DMI+')
    ax.plot(x, dmi_m, color='red', linewidth=1, label='DMI-')
    ax.plot(x, adx, color='black', linewidth=0.8, alpha=0.6, label='ADX')
    ax.axvline(entry_x, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(exit_x, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('DMI/ADX')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 5: Volume + P_center + Coherence ──
    ax = axes[4]
    ax2 = ax.twinx()
    ax.bar(x, volumes, color='#aaa', alpha=0.5, label='Volume')
    ax2.plot(x, p_center, color='blue', linewidth=1, label='P_center')
    ax2.plot(x, coherence, color='magenta', linewidth=0.8, alpha=0.7, label='Coherence')
    ax.axvline(entry_x, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(exit_x, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Volume')
    ax2.set_ylabel('P_center / Coherence')
    ax.set_xlabel('Bar Index (15s)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fname = f'trade_{trade["trade_id"]:04d}_{side}_{result_str.replace("$","").replace(" ","_")}.png'
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='I-MR Trade Chart')
    parser.add_argument('--source', default='is', help='is or oos')
    parser.add_argument('--n', type=int, default=10, help='Number of trades to plot')
    parser.add_argument('--filter', default=None,
                        choices=['winners', 'losers', 'giveback', 'stop_loss', 'all'],
                        help='Filter trades')
    parser.add_argument('--id', type=int, default=None, help='Plot specific trade ID')
    parser.add_argument('--output', default='reports/findings/imr_charts',
                        help='Output directory')
    args = parser.parse_args()

    replays = load_replays(args.source)
    if not replays:
        return

    print(f'Loaded {len(replays)} trade replays from {args.source}')

    # Filter
    if args.id is not None:
        replays = [r for r in replays if r['trade_id'] == args.id]
    elif args.filter == 'winners':
        replays = [r for r in replays if r['actual_pnl'] > 0]
    elif args.filter == 'losers':
        replays = [r for r in replays if r['actual_pnl'] <= 0]
    elif args.filter == 'giveback':
        replays = [r for r in replays if r['exit_reason'] == 'peak_giveback']
    elif args.filter == 'stop_loss':
        replays = [r for r in replays if r['exit_reason'] == 'stop_loss']

    # Sort by |PnL| descending (most impactful trades first)
    replays.sort(key=lambda r: abs(r['actual_pnl']), reverse=True)

    # Take top N
    replays = replays[:args.n]

    print(f'Plotting {len(replays)} trades...')
    for i, trade in enumerate(replays):
        path = plot_trade(trade, args.output, i)
        pnl = trade['actual_pnl']
        side = trade['side']
        exit_r = trade['exit_reason']
        print(f'  [{i+1}/{len(replays)}] Trade #{trade["trade_id"]} '
              f'{side} ${pnl:.1f} ({exit_r}) -> {path}')

    print(f'\nCharts saved to {args.output}/')


if __name__ == '__main__':
    main()
