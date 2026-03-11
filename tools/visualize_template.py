#!/usr/bin/env python
"""
Template Visualizer
===================
Multi-panel visualization of a single template or all templates.

Panels:
  1. Centroid radar chart (16D feature fingerprint)
  2. MFE distribution: template oracle expectation vs actual trade MFE
  3. Trade timeline: PnL scatter by hold time, colored by exit reason
  4. Anchor patience diagram: template threshold vs actual trade MFE
  5. Transition map: where trades go next (Markov chain)
  6. Stats card: key numbers at a glance

Usage:
    python tools/visualize_template.py 21              # single template
    python tools/visualize_template.py 21 54 0 38      # compare multiple
    python tools/visualize_template.py --all            # overview grid
    python tools/visualize_template.py --mode oos 21    # OOS trades
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.oracle_config import ORACLE_LOOKAHEAD_BARS

TICK_SIZE = 0.25
TICK_VALUE = 0.50

FEATURE_NAMES = ['|z|', 'log(vel)', 'log(mom)', 'entropy', 'tf_scale', 'depth',
                 'parent_ctx', 'adx', 'hurst', 'dmi_diff', 'parent_z',
                 'parent_dmi', 'root_roche', 'tf_align', 'pid', 'osc_ent']

TF_SECONDS = {'1s':1,'5s':5,'15s':15,'30s':30,'1m':60,'2m':120,'3m':180,
              '5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400}

EXIT_COLORS = {
    'stop_loss': '#e74c3c',
    'take_profit': '#2ecc71',
    'envelope_decay': '#3498db',
    'peak_giveback': '#f39c12',
    'maintenance_flat': '#95a5a6',
    'belief_flip': '#9b59b6',
}

OUT_DIR = 'reports/research/template_viz'


def load_data(mode='oos'):
    with open('checkpoints/pattern_library.pkl', 'rb') as f:
        lib = pickle.load(f)
    with open('checkpoints/clustering_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    name = 'oos_trade_log.csv' if mode in ('oos', 'oos2') else 'oracle_trade_log.csv'
    tl_path = os.path.join('checkpoints', name)
    tl = pd.read_csv(tl_path) if os.path.exists(tl_path) else None
    return lib, scaler, tl


def _radar_panel(ax, centroid, scaler, tid, name):
    """Radar chart of centroid feature values (normalized to scaler range)."""
    # Normalize centroid to 0-1 range using scaler stats
    scaled = (centroid - scaler.mean_) / scaler.scale_
    # Clip for display
    clipped = np.clip(scaled, -3, 3)
    norm = (clipped + 3) / 6  # map [-3,3] -> [0,1]

    N = len(FEATURE_NAMES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = norm.tolist()
    # Close the polygon
    angles += angles[:1]
    values += values[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, 'o-', linewidth=1.5, color='#2980b9', markersize=3)
    ax.fill(angles, values, alpha=0.15, color='#2980b9')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURE_NAMES, size=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['-1.5s', '0', '+1.5s'], size=5, alpha=0.5)
    ax.set_title(f'TID {tid}: {name}', size=9, fontweight='bold', pad=15)


def _mfe_panel(ax, entry, trades):
    """MFE distribution: template expectation vs actual trade MFE."""
    p75_mfe = entry.get('p75_mfe_ticks', 0)
    mean_mfe = entry.get('mean_mfe_ticks', 0)

    if trades is not None and len(trades) > 0:
        actual_mfe = trades['trade_mfe_ticks'].values
        # Histogram of actual trade MFE
        max_val = min(np.percentile(actual_mfe, 99), p75_mfe * 0.5) if p75_mfe > 0 else np.percentile(actual_mfe, 99)
        max_val = max(max_val, 10)
        bins = np.linspace(0, max_val, 30)
        ax.hist(actual_mfe[actual_mfe <= max_val], bins=bins, color='#3498db',
                alpha=0.7, label=f'Actual MFE (n={len(actual_mfe)})', edgecolor='white')

        # Mark percentiles
        p50 = np.median(actual_mfe)
        p75_actual = np.percentile(actual_mfe, 75)
        ax.axvline(p50, color='#2980b9', linestyle='--', linewidth=1.5,
                   label=f'Actual p50={p50:.0f}t')
        ax.axvline(p75_actual, color='#2980b9', linestyle=':', linewidth=1.5,
                   label=f'Actual p75={p75_actual:.0f}t')

    # Template expectations (will be off-chart if huge)
    if p75_mfe > 0:
        if p75_mfe <= ax.get_xlim()[1] * 1.5:
            ax.axvline(p75_mfe, color='#e74c3c', linestyle='-', linewidth=2,
                       label=f'Template p75={p75_mfe:.0f}t')
        else:
            ax.annotate(f'Template p75={p75_mfe:.0f}t\n({p75_mfe*TICK_SIZE:.0f}pts) -->',
                        xy=(ax.get_xlim()[1] * 0.85, ax.get_ylim()[1] * 0.8),
                        fontsize=7, color='#e74c3c', fontweight='bold',
                        ha='right')

    ax.set_xlabel('MFE (ticks)', size=8)
    ax.set_ylabel('Count', size=8)
    ax.set_title('MFE: Template Expectation vs Reality', size=9)
    ax.legend(fontsize=6, loc='upper right')
    ax.tick_params(labelsize=7)


def _timeline_panel(ax, trades):
    """Trade scatter: hold_bars vs PnL, colored by exit reason."""
    if trades is None or trades.empty:
        ax.text(0.5, 0.5, 'No trades', ha='center', va='center', transform=ax.transAxes)
        return

    for reason, color in EXIT_COLORS.items():
        mask = trades['exit_reason'] == reason
        if mask.sum() == 0:
            continue
        grp = trades[mask]
        hold_min = grp['hold_bars'] * 15 / 60
        ax.scatter(hold_min, grp['actual_pnl'], s=15, alpha=0.6,
                   color=color, label=f'{reason} ({mask.sum()})', edgecolors='none')

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Hold time (min)', size=8)
    ax.set_ylabel('PnL ($)', size=8)
    ax.set_title('Trade Outcomes by Hold Time', size=9)
    ax.legend(fontsize=5, loc='upper right', ncol=2)
    ax.tick_params(labelsize=7)


def _anchor_panel(ax, entry, trades):
    """Anchor patience diagram: threshold vs actual MFE."""
    if trades is None or trades.empty:
        ax.text(0.5, 0.5, 'No trades', ha='center', va='center', transform=ax.transAxes)
        return

    anchor_mfe = trades['anchor_mfe_ticks'].values
    trade_mfe = trades['trade_mfe_ticks'].values
    threshold = anchor_mfe * 0.3  # 30% patience threshold

    # Scatter: each trade's actual MFE vs the patience threshold
    colors = ['#e74c3c' if m < t else '#2ecc71' for m, t in zip(trade_mfe, threshold)]
    ax.scatter(threshold, trade_mfe, s=12, c=colors, alpha=0.6, edgecolors='none')

    # Diagonal: MFE = threshold (trades above this line have giveback enabled)
    max_val = max(np.percentile(threshold, 95), np.percentile(trade_mfe, 95))
    max_val = max(max_val, 10)
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.3, label='MFE = threshold')

    n_blocked = sum(1 for m, t in zip(trade_mfe, threshold) if m < t)
    n_free = len(trade_mfe) - n_blocked

    ax.set_xlabel('Anchor patience threshold (ticks)', size=8)
    ax.set_ylabel('Actual trade MFE (ticks)', size=8)
    ax.set_title(f'Anchor Patience: {n_blocked} blocked (red) vs {n_free} free (green)', size=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)


def _transition_panel(ax, entry, lib):
    """Top transitions from this template."""
    tmap = entry.get('transition_map', {})
    if not tmap:
        ax.text(0.5, 0.5, 'No transitions', ha='center', va='center', transform=ax.transAxes)
        return

    # Sort by probability
    sorted_trans = sorted(tmap.items(), key=lambda x: x[1], reverse=True)[:10]
    tids = [str(int(t)) for t, _ in sorted_trans]
    probs = [p for _, p in sorted_trans]

    bars = ax.barh(range(len(tids)), probs, color='#3498db', alpha=0.7)
    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f'TID {t}' for t in tids], size=7)
    ax.set_xlabel('P(next)', size=8)
    ax.set_title('Transition Probabilities', size=9)
    ax.tick_params(labelsize=7)
    ax.invert_yaxis()

    # Annotate with template names
    for i, (tid_str, prob) in enumerate(zip(tids, probs)):
        tid_int = int(tid_str)
        if tid_int in lib:
            name = lib[tid_int].get('semantic_name', '')[:20]
            ax.text(prob + 0.005, i, f'{prob:.0%} {name}', va='center', fontsize=5)


def _stats_panel(ax, entry, trades, tid):
    """Stats card with key numbers."""
    ax.axis('off')

    lines = []
    lines.append(f"TID {tid}: {entry.get('semantic_name', '?')}")
    lines.append(f"Members: {entry.get('member_count', 0)}")
    lines.append(f"Win Rate: {entry.get('stats_win_rate', 0):.0%}")
    lines.append(f"Long Bias: {entry.get('long_bias', 0.5):.0%}")
    lines.append(f"")
    lines.append(f"--- TEMPLATE (oracle) ---")
    lines.append(f"p75 MFE: {entry.get('p75_mfe_ticks', 0):.0f} ticks "
                 f"(${entry.get('p75_mfe_ticks', 0) * TICK_VALUE:.0f})")
    lines.append(f"mean MAE: {entry.get('mean_mae_ticks', 0):.0f} ticks")
    lines.append(f"avg MFE bar: {entry.get('avg_mfe_bar', 0):.0f}")
    lines.append(f"TP: {entry.get('params', {}).get('take_profit_ticks', 0)} ticks")
    lines.append(f"SL: {entry.get('params', {}).get('stop_loss_ticks', 0)} ticks")

    if trades is not None and len(trades) > 0:
        lines.append(f"")
        lines.append(f"--- ACTUAL ({len(trades)} trades) ---")
        lines.append(f"Avg PnL: ${trades['actual_pnl'].mean():.1f}")
        lines.append(f"Total PnL: ${trades['actual_pnl'].sum():.0f}")
        lines.append(f"Avg MFE: {trades['trade_mfe_ticks'].mean():.0f} ticks")
        lines.append(f"p75 MFE: {trades['trade_mfe_ticks'].quantile(0.75):.0f} ticks")
        lines.append(f"Avg hold: {trades['hold_bars'].mean():.0f} bars "
                     f"({trades['hold_bars'].mean() * 15 / 60:.1f}min)")
        lines.append(f"MFE ratio: {trades['trade_mfe_ticks'].mean() / max(1, entry.get('p75_mfe_ticks', 1)):.1%}")

        # Exit breakdown
        lines.append(f"")
        for reason, grp in trades.groupby('exit_reason'):
            lines.append(f"  {reason}: {len(grp)} ({100*len(grp)/len(trades):.0f}%)")

    text = '\n'.join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=6.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))


def visualize_template(tid, lib, scaler, tl, save=True):
    """Full 6-panel visualization for one template."""
    if tid not in lib:
        print(f"  TID {tid} not in library")
        return None

    entry = lib[tid]
    trades = tl[tl['template_id'] == tid] if tl is not None else None

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Template {tid}: {entry.get('semantic_name', '?')}  "
                 f"(members={entry.get('member_count', 0)}, "
                 f"trades={len(trades) if trades is not None else 0})",
                 fontsize=12, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # Panel 1: Radar
    ax_radar = fig.add_subplot(gs[0, 0], projection='polar')
    _radar_panel(ax_radar, entry['centroid'], scaler, tid, entry.get('semantic_name', ''))

    # Panel 2: MFE distribution
    ax_mfe = fig.add_subplot(gs[0, 1])
    _mfe_panel(ax_mfe, entry, trades)

    # Panel 3: Stats card
    ax_stats = fig.add_subplot(gs[0, 2])
    _stats_panel(ax_stats, entry, trades, tid)

    # Panel 4: Timeline
    ax_time = fig.add_subplot(gs[1, 0])
    _timeline_panel(ax_time, trades)

    # Panel 5: Anchor patience
    ax_anchor = fig.add_subplot(gs[1, 1])
    _anchor_panel(ax_anchor, entry, trades)

    # Panel 6: Transitions
    ax_trans = fig.add_subplot(gs[1, 2])
    _transition_panel(ax_trans, entry, lib)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f'template_{tid}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    else:
        plt.show()
        return None


def visualize_overview(lib, scaler, tl, top_n=20):
    """Grid overview of top-N templates by trade count."""
    if tl is None:
        print("  No trade log — can't rank templates")
        return

    # Top templates by trade count
    counts = tl.groupby('template_id').size().sort_values(ascending=False)
    top_tids = [t for t in counts.index[:top_n] if t in lib]

    n_cols = 4
    n_rows = (len(top_tids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows),
                             subplot_kw=dict(projection='polar'))
    fig.suptitle(f'Top {len(top_tids)} Templates by Trade Count (radar = feature fingerprint)',
                 fontsize=14, fontweight='bold')

    for i, tid in enumerate(top_tids):
        row, col = divmod(i, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[col]
        entry = lib[tid]
        trades = tl[tl['template_id'] == tid]
        n = len(trades)
        avg_pnl = trades['actual_pnl'].mean()
        _radar_panel(ax, entry['centroid'], scaler, tid,
                     f"{entry.get('semantic_name', '')[:18]} n={n} ${avg_pnl:.0f}/tr")

    # Hide empty subplots
    for i in range(len(top_tids), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.set_visible(False)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'overview_grid.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='Template Visualizer')
    parser.add_argument('tids', nargs='*', type=int, help='Template IDs to visualize')
    parser.add_argument('--all', action='store_true', help='Overview grid of top templates')
    parser.add_argument('--mode', choices=['is', 'oos', 'oos2'], default='oos')
    parser.add_argument('--top', type=int, default=20, help='Top N for --all mode')
    args = parser.parse_args()

    print(f"  Template Visualizer ({args.mode.upper()} mode)")
    lib, scaler, tl = load_data(args.mode)
    print(f"  Loaded {len(lib)} templates, {len(tl) if tl is not None else 0} trades")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.all:
        path = visualize_overview(lib, scaler, tl, top_n=args.top)
        if path:
            print(f"  Overview saved: {path}")

    if args.tids:
        for tid in args.tids:
            path = visualize_template(tid, lib, scaler, tl)
            if path:
                print(f"  TID {tid} saved: {path}")
    elif not args.all:
        # Default: visualize top 5 by trade count
        if tl is not None:
            top5 = tl.groupby('template_id').size().sort_values(ascending=False).index[:5]
            for tid in top5:
                if tid in lib:
                    path = visualize_template(tid, lib, scaler, tl)
                    if path:
                        print(f"  TID {tid} saved: {path}")

    print(f"  Output: {OUT_DIR}/")


if __name__ == '__main__':
    main()
