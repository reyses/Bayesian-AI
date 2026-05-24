"""Visualize pattern_library.pkl and live_brain.pkl checkpoint files.

Usage:
    python tools/checkpoint_viewer.py                        # default: checkpoints/
    python tools/checkpoint_viewer.py --dir checkpoints/snowflake
"""

import argparse
import pickle
import sys
from pathlib import Path

# ensure project root is importable (for TradeOutcome in brain pkl)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# ── feature labels for the 16D centroid ──────────────────────────────────────
FEATURE_NAMES = [
    'abs_z', 'log_v', 'log_m', 'entropy_normalized', 'tf_scale', 'depth',
    'parent_ctx', 'self_adx', 'self_hurst', 'self_dmi_diff',
    'parent_z', 'parent_dmi_diff', 'root_roche', 'tf_align',
    'self_pid', 'osc_coh',
]


def load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Page 1: Pattern Library Overview ─────────────────────────────────────────

def page_pattern_library(lib: dict, save_dir: Path):
    tids = sorted(lib.keys())
    names = [lib[t].get('semantic_name', str(t)) for t in tids]
    wr = [lib[t].get('stats_win_rate', 0) for t in tids]
    mfe = [lib[t].get('mean_mfe_ticks', 0) for t in tids]
    mae = [lib[t].get('mean_mae_ticks', 0) for t in tids]
    members = [lib[t].get('member_count', 0) for t in tids]
    depths = [lib[t]['centroid'][5] if 'centroid' in lib[t] else 0 for t in tids]

    # sort by win rate for bar chart
    order = np.argsort(wr)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Page 1 — Pattern Library Overview', fontsize=14, fontweight='bold')

    # ── top-left: WR bar sorted, colored by depth ──
    ax = axes[0, 0]
    wr_sorted = [wr[i] for i in order]
    depth_sorted = [depths[i] for i in order]
    norm = plt.Normalize(min(depths), max(depths) + 1e-9)
    colors = plt.cm.viridis(norm(depth_sorted))
    ax.bar(range(len(wr_sorted)), wr_sorted, color=colors, width=1.0)
    ax.set_title('Win Rate by Template (colored by depth)')
    ax.set_xlabel('Template (sorted)')
    ax.set_ylabel('Win Rate')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    fig.colorbar(sm, ax=ax, label='Depth')

    # ── top-right: MFE vs MAE scatter ──
    ax = axes[0, 1]
    ax.scatter(mae, mfe, c=wr, cmap='RdYlGn', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
    ax.set_title('Risk / Reward Map')
    ax.set_xlabel('Mean MAE (ticks)')
    ax.set_ylabel('Mean MFE (ticks)')
    ax.axline((0, 0), slope=1, color='grey', ls='--', lw=0.8, label='MFE=MAE')
    ax.legend(fontsize=8)
    sm2 = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(min(wr), max(wr)))
    fig.colorbar(sm2, ax=ax, label='Win Rate')

    # ── bottom-left: centroid heatmap ──
    ax = axes[1, 0]
    centroids = np.array([lib[t]['centroid'] for t in tids])
    im = ax.imshow(centroids, aspect='auto', cmap='coolwarm')
    ax.set_title('Centroid Heatmap')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Template index')
    ax.set_xticks(range(16))
    ax.set_xticklabels(FEATURE_NAMES, rotation=60, ha='right', fontsize=7)
    fig.colorbar(im, ax=ax)

    # ── bottom-right: top 10 table ──
    ax = axes[1, 1]
    ax.axis('off')
    top_idx = np.argsort(members)[::-1][:10]
    table_data = []
    for i in top_idx:
        t = tids[i]
        nm = lib[t].get('semantic_name', str(t))
        if len(nm) > 28:
            nm = nm[:26] + '..'
        table_data.append([
            str(t), nm,
            f"{wr[i]:.1%}",
            f"{mfe[i]:.0f}",
            f"{members[i]}",
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=['TID', 'Name', 'WR', 'MFE', 'Members'],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title('Top 10 Templates by Member Count')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_dir / 'page1_pattern_library.png', dpi=150)
    return fig


# ── Page 2: Direction & Depth ────────────────────────────────────────────────

def page_direction_depth(lib: dict, save_dir: Path):
    tids = sorted(lib.keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Page 2 — Direction & Depth', fontsize=14, fontweight='bold')

    # ── top-left: long/short bias diverging bar ──
    ax = axes[0, 0]
    long_b = [lib[t].get('long_bias', 0.5) for t in tids]
    short_b = [lib[t].get('short_bias', 0.5) for t in tids]
    net = [l - s for l, s in zip(long_b, short_b)]
    order = np.argsort(net)
    y = range(len(tids))
    colors = ['#2196F3' if n >= 0 else '#F44336' for n in [net[i] for i in order]]
    ax.barh(y, [net[i] for i in order], color=colors, height=1.0)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_title('Long - Short Bias')
    ax.set_xlabel('Net Bias (blue=LONG, red=SHORT)')
    ax.set_ylabel('Template (sorted)')

    # ── top-right: member count histogram ──
    ax = axes[0, 1]
    members = [lib[t].get('member_count', 0) for t in tids]
    ax.hist(members, bins=30, color='steelblue', edgecolor='white')
    ax.set_title('Member Count Distribution')
    ax.set_xlabel('Members')
    ax.set_ylabel('Count')
    ax.axvline(np.median(members), color='red', ls='--', label=f'median={np.median(members):.0f}')
    ax.legend()

    # ── bottom-left: model availability stacked bar ──
    ax = axes[1, 0]
    has_mfe = [1 if lib[t].get('mfe_coeff') is not None else 0 for t in tids]
    has_dir = [1 if lib[t].get('dir_coeff') is not None else 0 for t in tids]
    x = range(len(tids))
    ax.bar(x, has_mfe, label='MFE model', color='#4CAF50', width=1.0)
    ax.bar(x, has_dir, bottom=has_mfe, label='DIR model', color='#FF9800', width=1.0)
    ax.set_title('Model Availability per Template')
    ax.set_xlabel('Template index')
    ax.set_ylabel('Has model')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['None', 'One', 'Both'])
    ax.legend()

    # ── bottom-right: depth distribution ──
    ax = axes[1, 1]
    depths = [lib[t]['centroid'][5] for t in tids if 'centroid' in lib[t]]
    unique_d, counts_d = np.unique(np.round(depths, 1), return_counts=True)
    ax.bar(unique_d, counts_d, width=0.08, color='#9C27B0', edgecolor='white')
    ax.set_title('Depth Distribution')
    ax.set_xlabel('Depth (centroid[5])')
    ax.set_ylabel('Templates')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_dir / 'page2_direction_depth.png', dpi=150)
    return fig


# ── Page 3: Brain Table ──────────────────────────────────────────────────────

def page_brain(brain: dict, save_dir: Path):
    table = brain.get('table', {})
    history = brain.get('trade_history', [])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Page 3 — Brain Table', fontsize=14, fontweight='bold')

    # ── top-left: P(win) bar colored by confidence ──
    ax = axes[0, 0]
    tids = sorted(table.keys(), key=str)
    if tids:
        wins = [table[t]['wins'] for t in tids]
        losses = [table[t]['losses'] for t in tids]
        totals = [table[t]['total'] for t in tids]
        p_win = [(1 + w) / (11 + tot) for w, tot in zip(wins, totals)]
        conf = [min(tot / 100, 1.0) for tot in totals]
        colors = plt.cm.YlOrRd([c for c in conf])
        bars = ax.bar(range(len(tids)), p_win, color=colors, edgecolor='k', linewidth=0.3)
        ax.set_xticks(range(len(tids)))
        ax.set_xticklabels([str(t) for t in tids], rotation=45, ha='right', fontsize=8)
        ax.set_title('P(win) per Template (color = confidence)')
        ax.set_ylabel('P(win)')
        ax.axhline(0.5, color='grey', ls='--', lw=0.8)
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, label='Confidence')
    else:
        ax.text(0.5, 0.5, 'No brain table data', ha='center', va='center', transform=ax.transAxes)

    # ── top-right: wins vs losses scatter ──
    ax = axes[0, 1]
    if tids:
        sizes = [max(tot * 3, 20) for tot in totals]
        ax.scatter(losses, wins, s=sizes, c=p_win, cmap='RdYlGn', edgecolors='k',
                   linewidths=0.5, alpha=0.8)
        for i, t in enumerate(tids):
            ax.annotate(str(t), (losses[i], wins[i]), fontsize=7, ha='center', va='bottom')
        ax.set_title('Wins vs Losses (bubble = total trades)')
        ax.set_xlabel('Losses')
        ax.set_ylabel('Wins')
        ax.axline((0, 0), slope=1, color='grey', ls='--', lw=0.8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # ── bottom-left: PnL timeline ──
    ax = axes[1, 0]
    if history:
        pnls = [getattr(t, 'pnl', t.get('pnl', 0) if isinstance(t, dict) else 0) for t in history]
        cum_pnl = np.cumsum(pnls)
        ax.plot(cum_pnl, color='#2196F3', lw=1.5)
        ax.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.15, color='#2196F3')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'Cumulative PnL ({len(history)} trades)')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL ($)')
    else:
        ax.text(0.5, 0.5, 'No trade history', ha='center', va='center', transform=ax.transAxes)

    # ── bottom-right: exit reason pie ──
    ax = axes[1, 1]
    if history:
        reasons = {}
        for t in history:
            r = getattr(t, 'exit_reason', t.get('exit_reason', '?') if isinstance(t, dict) else '?')
            reasons[r] = reasons.get(r, 0) + 1
        labels = list(reasons.keys())
        sizes = list(reasons.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 8})
        ax.set_title('Exit Reasons')
    else:
        ax.text(0.5, 0.5, 'No trade history', ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_dir / 'page3_brain.png', dpi=150)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Checkpoint Visualizer')
    parser.add_argument('--dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--no-show', action='store_true', help='Save only, no interactive display')
    args = parser.parse_args()

    ckpt = Path(args.dir)
    save_dir = Path('tools/plots')
    save_dir.mkdir(parents=True, exist_ok=True)

    lib = load_pkl(ckpt / 'pattern_library.pkl')
    brain = load_pkl(ckpt / 'live_brain.pkl')

    if lib is None and brain is None:
        print(f"No pkl files found in {ckpt}/")
        sys.exit(1)

    figs = []
    if lib:
        print(f"Pattern library: {len(lib)} templates")
        figs.append(page_pattern_library(lib, save_dir))
        figs.append(page_direction_depth(lib, save_dir))
    else:
        print("No pattern_library.pkl found — skipping pages 1-2")

    if brain:
        tbl = brain.get('table', {})
        hist = brain.get('trade_history', [])
        print(f"Brain: {len(tbl)} entries, {len(hist)} trades")
        figs.append(page_brain(brain, save_dir))
    else:
        print("No live_brain.pkl found — skipping page 3")

    print(f"Saved {len(figs)} figures to {save_dir}/")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
