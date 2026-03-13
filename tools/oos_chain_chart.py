"""OOS Chain Comparison Chart — PnL curves + trade correlation across OOS1/OOS2/OOS3."""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def _load_trades(csv_path):
    """Load trades from CSV, return list of dicts with key fields."""
    if not os.path.exists(csv_path):
        return []
    trades = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trades.append({
                    'tid': row.get('template_id', ''),
                    'direction': row.get('direction', ''),
                    'entry_price': float(row.get('entry_price', 0)),
                    'exit_price': float(row.get('exit_price', 0)),
                    'entry_time': row.get('entry_time', ''),
                    'pnl': float(row.get('actual_pnl', 0)),
                    'exit_reason': row.get('exit_reason', ''),
                    'hold_bars': int(row.get('hold_bars', 0)),
                    'result': row.get('result', ''),
                    'dir_source': row.get('dir_source', ''),
                })
            except (ValueError, KeyError):
                continue
    return trades


def _cumulative_pnl(trades):
    """Return cumulative PnL array."""
    pnls = [t['pnl'] for t in trades]
    return np.cumsum(pnls) if pnls else np.array([])


def _match_trades(trades_a, trades_b, price_tol=2.0):
    """Match trades between two runs by entry_price + direction."""
    matched_a = set()
    matched_b = set()
    pairs = []
    for i, ta in enumerate(trades_a):
        for j, tb in enumerate(trades_b):
            if j in matched_b:
                continue
            if (ta['direction'] == tb['direction']
                    and abs(ta['entry_price'] - tb['entry_price']) <= price_tol):
                matched_a.add(i)
                matched_b.add(j)
                pairs.append((i, j))
                break
    return pairs, matched_a, matched_b


def generate_oos_chain_chart(checkpoint_dir, output_path):
    """Generate 6-panel OOS chain comparison chart."""
    oos1 = _load_trades(os.path.join(checkpoint_dir, 'oos1_trade_log.csv'))
    oos2 = _load_trades(os.path.join(checkpoint_dir, 'oos2_trade_log.csv'))
    oos3 = _load_trades(os.path.join(checkpoint_dir, 'oos3_trade_log.csv'))

    if not oos1 and not oos2 and not oos3:
        print("  No OOS trade logs found — skipping chart")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('OOS Chain Comparison: OOS1 vs OOS2 vs OOS3', fontsize=14, fontweight='bold')

    colors = {'OOS1': '#2196F3', 'OOS2': '#FF9800', 'OOS3': '#4CAF50'}

    # ── Panel 1: Cumulative PnL overlay ──
    ax = axes[0, 0]
    for label, trades, c in [('OOS1', oos1, colors['OOS1']),
                              ('OOS2', oos2, colors['OOS2']),
                              ('OOS3', oos3, colors['OOS3'])]:
        cum = _cumulative_pnl(trades)
        if len(cum) > 0:
            ax.plot(cum, label=f'{label} ({len(trades)} trades, ${cum[-1]:.0f})', color=c, linewidth=1.5)
    ax.set_title('Cumulative PnL')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('PnL ($)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Per-trade PnL scatter ──
    ax = axes[0, 1]
    for label, trades, c in [('OOS1', oos1, colors['OOS1']),
                              ('OOS2', oos2, colors['OOS2']),
                              ('OOS3', oos3, colors['OOS3'])]:
        pnls = [t['pnl'] for t in trades]
        if pnls:
            ax.scatter(range(len(pnls)), pnls, s=8, alpha=0.5, color=c, label=label)
    ax.set_title('Per-Trade PnL')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('PnL ($)')
    ax.axhline(0, color='red', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Trade overlap Venn-style bar chart ──
    ax = axes[0, 2]
    pairs_12, m1_12, m2_12 = _match_trades(oos1, oos2)
    pairs_23, m2_23, m3_23 = _match_trades(oos2, oos3)
    pairs_13, m1_13, m3_13 = _match_trades(oos1, oos3)

    overlap_data = {
        'OOS1∩OOS2': len(pairs_12),
        'OOS2∩OOS3': len(pairs_23),
        'OOS1∩OOS3': len(pairs_13),
        'OOS1 only': len(oos1) - len(m1_12) - len(m1_13 - m1_12),
        'OOS2 only': len(oos2) - len(m2_12) - len(m2_23 - m2_12),
        'OOS3 only': len(oos3) - len(m3_13) - len(m3_23 - m3_13),
    }
    bars = ax.barh(list(overlap_data.keys()), list(overlap_data.values()),
                   color=['#9C27B0', '#E91E63', '#009688', colors['OOS1'], colors['OOS2'], colors['OOS3']])
    for bar, val in zip(bars, overlap_data.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.set_title('Trade Overlap')
    ax.set_xlabel('# Trades')
    ax.grid(True, alpha=0.3, axis='x')

    # ── Panel 4: Exit reason distribution ──
    ax = axes[1, 0]
    exit_reasons = defaultdict(lambda: {'OOS1': 0, 'OOS2': 0, 'OOS3': 0})
    for label, trades in [('OOS1', oos1), ('OOS2', oos2), ('OOS3', oos3)]:
        for t in trades:
            reason = t['exit_reason'].split(':')[0].strip() if t['exit_reason'] else 'unknown'
            exit_reasons[reason][label] += 1

    reasons = sorted(exit_reasons.keys(), key=lambda r: sum(exit_reasons[r].values()), reverse=True)[:8]
    x = np.arange(len(reasons))
    w = 0.25
    for i, (label, c) in enumerate([('OOS1', colors['OOS1']), ('OOS2', colors['OOS2']), ('OOS3', colors['OOS3'])]):
        vals = [exit_reasons[r][label] for r in reasons]
        ax.bar(x + i*w, vals, w, label=label, color=c, alpha=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(reasons, rotation=45, ha='right', fontsize=7)
    ax.set_title('Exit Reasons')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 5: Direction accuracy ──
    ax = axes[1, 1]
    dir_data = {}
    for label, trades in [('OOS1', oos1), ('OOS2', oos2), ('OOS3', oos3)]:
        total = len(trades)
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        correct_dir = sum(1 for t in trades if t['pnl'] > 1.0)
        wrong_dir = sum(1 for t in trades if t['pnl'] < -1.0)
        be = total - correct_dir - wrong_dir
        dir_data[label] = {
            'WR': wins / max(1, total) * 100,
            'Correct Dir': correct_dir / max(1, total) * 100,
            'Wrong Dir': wrong_dir / max(1, total) * 100,
            'Breakeven': be / max(1, total) * 100,
        }

    categories = ['WR', 'Correct Dir', 'Wrong Dir', 'Breakeven']
    x = np.arange(len(categories))
    w = 0.25
    for i, (label, c) in enumerate([('OOS1', colors['OOS1']), ('OOS2', colors['OOS2']), ('OOS3', colors['OOS3'])]):
        vals = [dir_data[label][cat] for cat in categories]
        bars = ax.bar(x + i*w, vals, w, label=label, color=c, alpha=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.0f}%', ha='center', fontsize=7)
    ax.set_xticks(x + w)
    ax.set_xticklabels(categories)
    ax.set_title('Direction & Win Rate')
    ax.set_ylabel('%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 6: PnL correlation scatter (OOS1 vs OOS2 matched trades) ──
    ax = axes[1, 2]
    if pairs_12:
        pnl_1 = [oos1[i]['pnl'] for i, j in pairs_12]
        pnl_2 = [oos2[j]['pnl'] for i, j in pairs_12]
        ax.scatter(pnl_1, pnl_2, s=15, alpha=0.6, color='#9C27B0')
        # Correlation
        if len(pnl_1) > 2:
            corr = np.corrcoef(pnl_1, pnl_2)[0, 1]
            ax.set_title(f'PnL Correlation OOS1↔OOS2 (r={corr:.3f}, n={len(pairs_12)})')
        else:
            ax.set_title(f'PnL Correlation OOS1↔OOS2 (n={len(pairs_12)})')
        _lim = max(abs(min(pnl_1 + pnl_2)), abs(max(pnl_1 + pnl_2))) * 1.1
        ax.plot([-_lim, _lim], [-_lim, _lim], 'k--', linewidth=0.5, alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'No matched trades', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('PnL Correlation OOS1↔OOS2')
    ax.set_xlabel('OOS1 PnL ($)')
    ax.set_ylabel('OOS2 PnL ($)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OOS chain comparison chart')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--output', default='reports/oos_chain_comparison.png')
    args = parser.parse_args()
    generate_oos_chain_chart(args.checkpoint_dir, args.output)
