"""OOS2 vs OOS3 Trade Overlay Chart.

Matches trades by entry_price + side, shows:
1. Timeline chart: both trade sets on price axis, color by exit reason
2. Entry price matching: which OOS2 trades have OOS3 counterparts
3. Exit reason comparison for matched trades
4. MFE / bars_held scatter for matched vs unmatched
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_oos2(last_n_days=5):
    """Load OOS2 trades from CSV, filter to last N days."""
    df = pd.read_csv(ROOT / 'checkpoints' / 'oos_trade_log.csv')
    df['entry_dt'] = pd.to_datetime(df['entry_time'], unit='s')
    df['exit_dt'] = pd.to_datetime(df['exit_time'], unit='s')
    df['date'] = df['entry_dt'].dt.date
    last_days = sorted(df['date'].unique())[-last_n_days:]
    df = df[df['date'].isin(last_days)].reset_index(drop=True)
    # Normalize direction
    df['side'] = df['direction'].str.lower()
    df['pnl'] = df['actual_pnl']
    df['bars'] = df['hold_bars']
    df['mfe_ticks'] = df['trade_mfe_ticks']
    return df


def parse_oos3_from_report(report_path):
    """Parse OOS3 trades from parity report text."""
    text = Path(report_path).read_text()
    trades = []
    # Match lines like:   1  short   25,018.75  25,018.50 $    +0.77 stop_loss              2 template_bias
    pattern = re.compile(
        r'\s+(\d+)\s+(short|long)\s+([\d,]+\.\d+)\s+([\d,]+\.\d+)\s+\$\s+([+-]?[\d,]+\.\d+)\s+(\S+)\s+(\d+)\s+(\S+)'
    )
    for m in pattern.finditer(text):
        trades.append({
            'trade_num': int(m.group(1)),
            'side': m.group(2),
            'entry_price': float(m.group(3).replace(',', '')),
            'exit_price': float(m.group(4).replace(',', '')),
            'pnl': float(m.group(5).replace(',', '')),
            'exit_reason': m.group(6),
            'bars': int(m.group(7)),
            'dir_source': m.group(8),
        })
    return pd.DataFrame(trades)


def match_trades(oos2, oos3, price_tol=2.0):
    """Match OOS2 and OOS3 trades by entry_price + side within tolerance."""
    matches = []
    oos3_used = set()

    for i2, r2 in oos2.iterrows():
        best_j = None
        best_dist = price_tol + 1
        for i3, r3 in oos3.iterrows():
            if i3 in oos3_used:
                continue
            if r2['side'] != r3['side']:
                continue
            dist = abs(r2['entry_price'] - r3['entry_price'])
            if dist < best_dist:
                best_dist = dist
                best_j = i3
        if best_j is not None and best_dist <= price_tol:
            oos3_used.add(best_j)
            matches.append({
                'oos2_idx': i2,
                'oos3_idx': best_j,
                'side': r2['side'],
                'entry_price_oos2': r2['entry_price'],
                'entry_price_oos3': oos3.loc[best_j, 'entry_price'],
                'exit_price_oos2': r2['exit_price'],
                'exit_price_oos3': oos3.loc[best_j, 'exit_price'],
                'exit_reason_oos2': r2['exit_reason'],
                'exit_reason_oos3': oos3.loc[best_j, 'exit_reason'],
                'pnl_oos2': r2['pnl'],
                'pnl_oos3': oos3.loc[best_j, 'pnl'],
                'bars_oos2': r2['bars'],
                'bars_oos3': oos3.loc[best_j, 'bars'],
                'mfe_oos2': r2.get('mfe_ticks', 0),
                'price_diff': best_dist,
            })

    matched_df = pd.DataFrame(matches)
    oos2_unmatched = oos2[~oos2.index.isin([m['oos2_idx'] for m in matches])]
    oos3_unmatched = oos3[~oos3.index.isin(oos3_used)]
    return matched_df, oos2_unmatched, oos3_unmatched


EXIT_COLORS = {
    'stop_loss': '#888888',
    'envelope_decay': '#2196F3',
    'peak_giveback': '#FF9800',
    'take_profit': '#4CAF50',
    'eod_flatten': '#9C27B0',
    'belief_flip': '#E91E63',
    'band_urgent': '#795548',
}


def make_overlay(oos2, oos3, matched, oos2_unmatched, oos3_unmatched):
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle('OOS2 (Inline) vs OOS3 (AdvanceEngine) — Trade Overlay', fontsize=16, fontweight='bold')

    # --- Panel 1: Trade timeline by entry price, color by exit reason ---
    ax = axes[0, 0]
    for _, r in oos2.iterrows():
        c = EXIT_COLORS.get(r['exit_reason'], '#000000')
        ax.plot([r['entry_price'], r['exit_price']], [0, 0],
                color=c, alpha=0.3, linewidth=1)
        ax.scatter(r['entry_price'], 0.1, color=c, s=15, alpha=0.6, marker='v' if r['side']=='short' else '^')
    for _, r in oos3.iterrows():
        c = EXIT_COLORS.get(r['exit_reason'], '#000000')
        ax.scatter(r['entry_price'], -0.1, color=c, s=15, alpha=0.6, marker='v' if r['side']=='short' else '^')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_yticks([-0.1, 0.1])
    ax.set_yticklabels(['OOS3', 'OOS2'])
    ax.set_xlabel('Entry Price')
    ax.set_title('Entry Locations (color = exit reason)')
    handles = [mpatches.Patch(color=v, label=k) for k, v in EXIT_COLORS.items() if k in
               set(oos2['exit_reason'].unique()) | set(oos3['exit_reason'].unique())]
    ax.legend(handles=handles, fontsize=7, loc='upper left')

    # --- Panel 2: Venn-style match summary ---
    ax = axes[0, 1]
    ax.axis('off')
    n_matched = len(matched)
    n_oos2_only = len(oos2_unmatched)
    n_oos3_only = len(oos3_unmatched)
    total_oos2 = len(oos2)
    total_oos3 = len(oos3)

    summary_text = (
        f"TRADE MATCHING (price tol = 2.0 pts)\n"
        f"{'='*45}\n\n"
        f"OOS2 total:           {total_oos2:>4}\n"
        f"OOS3 total:           {total_oos3:>4}\n\n"
        f"MATCHED (same entry): {n_matched:>4}  ({n_matched/total_oos2*100:.0f}% of OOS2)\n"
        f"OOS2-only:            {n_oos2_only:>4}  (OOS2 entered, OOS3 didn't)\n"
        f"OOS3-only:            {n_oos3_only:>4}  (OOS3 entered, OOS2 didn't)\n"
    )
    if n_matched > 0:
        same_exit = (matched['exit_reason_oos2'] == matched['exit_reason_oos3']).sum()
        summary_text += (
            f"\nOf {n_matched} matched trades:\n"
            f"  Same exit reason:   {same_exit:>4}  ({same_exit/n_matched*100:.0f}%)\n"
            f"  Diff exit reason:   {n_matched-same_exit:>4}  ({(n_matched-same_exit)/n_matched*100:.0f}%)\n"
            f"\n  Avg PnL OOS2:     ${matched['pnl_oos2'].mean():>8.2f}\n"
            f"  Avg PnL OOS3:     ${matched['pnl_oos3'].mean():>8.2f}\n"
            f"\n  Avg Bars OOS2:    {matched['bars_oos2'].mean():>8.1f}\n"
            f"  Avg Bars OOS3:    {matched['bars_oos3'].mean():>8.1f}\n"
        )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Panel 3: Exit reason transition matrix (matched trades) ---
    ax = axes[1, 0]
    if n_matched > 0:
        reasons = sorted(set(matched['exit_reason_oos2'].unique()) |
                        set(matched['exit_reason_oos3'].unique()))
        matrix = np.zeros((len(reasons), len(reasons)))
        for _, r in matched.iterrows():
            i = reasons.index(r['exit_reason_oos2'])
            j = reasons.index(r['exit_reason_oos3'])
            matrix[i, j] += 1
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(reasons)))
        ax.set_xticklabels(reasons, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(reasons)))
        ax.set_yticklabels(reasons, fontsize=8)
        ax.set_xlabel('OOS3 exit reason')
        ax.set_ylabel('OOS2 exit reason')
        ax.set_title('Exit Reason Transition (matched trades)')
        for i in range(len(reasons)):
            for j in range(len(reasons)):
                if matrix[i, j] > 0:
                    ax.text(j, i, f'{int(matrix[i,j])}', ha='center', va='center',
                            fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'No matched trades', ha='center', va='center')
        ax.set_title('Exit Reason Transition')

    # --- Panel 4: Bars held comparison (matched) ---
    ax = axes[1, 1]
    if n_matched > 0:
        ax.scatter(matched['bars_oos2'], matched['bars_oos3'], c='steelblue', alpha=0.6, s=40)
        max_bars = max(matched['bars_oos2'].max(), matched['bars_oos3'].max()) + 5
        ax.plot([0, max_bars], [0, max_bars], 'k--', alpha=0.3, label='parity line')
        ax.set_xlabel('Bars Held OOS2')
        ax.set_ylabel('Bars Held OOS3')
        ax.set_title('Bars Held: OOS2 vs OOS3 (matched trades)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No matched trades', ha='center', va='center')

    # --- Panel 5: PnL comparison (matched) ---
    ax = axes[2, 0]
    if n_matched > 0:
        colors = ['green' if r['pnl_oos2'] > 0 and r['pnl_oos3'] > 0
                  else 'red' if r['pnl_oos2'] < 0 or r['pnl_oos3'] < 0
                  else 'orange' for _, r in matched.iterrows()]
        ax.scatter(matched['pnl_oos2'], matched['pnl_oos3'], c=colors, alpha=0.6, s=40)
        lim = max(abs(matched['pnl_oos2']).max(), abs(matched['pnl_oos3']).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        ax.set_xlabel('PnL OOS2 ($)')
        ax.set_ylabel('PnL OOS3 ($)')
        ax.set_title('PnL: OOS2 vs OOS3 (matched trades)')
        ax.axhline(0, color='gray', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', alpha=0.3, linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'No matched trades', ha='center', va='center')

    # --- Panel 6: Exit reason breakdown comparison ---
    ax = axes[2, 1]
    oos2_exits = oos2['exit_reason'].value_counts()
    oos3_exits = oos3['exit_reason'].value_counts()
    all_reasons = sorted(set(oos2_exits.index) | set(oos3_exits.index))
    x = np.arange(len(all_reasons))
    w = 0.35
    bars1 = ax.bar(x - w/2, [oos2_exits.get(r, 0) for r in all_reasons],
                   w, label='OOS2', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + w/2, [oos3_exits.get(r, 0) for r in all_reasons],
                   w, label='OOS3', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_reasons, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Trade Count')
    ax.set_title('Exit Reason Distribution')
    ax.legend()
    # Add count labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(bar.get_height())}', ha='center', fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(bar.get_height())}', ha='center', fontsize=8)

    plt.tight_layout()
    out_path = ROOT / 'reports' / 'live' / 'oos_parity_overlay.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

    # --- Text summary of matched trade details ---
    print('\n' + '='*80)
    print('  MATCHED TRADE DETAILS')
    print('='*80)
    if n_matched > 0:
        print(f'  {"Side":<6} {"Entry OOS2":>10} {"Entry OOS3":>10} {"Diff":>6} '
              f'{"Exit OOS2":<16} {"Exit OOS3":<16} {"PnL2":>8} {"PnL3":>8} '
              f'{"Bars2":>5} {"Bars3":>5}')
        print('-'*100)
        for _, r in matched.iterrows():
            same = '=' if r['exit_reason_oos2'] == r['exit_reason_oos3'] else '!'
            print(f'  {r["side"]:<6} {r["entry_price_oos2"]:>10.2f} {r["entry_price_oos3"]:>10.2f} '
                  f'{r["price_diff"]:>6.2f} '
                  f'{r["exit_reason_oos2"]:<16} {r["exit_reason_oos3"]:<16} '
                  f'${r["pnl_oos2"]:>7.2f} ${r["pnl_oos3"]:>7.2f} '
                  f'{r["bars_oos2"]:>5} {r["bars_oos3"]:>5} {same}')

    print('\n' + '='*80)
    print('  OOS2-ONLY TRADES (OOS2 entered, OOS3 did not)')
    print('='*80)
    if len(oos2_unmatched) > 0:
        print(f'  {"Side":<6} {"Entry":>10} {"Exit":>10} {"ExitReason":<16} {"PnL":>8} {"Bars":>5} {"MFE":>6}')
        print('-'*70)
        for _, r in oos2_unmatched.iterrows():
            print(f'  {r["side"]:<6} {r["entry_price"]:>10.2f} {r["exit_price"]:>10.2f} '
                  f'{r["exit_reason"]:<16} ${r["pnl"]:>7.2f} {r["bars"]:>5} {r.get("mfe_ticks",0):>6.1f}')

    print('\n' + '='*80)
    print('  OOS3-ONLY TRADES (OOS3 entered, OOS2 did not)')
    print('='*80)
    if len(oos3_unmatched) > 0:
        print(f'  {"Side":<6} {"Entry":>10} {"Exit":>10} {"ExitReason":<16} {"PnL":>8} {"Bars":>5}')
        print('-'*70)
        for _, r in oos3_unmatched.iterrows():
            print(f'  {r["side"]:<6} {r["entry_price"]:>10.2f} {r["exit_price"]:>10.2f} '
                  f'{r["exit_reason"]:<16} ${r["pnl"]:>7.2f} {r["bars"]:>5}')


def main():
    report_path = ROOT / 'reports' / 'live' / 'parity_report_20260312_183315.txt'
    if not report_path.exists():
        # Find latest
        reports = sorted(ROOT.glob('reports/live/parity_report_*.txt'))
        if not reports:
            print('No parity reports found')
            sys.exit(1)
        report_path = reports[-1]
        print(f'Using latest report: {report_path.name}')

    oos2 = load_oos2(last_n_days=5)
    oos3 = parse_oos3_from_report(report_path)

    print(f'OOS2: {len(oos2)} trades')
    print(f'OOS3: {len(oos3)} trades')

    matched, oos2_unmatched, oos3_unmatched = match_trades(oos2, oos3, price_tol=2.0)
    print(f'Matched: {len(matched)}, OOS2-only: {len(oos2_unmatched)}, OOS3-only: {len(oos3_unmatched)}')

    make_overlay(oos2, oos3, matched, oos2_unmatched, oos3_unmatched)


if __name__ == '__main__':
    main()
