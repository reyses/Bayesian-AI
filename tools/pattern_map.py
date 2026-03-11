"""
Pattern Location Map -- Shows WHERE every detected pattern sits on the price waveform.

Overlays all signals (traded + skipped) from the signal log onto the price chart,
color-coded by gate outcome. Answers: "Are workers detecting patterns? Where? What blocks them?"

Usage:
    python -m tools.pattern_map --month 2025_01
    python -m tools.pattern_map --month 2025_04 --min-oracle 50
"""
import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba


# Gate display config: (color, label, zorder, alpha)
GATE_STYLE = {
    'traded':             ('#00ff88', 'Traded',              8, 0.95),
    'score_loser':        ('#888800', 'Score loser',         4, 0.25),
    'gate3':              ('#4488ff', 'Gate 3 (conviction)', 5, 0.55),
    'gate1':              ('#ff8800', 'Gate 1 (no match)',   6, 0.65),
    'gate2':              ('#cc44cc', 'Gate 2 (brain)',      6, 0.65),
    'gate0_r4_struct':    ('#ff2222', 'Gate 0 (headroom)',   7, 0.70),
    'gate0_r4_nightmare': ('#ff0000', 'Gate 0 (nightmare)',  7, 0.70),
    'gate0_hurst':        ('#ff6600', 'Gate 0 (Hurst)',      7, 0.65),
    'gate0_momentum':     ('#ff4488', 'Gate 0 (momentum)',   7, 0.65),
    'gate0_tunnel':       ('#ff2288', 'Gate 0 (tunnel)',     7, 0.65),
    'gate0_5':            ('#aa4444', 'Gate 0.5 (depth)',    6, 0.60),
}


def main():
    parser = argparse.ArgumentParser(description='Pattern detection location map')
    parser.add_argument('--month', default='2025_01', help='Month stem (e.g. 2025_01)')
    parser.add_argument('--signal-log', default=None,
                        help='Signal log CSV (auto-detected from runs/ if omitted)')
    parser.add_argument('--atlas-dir', default='DATA/ATLAS/15s')
    parser.add_argument('--min-oracle', type=float, default=0,
                        help='Only show signals with oracle_pnl >= this (filter noise)')
    parser.add_argument('--output', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    # ── Auto-detect signal log ───────────────────────────────────────────
    if args.signal_log is None:
        # Map month to quarter
        month_num = int(args.month.split('_')[1])
        q = (month_num - 1) // 3 + 1
        year = args.month.split('_')[0]
        candidates = [
            f'reports/is/shards/signal_log_{year}_Q{q}.csv',
            f'reports/oos/shards/signal_log_{year}_Q{q}.csv',
            f'runs/2026-02-22_pre-depth-gate/signal_log_{year}_Q{q}.csv',
            f'checkpoints/signal_log_{year}_Q{q}.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                args.signal_log = c
                break
        if args.signal_log is None:
            print(f'ERROR: No signal log found for {args.month}. Tried: {candidates}')
            return

    # ── Load price ───────────────────────────────────────────────────────
    parquet_path = os.path.join(args.atlas_dir, f'{args.month}.parquet')
    print(f'Loading price: {parquet_path}')
    df = pd.read_parquet(parquet_path)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.set_index('dt').sort_index()

    # Build timestamp -> close lookup for signal price placement
    ts_to_close = dict(zip(df['timestamp'].values, df['close'].values))

    price_5m = df['close'].resample('5min').last().dropna()

    # ── Load signal log ──────────────────────────────────────────────────
    print(f'Loading signals: {args.signal_log}')
    sig = pd.read_csv(args.signal_log)
    sig = sig[sig['day'] == args.month].copy()
    print(f'  {len(sig)} signals in {args.month}')

    if args.min_oracle > 0:
        sig = sig[sig['oracle_pnl'].fillna(0).astype(float) >= args.min_oracle]
        print(f'  {len(sig)} after oracle_pnl >= {args.min_oracle}')

    sig['dt'] = pd.to_datetime(sig['ts'].astype(float), unit='s', utc=True)
    # Look up close price at each signal's timestamp
    sig['price'] = sig['ts'].astype(int).map(ts_to_close)
    # Fallback: forward-fill from nearest bar if exact match fails
    sig['price'] = sig['price'].ffill().fillna(df['close'].iloc[0])

    sig['oracle_pnl_f'] = sig['oracle_pnl'].fillna(0).astype(float)

    # ── Gate summary stats ───────────────────────────────────────────────
    gate_counts = sig['gate'].value_counts()
    total_signals = len(sig)
    total_oracle = sig['oracle_pnl_f'].sum()
    real_moves = (sig['oracle_pnl_f'] > 0).sum()

    print(f'\n  DETECTION SUMMARY ({args.month}):')
    print(f'    Total signals logged:  {total_signals:>7,}')
    print(f'    Real moves (oracle>0): {real_moves:>7,} ({real_moves/total_signals*100:.1f}%)')
    print(f'    Total oracle potential: ${total_oracle:>12,.0f}')
    print(f'    Gate breakdown:')
    for gate, count in gate_counts.items():
        oracle_sum = sig[sig['gate'] == gate]['oracle_pnl_f'].sum()
        pct = count / total_signals * 100
        print(f'      {gate:<22} {count:>6} ({pct:>5.1f}%)  oracle ${oracle_sum:>10,.0f}')

    # ── Unique bars vs total bars ────────────────────────────────────────
    total_bars = len(df)
    unique_signal_bars = sig['ts'].nunique()
    blind_bars = total_bars - unique_signal_bars
    print(f'\n    Total 15s bars in month: {total_bars:>7,}')
    print(f'    Bars with detection:    {unique_signal_bars:>7,} ({unique_signal_bars/total_bars*100:.1f}%)')
    print(f'    Bars with NO detection: {blind_bars:>7,} ({blind_bars/total_bars*100:.1f}%)')

    # ── Plot ─────────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    fig, (ax_main, ax_funnel) = plt.subplots(
        1, 2, figsize=(32, 10), gridspec_kw={'width_ratios': [4, 1]})

    # Price waveform
    ax_main.plot(price_5m.index, price_5m.values,
                 color='#aaaaaa', linewidth=0.5, alpha=0.8, zorder=1)

    # ── Plot signals by gate (score_losers first/background, traded last/foreground)
    gate_order = ['score_loser', 'gate3', 'gate0_5', 'gate1', 'gate2',
                  'gate0_r4_struct', 'gate0_r4_nightmare',
                  'gate0_hurst', 'gate0_momentum', 'gate0_tunnel', 'traded']

    for gate_name in gate_order:
        subset = sig[sig['gate'] == gate_name]
        if subset.empty:
            continue
        style = GATE_STYLE.get(gate_name, ('#ffffff', gate_name, 3, 0.5))
        color, label, zorder, alpha = style

        # Size proportional to oracle potential (bigger = more $ left on table)
        sizes = np.clip(subset['oracle_pnl_f'].values / 10, 5, 80)

        ax_main.scatter(subset['dt'], subset['price'],
                        color=color, s=sizes, alpha=alpha, zorder=zorder,
                        edgecolors='none', label=f'{label} ({len(subset)})')

    # ── Formatting ───────────────────────────────────────────────────────
    month_label = args.month.replace('_', '-')
    traded_n = gate_counts.get('traded', 0)
    ax_main.set_title(
        f'Pattern Detection Map  |  MNQ {month_label}  |  '
        f'{total_signals:,} detections  |  {traded_n} traded  |  '
        f'{blind_bars:,} blind bars ({blind_bars/total_bars*100:.0f}%)',
        fontsize=13, color='white', pad=15, fontweight='bold')
    ax_main.set_ylabel('Price', fontsize=11)
    ax_main.set_xlabel('Date (UTC)', fontsize=10)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax_main.xaxis.set_minor_locator(mdates.DayLocator())
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, fontsize=9)
    ax_main.grid(True, alpha=0.12, which='major')

    # Legend
    legend_elements = []
    for gate_name in gate_order:
        if gate_name not in gate_counts.index:
            continue
        style = GATE_STYLE.get(gate_name, ('#ffffff', gate_name, 3, 0.5))
        color, label, _, alpha = style
        cnt = gate_counts[gate_name]
        oracle = sig[sig['gate'] == gate_name]['oracle_pnl_f'].sum()
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=7, linestyle='None', alpha=alpha,
                   label=f'{label}: {cnt:,} (${oracle:,.0f})'))
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8,
                   framealpha=0.5, ncol=2)

    # ── Right panel: Gate funnel bar chart ───────────────────────────────
    funnel_gates = [g for g in gate_order if g in gate_counts.index]
    funnel_counts = [gate_counts[g] for g in funnel_gates]
    funnel_labels = [GATE_STYLE.get(g, ('#fff', g, 0, 0))[1] for g in funnel_gates]
    funnel_colors = [GATE_STYLE.get(g, ('#fff', g, 0, 0))[0] for g in funnel_gates]
    funnel_oracle = [sig[sig['gate'] == g]['oracle_pnl_f'].sum() for g in funnel_gates]

    y_pos = np.arange(len(funnel_gates))
    bars = ax_funnel.barh(y_pos, funnel_counts, color=funnel_colors, alpha=0.8)
    ax_funnel.set_yticks(y_pos)
    ax_funnel.set_yticklabels(funnel_labels, fontsize=9)
    ax_funnel.set_xlabel('Signal Count', fontsize=10)
    ax_funnel.set_title('Gate Funnel', fontsize=12, color='white', pad=10)
    ax_funnel.invert_yaxis()

    # Annotate bars with count and oracle $
    for i, (cnt, oracle) in enumerate(zip(funnel_counts, funnel_oracle)):
        ax_funnel.text(cnt + max(funnel_counts) * 0.02, i,
                       f'{cnt:,} (${oracle/1000:.0f}K)',
                       va='center', fontsize=8, color='#cccccc')

    # ── Top patterns table (text box) ────────────────────────────────────
    top_templates = (sig[sig['template_id'].notna()]
                     .groupby('template_id')
                     .agg(count=('ts', 'size'),
                          traded=('gate', lambda x: (x == 'traded').sum()),
                          oracle_total=('oracle_pnl_f', 'sum'),
                          oracle_avg=('oracle_pnl_f', 'mean'))
                     .sort_values('oracle_total', ascending=False)
                     .head(10))

    if not top_templates.empty:
        table_text = "TOP 10 TEMPLATES (by oracle $):\n"
        table_text += f"{'ID':>5} {'Det':>5} {'Trd':>4} {'Orc$':>8} {'Avg$':>6}\n"
        table_text += "-" * 32 + "\n"
        for tid, row in top_templates.iterrows():
            table_text += (f"{int(tid):>5} {int(row['count']):>5} "
                          f"{int(row['traded']):>4} "
                          f"${row['oracle_total']:>7,.0f} "
                          f"${row['oracle_avg']:>5.0f}\n")

        ax_main.text(0.99, 0.02, table_text, transform=ax_main.transAxes,
                     fontsize=7, verticalalignment='bottom',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#111111', alpha=0.85),
                     color='#cccccc', family='monospace')

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────
    output = args.output or f'reports/pattern_map_{args.month}.png'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'\nSaved: {output}  ({args.dpi} DPI)')


if __name__ == '__main__':
    main()
