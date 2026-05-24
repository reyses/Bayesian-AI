"""
Post-Run Regret Analysis
========================
Measures how much money each exit reason left on the table.
Regret = trade MFE achieved - actual PnL captured.

Embedded in OOS report generation and available as standalone tool.

Usage:
    python tools/regret_analysis.py                          # latest IS + OOS
    python tools/regret_analysis.py --file checkpoints/oos_trade_log.csv
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK_VALUE = 0.50  # MNQ


def run_regret_analysis(df: pd.DataFrame, label: str = '',
                        tick_value: float = TICK_VALUE) -> str:
    """Run regret analysis on a trade log DataFrame.

    Returns formatted report string (for embedding in reports or printing).
    """
    lines = []
    lines.append(f"{'='*75}")
    lines.append(f"  REGRET ANALYSIS{f' — {label}' if label else ''}")
    lines.append(f"  {len(df):,} trades analyzed")
    lines.append(f"{'='*75}")

    if 'trade_mfe_ticks' not in df.columns or 'actual_pnl' not in df.columns:
        lines.append("  ERROR: trade_mfe_ticks or actual_pnl column missing")
        return '\n'.join(lines)

    df = df.copy()
    df['actual_pnl_ticks'] = df['actual_pnl'] / tick_value
    df['regret_ticks'] = df['trade_mfe_ticks'] - df['actual_pnl_ticks']
    df['regret_dollars'] = df['regret_ticks'] * tick_value

    # --- By exit reason ---
    lines.append(f"\n  BY EXIT REASON (sorted by total regret):")
    lines.append(f"  {'Exit':25s} {'n':>5s} {'Avg Regret':>11s} {'Total Regret':>13s} "
                 f"{'Avg PnL':>9s} {'Avg MFE(t)':>10s} {'Capture%':>9s}")
    lines.append(f"  {'-'*85}")

    reason_regret = df.groupby('exit_reason')['regret_dollars'].sum().sort_values(ascending=False)
    for reason in reason_regret.index:
        sub = df[df['exit_reason'] == reason]
        n = len(sub)
        avg_reg = sub['regret_dollars'].mean()
        tot_reg = sub['regret_dollars'].sum()
        avg_pnl = sub['actual_pnl'].mean()
        avg_mfe = sub['trade_mfe_ticks'].mean()
        cap_pct = (sub['actual_pnl_ticks'] / sub['trade_mfe_ticks'].replace(0, np.nan)).mean() * 100
        lines.append(f"  {reason:25s} {n:>5d} {avg_reg:>+10.2f}$ "
                     f"{tot_reg:>+12.2f}$ {avg_pnl:>+8.2f}$ "
                     f"{avg_mfe:>9.1f}t {cap_pct:>+8.1f}%")

    total_regret = df['regret_dollars'].sum()
    total_pnl = df['actual_pnl'].sum()
    lines.append(f"\n  TOTAL PnL:    ${total_pnl:>12,.2f}")
    lines.append(f"  TOTAL REGRET: ${total_regret:>12,.2f}")
    lines.append(f"  CAPTURE RATE: {total_pnl / (total_pnl + total_regret) * 100:.1f}% "
                 f"of available profit captured")

    # --- Losing trades that were profitable ---
    losses = df[df['actual_pnl'] < 0]
    profitable_losses = losses[losses['trade_mfe_ticks'] > 10]
    lines.append(f"\n  LOSING TRADES THAT PEAKED > 5 TICKS:")
    lines.append(f"    {len(profitable_losses)} / {len(losses)} losses "
                 f"({len(profitable_losses)/max(1,len(losses))*100:.1f}%)")
    if len(profitable_losses) > 0:
        lines.append(f"    Avg peak MFE: {profitable_losses['trade_mfe_ticks'].mean():.1f} ticks "
                     f"(${profitable_losses['trade_mfe_ticks'].mean() * tick_value:.2f})")
        lines.append(f"    Avg actual PnL: ${profitable_losses['actual_pnl'].mean():.2f}")
        lines.append(f"    Total lost opportunity: "
                     f"${profitable_losses['regret_dollars'].sum():,.2f}")

        # Breakdown by exit reason for these reversals
        lines.append(f"\n    Exit reason breakdown (losing trades that peaked > 5t):")
        for reason in profitable_losses['exit_reason'].value_counts().index:
            sub = profitable_losses[profitable_losses['exit_reason'] == reason]
            lines.append(f"      {reason:25s} n={len(sub):>4d}  avg_peak={sub['trade_mfe_ticks'].mean():.0f}t  "
                         f"avg_pnl=${sub['actual_pnl'].mean():.2f}")

    # --- By depth ---
    if 'entry_depth' in df.columns:
        lines.append(f"\n  REGRET BY DEPTH:")
        lines.append(f"  {'Depth':>7s} {'n':>5s} {'Avg Regret':>11s} {'Capture%':>9s}")
        lines.append(f"  {'-'*35}")
        for d in sorted(df['entry_depth'].unique()):
            sub = df[df['entry_depth'] == d]
            avg_reg = sub['regret_dollars'].mean()
            cap = (sub['actual_pnl_ticks'] / sub['trade_mfe_ticks'].replace(0, np.nan)).mean() * 100
            lines.append(f"  {d:>7.0f} {len(sub):>5d} {avg_reg:>+10.2f}$ {cap:>+8.1f}%")

    # --- By duration ---
    if 'hold_bars' in df.columns:
        lines.append(f"\n  REGRET BY HOLD DURATION:")
        lines.append(f"  {'Duration':>10s} {'n':>5s} {'Avg Regret':>11s} {'Capture%':>9s}")
        lines.append(f"  {'-'*40}")
        df['hold_sec'] = df['hold_bars'] * 15
        for lo, hi, tag in [(0, 30, '<30s'), (30, 60, '30s-1m'), (60, 120, '1-2m'),
                            (120, 300, '2-5m'), (300, 99999, '5m+')]:
            sub = df[(df['hold_sec'] >= lo) & (df['hold_sec'] < hi)]
            if len(sub) == 0:
                continue
            avg_reg = sub['regret_dollars'].mean()
            cap = (sub['actual_pnl_ticks'] / sub['trade_mfe_ticks'].replace(0, np.nan)).mean() * 100
            lines.append(f"  {tag:>10s} {len(sub):>5d} {avg_reg:>+10.2f}$ {cap:>+8.1f}%")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Post-Run Regret Analysis')
    parser.add_argument('--file', default=None, help='Trade log CSV path')
    args = parser.parse_args()

    files_to_analyze = []
    if args.file:
        files_to_analyze.append((args.file, os.path.basename(args.file)))
    else:
        # Auto-detect IS + OOS trade logs
        candidates = [
            ('checkpoints/oracle_trade_log_old.csv', 'IS'),
            ('reports/is/oracle_trade_log.csv', 'IS'),
            ('checkpoints/oos_trade_log.csv', 'OOS'),
        ]
        for path, label in candidates:
            if os.path.exists(path):
                df_test = pd.read_csv(path, nrows=2)
                if len(df_test) > 1 and 'trade_mfe_ticks' in df_test.columns:
                    files_to_analyze.append((path, label))

    if not files_to_analyze:
        print("ERROR: No trade log found. Run training first.")
        return

    for path, label in files_to_analyze:
        df = pd.read_csv(path)
        report = run_regret_analysis(df, label)
        print(report)
        print()

    # Save report
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'reports/findings/regret_analysis_{ts}.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        for path, label in files_to_analyze:
            df = pd.read_csv(path)
            f.write(run_regret_analysis(df, label))
            f.write('\n\n')
    print(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()
