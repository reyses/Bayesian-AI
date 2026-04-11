"""
Report — end-of-run IS/OOS comparison with modes + typical month.

Reads per-day CSVs from the brain builder and produces the reality check.
Mode over mean: mode shows what a TYPICAL day looks like.
Typical month = winning_days × mode_win - losing_days × mode_loss.

Usage:
    python training/report.py                      # default: read latest per-day CSVs
    python training/report.py --is-csv PATH --oos-csv PATH
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TREE_DIR = 'training/output/reports'

# Typical trading days per month
TRADING_DAYS_PER_MONTH = 21


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='IS/OOS comparison report')
    p.add_argument('--is-csv', type=str,
                   default=os.path.join(TREE_DIR, 'perday_is_daily.csv'))
    p.add_argument('--oos-csv', type=str,
                   default=os.path.join(TREE_DIR, 'perday_oos_daily.csv'))
    return p.parse_args()


def mode_pnl(values):
    """Compute mode of PnL values, bucketed to nearest $25."""
    if len(values) == 0:
        return 0.0
    bucket_size = 25.0
    bucketed = [round(v / bucket_size) * bucket_size for v in values]
    counts = Counter(bucketed)
    return counts.most_common(1)[0][0]


def compute_stats(df, label):
    """Compute comprehensive stats for one dataset."""
    n_days = len(df)
    if n_days == 0:
        return {}

    total_pnl = df['pnl'].sum()
    total_trades = df['trades'].sum()
    winning_mask = df['pnl'] > 0
    losing_mask = df['pnl'] <= 0
    zero_mask = df['trades'] == 0

    winning_days = df[winning_mask]
    losing_days = df[losing_mask & ~zero_mask]  # exclude zero-trade days from losing
    zero_days = df[zero_mask]

    # Active days (had at least 1 trade)
    active_days = df[~zero_mask]
    n_active = len(active_days)

    # Low trade days
    low_trade_mask = (df['trades'] > 0) & (df['trades'] <= 10)
    low_trade_days = df[low_trade_mask]

    # Chains
    chained = df['chained'].sum() if 'chained' in df.columns else 0

    # Drawdown analysis
    cumul = df['pnl'].cumsum()
    peak_equity = cumul.cummax()
    drawdown = cumul - peak_equity

    # Consecutive losing days
    is_loss = (df['pnl'] <= 0).values
    max_consec_loss = 0
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Weekly PnL (group by 5-day blocks)
    weekly_pnls = []
    for i in range(0, n_days, 5):
        week = df.iloc[i:i+5]
        weekly_pnls.append(week['pnl'].sum())

    return {
        'label': label,
        'n_days': n_days,
        'n_active': n_active,
        'total_pnl': total_pnl,
        'per_day': total_pnl / max(n_active, 1),
        'total_trades': int(total_trades),
        'trades_per_day': total_trades / max(n_active, 1),
        # Winning days
        'win_count': len(winning_days),
        'win_pct': len(winning_days) / max(n_active, 1),
        'win_mode': mode_pnl(winning_days['pnl'].values),
        'win_avg': winning_days['pnl'].mean() if len(winning_days) > 0 else 0,
        'win_median': winning_days['pnl'].median() if len(winning_days) > 0 else 0,
        'win_trades_avg': winning_days['trades'].mean() if len(winning_days) > 0 else 0,
        # Losing days
        'loss_count': len(losing_days),
        'loss_pct': len(losing_days) / max(n_active, 1),
        'loss_mode': mode_pnl(losing_days['pnl'].values),
        'loss_avg': losing_days['pnl'].mean() if len(losing_days) > 0 else 0,
        'loss_median': losing_days['pnl'].median() if len(losing_days) > 0 else 0,
        'loss_trades_avg': losing_days['trades'].mean() if len(losing_days) > 0 else 0,
        # Zero days
        'zero_count': len(zero_days),
        # Low trade days
        'low_trade_count': len(low_trade_days),
        'low_trade_avg': low_trade_days['pnl'].mean() if len(low_trade_days) > 0 else 0,
        'low_trade_wins': int((low_trade_days['pnl'] > 0).sum()),
        'low_trade_losses': int((low_trade_days['pnl'] <= 0).sum()),
        # Chains
        'chained': int(chained),
        'chain_pct': chained / max(total_trades, 1),
        # Drawdown
        'max_dd': float(drawdown.min()),
        'max_consec_loss': max_consec_loss,
        'worst_day': float(df['pnl'].min()),
        'best_day': float(df['pnl'].max()),
        'worst_week': min(weekly_pnls) if weekly_pnls else 0,
        'best_week': max(weekly_pnls) if weekly_pnls else 0,
    }


def typical_month(stats):
    """Estimate typical month from mode stats."""
    win_days = round(TRADING_DAYS_PER_MONTH * stats['win_pct'])
    loss_days = round(TRADING_DAYS_PER_MONTH * stats['loss_pct'])
    carry = win_days * stats['win_mode']
    drag = loss_days * abs(stats['loss_mode'])
    net = carry - drag
    return win_days, loss_days, carry, drag, net


def format_report(is_stats, oos_stats):
    """Format the comparison report."""
    lines = []

    def out(s=''):
        lines.append(s)

    out(f'{"="*65}')
    out(f'  SYSTEM REPORT — IS vs OOS Comparison')
    out(f'{"="*65}')

    # Side-by-side table
    is_s = is_stats
    oos_s = oos_stats
    has_oos = oos_s.get('n_days', 0) > 0

    def row(label, is_val, oos_val='', fmt=''):
        if has_oos:
            out(f'  {label:<25} {is_val:>15}  {oos_val:>15}')
        else:
            out(f'  {label:<25} {is_val:>15}')

    header_is = f'IS ({is_s["n_days"]} days)'
    header_oos = f'OOS ({oos_s["n_days"]} days)' if has_oos else ''
    out(f'  {"":25} {header_is:>15}  {header_oos:>15}')
    out(f'  {"-"*60}')

    row('Total PnL', f'${is_s["total_pnl"]:,.0f}', f'${oos_s.get("total_pnl",0):,.0f}')
    row('$/day', f'${is_s["per_day"]:.0f}', f'${oos_s.get("per_day",0):.0f}')
    row('Total trades', f'{is_s["total_trades"]:,}', f'{oos_s.get("total_trades",0):,}')
    row('Trades/day', f'{is_s["trades_per_day"]:.1f}', f'{oos_s.get("trades_per_day",0):.1f}')

    out(f'\n  WINNING DAYS:')
    row('  Count', f'{is_s["win_count"]}', f'{oos_s.get("win_count",0)}')
    row('  Win %', f'{is_s["win_pct"]:.0%}', f'{oos_s.get("win_pct",0):.0%}')
    row('  Mode PnL', f'${is_s["win_mode"]:.0f}', f'${oos_s.get("win_mode",0):.0f}')
    row('  Avg PnL', f'${is_s["win_avg"]:.0f}', f'${oos_s.get("win_avg",0):.0f}')
    row('  Median PnL', f'${is_s["win_median"]:.0f}', f'${oos_s.get("win_median",0):.0f}')
    row('  Avg trades/day', f'{is_s["win_trades_avg"]:.1f}', f'{oos_s.get("win_trades_avg",0):.1f}')

    out(f'\n  LOSING DAYS:')
    row('  Count', f'{is_s["loss_count"]}', f'{oos_s.get("loss_count",0)}')
    row('  Loss %', f'{is_s["loss_pct"]:.0%}', f'{oos_s.get("loss_pct",0):.0%}')
    row('  Mode PnL', f'${is_s["loss_mode"]:.0f}', f'${oos_s.get("loss_mode",0):.0f}')
    row('  Avg PnL', f'${is_s["loss_avg"]:.0f}', f'${oos_s.get("loss_avg",0):.0f}')
    row('  Median PnL', f'${is_s["loss_median"]:.0f}', f'${oos_s.get("loss_median",0):.0f}')
    row('  Avg trades/day', f'{is_s["loss_trades_avg"]:.1f}', f'{oos_s.get("loss_trades_avg",0):.1f}')

    out(f'\n  LOW TRADE DAYS (1-10 trades):')
    row('  Count', f'{is_s["low_trade_count"]}', f'{oos_s.get("low_trade_count",0)}')
    row('  Avg PnL', f'${is_s["low_trade_avg"]:.0f}', f'${oos_s.get("low_trade_avg",0):.0f}')
    row('  Win/Loss', f'{is_s["low_trade_wins"]}/{is_s["low_trade_losses"]}',
        f'{oos_s.get("low_trade_wins",0)}/{oos_s.get("low_trade_losses",0)}')

    out(f'\n  CHAINS:')
    row('  Chained trades', f'{is_s["chained"]}', f'{oos_s.get("chained",0)}')
    row('  Chain %', f'{is_s["chain_pct"]:.0%}', f'{oos_s.get("chain_pct",0):.0%}')

    # Typical month
    out(f'\n  TYPICAL MONTH ESTIMATE:')
    is_wd, is_ld, is_carry, is_drag, is_net = typical_month(is_s)
    out(f'    IS:  {is_wd} winning × ${is_s["win_mode"]:.0f} = ${is_carry:,.0f} carry')
    out(f'         {is_ld} losing  × ${abs(is_s["loss_mode"]):.0f} = ${is_drag:,.0f} drag')
    out(f'         Net typical month: ${is_net:,.0f}')

    if has_oos:
        oos_wd, oos_ld, oos_carry, oos_drag, oos_net = typical_month(oos_s)
        out(f'    OOS: {oos_wd} winning × ${oos_s["win_mode"]:.0f} = ${oos_carry:,.0f} carry')
        out(f'         {oos_ld} losing  × ${abs(oos_s["loss_mode"]):.0f} = ${oos_drag:,.0f} drag')
        out(f'         Net typical month: ${oos_net:,.0f}')

    # Drawdown
    out(f'\n  DRAWDOWN ANALYSIS:')
    row('  Max drawdown', f'${is_s["max_dd"]:.0f}', f'${oos_s.get("max_dd",0):.0f}')
    row('  Max consec losses', f'{is_s["max_consec_loss"]}', f'{oos_s.get("max_consec_loss",0)}')
    row('  Worst day', f'${is_s["worst_day"]:.0f}', f'${oos_s.get("worst_day",0):.0f}')
    row('  Best day', f'${is_s["best_day"]:.0f}', f'${oos_s.get("best_day",0):.0f}')
    row('  Worst week', f'${is_s["worst_week"]:.0f}', f'${oos_s.get("worst_week",0):.0f}')
    row('  Best week', f'${is_s["best_week"]:.0f}', f'${oos_s.get("best_week",0):.0f}')

    # Risk guardrails
    out(f'\n  RISK GUARDRAILS (from drawdown data):')
    daily_stop = abs(is_s['loss_mode']) * 1.5
    weekly_pause = abs(is_s['worst_week']) * 0.8
    out(f'    Daily stop loss:  ${daily_stop:.0f} (mode loss × 1.5)')
    out(f'    Weekly equity pause: ${weekly_pause:.0f} (worst week × 0.8)')
    out(f'    Max drawdown seen: ${abs(is_s["max_dd"]):.0f}')

    out(f'\n{"="*65}')

    return '\n'.join(lines)


def main():
    args = parse_args()

    # Load IS
    is_stats = {}
    if os.path.exists(args.is_csv):
        is_df = pd.read_csv(args.is_csv)
        is_stats = compute_stats(is_df, 'IS')
        print(f'IS: {len(is_df)} days loaded from {args.is_csv}')
    else:
        print(f'IS CSV not found: {args.is_csv}')
        # Try AI report CSV as fallback
        alt = os.path.join(TREE_DIR, 'ai_is_daily.csv')
        if os.path.exists(alt):
            is_df = pd.read_csv(alt)
            is_stats = compute_stats(is_df, 'IS')
            print(f'IS: {len(is_df)} days loaded from {alt} (fallback)')

    # Load OOS
    oos_stats = {}
    if os.path.exists(args.oos_csv):
        oos_df = pd.read_csv(args.oos_csv)
        oos_stats = compute_stats(oos_df, 'OOS')
        print(f'OOS: {len(oos_df)} days loaded from {args.oos_csv}')
    else:
        # Try AI report CSV as fallback
        alt = os.path.join(TREE_DIR, 'ai_oos_daily.csv')
        if os.path.exists(alt):
            oos_df = pd.read_csv(alt)
            oos_stats = compute_stats(oos_df, 'OOS')
            print(f'OOS: {len(oos_df)} days loaded from {alt} (fallback)')
        else:
            print(f'OOS CSV not found (skipping OOS comparison)')

    if not is_stats:
        print('No data to report.')
        return

    report = format_report(is_stats, oos_stats)
    print(report)

    # Save
    os.makedirs(TREE_DIR, exist_ok=True)
    report_path = os.path.join(TREE_DIR, 'system_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nReport saved: {report_path}')


if __name__ == '__main__':
    main()
