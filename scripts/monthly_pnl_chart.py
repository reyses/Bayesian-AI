"""Monthly PnL bar chart + cumulative line overlay."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import sys, os

def plot_monthly_pnl(csv_path: str, title_suffix: str = ""):
    df = pd.read_csv(csv_path)
    df['entry_dt'] = pd.to_datetime(df['entry_time'], unit='s')
    df['month'] = df['entry_dt'].dt.to_period('M')

    monthly = df.groupby('month').agg(
        pnl=('actual_pnl', 'sum'),
        trades=('actual_pnl', 'count'),
        wr=('result', lambda x: (x == 'WIN').mean() * 100),
    )
    monthly['cumulative'] = monthly['pnl'].cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#1a1a2e')

    months = [str(m) for m in monthly.index]
    x = range(len(months))
    colors = ['#00cc66' if v >= 0 else '#ff4444' for v in monthly['pnl']]
    bars = ax1.bar(x, monthly['pnl'], color=colors, alpha=0.85, width=0.6, zorder=2)

    # Labels on bars
    for i, (v, t, w) in enumerate(zip(monthly['pnl'], monthly['trades'], monthly['wr'])):
        label = f"${v:,.0f}\n{t}t | {w:.0f}%"
        y_off = 200 if v >= 0 else -200
        va = 'bottom' if v >= 0 else 'top'
        ax1.text(i, v + y_off, label, ha='center', va=va, fontsize=8,
                 color='white', fontweight='bold')

    ax1.set_ylabel('Monthly PnL ($)', color='#cccccc', fontsize=11)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(months, rotation=45, ha='right', color='#cccccc')
    ax1.tick_params(axis='y', colors='#cccccc')
    ax1.axhline(y=0, color='#555555', linewidth=0.8, linestyle='--')
    ax1.grid(axis='y', alpha=0.2, color='#444444')

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(list(x), monthly['cumulative'].values, color='#ffaa00', linewidth=2.5,
             marker='o', markersize=6, zorder=3, label='Cumulative PnL')
    ax2.set_ylabel('Cumulative PnL ($)', color='#ffaa00', fontsize=11)
    ax2.tick_params(axis='y', colors='#ffaa00')

    # Format y-axes as dollars
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    title = f"Monthly PnL Breakdown{title_suffix}"
    total = monthly['pnl'].sum()
    total_trades = monthly['trades'].sum()
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold', y=0.98)
    ax1.set_title(f"Total: ${total:,.0f}  |  {total_trades} trades", color='#aaaaaa',
                  fontsize=10, pad=10)

    ax2.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#555555',
               labelcolor='#ffaa00')
    fig.tight_layout()

    out_dir = os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, 'monthly_pnl_chart.png')
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Chart saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_pre_snowflake/oracle_trade_log.csv'
    suffix = f" — {sys.argv[2]}" if len(sys.argv) > 2 else " — Pre-Snowflake Baseline"
    plot_monthly_pnl(csv, suffix)
