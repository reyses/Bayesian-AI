"""
Diagnose why ~28% of days produce ~$0 PnL in pivot_physics_chains.

Per day, capture:
  - n_trades (did we trade at all?)
  - gross_win, gross_loss, net_pnl
  - 1m_z_se activity: n_bars with |z|>0.5 (entry gate)
  - day range (price max - min in $)
  - day-of-week

Classify days:
  A) No trades: signal never fired. Why? (low residual activity / no 1s pivots)
  B) Trades fired, net ≈ 0: wins cancelled losses
  C) Trades fired, net small positive (<$25): mode bucket

Output:
  reports/findings/pivot_zero_day_diagnosis.md
  charts/pivot_zero_day_diagnosis.png
  tools/_cache/pivot_per_day.parquet   (for downstream reuse)

Usage:
    python tools/pivot_zero_day_diagnosis.py
"""
import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pivot_physics_exit import load_day, DOLLAR_PER_POINT
from tools.pivot_physics_chains import simulate as simulate_chains

ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_zero_day_diagnosis.md'
OUT_CHART = 'charts/pivot_zero_day_diagnosis.png'
OUT_PARQUET = 'tools/_cache/pivot_per_day.parquet'

MODE_LOW = 0
MODE_HIGH = 25  # "mode bucket" = days in [0, 25)


def day_to_dow(day_str):
    """Day string like 2025_06_09 → day-of-week 0=Mon..6=Sun."""
    return datetime.strptime(day_str, '%Y_%m_%d').weekday()


def collect_days(paths, label):
    rows = []
    for p in tqdm(paths, desc=label, unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        sec_path = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
        feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(sec_path) or not os.path.exists(feat_path):
            continue
        loaded = load_day(sec_path, p, feat_path)
        if loaded is None:
            continue
        sec, closes_1m, ts_1m, residuals_1s, residuals_1m = loaded

        # Diagnostics on raw data (independent of sim)
        closes_pts = np.asarray(closes_1m, dtype=np.float64)
        day_range_dollars = (closes_pts.max() - closes_pts.min()) * DOLLAR_PER_POINT
        res_1m_arr = np.asarray(residuals_1m, dtype=np.float64)
        res_1m_valid = res_1m_arr[~np.isnan(res_1m_arr)]
        n_strong_res_bars = int((np.abs(res_1m_valid) >= 0.5).sum()) if len(res_1m_valid) else 0
        pct_strong_res = (n_strong_res_bars / max(len(res_1m_valid), 1)) * 100

        trades = simulate_chains(sec, closes_1m, ts_1m, residuals_1s,
                                 residuals_1m,
                                 r_entry_pts=1.0, r_reg_pts=4.0,
                                 min_res=0.5, sniper_sec=30,
                                 max_chains=1)

        pnls = [t['pnl'] for t in trades]
        n_trades = len(pnls)
        gross_win = float(sum(p for p in pnls if p > 0))
        gross_loss = float(sum(p for p in pnls if p < 0))
        net = float(sum(pnls))
        n_wins = int(sum(1 for p in pnls if p > 0))
        n_losses = int(sum(1 for p in pnls if p < 0))

        # Classification
        if n_trades == 0:
            cls = 'A_no_trades'
        elif MODE_LOW <= net < MODE_HIGH:
            if gross_win > 0 and abs(gross_loss) > gross_win * 0.5:
                cls = 'B_wash_small_positive'
            else:
                cls = 'C_small_positive_few_trades'
        elif abs(net) < 5 and n_trades > 0:
            cls = 'B_wash_exact'
        else:
            cls = 'Z_non_mode'

        rows.append({
            'day': day,
            'dataset': label,
            'dow': day_to_dow(day),
            'n_1m_bars': len(closes_pts),
            'day_range_dollars': day_range_dollars,
            'n_strong_res_bars': n_strong_res_bars,
            'pct_strong_res': pct_strong_res,
            'n_trades': n_trades,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'gross_win': gross_win,
            'gross_loss': gross_loss,
            'net_pnl': net,
            'cls': cls,
        })
    return rows


def main():
    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    rows = collect_days(is_paths, 'IS') + collect_days(oos_paths, 'OOS')
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f'Wrote: {OUT_PARQUET}')

    # === Analysis ===
    lines = ['# Zero-day diagnosis — pivot_physics_chains chains=1', '']
    lines.append(f'Total days: {len(df)} (IS {(df.dataset=="IS").sum()} / '
                 f'OOS {(df.dataset=="OOS").sum()})')
    lines.append('')

    # Classification counts
    lines.append('## Classification')
    lines.append('')
    lines.append('| Class | Description | IS | OOS | All |')
    lines.append('|---|---|---:|---:|---:|')
    for cls_name, desc in [
        ('A_no_trades', 'No trades fired'),
        ('B_wash_small_positive', 'Trades fired, net ∈ [0,25), gross_loss/gross_win > 50%'),
        ('B_wash_exact', 'Trades fired, |net| < $5'),
        ('C_small_positive_few_trades', 'Trades fired, net ∈ [0,25), minimal offsets'),
        ('Z_non_mode', 'Days outside mode bucket'),
    ]:
        n_is = int(((df.cls == cls_name) & (df.dataset == 'IS')).sum())
        n_oos = int(((df.cls == cls_name) & (df.dataset == 'OOS')).sum())
        lines.append(f'| {cls_name} | {desc} | {n_is} | {n_oos} | {n_is + n_oos} |')
    lines.append('')

    # Focus: A_no_trades
    no_trade = df[df.cls == 'A_no_trades']
    lines.append(f'## A_no_trades: {len(no_trade)} days')
    if len(no_trade) > 0:
        lines.append('')
        lines.append('Why no trades? Check data availability + signal strength.')
        lines.append('')
        lines.append(f'- Mean day range: ${no_trade.day_range_dollars.mean():,.0f}')
        lines.append(f'- Median day range: ${no_trade.day_range_dollars.median():,.0f}')
        lines.append(f'- Mean n_strong_res_bars: {no_trade.n_strong_res_bars.mean():.0f}')
        lines.append(f'- Mean n_1m_bars: {no_trade.n_1m_bars.mean():.0f}')
        lines.append('')
        lines.append('Compare to days with trades:')
        with_trades = df[df.n_trades > 0]
        lines.append(f'- Mean day range (traded): ${with_trades.day_range_dollars.mean():,.0f}')
        lines.append(f'- Mean n_strong_res_bars: {with_trades.n_strong_res_bars.mean():.0f}')
        lines.append(f'- Mean n_1m_bars: {with_trades.n_1m_bars.mean():.0f}')
        lines.append('')
        # Top 10 no-trade days by range (days with activity but no trades)
        top = no_trade.sort_values('day_range_dollars', ascending=False).head(10)
        lines.append('### Top-10 no-trade days by range (puzzling — why no entry?)')
        lines.append('')
        lines.append('| Day | DoW | Range | n_1m | n_strong_res | pct_strong |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for _, r in top.iterrows():
            dow = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(r.dow)]
            lines.append(f'| {r.day} | {dow} | ${r.day_range_dollars:,.0f} | '
                         f'{r.n_1m_bars} | {r.n_strong_res_bars} | '
                         f'{r.pct_strong_res:.0f}% |')
        lines.append('')
        # DoW distribution
        lines.append('### No-trade day distribution by day-of-week')
        lines.append('')
        lines.append('| DoW | Count | % of no-trade |')
        lines.append('|---|---:|---:|')
        for d in range(7):
            n = int((no_trade.dow == d).sum())
            lines.append(f'| {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]} | '
                         f'{n} | {n/max(len(no_trade),1)*100:.0f}% |')
        lines.append('')

    # B_wash diagnosis
    wash = df[df.cls.isin(['B_wash_small_positive', 'B_wash_exact'])]
    lines.append(f'## B_wash: {len(wash)} days (trades fired, net near zero)')
    if len(wash) > 0:
        lines.append('')
        lines.append(f'- Mean n_trades: {wash.n_trades.mean():.1f}')
        lines.append(f'- Mean gross_win: ${wash.gross_win.mean():+,.0f}')
        lines.append(f'- Mean gross_loss: ${wash.gross_loss.mean():+,.0f}')
        lines.append(f'- Mean net: ${wash.net_pnl.mean():+,.0f}')
        lines.append('')

    # C_small_positive diagnosis
    smallpos = df[df.cls == 'C_small_positive_few_trades']
    lines.append(f'## C_small_positive: {len(smallpos)} days')
    if len(smallpos) > 0:
        lines.append('')
        lines.append(f'- Mean n_trades: {smallpos.n_trades.mean():.1f}')
        lines.append(f'- Mean gross_win: ${smallpos.gross_win.mean():+,.0f}')
        lines.append(f'- Mean gross_loss: ${smallpos.gross_loss.mean():+,.0f}')
        lines.append('')

    # Day range distribution by class
    lines.append('## Day range by class')
    lines.append('')
    lines.append('| Class | N | Mean range | Median range |')
    lines.append('|---|---:|---:|---:|')
    for cls in df.cls.unique():
        sub = df[df.cls == cls]
        lines.append(f'| {cls} | {len(sub)} | ${sub.day_range_dollars.mean():,.0f} | '
                     f'${sub.day_range_dollars.median():,.0f} |')
    lines.append('')

    # Write
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Wrote: {OUT_MD}')
    print()
    print('\n'.join(lines))

    # Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. n_trades vs day_range
    ax = axes[0][0]
    for ds, color in [('IS', 'tab:blue'), ('OOS', 'tab:orange')]:
        sub = df[df.dataset == ds]
        ax.scatter(sub.day_range_dollars, sub.n_trades, s=10, alpha=0.5,
                   label=ds, color=color)
    ax.set_xlabel('Day range ($)')
    ax.set_ylabel('N trades')
    ax.set_title('Trades vs day range')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. n_strong_res vs n_trades
    ax = axes[0][1]
    for ds, color in [('IS', 'tab:blue'), ('OOS', 'tab:orange')]:
        sub = df[df.dataset == ds]
        ax.scatter(sub.n_strong_res_bars, sub.n_trades, s=10, alpha=0.5,
                   label=ds, color=color)
    ax.set_xlabel('N bars with |residual| ≥ 0.5')
    ax.set_ylabel('N trades')
    ax.set_title('Trades vs residual activity')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. PnL distribution with classes colored
    ax = axes[1][0]
    for cls, color in [('A_no_trades', 'tab:gray'),
                        ('B_wash_small_positive', 'tab:orange'),
                        ('B_wash_exact', 'tab:red'),
                        ('C_small_positive_few_trades', 'tab:green'),
                        ('Z_non_mode', 'tab:blue')]:
        sub = df[df.cls == cls]
        ax.hist(sub.net_pnl, bins=50, alpha=0.6, label=f'{cls} ({len(sub)})',
                color=color)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Net PnL ($)')
    ax.set_ylabel('Days')
    ax.set_title('Per-day PnL colored by class')
    ax.set_xlim(-1500, 1500)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. DoW breakdown
    ax = axes[1][1]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, cls in enumerate(['A_no_trades', 'Z_non_mode']):
        sub = df[df.cls == cls]
        counts = [int((sub.dow == d).sum()) for d in range(7)]
        ax.bar(np.arange(7) + i * 0.4, counts, width=0.4, label=cls,
                alpha=0.7)
    ax.set_xticks(np.arange(7) + 0.2)
    ax.set_xticklabels(dow_names)
    ax.set_ylabel('Days')
    ax.set_title('No-trade vs traded days by DoW')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'Wrote: {OUT_CHART}')


if __name__ == '__main__':
    main()
