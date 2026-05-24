"""Quick PnL distribution analysis from trade CSV."""
import pandas as pd
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'advance_engine/results/trades.csv'
trades = pd.read_csv(path)

losers = trades[trades['pnl_dollars'] < 0]
winners = trades[trades['pnl_dollars'] > 0]
flat = trades[trades['pnl_dollars'] == 0]

print(f'=== PnL Distribution ({len(trades)} trades) ===')
print(f'Winners: {len(winners)} (+${winners["pnl_dollars"].sum():,.2f})')
print(f'Losers:  {len(losers)} (${losers["pnl_dollars"].sum():,.2f})')
print(f'Flat:    {len(flat)}')
print(f'Net:     ${trades["pnl_dollars"].sum():,.2f}')
print()

if len(losers) > 0:
    pnls = losers['pnl_dollars'].values
    print(f'=== LOSERS ===')
    print(f'  Avg: ${pnls.mean():.2f} | Median: ${np.median(pnls):.2f} | Worst: ${pnls.min():.2f}')
    print(f'  Buckets:')
    for lo, hi, label in [(0,-2,'$0 to -$2'),(-2,-5,'-$2 to -$5'),(-5,-10,'-$5 to -$10'),
                           (-10,-20,'-$10 to -$20'),(-20,-40,'-$20 to -$40'),(-40,-80,'-$40 to -$80'),
                           (-80,-999,'-$80+')]:
        mask = (pnls <= lo) & (pnls > hi)
        n = mask.sum()
        if n > 0:
            print(f'    {label:>15}: {n:>4} trades  ${pnls[mask].sum():>+10,.2f}')
    print(f'  By exit reason:')
    for reason, grp in losers.groupby('reason'):
        print(f'    {reason:<35} {len(grp):>4}  ${grp["pnl_dollars"].sum():>+10,.2f}  avg=${grp["pnl_dollars"].mean():>+.2f}')

print()
if len(winners) > 0:
    pnls = winners['pnl_dollars'].values
    print(f'=== WINNERS ===')
    print(f'  Avg: ${pnls.mean():.2f} | Median: ${np.median(pnls):.2f} | Best: ${pnls.max():.2f}')
    print(f'  Buckets:')
    for lo, hi, label in [(0,2,'$0 to $2'),(2,5,'$2 to $5'),(5,10,'$5 to $10'),
                           (10,20,'$10 to $20'),(20,40,'$20 to $40'),(40,80,'$40 to $80'),
                           (80,999,'$80+')]:
        mask = (pnls >= lo) & (pnls < hi)
        n = mask.sum()
        if n > 0:
            print(f'    {label:>15}: {n:>4} trades  ${pnls[mask].sum():>+10,.2f}')

print()
print(f'=== DIRECTION ===')
for d in trades['direction'].unique():
    sub = trades[trades['direction'] == d]
    w = (sub['pnl_dollars'] > 0).sum()
    print(f'  {d.upper():>5}: {len(sub)} trades, WR={w/len(sub)*100:.1f}%, ${sub["pnl_dollars"].sum():+,.2f}')
