"""Per-pattern regret report — sorted by WR, high to low."""
import pickle
import numpy as np
import pandas as pd

# Load pattern assignments
patterns = {}
for tier in ['cascade', 'kill_shot', 'base_nmp']:
    path = f'nn_v2/output/entry/patterns_{tier}.pkl'
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for d in data:
            patterns[d['trade_idx']] = d
    except FileNotFoundError:
        pass

# Load regret + trades
regret = pd.read_csv('nn_v2/output/tree/regret_analysis.csv')
with open('nn_v2/output/trades/blended_is.pkl', 'rb') as f:
    trades = pickle.load(f)

# Build table
rows = []
for i in range(min(len(trades), len(regret))):
    label = patterns[i]['pattern_label'] if i in patterns else ''
    if not label:
        continue
    rows.append({
        'pattern': label,
        'actual_pnl': trades[i].get('pnl', 0),
        'best_pnl': regret.iloc[i]['best_pnl'],
        'best_action': regret.iloc[i]['best_action'],
        'early_bars': regret.iloc[i].get('best_early_bars_before', 0),
    })

df = pd.DataFrame(rows)

# Stats per pattern
print('BLENDED BY PATTERN (sorted by WR)')
print('=' * 95)
hdr = f'  {"Pattern":<16} {"N":>5} {"WR":>5} {"ActPnL":>10} {"OptPnL":>10} {"Capt":>6} {"Counter%":>8} {"EarlyBars":>9}'
print(hdr)
print('  ' + '-' * 90)

stats = []
for label in sorted(df['pattern'].unique()):
    sub = df[df['pattern'] == label]
    n = len(sub)
    wr = (sub['actual_pnl'] > 0).mean() * 100
    actual = sub['actual_pnl'].sum()
    optimal = sub['best_pnl'].sum()
    capture = actual / max(optimal, 1) * 100
    counter = sub['best_action'].str.contains('counter').mean() * 100
    early = sub['early_bars'].mean()
    stats.append((label, n, wr, actual, optimal, capture, counter, early))

stats.sort(key=lambda x: -x[2])
for label, n, wr, actual, optimal, capture, counter, early in stats:
    print(f'  {label:<16} {n:>5} {wr:>4.0f}% ${actual:>9,.0f} ${optimal:>9,.0f} '
          f'{capture:>5.1f}% {counter:>7.0f}% {early:>8.0f}')

# Deep dive: best action distribution per BASE_NMP pattern
print('\n' + '=' * 95)
print('BASE_NMP DEEP DIVE — Best Action by Pattern')
print('=' * 95)

for label, n, wr, actual, optimal, capture, counter, early in stats:
    if not label.startswith('BASE_NMP'):
        continue
    sub = df[df['pattern'] == label]
    print(f'\n  {label} ({n} trades, WR={wr:.0f}%, actual=${actual:,.0f}, optimal=${optimal:,.0f})')
    for action, count in sub['best_action'].value_counts().items():
        pct = count / len(sub) * 100
        avg_best = sub[sub['best_action'] == action]['best_pnl'].mean()
        avg_actual = sub[sub['best_action'] == action]['actual_pnl'].mean()
        print(f'    {action:<22} {count:>5} ({pct:>4.0f}%)  '
              f'avg_best=${avg_best:.1f}  avg_actual=${avg_actual:.1f}')

# Simulation
print('\n' + '=' * 95)
print('SIMULATION: Filter by pattern')
print('=' * 95)

for threshold in [50, 30, 25]:
    good = [s for s in stats if s[2] > threshold]
    bad = [s for s in stats if s[2] <= threshold]
    good_pnl = sum(s[3] for s in good)
    good_n = sum(s[1] for s in good)
    bad_pnl = sum(s[3] for s in bad)
    bad_n = sum(s[1] for s in bad)
    total = good_pnl + bad_pnl
    kept = ', '.join(s[0] for s in good)
    print(f'  WR > {threshold}%: KEEP {good_n} trades (${good_pnl:,.0f}) | '
          f'KILL {bad_n} trades (${bad_pnl:,.0f}) | '
          f'vs baseline ${total:,.0f} | delta=${good_pnl - total:+,.0f}')
    print(f'    Kept: {kept}')
