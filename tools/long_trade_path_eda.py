"""
Long Trade Path EDA — What happens during 18+ minute trades?

Look at the 79D trajectory through the trade path for the longest-held optimal trades.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

# Load data
with open('training/output/trades/blended_is.pkl', 'rb') as f:
    trades = pickle.load(f)
regret = pd.read_csv('training/output/nn/regret_analysis.csv')

FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES_79D)}
n = min(len(trades), len(regret))

# Key features to track through the path
TRACK = ['1m_z_se', '1m_p_at_center', '1m_velocity', '1m_variance_ratio',
         '1m_wick_ratio', '1m_reversion_prob', '5m_z_se', '5m_velocity']

# Find long-hold BASE_NMP trades (P90 = 224 bars = ~18 min)
long_trades = []
for i in range(n):
    t = trades[i]
    r = regret.iloc[i]
    tier = t.get('entry_tier', '')
    if tier != 'BASE_NMP':
        continue

    best_action = r['best_action']
    if 'same' in best_action:
        best_bar = int(r['same_best_bar'])
    else:
        best_bar = int(r['counter_best_bar'])

    path = t.get('path', [])
    if best_bar >= 200 and len(path) >= best_bar:
        long_trades.append((i, t, r, best_bar))

print(f'Long trades (optimal hold >= 200 bars): {len(long_trades)}')

# Analyze path of first 10 long trades in detail
for rank, (idx, t, r, best_bar) in enumerate(long_trades[:10]):
    path = t.get('path', [])
    direction = t.get('dir', '')
    entry_price = t.get('entry_price', 0)

    print(f'\n{"="*80}')
    print(f'TRADE #{rank+1} (idx={idx}, day={t.get("day","")}, dir={direction})')
    print(f'  Entry: ${entry_price:.2f}, Optimal hold: {best_bar} bars')
    print(f'  Actual PnL: ${r["actual_pnl"]:.1f}, Optimal PnL: ${r["best_pnl"]:.1f}')
    print(f'  Best action: {r["best_action"]}')
    print(f'{"="*80}')

    # Sample path at key points: entry, 25%, 50%, 75%, optimal exit, actual exit
    sample_bars = [0]
    for pct in [0.25, 0.50, 0.75]:
        sample_bars.append(int(best_bar * pct))
    sample_bars.append(best_bar)
    if int(r['actual_held']) < len(path):
        sample_bars.append(int(r['actual_held']))
    # Add every 60th bar for minute-by-minute
    sample_bars.extend(range(0, min(best_bar, len(path)), 60))
    sample_bars = sorted(set(b for b in sample_bars if b < len(path)))

    print(f'  {"Bar":>5} {"Min":>4} {"Price":>10} {"PnL":>8} '
          f'{"z_se":>6} {"p_ctr":>6} {"vel":>7} {"vr":>5} '
          f'{"wick":>5} {"rev_p":>6} {"5m_z":>6} {"5m_v":>7} {"Note":>10}')
    print(f'  {"-"*95}')

    for bar_idx in sample_bars:
        p = path[bar_idx]
        feat = np.array(p.get('features_79d', []))
        if len(feat) != len(FEATURE_NAMES_79D):
            continue

        price = p.get('price', 0)
        pnl = p.get('pnl', 0)
        minutes = bar_idx * 5 / 60  # 5s bars

        z = feat[FEAT_IDX['1m_z_se']]
        pc = feat[FEAT_IDX['1m_p_at_center']]
        vel = feat[FEAT_IDX['1m_velocity']]
        vr = feat[FEAT_IDX['1m_variance_ratio']]
        wick = feat[FEAT_IDX['1m_wick_ratio']]
        rev = feat[FEAT_IDX['1m_reversion_prob']]
        z5 = feat[FEAT_IDX['5m_z_se']]
        v5 = feat[FEAT_IDX['5m_velocity']]

        note = ''
        if bar_idx == 0:
            note = 'ENTRY'
        elif bar_idx == best_bar:
            note = 'OPT_EXIT'
        elif bar_idx == int(r['actual_held']):
            note = 'ACT_EXIT'
        elif abs(z) < 0.5:
            note = 'z<0.5'
        elif pc > 0.6:
            note = 'p>0.6'

        print(f'  {bar_idx:>5} {minutes:>4.1f} {price:>10.2f} ${pnl:>7.1f} '
              f'{z:>+6.2f} {pc:>6.3f} {vel:>+7.2f} {vr:>5.2f} '
              f'{wick:>5.2f} {rev:>6.3f} {z5:>+6.2f} {v5:>+7.2f} {note:>10}')

# Summary: what's happening at different stages of long trades
print(f'\n{"="*80}')
print(f'AGGREGATE: What happens during long trades ({len(long_trades)} trades)')
print(f'{"="*80}')

# Collect features at 25%, 50%, 75%, and optimal exit
stages = {'25%': [], '50%': [], '75%': [], 'exit': []}
for idx, t, r, best_bar in long_trades:
    path = t.get('path', [])
    for stage_name, stage_pct in [('25%', 0.25), ('50%', 0.50), ('75%', 0.75)]:
        bar = int(best_bar * stage_pct)
        if bar < len(path):
            feat = np.array(path[bar].get('features_79d', []))
            if len(feat) == len(FEATURE_NAMES_79D):
                stages[stage_name].append({
                    name: float(feat[FEAT_IDX[name]]) for name in TRACK
                })
    if best_bar < len(path):
        feat = np.array(path[best_bar].get('features_79d', []))
        if len(feat) == len(FEATURE_NAMES_79D):
            stages['exit'].append({
                name: float(feat[FEAT_IDX[name]]) for name in TRACK
            })

print(f'\n  {"Feature":<25} {"25%":>8} {"50%":>8} {"75%":>8} {"Exit":>8}')
print(f'  {"-"*60}')
for feat in TRACK:
    vals = []
    for stage in ['25%', '50%', '75%', 'exit']:
        if stages[stage]:
            mean = np.mean([s[feat] for s in stages[stage]])
            vals.append(f'{mean:>+8.3f}')
        else:
            vals.append(f'{"N/A":>8}')
    print(f'  {feat:<25} {" ".join(vals)}')

# Key question: do long trades show a pattern where z crosses zero then comes back?
print(f'\n  Z-CROSSING ANALYSIS (does z cross zero during trade?):')
n_cross = 0
n_multi_cross = 0
for idx, t, r, best_bar in long_trades:
    path = t.get('path', [])
    entry_z = np.array(t.get('entry_79d', []))
    if len(entry_z) != len(FEATURE_NAMES_79D):
        continue
    entry_z_sign = 1 if entry_z[FEAT_IDX['1m_z_se']] > 0 else -1

    crossings = 0
    prev_sign = entry_z_sign
    for bar in range(min(best_bar, len(path))):
        feat = np.array(path[bar].get('features_79d', []))
        if len(feat) != len(FEATURE_NAMES_79D):
            continue
        z = feat[FEAT_IDX['1m_z_se']]
        curr_sign = 1 if z > 0 else -1
        if curr_sign != prev_sign:
            crossings += 1
            prev_sign = curr_sign

    if crossings > 0:
        n_cross += 1
    if crossings > 1:
        n_multi_cross += 1

print(f'  {n_cross}/{len(long_trades)} ({n_cross/len(long_trades)*100:.0f}%) cross zero at least once')
print(f'  {n_multi_cross}/{len(long_trades)} ({n_multi_cross/len(long_trades)*100:.0f}%) cross zero multiple times (oscillation)')
