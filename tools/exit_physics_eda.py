"""
Exit Physics EDA — What does the 79D look like at regret's optimal exit?

For each pattern, extract 79D at the optimal exit bar and find common conditions.
Goal: readable exit rules per pattern, not CNN black box.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features import FEATURE_NAMES, TF_ORDER

# Load data
print('Loading data...')
with open('training/output/trades/blended_is.pkl', 'rb') as f:
    trades = pickle.load(f)
regret = pd.read_csv('training/output/nn/regret_analysis.csv')

# Load pattern assignments
patterns = {}
for tier in ['cascade', 'kill_shot', 'base_nmp']:
    path = f'training/output/entry/patterns_{tier}.pkl'
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for d in data:
            patterns[d['trade_idx']] = d
    except FileNotFoundError:
        pass

n = min(len(trades), len(regret))
print(f'Trades: {n}, Patterns: {len(patterns)}')

# Key 79D feature indices
FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Features we care about for exit analysis
EXIT_FEATURES = [
    '1m_z_se', '1m_variance_ratio', '1m_velocity', '1m_p_at_center',
    '1m_reversion_prob', '1m_bar_range', '1m_wick_ratio',
    '5m_z_se', '5m_velocity', '5m_wick_ratio',
    '15m_z_se', '15m_velocity',
    '1h_z_se',
]

# Collect exit 79D per pattern
print('\nExtracting exit conditions...')
pattern_exits = defaultdict(list)
pattern_entries = defaultdict(list)

for i in range(n):
    t = trades[i]
    r = regret.iloc[i]
    label = patterns[i]['pattern_label'] if i in patterns else ''
    if not label:
        continue

    # Get the trade path
    path = t.get('path', [])
    if not path:
        continue

    # Regret's optimal exit bar (relative to entry)
    best_action = r['best_action']
    if 'same' in best_action:
        best_bar = int(r['same_best_bar'])
    else:
        best_bar = int(r['counter_best_bar'])

    # Find 79D at optimal exit bar
    if best_bar < len(path) and 'features' in path[best_bar]:
        exit_79d = np.array(path[best_bar]['features'])
        if len(exit_79d) == len(FEATURE_NAMES):
            exit_feats = {name: float(exit_79d[FEAT_IDX[name]]) for name in EXIT_FEATURES}
            exit_feats['bars_held'] = best_bar
            exit_feats['pnl'] = float(r['best_pnl'])
            exit_feats['actual_pnl'] = float(r['actual_pnl'])
            pattern_exits[label].append(exit_feats)

    # Also get entry 79D for comparison
    entry_79d = np.array(t.get('entry_79d', []))
    if len(entry_79d) == len(FEATURE_NAMES):
        entry_feats = {name: float(entry_79d[FEAT_IDX[name]]) for name in EXIT_FEATURES}
        pattern_entries[label].append(entry_feats)

# Report
print(f'\n{"="*90}')
print(f'EXIT PHYSICS BY PATTERN')
print(f'{"="*90}')

for label in sorted(pattern_exits.keys()):
    exits = pattern_exits[label]
    entries = pattern_entries.get(label, [])
    n_trades = len(exits)

    if n_trades < 5:
        continue

    print(f'\n{"="*90}')
    print(f'{label} ({n_trades} trades)')
    print(f'{"="*90}')

    df_exit = pd.DataFrame(exits)
    df_entry = pd.DataFrame(entries[:n_trades])

    # Summary stats
    print(f'  Avg optimal hold: {df_exit["bars_held"].mean():.0f} bars '
          f'(mode={df_exit["bars_held"].mode().iloc[0]:.0f})')
    print(f'  Avg optimal PnL: ${df_exit["pnl"].mean():.1f}')
    print(f'  Avg actual PnL:  ${df_exit["actual_pnl"].mean():.1f}')

    # Entry vs Exit comparison for key features
    print(f'\n  {"Feature":<25} {"Entry Mean":>10} {"Exit Mean":>10} {"Delta":>10} {"Exit P25":>10} {"Exit P75":>10}')
    print(f'  {"-"*75}')

    for feat in EXIT_FEATURES:
        if feat not in df_exit.columns:
            continue
        entry_mean = df_entry[feat].mean() if feat in df_entry.columns else 0
        exit_mean = df_exit[feat].mean()
        delta = exit_mean - entry_mean
        p25 = df_exit[feat].quantile(0.25)
        p75 = df_exit[feat].quantile(0.75)
        # Flag features with big delta
        flag = ' <<<' if abs(delta) > abs(entry_mean) * 0.3 and abs(delta) > 0.1 else ''
        print(f'  {feat:<25} {entry_mean:>10.3f} {exit_mean:>10.3f} {delta:>+10.3f} '
              f'{p25:>10.3f} {p75:>10.3f}{flag}')

    # Exit z_se distribution — where does price sit relative to bands?
    z_exit = df_exit['1m_z_se']
    print(f'\n  z_se at exit: mean={z_exit.mean():.2f}, '
          f'|z|<0.5={((z_exit.abs() < 0.5).sum()/n_trades*100):.0f}%, '
          f'|z|<1.0={((z_exit.abs() < 1.0).sum()/n_trades*100):.0f}%, '
          f'z_flipped={(((z_exit * df_entry["1m_z_se"] if "1m_z_se" in df_entry else z_exit) < 0).sum()/n_trades*100):.0f}%')

    # p_center at exit
    pc_exit = df_exit['1m_p_at_center']
    print(f'  p_center at exit: mean={pc_exit.mean():.2f}, '
          f'>0.5={((pc_exit > 0.5).sum()/n_trades*100):.0f}%, '
          f'>0.6={((pc_exit > 0.6).sum()/n_trades*100):.0f}%, '
          f'>0.7={((pc_exit > 0.7).sum()/n_trades*100):.0f}%')

    # Velocity at exit — momentum exhausted?
    vel_exit = df_exit['1m_velocity']
    print(f'  velocity at exit: mean={vel_exit.mean():.3f}, '
          f'|v|<0.3={((vel_exit.abs() < 0.3).sum()/n_trades*100):.0f}%, '
          f'|v|<0.5={((vel_exit.abs() < 0.5).sum()/n_trades*100):.0f}%')

# Summary: common exit conditions across all BASE_NMP patterns
print(f'\n{"="*90}')
print(f'BASE_NMP EXIT CONDITIONS SUMMARY')
print(f'{"="*90}')

all_base = []
for label, exits in pattern_exits.items():
    if label.startswith('BASE_NMP'):
        all_base.extend(exits)

if all_base:
    df = pd.DataFrame(all_base)
    print(f'Total BASE_NMP exits: {len(df)}')
    print(f'  Avg hold bars: {df["bars_held"].mean():.0f} (mode={df["bars_held"].mode().iloc[0]:.0f})')
    print(f'  z_se at exit: mean={df["1m_z_se"].mean():.2f}')
    print(f'  p_center > 0.5: {(df["1m_p_at_center"] > 0.5).sum()/len(df)*100:.0f}%')
    print(f'  p_center > 0.6: {(df["1m_p_at_center"] > 0.6).sum()/len(df)*100:.0f}%')
    print(f'  |velocity| < 0.3: {(df["1m_velocity"].abs() < 0.3).sum()/len(df)*100:.0f}%')
    print(f'  |z| < 0.5: {(df["1m_z_se"].abs() < 0.5).sum()/len(df)*100:.0f}%')
    print(f'  |z| < 1.0: {(df["1m_z_se"].abs() < 1.0).sum()/len(df)*100:.0f}%')
    print(f'  variance_ratio > 1.0: {(df["1m_variance_ratio"] > 1.0).sum()/len(df)*100:.0f}%')

    # What % of optimal exits happen at z_sign flip?
    # Can't directly check without entry z, so use bars_held distribution
    print(f'\n  Hold bars distribution:')
    for pct in [10, 25, 50, 75, 90]:
        print(f'    P{pct}: {df["bars_held"].quantile(pct/100):.0f} bars')

# Save full analysis
os.makedirs('reports/findings', exist_ok=True)
results = []
for label in sorted(pattern_exits.keys()):
    for e in pattern_exits[label]:
        e['pattern'] = label
        results.append(e)
import os
pd.DataFrame(results).to_csv('reports/findings/exit_physics_eda.csv', index=False)
print(f'\nSaved: reports/findings/exit_physics_eda.csv')
