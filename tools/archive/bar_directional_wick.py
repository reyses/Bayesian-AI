"""Directional wick analysis: upper vs lower wick as continuation/reversal signal.

For GREEN bars:
  upper_wick = high - close  (rejection at top = sellers absorbing)
  lower_wick = open - low    (support from below = buyers stepping in)

For RED bars:
  upper_wick = high - open   (rejection at top = failed to hold)
  lower_wick = close - low   (support from below = buyers absorbing)

Wick ratio = wick / range (0 = no wick, 1 = all wick)
"""
import numpy as np, pandas as pd, sys, os, glob
from scipy import stats

tf = sys.argv[1] if len(sys.argv) > 1 else '1m'

files = sorted(glob.glob(f'DATA/ATLAS/{tf}/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files[-3:]], ignore_index=True)
df = df.sort_values('timestamp').reset_index(drop=True)

closes = df['close'].values
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
n = len(df)

green = closes > opens
red = closes < opens

# Compute directional wicks
upper_wick = np.zeros(n)
lower_wick = np.zeros(n)
upper_wick_ratio = np.zeros(n)
lower_wick_ratio = np.zeros(n)

for i in range(n):
    rng = highs[i] - lows[i]
    if rng <= 0:
        continue
    if green[i]:
        upper_wick[i] = highs[i] - closes[i]   # rejection at top
        lower_wick[i] = opens[i] - lows[i]      # support from below
    elif red[i]:
        upper_wick[i] = highs[i] - opens[i]     # failed to hold high
        lower_wick[i] = closes[i] - lows[i]     # absorption at bottom

    upper_wick_ratio[i] = upper_wick[i] / rng
    lower_wick_ratio[i] = lower_wick[i] / rng

print(f'DIRECTIONAL WICK vs CONTINUATION ({tf}, {n:,} bars)')
print()

# Green bars: upper wick ratio vs P(next green)
print(f'GREEN BARS: upper wick (rejection at top)')
print(f'{"Upper wick %":<16} {"P(next green)":<14} {"N":<10} {"Meaning":<20}')
print(f'{"="*60}')

buckets = [(0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 1.01)]
labels = ['NO wick', 'TINY wick', 'SMALL wick', 'MEDIUM wick', 'BIG wick', 'HUGE wick']

for (lo, hi), label in zip(buckets, labels):
    same = total = 0
    for i in range(n - 1):
        if not green[i]:
            continue
        if upper_wick_ratio[i] < lo or upper_wick_ratio[i] >= hi:
            continue
        total += 1
        if green[i+1]:
            same += 1
    if total < 30:
        continue
    p = same / total * 100
    print(f'{lo:.0%}-{hi:.0%} ({label:<10}) {p:>5.1f}%      {total:>8,}  {"MOMENTUM" if p > 55 else "BLOCKED" if p < 48 else ""}')

# Green bars: lower wick ratio vs P(next green)
print()
print(f'GREEN BARS: lower wick (support from below)')
print(f'{"Lower wick %":<16} {"P(next green)":<14} {"N":<10} {"Meaning":<20}')
print(f'{"="*60}')

for (lo, hi), label in zip(buckets, labels):
    same = total = 0
    for i in range(n - 1):
        if not green[i]:
            continue
        if lower_wick_ratio[i] < lo or lower_wick_ratio[i] >= hi:
            continue
        total += 1
        if green[i+1]:
            same += 1
    if total < 30:
        continue
    p = same / total * 100
    print(f'{lo:.0%}-{hi:.0%} ({label:<10}) {p:>5.1f}%      {total:>8,}  {"STRONG SUPPORT" if p > 55 else "WEAK" if p < 48 else ""}')

# RED bars: same analysis
print()
print(f'RED BARS: lower wick (absorption at bottom)')
print(f'{"Lower wick %":<16} {"P(next red)":<14} {"N":<10} {"Meaning":<20}')
print(f'{"="*60}')

for (lo, hi), label in zip(buckets, labels):
    same = total = 0
    for i in range(n - 1):
        if not red[i]:
            continue
        if lower_wick_ratio[i] < lo or lower_wick_ratio[i] >= hi:
            continue
        total += 1
        if red[i+1]:
            same += 1
    if total < 30:
        continue
    p = same / total * 100
    print(f'{lo:.0%}-{hi:.0%} ({label:<10}) {p:>5.1f}%      {total:>8,}  {"MOMENTUM" if p > 55 else "ABSORBED" if p < 48 else ""}')

print()
print(f'RED BARS: upper wick (failed to hold high)')
print(f'{"Upper wick %":<16} {"P(next red)":<14} {"N":<10} {"Meaning":<20}')
print(f'{"="*60}')

for (lo, hi), label in zip(buckets, labels):
    same = total = 0
    for i in range(n - 1):
        if not red[i]:
            continue
        if upper_wick_ratio[i] < lo or upper_wick_ratio[i] >= hi:
            continue
        total += 1
        if red[i+1]:
            same += 1
    if total < 30:
        continue
    p = same / total * 100
    print(f'{lo:.0%}-{hi:.0%} ({label:<10}) {p:>5.1f}%      {total:>8,}')

# Combined: streak + directional wick
print()
print(f'COMBINED: 3+ streak + directional wick')
print(f'{"Streak":<8} {"Wick type":<20} {"P(continue)":<14} {"N":<10}')
print(f'{"="*55}')

for min_streak in [3, 5]:
    for wick_label, wick_lo, wick_hi, wick_type in [
        ('no upper wick', 0, 0.05, 'upper'),
        ('small upper', 0.05, 0.25, 'upper'),
        ('big upper', 0.25, 1.01, 'upper'),
        ('no lower wick', 0, 0.05, 'lower'),
        ('small lower', 0.05, 0.25, 'lower'),
        ('big lower', 0.25, 1.01, 'lower'),
    ]:
        same = total = 0
        for i in range(min_streak, n - 1):
            if not (green[i] or red[i]):
                continue

            streak_color = green[i]
            has_streak = all(
                (green[i-j] if streak_color else red[i-j])
                for j in range(min_streak)
            )
            if not has_streak:
                continue

            wr = upper_wick_ratio[i] if wick_type == 'upper' else lower_wick_ratio[i]
            if wr < wick_lo or wr >= wick_hi:
                continue

            total += 1
            is_same = (green[i] and green[i+1]) or (red[i] and red[i+1])
            if is_same:
                same += 1

        if total < 30:
            continue
        p = same / total * 100
        marker = ' <-- SIGNAL' if p > 60 or p < 45 else ''
        print(f'{min_streak}+     {wick_label:<20} {p:>5.1f}%      {total:>8,}{marker}')
    print()
