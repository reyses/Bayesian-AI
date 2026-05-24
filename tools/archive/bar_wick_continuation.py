"""Body-to-wick ratio vs continuation probability.
Does a bar with a big body (conviction) continue more than a bar with big wick (rejection)?"""
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

print(f'BODY/WICK RATIO vs CONTINUATION ({tf}, {n:,} bars)')
print()

# For each bar: compute body ratio = body / range (0=all wick, 1=all body)
body_ratio = np.zeros(n)
for i in range(n):
    rng = highs[i] - lows[i]
    if rng > 0:
        body_ratio[i] = abs(closes[i] - opens[i]) / rng
    else:
        body_ratio[i] = 0.5  # flat bar

# Bucket body_ratio and measure P(same color next bar)
print(f'{"Body ratio":<14} {"P(continue)":<14} {"N":<10} {"Avg body":<12} {"Avg next body":<14}')
print(f'{"="*65}')

buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

br_vals = []
cont_vals = []

for lo, hi in buckets:
    same = 0
    total = 0
    bodies = []
    next_bodies = []

    for i in range(n - 1):
        if not (green[i] or red[i]):
            continue
        if body_ratio[i] < lo or body_ratio[i] >= hi:
            continue

        total += 1
        bodies.append(abs(closes[i] - opens[i]))
        next_bodies.append(abs(closes[i+1] - opens[i+1]))

        is_same = (green[i] and green[i+1]) or (red[i] and red[i+1])
        if is_same:
            same += 1

    if total < 50:
        continue

    p_cont = same / total * 100
    br_vals.append((lo + hi) / 2)
    cont_vals.append(p_cont)

    avg_b = np.mean(bodies)
    avg_nb = np.mean(next_bodies)
    bar = '#' * int(p_cont)

    label = 'ALL WICK' if hi <= 0.2 else 'MOSTLY WICK' if hi <= 0.4 else 'MIXED' if hi <= 0.6 else 'MOSTLY BODY' if hi <= 0.8 else 'ALL BODY'

    print(f'{lo:.0%}-{hi:.0%} ({label:<11}) {p_cont:>5.1f}%  {total:>8,}  {avg_b:>8.2f}pts  {avg_nb:>10.2f}pts  {bar}')

# Regression: body_ratio -> continuation
br_all = []
cont_all = []
for i in range(n - 1):
    if not (green[i] or red[i]):
        continue
    br_all.append(body_ratio[i])
    is_same = (green[i] and green[i+1]) or (red[i] and red[i+1])
    cont_all.append(1.0 if is_same else 0.0)

br_all = np.array(br_all)
cont_all = np.array(cont_all)

slope, intercept, r, p_val, se = stats.linregress(br_all, cont_all)
print(f'\nRegression: body_ratio -> P(continue)')
print(f'  R2={r**2:.4f} ({r**2*100:.3f}%)  slope={slope:.4f}  p={p_val:.2e}')
print(f'  At body_ratio=0.0 (all wick): P(cont) = {intercept*100:.1f}%')
print(f'  At body_ratio=1.0 (all body): P(cont) = {(intercept+slope)*100:.1f}%')

# Now combine: streak + body ratio
print(f'\n\nCOMBINED: streak length + body ratio')
print(f'{"Streak":<8} {"Body ratio":<14} {"P(continue)":<14} {"N":<10}')
print(f'{"="*50}')

for min_streak in [1, 3, 5]:
    for body_cat, lo, hi in [('WICK', 0, 0.3), ('MIXED', 0.3, 0.6), ('BODY', 0.6, 1.01)]:
        same = 0
        total = 0

        for i in range(min_streak, n - 1):
            if not (green[i] or red[i]):
                continue

            # Check streak: last min_streak bars same color
            streak_color = green[i]
            has_streak = all(
                (green[i-j] if streak_color else red[i-j])
                for j in range(min_streak)
            )
            if not has_streak:
                continue

            if body_ratio[i] < lo or body_ratio[i] >= hi:
                continue

            total += 1
            is_same = (green[i] and green[i+1]) or (red[i] and red[i+1])
            if is_same:
                same += 1

        if total < 30:
            continue

        p_cont = same / total * 100
        print(f'{min_streak}+     {body_cat:<14} {p_cont:>5.1f}%      {total:>8,}')
