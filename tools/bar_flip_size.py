"""When bar color flips, how big is the flip bar vs the continuation bars?"""
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
TICK = 0.25

print(f'BAR FLIP SIZE ANALYSIS ({tf}, {n:,} bars)')
print()

# For streaks of same color: measure the flip bar vs the streak bars
for min_streak in [1, 2, 3, 5]:
    cont_bodies = []  # body sizes during streak
    flip_bodies = []  # body size of the bar that breaks the streak
    flip_as_pct_of_avg = []  # flip body / avg streak body

    i = 1
    while i < n - 1:
        # Find start of a streak
        if not (green[i] or red[i]):
            i += 1
            continue

        streak_color = green[i]
        streak_start = i
        streak_bodies = [abs(closes[i] - opens[i])]

        j = i + 1
        while j < n and ((green[j] if streak_color else red[j])):
            streak_bodies.append(abs(closes[j] - opens[j]))
            j += 1

        streak_len = j - streak_start

        if streak_len >= min_streak and j < n and (green[j] or red[j]):
            # j is the flip bar
            flip_body = abs(closes[j] - opens[j])
            avg_streak_body = np.mean(streak_bodies)

            cont_bodies.extend(streak_bodies)
            flip_bodies.append(flip_body)
            if avg_streak_body > 0:
                flip_as_pct_of_avg.append(flip_body / avg_streak_body)

        i = j

    if not flip_bodies:
        continue

    cb = np.array(cont_bodies)
    fb = np.array(flip_bodies)
    fp = np.array(flip_as_pct_of_avg)

    t_stat, p_val = stats.ttest_ind(fb, cb)

    print(f'After {min_streak}+ bar streak:')
    print(f'  Streak bar body:  mean={cb.mean():.2f}  median={np.median(cb):.2f}  (N={len(cb):,})')
    print(f'  Flip bar body:    mean={fb.mean():.2f}  median={np.median(fb):.2f}  (N={len(fb):,})')
    print(f'  Ratio (flip/streak): {fb.mean()/cb.mean():.2f}x')
    print(f'  t-test: t={t_stat:.2f}, p={p_val:.2e}')
    print()

    # Distribution of flip size relative to streak average
    print(f'  Flip bar as % of avg streak bar:')
    pcts = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for p in pcts:
        below = (fp <= p).mean() * 100
        print(f'    <= {p:.0%} of streak avg: {below:.1f}% of flips')

    # How many flips are tiny (less than half the streak bar)?
    tiny_flips = (fp < 0.5).mean() * 100
    big_flips = (fp > 1.5).mean() * 100
    print(f'  Tiny flips (<50% of streak): {tiny_flips:.1f}%')
    print(f'  Big flips (>150% of streak): {big_flips:.1f}%')
    print()
