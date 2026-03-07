"""When do counter-trend scalps happen? Are they wave riding within a trend?"""
import pandas as pd, json, numpy as np

df = pd.read_csv('checkpoints/oos_trade_log.csv')
df['dir_correct'] = ((df['direction']=='LONG')&(df['oracle_label']>0))|((df['direction']=='SHORT')&(df['oracle_label']<0))
df['is_scalp'] = (~df['dir_correct']) & (df['oracle_label']!=0) & (df['actual_pnl']>0)
df['hold_mins'] = df['hold_bars'] * 15 / 60
df['entry_dt'] = pd.to_datetime(df['entry_time'], unit='s')
df['exit_dt'] = pd.to_datetime(df['exit_time'], unit='s')

scalps = df[df['is_scalp']].copy()
correct = df[df['dir_correct']].copy()

print(f"Total scalps: {len(scalps)}")
print()

# 1. Time clustering: do scalps happen in bursts (wave tops/bottoms)?
print("=== SCALP TIME CLUSTERING ===")
scalps_sorted = scalps.sort_values('entry_time')
time_diffs = scalps_sorted['entry_time'].diff().dropna()
print(f"  Median time between scalps: {time_diffs.median()/60:.1f} min")
print(f"  Mean time between scalps:   {time_diffs.mean()/60:.1f} min")
print(f"  Scalps within 2 min of each other: {(time_diffs < 120).sum()}")
print(f"  Scalps within 5 min of each other: {(time_diffs < 300).sum()}")
print()

# 2. Wave maturity at entry — are scalps at wave peaks?
print("=== WAVE MATURITY AT ENTRY ===")
print(f"  Scalps  wave_maturity:  {scalps['wave_maturity'].mean():.4f}")
print(f"  Correct wave_maturity:  {correct['wave_maturity'].mean():.4f}")
print(f"  Scalps  decision_wave:  {scalps['decision_wave_maturity'].mean():.4f}")
print(f"  Correct decision_wave:  {correct['decision_wave_maturity'].mean():.4f}")
print()

# 3. Do scalps happen right after correct-direction trades? (wave alternation)
print("=== TRADE SEQUENCE: Do scalps follow correct-dir trades? ===")
df_sorted = df.sort_values('entry_time').reset_index(drop=True)
prev_class = []
for i in range(len(df_sorted)):
    if i == 0:
        prev_class.append('none')
    else:
        prev = df_sorted.iloc[i-1]
        if prev['dir_correct']:
            prev_class.append('correct')
        elif prev['is_scalp']:
            prev_class.append('scalp')
        else:
            prev_class.append('wrong_loss')
df_sorted['prev_class'] = prev_class

scalp_rows = df_sorted[df_sorted['is_scalp']]
print(f"  Previous trade was correct dir:  {(scalp_rows['prev_class']=='correct').sum()} ({(scalp_rows['prev_class']=='correct').mean()*100:.1f}%)")
print(f"  Previous trade was scalp:        {(scalp_rows['prev_class']=='scalp').sum()} ({(scalp_rows['prev_class']=='scalp').mean()*100:.1f}%)")
print(f"  Previous trade was wrong/loss:   {(scalp_rows['prev_class']=='wrong_loss').sum()} ({(scalp_rows['prev_class']=='wrong_loss').mean()*100:.1f}%)")
print()

# 4. Time since last correct-dir trade
print("=== TIME GAP FROM LAST CORRECT-DIR TRADE ===")
correct_times = set(correct['exit_time'].values)
gaps = []
for _, row in scalps.iterrows():
    prev_correct = df_sorted[(df_sorted['dir_correct']) & (df_sorted['exit_time'] < row['entry_time'])]
    if len(prev_correct) > 0:
        gap = row['entry_time'] - prev_correct.iloc[-1]['exit_time']
        gaps.append(gap / 60)  # minutes
if gaps:
    gaps = np.array(gaps)
    print(f"  Median gap: {np.median(gaps):.1f} min")
    print(f"  Mean gap:   {np.mean(gaps):.1f} min")
    print(f"  <1 min:     {(gaps < 1).sum()} ({(gaps < 1).mean()*100:.1f}%)")
    print(f"  <5 min:     {(gaps < 5).sum()} ({(gaps < 5).mean()*100:.1f}%)")
    print(f"  <15 min:    {(gaps < 15).sum()} ({(gaps < 15).mean()*100:.1f}%)")
print()

# 5. MFE analysis — how much move was there in the scalp direction?
print("=== SCALP MFE: How real is the counter-move? ===")
print(f"  Avg trade MFE (ticks in our favor): {scalps['trade_mfe_ticks'].mean():.1f}")
print(f"  Avg actual PnL:                     {scalps['actual_pnl'].mean():.2f}")
print(f"  Avg capture (PnL/MFE):              {(scalps['actual_pnl'] / (scalps['trade_mfe_ticks']*0.5+0.001)).mean():.1%}")
print(f"  Avg hold time:                      {scalps['hold_mins'].mean():.1f} min")
print()

# Bucket by MFE size
print("=== SCALP MFE DISTRIBUTION (ticks) ===")
for lo, hi in [(0,8), (8,16), (16,32), (32,64), (64,128), (128,999)]:
    sub = scalps[(scalps['trade_mfe_ticks'] >= lo) & (scalps['trade_mfe_ticks'] < hi)]
    if len(sub) > 0:
        print(f"  MFE {lo:>3d}-{hi:>3d}t: {len(sub):4d} trades, avg PnL {sub['actual_pnl'].mean():.2f}, avg hold {sub['hold_mins'].mean():.1f}m")
print()

# 6. Physics state — are scalps at extremes (Roche limit)?
print("=== PHYSICS AT ENTRY: Scalps vs Correct ===")
for col in ['velocity', 'sigma', 'hurst', 'tunnel_prob', 'F_momentum', 'F_reversion', 'mom_rev_ratio']:
    if col in scalps.columns:
        s_mean = scalps[col].mean()
        c_mean = correct[col].mean()
        print(f"  {col:20s}  scalps={s_mean:10.3f}  correct={c_mean:10.3f}  delta={c_mean-s_mean:+10.3f}")
print()

# 7. Session distribution
print("=== SCALPS BY HOUR (ET) ===")
scalps['hour'] = scalps['entry_dt'].dt.hour
hourly = scalps.groupby('hour').agg(n=('actual_pnl','count'), avg_pnl=('actual_pnl','mean')).reset_index()
correct['hour'] = correct['entry_dt'].dt.hour
hourly_c = correct.groupby('hour').agg(n_c=('actual_pnl','count')).reset_index()
merged = hourly.merge(hourly_c, on='hour', how='left')
merged['scalp_ratio'] = merged['n'] / (merged['n'] + merged['n_c'].fillna(0))
print(f"  {'Hour':>4s}  {'Scalps':>6s}  {'Correct':>7s}  {'Scalp%':>7s}  {'AvgPnL':>8s}")
for _, r in merged.iterrows():
    print(f"  {int(r['hour']):4d}  {int(r['n']):6d}  {int(r.get('n_c',0)):7d}  {r['scalp_ratio']:6.1%}  {r['avg_pnl']:8.2f}")
