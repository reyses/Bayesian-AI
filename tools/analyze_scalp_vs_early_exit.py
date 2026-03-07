"""Are 'too early' correct-dir exits and counter-trend scalps happening in the same time windows?
If so: we're catching half the wave as a correct trade (exiting too early) and the other half
as a counter-trend scalp (entering wrong side of the same micro-pullback)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, json

df = pd.read_csv('checkpoints/oos_trade_log.csv')

# Classify trades
df['dir_correct'] = ((df['direction']=='LONG')&(df['oracle_label']>0))|((df['direction']=='SHORT')&(df['oracle_label']<0))
df['is_scalp'] = (~df['dir_correct']) & (df['oracle_label']!=0) & (df['actual_pnl']>0)
df['entry_dt'] = pd.to_datetime(df['entry_time'], unit='s')
df['exit_dt'] = pd.to_datetime(df['exit_time'], unit='s')

# Exit quality buckets (from report logic)
correct = df[df['dir_correct']].copy()
correct['oracle_mfe'] = correct['oracle_label'].abs()
correct['capture_pct'] = correct['actual_pnl'] / (correct['oracle_mfe'] * 0.5 + 0.001)
correct['is_too_early'] = correct['capture_pct'] < 0.20  # <20% captured = "too early"
correct['is_reversed'] = correct['actual_pnl'] < 0  # lost money on correct dir = reversed

scalps = df[df['is_scalp']].copy()
too_early = correct[correct['is_too_early']].copy()
reversed_trades = correct[correct['is_reversed']].copy()

print(f"Correct-dir trades: {len(correct)}")
print(f"  Too early exits:  {len(too_early)} ({len(too_early)/len(correct)*100:.1f}%)")
print(f"  Reversed:         {len(reversed_trades)} ({len(reversed_trades)/len(correct)*100:.1f}%)")
print(f"Counter-trend scalps: {len(scalps)}")
print()

# === 1. TIME PROXIMITY: Do scalps happen near too-early exits? ===
print("=" * 70)
print("1. TIME PROXIMITY: Scalps near too-early correct-dir exits")
print("=" * 70)

# For each scalp, find nearest too-early exit
proximity_results = []
for _, scalp in scalps.iterrows():
    s_entry = scalp['entry_time']
    # Find too-early exits within ±10 min
    near = too_early[(too_early['exit_time'] > s_entry - 600) &
                     (too_early['exit_time'] < s_entry + 600)]
    if len(near) > 0:
        closest = near.loc[(near['exit_time'] - s_entry).abs().idxmin()]
        gap = s_entry - closest['exit_time']
        proximity_results.append({
            'scalp_dir': scalp['direction'],
            'scalp_pnl': scalp['actual_pnl'],
            'early_dir': closest['direction'],
            'early_pnl': closest['actual_pnl'],
            'gap_sec': gap,
            'same_dir': scalp['direction'] == closest['direction'],
            'opposite_dir': scalp['direction'] != closest['direction'],
            'scalp_after_exit': gap > 0,
        })

prox = pd.DataFrame(proximity_results)
if len(prox) > 0:
    print(f"  Scalps within 10min of a too-early exit: {len(prox)} / {len(scalps)} ({len(prox)/len(scalps)*100:.1f}%)")
    print(f"  Scalp AFTER too-early exit:  {prox['scalp_after_exit'].sum()} ({prox['scalp_after_exit'].mean()*100:.1f}%)")
    print(f"  Opposite direction (wave flip): {prox['opposite_dir'].sum()} ({prox['opposite_dir'].mean()*100:.1f}%)")
    print(f"  Same direction (re-entry):      {prox[prox['same_dir']].shape[0]} ({prox['same_dir'].mean()*100:.1f}%)")

    # The KEY pattern: scalp is opposite dir AND happens after the early exit
    wave_flip = prox[(prox['opposite_dir']) & (prox['scalp_after_exit'])]
    print(f"\n  WAVE FLIP pattern (opposite dir, scalp after exit): {len(wave_flip)}")
    if len(wave_flip) > 0:
        print(f"    Avg gap: {wave_flip['gap_sec'].mean():.0f}s ({wave_flip['gap_sec'].mean()/60:.1f}min)")
        print(f"    Avg scalp PnL: ${wave_flip['scalp_pnl'].mean():.2f}")
        print(f"    Avg early-exit PnL: ${wave_flip['early_pnl'].mean():.2f}")
        print(f"    Combined avg: ${(wave_flip['scalp_pnl'] + wave_flip['early_pnl']).mean():.2f}")
print()

# === 2. DIRECTION PATTERN: Are scalps the opposite side of too-early exits? ===
print("=" * 70)
print("2. DIRECTION SEQUENCE: What happens after too-early exits?")
print("=" * 70)

df_sorted = df.sort_values('entry_time').reset_index(drop=True)
for i, row in df_sorted.iterrows():
    df_sorted.at[i, 'is_too_early'] = row['dir_correct'] and (row['actual_pnl'] / (abs(row['oracle_label']) * 0.5 + 0.001) < 0.20 if row['oracle_label'] != 0 else False)

# For each too-early exit, what's the NEXT trade?
too_early_idx = df_sorted[df_sorted['is_too_early']].index
next_trades = []
for idx in too_early_idx:
    if idx + 1 < len(df_sorted):
        te = df_sorted.iloc[idx]
        nt = df_sorted.iloc[idx + 1]
        gap = nt['entry_time'] - te['exit_time']
        next_trades.append({
            'early_dir': te['direction'],
            'next_dir': nt['direction'],
            'next_is_scalp': bool(nt['is_scalp']),
            'next_is_correct': bool(nt['dir_correct']),
            'next_pnl': nt['actual_pnl'],
            'gap_sec': gap,
            'flipped_dir': te['direction'] != nt['direction'],
        })

nt_df = pd.DataFrame(next_trades)
if len(nt_df) > 0:
    print(f"  Too-early exits: {len(nt_df)}")
    print(f"  Next trade is scalp:   {nt_df['next_is_scalp'].sum()} ({nt_df['next_is_scalp'].mean()*100:.1f}%)")
    print(f"  Next trade is correct: {nt_df['next_is_correct'].sum()} ({nt_df['next_is_correct'].mean()*100:.1f}%)")
    print(f"  Next trade flips dir:  {nt_df['flipped_dir'].sum()} ({nt_df['flipped_dir'].mean()*100:.1f}%)")

    # Too-early -> scalp -> what's the combined story?
    te_then_scalp = nt_df[nt_df['next_is_scalp']]
    if len(te_then_scalp) > 0:
        print(f"\n  Too-early -> Scalp chains ({len(te_then_scalp)}):")
        print(f"    Flipped direction: {te_then_scalp['flipped_dir'].sum()} ({te_then_scalp['flipped_dir'].mean()*100:.1f}%)")
        print(f"    Avg gap: {te_then_scalp['gap_sec'].mean():.0f}s ({te_then_scalp['gap_sec'].mean()/60:.1f}min)")
        print(f"    Avg scalp PnL: ${te_then_scalp['next_pnl'].mean():.2f}")
print()

# === 3. HOURLY OVERLAP: Do scalps and too-early exits cluster in same hours? ===
print("=" * 70)
print("3. HOURLY OVERLAP: Scalps vs Too-Early exits by hour")
print("=" * 70)

scalps['hour'] = scalps['entry_dt'].dt.hour
too_early['hour'] = too_early['entry_dt'].dt.hour

h_scalp = scalps.groupby('hour').size().rename('scalps')
h_early = too_early.groupby('hour').size().rename('too_early')
h_all = correct.groupby(correct['entry_dt'].dt.hour).size().rename('all_correct')

hourly = pd.concat([h_scalp, h_early, h_all], axis=1).fillna(0).astype(int)
hourly['scalp_rate'] = hourly['scalps'] / (hourly['scalps'] + hourly['all_correct'] + 0.001)
hourly['early_rate'] = hourly['too_early'] / (hourly['all_correct'] + 0.001)

print(f"  {'Hour':>4s}  {'Scalps':>6s}  {'TooEarly':>8s}  {'Correct':>7s}  {'Scalp%':>7s}  {'Early%':>7s}  {'Correlation?':>12s}")
for h, r in hourly.iterrows():
    flag = " <-- BOTH HIGH" if r['scalp_rate'] > 0.3 and r['early_rate'] > 0.15 else ""
    print(f"  {h:4d}  {int(r['scalps']):6d}  {int(r['too_early']):8d}  {int(r['all_correct']):7d}  {r['scalp_rate']:6.1%}  {r['early_rate']:6.1%}{flag}")

# Correlation
from scipy.stats import spearmanr
if len(hourly) > 3:
    corr, pval = spearmanr(hourly['scalps'], hourly['too_early'])
    print(f"\n  Spearman correlation (scalps vs too-early by hour): r={corr:.3f}, p={pval:.3f}")
print()

# === 4. SAME ORACLE MOVE: Are they fighting the same underlying move? ===
print("=" * 70)
print("4. ORACLE LABEL OVERLAP: Same underlying move?")
print("=" * 70)

# If scalp and too-early exit are near each other AND have same oracle_label_name,
# they're literally part of the same wave
print("\n  Too-early exits by oracle label:")
for label in sorted(too_early['oracle_label_name'].unique()):
    sub = too_early[too_early['oracle_label_name'] == label]
    print(f"    {label:20s}: {len(sub)} trades, avg captured ${sub['actual_pnl'].mean():.2f}")

print("\n  Scalps by oracle label (what they're fighting):")
for label in sorted(scalps['oracle_label_name'].unique()):
    sub = scalps[scalps['oracle_label_name'] == label]
    print(f"    {label:20s}: {len(sub)} trades, avg PnL ${sub['actual_pnl'].mean():.2f}")

# === 5. THE MONEY QUESTION: Combined PnL of wave pairs ===
print()
print("=" * 70)
print("5. WAVE PAIR PnL: If we held instead of flipping, would we make more?")
print("=" * 70)

# Find pairs: too-early exit followed by scalp within 5min, opposite direction
pairs = []
for idx in too_early_idx:
    if idx + 1 >= len(df_sorted):
        continue
    te = df_sorted.iloc[idx]
    nt = df_sorted.iloc[idx + 1]
    gap = nt['entry_time'] - te['exit_time']
    if gap < 300 and gap > 0 and nt['is_scalp'] and te['direction'] != nt['direction']:
        pairs.append({
            'early_dir': te['direction'],
            'early_pnl': te['actual_pnl'],
            'early_oracle': te['oracle_label_name'],
            'scalp_dir': nt['direction'],
            'scalp_pnl': nt['actual_pnl'],
            'combined_pnl': te['actual_pnl'] + nt['actual_pnl'],
            'gap_sec': gap,
            'early_hold': te['hold_bars'],
            'scalp_hold': nt['hold_bars'],
        })

pairs_df = pd.DataFrame(pairs)
if len(pairs_df) > 0:
    print(f"  Wave pairs found (too-early -> opposite scalp within 5min): {len(pairs_df)}")
    print(f"  Avg early-exit PnL:  ${pairs_df['early_pnl'].mean():.2f}")
    print(f"  Avg scalp PnL:       ${pairs_df['scalp_pnl'].mean():.2f}")
    print(f"  Avg combined PnL:    ${pairs_df['combined_pnl'].mean():.2f}")
    print(f"  Total early PnL:     ${pairs_df['early_pnl'].sum():.2f}")
    print(f"  Total scalp PnL:     ${pairs_df['scalp_pnl'].sum():.2f}")
    print(f"  Total combined PnL:  ${pairs_df['combined_pnl'].sum():.2f}")
    print(f"  Avg gap between:     {pairs_df['gap_sec'].mean():.0f}s")
    print(f"  Avg early hold:      {pairs_df['early_hold'].mean():.1f} bars")
    print(f"  Avg scalp hold:      {pairs_df['scalp_hold'].mean():.1f} bars")

    print(f"\n  Direction pattern:")
    for d in ['LONG', 'SHORT']:
        sub = pairs_df[pairs_df['early_dir'] == d]
        if len(sub) > 0:
            print(f"    {d} exit-early -> {('SHORT' if d=='LONG' else 'LONG')} scalp: {len(sub)} pairs, combined ${sub['combined_pnl'].mean():.2f}/pair")
else:
    print("  No wave pairs found.")
print()

# === 6. REVERSED trades near scalps (even worse pattern) ===
print("=" * 70)
print("6. REVERSED correct-dir trades near scalps")
print("=" * 70)
print("   (We entered RIGHT, market reversed, we lost. Then we scalped the reversal.)")

reversed_idx = df_sorted[df_sorted['is_reversed'] if 'is_reversed' in df_sorted.columns else df_sorted['dir_correct'] & (df_sorted['actual_pnl'] < 0)].index
rev_pairs = []
for idx in reversed_idx:
    if idx + 1 >= len(df_sorted):
        continue
    rev = df_sorted.iloc[idx]
    nt = df_sorted.iloc[idx + 1]
    gap = nt['entry_time'] - rev['exit_time']
    if gap < 300 and gap > 0 and nt['is_scalp']:
        rev_pairs.append({
            'rev_pnl': rev['actual_pnl'],
            'scalp_pnl': nt['actual_pnl'],
            'combined': rev['actual_pnl'] + nt['actual_pnl'],
            'gap_sec': gap,
        })

if rev_pairs:
    rp = pd.DataFrame(rev_pairs)
    print(f"  Reversed -> Scalp pairs (within 5min): {len(rp)}")
    print(f"  Avg reversed loss: ${rp['rev_pnl'].mean():.2f}")
    print(f"  Avg scalp profit:  ${rp['scalp_pnl'].mean():.2f}")
    print(f"  Net combined:      ${rp['combined'].mean():.2f}")
    print(f"  Total combined:    ${rp['combined'].sum():.2f}")
else:
    print("  No reversed->scalp pairs found.")
