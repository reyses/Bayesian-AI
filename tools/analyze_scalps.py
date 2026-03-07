"""Analyze why the brain isn't picking up trend signals on counter-trend scalps."""
import pandas as pd, json, numpy as np

df = pd.read_csv('checkpoints/oos_trade_log.csv')

df['dir_correct'] = ((df['direction']=='LONG')&(df['oracle_label']>0))|((df['direction']=='SHORT')&(df['oracle_label']<0))
df['is_scalp'] = (~df['dir_correct']) & (df['oracle_label']!=0) & (df['actual_pnl']>0)

scalps = df[df['is_scalp']]
correct = df[df['dir_correct']]

def get_workers(row):
    try:
        w = json.loads(row['entry_workers'])
    except:
        return {}
    out = {}
    for tf in ['4h','1h','30m','15m','5m','3m','1m']:
        if tf in w:
            out[f'{tf}_d'] = w[tf].get('d', 0.5)
            out[f'{tf}_c'] = w[tf].get('c', 0.5)
            out[f'{tf}_band'] = w[tf].get('band', 0)
            out[f'{tf}_z'] = w[tf].get('z', 0.0)
    return out

scalp_w = scalps.apply(get_workers, axis=1).apply(pd.Series)
correct_w = correct.apply(get_workers, axis=1).apply(pd.Series)

print("=== SLOW WORKER SIGNALS: Counter-trend scalps vs Correct dir ===")
print("(d = P(SHORT), so high d = worker says SHORT)")
print()
print(f"  {'TF':4s}  {'Scalp d':>8s}  {'Correct d':>10s}  {'Delta':>7s}  {'Scalp band':>11s}  {'Correct band':>13s}")
for tf in ['4h','1h','30m','15m','5m','3m','1m']:
    d_col = f'{tf}_d'
    b_col = f'{tf}_band'
    if d_col in scalp_w.columns and d_col in correct_w.columns:
        sd = scalp_w[d_col].mean()
        cd = correct_w[d_col].mean()
        sb = scalp_w[b_col].mean() if b_col in scalp_w.columns else 0
        cb = correct_w[b_col].mean() if b_col in correct_w.columns else 0
        print(f"  {tf:4s}  {sd:8.3f}  {cd:10.3f}  {cd-sd:+7.3f}  {sb:11.2f}  {cb:13.2f}")

print()
print("=== SCALP DIRECTION BREAKDOWN ===")
for d in ['LONG','SHORT']:
    sub = scalps[scalps['direction']==d]
    if len(sub) > 0:
        print(f"  {d}: {len(sub)} scalps, avg PnL {sub['actual_pnl'].mean():.2f}")

print()
print("=== WHAT ORACLE MOVE ARE SCALPS FIGHTING? ===")
for label in sorted(scalps['oracle_label_name'].unique()):
    sub = scalps[scalps['oracle_label_name']==label]
    print(f"  {label:20s}: {len(sub)} trades, avg PnL {sub['actual_pnl'].mean():.2f}")

print()
print("=== CONVICTION: Scalps vs Correct ===")
print(f"  Scalp conviction:   {scalps['belief_conviction'].mean():.4f}")
print(f"  Correct conviction: {correct['belief_conviction'].mean():.4f}")
print(f"  Delta: {correct['belief_conviction'].mean() - scalps['belief_conviction'].mean():+.4f}")

print()
print("=== KEY QUESTION: Are slow TFs (1h/4h) signaling the right trend? ===")
# For scalps: we went SHORT but oracle says LONG. Does 1h worker agree with oracle (LONG)?
scalp_short_vs_long = scalps[(scalps['direction']=='SHORT') & (scalps['oracle_label']>0)]
scalp_long_vs_short = scalps[(scalps['direction']=='LONG') & (scalps['oracle_label']<0)]

if len(scalp_short_vs_long) > 0:
    w_data = scalp_short_vs_long.apply(get_workers, axis=1).apply(pd.Series)
    print(f"\n  SHORT scalps fighting LONG oracle ({len(scalp_short_vs_long)} trades):")
    for tf in ['4h','1h','30m','15m']:
        d_col = f'{tf}_d'
        if d_col in w_data.columns:
            avg_d = w_data[d_col].mean()
            # d < 0.5 = worker says LONG (agrees with oracle)
            # d > 0.5 = worker says SHORT (agrees with trade, disagrees with oracle)
            oracle_agree_pct = (w_data[d_col] < 0.5).mean() * 100
            print(f"    {tf:4s}: avg d={avg_d:.3f}, worker agrees with oracle (LONG): {oracle_agree_pct:.1f}%")

if len(scalp_long_vs_short) > 0:
    w_data = scalp_long_vs_short.apply(get_workers, axis=1).apply(pd.Series)
    print(f"\n  LONG scalps fighting SHORT oracle ({len(scalp_long_vs_short)} trades):")
    for tf in ['4h','1h','30m','15m']:
        d_col = f'{tf}_d'
        if d_col in w_data.columns:
            avg_d = w_data[d_col].mean()
            oracle_agree_pct = (w_data[d_col] > 0.5).mean() * 100
            print(f"    {tf:4s}: avg d={avg_d:.3f}, worker agrees with oracle (SHORT): {oracle_agree_pct:.1f}%")

print()
print("=== TEMPLATE CONCENTRATION: Which templates produce most scalps? ===")
top_templates = scalps.groupby('template_id').agg(
    n=('actual_pnl','count'),
    avg_pnl=('actual_pnl','mean'),
    total_pnl=('actual_pnl','sum'),
    playbook=('playbook','first')
).sort_values('n', ascending=False).head(10)
print(f"  {'TID':>5s}  {'N':>4s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  Playbook")
for tid, row in top_templates.iterrows():
    print(f"  {tid:5d}  {row['n']:4.0f}  {row['avg_pnl']:8.2f}  {row['total_pnl']:10.2f}  {row['playbook']}")
