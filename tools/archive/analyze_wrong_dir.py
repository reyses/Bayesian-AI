"""Analyze wrong-direction trades and simulate filters."""
import pandas as pd, numpy as np, json, sys

df = pd.read_csv('checkpoints/oos_trade_log.csv')

df['dir_correct'] = ((df['direction'] == 'LONG') & (df['oracle_label'] > 0)) | \
                    ((df['direction'] == 'SHORT') & (df['oracle_label'] < 0))
df['is_noise'] = df['oracle_label'] == 0

correct = df[df['dir_correct']]
wrong = df[~df['dir_correct'] & ~df['is_noise']]

print(f"Correct: {len(correct)}  Wrong: {len(wrong)}  Noise: {df.is_noise.sum()}")
print()

# Parse worker agreement
def count_worker_agreement(row):
    try:
        workers = json.loads(row['entry_workers'])
    except:
        return {}
    trade_dir = row['direction']
    result = {}
    for tf, w in workers.items():
        d = w.get('d', 0.5)
        if trade_dir == 'LONG':
            agrees = d < 0.5
        else:
            agrees = d > 0.5
        result[tf] = 1 if agrees else 0
    return result

agreements = df.apply(count_worker_agreement, axis=1).apply(pd.Series)

print("=== WORKER AGREEMENT: Correct vs Wrong direction ===")
print("(agree% = worker direction matches trade direction)")
print()
tfs = ['4h','1h','30m','15m','5m','3m','1m','30s','15s','5s','1s']
for tf in tfs:
    if tf not in agreements.columns:
        continue
    c_agree = agreements.loc[correct.index, tf].mean()
    w_agree = agreements.loc[wrong.index, tf].mean()
    delta = c_agree - w_agree
    flag = "  <-- FILTER" if delta > 0.05 else ""
    print(f"  {tf:4s}  correct={c_agree:.3f}  wrong={w_agree:.3f}  delta={delta:+.3f}{flag}")

print()
print("=== WRONG-DIR TRADES: Winners vs Losers ===")
wrong_win = wrong[wrong['actual_pnl'] > 0]
wrong_loss = wrong[wrong['actual_pnl'] <= 0]
print(f"  Wrong-dir winners: {len(wrong_win)} ({len(wrong_win)/len(wrong)*100:.1f}%) avg PnL: {wrong_win['actual_pnl'].mean():.2f}")
print(f"  Wrong-dir losers:  {len(wrong_loss)} ({len(wrong_loss)/len(wrong)*100:.1f}%) avg PnL: {wrong_loss['actual_pnl'].mean():.2f}")
print(f"  Wrong-dir NET PnL: {wrong['actual_pnl'].sum():.2f}")
print()

# Conviction threshold
print("=== CONVICTION THRESHOLD SIMULATION ===")
print("  (reject trades below conviction X)")
print(f"  {'Thresh':>8s}  {'Trades':>6s}  {'Dir%':>6s}  {'WR':>6s}  {'TotalPnL':>10s}  {'Avg':>8s}")
for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65]:
    kept = df[df['belief_conviction'] >= thresh]
    k_correct = ((kept['direction'] == 'LONG') & (kept['oracle_label'] > 0)) | \
                ((kept['direction'] == 'SHORT') & (kept['oracle_label'] < 0))
    k_noise = kept['oracle_label'] == 0
    n = len(kept)
    if n == 0: continue
    n_correct = k_correct.sum()
    n_wrong = (~k_correct & ~k_noise).sum()
    pnl = kept['actual_pnl'].sum()
    wr = (kept['actual_pnl'] > 0).mean() * 100
    dir_pct = n_correct / (n_correct + n_wrong) * 100 if (n_correct + n_wrong) > 0 else 0
    print(f"  >= {thresh:.2f}   {n:6d}  {dir_pct:5.1f}%  {wr:5.1f}%  {pnl:10.0f}  {pnl/n:8.2f}")

print()
print("=== WORKER TF AGREEMENT FILTER ===")
for tf_filter in ['15m', '1h', '30m']:
    if tf_filter not in agreements.columns:
        continue
    print(f"  Filter: require {tf_filter} worker agrees with trade direction")
    agrees_col = agreements[tf_filter]

    kept_idx = agrees_col[agrees_col == 1].index
    kept = df.loc[kept_idx]
    k_correct = ((kept['direction'] == 'LONG') & (kept['oracle_label'] > 0)) | \
                ((kept['direction'] == 'SHORT') & (kept['oracle_label'] < 0))
    n = len(kept)
    n_correct = k_correct.sum()
    n_wrong = n - n_correct - (kept['oracle_label'] == 0).sum()
    pnl = kept['actual_pnl'].sum()
    wr = (kept['actual_pnl'] > 0).mean() * 100
    dir_pct = n_correct / (n_correct + n_wrong) * 100 if (n_correct + n_wrong) > 0 else 0
    print(f"    Agree:   {n:4d} trades, dir {dir_pct:.1f}%, WR {wr:.1f}%, PnL {pnl:.0f}, avg {pnl/n:.2f}")

    rej_idx = agrees_col[agrees_col == 0].index
    rej = df.loc[rej_idx]
    r_pnl = rej['actual_pnl'].sum()
    rn = len(rej)
    print(f"    Reject:  {rn:4d} trades, PnL {r_pnl:.0f}, avg {r_pnl/rn:.2f}")
    print()

# Combined filter: conviction + 15m agreement
print("=== COMBINED FILTERS ===")
print(f"  {'Filter':40s}  {'N':>5s}  {'Dir%':>6s}  {'WR':>6s}  {'PnL':>10s}  {'Avg':>8s}")
combos = [
    ("Baseline (no filter)", df.index),
    ("conv >= 0.54", df[df['belief_conviction'] >= 0.54].index),
    ("conv >= 0.56", df[df['belief_conviction'] >= 0.56].index),
    ("15m agrees", agreements[agreements.get('15m', pd.Series()) == 1].index if '15m' in agreements.columns else pd.Index([])),
    ("conv>=0.54 + 15m agrees", df[(df['belief_conviction'] >= 0.54) & (agreements.get('15m', pd.Series(dtype=float)).reindex(df.index, fill_value=0) == 1)].index if '15m' in agreements.columns else pd.Index([])),
    ("conv>=0.56 + 15m agrees", df[(df['belief_conviction'] >= 0.56) & (agreements.get('15m', pd.Series(dtype=float)).reindex(df.index, fill_value=0) == 1)].index if '15m' in agreements.columns else pd.Index([])),
]

for name, idx in combos:
    kept = df.loc[idx]
    n = len(kept)
    if n == 0:
        print(f"  {name:40s}  {0:5d}  --")
        continue
    k_correct = ((kept['direction'] == 'LONG') & (kept['oracle_label'] > 0)) | \
                ((kept['direction'] == 'SHORT') & (kept['oracle_label'] < 0))
    k_noise = kept['oracle_label'] == 0
    n_correct = k_correct.sum()
    n_wrong = (~k_correct & ~k_noise).sum()
    pnl = kept['actual_pnl'].sum()
    wr = (kept['actual_pnl'] > 0).mean() * 100
    dir_pct = n_correct / (n_correct + n_wrong) * 100 if (n_correct + n_wrong) > 0 else 0
    print(f"  {name:40s}  {n:5d}  {dir_pct:5.1f}%  {wr:5.1f}%  {pnl:10.0f}  {pnl/n:8.2f}")

# DMI filter
print()
print("=== DMI_DIFF FILTER (negative = bearish) ===")
# For SHORT trades, negative dmi_diff should confirm. For LONG, positive.
df['dmi_confirms'] = ((df['direction'] == 'SHORT') & (df['dmi_diff'] < 0)) | \
                     ((df['direction'] == 'LONG') & (df['dmi_diff'] > 0))
for confirm_val in [True, False]:
    sub = df[df['dmi_confirms'] == confirm_val]
    n = len(sub)
    if n == 0: continue
    s_correct = ((sub['direction'] == 'LONG') & (sub['oracle_label'] > 0)) | \
                ((sub['direction'] == 'SHORT') & (sub['oracle_label'] < 0))
    s_noise = sub['oracle_label'] == 0
    nc = s_correct.sum()
    nw = (~s_correct & ~s_noise).sum()
    pnl = sub['actual_pnl'].sum()
    wr = (sub['actual_pnl'] > 0).mean() * 100
    dp = nc / (nc + nw) * 100 if (nc + nw) > 0 else 0
    label = "DMI confirms" if confirm_val else "DMI conflicts"
    print(f"  {label:20s}  {n:4d} trades, dir {dp:.1f}%, WR {wr:.1f}%, PnL {pnl:.0f}, avg {pnl/n:.2f}")

# Hurst filter
print()
print("=== HURST FILTER (>0.5 = trending) ===")
for h_thresh in [0.50, 0.55, 0.60, 0.65]:
    sub = df[df['hurst'] >= h_thresh]
    n = len(sub)
    if n == 0: continue
    s_correct = ((sub['direction'] == 'LONG') & (sub['oracle_label'] > 0)) | \
                ((sub['direction'] == 'SHORT') & (sub['oracle_label'] < 0))
    s_noise = sub['oracle_label'] == 0
    nc = s_correct.sum()
    nw = (~s_correct & ~s_noise).sum()
    pnl = sub['actual_pnl'].sum()
    wr = (sub['actual_pnl'] > 0).mean() * 100
    dp = nc / (nc + nw) * 100 if (nc + nw) > 0 else 0
    print(f"  hurst >= {h_thresh:.2f}  {n:4d} trades, dir {dp:.1f}%, WR {wr:.1f}%, PnL {pnl:.0f}, avg {pnl/n:.2f}")
