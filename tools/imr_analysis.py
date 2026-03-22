"""I-MR Trade Analysis — IS trades normalized to seconds + trend alignment."""
import json
import pandas as pd
import numpy as np

# Load IS trade replays
with open('reports/trade_replays/is_replays.json') as f:
    trades = json.load(f)
print(f'IS trades: {len(trades)}')

# Load trend seeds
with open('DATA/regime_seeds/auto_swing/auto_seeds_all_20260313_200730.json') as f:
    auto = json.load(f)
auto_seeds = []
for d in auto['days'].values():
    auto_seeds.extend(d['seeds'])
auto_starts = np.array([s['ts_start'] for s in auto_seeds])
auto_ends = np.array([s['ts_end'] for s in auto_seeds])
auto_dirs = [s['direction'] for s in auto_seeds]

results = []
for t in trades:
    bars = t['bars']
    states = t['states']
    if not bars or not states:
        continue

    entry_ts = bars[0][0] if bars else 0
    hold_s = t['hold_bars'] * 15
    pnl = t['actual_pnl']
    side = t['side']
    exit_reason = t['exit_reason']

    # Find trend seed
    idx = np.searchsorted(auto_starts, entry_ts, side='right') - 1
    trend_dir = None
    trend_progress = None
    for j in range(max(0, idx - 2), min(len(auto_seeds), idx + 3)):
        if auto_starts[j] <= entry_ts <= auto_ends[j]:
            trend_dir = auto_dirs[j]
            trend_progress = (entry_ts - auto_starts[j]) / max(auto_seeds[j]['duration_mins'] * 60, 1)
            break

    peak_dir = 'LONG' if side == 'long' else 'SHORT'
    aligned = (trend_dir == peak_dir) if trend_dir else None

    entry_state = states[min(t['entry_bar'], len(states) - 1)]
    exit_state = states[min(t['exit_bar'], len(states) - 1)]

    fm_entry = entry_state.get('f_mom', 0)
    fm_exit = exit_state.get('f_mom', 0)
    z_entry = entry_state.get('z', 0)
    z_exit = exit_state.get('z', 0)
    dmi_diff_entry = entry_state.get('dmi_p', 0) - entry_state.get('dmi_m', 0)
    dmi_diff_exit = exit_state.get('dmi_p', 0) - exit_state.get('dmi_m', 0)

    results.append({
        'trade_id': t['trade_id'], 'pnl': pnl, 'hold_s': hold_s,
        'hold_min': hold_s / 60, 'side': side, 'exit_reason': exit_reason,
        'aligned': aligned, 'trend_progress': trend_progress,
        'fm_entry': fm_entry, 'fm_exit': fm_exit, 'fm_delta': fm_exit - fm_entry,
        'z_entry': z_entry, 'z_exit': z_exit,
        'dmi_diff_entry': dmi_diff_entry, 'dmi_diff_exit': dmi_diff_exit,
        'mfe': t.get('trade_mfe_ticks', 0),
    })

df = pd.DataFrame(results)
wins = df[df.pnl > 0]
losses = df[df.pnl <= 0]

print()
print('=' * 70)
print('IS TRADE I-MR ANALYSIS (normalized seconds/minutes)')
print('=' * 70)

print()
print('=== HOLD DURATION ===')
print(f'  Winners:  mean={wins.hold_s.mean():.0f}s ({wins.hold_min.mean():.1f}m)  median={wins.hold_s.median():.0f}s')
print(f'  Losers:   mean={losses.hold_s.mean():.0f}s ({losses.hold_min.mean():.1f}m)  median={losses.hold_s.median():.0f}s')

print()
print('=== PnL BY HOLD (seconds) ===')
buckets = [(0, 30, '<30s'), (30, 60, '30s-1m'), (60, 120, '1-2m'), (120, 300, '2-5m'),
           (300, 600, '5-10m'), (600, 99999, '10m+')]
hdr = f'  {"Bucket":<10} {"N":>5} {"WR":>5} {"AvgPnL":>8} {"TotalPnL":>10} {"AvgMFE":>7}'
print(hdr)
for lo, hi, label in buckets:
    mask = (df.hold_s >= lo) & (df.hold_s < hi)
    sub = df[mask]
    if len(sub) > 0:
        wr = (sub.pnl > 0).mean() * 100
        print(f'  {label:<10} {len(sub):>5} {wr:>4.0f}% {sub.pnl.mean():>+7.1f} {sub.pnl.sum():>+9.0f} {sub.mfe.mean():>6.1f}t')

print()
print('=== TREND ALIGNMENT (peak vs trend seed direction) ===')
al = df[df.aligned == True]
ct = df[df.aligned == False]
nt = df[df.aligned.isna()]
if len(al) > 0:
    print(f'  Aligned:  N={len(al)} WR={(al.pnl > 0).mean() * 100:.0f}% avg=${al.pnl.mean():.1f} total=${al.pnl.sum():.0f}')
if len(ct) > 0:
    print(f'  Counter:  N={len(ct)} WR={(ct.pnl > 0).mean() * 100:.0f}% avg=${ct.pnl.mean():.1f} total=${ct.pnl.sum():.0f}')
if len(nt) > 0:
    print(f'  No trend: N={len(nt)} WR={(nt.pnl > 0).mean() * 100:.0f}% avg=${nt.pnl.mean():.1f} total=${nt.pnl.sum():.0f}')

print()
print('=== ENTRY POSITION IN TREND SEED ===')
has_trend = df[df.trend_progress.notna()]
if len(has_trend) > 0:
    for lo, hi, label in [(0, 0.2, 'Start 0-20%'), (0.2, 0.5, 'Early 20-50%'),
                           (0.5, 0.8, 'Late 50-80%'), (0.8, 1.01, 'End 80-100%')]:
        mask = (has_trend.trend_progress >= lo) & (has_trend.trend_progress < hi)
        sub = has_trend[mask]
        if len(sub) > 0:
            wr = (sub.pnl > 0).mean() * 100
            print(f'  {label:<15} N={len(sub):>4} WR={wr:.0f}% avg=${sub.pnl.mean():.1f} MFE={sub.mfe.mean():.0f}t')

print()
print('=== STATE AT ENTRY (wins vs losses) ===')
for col in ['fm_entry', 'z_entry', 'dmi_diff_entry']:
    w = wins[col].mean()
    l = losses[col].mean()
    print(f'  {col:<18} WIN={w:>+8.2f}  LOSS={l:>+8.2f}  delta={w - l:>+8.2f}')

print()
print('=== STATE DELTA (entry -> exit) ===')
for col in ['fm_delta', 'z_exit', 'dmi_diff_exit']:
    w = wins[col].mean()
    l = losses[col].mean()
    print(f'  {col:<18} WIN={w:>+8.2f}  LOSS={l:>+8.2f}  delta={w - l:>+8.2f}')

# Save report
output = 'reports/findings/imr_trade_analysis.txt'
import sys, io
# Re-run with capture
buf = io.StringIO()
old_stdout = sys.stdout
sys.stdout = buf
# (already printed above, just save the key findings)
sys.stdout = old_stdout
print(f'\nAnalysis complete. Results printed above.')
print(f'Trades analyzed: {len(df)} (wins={len(wins)}, losses={len(losses)})')
