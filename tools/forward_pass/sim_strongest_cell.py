"""Test the strongest empirical reversion cell as a standalone strategy.

The probability table from sim_decay_rules.py identified:
  LONG bounce: z_1h_low ≤ -3σ AND slope_15m ≤ -0.5 → P_revert = 0.71  (n=1,529 IS)
  LONG bounce: z_1h_low ≤ -3σ AND slope_15m ≤ -0.1 → P_revert = 0.64

This is the strongest cell in the data. If it trade-validates, the framework
is real (just at lower frequency than the user's manual picks); if not, the
0.71 prob is partly lookahead artifact.

Strategy:
  Fire LONG when z_1h_low ≤ -3σ AND slope_15m ≤ -0.5
  Exit:
    target: hit z_1h_low = 0 (reverted to lower rail)  ≈ +30 to +60 ticks
    stop:   z_1h_low ≤ -5σ (deeper crash)  ≈ -20 to -40 ticks
    time:   60 min
  Cooldown: 30 min
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars, OUT_DIR


TICK, DOL = 0.25, 0.50
COOLDOWN_MIN = 30
MAX_HOLD_MIN = 60
HARD_STOP_Z = -99   # disabled — use tick-based stop instead
TARGET_TICKS = 20   # match the probability measurement window
STOP_TICKS = 20
ENTRY_Z_THR = -3.0
ENTRY_SLOPE_THR = -0.5


def simulate(close, high, low, ts, M_15m, S_15m, Mh, Sh, Ml, Sl):
    n = len(close)
    z_lo = np.where(Sl > 0, (close - Ml) / Sl, np.nan)
    z_hi = np.where(Sh > 0, (close - Mh) / Sh, np.nan)
    z_15m = np.where(S_15m > 0, (close - M_15m) / S_15m, np.nan)
    N = 15
    slope_15m = np.full(n, np.nan)
    slope_15m[N:] = (M_15m[N:] - M_15m[:-N]) / N

    trades = []
    open_trade = None
    cooldown_until = 0.0

    for i in range(N + 1, n):
        if open_trade is not None:
            pnl_ticks = (close[i] - open_trade['entry_price']) / TICK
            open_trade['peak_pnl'] = max(open_trade['peak_pnl'], pnl_ticks)
            open_trade['worst_pnl'] = min(open_trade['worst_pnl'], pnl_ticks)
            dur = (ts[i] - open_trade['entry_ts']) / 60.0
            close_now = False; reason = ''
            # Tick-based 1:1 R/R exits (match the probability measurement)
            if pnl_ticks >= TARGET_TICKS:
                close_now = True; reason = 'target_20t'
            elif pnl_ticks <= -STOP_TICKS:
                close_now = True; reason = 'stop_20t'
            elif dur >= MAX_HOLD_MIN:
                close_now = True; reason = 'time_stop'
            if close_now:
                open_trade['exit_ts'] = ts[i]
                open_trade['exit_price'] = close[i]
                open_trade['exit_reason'] = reason
                open_trade['duration_min'] = dur
                trades.append(open_trade)
                cooldown_until = ts[i] + COOLDOWN_MIN * 60
                open_trade = None
            else:
                continue
        if ts[i] < cooldown_until:
            continue
        # Entry: TRUE CUSP on z_1h_low — z was below -3σ at t-1, started rising at t.
        # Detects the bottom of the crash, not the start.
        prev_z_lo = z_lo[i-1] if i > 0 and not np.isnan(z_lo[i-1]) else np.nan
        prev2_z_lo = z_lo[i-2] if i > 1 and not np.isnan(z_lo[i-2]) else np.nan
        if (not np.isnan(z_lo[i]) and not np.isnan(prev_z_lo) and not np.isnan(prev2_z_lo)
            and prev2_z_lo <= ENTRY_Z_THR        # was below -3 two bars back
            and prev_z_lo <= ENTRY_Z_THR         # still below at t-1
            and z_lo[i] > prev_z_lo              # NOW recovering (z rising)
            and not np.isnan(slope_15m[i]) and slope_15m[i] <= ENTRY_SLOPE_THR):
            open_trade = {
                'entry_ts': ts[i], 'entry_price': close[i],
                'entry_z_lo': z_lo[i], 'entry_slope_15m': slope_15m[i],
                'peak_pnl': 0.0, 'worst_pnl': 0.0,
                'exit_ts': 0, 'exit_price': 0, 'exit_reason': '', 'duration_min': 0,
            }

    if open_trade is not None:
        open_trade['exit_ts'] = ts[-1]
        open_trade['exit_price'] = close[-1]
        open_trade['exit_reason'] = 'eod'
        open_trade['duration_min'] = (ts[-1] - open_trade['entry_ts']) / 60.0
        trades.append(open_trade)
    return trades


def report(trades, name):
    if not trades:
        print(f'{name}: NO TRADES')
        return
    pnls = np.array([(t['exit_price'] - t['entry_price']) / TICK * DOL for t in trades])
    n = len(trades)
    n_win = (pnls > 0).sum()
    win = pnls[pnls > 0].sum()
    lose = -pnls[pnls < 0].sum()
    pf_wr = (win / lose - 1) if lose > 0 else float('inf')
    from collections import Counter, defaultdict
    reasons = Counter(t['exit_reason'] for t in trades)
    daily = defaultdict(float)
    for t in trades:
        d = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
        daily[d] += (t['exit_price'] - t['entry_price']) / TICK * DOL
    ndays = max(1, len(daily))
    print(f'=== {name} (STRONGEST CELL: z_1h_low<=-3 AND slope_15m<=-0.5) ===')
    print(f'  n_trades = {n}  win_rate(cnt) = {100*n_win/n:.1f}%  PF-WR = {pf_wr:+.3f}')
    print(f'  total ${pnls.sum():.0f}  $/day ${pnls.sum()/ndays:.1f}  mean/t ${pnls.mean():.2f}  med ${np.median(pnls):.2f}')
    print(f'  active days {ndays}  reasons {dict(reasons)}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f'strongest_cell_{name}_trades.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['entry_utc','exit_utc','entry_px','exit_px','pnl','dur_min',
                       'reason','entry_z_lo','entry_slope_15m','peak_pnl','worst_pnl'])
        for t in trades:
            w.writerow([
                datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                round(t['entry_price'], 2), round(t['exit_price'], 2),
                round((t['exit_price']-t['entry_price'])/TICK*DOL, 2),
                round(t['duration_min'], 1),
                t['exit_reason'],
                round(t['entry_z_lo'], 3), round(t['entry_slope_15m'], 3),
                round(t['peak_pnl'], 1), round(t['worst_pnl'], 1),
            ])


def _ts(d):
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def run(start, end, name):
    t_s = _ts(start); t_e = _ts(end) + 86400
    print(f'Loading {start} → {end}...')
    df = load_1m_bars(t_s, t_e)
    if df.empty:
        print('No data'); return
    ts = df['timestamp'].values.astype(np.int64)
    print('Anchors...')
    M_15m, S_15m = compute_anchor('15m', ts, t_s, t_e, window=12, column='close')
    Mh, Sh = compute_anchor('1h', ts, t_s, t_e, window=12, column='high')
    Ml, Sl = compute_anchor('1h', ts, t_s, t_e, window=12, column='low')
    print('Simulating...')
    trades = simulate(df['close'].values, df['high'].values, df['low'].values,
                          ts, M_15m, S_15m, Mh, Sh, Ml, Sl)
    print(f'  {len(trades)} trades')
    report(trades, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    if not args.start:
        # Default: run all three samples
        for s, e, n in [
            ('2025-04-01', '2025-10-31', 'strongest_IS'),
            ('2025-11-01', '2025-12-31', 'strongest_valid'),
            ('2026-01-01', '2026-02-28', 'strongest_OOS'),
        ]:
            run(s, e, n)
            print()
    else:
        run(args.start, args.end, args.name or f'{args.start}_{args.end}')


if __name__ == '__main__':
    main()
