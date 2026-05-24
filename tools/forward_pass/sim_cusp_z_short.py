"""Symmetric SHORT-side test of cusp-on-z detection.
Mirror of sim_strongest_cell.py: fire SHORT when z_1h_high just stopped rising
after sustained >+1σ AND slope_15m is decelerating.

Tests whether the IS-only edge / OOS failure pattern is symmetric between
LONG (bottom cusps) and SHORT (top cusps), or if one side has structural
edge while the other doesn't.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars

TICK, DOL = 0.25, 0.50
COOLDOWN_MIN = 30
MAX_HOLD_MIN = 60
TARGET_TICKS = 20
STOP_TICKS = 20
ENTRY_Z_THR = 1.0
ENTRY_SLOPE_MAX = -0.1  # 15m flipped down


def simulate(close, high, low, ts, M_15m, S_15m, Mh, Sh, Ml, Sl):
    n = len(close)
    z_hi = np.where(Sh > 0, (close - Mh) / Sh, np.nan)
    N = 15
    slope_15m = np.full(n, np.nan)
    slope_15m[N:] = (M_15m[N:] - M_15m[:-N]) / N

    trades = []
    open_trade = None
    cooldown_until = 0.0

    for i in range(N + 1, n):
        if open_trade is not None:
            pnl = (open_trade['entry_price'] - close[i]) / TICK  # SHORT
            dur = (ts[i] - open_trade['entry_ts']) / 60.0
            close_now = False; reason = ''
            if pnl >= TARGET_TICKS:
                close_now = True; reason = 'target'
            elif pnl <= -STOP_TICKS:
                close_now = True; reason = 'stop'
            elif dur >= MAX_HOLD_MIN:
                close_now = True; reason = 'time'
            if close_now:
                open_trade['exit_ts'] = ts[i]; open_trade['exit_price'] = close[i]
                open_trade['reason'] = reason; open_trade['dur'] = dur
                trades.append(open_trade); open_trade = None
                cooldown_until = ts[i] + COOLDOWN_MIN * 60
            else:
                continue
        if ts[i] < cooldown_until:
            continue
        # Cusp on z_1h_high: was rising and sustained above +1σ, just turned down
        if i < 2: continue
        z = z_hi[i]; z1 = z_hi[i-1]; z2 = z_hi[i-2]
        s = slope_15m[i]
        if (not np.isnan(z) and not np.isnan(z1) and not np.isnan(z2)
            and z2 >= ENTRY_Z_THR and z1 >= ENTRY_Z_THR
            and z < z1                            # peak just passed
            and not np.isnan(s) and s <= ENTRY_SLOPE_MAX):
            open_trade = {'entry_ts': ts[i], 'entry_price': close[i],
                              'z_at_entry': z1, 'slope_15m': s}

    if open_trade is not None:
        open_trade['exit_ts'] = ts[-1]; open_trade['exit_price'] = close[-1]
        open_trade['reason'] = 'eod'
        open_trade['dur'] = (ts[-1] - open_trade['entry_ts']) / 60.0
        trades.append(open_trade)
    return trades


def report(trades, name):
    if not trades:
        print(f'{name}: NO TRADES'); return
    pnls = np.array([(t['entry_price'] - t['exit_price']) / TICK * DOL for t in trades])
    n = len(trades)
    nw = (pnls > 0).sum()
    win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf_wr = (win / lose - 1) if lose > 0 else float('inf')
    from collections import defaultdict
    daily = defaultdict(float)
    for t in trades:
        daily[datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')] += (t['entry_price']-t['exit_price'])/TICK*DOL
    ndays = max(1, len(daily))
    print(f'{name}: n={n} wr={100*nw/n:.1f}% PF-WR={pf_wr:+.3f} total ${pnls.sum():.0f} $/day ${pnls.sum()/ndays:.1f}')


def _ts(d): return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def run(s, e, name):
    t_s = _ts(s); t_e = _ts(e) + 86400
    df = load_1m_bars(t_s, t_e)
    if df.empty: print(f'{name}: no data'); return
    ts = df['timestamp'].values.astype(np.int64)
    M_15m, S_15m = compute_anchor('15m', ts, t_s, t_e, window=12, column='close')
    Mh, Sh = compute_anchor('1h', ts, t_s, t_e, window=12, column='high')
    Ml, Sl = compute_anchor('1h', ts, t_s, t_e, window=12, column='low')
    trades = simulate(df['close'].values, df['high'].values, df['low'].values,
                          ts, M_15m, S_15m, Mh, Sh, Ml, Sl)
    report(trades, name)


def main():
    for s, e, n in [
        ('2025-04-01', '2025-10-31', 'SHORT-cusp_IS'),
        ('2025-11-01', '2025-12-31', 'SHORT-cusp_valid'),
        ('2026-01-01', '2026-02-28', 'SHORT-cusp_OOS'),
    ]:
        run(s, e, n)


if __name__ == '__main__':
    main()
