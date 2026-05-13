"""v5: PURE probability-gate sim — no slope-flip cusp, no anchor cusps.
At each 1m bar, query the empirical P_revert table on (z_1h_high or z_1h_low,
slope_15m). Fire trade when P_revert >= 0.55. Exit on +20 ticks target or
-20 ticks stop (matching the table's threshold).

This is the cleanest test of whether the empirical conditional probability
has tradeable edge by itself, decoupled from any cusp detection.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import (
    load_1m_bars, compute_anchor_set, p_revert_short, p_revert_long,
    TICK, DOL, OUT_DIR
)


TARGET_TICKS = 20
STOP_TICKS = 20
COOLDOWN_MIN = 30
MAX_HOLD_MIN = 120
P_FIRE = 0.55


@dataclass
class Trade:
    side: str
    entry_ts: float
    entry_price: float
    exit_ts: float = 0
    exit_price: float = 0
    exit_reason: str = ''
    duration_min: float = 0
    p_at_entry: float = 0
    z_at_entry: float = 0
    slope_15m_at_entry: float = 0

    @property
    def pnl_dollars(self) -> float:
        if self.exit_ts == 0:
            return 0
        if self.side == 'LONG':
            return ((self.exit_price - self.entry_price) / TICK) * DOL
        return ((self.entry_price - self.exit_price) / TICK) * DOL


def simulate(close, high, low, ts, a):
    """Pure prob-gate sim. SHORT on high P_revert_short; LONG on high P_revert_long."""
    # Vectorize features
    Mh = a['Mh_1h']; Sh = a['Sh_1h']
    Ml = a['Ml_1h']; Sl = a['Sl_1h']
    M_15m = a['M_15m']
    n = len(close)
    z_hi = np.where(Sh > 0, (close - Mh) / Sh, np.nan)
    z_lo = np.where(Sl > 0, (close - Ml) / Sl, np.nan)
    N = 15
    slope_15m = np.full(n, np.nan)
    slope_15m[N:] = (M_15m[N:] - M_15m[:-N]) / N

    trades = []
    open_trade: Optional[Trade] = None
    cooldown_until = 0.0

    for i in range(N + 1, n):
        # Manage open
        if open_trade is not None:
            if open_trade.side == 'LONG':
                pnl_ticks = (close[i] - open_trade.entry_price) / TICK
            else:
                pnl_ticks = (open_trade.entry_price - close[i]) / TICK
            dur = (ts[i] - open_trade.entry_ts) / 60.0
            close_now = False; reason = ''
            if pnl_ticks >= TARGET_TICKS:
                close_now = True; reason = 'target'
            elif pnl_ticks <= -STOP_TICKS:
                close_now = True; reason = 'stop'
            elif dur >= MAX_HOLD_MIN:
                close_now = True; reason = 'time_stop'
            if close_now:
                open_trade.exit_ts = ts[i]; open_trade.exit_price = close[i]
                open_trade.exit_reason = reason; open_trade.duration_min = dur
                trades.append(open_trade)
                cooldown_until = ts[i] + COOLDOWN_MIN * 60
                open_trade = None
            else:
                continue

        if ts[i] < cooldown_until:
            continue
        if np.isnan(z_hi[i]) or np.isnan(z_lo[i]) or np.isnan(slope_15m[i]):
            continue

        p_s = p_revert_short(z_hi[i], slope_15m[i])
        p_l = p_revert_long(z_lo[i], slope_15m[i])

        if p_s >= P_FIRE and p_s >= p_l:
            open_trade = Trade(side='SHORT', entry_ts=ts[i], entry_price=close[i],
                                  p_at_entry=p_s, z_at_entry=z_hi[i],
                                  slope_15m_at_entry=slope_15m[i])
        elif p_l >= P_FIRE:
            open_trade = Trade(side='LONG', entry_ts=ts[i], entry_price=close[i],
                                  p_at_entry=p_l, z_at_entry=z_lo[i],
                                  slope_15m_at_entry=slope_15m[i])

    if open_trade is not None:
        open_trade.exit_ts = ts[-1]; open_trade.exit_price = close[-1]
        open_trade.exit_reason = 'eod'
        open_trade.duration_min = (ts[-1] - open_trade.entry_ts) / 60.0
        trades.append(open_trade)
    return trades


def report(trades, run_name, oracle=0):
    pnls = np.array([t.pnl_dollars for t in trades]) if trades else np.array([0.0])
    n = len(trades)
    from collections import Counter, defaultdict
    daily = defaultdict(float)
    for t in trades:
        day = datetime.fromtimestamp(t.entry_ts, tz=timezone.utc).strftime('%Y-%m-%d')
        daily[day] += t.pnl_dollars
    n_days = max(1, len(daily))
    n_win = (pnls > 0).sum()
    n_lose = (pnls < 0).sum()
    win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf_wr = (win / lose - 1) if lose > 0 else float('inf')
    reasons = Counter(t.exit_reason for t in trades)
    long_n = sum(1 for t in trades if t.side == 'LONG')
    short_n = n - long_n

    summary = (
        f'=== {run_name} (PURE prob-gate; P_FIRE={P_FIRE}, target/stop=±{TARGET_TICKS}t) ===\n'
        f'n_trades={n}  L={long_n} S={short_n}  days_active={n_days}\n'
        f'Total ${pnls.sum():.0f}  $/day ${pnls.sum()/n_days:.1f}  '
        f'mean ${pnls.mean():.2f}  median ${np.median(pnls):.2f}\n'
        f'win_rate(cnt) {100*n_win/n:.1f}% ({n_win}W/{n_lose}L)  PF-WR {pf_wr:+.3f}\n'
        f'reasons {dict(reasons)}\n')
    if oracle > 0:
        summary += f'capture {100*pnls.sum()/oracle:.0f}% (sim ${pnls.sum():.0f} / oracle ${oracle:.0f})\n'
    print(summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f'{run_name}_summary.txt', 'w') as f:
        f.write(summary)
    with open(OUT_DIR / f'{run_name}_trades.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['side', 'entry_utc', 'exit_utc', 'entry_px', 'exit_px',
                       'pnl', 'duration_min', 'reason', 'p_at_entry',
                       'z_at_entry', 'slope_15m_at_entry'])
        for t in trades:
            w.writerow([t.side,
                            datetime.fromtimestamp(t.entry_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            datetime.fromtimestamp(t.exit_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            round(t.entry_price, 2), round(t.exit_price, 2),
                            round(t.pnl_dollars, 2), round(t.duration_min, 1),
                            t.exit_reason, round(t.p_at_entry, 3),
                            round(t.z_at_entry, 3), round(t.slope_15m_at_entry, 3)])


def _ts_of(d):
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--days')
    ap.add_argument('--name')
    ap.add_argument('--oracle', type=float, default=0)
    args = ap.parse_args()

    if args.days:
        day_list = [d.strip() for d in args.days.split(',')]
        sd = sorted(day_list)
        t_start = _ts_of(sd[0]); t_end = _ts_of(sd[-1]) + 86400
        run_name = args.name or f'days_{sd[0]}_to_{sd[-1]}'
    else:
        t_start = _ts_of(args.start); t_end = _ts_of(args.end) + 86400
        run_name = args.name or f'range_{args.start}_to_{args.end}'

    print(f'Loading bars {datetime.fromtimestamp(t_start, tz=timezone.utc)}-{datetime.fromtimestamp(t_end, tz=timezone.utc)}')
    df = load_1m_bars(t_start, t_end)
    if args.days:
        keep = pd.Series(False, index=df.index)
        for d in [_ts_of(x) for x in day_list]:
            keep |= (df['timestamp'] >= d) & (df['timestamp'] < d + 86400)
        df = df[keep].reset_index(drop=True)
    print(f'  {len(df)} bars')
    if df.empty:
        return

    ts = df['timestamp'].values.astype(np.int64)
    print('Anchors...')
    a = compute_anchor_set(ts, t_start, t_end)
    print('Simulating...')
    trades = simulate(df['close'].values, df['high'].values,
                          df['low'].values, ts, a)
    print(f'  {len(trades)} trades')
    report(trades, run_name, args.oracle)


if __name__ == '__main__':
    main()
