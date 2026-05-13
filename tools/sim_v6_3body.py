"""v6 — 3-body architecture (proper anchor separation per user 2026-05-11).

1h HL   = POSITION (slow context — where in the structural band; gate)
15m CRM = TARGET   (mean-reversion target — where price wants to return)
15s CRM = TRIGGER  (fast cusp signal — when to fire)

Rules:
  SHORT cusp:
    z_15m_crm >= +1.5         (extension above 15m mean → room to revert)
    slope_15s flipped + → -   (cusp at top of 15s oscillation)
    pos_in_band <= 1.2        (1h context: not in extreme crash extension)

  LONG cusp:
    z_15m_crm <= -1.5         (extension below 15m mean)
    slope_15s flipped - → +   (cusp at bottom)
    pos_in_band >= -0.2       (1h context: not in extreme rally extension)

Exits:
  TARGET: z_15m_crm crosses back through 0 (reverted to 15m mean)
  STOP:   opposite extension (e.g. SHORT stops at z_15m_crm >= +2.5)
  TIME:   45 min

This is the small-target mean-reversion trade the user's marker captured.
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
COOLDOWN_MIN = 15
MAX_HOLD_MIN = 45

# Entry thresholds (from user picks distribution)
SHORT_Z_15M_ENTRY = +1.5
LONG_Z_15M_ENTRY = -1.5

# Position-context bounds (1h HL — loose for SHORT, tighter for LONG)
SHORT_MAX_POS_IN_BAND = 99    # 1h is just context, very loose for shorts
LONG_MIN_POS_IN_BAND  = -2.0  # don't long when in extreme rally above 1h_high

# Slope-flip detection (cusp trigger on 15s CRM)
SLOPE_15S_LB_LONG = 10   # 10-min slope (recent direction)
SLOPE_15S_LB_SHORT = 3   # 3-min slope (now)
SLOPE_LONG_THR_RISING = +0.30
SLOPE_LONG_THR_FALLING = -0.30
SLOPE_SHORT_THR_FLIP_DOWN = -0.10
SLOPE_SHORT_THR_FLIP_UP = +0.10

# Exit thresholds — target the 15m mean, stop on more extension
SHORT_TARGET_Z_15M = 0.0    # cross back through 15m mean
SHORT_STOP_Z_15M   = +2.7   # extension grew another σ → stop
LONG_TARGET_Z_15M  = 0.0
LONG_STOP_Z_15M    = -2.7


def simulate(close, high, low, ts, M_15s, M_15m, S_15m, Mh, Sh, Ml, Sl):
    n = len(close)
    band_w = Mh - Ml
    pos = np.where(band_w > 0, (M_15s - Ml) / band_w, np.nan)
    z_15m = np.where(S_15m > 0, (close - M_15m) / S_15m, np.nan)
    # Slopes on M_15s
    slope_long = np.full(n, np.nan)
    slope_short = np.full(n, np.nan)
    if n > SLOPE_15S_LB_LONG:
        slope_long[SLOPE_15S_LB_LONG:] = (
            M_15s[SLOPE_15S_LB_LONG:] - M_15s[:-SLOPE_15S_LB_LONG]) / SLOPE_15S_LB_LONG
    if n > SLOPE_15S_LB_SHORT:
        slope_short[SLOPE_15S_LB_SHORT:] = (
            M_15s[SLOPE_15S_LB_SHORT:] - M_15s[:-SLOPE_15S_LB_SHORT]) / SLOPE_15S_LB_SHORT

    trades = []
    open_trade = None
    cooldown_until = 0.0

    for i in range(SLOPE_15S_LB_LONG + 1, n):
        if open_trade is not None:
            if open_trade['side'] == 'LONG':
                pnl_ticks = (close[i] - open_trade['entry_price']) / TICK
            else:
                pnl_ticks = (open_trade['entry_price'] - close[i]) / TICK
            open_trade['peak'] = max(open_trade['peak'], pnl_ticks)
            open_trade['worst'] = min(open_trade['worst'], pnl_ticks)
            dur = (ts[i] - open_trade['entry_ts']) / 60.0

            close_now = False; reason = ''
            if not np.isnan(z_15m[i]):
                if open_trade['side'] == 'SHORT':
                    if z_15m[i] <= SHORT_TARGET_Z_15M:
                        close_now = True; reason = 'target_15m_mean'
                    elif z_15m[i] >= SHORT_STOP_Z_15M:
                        close_now = True; reason = 'stop_z_extension'
                else:  # LONG
                    if z_15m[i] >= LONG_TARGET_Z_15M:
                        close_now = True; reason = 'target_15m_mean'
                    elif z_15m[i] <= LONG_STOP_Z_15M:
                        close_now = True; reason = 'stop_z_extension'
            if not close_now and dur >= MAX_HOLD_MIN:
                close_now = True; reason = 'time_stop'

            if close_now:
                open_trade['exit_ts'] = ts[i]
                open_trade['exit_price'] = close[i]
                open_trade['reason'] = reason
                open_trade['dur_min'] = dur
                trades.append(open_trade)
                cooldown_until = ts[i] + COOLDOWN_MIN * 60
                open_trade = None
            else:
                continue

        if ts[i] < cooldown_until:
            continue
        if (np.isnan(z_15m[i]) or np.isnan(slope_long[i]) or np.isnan(slope_short[i])
            or np.isnan(pos[i])):
            continue

        # SHORT cusp: z_15m above +1.5, slope_15s flipped from rising to falling
        if (z_15m[i] >= SHORT_Z_15M_ENTRY
            and slope_long[i] >= SLOPE_LONG_THR_RISING
            and slope_short[i] <= SLOPE_SHORT_THR_FLIP_DOWN
            and pos[i] <= SHORT_MAX_POS_IN_BAND):
            open_trade = {'side': 'SHORT', 'entry_ts': ts[i], 'entry_price': close[i],
                              'entry_z_15m': z_15m[i], 'entry_pos': pos[i],
                              'slope_long': slope_long[i], 'slope_short': slope_short[i],
                              'peak': 0.0, 'worst': 0.0,
                              'exit_ts': 0, 'exit_price': 0, 'reason': '', 'dur_min': 0}
        # LONG cusp: z_15m below -1.5, slope_15s flipped from falling to rising
        elif (z_15m[i] <= LONG_Z_15M_ENTRY
              and slope_long[i] <= SLOPE_LONG_THR_FALLING
              and slope_short[i] >= SLOPE_SHORT_THR_FLIP_UP
              and pos[i] >= LONG_MIN_POS_IN_BAND):
            open_trade = {'side': 'LONG', 'entry_ts': ts[i], 'entry_price': close[i],
                              'entry_z_15m': z_15m[i], 'entry_pos': pos[i],
                              'slope_long': slope_long[i], 'slope_short': slope_short[i],
                              'peak': 0.0, 'worst': 0.0,
                              'exit_ts': 0, 'exit_price': 0, 'reason': '', 'dur_min': 0}

    if open_trade is not None:
        open_trade['exit_ts'] = ts[-1]; open_trade['exit_price'] = close[-1]
        open_trade['reason'] = 'eod'; open_trade['dur_min'] = (ts[-1] - open_trade['entry_ts']) / 60.0
        trades.append(open_trade)
    return trades


def report(trades, name, oracle=0):
    if not trades:
        print(f'{name}: 0 trades'); return
    pnls = np.array([(t['exit_price'] - t['entry_price']) / TICK * DOL if t['side']=='LONG'
                          else (t['entry_price'] - t['exit_price']) / TICK * DOL
                          for t in trades])
    n = len(trades)
    longs = sum(1 for t in trades if t['side']=='LONG')
    shorts = n - longs
    nw = (pnls > 0).sum()
    win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf_wr = (win/lose - 1) if lose > 0 else float('inf')
    from collections import Counter, defaultdict
    reasons = Counter(t['reason'] for t in trades)
    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        d = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
        daily[d] += p
    ndays = max(1, len(daily))
    print(f'=== {name} (v6 3-body — z_15m trigger + 1h pos context + 15m target) ===')
    print(f'  n_trades={n} (L={longs} S={shorts})  win_rate={100*nw/n:.1f}%  PF-WR={pf_wr:+.3f}')
    print(f'  total ${pnls.sum():.0f}  $/day ${pnls.sum()/ndays:.1f}  mean/t ${pnls.mean():.2f}  '
              f'med ${np.median(pnls):.2f}  days_active={ndays}')
    print(f'  reasons: {dict(reasons)}')
    if oracle > 0:
        print(f'  capture: {100*pnls.sum()/oracle:.0f}% (${pnls.sum():.0f} / ${oracle:.0f})')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f'v6_{name}_trades.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['side','entry_utc','exit_utc','entry_px','exit_px','pnl','dur_min',
                       'reason','entry_z_15m','entry_pos','slope_long','slope_short',
                       'peak_ticks','worst_ticks'])
        for t, p in zip(trades, pnls):
            w.writerow([t['side'],
                            datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            round(t['entry_price'],2), round(t['exit_price'],2),
                            round(p,2), round(t['dur_min'],1), t['reason'],
                            round(t['entry_z_15m'],3), round(t['entry_pos'],3),
                            round(t['slope_long'],3), round(t['slope_short'],3),
                            round(t['peak'],1), round(t['worst'],1)])


def _ts(d): return datetime.strptime(d,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def run(start, end, name, oracle=0):
    t_s = _ts(start); t_e = _ts(end) + 86400
    print(f'Loading {start} → {end} ...')
    df = load_1m_bars(t_s, t_e)
    if df.empty:
        print('No data'); return
    ts = df['timestamp'].values.astype(np.int64)
    M_15s, S_15s = compute_anchor('15s', ts, t_s, t_e, window=20, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_s, t_e, window=12, column='close')
    Mh, Sh = compute_anchor('1h', ts, t_s, t_e, window=12, column='high')
    Ml, Sl = compute_anchor('1h', ts, t_s, t_e, window=12, column='low')
    trades = simulate(df['close'].values, df['high'].values, df['low'].values,
                          ts, M_15s, M_15m, S_15m, Mh, Sh, Ml, Sl)
    report(trades, name, oracle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--days')
    ap.add_argument('--name')
    ap.add_argument('--oracle', type=float, default=0)
    args = ap.parse_args()
    if args.days:
        days = [d.strip() for d in args.days.split(',')]
        # run as single contiguous range
        run(min(days), max(days), args.name or 'days_run', args.oracle)
    elif args.start:
        run(args.start, args.end, args.name or f'{args.start}_{args.end}', args.oracle)
    else:
        # default: full sweep
        for s, e, n, o in [
            ('2025-06-06', '2025-06-06', 'v6_06-06_oracle_$908', 908),
            ('2025-09-08', '2025-09-10', 'v6_09-08_to_09-10_oracle_$2400', 2400),
            ('2025-07-01', '2025-07-31', 'v6_unmarked_Jul_IS', 0),
            ('2026-01-01', '2026-02-28', 'v6_OOS', 0),
        ]:
            run(s, e, n, o)
            print()


if __name__ == '__main__':
    main()
