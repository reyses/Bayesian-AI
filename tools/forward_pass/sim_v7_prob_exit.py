"""v7 — Same v6 3-body entry; PROBABILITY-BASED exit at each bar.

Per user 2026-05-12: exit should ask at every bar: "given current structure,
what's the probability my move continues / reverses?" — exit when the cell
flips to high-continuation-against-me probability.

Architecture (same as v6 for entries):
    1h HL  = POSITION (slow context)
    15m CRM = TARGET (mean-reversion target)
    15s CRM = TRIGGER (slope flip cusp)

NEW exit logic — at every bar in trade:
    For SHORT: P_continue_up = 1 − P_revert_short(z_1h_high, slope_15m)
    For LONG : P_continue_dn = 1 − P_revert_long(z_1h_low,  slope_15m)
    Exit when P_continue >= EXIT_PROB_HIGH (0.65 default)

Hard PnL stop remains as safety net.
"""
from __future__ import annotations
import argparse, csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import (
    load_1m_bars, p_revert_short, p_revert_long, OUT_DIR,
)


TICK, DOL = 0.25, 0.50

# v6 entry thresholds (3-body)
SHORT_Z_15M_ENTRY = +1.5
LONG_Z_15M_ENTRY = -1.5
SHORT_MAX_POS = 99
LONG_MIN_POS = -2.0
SLOPE_LONG_RISING = +0.30
SLOPE_LONG_FALLING = -0.30
SLOPE_SHORT_FLIP_DN = -0.10
SLOPE_SHORT_FLIP_UP = +0.10
SHORT_MIN_POS_IN_BAND = 0.50
LONG_MAX_POS_IN_BAND = 0.50

# Exit logic
EXIT_PROB_HIGH = 0.55   # P_continue >= 0.55 → tilt against me → exit
HARD_STOP_TICKS = 60.0   # max-pain stop (match v6)
COOLDOWN_MIN = 15
MAX_HOLD_MIN = 60
SLOPE_15S_LB_LONG = 10
SLOPE_15S_LB_SHORT = 3
SLOPE_15M_LB = 15


def simulate(close, high, low, ts, M_15s, M_15m, S_15m, Mh, Sh, Ml, Sl):
    n = len(close)
    band_w = Mh - Ml
    pos = np.where(band_w > 0, (M_15s - Ml) / band_w, np.nan)
    z_hi = np.where(Sh > 0, (close - Mh) / Sh, np.nan)
    z_lo = np.where(Sl > 0, (close - Ml) / Sl, np.nan)
    z_15m = np.where(S_15m > 0, (close - M_15m) / S_15m, np.nan)
    slope_long = np.full(n, np.nan)
    slope_short = np.full(n, np.nan)
    slope_15m = np.full(n, np.nan)
    if n > SLOPE_15S_LB_LONG:
        slope_long[SLOPE_15S_LB_LONG:] = (
            M_15s[SLOPE_15S_LB_LONG:] - M_15s[:-SLOPE_15S_LB_LONG]) / SLOPE_15S_LB_LONG
    if n > SLOPE_15S_LB_SHORT:
        slope_short[SLOPE_15S_LB_SHORT:] = (
            M_15s[SLOPE_15S_LB_SHORT:] - M_15s[:-SLOPE_15S_LB_SHORT]) / SLOPE_15S_LB_SHORT
    if n > SLOPE_15M_LB:
        slope_15m[SLOPE_15M_LB:] = (
            M_15m[SLOPE_15M_LB:] - M_15m[:-SLOPE_15M_LB]) / SLOPE_15M_LB

    trades = []
    open_t = None
    cooldown_until = 0.0

    for i in range(SLOPE_15S_LB_LONG + 1, n):
        # ── Manage open ─────────────────────────────────────
        if open_t is not None:
            if open_t['side'] == 'LONG':
                pnl_ticks = (close[i] - open_t['entry_price']) / TICK
                # Probability the ADVERSE direction (continue down) is likely
                p_revert_long_now = p_revert_long(z_lo[i], slope_15m[i]) if not np.isnan(z_lo[i]) and not np.isnan(slope_15m[i]) else 0.5
                p_continue_against = 1 - p_revert_long_now
            else:
                pnl_ticks = (open_t['entry_price'] - close[i]) / TICK
                p_revert_short_now = p_revert_short(z_hi[i], slope_15m[i]) if not np.isnan(z_hi[i]) and not np.isnan(slope_15m[i]) else 0.5
                p_continue_against = 1 - p_revert_short_now
            open_t['peak'] = max(open_t['peak'], pnl_ticks)
            open_t['worst'] = min(open_t['worst'], pnl_ticks)
            dur = (ts[i] - open_t['entry_ts']) / 60.0

            close_now = False; reason = ''
            # 1) Hard PnL stop (safety net)
            if pnl_ticks <= -HARD_STOP_TICKS:
                close_now = True; reason = 'hard_stop'
            # 2) PROBABILITY exit — current state says adverse continuation likely
            elif p_continue_against >= EXIT_PROB_HIGH:
                close_now = True; reason = f'p_continue={p_continue_against:.2f}'
            # 3) Profit-take: revert to 15m mean
            elif open_t['side'] == 'SHORT' and z_15m[i] <= 0:
                close_now = True; reason = 'target_15m_mean'
            elif open_t['side'] == 'LONG' and z_15m[i] >= 0:
                close_now = True; reason = 'target_15m_mean'
            # 4) Time stop
            elif dur >= MAX_HOLD_MIN:
                close_now = True; reason = 'time_stop'

            if close_now:
                open_t['exit_ts'] = ts[i]; open_t['exit_price'] = close[i]
                open_t['reason'] = reason; open_t['dur_min'] = dur
                trades.append(open_t); open_t = None
                cooldown_until = ts[i] + COOLDOWN_MIN * 60
            else:
                continue

        # ── Entry ───────────────────────────────────────────
        if ts[i] < cooldown_until or np.isnan(slope_long[i]) or np.isnan(z_15m[i]):
            continue

        # SHORT cusp entry
        if (z_15m[i] >= SHORT_Z_15M_ENTRY
            and slope_long[i] >= SLOPE_LONG_RISING
            and slope_short[i] <= SLOPE_SHORT_FLIP_DN
            and pos[i] >= SHORT_MIN_POS_IN_BAND):
            open_t = {'side': 'SHORT', 'entry_ts': ts[i], 'entry_price': close[i],
                          'entry_z_15m': z_15m[i], 'entry_z_1h_high': z_hi[i],
                          'entry_slope_15m': slope_15m[i],
                          'peak': 0.0, 'worst': 0.0}
        # LONG cusp entry
        elif (z_15m[i] <= LONG_Z_15M_ENTRY
              and slope_long[i] <= SLOPE_LONG_FALLING
              and slope_short[i] >= SLOPE_SHORT_FLIP_UP
              and pos[i] <= LONG_MAX_POS_IN_BAND):
            open_t = {'side': 'LONG', 'entry_ts': ts[i], 'entry_price': close[i],
                          'entry_z_15m': z_15m[i], 'entry_z_1h_low': z_lo[i],
                          'entry_slope_15m': slope_15m[i],
                          'peak': 0.0, 'worst': 0.0}

    if open_t is not None:
        open_t['exit_ts'] = ts[-1]; open_t['exit_price'] = close[-1]
        open_t['reason'] = 'eod'; open_t['dur_min'] = (ts[-1] - open_t['entry_ts']) / 60.0
        trades.append(open_t)
    return trades


def report(trades, name):
    if not trades:
        print(f'{name}: 0 trades'); return
    pnls = np.array([(t['exit_price']-t['entry_price'])/TICK*DOL if t['side']=='LONG'
                          else (t['entry_price']-t['exit_price'])/TICK*DOL for t in trades])
    n = len(trades)
    nw = (pnls > 0).sum()
    win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf_wr = (win/lose - 1) if lose > 0 else float('inf')
    from collections import Counter, defaultdict
    reasons = Counter(t['reason'].split('=')[0] for t in trades)
    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        d = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
        daily[d] += p
    ndays = max(1, len(daily))
    print(f'{name}: n={n} wr={100*nw/n:.1f}% PF-WR={pf_wr:+.3f} total ${pnls.sum():.0f} $/day ${pnls.sum()/ndays:.1f} reasons={dict(reasons)}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR/f'v7_{name}_trades.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['side','entry_utc','exit_utc','entry_px','exit_px','pnl','dur_min','reason','peak_ticks','worst_ticks'])
        for t,p in zip(trades, pnls):
            w.writerow([t['side'],
                            datetime.fromtimestamp(t['entry_ts'],tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            datetime.fromtimestamp(t['exit_ts'],tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            round(t['entry_price'],2), round(t['exit_price'],2),
                            round(p,2), round(t['dur_min'],1), t['reason'],
                            round(t['peak'],1), round(t['worst'],1)])


def _ts(d): return datetime.strptime(d,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def run(start, end, name):
    t_s = _ts(start); t_e = _ts(end) + 86400
    print(f'Loading {start} → {end}...')
    df = load_1m_bars(t_s, t_e)
    if df.empty: print('No data'); return
    ts = df['timestamp'].values.astype(np.int64)
    M_15s, S_15s = compute_anchor('15s', ts, t_s, t_e, window=20, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_s, t_e, window=12, column='close')
    Mh, Sh = compute_anchor('1h', ts, t_s, t_e, window=12, column='high')
    Ml, Sl = compute_anchor('1h', ts, t_s, t_e, window=12, column='low')
    trades = simulate(df['close'].values, df['high'].values, df['low'].values,
                          ts, M_15s, M_15m, S_15m, Mh, Sh, Ml, Sl)
    report(trades, name)


def main():
    for s, e, n in [
        ('2025-07-01', '2025-07-31', 'v7_unmarked_Jul'),
        ('2026-01-01', '2026-02-28', 'v7_OOS'),
    ]:
        run(s, e, n)


if __name__ == '__main__':
    main()
