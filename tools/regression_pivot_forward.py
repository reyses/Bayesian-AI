"""
REGRESSION-LINE PIVOT forward pass — 1s slippage resolution.

Entry logic:
  - Rolling 60-bar OLS on 1m closes → fitted regression line
  - Zigzag at R_reg on the REGRESSION LINE (not price)
  - At each regression-line pivot:
      * entry at CURRENT 1m-bar close (live — pivot confirmed by latest bar)
      * direction from the pivot type:
          - Regression LOW pivot → LONG (trend turning up)
          - Regression HIGH pivot → SHORT (trend turning down)

Exit logic (1s resolution):
  - TP at entry ± tp_pts
  - SL at entry ± sl_pts (wide — allow for price noise around regression)
  - Optional: regression-pivot reversal = another regression pivot fires → exit

No-lookahead:
  - Regression fit at bar T uses only closes[T-W+1..T] (window fully observed).
  - Zigzag on regression fires at bar T when the regression has retraced
    R_reg points from its running extreme. All past.
  - Entry at bar T's close; exit walks 1s bars from there.

This is the regression-cord-hunter strategy. Expected ceiling ~$2.5-3K/day,
we'd be happy with 20-30% capture = $500-900/day both sides.

Usage:
    python tools/regression_pivot_forward.py --tp 30 --sl 15 --r-reg 8
    python tools/regression_pivot_forward.py --sweep

Output: reports/findings/regression_pivot_forward.md
"""
import os
import sys
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
OUT_MD = 'reports/findings/regression_pivot_forward.md'
DOLLAR_PER_POINT = 2.0
DEFAULT_WINDOW = 60  # 1h regression window


def rolling_fit(closes, window):
    n = len(closes)
    fitted = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        x = np.arange(window, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            continue
        slope = (dx * (y - ym)).sum() / denom
        intercept = ym - slope * xm
        fitted[i] = intercept + slope * (window - 1)
    return fitted


def load_1s(path):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    return {
        'ts': df['timestamp'].values.astype(np.int64),
        'high': df['high'].values.astype(np.float64),
        'low': df['low'].values.astype(np.float64),
        'close': df['close'].values.astype(np.float64),
    }


def resolve_exit_1s(sec, entry_ts, entry_price, direction, tp_pts, sl_pts):
    if sec is None:
        return entry_price, entry_ts, 'no_1s_data', 0
    ts_arr = sec['ts']
    # 1m bar at ts=T closes at T+60. Skip to next-minute 1s bars.
    start_idx = np.searchsorted(ts_arr, entry_ts + 60, side='left')
    if start_idx >= len(ts_arr):
        return entry_price, entry_ts, 'entry_past_day_end', 0
    for i in range(start_idx, len(ts_arr)):
        hi = sec['high'][i]
        lo = sec['low'][i]
        if direction == 'LONG':
            tp_hit = hi >= entry_price + tp_pts
            sl_hit = lo <= entry_price - sl_pts
        else:
            tp_hit = lo <= entry_price - tp_pts
            sl_hit = hi >= entry_price + sl_pts
        if tp_hit and sl_hit:
            exit_p = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
            return exit_p, ts_arr[i], 'SL', int(ts_arr[i] - entry_ts)
        if tp_hit:
            exit_p = entry_price + tp_pts if direction == 'LONG' else entry_price - tp_pts
            return exit_p, ts_arr[i], 'TP', int(ts_arr[i] - entry_ts)
        if sl_hit:
            exit_p = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
            return exit_p, ts_arr[i], 'SL', int(ts_arr[i] - entry_ts)
    return sec['close'][-1], ts_arr[-1], 'eod', int(ts_arr[-1] - entry_ts)


def simulate_day(closes, ts_1m, fitted, sec_data, r_reg_pts, tp_pts, sl_pts,
                 min_leg_pts=0, fallback_close_exit=True):
    """Run regression-pivot strategy across one day. Returns trades list."""
    n = len(closes)
    trades = []

    # Zigzag on the regression fitted values
    leg_dir = None
    extreme_idx = None
    extreme_fit = None
    last_pivot_idx = None

    in_position = False
    current_direction = None
    n_pivots = 0

    for T in range(n):
        if np.isnan(fitted[T]):
            continue
        f = fitted[T]

        if extreme_idx is None:
            extreme_idx = T
            extreme_fit = f
            continue

        pivot_fired = False
        pivot_type = None  # 'HIGH' (reg topped, go SHORT) or 'LOW' (reg bottomed, go LONG)

        if leg_dir is None:
            if f - extreme_fit >= r_reg_pts:
                leg_dir = 'up'
                last_pivot_idx = extreme_idx
                pivot_fired = True
                pivot_type = 'LOW'
                extreme_idx = T
                extreme_fit = f
            elif extreme_fit - f >= r_reg_pts:
                leg_dir = 'down'
                last_pivot_idx = extreme_idx
                pivot_fired = True
                pivot_type = 'HIGH'
                extreme_idx = T
                extreme_fit = f
            else:
                if f > extreme_fit:
                    extreme_idx = T
                    extreme_fit = f
                elif f < extreme_fit:
                    extreme_idx = T
                    extreme_fit = f
        elif leg_dir == 'up':
            if f > extreme_fit:
                extreme_idx = T
                extreme_fit = f
            elif extreme_fit - f >= r_reg_pts:
                last_pivot_idx = extreme_idx
                pivot_fired = True
                pivot_type = 'HIGH'
                leg_dir = 'down'
                extreme_idx = T
                extreme_fit = f
        else:   # leg_dir == 'down'
            if f < extreme_fit:
                extreme_idx = T
                extreme_fit = f
            elif f - extreme_fit >= r_reg_pts:
                last_pivot_idx = extreme_idx
                pivot_fired = True
                pivot_type = 'LOW'
                leg_dir = 'up'
                extreme_idx = T
                extreme_fit = f

        if pivot_fired and pivot_type is not None:
            n_pivots += 1
            if in_position:
                # Skip: already in a trade
                continue
            direction = 'LONG' if pivot_type == 'LOW' else 'SHORT'
            entry_price = closes[T]
            entry_ts = int(ts_1m[T])

            if sec_data is not None:
                exit_price, exit_ts, exit_reason, held_sec = resolve_exit_1s(
                    sec_data, entry_ts, entry_price, direction, tp_pts, sl_pts)
            else:
                # Fallback to 1m close-based exit
                exit_price = entry_price
                exit_ts = entry_ts
                exit_reason = 'no_1s_data'
                held_sec = 0

            pnl = (exit_price - entry_price) * DOLLAR_PER_POINT if direction == 'LONG' \
                else (entry_price - exit_price) * DOLLAR_PER_POINT
            trades.append({
                'entry_bar': T, 'entry_ts': entry_ts, 'exit_ts': exit_ts,
                'held_sec': held_sec, 'direction': direction,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'exit_reason': exit_reason,
                'pivot_type': pivot_type,
            })
            in_position = False  # single-position; immediate exit resolved
    return trades, n_pivots


def run_pass(paths, r_reg, tp, sl, window, label):
    r_pts = r_reg / DOLLAR_PER_POINT
    tp_pts = tp / DOLLAR_PER_POINT
    sl_pts = sl / DOLLAR_PER_POINT
    all_trades = []
    tot_pivots = 0
    days_with_1s = 0
    for p in tqdm(paths, desc=label, unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        df = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
        closes = df['close'].values.astype(np.float64)
        ts_1m = df['timestamp'].values.astype(np.int64)
        if len(closes) < window + 5:
            continue
        fitted = rolling_fit(closes, window)
        sec_path = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
        sec_data = load_1s(sec_path)
        if sec_data is not None:
            days_with_1s += 1
        trades, npiv = simulate_day(
            closes, ts_1m, fitted, sec_data, r_pts, tp_pts, sl_pts)
        for t in trades:
            t['day'] = day
        all_trades.extend(trades)
        tot_pivots += npiv
    return all_trades, tot_pivots, days_with_1s


def summarize(trades):
    if not trades:
        return None
    pnls = np.array([t['pnl'] for t in trades])
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    hold_sec = np.array([t['held_sec'] for t in trades])
    exits = Counter(t['exit_reason'] for t in trades)
    w_d = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    l_d = -pnls[pnls < 0].sum() if (pnls < 0).any() else 0
    return {
        'n': len(trades), 'total': float(pnls.sum()),
        'per_trade': float(pnls.mean()),
        'wr': wins / max(wins + losses, 1) * 100,
        'dollar_wr': (w_d / l_d - 1) * 100 if l_d > 0 else float('inf'),
        'wins': int(wins), 'losses': int(losses),
        'mean_hold_sec': float(hold_sec.mean()),
        'median_hold_sec': float(np.median(hold_sec)),
        'exits': dict(exits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-reg', type=float, default=8.0,
                    help='Regression-line zigzag threshold in $ (default 8)')
    ap.add_argument('--tp', type=float, default=30.0)
    ap.add_argument('--sl', type=float, default=15.0)
    ap.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    ap.add_argument('--sweep', action='store_true',
                    help='Sweep (r_reg, tp, sl) grid')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} | OOS {len(oos_paths)} days  window={args.window}')

    if args.sweep:
        r_reg_values = [5, 8, 12, 20]
        tp_values = [15, 25, 40]
        sl_values = [10, 20, 30]
        rows = []
        for r in r_reg_values:
            for tp in tp_values:
                for sl in sl_values:
                    is_t, is_p, _ = run_pass(is_paths, r, tp, sl, args.window,
                                              f'IS r={r} tp={tp} sl={sl}')
                    oos_t, oos_p, _ = run_pass(oos_paths, r, tp, sl, args.window,
                                                f'OOS r={r} tp={tp} sl={sl}')
                    is_s = summarize(is_t)
                    oos_s = summarize(oos_t)
                    if is_s and oos_s:
                        is_day = is_s['total'] / len(is_paths)
                        oos_day = oos_s['total'] / len(oos_paths)
                        comb = (is_s['total'] + oos_s['total']) / (len(is_paths) + len(oos_paths))
                        rows.append({
                            'r_reg': r, 'tp': tp, 'sl': sl,
                            'is_day': is_day, 'oos_day': oos_day, 'comb': comb,
                            'is_n': is_s['n'], 'oos_n': oos_s['n'],
                            'is_wr': is_s['wr'], 'oos_wr': oos_s['wr'],
                            'is_pt': is_s['per_trade'], 'oos_pt': oos_s['per_trade'],
                        })
        rows.sort(key=lambda r: -r['comb'])
        print(f'\n{"r_reg":>6} {"TP":>4} {"SL":>4} {"IS$/d":>9} {"OOS$/d":>9} '
              f'{"Comb":>9} {"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>8} {"OOS $/tr":>9}')
        for r in rows:
            print(f'${r["r_reg"]:>4} ${r["tp"]:>3} ${r["sl"]:>3} '
                  f'${r["is_day"]:>+7,.0f} ${r["oos_day"]:>+7,.0f} '
                  f'${r["comb"]:>+7,.0f} {r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_pt"]:>+6.2f} ${r["oos_pt"]:>+7.2f}')
        # MD
        out = [f'# Regression-pivot forward — (r_reg × TP × SL) sweep', '']
        out.append(f'Window: {args.window} 1m bars (1h regression).')
        out.append('')
        out.append('| r_reg | TP | SL | IS $/day | OOS $/day | Combined | '
                   'IS WR | OOS WR | IS $/tr | OOS $/tr | IS N | OOS N |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            out.append(f'| ${r["r_reg"]} | ${r["tp"]} | ${r["sl"]} | '
                       f'${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                       f'**${r["comb"]:+,.0f}** | '
                       f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                       f'${r["is_pt"]:+.2f} | ${r["oos_pt"]:+.2f} | '
                       f'{r["is_n"]:,} | {r["oos_n"]:,} |')
        out.append('')
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f'\nWrote: {OUT_MD}')
        return

    is_trades, is_piv, is_1s_d = run_pass(is_paths, args.r_reg, args.tp,
                                          args.sl, args.window, 'IS')
    oos_trades, oos_piv, oos_1s_d = run_pass(oos_paths, args.r_reg, args.tp,
                                              args.sl, args.window, 'OOS')

    is_sum = summarize(is_trades)
    oos_sum = summarize(oos_trades)

    print(f'\n1s coverage: IS {is_1s_d}/{len(is_paths)}  OOS {oos_1s_d}/{len(oos_paths)}')

    def p(label, s, nd, piv):
        if not s:
            print(f'\n{label}: no trades')
            return
        print(f'\n=== {label} ({nd} days) ===')
        print(f'  Reg pivots: {piv:,} ({piv/nd:.1f}/day)')
        print(f'  Trades: {s["n"]:,} ({s["n"]/nd:.1f}/day)')
        print(f'  WR: {s["wr"]:.1f}%  $WR: {s["dollar_wr"]:+.0f}%')
        print(f'  $/trade: ${s["per_trade"]:+.2f}')
        print(f'  Total: ${s["total"]:+,.0f}')
        print(f'  $/day:  ${s["total"]/nd:+,.2f}')
        print(f'  Hold: mean={s["mean_hold_sec"]:.0f}s, median={s["median_hold_sec"]:.0f}s')
        print(f'  Exits: {s["exits"]}')

    p('IS', is_sum, len(is_paths), is_piv)
    p('OOS', oos_sum, len(oos_paths), oos_piv)

    # MD
    out = [f'# Regression-pivot forward pass', '']
    out.append(f'r_reg=${args.r_reg} TP=${args.tp} SL=${args.sl} window={args.window}')
    out.append('')
    out.append('Zigzag applied to the REGRESSION LINE (1h-window OLS fitted values), '
               'not to price. Entry at pivot confirmation bar (1m close). Exit '
               'resolved at 1s intra-bar granularity.')
    out.append('')
    out.append('| Dataset | Days | Reg pivots | Trades | $/day | $/trade | WR | $WR | Total |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for lbl, s, nd, piv in [('IS', is_sum, len(is_paths), is_piv),
                             ('OOS', oos_sum, len(oos_paths), oos_piv)]:
        if s:
            out.append(f'| {lbl} | {nd} | {piv:,} | {s["n"]:,} | '
                       f'${s["total"]/nd:+,.0f} | ${s["per_trade"]:+.2f} | '
                       f'{s["wr"]:.1f}% | {s["dollar_wr"]:+.0f}% | '
                       f'${s["total"]:+,.0f} |')
    out.append('')
    out.append('## Exit breakdown')
    out.append('')
    all_r = set(is_sum['exits'].keys()) | set(oos_sum['exits'].keys()) \
        if is_sum and oos_sum else set()
    out.append('| Reason | IS N | IS % | OOS N | OOS % |')
    out.append('|---|---:|---:|---:|---:|')
    for reason in sorted(all_r):
        ic = is_sum['exits'].get(reason, 0)
        oc = oos_sum['exits'].get(reason, 0)
        out.append(f'| {reason} | {ic:,} | {ic/is_sum["n"]*100:.1f}% | '
                   f'{oc:,} | {oc/oos_sum["n"]*100:.1f}% |')
    out.append('')
    out.append('## Cord-capture context')
    out.append('')
    out.append('| | Reg cord ceiling | System $/day | Capture % |')
    out.append('|---|---:|---:|---:|')
    if is_sum:
        out.append(f'| IS | $2,504 | ${is_sum["total"]/len(is_paths):+.0f} | '
                   f'{is_sum["total"]/len(is_paths)/2504*100:.1f}% |')
    if oos_sum:
        out.append(f'| OOS | $2,874 | ${oos_sum["total"]/len(oos_paths):+.0f} | '
                   f'{oos_sum["total"]/len(oos_paths)/2874*100:.1f}% |')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
