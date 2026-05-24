"""
FULL 1s forward pass: pivot detection AND execution both at 1s resolution.

Previous sims (pivot_forward_1s.py) used 1m bars for pivot detection and
1s for exit. That meant confirmation took 60+ seconds on average.

This version confirms pivots at 1-second resolution. With r_confirm=$2,
pivots confirm within seconds of the true extreme — much closer entry.

No-lookahead:
  - Pivot detection at 1s uses only closes[0..T] at time T.
  - Residual (1m_z_se) feature at the 1s pivot timestamp uses the nearest
    prior 5s FEATURES bar (no lookahead; N-1 higher-TF already baked in).
  - Entry at the 1s confirmation bar's close.
  - Exit walks forward 1s bars from T+1.

Usage:
    python tools/pivot_1s_forward.py --r-confirm 2 --tp 20 --sl 3
    python tools/pivot_1s_forward.py --sweep

Output: reports/findings/pivot_1s_forward.md
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


ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_1s_forward.md'
DOLLAR_PER_POINT = 2.0


def load_day(sec_path, feat_path):
    """Load 1s OHLC + nearest-prior 1m_z_se for each 1s bar."""
    df_sec = pd.read_parquet(sec_path).sort_values('timestamp').reset_index(drop=True)
    closes = df_sec['close'].values.astype(np.float64)
    highs = df_sec['high'].values.astype(np.float64)
    lows = df_sec['low'].values.astype(np.float64)
    ts = df_sec['timestamp'].values.astype(np.int64)

    df_feat = pd.read_parquet(feat_path).sort_values('timestamp').reset_index(drop=True)
    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    if '1m_z_se' not in df_feat.columns:
        return closes, highs, lows, ts, np.zeros(len(closes))
    res_feat = df_feat['1m_z_se'].values.astype(np.float64)
    idx = np.searchsorted(ts_feat, ts, side='right') - 1
    idx = np.clip(idx, 0, len(ts_feat) - 1)
    residuals_per_1s = res_feat[idx]
    return closes, highs, lows, ts, residuals_per_1s


def simulate_day(closes, highs, lows, ts, residuals,
                 r_confirm_pts, tp_pts, sl_pts, min_res_strength,
                 cooldown_sec=0, min_leg_pts=0.0):
    """
    cooldown_sec: seconds to wait after any trade exit before taking new pivot.
    min_leg_pts: require previous leg (last_pivot to this_pivot) >= this magnitude.
    """
    n = len(closes)
    trades = []

    # Pivot tracking state (at 1s granularity)
    leg_dir = None
    extreme_idx = 0
    extreme_price = closes[0]
    last_pivot_idx = None
    last_pivot_price = closes[0]
    last_exit_ts = 0  # cooldown anchor

    in_position = False
    entry_idx = None
    entry_price = None
    entry_ts = None
    direction = None
    trade_sl_pts = sl_pts
    n_pivots = 0
    n_skipped_cooldown = 0
    n_skipped_small_leg = 0

    for T in range(n):
        price = closes[T]

        # Handle position exit (in-position, so we check TP/SL first)
        if in_position and T > entry_idx:
            hi = highs[T]
            lo = lows[T]
            if direction == 'LONG':
                tp_hit = hi >= entry_price + tp_pts
                sl_hit = lo <= entry_price - sl_pts
            else:
                tp_hit = lo <= entry_price - tp_pts
                sl_hit = hi >= entry_price + sl_pts
            if tp_hit and sl_hit:
                exit_p = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': T,
                    'entry_ts': entry_ts, 'exit_ts': int(ts[T]),
                    'direction': direction, 'entry_price': entry_price,
                    'exit_price': exit_p,
                    'pnl': -sl_pts * DOLLAR_PER_POINT,
                    'exit_reason': 'SL', 'held_sec': int(ts[T] - entry_ts),
                })
                in_position = False
                last_exit_ts = int(ts[T])
                continue
            if tp_hit:
                exit_p = entry_price + tp_pts if direction == 'LONG' else entry_price - tp_pts
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': T,
                    'entry_ts': entry_ts, 'exit_ts': int(ts[T]),
                    'direction': direction, 'entry_price': entry_price,
                    'exit_price': exit_p,
                    'pnl': tp_pts * DOLLAR_PER_POINT,
                    'exit_reason': 'TP', 'held_sec': int(ts[T] - entry_ts),
                })
                in_position = False
                last_exit_ts = int(ts[T])
                continue
            if sl_hit:
                exit_p = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': T,
                    'entry_ts': entry_ts, 'exit_ts': int(ts[T]),
                    'direction': direction, 'entry_price': entry_price,
                    'exit_price': exit_p,
                    'pnl': -sl_pts * DOLLAR_PER_POINT,
                    'exit_reason': 'SL', 'held_sec': int(ts[T] - entry_ts),
                })
                in_position = False
                last_exit_ts = int(ts[T])
                continue

        # Pivot tracking
        pivot_fired = False
        pivot_at = None
        if leg_dir is None:
            if price - extreme_price >= r_confirm_pts:
                leg_dir = 'up'
                pivot_at = extreme_idx
                pivot_fired = True
                extreme_idx = T
                extreme_price = price
            elif extreme_price - price >= r_confirm_pts:
                leg_dir = 'down'
                pivot_at = extreme_idx
                pivot_fired = True
                extreme_idx = T
                extreme_price = price
            elif price > extreme_price:
                extreme_idx = T
                extreme_price = price
            elif price < extreme_price:
                extreme_idx = T
                extreme_price = price
        elif leg_dir == 'up':
            if price > extreme_price:
                extreme_idx = T
                extreme_price = price
            elif extreme_price - price >= r_confirm_pts:
                pivot_at = extreme_idx
                pivot_fired = True
                leg_dir = 'down'
                extreme_idx = T
                extreme_price = price
        else:
            if price < extreme_price:
                extreme_idx = T
                extreme_price = price
            elif price - extreme_price >= r_confirm_pts:
                pivot_at = extreme_idx
                pivot_fired = True
                leg_dir = 'up'
                extreme_idx = T
                extreme_price = price

        if pivot_fired and pivot_at is not None:
            n_pivots += 1
            # Measure previous leg magnitude for "real pivot" filter
            pivot_price = closes[pivot_at]
            prev_leg_pts = abs(pivot_price - last_pivot_price)
            # Update last_pivot_price AFTER using prev leg
            last_pivot_price = pivot_price

            if in_position:
                continue
            # Cooldown filter
            if cooldown_sec > 0 and int(ts[T]) - last_exit_ts < cooldown_sec:
                n_skipped_cooldown += 1
                continue
            # Min-leg filter
            if prev_leg_pts < min_leg_pts:
                n_skipped_small_leg += 1
                continue
            # Residual at the pivot 1s bar
            r = float(residuals[pivot_at])
            if np.isnan(r) or abs(r) < min_res_strength:
                continue
            direction = 'LONG' if r < 0 else 'SHORT'
            entry_idx = T
            entry_price = closes[T]
            entry_ts = int(ts[T])
            in_position = True

    return trades, n_pivots


def run_pass(paths, r_confirm, tp, sl, min_res, label,
             cooldown_sec=0, min_leg_dollars=0.0):
    r_pts = r_confirm / DOLLAR_PER_POINT
    tp_pts = tp / DOLLAR_PER_POINT
    sl_pts = sl / DOLLAR_PER_POINT
    min_leg_pts = min_leg_dollars / DOLLAR_PER_POINT
    all_trades = []
    tot_pivots = 0
    n_days_with_1s = 0
    for p in tqdm(paths, desc=label, unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(feat_path):
            continue
        try:
            closes, highs, lows, ts, residuals = load_day(p, feat_path)
        except Exception as e:
            continue
        if len(closes) < 30:
            continue
        n_days_with_1s += 1
        trades, npiv = simulate_day(closes, highs, lows, ts, residuals,
                                     r_pts, tp_pts, sl_pts, min_res,
                                     cooldown_sec, min_leg_pts)
        for t in trades:
            t['day'] = day
        all_trades.extend(trades)
        tot_pivots += npiv
    return all_trades, tot_pivots, n_days_with_1s


def summarize(trades):
    if not trades:
        return None
    pnls = np.array([t['pnl'] for t in trades])
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    hold = np.array([t['held_sec'] for t in trades])
    exits = Counter(t['exit_reason'] for t in trades)
    w_d = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    l_d = -pnls[pnls < 0].sum() if (pnls < 0).any() else 0
    return {
        'n': len(trades), 'total': float(pnls.sum()),
        'per_trade': float(pnls.mean()),
        'wr': wins / max(wins + losses, 1) * 100,
        'dollar_wr': (w_d / l_d - 1) * 100 if l_d > 0 else float('inf'),
        'wins': int(wins), 'losses': int(losses),
        'mean_hold_sec': float(hold.mean()),
        'median_hold_sec': float(np.median(hold)),
        'exits': dict(exits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-confirm', type=float, default=2.0,
                    help='1s pivot retracement in $ (default 2)')
    ap.add_argument('--tp', type=float, default=20.0)
    ap.add_argument('--sl', type=float, default=3.0)
    ap.add_argument('--min-res-strength', type=float, default=0.5)
    ap.add_argument('--cooldown', type=float, default=0,
                    help='Seconds of wait after a trade exit before taking new pivot')
    ap.add_argument('--min-leg', type=float, default=0.0,
                    help='Require previous leg (pivot-to-pivot) >= this $ magnitude')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()

    # 1s files exist only for days in 1s dir
    is_paths = sorted(glob.glob(os.path.join(ATLAS_1S_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1S_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} | OOS {len(oos_paths)} days')

    if args.sweep:
        r_values = [1, 2, 3, 5]
        tp_values = [10, 15, 20, 30, 50]
        sl_values = [2, 3, 5, 10]
        rows = []
        for r in r_values:
            for tp in tp_values:
                for sl in sl_values:
                    is_t, is_p, _ = run_pass(is_paths, r, tp, sl,
                                              args.min_res_strength,
                                              f'IS r={r} tp={tp} sl={sl}')
                    oos_t, oos_p, _ = run_pass(oos_paths, r, tp, sl,
                                                args.min_res_strength,
                                                f'OOS r={r} tp={tp} sl={sl}')
                    is_s = summarize(is_t)
                    oos_s = summarize(oos_t)
                    if is_s and oos_s:
                        nd_is = len(is_paths)
                        nd_oos = len(oos_paths)
                        is_day = is_s['total'] / nd_is
                        oos_day = oos_s['total'] / nd_oos
                        comb = (is_s['total'] + oos_s['total']) / (nd_is + nd_oos)
                        rows.append({
                            'r': r, 'tp': tp, 'sl': sl,
                            'is_day': is_day, 'oos_day': oos_day, 'comb': comb,
                            'is_wr': is_s['wr'], 'oos_wr': oos_s['wr'],
                            'is_pt': is_s['per_trade'], 'oos_pt': oos_s['per_trade'],
                            'is_n': is_s['n'], 'oos_n': oos_s['n'],
                        })
        rows.sort(key=lambda r: -r['comb'])
        print(f'\n{"r":>3} {"TP":>4} {"SL":>3} {"IS$/d":>9} {"OOS$/d":>9} '
              f'{"Comb":>9} {"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>8} {"OOS N":>7}')
        for r in rows[:20]:
            print(f'${r["r"]:>2} ${r["tp"]:>3} ${r["sl"]:>2} '
                  f'${r["is_day"]:>+7,.0f} ${r["oos_day"]:>+7,.0f} '
                  f'${r["comb"]:>+7,.0f} {r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_pt"]:>+6.2f} {r["oos_n"]:>7,}')
        # MD
        out = [f'# 1s-pivot forward pass — (r × TP × SL) sweep', '']
        out.append('Pivot detection AND execution both at 1s resolution.')
        out.append('')
        out.append('| r_confirm | TP | SL | IS $/day | OOS $/day | Combined | '
                   'IS WR | OOS WR | IS $/tr | OOS $/tr | IS N | OOS N |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            out.append(f'| ${r["r"]} | ${r["tp"]} | ${r["sl"]} | '
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

    is_trades, is_piv, is_d = run_pass(is_paths, args.r_confirm, args.tp,
                                        args.sl, args.min_res_strength, 'IS',
                                        args.cooldown, args.min_leg)
    oos_trades, oos_piv, oos_d = run_pass(oos_paths, args.r_confirm, args.tp,
                                           args.sl, args.min_res_strength, 'OOS',
                                           args.cooldown, args.min_leg)
    is_sum = summarize(is_trades)
    oos_sum = summarize(oos_trades)

    def p(label, s, nd, piv):
        if not s:
            print(f'\n{label}: no trades')
            return
        print(f'\n=== {label} ({nd} days) ===')
        print(f'  1s pivots: {piv:,} ({piv/nd:.0f}/day)')
        print(f'  Trades: {s["n"]:,} ({s["n"]/nd:.1f}/day)')
        print(f'  WR: {s["wr"]:.1f}%  $WR: {s["dollar_wr"]:+.0f}%')
        print(f'  $/trade: ${s["per_trade"]:+.2f}')
        print(f'  Total: ${s["total"]:+,.0f}')
        print(f'  $/day:  ${s["total"]/nd:+,.2f}')
        print(f'  Hold: mean={s["mean_hold_sec"]:.0f}s, median={s["median_hold_sec"]:.0f}s')
        print(f'  Exits: {s["exits"]}')

    p('IS', is_sum, is_d, is_piv)
    p('OOS', oos_sum, oos_d, oos_piv)


if __name__ == '__main__':
    main()
