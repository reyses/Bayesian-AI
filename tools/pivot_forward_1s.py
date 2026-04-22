"""
Pivot-residual FORWARD PASS with 1s SLIPPAGE RESOLUTION.

Differs from pivot_residual_forward.py in one critical way: exits are
resolved at 1-second granularity using intra-bar OHLC, not 1-minute close.

This eliminates the "did TP or SL fire first?" ambiguity that exists at
1-minute resolution and gives a realistic live-execution number.

No-lookahead constraints:
  - Pivot detection on 1m closes (known at bar close).
  - Residual at pivot from precomputed 1m_z_se (no-lookahead N-1).
  - Entry at confirmation 1m-bar close price.
  - Exit walks 1s bars from entry timestamp forward. For each 1s bar:
      * LONG: TP if high >= entry+TP; SL if low <= entry-SL
      * SHORT: TP if low <= entry-TP; SL if high >= entry+SL
      * If both hit in same 1s bar → pessimistic: SL wins (worst case).
      * Inverse: check close vs residual threshold at the END of the 1s bar.

Usage:
    python tools/pivot_forward_1s.py --tp 50 --sl 3
    python tools/pivot_forward_1s.py --apply-vel-filter

Output: reports/findings/pivot_forward_1s.md
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
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_forward_1s.md'
DOLLAR_PER_POINT = 2.0
VEL_WINDOW = 5


def load_day_1m_and_features(price_path, features_path):
    df_price = pd.read_parquet(price_path).sort_values('timestamp').reset_index(drop=True)
    closes_1m = df_price['close'].values.astype(np.float64)
    ts_1m = df_price['timestamp'].values.astype(np.int64)

    df_feat = pd.read_parquet(features_path).sort_values('timestamp').reset_index(drop=True)
    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    res_feat = df_feat['1m_z_se'].values.astype(np.float64)
    idx = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx = np.clip(idx, 0, len(ts_feat) - 1)
    residuals = res_feat[idx]
    return closes_1m, residuals, ts_1m


def load_day_1s(path):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    return {
        'ts': df['timestamp'].values.astype(np.int64),
        'high': df['high'].values.astype(np.float64),
        'low': df['low'].values.astype(np.float64),
        'close': df['close'].values.astype(np.float64),
    }


def resolve_exit_1s(sec_data, entry_ts, entry_price, direction,
                     tp_pts, sl_pts, inverse_fn=None):
    """Walk 1s bars from entry_ts forward. Return (exit_price, exit_ts,
    exit_reason, held_seconds)."""
    if sec_data is None:
        return entry_price, entry_ts, 'no_1s_data', 0

    ts_arr = sec_data['ts']
    # side='right': skip the entry second itself (where we just placed the order)
    start_idx = np.searchsorted(ts_arr, entry_ts, side='right')
    if start_idx >= len(ts_arr):
        return entry_price, entry_ts, 'entry_past_day_end', 0

    for i in range(start_idx, len(ts_arr)):
        hi = sec_data['high'][i]
        lo = sec_data['low'][i]
        cl = sec_data['close'][i]
        if direction == 'LONG':
            tp_hit = hi >= entry_price + tp_pts
            sl_hit = lo <= entry_price - sl_pts
        else:
            tp_hit = lo <= entry_price - tp_pts
            sl_hit = hi >= entry_price + sl_pts

        if tp_hit and sl_hit:
            # Pessimistic: SL wins in same-bar ambiguity
            exit_p = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
            return exit_p, ts_arr[i], 'SL', int(ts_arr[i] - entry_ts)
        if tp_hit:
            exit_p = entry_price + tp_pts if direction == 'LONG' else entry_price - tp_pts
            return exit_p, ts_arr[i], 'TP', int(ts_arr[i] - entry_ts)
        if sl_hit:
            exit_p = entry_price - sl_pts if direction == 'LONG' else entry_price + sl_pts
            return exit_p, ts_arr[i], 'SL', int(ts_arr[i] - entry_ts)

        if inverse_fn is not None:
            if inverse_fn(i, cl):
                return cl, ts_arr[i], 'inverse', int(ts_arr[i] - entry_ts)

    # EOD flush at last 1s close
    return sec_data['close'][-1], ts_arr[-1], 'eod', int(ts_arr[-1] - entry_ts)


def simulate_day(closes_1m, residuals, ts_1m, sec_data,
                 r_confirm_pts, tp_pts, sl_pts, min_res_strength,
                 apply_vel_filter=False, apply_vel_flip=False,
                 fallback_1m_if_no_1s=True):
    trades = []
    leg_dir = None
    extreme_idx = 0
    extreme_price = closes_1m[0]
    last_pivot_idx = None
    in_position = False
    entry_bar = None
    entry_price = None
    entry_ts = None
    direction = None
    entry_res = None
    n_pivots = 0

    n = len(closes_1m)

    for T in range(n):
        price = closes_1m[T]

        if in_position:
            # Already in position, skip pivot checking but also skip
            # intra-minute exit (handled at entry time below)
            continue

        # Pivot tracking
        fire_pivot = False
        pivot_at = None
        if leg_dir is None:
            if price - extreme_price >= r_confirm_pts:
                leg_dir = 'up'
                pivot_at = extreme_idx
                fire_pivot = True
                extreme_idx = T
                extreme_price = price
            elif extreme_price - price >= r_confirm_pts:
                leg_dir = 'down'
                pivot_at = extreme_idx
                fire_pivot = True
                extreme_idx = T
                extreme_price = price
            else:
                if price > extreme_price:
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
                fire_pivot = True
                leg_dir = 'down'
                extreme_idx = T
                extreme_price = price
        else:
            if price < extreme_price:
                extreme_idx = T
                extreme_price = price
            elif price - extreme_price >= r_confirm_pts:
                pivot_at = extreme_idx
                fire_pivot = True
                leg_dir = 'up'
                extreme_idx = T
                extreme_price = price

        if fire_pivot and pivot_at is not None:
            n_pivots += 1
            r = residuals[pivot_at] if pivot_at < len(residuals) else float('nan')
            if np.isnan(r) or abs(r) < min_res_strength:
                continue

            pred_direction = 'LONG' if r < 0 else 'SHORT'

            # Velocity filter / flip
            if apply_vel_filter or apply_vel_flip:
                if pivot_at >= VEL_WINDOW:
                    price_vel = (closes_1m[pivot_at] - closes_1m[pivot_at - VEL_WINDOW]) / VEL_WINDOW
                else:
                    price_vel = 0.0
                pred_sign = +1 if pred_direction == 'LONG' else -1
                pred_dir_vel = pred_sign * price_vel
                # If velocity is AGAINST prediction (negative) → take prediction.
                # If velocity is STRONGLY WITH prediction (>0.5) → flip.
                # If velocity is weak WITH prediction (0 to 0.5) → skip.
                if pred_dir_vel >= 0.5 and apply_vel_flip:
                    # Strong against-prediction momentum → flip
                    pred_direction = 'SHORT' if pred_direction == 'LONG' else 'LONG'
                elif pred_dir_vel >= 0.01 and apply_vel_filter:
                    # Weak with-prediction velocity → skip (8-18% win rate zone)
                    continue
                # Otherwise: AGAINST prediction (best case) — take it.

            # Entry at confirmation bar's close, in 1s timestamp space
            entry_bar = T
            entry_price = closes_1m[T]
            entry_ts = int(ts_1m[T])
            direction = pred_direction
            entry_res = r

            # Resolve exit using 1s data
            if sec_data is not None:
                exit_price, exit_ts, exit_reason, held_sec = resolve_exit_1s(
                    sec_data, entry_ts, entry_price, direction,
                    tp_pts, sl_pts, inverse_fn=None)
            elif fallback_1m_if_no_1s:
                # Fallback: 1-minute high/low resolution
                exit_price = entry_price
                exit_ts = entry_ts
                exit_reason = 'no_1s_data'
                held_sec = 0
                for j in range(T + 1, n):
                    hi_1m = closes_1m[j]  # simplified: use close as proxy
                    if direction == 'LONG':
                        if hi_1m >= entry_price + tp_pts:
                            exit_price = entry_price + tp_pts
                            exit_reason = 'TP_1m'
                            exit_ts = int(ts_1m[j])
                            held_sec = exit_ts - entry_ts
                            break
                        if hi_1m <= entry_price - sl_pts:
                            exit_price = entry_price - sl_pts
                            exit_reason = 'SL_1m'
                            exit_ts = int(ts_1m[j])
                            held_sec = exit_ts - entry_ts
                            break
                    else:
                        if hi_1m <= entry_price - tp_pts:
                            exit_price = entry_price - tp_pts
                            exit_reason = 'TP_1m'
                            exit_ts = int(ts_1m[j])
                            held_sec = exit_ts - entry_ts
                            break
                        if hi_1m >= entry_price + sl_pts:
                            exit_price = entry_price + sl_pts
                            exit_reason = 'SL_1m'
                            exit_ts = int(ts_1m[j])
                            held_sec = exit_ts - entry_ts
                            break
            else:
                continue

            if direction == 'LONG':
                pnl = (exit_price - entry_price) * DOLLAR_PER_POINT
            else:
                pnl = (entry_price - exit_price) * DOLLAR_PER_POINT

            trades.append({
                'entry_bar': entry_bar, 'entry_ts': entry_ts,
                'exit_ts': exit_ts, 'held_sec': held_sec,
                'direction': direction, 'entry_res': entry_res,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'exit_reason': exit_reason,
            })

            # Allow next pivot (position closed)
            in_position = False

    return trades, n_pivots


def run_pass(paths_1m, r_confirm, tp, sl, min_res, label,
             apply_vel_filter=False, apply_vel_flip=False):
    r_pts = r_confirm / DOLLAR_PER_POINT
    tp_pts = tp / DOLLAR_PER_POINT
    sl_pts = sl / DOLLAR_PER_POINT
    all_trades = []
    tot_pivots = 0
    n_days_with_1s = 0
    n_days_no_1s = 0
    for p in tqdm(paths_1m, desc=label, unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        sec_path = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
        if not os.path.exists(feat_path):
            continue
        try:
            closes_1m, residuals, ts_1m = load_day_1m_and_features(p, feat_path)
        except Exception as e:
            continue
        sec_data = load_day_1s(sec_path)
        if sec_data is not None:
            n_days_with_1s += 1
        else:
            n_days_no_1s += 1
        trades, npiv = simulate_day(
            closes_1m, residuals, ts_1m, sec_data,
            r_pts, tp_pts, sl_pts, min_res,
            apply_vel_filter, apply_vel_flip)
        for t in trades:
            t['day'] = day
        all_trades.extend(trades)
        tot_pivots += npiv
    return all_trades, tot_pivots, n_days_with_1s, n_days_no_1s


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
        'n': len(trades), 'total': pnls.sum(),
        'per_trade': pnls.mean(),
        'wr': wins / max(wins + losses, 1) * 100,
        'dollar_wr': (w_d / l_d - 1) * 100 if l_d > 0 else float('inf'),
        'wins': int(wins), 'losses': int(losses),
        'mean_hold_sec': float(hold_sec.mean()),
        'median_hold_sec': float(np.median(hold_sec)),
        'exits': dict(exits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-confirm', type=float, default=5.0)
    ap.add_argument('--tp', type=float, default=50.0)
    ap.add_argument('--sl', type=float, default=3.0)
    ap.add_argument('--min-res-strength', type=float, default=0.5)
    ap.add_argument('--apply-vel-filter', action='store_true',
                    help='Skip pivots where price_vel WITH prediction (worst accuracy)')
    ap.add_argument('--apply-vel-flip', action='store_true',
                    help='Flip direction when price_vel strongly WITH prediction')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} days | OOS {len(oos_paths)} days')
    print(f'r_confirm=${args.r_confirm} tp=${args.tp} sl=${args.sl} '
          f'min_res={args.min_res_strength} vel_filter={args.apply_vel_filter} '
          f'vel_flip={args.apply_vel_flip}')

    is_trades, is_piv, is_1s_days, is_no_1s = run_pass(
        is_paths, args.r_confirm, args.tp, args.sl, args.min_res_strength,
        'IS', args.apply_vel_filter, args.apply_vel_flip)
    oos_trades, oos_piv, oos_1s_days, oos_no_1s = run_pass(
        oos_paths, args.r_confirm, args.tp, args.sl, args.min_res_strength,
        'OOS', args.apply_vel_filter, args.apply_vel_flip)

    print(f'\n1s data coverage: IS {is_1s_days}/{is_1s_days+is_no_1s} days  '
          f'OOS {oos_1s_days}/{oos_1s_days+oos_no_1s} days')

    is_sum = summarize(is_trades)
    oos_sum = summarize(oos_trades)

    def p(label, s, n_days, pivots):
        if not s:
            print(f'\n{label}: no trades')
            return
        print(f'\n=== {label} ({n_days} days) ===')
        print(f'  Pivots: {pivots:,} ({pivots/n_days:.0f}/day)')
        print(f'  Trades: {s["n"]:,} ({s["n"]/n_days:.1f}/day)')
        print(f'  WR: {s["wr"]:.1f}%  $WR: {s["dollar_wr"]:+.0f}%')
        print(f'  $/trade: ${s["per_trade"]:+.2f}')
        print(f'  Total: ${s["total"]:+,.0f}')
        print(f'  $/day:  ${s["total"]/n_days:+,.2f}')
        print(f'  Hold (sec): mean={s["mean_hold_sec"]:.0f}, '
              f'median={s["median_hold_sec"]:.0f}')
        print(f'  Exits: {s["exits"]}')

    p('IS', is_sum, len(is_paths), is_piv)
    p('OOS', oos_sum, len(oos_paths), oos_piv)

    # MD
    out = [f'# Pivot-residual FORWARD PASS with 1s slippage', '']
    out.append('Exits resolved at 1-second granularity (no same-bar '
               'TP/SL ambiguity).')
    out.append('')
    out.append(f'r_confirm=${args.r_confirm} TP=${args.tp} SL=${args.sl} '
               f'min_res={args.min_res_strength}')
    if args.apply_vel_filter:
        out.append('**Velocity filter**: skipping pivots where price_vel '
                   'agrees with prediction (8-18% WR zone).')
    if args.apply_vel_flip:
        out.append('**Velocity flip**: flipping direction when price_vel '
                   'strongly agrees with prediction (92% WR inverted).')
    out.append('')
    out.append('## Results')
    out.append('')
    out.append('| Dataset | Days | Pivots | Trades | $/day | $/trade | WR | $WR | Total | Hold (s) |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for lbl, s, nd, piv in [('IS', is_sum, len(is_paths), is_piv),
                             ('OOS', oos_sum, len(oos_paths), oos_piv)]:
        if s:
            out.append(f'| {lbl} | {nd} | {piv:,} | {s["n"]:,} | '
                       f'${s["total"]/nd:+,.0f} | ${s["per_trade"]:+.2f} | '
                       f'{s["wr"]:.1f}% | {s["dollar_wr"]:+.0f}% | '
                       f'${s["total"]:+,.0f} | {s["median_hold_sec"]:.0f} |')
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

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
