"""
Physics-exit pivot sim with CHAIN MULTIPLIER (multiple concurrent positions).

Same entry + exit physics as pivot_physics_exit.py at its BASELINE settings
(r_entry=$2 1s zigzag, |1m_z_se|>=0.5 direction, physics-exit signal,
30s sniper, no thesis_broken, no adverse_reg), but allows up to `max_chains`
positions open at once. Each position tracks its own state.

Each 1s bar:
  1. For every open position independently:
     - If in sniper window: update best price, check timeout
     - Else (at each 1m close): check exit signal
  2. Check EOD force-close on all positions.
  3. If a 1s pivot fires AND len(positions) < max_chains: open a new position.

Sweep:
    python tools/pivot_physics_chains.py --sweep   # {1,2,3,4,5}
    python tools/pivot_physics_chains.py --max-chains 3

Output: reports/findings/pivot_physics_chains.md
"""
import os
import sys
import glob
import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pivot_physics_exit import rolling_fit, zigzag_pivots_realtime


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_physics_chains.md'
DOLLAR_PER_POINT = 2.0
REG_WINDOW = 60
EOD_UTC_SECONDS = 20 * 3600 + 55 * 60
FILL_SLIP_TICKS = 1
TICK_SIZE_POINTS = 0.25


@dataclass
class Position:
    entry_ts: int
    entry_price: float
    direction: str
    entry_res_sign: int
    last_reg_pivot_bar_at_entry: int
    sniper_active: bool = False
    sniper_start_ts: Optional[int] = None
    sniper_best_price: Optional[float] = None


def load_day(sec_path, min_path, feat_path):
    df_sec = pd.read_parquet(sec_path).sort_values('timestamp').reset_index(drop=True)
    df_min = pd.read_parquet(min_path).sort_values('timestamp').reset_index(drop=True)
    df_feat = pd.read_parquet(feat_path).sort_values('timestamp').reset_index(drop=True)

    sec = {
        'ts': df_sec['timestamp'].values.astype(np.int64),
        'high': df_sec['high'].values.astype(np.float64),
        'low': df_sec['low'].values.astype(np.float64),
        'close': df_sec['close'].values.astype(np.float64),
    }
    closes_1m = df_min['close'].values.astype(np.float64)
    ts_1m = df_min['timestamp'].values.astype(np.int64)

    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    if '1m_z_se' not in df_feat.columns:
        return None
    res_feat = df_feat['1m_z_se'].values.astype(np.float64)
    idx_sec = np.searchsorted(ts_feat, sec['ts'], side='right') - 1
    idx_sec = np.clip(idx_sec, 0, len(ts_feat) - 1)
    residuals_1s = res_feat[idx_sec]
    idx_1m = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx_1m = np.clip(idx_1m, 0, len(ts_feat) - 1)
    residuals_1m = res_feat[idx_1m]
    return sec, closes_1m, ts_1m, residuals_1s, residuals_1m


def seconds_past_midnight(ts):
    return int(ts) % 86400


def simulate(sec, closes_1m, ts_1m, residuals_1s, residuals_1m,
             r_entry_pts, r_reg_pts, min_res, sniper_sec, max_chains):
    n_sec = len(sec['close'])
    n_min = len(closes_1m)
    trades = []
    fill_slip_pts = FILL_SLIP_TICKS * TICK_SIZE_POINTS

    fitted_1m = rolling_fit(closes_1m, REG_WINDOW)
    reg_pivots = zigzag_pivots_realtime(fitted_1m, r_reg_pts)

    # 1s pivot state (shared across all positions)
    leg_dir = None
    extreme_idx = 0
    extreme_price = sec['close'][0]

    # Multiple positions
    positions: List[Position] = []

    ts_1m_set = dict(zip(ts_1m.tolist(), range(n_min)))

    def last_closed_1m_idx(t_sec):
        return np.searchsorted(ts_1m, t_sec - 60, side='right') - 1

    for T in range(n_sec):
        t_sec = int(sec['ts'][T])
        price = sec['close'][T]
        hi = sec['high'][T]
        lo = sec['low'][T]
        sec_in_day = seconds_past_midnight(t_sec)

        # ── Process each open position ──
        closing = []
        for pi, pos in enumerate(positions):
            # Sniper window
            if pos.sniper_active:
                if pos.direction == 'LONG':
                    if hi > pos.sniper_best_price:
                        pos.sniper_best_price = hi
                else:
                    if lo < pos.sniper_best_price:
                        pos.sniper_best_price = lo
                if t_sec - pos.sniper_start_ts >= sniper_sec:
                    if pos.direction == 'LONG':
                        exit_price = pos.sniper_best_price - fill_slip_pts
                    else:
                        exit_price = pos.sniper_best_price + fill_slip_pts
                    pnl = ((exit_price - pos.entry_price) if pos.direction == 'LONG'
                           else (pos.entry_price - exit_price)) * DOLLAR_PER_POINT
                    trades.append({
                        'entry_ts': pos.entry_ts, 'exit_ts': t_sec,
                        'entry_price': pos.entry_price, 'exit_price': exit_price,
                        'direction': pos.direction, 'pnl': pnl,
                        'exit_reason': 'sniper',
                        'held_sec': t_sec - pos.entry_ts,
                    })
                    closing.append(pi)
                continue

            # EOD force close
            if sec_in_day >= EOD_UTC_SECONDS:
                exit_price = price
                pnl = ((exit_price - pos.entry_price) if pos.direction == 'LONG'
                       else (pos.entry_price - exit_price)) * DOLLAR_PER_POINT
                trades.append({
                    'entry_ts': pos.entry_ts, 'exit_ts': t_sec,
                    'entry_price': pos.entry_price, 'exit_price': exit_price,
                    'direction': pos.direction, 'pnl': pnl,
                    'exit_reason': 'eod',
                    'held_sec': t_sec - pos.entry_ts,
                })
                closing.append(pi)
                continue

            # Exit signal check (on 1m close)
            if t_sec in ts_1m_set:
                just_closed_1m_idx = ts_1m_set[t_sec] - 1
                if just_closed_1m_idx >= 0 and not np.isnan(fitted_1m[just_closed_1m_idx]):
                    res_now = residuals_1m[just_closed_1m_idx]
                    res_sign_flipped = (np.sign(res_now) != pos.entry_res_sign
                                        and np.sign(res_now) != 0)
                    reg_flip = False
                    for k in range(pos.last_reg_pivot_bar_at_entry + 1, just_closed_1m_idx + 1):
                        if k in reg_pivots:
                            pivot_type = reg_pivots[k]
                            if (pos.direction == 'LONG' and pivot_type == 'HIGH') or \
                               (pos.direction == 'SHORT' and pivot_type == 'LOW'):
                                reg_flip = True
                                break
                    if res_sign_flipped and reg_flip:
                        pos.sniper_active = True
                        pos.sniper_start_ts = t_sec
                        pos.sniper_best_price = hi if pos.direction == 'LONG' else lo

        # Remove closed positions (in reverse order to preserve indices)
        for pi in reversed(closing):
            positions.pop(pi)

        # ── 1s pivot tracking (for entries) ──
        pivot_fired = False
        pivot_at = None
        if leg_dir is None:
            if price - extreme_price >= r_entry_pts:
                leg_dir = 'up'
                pivot_at = extreme_idx
                pivot_fired = True
                extreme_idx = T
                extreme_price = price
            elif extreme_price - price >= r_entry_pts:
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
            elif extreme_price - price >= r_entry_pts:
                pivot_at = extreme_idx
                pivot_fired = True
                leg_dir = 'down'
                extreme_idx = T
                extreme_price = price
        else:
            if price < extreme_price:
                extreme_idx = T
                extreme_price = price
            elif price - extreme_price >= r_entry_pts:
                pivot_at = extreme_idx
                pivot_fired = True
                leg_dir = 'up'
                extreme_idx = T
                extreme_price = price

        if pivot_fired and pivot_at is not None and len(positions) < max_chains:
            if sec_in_day >= EOD_UTC_SECONDS:
                continue
            r = float(residuals_1s[pivot_at])
            if np.isnan(r) or abs(r) < min_res:
                continue
            direction = 'LONG' if r < 0 else 'SHORT'
            entry_1m_idx = last_closed_1m_idx(t_sec)
            last_reg_pivot_bar = -1
            for k in range(entry_1m_idx, -1, -1):
                if k in reg_pivots:
                    last_reg_pivot_bar = k
                    break
            positions.append(Position(
                entry_ts=t_sec, entry_price=price, direction=direction,
                entry_res_sign=int(np.sign(r)),
                last_reg_pivot_bar_at_entry=last_reg_pivot_bar,
            ))

    # Close any remaining positions at last 1s bar
    for pos in positions:
        exit_price = sec['close'][-1]
        t_sec = int(sec['ts'][-1])
        pnl = ((exit_price - pos.entry_price) if pos.direction == 'LONG'
               else (pos.entry_price - exit_price)) * DOLLAR_PER_POINT
        trades.append({
            'entry_ts': pos.entry_ts, 'exit_ts': t_sec,
            'entry_price': pos.entry_price, 'exit_price': exit_price,
            'direction': pos.direction, 'pnl': pnl,
            'exit_reason': 'day_end',
            'held_sec': t_sec - pos.entry_ts,
        })

    return trades


def run_pass(paths_1m, r_entry, r_reg, min_res, sniper_sec, max_chains, label):
    r_entry_pts = r_entry / DOLLAR_PER_POINT
    r_reg_pts = r_reg / DOLLAR_PER_POINT
    all_trades = []
    n_days = 0
    for p in tqdm(paths_1m, desc=label, unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        sec_path = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
        feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(sec_path) or not os.path.exists(feat_path):
            continue
        try:
            loaded = load_day(sec_path, p, feat_path)
            if loaded is None:
                continue
            sec, closes_1m, ts_1m, residuals_1s, residuals_1m = loaded
        except Exception:
            continue
        n_days += 1
        trades = simulate(sec, closes_1m, ts_1m, residuals_1s, residuals_1m,
                          r_entry_pts, r_reg_pts, min_res, sniper_sec, max_chains)
        for t in trades:
            t['day'] = day
        all_trades.extend(trades)
    return all_trades, n_days


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
        'median_hold_sec': float(np.median(hold)),
        'max_win': float(pnls.max()),
        'max_loss': float(pnls.min()),
        'exits': dict(exits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-entry', type=float, default=2.0)
    ap.add_argument('--r-reg', type=float, default=4.0)
    ap.add_argument('--min-res', type=float, default=0.5)
    ap.add_argument('--sniper-sec', type=int, default=30)
    ap.add_argument('--max-chains', type=int, default=1)
    ap.add_argument('--sweep', action='store_true',
                    help='Sweep max_chains in {1, 2, 3, 4, 5}')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    chain_values = [1, 2, 3, 4, 5] if args.sweep else [args.max_chains]

    rows = []
    for mc in chain_values:
        print(f'\n--- max_chains={mc} ---')
        is_trades, is_days = run_pass(is_paths, args.r_entry, args.r_reg,
                                       args.min_res, args.sniper_sec,
                                       mc, f'IS mc={mc}')
        oos_trades, oos_days = run_pass(oos_paths, args.r_entry, args.r_reg,
                                         args.min_res, args.sniper_sec,
                                         mc, f'OOS mc={mc}')
        is_s = summarize(is_trades)
        oos_s = summarize(oos_trades)
        if not is_s or not oos_s:
            continue
        is_day = is_s['total'] / is_days if is_days else 0
        oos_day = oos_s['total'] / oos_days if oos_days else 0
        rows.append({
            'mc': mc, 'is_day': is_day, 'oos_day': oos_day,
            'is_n': is_s['n'], 'oos_n': oos_s['n'],
            'is_wr': is_s['wr'], 'oos_wr': oos_s['wr'],
            'is_pt': is_s['per_trade'], 'oos_pt': oos_s['per_trade'],
            'is_maxloss': is_s['max_loss'], 'oos_maxloss': oos_s['max_loss'],
            'is_days': is_days, 'oos_days': oos_days,
        })
        print(f'  IS  ${is_day:+,.0f}/day ({is_s["n"]:,} trades WR {is_s["wr"]:.1f}% '
              f'${is_s["per_trade"]:+.2f}/tr, max_loss ${is_s["max_loss"]:+,.0f})')
        print(f'  OOS ${oos_day:+,.0f}/day ({oos_s["n"]:,} trades WR {oos_s["wr"]:.1f}% '
              f'${oos_s["per_trade"]:+.2f}/tr, max_loss ${oos_s["max_loss"]:+,.0f})')

    if len(rows) > 1:
        print('\n=== SWEEP SUMMARY ===')
        print(f'{"max_chains":>10} {"IS $/day":>10} {"OOS $/day":>11} '
              f'{"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>9} {"OOS $/tr":>10} '
              f'{"IS N":>8} {"OOS N":>8}')
        for r in rows:
            print(f'{r["mc"]:>10} ${r["is_day"]:>+8,.0f} ${r["oos_day"]:>+9,.0f} '
                  f'{r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_pt"]:>+7.2f} ${r["oos_pt"]:>+8.2f} '
                  f'{r["is_n"]:>8,} {r["oos_n"]:>8,}')

    out = [f'# Chain-multiplier sweep', '']
    out.append(f'r_entry=${args.r_entry} r_reg=${args.r_reg} min_res={args.min_res} '
               f'sniper={args.sniper_sec}s')
    out.append('')
    out.append('| max_chains | IS $/day | OOS $/day | IS WR | OOS WR | '
               'IS $/tr | OOS $/tr | IS N | OOS N | Max loss |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        out.append(f'| {r["mc"]} | ${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                   f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                   f'${r["is_pt"]:+.2f} | ${r["oos_pt"]:+.2f} | '
                   f'{r["is_n"]:,} | {r["oos_n"]:,} | '
                   f'${r["is_maxloss"]:+.0f} |')
    out.append('')
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
