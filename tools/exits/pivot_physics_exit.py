"""
Physics-only exit simulator — entry on 1s pivot, exit on 1m signal + 30s sniper.

No stop-loss. No early cuts. Pure signal measurement: let the trade ride from
one regression-mean-cross-and-reg-direction-flip event to the next.

Entry (1s precision):
  - 1s zigzag pivot (r_confirm = $2 default)
  - Residual (1m_z_se, nearest prior 5s feature) sign = direction
  - Skip if |residual| < min_res_strength
  - Enter at 1s pivot bar close

Exit signal (1m decision):
  - At each 1m bar close while in position, check:
      1. Regression direction has flipped since entry (1m regression zigzag
         pivot fired in opposite direction to entry)
      2. Price has crossed regression mean since entry (residual sign flipped
         from entry's sign)
  - When BOTH true → signal "exit window open"

Exit execution (1s sniper):
  - Start 30s window from signal moment
  - Track running_max (LONG) / running_min (SHORT) of 1s highs/lows in window
  - At end of 30s: exit at best extreme minus 1 tick slip
  - Same procedure whether winning or losing (measure raw signal quality)

End-of-day:
  - Force-close at 20:55 UTC (5 min before 21:00 UTC maintenance)
  - Trade's exit price = last 1s close at or before cutoff

Usage:
    python tools/pivot_physics_exit.py --r-entry 2 --r-reg 8
    python tools/pivot_physics_exit.py --sniper-sec 30 --min-res 0.5

Output: reports/findings/pivot_physics_exit.md
"""
import os
import sys
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_physics_exit.md'
DOLLAR_PER_POINT = 2.0
REG_WINDOW = 60     # 1m bars in rolling regression fit
EOD_UTC_SECONDS = 20 * 3600 + 55 * 60   # 20:55 UTC = 75300 seconds past midnight
FILL_SLIP_TICKS = 1
TICK_SIZE_POINTS = 0.25    # 1 tick = 0.25 points on MNQ


def load_day(sec_path, min_path, feat_path):
    """Load 1s OHLC, 1m closes+ts, and 1m_z_se feature + sampled residuals."""
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

    # Residual at every 1s timestamp (nearest prior 5s feature)
    idx_sec = np.searchsorted(ts_feat, sec['ts'], side='right') - 1
    idx_sec = np.clip(idx_sec, 0, len(ts_feat) - 1)
    residuals_1s = res_feat[idx_sec]

    # Residual at every 1m timestamp (similarly)
    idx_1m = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx_1m = np.clip(idx_1m, 0, len(ts_feat) - 1)
    residuals_1m = res_feat[idx_1m]

    return sec, closes_1m, ts_1m, residuals_1s, residuals_1m


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


def zigzag_pivots_realtime(fitted_or_closes, r_threshold):
    """Return mapping {bar_idx: 'HIGH'|'LOW'} of pivots as they confirm in
    chronological order. Each pivot confirms at the bar where retracement
    crosses r_threshold."""
    n = len(fitted_or_closes)
    pivots = {}
    if n < 2:
        return pivots
    leg_dir = None
    extreme_idx = 0
    extreme_val = fitted_or_closes[0]
    for i in range(1, n):
        v = fitted_or_closes[i]
        if np.isnan(v):
            continue
        if np.isnan(extreme_val):
            extreme_idx = i
            extreme_val = v
            continue
        if leg_dir is None:
            if v - extreme_val >= r_threshold:
                leg_dir = 'up'
                pivots[i] = 'LOW'     # pivot type at extreme_idx was LOW
                pivots[extreme_idx] = 'LOW'
                extreme_idx = i
                extreme_val = v
            elif extreme_val - v >= r_threshold:
                leg_dir = 'down'
                pivots[i] = 'HIGH'
                pivots[extreme_idx] = 'HIGH'
                extreme_idx = i
                extreme_val = v
            elif v > extreme_val:
                extreme_idx = i
                extreme_val = v
            elif v < extreme_val:
                extreme_idx = i
                extreme_val = v
        elif leg_dir == 'up':
            if v > extreme_val:
                extreme_idx = i
                extreme_val = v
            elif extreme_val - v >= r_threshold:
                pivots[i] = 'HIGH'
                pivots[extreme_idx] = 'HIGH'
                leg_dir = 'down'
                extreme_idx = i
                extreme_val = v
        else:
            if v < extreme_val:
                extreme_idx = i
                extreme_val = v
            elif v - extreme_val >= r_threshold:
                pivots[i] = 'LOW'
                pivots[extreme_idx] = 'LOW'
                leg_dir = 'up'
                extreme_idx = i
                extreme_val = v
    return pivots


def seconds_past_midnight(ts):
    """Return seconds past UTC midnight for a unix timestamp."""
    return int(ts) % 86400


def simulate_day(sec, closes_1m, ts_1m, residuals_1s, residuals_1m,
                 r_entry_pts, r_reg_pts, min_res_strength, sniper_sec,
                 eod_utc_sec=EOD_UTC_SECONDS,
                 thesis_broken=True, adverse_bars=5):
    """Run the physics-exit strategy on one day.

    Exit conditions (first-wins):
      1. PHYSICS SIGNAL: reg direction flipped AND residual sign flipped
         → sniper window, take best extreme
      2. THESIS_BROKEN: new reg pivot forms at a level worse than entry
         pivot (lower for LONG, higher for SHORT) → exit immediately
      3. ADVERSE_REG: regression has moved against us for `adverse_bars`
         consecutive 1m bars → exit immediately
      4. EOD at 20:55 UTC
    """
    n_sec = len(sec['close'])
    n_min = len(closes_1m)
    trades = []

    # Precompute 1m regression fitted values and pivot timeline
    fitted_1m = rolling_fit(closes_1m, REG_WINDOW)
    reg_pivots = zigzag_pivots_realtime(fitted_1m, r_reg_pts)

    # 1s pivot tracking state (for entries)
    leg_dir = None
    extreme_idx = 0
    extreme_price = sec['close'][0]

    # Position state
    in_position = False
    entry_price = None
    entry_ts = None
    entry_dir = None
    entry_res_sign = None
    entry_1m_idx = None   # 1m bar index AT OR AFTER entry
    last_reg_pivot_bar_at_entry = None
    entry_pivot_fit_level = None   # fitted[last_reg_pivot_bar_at_entry]
    last_1m_checked_idx = -1
    adverse_streak = 0
    # Sniper state
    sniper_active = False
    sniper_start_ts = None
    sniper_best_price = None

    fill_slip_pts = FILL_SLIP_TICKS * TICK_SIZE_POINTS

    # Map ts_1m to index for fast 1m-close detection while walking 1s
    ts_1m_set = dict(zip(ts_1m.tolist(), range(n_min)))

    # Helper: compute the last 1m bar index at or before a given 1s timestamp
    def last_closed_1m_idx(t_sec):
        # 1m bar at ts T closes at T+60. So "last closed" is largest ts_1m[k]
        # where ts_1m[k] + 60 <= t_sec
        idx = np.searchsorted(ts_1m, t_sec - 60, side='right') - 1
        if idx < 0:
            return -1
        return idx

    for T in range(n_sec):
        t_sec = int(sec['ts'][T])
        price = sec['close'][T]
        hi = sec['high'][T]
        lo = sec['low'][T]
        sec_in_day = seconds_past_midnight(t_sec)

        # ── SNIPER WINDOW (highest priority once active) ──
        if sniper_active:
            # Update running extreme
            if entry_dir == 'LONG':
                if hi > sniper_best_price:
                    sniper_best_price = hi
            else:
                if lo < sniper_best_price:
                    sniper_best_price = lo
            # Window timeout?
            if t_sec - sniper_start_ts >= sniper_sec:
                # Exit at best_price minus slip
                if entry_dir == 'LONG':
                    exit_price = sniper_best_price - fill_slip_pts
                else:
                    exit_price = sniper_best_price + fill_slip_pts
                pnl = (exit_price - entry_price if entry_dir == 'LONG'
                       else entry_price - exit_price) * DOLLAR_PER_POINT
                trades.append({
                    'entry_ts': entry_ts, 'exit_ts': t_sec,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'direction': entry_dir, 'pnl': pnl,
                    'exit_reason': 'sniper',
                    'held_sec': t_sec - entry_ts,
                })
                in_position = False
                sniper_active = False
            continue    # stay in sniper mode until window ends

        # ── EOD FORCE CLOSE ──
        if in_position and sec_in_day >= eod_utc_sec:
            exit_price = price
            pnl = (exit_price - entry_price if entry_dir == 'LONG'
                   else entry_price - exit_price) * DOLLAR_PER_POINT
            trades.append({
                'entry_ts': entry_ts, 'exit_ts': t_sec,
                'entry_price': entry_price, 'exit_price': exit_price,
                'direction': entry_dir, 'pnl': pnl,
                'exit_reason': 'eod',
                'held_sec': t_sec - entry_ts,
            })
            in_position = False
            continue

        # ── EXIT SIGNAL CHECK (on 1m-bar-close events) ──
        if in_position and t_sec in ts_1m_set:
            just_closed_1m_idx = ts_1m_set[t_sec] - 1
            if just_closed_1m_idx >= 0 and just_closed_1m_idx > last_1m_checked_idx \
                    and not np.isnan(fitted_1m[just_closed_1m_idx]):
                last_1m_checked_idx = just_closed_1m_idx

                # ── EXIT 2: THESIS_BROKEN ──
                # Did a new reg pivot form at a level WORSE than entry pivot?
                if thesis_broken and entry_pivot_fit_level is not None:
                    for k in range(last_reg_pivot_bar_at_entry + 1, just_closed_1m_idx + 1):
                        if k in reg_pivots:
                            pivot_type = reg_pivots[k]
                            level = fitted_1m[k]
                            if np.isnan(level):
                                continue
                            # LONG entry at a LOW pivot at level L_entry.
                            # New LOW pivot below L_entry = thesis broken.
                            # (Same sign pivot type going deeper.)
                            if entry_dir == 'LONG' and pivot_type == 'LOW' \
                                    and level < entry_pivot_fit_level:
                                # Thesis broken — exit immediately
                                exit_price = price
                                pnl = (exit_price - entry_price) * DOLLAR_PER_POINT
                                trades.append({
                                    'entry_ts': entry_ts, 'exit_ts': t_sec,
                                    'entry_price': entry_price, 'exit_price': exit_price,
                                    'direction': entry_dir, 'pnl': pnl,
                                    'exit_reason': 'thesis_broken',
                                    'held_sec': t_sec - entry_ts,
                                })
                                in_position = False
                                break
                            if entry_dir == 'SHORT' and pivot_type == 'HIGH' \
                                    and level > entry_pivot_fit_level:
                                exit_price = price
                                pnl = (entry_price - exit_price) * DOLLAR_PER_POINT
                                trades.append({
                                    'entry_ts': entry_ts, 'exit_ts': t_sec,
                                    'entry_price': entry_price, 'exit_price': exit_price,
                                    'direction': entry_dir, 'pnl': pnl,
                                    'exit_reason': 'thesis_broken',
                                    'held_sec': t_sec - entry_ts,
                                })
                                in_position = False
                                break
                    if not in_position:
                        continue

                # ── EXIT 3: ADVERSE_REG (5 consecutive adverse bars) ──
                if adverse_bars > 0 and just_closed_1m_idx >= 1:
                    prev_fit = fitted_1m[just_closed_1m_idx - 1]
                    cur_fit = fitted_1m[just_closed_1m_idx]
                    if not np.isnan(prev_fit) and not np.isnan(cur_fit):
                        slope_sign = 1 if cur_fit > prev_fit else (-1 if cur_fit < prev_fit else 0)
                        # For LONG, adverse = slope going DOWN.
                        # For SHORT, adverse = slope going UP.
                        adverse = (entry_dir == 'LONG' and slope_sign < 0) or \
                                  (entry_dir == 'SHORT' and slope_sign > 0)
                        if adverse:
                            adverse_streak += 1
                        else:
                            adverse_streak = 0
                        if adverse_streak >= adverse_bars:
                            exit_price = price
                            pnl = (exit_price - entry_price if entry_dir == 'LONG'
                                   else entry_price - exit_price) * DOLLAR_PER_POINT
                            trades.append({
                                'entry_ts': entry_ts, 'exit_ts': t_sec,
                                'entry_price': entry_price, 'exit_price': exit_price,
                                'direction': entry_dir, 'pnl': pnl,
                                'exit_reason': 'adverse_reg',
                                'held_sec': t_sec - entry_ts,
                            })
                            in_position = False
                            continue

                # ── EXIT 1: PHYSICS SIGNAL (reg flip + residual flip) ──
                res_now = residuals_1m[just_closed_1m_idx]
                res_sign_flipped = (np.sign(res_now) != entry_res_sign
                                    and np.sign(res_now) != 0)
                reg_flip = False
                for k in range(last_reg_pivot_bar_at_entry + 1, just_closed_1m_idx + 1):
                    if k in reg_pivots:
                        pivot_type = reg_pivots[k]
                        if (entry_dir == 'LONG' and pivot_type == 'HIGH') or \
                           (entry_dir == 'SHORT' and pivot_type == 'LOW'):
                            reg_flip = True
                            break

                if res_sign_flipped and reg_flip:
                    sniper_active = True
                    sniper_start_ts = t_sec
                    sniper_best_price = hi if entry_dir == 'LONG' else lo
                    continue

        # ── 1s PIVOT TRACKING (for entries) ──
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

        if pivot_fired and pivot_at is not None and not in_position:
            if sec_in_day >= eod_utc_sec:
                continue   # don't enter new trades near EOD
            r = float(residuals_1s[pivot_at])
            if np.isnan(r) or abs(r) < min_res_strength:
                continue
            direction = 'LONG' if r < 0 else 'SHORT'
            entry_price = price
            entry_ts = t_sec
            entry_dir = direction
            entry_res_sign = np.sign(r)
            # Find last reg pivot bar AT OR BEFORE entry
            entry_1m_idx = last_closed_1m_idx(t_sec)
            last_reg_pivot_bar_at_entry = -1
            entry_pivot_fit_level = None
            for k in range(entry_1m_idx, -1, -1):
                if k in reg_pivots:
                    last_reg_pivot_bar_at_entry = k
                    if not np.isnan(fitted_1m[k]):
                        entry_pivot_fit_level = float(fitted_1m[k])
                    break
            last_1m_checked_idx = entry_1m_idx - 1
            adverse_streak = 0
            in_position = True

    # Close any open position at last 1s bar
    if in_position:
        exit_price = sec['close'][-1]
        t_sec = int(sec['ts'][-1])
        pnl = (exit_price - entry_price if entry_dir == 'LONG'
               else entry_price - exit_price) * DOLLAR_PER_POINT
        trades.append({
            'entry_ts': entry_ts, 'exit_ts': t_sec,
            'entry_price': entry_price, 'exit_price': exit_price,
            'direction': entry_dir, 'pnl': pnl,
            'exit_reason': 'day_end',
            'held_sec': t_sec - entry_ts,
        })

    return trades


def run_pass(paths_1m, r_entry, r_reg, min_res, sniper_sec, label,
             thesis_broken=True, adverse_bars=5):
    r_entry_pts = r_entry / DOLLAR_PER_POINT
    r_reg_pts = r_reg / DOLLAR_PER_POINT
    all_trades = []
    n_days_used = 0
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
        except Exception as e:
            continue
        n_days_used += 1
        trades = simulate_day(sec, closes_1m, ts_1m, residuals_1s, residuals_1m,
                               r_entry_pts, r_reg_pts, min_res, sniper_sec,
                               thesis_broken=thesis_broken,
                               adverse_bars=adverse_bars)
        for t in trades:
            t['day'] = day
        all_trades.extend(trades)
    return all_trades, n_days_used


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
        'n': len(trades),
        'total': float(pnls.sum()),
        'per_trade': float(pnls.mean()),
        'wr': wins / max(wins + losses, 1) * 100,
        'dollar_wr': (w_d / l_d - 1) * 100 if l_d > 0 else float('inf'),
        'wins': int(wins), 'losses': int(losses),
        'mean_hold_sec': float(hold.mean()),
        'median_hold_sec': float(np.median(hold)),
        'p90_hold_sec': float(np.percentile(hold, 90)),
        'exits': dict(exits),
        'max_win': float(pnls.max()),
        'max_loss': float(pnls.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-entry', type=float, default=2.0,
                    help='1s entry zigzag retracement in $ (default 2)')
    ap.add_argument('--r-reg', type=float, default=8.0,
                    help='1m regression zigzag threshold in $ (default 8)')
    ap.add_argument('--min-res', type=float, default=0.5,
                    help='Min |1m_z_se| for entry (default 0.5)')
    ap.add_argument('--sniper-sec', type=int, default=30,
                    help='Sniper window duration in seconds (default 30)')
    ap.add_argument('--no-thesis-broken', action='store_true',
                    help='Disable thesis-broken exit')
    ap.add_argument('--adverse-bars', type=int, default=5,
                    help='Consecutive adverse-reg 1m bars to trigger exit (0 = disabled)')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} | OOS {len(oos_paths)} days')
    print(f'Config: r_entry=${args.r_entry}  r_reg=${args.r_reg}  '
          f'min_res={args.min_res}  sniper={args.sniper_sec}s')

    is_trades, is_days = run_pass(is_paths, args.r_entry, args.r_reg,
                                   args.min_res, args.sniper_sec, 'IS',
                                   thesis_broken=not args.no_thesis_broken,
                                   adverse_bars=args.adverse_bars)
    oos_trades, oos_days = run_pass(oos_paths, args.r_entry, args.r_reg,
                                      args.min_res, args.sniper_sec, 'OOS',
                                      thesis_broken=not args.no_thesis_broken,
                                      adverse_bars=args.adverse_bars)

    is_sum = summarize(is_trades)
    oos_sum = summarize(oos_trades)

    def p(label, s, nd):
        if not s:
            print(f'\n{label}: no trades')
            return
        print(f'\n=== {label} ({nd} days) ===')
        print(f'  Trades: {s["n"]:,} ({s["n"]/nd:.1f}/day)')
        print(f'  WR: {s["wr"]:.1f}%  ({s["wins"]:,}W / {s["losses"]:,}L)')
        print(f'  $WR: {s["dollar_wr"]:+.0f}%')
        print(f'  $/trade: ${s["per_trade"]:+.2f}')
        print(f'  Total: ${s["total"]:+,.0f}')
        print(f'  $/day:  ${s["total"]/nd:+,.2f}')
        print(f'  Hold: mean={s["mean_hold_sec"]:.0f}s  '
              f'median={s["median_hold_sec"]:.0f}s  '
              f'p90={s["p90_hold_sec"]:.0f}s')
        print(f'  Max win: ${s["max_win"]:+,.0f}  '
              f'max loss: ${s["max_loss"]:+,.0f}')
        print(f'  Exits: {s["exits"]}')

    p('IS', is_sum, is_days)
    p('OOS', oos_sum, oos_days)

    out = [f'# Physics-only exit simulator', '']
    out.append(f'r_entry=${args.r_entry}  r_reg=${args.r_reg}  '
               f'min_res={args.min_res}  sniper={args.sniper_sec}s')
    out.append('')
    out.append('Entry: 1s zigzag pivot + residual direction.')
    out.append('Exit signal: 1m regression direction flipped AND residual sign flipped vs entry.')
    out.append('Exit execution: 30-second sniper window, take running extreme - 1 tick slip.')
    out.append('No stop-loss. EOD force-close at 20:55 UTC.')
    out.append('')
    out.append('| Dataset | Days | Trades | $/day | $/trade | WR | $WR | Mean hold | p90 hold | Max win | Max loss |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for lbl, s, nd in [('IS', is_sum, is_days), ('OOS', oos_sum, oos_days)]:
        if s:
            out.append(f'| {lbl} | {nd} | {s["n"]:,} | ${s["total"]/nd:+,.0f} | '
                       f'${s["per_trade"]:+.2f} | {s["wr"]:.1f}% | '
                       f'{s["dollar_wr"]:+.0f}% | {s["mean_hold_sec"]:.0f}s | '
                       f'{s["p90_hold_sec"]:.0f}s | ${s["max_win"]:+,.0f} | '
                       f'${s["max_loss"]:+,.0f} |')
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
