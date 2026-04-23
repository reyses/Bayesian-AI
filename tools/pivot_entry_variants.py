"""
Test three entry-side variants with identical physics-exit logic.

Baseline for comparison: the existing pivot_physics_exit defaults
(r_entry=$2 zigzag, |res|>=0.5, sniper=30s, no adverse, no thesis_broken):
  IS: +$321/day, OOS: +$503/day.

Variants:
  A — FILTERS: current zigzag ($2 retrace) + require wick AND vol at pivot bar
      • 1m_wick_ratio >= 0.30 (moderate rejection)
      • 1m_vol_rel >= 1.3 (elevated volume)
  B — TIGHT: reduce zigzag retrace threshold to $1 (catch pivots sooner)
  C — RESIDUAL: no zigzag; enter immediately when |1m_z_se| >= 1.5 at a
      1s bar (after not being in position). No retracement required.

All variants keep: physics-only exit (reg flip + residual flip) + 30s sniper + EOD.

Usage:
    python tools/pivot_entry_variants.py

Output: reports/findings/pivot_entry_variants.md
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
from tools.pivot_physics_exit import rolling_fit, zigzag_pivots_realtime


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
ATLAS_1S_DIR = 'DATA/ATLAS/1s'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_entry_variants.md'
DOLLAR_PER_POINT = 2.0
REG_WINDOW = 60
EOD_UTC_SECONDS = 20 * 3600 + 55 * 60
FILL_SLIP_TICKS = 1
TICK_SIZE_POINTS = 0.25


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
    wick_feat = df_feat['1m_wick_ratio'].values.astype(np.float64) \
        if '1m_wick_ratio' in df_feat.columns else np.zeros(len(df_feat))
    vol_feat = df_feat['1m_vol_rel'].values.astype(np.float64) \
        if '1m_vol_rel' in df_feat.columns else np.zeros(len(df_feat))

    idx_sec = np.searchsorted(ts_feat, sec['ts'], side='right') - 1
    idx_sec = np.clip(idx_sec, 0, len(ts_feat) - 1)
    residuals_1s = res_feat[idx_sec]
    wick_1s = wick_feat[idx_sec]
    vol_1s = vol_feat[idx_sec]

    idx_1m = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx_1m = np.clip(idx_1m, 0, len(ts_feat) - 1)
    residuals_1m = res_feat[idx_1m]

    return sec, closes_1m, ts_1m, residuals_1s, residuals_1m, wick_1s, vol_1s


def seconds_past_midnight(ts):
    return int(ts) % 86400


def simulate(sec, closes_1m, ts_1m, residuals_1s, residuals_1m, wick_1s, vol_1s,
             entry_mode, r_entry_pts, min_res, res_trigger,
             wick_min, vol_min, sniper_sec,
             r_reg_pts=4.0, eod_utc_sec=EOD_UTC_SECONDS):
    """Run strategy with variant entry logic + standard physics exit.

    entry_mode: 'filters' | 'tight' | 'residual'
      filters: zigzag ($2) + wick + vol filters
      tight:   zigzag ($1 typically) only
      residual: |res| >= res_trigger, no zigzag
    """
    n_sec = len(sec['close'])
    n_min = len(closes_1m)
    trades = []

    fitted_1m = rolling_fit(closes_1m, REG_WINDOW)
    reg_pivots = zigzag_pivots_realtime(fitted_1m, r_reg_pts)

    # 1s pivot state
    leg_dir = None
    extreme_idx = 0
    extreme_price = sec['close'][0]

    # Position state
    in_position = False
    entry_price = None
    entry_ts = None
    entry_dir = None
    entry_res_sign = None
    last_reg_pivot_bar_at_entry = None

    sniper_active = False
    sniper_start_ts = None
    sniper_best_price = None

    # For residual mode, track whether |res| is currently above threshold
    res_above_trigger = False
    fill_slip_pts = FILL_SLIP_TICKS * TICK_SIZE_POINTS
    ts_1m_set = dict(zip(ts_1m.tolist(), range(n_min)))

    def last_closed_1m_idx(t_sec):
        idx = np.searchsorted(ts_1m, t_sec - 60, side='right') - 1
        return idx

    def try_enter(T, direction, price, t_sec):
        nonlocal in_position, entry_price, entry_ts, entry_dir
        nonlocal entry_res_sign, last_reg_pivot_bar_at_entry
        sec_in_day = seconds_past_midnight(t_sec)
        if sec_in_day >= eod_utc_sec:
            return
        entry_price = price
        entry_ts = t_sec
        entry_dir = direction
        entry_res_sign = -1 if direction == 'LONG' else +1
        entry_1m_idx = last_closed_1m_idx(t_sec)
        last_reg_pivot_bar_at_entry = -1
        for k in range(entry_1m_idx, -1, -1):
            if k in reg_pivots:
                last_reg_pivot_bar_at_entry = k
                break
        in_position = True

    for T in range(n_sec):
        t_sec = int(sec['ts'][T])
        price = sec['close'][T]
        hi = sec['high'][T]
        lo = sec['low'][T]
        sec_in_day = seconds_past_midnight(t_sec)

        # Sniper window
        if sniper_active:
            if entry_dir == 'LONG':
                if hi > sniper_best_price:
                    sniper_best_price = hi
            else:
                if lo < sniper_best_price:
                    sniper_best_price = lo
            if t_sec - sniper_start_ts >= sniper_sec:
                if entry_dir == 'LONG':
                    exit_price = sniper_best_price - fill_slip_pts
                else:
                    exit_price = sniper_best_price + fill_slip_pts
                pnl = ((exit_price - entry_price) if entry_dir == 'LONG'
                       else (entry_price - exit_price)) * DOLLAR_PER_POINT
                trades.append({
                    'entry_ts': entry_ts, 'exit_ts': t_sec,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'direction': entry_dir, 'pnl': pnl,
                    'exit_reason': 'sniper',
                    'held_sec': t_sec - entry_ts,
                })
                in_position = False
                sniper_active = False
            continue

        # EOD
        if in_position and sec_in_day >= eod_utc_sec:
            exit_price = price
            pnl = ((exit_price - entry_price) if entry_dir == 'LONG'
                   else (entry_price - exit_price)) * DOLLAR_PER_POINT
            trades.append({
                'entry_ts': entry_ts, 'exit_ts': t_sec,
                'entry_price': entry_price, 'exit_price': exit_price,
                'direction': entry_dir, 'pnl': pnl,
                'exit_reason': 'eod',
                'held_sec': t_sec - entry_ts,
            })
            in_position = False
            continue

        # Exit signal check (on 1m-close)
        if in_position and t_sec in ts_1m_set:
            just_closed_1m_idx = ts_1m_set[t_sec] - 1
            if just_closed_1m_idx >= 0 and not np.isnan(fitted_1m[just_closed_1m_idx]):
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

        # ENTRY LOGIC — varies by mode
        if not in_position:
            if entry_mode == 'residual':
                # Residual-threshold trigger: enter on crossing |res| >= threshold
                r = float(residuals_1s[T])
                abs_r = abs(r) if not np.isnan(r) else 0
                is_above = abs_r >= res_trigger
                # Rising-edge only (don't re-enter while above)
                if is_above and not res_above_trigger:
                    direction = 'LONG' if r < 0 else 'SHORT'
                    try_enter(T, direction, price, t_sec)
                res_above_trigger = is_above
                continue  # skip zigzag logic for residual mode

            # Zigzag-based modes (filters, tight)
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

            if pivot_fired and pivot_at is not None:
                r = float(residuals_1s[pivot_at])
                if np.isnan(r) or abs(r) < min_res:
                    continue
                if entry_mode == 'filters':
                    # Require wick AND volume above thresholds
                    w = float(wick_1s[pivot_at])
                    v = float(vol_1s[pivot_at])
                    if w < wick_min or v < vol_min:
                        continue
                direction = 'LONG' if r < 0 else 'SHORT'
                try_enter(T, direction, price, t_sec)
        else:
            # Update zigzag state while in position (so we know extreme after exit)
            if leg_dir == 'up':
                if price > extreme_price:
                    extreme_idx = T
                    extreme_price = price
            elif leg_dir == 'down':
                if price < extreme_price:
                    extreme_idx = T
                    extreme_price = price

    if in_position:
        exit_price = sec['close'][-1]
        t_sec = int(sec['ts'][-1])
        pnl = ((exit_price - entry_price) if entry_dir == 'LONG'
               else (entry_price - exit_price)) * DOLLAR_PER_POINT
        trades.append({
            'entry_ts': entry_ts, 'exit_ts': t_sec,
            'entry_price': entry_price, 'exit_price': exit_price,
            'direction': entry_dir, 'pnl': pnl,
            'exit_reason': 'day_end',
            'held_sec': t_sec - entry_ts,
        })

    return trades


def run_variant(paths_1m, cfg, label):
    """cfg dict: entry_mode, r_entry, res_trigger, wick_min, vol_min, sniper_sec, min_res"""
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
            sec, closes_1m, ts_1m, residuals_1s, residuals_1m, wick_1s, vol_1s = loaded
        except Exception:
            continue
        n_days += 1
        r_entry_pts = cfg['r_entry'] / DOLLAR_PER_POINT
        trades = simulate(sec, closes_1m, ts_1m, residuals_1s, residuals_1m,
                          wick_1s, vol_1s,
                          entry_mode=cfg['entry_mode'],
                          r_entry_pts=r_entry_pts,
                          min_res=cfg['min_res'],
                          res_trigger=cfg['res_trigger'],
                          wick_min=cfg['wick_min'],
                          vol_min=cfg['vol_min'],
                          sniper_sec=cfg['sniper_sec'])
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
        'mean_hold_sec': float(hold.mean()),
        'median_hold_sec': float(np.median(hold)),
        'max_win': float(pnls.max()),
        'max_loss': float(pnls.min()),
        'exits': dict(exits),
    }


def main():
    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    variants = [
        ('BASELINE',  {'entry_mode': 'tight',    'r_entry': 2.0, 'res_trigger': 0,
                        'min_res': 0.5, 'wick_min': 0, 'vol_min': 0, 'sniper_sec': 30}),
        ('A_FILTERS', {'entry_mode': 'filters',  'r_entry': 2.0, 'res_trigger': 0,
                        'min_res': 0.5, 'wick_min': 0.30, 'vol_min': 1.3, 'sniper_sec': 30}),
        ('B_TIGHT',   {'entry_mode': 'tight',    'r_entry': 1.0, 'res_trigger': 0,
                        'min_res': 0.5, 'wick_min': 0, 'vol_min': 0, 'sniper_sec': 30}),
        ('C_RESID',   {'entry_mode': 'residual', 'r_entry': 0,   'res_trigger': 1.5,
                        'min_res': 0.5, 'wick_min': 0, 'vol_min': 0, 'sniper_sec': 30}),
    ]

    results = []
    for name, cfg in variants:
        print(f'\n--- {name} ---')
        is_trades, is_days = run_variant(is_paths, cfg, f'IS {name}')
        oos_trades, oos_days = run_variant(oos_paths, cfg, f'OOS {name}')
        is_s = summarize(is_trades)
        oos_s = summarize(oos_trades)
        if not is_s or not oos_s:
            continue
        is_day = is_s['total'] / is_days if is_days else 0
        oos_day = oos_s['total'] / oos_days if oos_days else 0
        print(f'{name}: IS ${is_day:+,.0f}/day ({is_s["n"]:,} trades, WR {is_s["wr"]:.1f}%) | '
              f'OOS ${oos_day:+,.0f}/day ({oos_s["n"]:,} trades, WR {oos_s["wr"]:.1f}%)')
        results.append({
            'name': name, 'cfg': cfg,
            'is_day': is_day, 'oos_day': oos_day,
            'is_n': is_s['n'], 'oos_n': oos_s['n'],
            'is_wr': is_s['wr'], 'oos_wr': oos_s['wr'],
            'is_pt': is_s['per_trade'], 'oos_pt': oos_s['per_trade'],
            'is_hold': is_s['median_hold_sec'], 'oos_hold': oos_s['median_hold_sec'],
            'is_maxloss': is_s['max_loss'], 'oos_maxloss': oos_s['max_loss'],
            'is_days': is_days, 'oos_days': oos_days,
        })

    print('\n=== SUMMARY ===')
    print(f'{"Variant":<12} {"IS $/day":>10} {"OOS $/day":>11} '
          f'{"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>9} {"OOS $/tr":>10} '
          f'{"IS trades":>10} {"OOS trades":>11}')
    for r in results:
        print(f'{r["name"]:<12} ${r["is_day"]:>+8,.0f} ${r["oos_day"]:>+9,.0f} '
              f'{r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
              f'${r["is_pt"]:>+7.2f} ${r["oos_pt"]:>+8.2f} '
              f'{r["is_n"]:>10,} {r["oos_n"]:>11,}')

    out = ['# Entry-variant comparison', '']
    out.append('All variants use same physics exit: reg flip + residual flip → '
               '30s sniper. No SL, no thesis_broken, no adverse_reg.')
    out.append('')
    out.append('| Variant | IS $/day | OOS $/day | IS WR | OOS WR | IS $/tr | OOS $/tr | '
               'IS N | OOS N | IS hold | Max loss |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        out.append(f'| {r["name"]} | ${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                   f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                   f'${r["is_pt"]:+.2f} | ${r["oos_pt"]:+.2f} | '
                   f'{r["is_n"]:,} | {r["oos_n"]:,} | '
                   f'{r["is_hold"]:.0f}s | ${r["is_maxloss"]:+.0f} |')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
