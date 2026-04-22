"""
Pivot-residual saturation sim — enter at each zigzag pivot in the direction
predicted by res_10_norm, exit on TP / SL / residual-sign-flip (inverse signal).

No timeout — let the trade run until one of the three exit conditions fires.
Single position at a time (skip new pivots while open).

Entry physics:
  - On zigzag pivot confirmed at bar i, compute res_10_norm at bar i
  - res_10_norm < 0 → LONG (price below mean, expect reversion up)
  - res_10_norm > 0 → SHORT (price above mean, expect reversion down)
  - |res_10_norm| < min_strength → SKIP (noise)

Exit physics (first fires wins):
  - TP: price hits entry_price + $15 (LONG) or entry_price - $15 (SHORT)
  - SL: price hits entry_price - $3 (LONG) or entry_price + $3 (SHORT)
  - Inverse signal: res_10_norm flips sign (mean reversion completed)

Reports per day & aggregate:
  - N trades, WR, $/trade, total $/day
  - Exit reason breakdown
  - IS + OOS walk-forward comparison

Usage:
    python tools/pivot_residual_sim.py
    python tools/pivot_residual_sim.py --tp 20 --sl 5
    python tools/pivot_residual_sim.py --min-res-strength 1.0

Output: reports/findings/pivot_residual_sim.md
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
from tools.regression_line_cohen_d import zigzag_pivots, compute_regression_features


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/pivot_residual_sim.md'
WINDOWS = [10, 20, 60, 180, 720]
DOLLAR_PER_POINT = 2.0


def compute_res10_single(closes, bar_idx, window=10):
    """Fast inline res_10_norm at a single bar."""
    if bar_idx < window - 1:
        return None
    y = closes[bar_idx - window + 1: bar_idx + 1]
    x = np.arange(window, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    denom = (dx * dx).sum()
    if denom < 1e-9:
        return None
    slope = (dx * (y - ym)).sum() / denom
    intercept = ym - slope * xm
    fit_now = intercept + slope * (window - 1)
    residual = closes[bar_idx] - fit_now
    fits = intercept + slope * x
    resid_std = np.std(y - fits, ddof=1) if window > 2 else 1.0
    if resid_std < 1e-9:
        return 0.0
    return residual / resid_std


def simulate_day(closes, pivot_indices, tp_dollars, sl_dollars,
                  min_res_strength, inverse_threshold=0.0):
    """Run the strategy through the day's 1m bars.

    Returns list of trade dicts + unfilled-pivot count.
    """
    tp_pts = tp_dollars / DOLLAR_PER_POINT
    sl_pts = sl_dollars / DOLLAR_PER_POINT
    trades = []
    skipped_noise = 0
    skipped_in_position = 0
    n = len(closes)

    # Map pivot index → True for O(1) check
    pivot_set = set(pivot_indices)

    in_position = False
    entry_price = None
    direction = None
    entry_bar = None
    entry_res = None
    exit_reason = None

    for i in range(n):
        # Entry check — only at confirmed pivot, not in position
        if (i in pivot_set) and (not in_position):
            r = compute_res10_single(closes, i, window=10)
            if r is None:
                continue
            if abs(r) < min_res_strength:
                skipped_noise += 1
                continue
            direction = 'LONG' if r < 0 else 'SHORT'
            entry_price = closes[i]
            entry_bar = i
            entry_res = r
            in_position = True
            continue

        if in_position:
            price = closes[i]
            if direction == 'LONG':
                pnl_pts = price - entry_price
            else:
                pnl_pts = entry_price - price
            pnl_dollars = pnl_pts * DOLLAR_PER_POINT

            # TP check
            if pnl_pts >= tp_pts:
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'direction': direction, 'entry_res': entry_res,
                    'pnl': tp_dollars, 'exit_reason': 'TP',
                    'held_bars': i - entry_bar,
                })
                in_position = False
                continue
            # SL check
            if pnl_pts <= -sl_pts:
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'direction': direction, 'entry_res': entry_res,
                    'pnl': -sl_dollars, 'exit_reason': 'SL',
                    'held_bars': i - entry_bar,
                })
                in_position = False
                continue
            # Inverse signal check (every bar)
            r_now = compute_res10_single(closes, i, window=10)
            if r_now is None:
                continue
            if (direction == 'LONG' and r_now > inverse_threshold) or \
               (direction == 'SHORT' and r_now < -inverse_threshold):
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'direction': direction, 'entry_res': entry_res,
                    'pnl': pnl_dollars, 'exit_reason': 'inverse',
                    'held_bars': i - entry_bar,
                })
                in_position = False
                continue

            # Pivot that fires while in position — just count, don't act
            if i in pivot_set:
                skipped_in_position += 1

    # End-of-day: flatten any open position
    if in_position:
        price = closes[-1]
        if direction == 'LONG':
            pnl_pts = price - entry_price
        else:
            pnl_pts = entry_price - price
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': n - 1,
            'direction': direction, 'entry_res': entry_res,
            'pnl': pnl_pts * DOLLAR_PER_POINT, 'exit_reason': 'eod',
            'held_bars': n - 1 - entry_bar,
        })

    return trades, skipped_noise, skipped_in_position


def run_pass(paths, pivot_threshold, tp, sl, min_res_strength, label,
             inverse_threshold=0.0):
    all_trades = []
    per_day = []
    total_skipped_noise = 0
    total_skipped_in_pos = 0

    for p in tqdm(paths, desc=label, unit='day'):
        day_name = os.path.basename(p).replace('.parquet', '')
        df = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
        closes = df['close'].values.astype(np.float64)
        pivots = zigzag_pivots(closes, pivot_threshold)
        trades, sk_noise, sk_inpos = simulate_day(
            closes, pivots, tp, sl, min_res_strength, inverse_threshold)
        for t in trades:
            t['day'] = day_name
        all_trades.extend(trades)
        total_skipped_noise += sk_noise
        total_skipped_in_pos += sk_inpos
        day_pnl = sum(t['pnl'] for t in trades)
        per_day.append({
            'day': day_name,
            'n_trades': len(trades),
            'pnl': day_pnl,
            'pivots': len(pivots) - 1,
        })
    return all_trades, per_day, total_skipped_noise, total_skipped_in_pos


def summarize(trades, label):
    if not trades:
        return None
    pnls = np.array([t['pnl'] for t in trades])
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    total = pnls.sum()
    win_d = pnls[pnls > 0].sum()
    loss_d = -pnls[pnls < 0].sum()
    exit_counter = Counter(t['exit_reason'] for t in trades)
    hold_bars = np.array([t['held_bars'] for t in trades])
    dollar_wr = (win_d / loss_d - 1) * 100 if loss_d > 0 else float('inf')
    return {
        'n': len(trades),
        'total': total,
        'per_trade': total / len(trades),
        'wins': wins,
        'losses': losses,
        'wr': wins / (wins + losses) * 100 if (wins + losses) else 0,
        'dollar_wr': dollar_wr,
        'win_d': win_d,
        'loss_d': loss_d,
        'exits': dict(exit_counter),
        'mean_hold': float(hold_bars.mean()),
        'median_hold': float(np.median(hold_bars)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pivot-threshold', type=float, default=15.0)
    ap.add_argument('--tp', type=float, default=15.0)
    ap.add_argument('--sl', type=float, default=3.0,
                    help='SL in $ (use very large, e.g. 9999, to disable)')
    ap.add_argument('--min-res-strength', type=float, default=0.5,
                    help='Skip entries when |res_10_norm| < this')
    ap.add_argument('--sweep-sl', action='store_true',
                    help='Sweep SL across {3, 5, 8, 10, 15, 20, 9999=no-SL}')
    ap.add_argument('--sweep-inverse', action='store_true',
                    help='Sweep inverse exit threshold {0, 0.5, 1.0, 1.5, 2.0, 3.0, 99=never}')
    ap.add_argument('--inverse-threshold', type=float, default=0.0,
                    help='Exit only when residual crosses this magnitude '
                         '(0 = current behavior, exits at mean)')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} days | OOS {len(oos_paths)}')

    if args.sweep_sl:
        sl_values = [3, 5, 8, 10, 15, 20, 9999]
        print(f'Sweeping SL in {sl_values}')
        all_rows = []
        for sl in sl_values:
            print(f'\n--- SL=${sl} ---')
            is_trades, _, _, _ = run_pass(
                is_paths, args.pivot_threshold, args.tp, sl,
                args.min_res_strength, f'IS sl=${sl}')
            oos_trades, _, _, _ = run_pass(
                oos_paths, args.pivot_threshold, args.tp, sl,
                args.min_res_strength, f'OOS sl=${sl}')
            is_sum = summarize(is_trades, 'IS')
            oos_sum = summarize(oos_trades, 'OOS')
            if is_sum and oos_sum:
                all_rows.append({
                    'sl': sl,
                    'is_day': is_sum['total'] / len(is_paths),
                    'oos_day': oos_sum['total'] / len(oos_paths),
                    'is_n': is_sum['n'],
                    'oos_n': oos_sum['n'],
                    'is_wr': is_sum['wr'],
                    'oos_wr': oos_sum['wr'],
                    'is_per_trade': is_sum['per_trade'],
                    'oos_per_trade': oos_sum['per_trade'],
                    'is_exits': is_sum['exits'],
                    'oos_exits': oos_sum['exits'],
                    'is_mean_hold': is_sum['mean_hold'],
                    'oos_mean_hold': oos_sum['mean_hold'],
                })

        # Console
        print(f'\n{"SL":>6} {"IS $/day":>10} {"OOS $/day":>10} '
              f'{"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>8} {"OOS $/tr":>9} '
              f'{"IS hold":>8} {"OOS hold":>9}')
        for r in all_rows:
            sl_label = 'none' if r['sl'] == 9999 else f'${r["sl"]}'
            print(f'{sl_label:>6} ${r["is_day"]:>+8,.0f} ${r["oos_day"]:>+8,.0f} '
                  f'{r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_per_trade"]:>+7.2f} ${r["oos_per_trade"]:>+7.2f} '
                  f'{r["is_mean_hold"]:>7.1f} {r["oos_mean_hold"]:>8.1f}')

        # MD
        out = [f'# Pivot-residual sim — SL sweep', '']
        out.append(f'**Entry**: zigzag pivots at ${args.pivot_threshold}, '
                   f'direction from `res_10_norm` sign (|res|>={args.min_res_strength}).')
        out.append(f'**Exit**: TP=${args.tp}, SL varies, or residual sign flip '
                   '(inverse signal).')
        out.append('')
        out.append('## Sweep results')
        out.append('')
        out.append('| SL | IS $/day | OOS $/day | IS WR | OOS WR | '
                   'IS $/trade | OOS $/trade | IS trades | OOS trades | '
                   'IS mean hold | OOS mean hold |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in all_rows:
            sl_label = 'none' if r['sl'] == 9999 else f'${r["sl"]}'
            out.append(f'| {sl_label} | ${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                       f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                       f'${r["is_per_trade"]:+.2f} | ${r["oos_per_trade"]:+.2f} | '
                       f'{r["is_n"]:,} | {r["oos_n"]:,} | '
                       f'{r["is_mean_hold"]:.1f} | {r["oos_mean_hold"]:.1f} |')
        out.append('')

        out.append('## Exit reason breakdown by SL')
        out.append('')
        for r in all_rows:
            sl_label = 'none' if r['sl'] == 9999 else f'${r["sl"]}'
            out.append(f'**SL={sl_label}**:')
            out.append('')
            all_reasons = set(r['is_exits'].keys()) | set(r['oos_exits'].keys())
            out.append('| Reason | IS N | IS % | OOS N | OOS % |')
            out.append('|---|---:|---:|---:|---:|')
            for reason in sorted(all_reasons):
                is_c = r['is_exits'].get(reason, 0)
                oos_c = r['oos_exits'].get(reason, 0)
                is_pct = is_c / r['is_n'] * 100 if r['is_n'] else 0
                oos_pct = oos_c / r['oos_n'] * 100 if r['oos_n'] else 0
                out.append(f'| {reason} | {is_c:,} | {is_pct:.1f}% | '
                           f'{oos_c:,} | {oos_pct:.1f}% |')
            out.append('')

        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f'\nWrote: {OUT_MD}')
        return

    if args.sweep_inverse:
        inv_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 99.0]
        sl = args.sl
        print(f'Sweeping inverse-threshold with SL=${sl}, TP=${args.tp}')
        all_rows = []
        for inv in inv_values:
            print(f'\n--- inverse_thr={inv} ---')
            is_trades, _, _, _ = run_pass(
                is_paths, args.pivot_threshold, args.tp, sl,
                args.min_res_strength, f'IS inv={inv}', inv)
            oos_trades, _, _, _ = run_pass(
                oos_paths, args.pivot_threshold, args.tp, sl,
                args.min_res_strength, f'OOS inv={inv}', inv)
            is_sum = summarize(is_trades, 'IS')
            oos_sum = summarize(oos_trades, 'OOS')
            if is_sum and oos_sum:
                all_rows.append({
                    'inv': inv,
                    'is_day': is_sum['total'] / len(is_paths),
                    'oos_day': oos_sum['total'] / len(oos_paths),
                    'is_n': is_sum['n'], 'oos_n': oos_sum['n'],
                    'is_wr': is_sum['wr'], 'oos_wr': oos_sum['wr'],
                    'is_per_trade': is_sum['per_trade'],
                    'oos_per_trade': oos_sum['per_trade'],
                    'is_exits': is_sum['exits'],
                    'oos_exits': oos_sum['exits'],
                    'is_mean_hold': is_sum['mean_hold'],
                    'oos_mean_hold': oos_sum['mean_hold'],
                })
        print(f'\n{"inv_thr":>8} {"IS $/day":>10} {"OOS $/day":>10} '
              f'{"IS WR":>7} {"OOS WR":>7} {"IS $/tr":>9} {"OOS $/tr":>9} '
              f'{"IS hold":>8} {"OOS hold":>9}')
        for r in all_rows:
            lbl = 'none' if r['inv'] == 99.0 else f'{r["inv"]}'
            print(f'{lbl:>8} ${r["is_day"]:>+8,.0f} ${r["oos_day"]:>+8,.0f} '
                  f'{r["is_wr"]:>6.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_per_trade"]:>+7.2f} ${r["oos_per_trade"]:>+7.2f} '
                  f'{r["is_mean_hold"]:>7.1f} {r["oos_mean_hold"]:>8.1f}')
        out = [f'# Pivot-residual sim — inverse-threshold sweep', '']
        out.append(f'**Entry**: zigzag pivots at ${args.pivot_threshold}. '
                   f'TP=${args.tp}, SL=${sl}.')
        out.append('**Inverse exit threshold** = exit only when residual crosses '
                   'this magnitude on opposite side. 0 = exits at mean-crossing '
                   '(current). Higher = more breathing room.')
        out.append('')
        out.append('| inv_thr | IS $/day | OOS $/day | IS WR | OOS WR | '
                   'IS $/tr | OOS $/tr | IS N | OOS N | IS hold | OOS hold |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in all_rows:
            lbl = 'none' if r['inv'] == 99.0 else f'{r["inv"]}'
            out.append(f'| {lbl} | ${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                       f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                       f'${r["is_per_trade"]:+.2f} | ${r["oos_per_trade"]:+.2f} | '
                       f'{r["is_n"]:,} | {r["oos_n"]:,} | '
                       f'{r["is_mean_hold"]:.1f} | {r["oos_mean_hold"]:.1f} |')
        out.append('')
        out.append('## Exit reasons by inverse_threshold')
        out.append('')
        for r in all_rows:
            lbl = 'none' if r['inv'] == 99.0 else str(r['inv'])
            out.append(f'**inv_thr={lbl}**:')
            out.append('')
            all_reasons = set(r['is_exits'].keys()) | set(r['oos_exits'].keys())
            out.append('| Reason | IS N | IS % | OOS N | OOS % |')
            out.append('|---|---:|---:|---:|---:|')
            for reason in sorted(all_reasons):
                is_c = r['is_exits'].get(reason, 0)
                oos_c = r['oos_exits'].get(reason, 0)
                is_pct = is_c / r['is_n'] * 100 if r['is_n'] else 0
                oos_pct = oos_c / r['oos_n'] * 100 if oos_sum['n'] else 0
                out.append(f'| {reason} | {is_c:,} | {is_pct:.1f}% | '
                           f'{oos_c:,} | {oos_pct:.1f}% |')
            out.append('')
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f'\nWrote: {OUT_MD}')
        return

    print(f'Pivot ${args.pivot_threshold} | TP ${args.tp} | SL ${args.sl} | '
          f'Min |res| {args.min_res_strength} | Inv thr {args.inverse_threshold}')

    is_trades, is_days, is_skip_n, is_skip_p = run_pass(
        is_paths, args.pivot_threshold, args.tp, args.sl,
        args.min_res_strength, 'IS', args.inverse_threshold)
    oos_trades, oos_days, oos_skip_n, oos_skip_p = run_pass(
        oos_paths, args.pivot_threshold, args.tp, args.sl,
        args.min_res_strength, 'OOS', args.inverse_threshold)

    is_sum = summarize(is_trades, 'IS')
    oos_sum = summarize(oos_trades, 'OOS')

    def print_sum(label, s, n_days):
        if s is None:
            print(f'\n{label}: no trades')
            return
        print(f'\n=== {label} ({n_days} days) ===')
        print(f'  Trades: {s["n"]:,}  ({s["n"]/n_days:.1f}/day)')
        print(f'  WR: {s["wr"]:.1f}% ({s["wins"]:,}W / {s["losses"]:,}L)')
        print(f'  $WR: {s["dollar_wr"]:+.0f}%')
        print(f'  $/trade: ${s["per_trade"]:+.2f}')
        print(f'  Total $: ${s["total"]:+,.0f}')
        print(f'  $/day: ${s["total"]/n_days:+,.2f}')
        print(f'  Mean hold: {s["mean_hold"]:.1f} bars  '
              f'Median: {s["median_hold"]:.0f} bars')
        print(f'  Exits: {s["exits"]}')

    print_sum('IS', is_sum, len(is_paths))
    print_sum('OOS', oos_sum, len(oos_paths))
    print(f'\nIS skipped (low |res|): {is_skip_n:,}  |  (in position): {is_skip_p:,}')
    print(f'OOS skipped (low |res|): {oos_skip_n:,}  |  (in position): {oos_skip_p:,}')

    # MD
    out = [f'# Pivot-residual saturation sim', '']
    out.append(f'**Entry**: zigzag pivots at ${args.pivot_threshold}, direction '
               f'from sign of `res_10_norm` at pivot (|res|>={args.min_res_strength}).')
    out.append(f'**Exit**: TP=${args.tp}, SL=${args.sl}, or residual sign flip.')
    out.append(f'**Sizing**: 1 contract, no chains.')
    out.append('')

    out.append('## Headline')
    out.append('')
    out.append('| Dataset | Days | Trades | $/day | $/trade | WR | $WR | Total $ |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for label, s, n_days in [('IS', is_sum, len(is_paths)),
                              ('OOS', oos_sum, len(oos_paths))]:
        if s:
            out.append(f'| {label} | {n_days} | {s["n"]:,} | '
                       f'${s["total"]/n_days:+.0f} | '
                       f'${s["per_trade"]:+.2f} | '
                       f'{s["wr"]:.1f}% | '
                       f'{s["dollar_wr"]:+.0f}% | '
                       f'${s["total"]:+,.0f} |')
    out.append('')

    out.append('## Exit reason breakdown')
    out.append('')
    out.append('| Exit reason | IS count | IS % | OOS count | OOS % |')
    out.append('|---|---:|---:|---:|---:|')
    all_reasons = set(is_sum['exits'].keys()) | set(oos_sum['exits'].keys()) if is_sum and oos_sum else set()
    for reason in sorted(all_reasons):
        is_c = is_sum['exits'].get(reason, 0) if is_sum else 0
        oos_c = oos_sum['exits'].get(reason, 0) if oos_sum else 0
        is_pct = is_c / is_sum['n'] * 100 if is_sum['n'] else 0
        oos_pct = oos_c / oos_sum['n'] * 100 if oos_sum['n'] else 0
        out.append(f'| {reason} | {is_c:,} | {is_pct:.1f}% | '
                   f'{oos_c:,} | {oos_pct:.1f}% |')
    out.append('')

    out.append('## Hold time')
    out.append('')
    out.append('| Dataset | Mean hold (1m bars) | Median hold |')
    out.append('|---|---:|---:|')
    out.append(f'| IS | {is_sum["mean_hold"]:.1f} | {is_sum["median_hold"]:.0f} |')
    out.append(f'| OOS | {oos_sum["mean_hold"]:.1f} | {oos_sum["median_hold"]:.0f} |')
    out.append('')

    out.append('## vs reference baselines')
    out.append('')
    out.append('| System | IS $/day | OOS $/day | Combined $/day |')
    out.append('|---|---:|---:|---:|')
    is_day = is_sum['total'] / len(is_paths)
    oos_day = oos_sum['total'] / len(oos_paths)
    comb_day = (is_sum['total'] + oos_sum['total']) / (len(is_paths) + len(oos_paths))
    out.append(f'| **Pivot-residual sim** | ${is_day:+.0f} | '
               f'${oos_day:+.0f} | ${comb_day:+.0f} |')
    out.append(f'| Current iso engine | +$311 | +$67 | +$261 |')
    out.append(f'| Saturation ($20/$3/8min) | +$70 | +$259 | +$109 |')
    out.append(f'| Blended engine (CNN off) | -$52 | +$172 | -$8 |')
    out.append('')

    out.append('## Cord capture')
    out.append('')
    out.append('| | Cord ceiling $/day | System $/day | Capture % |')
    out.append('|---|---:|---:|---:|')
    out.append(f'| IS | $8,375 | ${is_day:+.0f} | {is_day/8375*100:.1f}% |')
    out.append(f'| OOS | $10,285 | ${oos_day:+.0f} | {oos_day/10285*100:.1f}% |')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
