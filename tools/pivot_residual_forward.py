"""
Pivot-residual FORWARD PASS — no lookahead, real-time pivot detection.

Uses precomputed `1m_z_se` from FEATURES_5s (the SFE regression-residual
feature, already built with no-lookahead N-1 higher-TF convention). This
is the same signal the live engine sees, and sampling it at 1m boundaries
is O(lookup) — much faster than re-running rolling regression.

Key difference from pivot_residual_sim.py (oracle):
  - Oracle: magically knew pivot bar index, entered at pivot price
  - Forward: walks bar-by-bar. Pivot only known AFTER price has retraced
    $R from the running extreme. Entry happens at the CONFIRMATION bar
    (some bars after the true pivot). Entry price = confirmation-bar close.

No-lookahead constraints:
  - Pivot detection: only uses closes[0..T] at time T.
  - `1m_z_se` precomputed with N-1 higher-TF convention (post 2026-04-17 fix).
  - 1m_z_se at the pivot bar: historical, known at confirmation time.
  - 1m_z_se at live bar T: current value, computed from closes ≤ T.

Trade physics:
  - Zigzag with R_confirm = $5 (smaller than TP) detects pivot in real-time.
  - When pivot confirmed at bar T (extreme was at bar idx_H):
      1. Direction from residual at idx_H (historical, no lookahead)
      2. Skip if |residual| < min_res_strength
      3. Enter at closes[T] in direction indicated by residual
      4. TP = entry ± $15, SL = entry ± $10
      5. Inverse exit: residual crosses ± inverse_threshold
  - One position at a time. New pivots while in position are skipped.

Usage:
    python tools/pivot_residual_forward.py
    python tools/pivot_residual_forward.py --r-confirm 5 --tp 15 --sl 10
    python tools/pivot_residual_forward.py --sweep-r-confirm

Output: reports/findings/pivot_residual_forward.md
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
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
RESIDUAL_FEATURE = '1m_z_se'   # SFE regression residual, precomputed
VOL_FEATURE = '1m_bar_range'   # 1m bar high-low, proxy for local volatility
OUT_MD = 'reports/findings/pivot_residual_forward.md'
DOLLAR_PER_POINT = 2.0
VOL_WINDOW = 10                # rolling average window for vol metric (1m bars)


def load_day_with_residuals(price_path, features_path):
    """Load 1m bars and sample residual + volatility features at 1m boundaries.
    Returns (closes, residuals, vol_pts, timestamps).
      residuals[i]: 1m_z_se at timestamp[i]
      vol_pts[i]: rolling mean of 1m_bar_range over last VOL_WINDOW bars (points)
    All no-lookahead.
    """
    df_price = pd.read_parquet(price_path).sort_values('timestamp').reset_index(drop=True)
    closes = df_price['close'].values.astype(np.float64)
    ts_1m = df_price['timestamp'].values.astype(np.int64)

    df_feat = pd.read_parquet(features_path).sort_values('timestamp').reset_index(drop=True)
    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    res_feat = df_feat[RESIDUAL_FEATURE].values.astype(np.float64)
    vol_feat = df_feat[VOL_FEATURE].values.astype(np.float64) if VOL_FEATURE in df_feat.columns else np.zeros(len(df_feat))

    # Sample features at each 1m timestamp (no-lookahead).
    idx = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx = np.clip(idx, 0, len(ts_feat) - 1)
    residuals = res_feat[idx].astype(np.float64)
    bar_range_1m = vol_feat[idx].astype(np.float64)

    # Rolling mean of 1m_bar_range as "local vol" — use only PAST bars.
    vol_pts = np.zeros(len(closes), dtype=np.float64)
    for i in range(len(closes)):
        lo = max(0, i - VOL_WINDOW + 1)
        vol_pts[i] = bar_range_1m[lo:i+1].mean() if i >= lo else 0.0
    # bar_range is in points already (MNQ tick 0.25, so range in pts)
    return closes, residuals, vol_pts, ts_1m


def simulate_forward(closes, residuals, vol_pts, r_confirm_pts, tp_pts, sl_pts,
                     min_res_strength, inverse_threshold,
                     sl_vol_mult=None, sl_vol_floor_pts=1.5):
    """Real-time pivot detection + residual-based entry.

    residuals[i] = precomputed 1m_z_se at timestamp i (no lookahead).
    vol_pts[i] = rolling mean 1m_bar_range over last VOL_WINDOW bars (points).
    Args in POINTS (not dollars). Multiply by $2/pt for dollar equivalents.

    If sl_vol_mult is not None, SL at entry = max(sl_vol_floor_pts,
    sl_vol_mult × vol_pts[entry_bar]). Otherwise fixed sl_pts is used.
    """
    n = len(closes)
    trades = []
    skipped_noise = 0
    skipped_in_position = 0
    n_pivots = 0

    # Pivot-tracking state
    leg_dir = None           # None / 'up' / 'down'
    extreme_idx = 0
    extreme_price = closes[0]
    # Last confirmed pivot
    last_pivot_idx = None

    # Position state
    in_position = False
    entry_bar = None
    entry_price = None
    direction = None
    entry_res = None
    trade_sl_pts = sl_pts

    for T in range(n):
        price = closes[T]

        # ── Pivot tracking (NO LOOKAHEAD: only uses current + past closes) ──
        if leg_dir is None:
            # Haven't established direction yet
            if price - extreme_price >= r_confirm_pts:
                leg_dir = 'up'
                last_pivot_idx = extreme_idx
                n_pivots += 1
                # Entry opportunity: pivot was LOW → predict direction
                r = residuals[last_pivot_idx] if last_pivot_idx < len(residuals) else float('nan')
                if not np.isnan(r):
                    if abs(r) < min_res_strength:
                        skipped_noise += 1
                    elif not in_position:
                        direction = 'LONG' if r < 0 else 'SHORT'
                        entry_bar = T
                        entry_price = price
                        entry_res = r
                        # Dynamic SL based on local volatility at entry
                        if sl_vol_mult is not None and T < len(vol_pts):
                            trade_sl_pts = max(sl_vol_floor_pts, sl_vol_mult * vol_pts[T])
                        else:
                            trade_sl_pts = sl_pts
                        in_position = True
                extreme_idx = T
                extreme_price = price
            elif extreme_price - price >= r_confirm_pts:
                leg_dir = 'down'
                last_pivot_idx = extreme_idx
                n_pivots += 1
                r = residuals[last_pivot_idx] if last_pivot_idx < len(residuals) else float('nan')
                if not np.isnan(r):
                    if abs(r) < min_res_strength:
                        skipped_noise += 1
                    elif not in_position:
                        direction = 'LONG' if r < 0 else 'SHORT'
                        entry_bar = T
                        entry_price = price
                        entry_res = r
                        # Dynamic SL based on local volatility at entry
                        if sl_vol_mult is not None and T < len(vol_pts):
                            trade_sl_pts = max(sl_vol_floor_pts, sl_vol_mult * vol_pts[T])
                        else:
                            trade_sl_pts = sl_pts
                        in_position = True
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
                # New high in up-leg
                extreme_idx = T
                extreme_price = price
            elif extreme_price - price >= r_confirm_pts:
                # Down confirmation — HIGH pivot was at extreme_idx
                last_pivot_idx = extreme_idx
                n_pivots += 1
                r = residuals[last_pivot_idx] if last_pivot_idx < len(residuals) else float('nan')
                if not np.isnan(r):
                    if abs(r) < min_res_strength:
                        skipped_noise += 1
                    elif not in_position:
                        direction = 'LONG' if r < 0 else 'SHORT'
                        entry_bar = T
                        entry_price = price
                        entry_res = r
                        # Dynamic SL based on local volatility at entry
                        if sl_vol_mult is not None and T < len(vol_pts):
                            trade_sl_pts = max(sl_vol_floor_pts, sl_vol_mult * vol_pts[T])
                        else:
                            trade_sl_pts = sl_pts
                        in_position = True
                    else:
                        skipped_in_position += 1
                leg_dir = 'down'
                extreme_idx = T
                extreme_price = price
        elif leg_dir == 'down':
            if price < extreme_price:
                extreme_idx = T
                extreme_price = price
            elif price - extreme_price >= r_confirm_pts:
                last_pivot_idx = extreme_idx
                n_pivots += 1
                r = residuals[last_pivot_idx] if last_pivot_idx < len(residuals) else float('nan')
                if not np.isnan(r):
                    if abs(r) < min_res_strength:
                        skipped_noise += 1
                    elif not in_position:
                        direction = 'LONG' if r < 0 else 'SHORT'
                        entry_bar = T
                        entry_price = price
                        entry_res = r
                        # Dynamic SL based on local volatility at entry
                        if sl_vol_mult is not None and T < len(vol_pts):
                            trade_sl_pts = max(sl_vol_floor_pts, sl_vol_mult * vol_pts[T])
                        else:
                            trade_sl_pts = sl_pts
                        in_position = True
                    else:
                        skipped_in_position += 1
                leg_dir = 'up'
                extreme_idx = T
                extreme_price = price

        # ── Position management ──
        if in_position and entry_bar is not None and T > entry_bar:
            if direction == 'LONG':
                pnl_pts = price - entry_price
            else:
                pnl_pts = entry_price - price

            # TP
            if pnl_pts >= tp_pts:
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': T,
                    'direction': direction, 'entry_res': entry_res,
                    'pnl': tp_pts * DOLLAR_PER_POINT, 'exit_reason': 'TP',
                    'held_bars': T - entry_bar,
                })
                in_position = False
                continue
            # SL (may be per-trade if sl_vol_mult is set)
            active_sl_pts = trade_sl_pts if sl_vol_mult is not None else sl_pts
            if pnl_pts <= -active_sl_pts:
                trades.append({
                    'entry_bar': entry_bar, 'exit_bar': T,
                    'direction': direction, 'entry_res': entry_res,
                    'pnl': -active_sl_pts * DOLLAR_PER_POINT, 'exit_reason': 'SL',
                    'held_bars': T - entry_bar,
                    'sl_pts': active_sl_pts,
                })
                in_position = False
                continue
            # Inverse residual (no lookahead — uses closes through T)
            r_now = residuals[T] if T < len(residuals) else float('nan')
            if not np.isnan(r_now):
                if (direction == 'LONG' and r_now > inverse_threshold) or \
                   (direction == 'SHORT' and r_now < -inverse_threshold):
                    trades.append({
                        'entry_bar': entry_bar, 'exit_bar': T,
                        'direction': direction, 'entry_res': entry_res,
                        'pnl': pnl_pts * DOLLAR_PER_POINT,
                        'exit_reason': 'inverse',
                        'held_bars': T - entry_bar,
                    })
                    in_position = False
                    continue

    # End-of-day flush
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

    return trades, skipped_noise, skipped_in_position, n_pivots


def run_pass(paths, r_confirm, tp, sl, min_res, inv_thr, label,
             sl_vol_mult=None):
    r_pts = r_confirm / DOLLAR_PER_POINT
    tp_pts = tp / DOLLAR_PER_POINT
    sl_pts = sl / DOLLAR_PER_POINT
    all_trades = []
    tot_pivots = 0
    tot_skip_noise = 0
    tot_skip_inpos = 0
    for p in tqdm(paths, desc=label, unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(feat_path):
            continue
        try:
            closes, residuals, vol_pts, _ts = load_day_with_residuals(p, feat_path)
        except Exception as e:
            print(f'skip {day}: {e}')
            continue
        trades, sk_n, sk_p, npiv = simulate_forward(
            closes, residuals, vol_pts, r_pts, tp_pts, sl_pts,
            min_res, inv_thr, sl_vol_mult)
        for t in trades:
            t['day'] = day
        all_trades.extend(trades)
        tot_pivots += npiv
        tot_skip_noise += sk_n
        tot_skip_inpos += sk_p
    return all_trades, tot_pivots, tot_skip_noise, tot_skip_inpos


def summarize(trades):
    if not trades:
        return None
    pnls = np.array([t['pnl'] for t in trades])
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    hold = np.array([t['held_bars'] for t in trades])
    exits = Counter(t['exit_reason'] for t in trades)
    w_d = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    l_d = -pnls[pnls < 0].sum() if (pnls < 0).any() else 0
    return {
        'n': len(trades), 'total': pnls.sum(),
        'per_trade': pnls.mean(),
        'wr': wins / max(wins + losses, 1) * 100,
        'dollar_wr': (w_d / l_d - 1) * 100 if l_d > 0 else float('inf'),
        'wins': int(wins), 'losses': int(losses),
        'mean_hold': float(hold.mean()),
        'median_hold': float(np.median(hold)),
        'exits': dict(exits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r-confirm', type=float, default=5.0,
                    help='$ retracement required to confirm a pivot (default 5)')
    ap.add_argument('--tp', type=float, default=15.0)
    ap.add_argument('--sl', type=float, default=10.0)
    ap.add_argument('--min-res-strength', type=float, default=0.5)
    ap.add_argument('--inverse-threshold', type=float, default=2.0)
    ap.add_argument('--sweep-r-confirm', action='store_true')
    ap.add_argument('--sweep-tp-sl', action='store_true',
                    help='2D sweep of TP x SL combinations')
    ap.add_argument('--sweep-sl-vol', action='store_true',
                    help='Sweep volatility-adjusted SL multipliers')
    ap.add_argument('--sl-vol-mult', type=float, default=None,
                    help='SL = sl_vol_mult × rolling 1m_bar_range (points). '
                         'Overrides fixed --sl when set.')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} days | OOS {len(oos_paths)} days')

    if args.sweep_sl_vol:
        tp_values = [20, 30, 40, 50]
        sl_vol_mults = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
        print(f'Sweeping TP {tp_values} x sl_vol_mult {sl_vol_mults}')
        rows = []
        for tp in tp_values:
            for mult in sl_vol_mults:
                is_trades, _, _, _ = run_pass(
                    is_paths, args.r_confirm, tp, 9999,
                    args.min_res_strength, args.inverse_threshold,
                    f'IS tp={tp} mult={mult}', sl_vol_mult=mult)
                oos_trades, _, _, _ = run_pass(
                    oos_paths, args.r_confirm, tp, 9999,
                    args.min_res_strength, args.inverse_threshold,
                    f'OOS tp={tp} mult={mult}', sl_vol_mult=mult)
                is_s = summarize(is_trades)
                oos_s = summarize(oos_trades)
                if is_s and oos_s:
                    is_day = is_s['total'] / len(is_paths)
                    oos_day = oos_s['total'] / len(oos_paths)
                    comb = (is_s['total'] + oos_s['total']) / (len(is_paths) + len(oos_paths))
                    # Mean SL dollars fired
                    sl_dollars = [abs(t['pnl']) for t in is_trades
                                   if t['exit_reason'] == 'SL']
                    avg_sl = sum(sl_dollars) / len(sl_dollars) if sl_dollars else 0
                    rows.append({
                        'tp': tp, 'mult': mult,
                        'is_day': is_day, 'oos_day': oos_day, 'comb': comb,
                        'is_wr': is_s['wr'], 'oos_wr': oos_s['wr'],
                        'is_pt': is_s['per_trade'], 'oos_pt': oos_s['per_trade'],
                        'is_n': is_s['n'], 'oos_n': oos_s['n'],
                        'avg_sl_dollars': avg_sl,
                    })
        rows.sort(key=lambda r: -r['comb'])
        print(f'\n{"TP":>4} {"mult":>6} {"avgSL":>7} {"IS$/d":>9} {"OOS$/d":>9} '
              f'{"Comb":>9} {"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>8} {"OOS $/tr":>9}')
        print('-' * 95)
        for r in rows:
            print(f'${r["tp"]:>3} {r["mult"]:>5.2f} ${r["avg_sl_dollars"]:>5.1f} '
                  f'${r["is_day"]:>+7,.0f} ${r["oos_day"]:>+7,.0f} '
                  f'${r["comb"]:>+7,.0f} {r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_pt"]:>+6.2f} ${r["oos_pt"]:>+7.2f}')
        # MD
        out = [f'# Pivot-residual forward — volatility-adaptive SL sweep', '']
        out.append(f'SL is set per-trade as `sl_vol_mult × rolling mean of '
                   f'1m_bar_range (last {VOL_WINDOW} bars)`, floored at '
                   '1.5 points ($3).')
        out.append('')
        out.append(f'r_confirm=${args.r_confirm} inv_thr={args.inverse_threshold} '
                   f'min_res={args.min_res_strength}')
        out.append('')
        out.append('## Ranked by combined $/day')
        out.append('')
        out.append('| TP | vol_mult | Avg SL $ | IS $/day | OOS $/day | '
                   'Combined | IS WR | OOS WR | IS $/tr | OOS $/tr | IS N |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            out.append(f'| ${r["tp"]} | {r["mult"]:.2f} | ${r["avg_sl_dollars"]:.1f} | '
                       f'${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                       f'**${r["comb"]:+,.0f}** | '
                       f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                       f'${r["is_pt"]:+.2f} | ${r["oos_pt"]:+.2f} | '
                       f'{r["is_n"]:,} |')
        out.append('')
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f'\nWrote: {OUT_MD}')
        return

    if args.sweep_tp_sl:
        tp_values = [10, 15, 20, 25, 30, 40, 50]
        sl_values = [3, 5, 8, 10, 15, 20]
        print(f'Sweeping TP {tp_values} x SL {sl_values} = {len(tp_values)*len(sl_values)} combos')
        rows = []
        for tp in tp_values:
            for sl in sl_values:
                is_trades, is_piv, _, _ = run_pass(
                    is_paths, args.r_confirm, tp, sl,
                    args.min_res_strength, args.inverse_threshold,
                    f'IS tp={tp} sl={sl}')
                oos_trades, oos_piv, _, _ = run_pass(
                    oos_paths, args.r_confirm, tp, sl,
                    args.min_res_strength, args.inverse_threshold,
                    f'OOS tp={tp} sl={sl}')
                is_s = summarize(is_trades)
                oos_s = summarize(oos_trades)
                if is_s and oos_s:
                    is_day = is_s['total'] / len(is_paths)
                    oos_day = oos_s['total'] / len(oos_paths)
                    comb = (is_s['total'] + oos_s['total']) / (len(is_paths) + len(oos_paths))
                    rows.append({
                        'tp': tp, 'sl': sl,
                        'is_day': is_day, 'oos_day': oos_day, 'comb': comb,
                        'is_wr': is_s['wr'], 'oos_wr': oos_s['wr'],
                        'is_pt': is_s['per_trade'], 'oos_pt': oos_s['per_trade'],
                        'is_n': is_s['n'], 'oos_n': oos_s['n'],
                    })

        # Rank by combined
        rows.sort(key=lambda r: -r['comb'])
        print(f'\n{"TP":>4} {"SL":>4} {"IS$/d":>9} {"OOS$/d":>9} {"Comb":>9} '
              f'{"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>8} {"OOS $/tr":>9}')
        print('-' * 85)
        for r in rows:
            print(f'${r["tp"]:>3} ${r["sl"]:>3} ${r["is_day"]:>+7,.0f} ${r["oos_day"]:>+7,.0f} '
                  f'${r["comb"]:>+7,.0f} {r["is_wr"]:>5.1f}% {r["oos_wr"]:>6.1f}% '
                  f'${r["is_pt"]:>+6.2f} ${r["oos_pt"]:>+7.2f}')

        # MD
        out = [f'# Pivot-residual forward — TP × SL 2D sweep', '']
        out.append(f'r_confirm=${args.r_confirm} min_res={args.min_res_strength} '
                   f'inv_thr={args.inverse_threshold}')
        out.append('')

        # Sorted by combined
        out.append('## Ranked by combined $/day (best first)')
        out.append('')
        out.append('| TP | SL | RR | IS $/day | OOS $/day | Combined $/day | '
                   'IS WR | OOS WR | IS $/trade | OOS $/trade | IS N | OOS N |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            rr = f'{r["tp"]/r["sl"]:.2f}:1'
            out.append(f'| ${r["tp"]} | ${r["sl"]} | {rr} | '
                       f'${r["is_day"]:+,.0f} | ${r["oos_day"]:+,.0f} | '
                       f'**${r["comb"]:+,.0f}** | '
                       f'{r["is_wr"]:.1f}% | {r["oos_wr"]:.1f}% | '
                       f'${r["is_pt"]:+.2f} | ${r["oos_pt"]:+.2f} | '
                       f'{r["is_n"]:,} | {r["oos_n"]:,} |')
        out.append('')

        # IS $/day heatmap
        out.append('## IS $/day by (TP, SL)')
        out.append('')
        hdr = ['TP \\\\ SL'] + [f'${sl}' for sl in sl_values]
        out.append('| ' + ' | '.join(hdr) + ' |')
        out.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
        for tp in tp_values:
            cells = [f'**${tp}**']
            for sl in sl_values:
                v = next((r['is_day'] for r in rows if r['tp'] == tp and r['sl'] == sl), None)
                cells.append(f'${v:+,.0f}' if v is not None else '—')
            out.append('| ' + ' | '.join(cells) + ' |')
        out.append('')

        # OOS heatmap
        out.append('## OOS $/day by (TP, SL)')
        out.append('')
        out.append('| ' + ' | '.join(hdr) + ' |')
        out.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
        for tp in tp_values:
            cells = [f'**${tp}**']
            for sl in sl_values:
                v = next((r['oos_day'] for r in rows if r['tp'] == tp and r['sl'] == sl), None)
                cells.append(f'${v:+,.0f}' if v is not None else '—')
            out.append('| ' + ' | '.join(cells) + ' |')
        out.append('')

        # Combined
        out.append('## Combined $/day by (TP, SL)')
        out.append('')
        out.append('| ' + ' | '.join(hdr) + ' |')
        out.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
        for tp in tp_values:
            cells = [f'**${tp}**']
            for sl in sl_values:
                v = next((r['comb'] for r in rows if r['tp'] == tp and r['sl'] == sl), None)
                cells.append(f'${v:+,.0f}' if v is not None else '—')
            out.append('| ' + ' | '.join(cells) + ' |')
        out.append('')

        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f'\nWrote: {OUT_MD}')
        return

    if args.sweep_r_confirm:
        r_values = [3, 5, 7.5, 10, 15]
        rows = []
        for r in r_values:
            print(f'\n--- r_confirm=${r} ---')
            is_trades, is_piv, _, _ = run_pass(is_paths, r, args.tp, args.sl,
                args.min_res_strength, args.inverse_threshold, f'IS r=${r}')
            oos_trades, oos_piv, _, _ = run_pass(oos_paths, r, args.tp, args.sl,
                args.min_res_strength, args.inverse_threshold, f'OOS r=${r}')
            is_sum = summarize(is_trades)
            oos_sum = summarize(oos_trades)
            rows.append({
                'r': r, 'is_piv': is_piv, 'oos_piv': oos_piv,
                'is': is_sum, 'oos': oos_sum,
            })
        print(f'\n{"r_confirm":>10} {"IS $/day":>10} {"OOS $/day":>10} '
              f'{"IS WR":>6} {"OOS WR":>7} {"IS $/tr":>8} {"OOS $/tr":>9} '
              f'{"IS N":>7} {"OOS N":>7}')
        for row in rows:
            is_d = row['is']['total'] / len(is_paths) if row['is'] else 0
            oos_d = row['oos']['total'] / len(oos_paths) if row['oos'] else 0
            print(f'${row["r"]:>8.1f} ${is_d:>+8,.0f} ${oos_d:>+8,.0f} '
                  f'{row["is"]["wr"]:>5.1f}% {row["oos"]["wr"]:>6.1f}% '
                  f'${row["is"]["per_trade"]:>+7.2f} ${row["oos"]["per_trade"]:>+7.2f} '
                  f'{row["is"]["n"]:>7,} {row["oos"]["n"]:>7,}')
        # MD
        out = [f'# Pivot-residual FORWARD PASS — r_confirm sweep', '']
        out.append(f'Forward pass with NO LOOKAHEAD. Pivot confirmed when price '
                   f'retraces r_confirm from the running extreme. Entry at '
                   f'confirmation bar.')
        out.append(f'TP=${args.tp} SL=${args.sl} inv_thr={args.inverse_threshold} '
                   f'min_res={args.min_res_strength}')
        out.append('')
        out.append('| r_confirm | IS pivots | IS trades | IS $/day | IS WR | IS $/tr | '
                   'OOS pivots | OOS trades | OOS $/day | OOS WR | OOS $/tr |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for row in rows:
            is_d = row['is']['total'] / len(is_paths) if row['is'] else 0
            oos_d = row['oos']['total'] / len(oos_paths) if row['oos'] else 0
            out.append(f'| ${row["r"]:.1f} | {row["is_piv"]:,} | {row["is"]["n"]:,} | '
                       f'${is_d:+,.0f} | {row["is"]["wr"]:.1f}% | '
                       f'${row["is"]["per_trade"]:+.2f} | '
                       f'{row["oos_piv"]:,} | {row["oos"]["n"]:,} | '
                       f'${oos_d:+,.0f} | {row["oos"]["wr"]:.1f}% | '
                       f'${row["oos"]["per_trade"]:+.2f} |')
        out.append('')
        for row in rows:
            out.append(f'**r_confirm=${row["r"]:.1f} exit breakdown:**')
            out.append('')
            out.append('| Reason | IS N | IS % | OOS N | OOS % |')
            out.append('|---|---:|---:|---:|---:|')
            all_r = set(row['is']['exits'].keys()) | set(row['oos']['exits'].keys())
            for reason in sorted(all_r):
                ic = row['is']['exits'].get(reason, 0)
                oc = row['oos']['exits'].get(reason, 0)
                out.append(f'| {reason} | {ic:,} | {ic/row["is"]["n"]*100:.1f}% | '
                           f'{oc:,} | {oc/row["oos"]["n"]*100:.1f}% |')
            out.append('')
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f'\nWrote: {OUT_MD}')
        return

    is_trades, is_piv, is_sk_n, is_sk_p = run_pass(
        is_paths, args.r_confirm, args.tp, args.sl,
        args.min_res_strength, args.inverse_threshold, 'IS',
        sl_vol_mult=args.sl_vol_mult)
    oos_trades, oos_piv, oos_sk_n, oos_sk_p = run_pass(
        oos_paths, args.r_confirm, args.tp, args.sl,
        args.min_res_strength, args.inverse_threshold, 'OOS',
        sl_vol_mult=args.sl_vol_mult)

    is_sum = summarize(is_trades)
    oos_sum = summarize(oos_trades)

    def p(label, s, n_days, pivots):
        if not s:
            print(f'\n{label}: no trades')
            return
        print(f'\n=== {label} ({n_days} days) ===')
        print(f'  Pivots detected: {pivots:,}  ({pivots/n_days:.0f}/day)')
        print(f'  Trades: {s["n"]:,}  ({s["n"]/n_days:.0f}/day)')
        print(f'  WR: {s["wr"]:.1f}%  ({s["wins"]:,}W / {s["losses"]:,}L)')
        print(f'  $WR: {s["dollar_wr"]:+.0f}%')
        print(f'  $/trade: ${s["per_trade"]:+.2f}')
        print(f'  Total: ${s["total"]:+,.0f}')
        print(f'  $/day:  ${s["total"]/n_days:+,.2f}')
        print(f'  Mean hold: {s["mean_hold"]:.1f} bars  median {s["median_hold"]:.0f}')
        print(f'  Exits: {s["exits"]}')

    p('IS', is_sum, len(is_paths), is_piv)
    p('OOS', oos_sum, len(oos_paths), oos_piv)

    out = [f'# Pivot-residual FORWARD PASS', '']
    out.append('Forward pass with **NO LOOKAHEAD**. Pivot confirmed when price '
               'retraces $r_confirm from running extreme; entry at the '
               'confirmation bar (not the true pivot).')
    out.append('')
    out.append(f'**Config**: r_confirm=${args.r_confirm}, TP=${args.tp}, '
               f'SL=${args.sl}, min\\_res={args.min_res_strength}, '
               f'inverse\\_thr={args.inverse_threshold}')
    out.append('')
    out.append('## Results')
    out.append('')
    out.append('| Dataset | Days | Pivots | Trades | $/day | $/trade | WR | $WR | Total $ |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for lbl, s, nd, piv in [('IS', is_sum, len(is_paths), is_piv),
                             ('OOS', oos_sum, len(oos_paths), oos_piv)]:
        if s:
            out.append(f'| {lbl} | {nd} | {piv:,} | {s["n"]:,} | '
                       f'${s["total"]/nd:+.0f} | ${s["per_trade"]:+.2f} | '
                       f'{s["wr"]:.1f}% | {s["dollar_wr"]:+.0f}% | '
                       f'${s["total"]:+,.0f} |')
    out.append('')
    out.append('## Exit breakdown')
    out.append('')
    all_r = set(is_sum['exits'].keys()) | set(oos_sum['exits'].keys())
    out.append('| Reason | IS N | IS % | OOS N | OOS % |')
    out.append('|---|---:|---:|---:|---:|')
    for reason in sorted(all_r):
        ic = is_sum['exits'].get(reason, 0)
        oc = oos_sum['exits'].get(reason, 0)
        out.append(f'| {reason} | {ic:,} | {ic/is_sum["n"]*100:.1f}% | '
                   f'{oc:,} | {oc/oos_sum["n"]*100:.1f}% |')
    out.append('')
    out.append('## Vs oracle')
    out.append('')
    out.append('| | Oracle (pivot lookahead) | Forward pass (realistic) |')
    out.append('|---|---:|---:|')
    out.append(f'| IS $/day  | +$1,358 | ${is_sum["total"]/len(is_paths):+,.0f} |')
    out.append(f'| OOS $/day | +$1,664 | ${oos_sum["total"]/len(oos_paths):+,.0f} |')
    out.append('')
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
