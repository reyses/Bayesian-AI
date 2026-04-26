"""
zigzag_5s_perfect_fill.py -- 5s-grid simulator with perfect deterministic
fills + configurable commission and slippage.

Model:
  Decision cadence  : 1m close detects pivots (unchanged from NT8 v1.0/v1.2-RC)
  Entry fill        : EXACT 5s close immediately after pivot 1m bar closes
  Trail check       : every 5s close (5s primary cadence simulated)
  Trail exit fill   : EXACT 5s close where trail breach detected
  Pivot exit fill   : EXACT 5s close immediately after opposite-pivot 1m bar
  EOD               : EXACT 5s close at EOD threshold
  HWM tracking      : 1m closes (matches NT8 v1.2-RC behavior)
  Commission        : --commission   USD round-trip (default $2)
  Slippage          : --slippage     USD round-trip (default $1 = 1 tick each side)
  Total trade cost  : commission + slippage (default $3 round-trip)

Compared to the bar-by-bar 1s tick simulator, this version:
  - Eliminates within-1s-bar random fill variance (deterministic)
  - Decisions on 5s grid (less aggressive than 1s, more aggressive than 1m)
  - Models slippage as a fixed tax (configurable)

Usage:
    python tools/zigzag_5s_perfect_fill.py all --r 30 --sl-pts 10
    python tools/zigzag_5s_perfect_fill.py all --r 30 --commission 2 --slippage 2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.zigzag_backtest import zigzag_pivots_with_confirmation

DOLLAR_PER_POINT = 2.0
DEFAULT_R                  = 30.0
DEFAULT_TRAIL_ACTIVATE_PTS = 10.0
DEFAULT_TRAIL_DIST_PTS     = 5.0
DEFAULT_TRAIL_PCT          = 0.10
DEFAULT_COMMISSION_USD     = 2.0
DEFAULT_SLIPPAGE_USD       = 1.0
EOD_HOUR_UTC, EOD_MIN_UTC = 20, 55
ENTRY_CUTOFF_HOUR_UTC, ENTRY_CUTOFF_MIN_UTC = 20, 30
ATLAS_ROOT = 'DATA/ATLAS'


def simulate_day(day_label: str, r: float,
                 trail_activate: float, trail_dist: float, trail_pct: float,
                 sl_pts: float, commission: float, slippage: float,
                 atlas_root: str = ATLAS_ROOT) -> tuple[list[dict], dict]:
    p_1m = os.path.join(atlas_root, '1m', f'{day_label}.parquet')
    p_5s = os.path.join(atlas_root, '5s', f'{day_label}.parquet')
    if not (os.path.exists(p_1m) and os.path.exists(p_5s)):
        return [], dict(day=day_label, n_trades=0)

    df_1m = pd.read_parquet(p_1m).sort_values('timestamp').reset_index(drop=True)
    df_5s = pd.read_parquet(p_5s).sort_values('timestamp').reset_index(drop=True)
    if len(df_1m) < 2 or len(df_5s) < 12:
        return [], dict(day=day_label, n_trades=0)

    closes_1m = df_1m['close'].values.astype(np.float64)
    ts_1m     = df_1m['timestamp'].values.astype(np.int64)
    pivots = zigzag_pivots_with_confirmation(closes_1m, r)
    if len(pivots) < 2:
        return [], dict(day=day_label, n_trades=0)

    # Map 1m confirm bar -> (kind, ext_price)
    pivot_by_confirm = {int(c): (k, float(p)) for _, p, k, c in pivots}

    ts_5s     = df_5s['timestamp'].values.astype(np.int64)
    closes_5s = df_5s['close'].values.astype(np.float64)
    highs_5s  = df_5s['high'].values.astype(np.float64)
    lows_5s   = df_5s['low'].values.astype(np.float64)
    n_5s = len(ts_5s)
    cost_per_trade = commission + slippage

    trades: list[dict] = []
    n_skipped = n_trail = n_pivot = n_eod = n_sl = 0

    in_pos = False
    direction = 0
    entry_price = 0.0
    entry_ts = 0
    peak_close_pts = 0.0
    peak_close_price = 0.0
    peak_pnl_pts = 0.0
    worst_pnl_pts = 0.0
    trail_armed = False
    pending_entry: tuple[str, float, int] | None = None  # (kind, ext_price, fire_at_ts)

    def find_5s_close_at_or_after(t):
        idx = int(np.searchsorted(ts_5s, t, side='left'))
        return idx if idx < n_5s else -1

    # Walk pivots in order; for each, set entry pending; then walk 5s bars
    # forward until next-pivot or trail/SL/EOD fires.
    for i in range(len(pivots) - 1):
        ext_idx, ext_price, kind, confirm_idx = pivots[i]
        _, _, _, next_confirm_idx = pivots[i + 1]

        # Entry: first 5s bar whose timestamp > 1m confirm bar close
        confirm_ts_1m = int(ts_1m[confirm_idx]) + 60
        entry_idx_5s = find_5s_close_at_or_after(confirm_ts_1m)
        if entry_idx_5s < 0: continue

        # Past entry cutoff?
        entry_ts_actual = int(ts_5s[entry_idx_5s])
        entry_dt = datetime.fromtimestamp(entry_ts_actual, tz=timezone.utc)
        mins = entry_dt.hour * 60 + entry_dt.minute
        if mins >= ENTRY_CUTOFF_HOUR_UTC * 60 + ENTRY_CUTOFF_MIN_UTC:
            n_skipped += 1
            continue

        # Place entry
        direction = +1 if kind == 'low' else -1
        entry_price = float(closes_5s[entry_idx_5s])
        entry_ts = entry_ts_actual
        peak_close_pts = 0.0
        peak_close_price = entry_price
        peak_pnl_pts = 0.0
        worst_pnl_pts = 0.0
        trail_armed = False
        in_pos = True

        # Walk forward 5s bars until next-pivot's confirm or trail/SL/EOD
        next_pivot_confirm_ts = int(ts_1m[next_confirm_idx]) + 60
        end_idx_for_pivot = find_5s_close_at_or_after(next_pivot_confirm_ts)
        if end_idx_for_pivot < 0:
            end_idx_for_pivot = n_5s - 1

        # SL price level
        if sl_pts > 0:
            sl_price = entry_price - direction * sl_pts
        else:
            sl_price = None

        exit_idx_5s = None
        exit_reason = None

        for j in range(entry_idx_5s + 1, n_5s):
            tick_close = float(closes_5s[j])
            tick_high = float(highs_5s[j])
            tick_low  = float(lows_5s[j])
            tick_ts = int(ts_5s[j])

            # MFE / MAE tracking on 5s closes
            if direction > 0:
                pnl = tick_close - entry_price
                worst = tick_low - entry_price
            else:
                pnl = entry_price - tick_close
                worst = entry_price - tick_high
            if pnl > peak_pnl_pts: peak_pnl_pts = pnl
            if worst < worst_pnl_pts: worst_pnl_pts = worst

            # SL check (intra-5s, on tick high/low)
            if sl_price is not None:
                if direction > 0 and tick_low <= sl_price:
                    exit_idx_5s = j
                    exit_reason = 'sl'
                    break
                elif direction < 0 and tick_high >= sl_price:
                    exit_idx_5s = j
                    exit_reason = 'sl'
                    break

            # HWM update on 5s close (instead of 1m close — slightly more aggressive
            # but keeps trail logic intact)
            if direction > 0 and tick_close > peak_close_price:
                peak_close_price = tick_close
                peak_close_pts = tick_close - entry_price
            elif direction < 0 and tick_close < peak_close_price:
                peak_close_price = tick_close
                peak_close_pts = entry_price - tick_close

            # Trail arm
            if trail_activate > 0 and not trail_armed and peak_close_pts >= trail_activate:
                trail_armed = True

            # Trail breach
            if trail_armed:
                eff_dist = max(trail_dist, trail_pct * peak_close_pts)
                if direction > 0:
                    stop_px = peak_close_price - eff_dist
                    if tick_close <= stop_px:
                        exit_idx_5s = j
                        exit_reason = 'trail'
                        break
                else:
                    stop_px = peak_close_price + eff_dist
                    if tick_close >= stop_px:
                        exit_idx_5s = j
                        exit_reason = 'trail'
                        break

            # EOD check
            tick_dt = datetime.fromtimestamp(tick_ts, tz=timezone.utc)
            mins_now = tick_dt.hour * 60 + tick_dt.minute
            if mins_now >= EOD_HOUR_UTC * 60 + EOD_MIN_UTC:
                exit_idx_5s = j
                exit_reason = 'eod'
                break

            # Reach next pivot's confirm 1m close — natural pivot exit
            if j >= end_idx_for_pivot:
                exit_idx_5s = j
                exit_reason = 'pivot'
                break

        if exit_idx_5s is None:
            exit_idx_5s = n_5s - 1
            exit_reason = 'eod_final'

        # Compute trade outcome with deterministic 5s close fill
        if exit_reason == 'sl':
            exit_price = sl_price
        else:
            exit_price = float(closes_5s[exit_idx_5s])
        exit_ts_actual = int(ts_5s[exit_idx_5s])

        pnl_pts = direction * (exit_price - entry_price)
        pnl_usd = pnl_pts * DOLLAR_PER_POINT - cost_per_trade

        trades.append({
            'day': day_label,
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'exit_ts': exit_ts_actual,
            'exit_price': float(exit_price),
            'direction': direction,
            'pnl_pts': float(pnl_pts),
            'pnl_usd': float(pnl_usd),
            'mfe_pts': float(peak_pnl_pts),
            'mae_pts': float(-worst_pnl_pts),
            'leg_min': (exit_ts_actual - entry_ts) / 60.0,
            'exit_reason': exit_reason,
        })
        if exit_reason == 'trail': n_trail += 1
        elif exit_reason == 'pivot': n_pivot += 1
        elif exit_reason == 'sl': n_sl += 1
        else: n_eod += 1
        in_pos = False

    summary = dict(
        day=day_label,
        n_trades=len(trades),
        n_skipped=n_skipped,
        n_trail=n_trail, n_pivot=n_pivot, n_eod=n_eod, n_sl=n_sl,
        total_usd=sum(t['pnl_usd'] for t in trades),
    )
    return trades, summary


def is_2025(d): return d.startswith('2025_')
def is_2026(d): return d.startswith('2026_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('day', nargs='?', default='all')
    ap.add_argument('--atlas', default=ATLAS_ROOT)
    ap.add_argument('--r', type=float, default=DEFAULT_R)
    ap.add_argument('--trail-activate', type=float, default=DEFAULT_TRAIL_ACTIVATE_PTS)
    ap.add_argument('--trail-dist', type=float, default=DEFAULT_TRAIL_DIST_PTS)
    ap.add_argument('--trail-pct', type=float, default=DEFAULT_TRAIL_PCT)
    ap.add_argument('--sl-pts', type=float, default=10.0,
                    help='Hard SL points (default 10 = v1.2-RC default; 0 disables)')
    ap.add_argument('--commission', type=float, default=DEFAULT_COMMISSION_USD,
                    help='Round-trip commission USD per contract (default 2)')
    ap.add_argument('--slippage', type=float, default=DEFAULT_SLIPPAGE_USD,
                    help='Round-trip slippage tax USD per contract (default 1 = 1 tick each side)')
    ap.add_argument('--out', default='reports/findings/zigzag_5s_perfect_fill.csv')
    args = ap.parse_args()

    if args.day == 'all':
        files = sorted(glob.glob(os.path.join(args.atlas, '1m', '*.parquet')))
        days = [os.path.splitext(os.path.basename(p))[0] for p in files]
    else:
        days = [args.day.replace('-', '_')]

    print('=' * 110)
    print(f'V1.2-RC 5s PERFECT-FILL SIMULATOR')
    print(f'R={args.r:g}  trail_act={args.trail_activate:g}  trail_dist={args.trail_dist:g}  '
          f'trail_pct={args.trail_pct:g}  SL={args.sl_pts:g}pt')
    print(f'Commission ${args.commission:g} round-trip + slippage ${args.slippage:g} '
          f'= ${args.commission+args.slippage:g} total tax per trade')
    print(f'Days: {len(days)}')
    print('=' * 110)

    all_trades: list[dict] = []
    iterator = days if len(days) == 1 else tqdm(days, desc='days')
    for d in iterator:
        trades, _summary = simulate_day(
            d, args.r, args.trail_activate, args.trail_dist, args.trail_pct,
            args.sl_pts, args.commission, args.slippage, atlas_root=args.atlas)
        all_trades.extend(trades)

    if not all_trades:
        print('No trades.')
        return

    df = pd.DataFrame(all_trades)
    daily = df.groupby('day')['pnl_usd'].sum()
    is_d  = daily[daily.index.map(is_2025)]
    oos_d = daily[daily.index.map(is_2026)]

    def stats(s):
        if len(s) == 0: return None
        a = s.values
        return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                    dWR=float((a>0).mean()*100), best=float(a.max()), worst=float(a.min()),
                    std=float(a.std()))

    print(f'\nTotal trades: {len(df):,}  total $: {df["pnl_usd"].sum():+.0f}')
    print(f'\nEXIT REASONS')
    for reason, sub in df.groupby('exit_reason'):
        wr = (sub['pnl_usd']>0).mean()*100
        print(f'  {reason:>12}  N={len(sub):>5,}  WR={wr:>5.1f}%  '
              f'mean=${sub["pnl_usd"].mean():>+7.2f}  '
              f'mean_MFE=${sub["mfe_pts"].mean()*DOLLAR_PER_POINT:>+7.2f}')

    print(f'\nDAILY P&L')
    s_is, s_oos = stats(is_d), stats(oos_d)
    if s_is:
        print(f'  IS  {s_is["n"]:>3} days  mean=${s_is["mean"]:+8.2f}/d  median=${s_is["median"]:+7.2f}  '
              f'dWR={s_is["dWR"]:5.1f}%  best=${s_is["best"]:+5.0f}  worst=${s_is["worst"]:+6.0f}  std=${s_is["std"]:5.0f}')
    if s_oos:
        print(f'  OOS {s_oos["n"]:>3} days  mean=${s_oos["mean"]:+8.2f}/d  median=${s_oos["median"]:+7.2f}  '
              f'dWR={s_oos["dWR"]:5.1f}%  best=${s_oos["best"]:+5.0f}  worst=${s_oos["worst"]:+6.0f}  std=${s_oos["std"]:5.0f}')

    # PF / dollar-share WR
    p = df.loc[df["pnl_usd"]>0, "pnl_usd"].sum()
    l = abs(df.loc[df["pnl_usd"]<0, "pnl_usd"].sum())
    pf = p / max(l, 1)
    print(f'\nProfit factor (sum_profit/|sum_loss|): {pf:.4f}  '
          f'(=> dollar-share WR = {pf/(pf+1)*100:.2f}%)')
    print(f'PF-based Trade WR: {pf - 1:+.4f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
