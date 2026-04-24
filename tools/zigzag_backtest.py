"""
Zigzag-only backtest: is there edge in pure-rule pivot trading?
================================================================

Strategy: no CNN, no features, no engine. Pure geometric rule:
  1. Compute zigzag pivots on 1m closes at threshold R (points).
  2. When a pivot is CONFIRMED (retrace R from current extreme on 1m close),
     place a market order -- fills in the NEXT 1s bar after the 1m close.
  3. On the opposite pivot confirmation, reverse (exit + new entry).

Two PnL models reported per R threshold:

  THEORETICAL  (upper-bound, idealized):
    Entry at (extreme +- R), exit at (next_extreme +- R).
    Profit per leg = leg_magnitude - 2*R.

  REALISTIC  (with slippage + commission):
    Entry fill  = uniform random in [low, high] of the 1s bar immediately
                  following the 1m confirmation close.
    Exit fill   = same sampling method on the 1s bar following next
                  confirmation.
    Commission  = $1 entry + $1 exit per contract = $2 round-trip.
    Net $ per trade = (exit_fill - entry_fill) * sign * $2/pt - $2.

  Fallback: if the 1s bar is missing for a required second (thin session),
  fall through to the 1m bar's [low, high] for that sample.

Usage:
    python tools/zigzag_backtest.py                    # ATLAS, 8-way R sweep, both models
    python tools/zigzag_backtest.py --atlas DATA/ATLAS_NT8
    python tools/zigzag_backtest.py --r 20 30 50
    python tools/zigzag_backtest.py --no-realistic     # skip 1s-bar loading for speed
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# MNQ contract
DOLLAR_PER_POINT = 2.0

# Execution costs (realistic model only)
COMMISSION_PER_SIDE_USD = 1.0     # per contract per side -> $2 round-trip
SLIPPAGE_SEED = 42                # reproducible random fills

DEFAULT_R_POINTS = [5, 10, 15, 20, 30, 50, 75, 100]
DEFAULT_ATLAS = 'DATA/ATLAS'


# ─── Zigzag ───────────────────────────────────────────────────────────────

def zigzag_pivots_with_confirmation(closes: np.ndarray, r: float):
    """Zigzag pivots with CONFIRMATION bar index.

    Returns list of (extreme_idx, extreme_price, kind, confirm_idx):
      extreme_idx  = bar where the pivot's extreme occurred
      extreme_price = price at the extreme
      kind          = 'high' or 'low'
      confirm_idx   = bar index where retracement of R was confirmed (always > extreme_idx)
    """
    n = len(closes)
    if n < 2:
        return []

    pivots = []
    direction = None
    ext_idx = 0
    ext_price = float(closes[0])

    for i in range(1, n):
        c = float(closes[i])
        if direction is None:
            if c - ext_price >= r:
                pivots.append((ext_idx, ext_price, 'low', i))
                direction = 'up'
                ext_idx, ext_price = i, c
            elif ext_price - c >= r:
                pivots.append((ext_idx, ext_price, 'high', i))
                direction = 'down'
                ext_idx, ext_price = i, c
        elif direction == 'up':
            if c > ext_price:
                ext_idx, ext_price = i, c
            elif ext_price - c >= r:
                pivots.append((ext_idx, ext_price, 'high', i))
                direction = 'down'
                ext_idx, ext_price = i, c
        else:  # down
            if c < ext_price:
                ext_idx, ext_price = i, c
            elif c - ext_price >= r:
                pivots.append((ext_idx, ext_price, 'low', i))
                direction = 'up'
                ext_idx, ext_price = i, c

    return pivots


def theoretical_leg_profits(pivots, r: float) -> list[float]:
    """Idealized profit per leg in POINTS: leg_mag - 2*R."""
    profits = []
    for i in range(len(pivots) - 1):
        _, p1, _, _ = pivots[i]
        _, p2, _, _ = pivots[i + 1]
        profits.append(abs(p1 - p2) - 2.0 * r)
    return profits


# ─── Realistic fill simulation ────────────────────────────────────────────

def _random_fill_in_1s_bar(ts_target: int,
                            ts_1s: np.ndarray,
                            low_1s: np.ndarray,
                            high_1s: np.ndarray,
                            df_1m_fallback: dict,
                            rng: np.random.Generator) -> float | None:
    """Uniform-random fill price in [low, high] of the 1s bar at ts_target.

    If the 1s bar for ts_target is missing (thin trading second), fall back
    to uniform random in the 1m bar's [low, high] containing ts_target.
    Returns None if nothing available.
    """
    # Binary search the 1s bar
    idx = np.searchsorted(ts_1s, ts_target)
    if idx < len(ts_1s) and ts_1s[idx] == ts_target:
        lo, hi = low_1s[idx], high_1s[idx]
    else:
        # Fallback: use the 1m bar containing ts_target
        ts_1m_start = (ts_target // 60) * 60
        if ts_1m_start not in df_1m_fallback:
            return None
        lo, hi = df_1m_fallback[ts_1m_start]

    if hi <= lo:
        return float(lo)
    return float(rng.uniform(lo, hi))


def realistic_leg_pnl_usd(pivots,
                           ts_1m: np.ndarray,
                           df_1s: pd.DataFrame | None,
                           df_1m_lows: np.ndarray,
                           df_1m_highs: np.ndarray,
                           rng: np.random.Generator,
                           commission_per_side: float) -> list[float]:
    """Per-leg $ PnL with random-in-1s-bar fills and commission.

    For each consecutive pivot pair (p_i, p_{i+1}):
      entry 1s bar  = 1s bar at ts_1m[confirm_idx_of_p_i] + 60  (first sec of next 1m)
      exit  1s bar  = 1s bar at ts_1m[confirm_idx_of_p_{i+1}] + 60
      direction     = long if p_i is 'low' else short

    Missing 1s bar -> fallback to 1m bar OHLC random sample.
    Commission applied at $commission_per_side each side -> 2x round-trip.
    """
    if df_1s is not None and len(df_1s) > 0:
        ts_1s = df_1s['timestamp'].values.astype(np.int64)
        low_1s = df_1s['low'].values.astype(np.float64)
        high_1s = df_1s['high'].values.astype(np.float64)
    else:
        ts_1s = np.array([], dtype=np.int64)
        low_1s = np.array([], dtype=np.float64)
        high_1s = np.array([], dtype=np.float64)

    # Build {1m_ts: (low, high)} for fallback
    fallback_1m = {int(ts_1m[i]): (float(df_1m_lows[i]), float(df_1m_highs[i]))
                    for i in range(len(ts_1m))}

    pnls = []
    for i in range(len(pivots) - 1):
        _, _, kind_entry, confirm_idx_entry = pivots[i]
        _, _, _, confirm_idx_exit = pivots[i + 1]
        if confirm_idx_entry >= len(ts_1m) or confirm_idx_exit >= len(ts_1m):
            continue

        entry_ts = int(ts_1m[confirm_idx_entry]) + 60
        exit_ts = int(ts_1m[confirm_idx_exit]) + 60

        entry_fill = _random_fill_in_1s_bar(entry_ts, ts_1s, low_1s, high_1s, fallback_1m, rng)
        exit_fill = _random_fill_in_1s_bar(exit_ts, ts_1s, low_1s, high_1s, fallback_1m, rng)
        if entry_fill is None or exit_fill is None:
            continue

        # Direction: 'low' pivot -> next leg is UP -> LONG
        #            'high' pivot -> next leg is DOWN -> SHORT
        is_long = (kind_entry == 'low')
        pts = (exit_fill - entry_fill) if is_long else (entry_fill - exit_fill)
        usd = pts * DOLLAR_PER_POINT - 2.0 * commission_per_side
        pnls.append(usd)

    return pnls


# ─── Day loading ──────────────────────────────────────────────────────────

def _list_1m_days(atlas_root: str) -> list[tuple[str, str]]:
    pat = os.path.join(atlas_root, '1m', '*.parquet')
    files = sorted(glob.glob(pat))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in files]


def _is_sample(day_label: str) -> bool:
    return day_label.startswith('2025_')


def _is_oos(day_label: str) -> bool:
    return day_label.startswith('2026_')


def _load_1s_for_day(atlas_root: str, day: str) -> pd.DataFrame | None:
    p = os.path.join(atlas_root, '1s', f'{day}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


# ─── Main backtest driver ──────────────────────────────────────────────────

def backtest_one_R(r: float,
                   day_paths: list[tuple[str, str]],
                   atlas_root: str,
                   realistic: bool,
                   commission_per_side: float) -> dict:
    """Aggregate stats at a single R threshold across all days."""
    rng = np.random.default_rng(SLIPPAGE_SEED)

    # Theoretical totals
    theo_profits_is_pts: list[float] = []
    theo_profits_oos_pts: list[float] = []
    theo_daily_is: list[float] = []
    theo_daily_oos: list[float] = []

    # Realistic totals
    real_pnl_is_usd: list[float] = []
    real_pnl_oos_usd: list[float] = []
    real_daily_is: list[float] = []
    real_daily_oos: list[float] = []

    total_legs = 0
    n_days_is, n_days_oos = 0, 0

    for day, path in day_paths:
        try:
            df1m = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
        except Exception:
            continue
        if len(df1m) < 2:
            continue
        closes = df1m['close'].values.astype(np.float64)
        ts_1m = df1m['timestamp'].values.astype(np.int64)
        lows_1m = df1m['low'].values.astype(np.float64)
        highs_1m = df1m['high'].values.astype(np.float64)

        pivots = zigzag_pivots_with_confirmation(closes, r)
        if len(pivots) < 2:
            continue

        theo_profits = theoretical_leg_profits(pivots, r)
        total_legs += len(theo_profits)

        day_theo_pts = float(sum(theo_profits))
        day_real_usd = 0.0

        if realistic:
            df1s = _load_1s_for_day(atlas_root, day)
            real_usds = realistic_leg_pnl_usd(pivots, ts_1m, df1s, lows_1m, highs_1m,
                                               rng, commission_per_side)
            day_real_usd = float(sum(real_usds))
        else:
            real_usds = []

        if _is_sample(day):
            n_days_is += 1
            theo_profits_is_pts.extend(theo_profits)
            theo_daily_is.append(day_theo_pts)
            if realistic:
                real_pnl_is_usd.extend(real_usds)
                real_daily_is.append(day_real_usd)
        elif _is_oos(day):
            n_days_oos += 1
            theo_profits_oos_pts.extend(theo_profits)
            theo_daily_oos.append(day_theo_pts)
            if realistic:
                real_pnl_oos_usd.extend(real_usds)
                real_daily_oos.append(day_real_usd)

    def _aggregate(profits, daily, n_days, per_point=False):
        if not profits or n_days == 0:
            return dict(total=0.0, per_day=0.0, wr=0.0, day_wr=0.0, n=0, median=0.0)
        arr = np.array(profits)
        daily_arr = np.array(daily)
        mult = DOLLAR_PER_POINT if per_point else 1.0
        return dict(
            total=float(arr.sum() * mult),
            per_day=float(arr.sum() * mult / n_days),
            wr=float(100.0 * (arr > 0).mean()),
            day_wr=float(100.0 * (daily_arr > 0).mean()),
            n=len(arr),
            median=float(np.median(arr) * mult),
        )

    theo_is = _aggregate(theo_profits_is_pts, theo_daily_is, n_days_is, per_point=True)
    theo_oos = _aggregate(theo_profits_oos_pts, theo_daily_oos, n_days_oos, per_point=True)
    real_is = _aggregate(real_pnl_is_usd, real_daily_is, n_days_is)
    real_oos = _aggregate(real_pnl_oos_usd, real_daily_oos, n_days_oos)

    return {
        'R_pts': r,
        'n_days_is': n_days_is,
        'n_days_oos': n_days_oos,
        'total_legs': total_legs,
        'legs_per_day': total_legs / max(n_days_is + n_days_oos, 1),
        'theo_is_per_day_usd': theo_is['per_day'],
        'theo_oos_per_day_usd': theo_oos['per_day'],
        'theo_is_day_wr_pct': theo_is['day_wr'],
        'theo_oos_day_wr_pct': theo_oos['day_wr'],
        'real_is_per_day_usd': real_is['per_day'],
        'real_oos_per_day_usd': real_oos['per_day'],
        'real_is_day_wr_pct': real_is['day_wr'],
        'real_oos_day_wr_pct': real_oos['day_wr'],
        'real_is_trade_wr_pct': real_is['wr'],
        'real_oos_trade_wr_pct': real_oos['wr'],
        'real_median_trade_usd': real_oos['median'] if real_oos['n'] > 0 else real_is['median'],
    }


def main():
    ap = argparse.ArgumentParser(description='Zigzag-only backtest (pure rule)')
    ap.add_argument('--atlas', default=DEFAULT_ATLAS)
    ap.add_argument('--r', type=float, nargs='+', default=DEFAULT_R_POINTS,
                    help='R thresholds in POINTS. MNQ: 1 pt = $2.')
    ap.add_argument('--commission', type=float, default=COMMISSION_PER_SIDE_USD,
                    help='Per-side commission in $ (default $1 = $2 round-trip)')
    ap.add_argument('--no-realistic', action='store_true',
                    help='Skip realistic simulation (faster; theoretical only)')
    args = ap.parse_args()

    day_paths = _list_1m_days(args.atlas)
    if not day_paths:
        print(f'No 1m parquets under {args.atlas}/1m/')
        return

    realistic = not args.no_realistic
    print(f'Zigzag backtest -- {len(day_paths)} day files under {args.atlas}/1m/')
    print(f'R sweep (points): {args.r}')
    print(f'MNQ: 1 point = ${DOLLAR_PER_POINT}')
    if realistic:
        print(f'Realistic model: 1s-bar uniform-random fills, '
              f'${args.commission}/side commission (= ${2 * args.commission} round-trip)')
    else:
        print('Theoretical only (use --no-realistic to skip 1s-bar loading)')
    print()

    rows = []
    for r in tqdm(args.r, desc='R sweep'):
        rows.append(backtest_one_R(r, day_paths, args.atlas, realistic, args.commission))

    df = pd.DataFrame(rows)

    print()
    print('=' * 130)
    header = (f'{"R(pts)":>6} {"R($)":>6}  {"legs/day":>8}  '
              f'{"theo_IS":>9} {"theo_OOS":>9}  '
              f'{"real_IS":>9} {"real_OOS":>9}  '
              f'{"real_IS_dWR":>11} {"real_OOS_dWR":>12}  '
              f'{"real_tWR":>9} {"med_trade":>10}')
    print(header)
    print('-' * 130)
    for r in rows:
        print(f'{r["R_pts"]:>6.0f} {r["R_pts"]*DOLLAR_PER_POINT:>5.0f}$  '
              f'{r["legs_per_day"]:>8.1f}  '
              f'${r["theo_is_per_day_usd"]:>+8.0f} ${r["theo_oos_per_day_usd"]:>+8.0f}  '
              f'${r["real_is_per_day_usd"]:>+8.0f} ${r["real_oos_per_day_usd"]:>+8.0f}  '
              f'{r["real_is_day_wr_pct"]:>10.1f}% {r["real_oos_day_wr_pct"]:>11.1f}%  '
              f'{r["real_oos_trade_wr_pct"]:>7.1f}%  ${r["real_median_trade_usd"]:>+8.2f}')
    print('=' * 130)
    print()
    print('Column key:')
    print('  theo_* : idealized (extreme +- R fills, no commission)  -- UPPER BOUND')
    print('  real_* : 1s-bar random fill + $2 round-trip commission -- DEPLOYABLE ESTIMATE')
    print('  dWR    : day win rate   tWR : trade win rate   med_trade : median $ per trade (OOS)')

    # Save
    out_csv = 'reports/findings/zigzag_backtest.csv'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')


if __name__ == '__main__':
    main()
