"""Extend OOS sample with May 2026 NT8 days.

For each new May day (where ATLAS_NT8/{1m,5s}/2026_05_*.parquet exists):
  1. Detect pivots inline using detect_swings on 5s closes (NT8 data)
  2. Run hardened forward pass: R-trigger entries/exits, flat sizing, $6 friction
  3. Aggregate per-day P&L

Combines with existing OOS hardened CSV (composite_forward_pass_hardened.csv,
which has the 23 days from 2026-03-20 to 2026-04-26) to produce an
expanded OOS leg list + per-day P&L.

Uses the FLAT target (no gbm_ev sizing because B7 requires V2 features
not yet computed for May days). For the DRS evaluation, we'll also need
to recompute the IS target as flat to maintain apples-to-apples.

Output:
  reports/findings/drs/oos_extended_hardened_legs.csv (34 days)
  reports/findings/drs/oos_extended_day_pnl.csv (per-day aggregate)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_swings(close: np.ndarray, min_reversal: float, min_bars: int,
                    max_bars: int = 0) -> list:
    """Inlined ZigZag pivot detector (from auto_swing_marker.detect_swings).
    Tracks running high/low. When price reverses from extreme by >= min_reversal
    ticks AND at least min_bars elapsed, the extreme becomes a pivot.
    """
    n = len(close)
    if n < 3:
        return []
    ct = close / 0.25  # work in tick space
    pivots = [0]
    direction = 0
    extreme_idx = 0
    extreme_val = ct[0]
    for i in range(1, n):
        price = ct[i]
        last_pivot = pivots[-1]
        if max_bars > 0 and direction != 0 and i - last_pivot >= max_bars:
            pivot_val = ct[last_pivot]
            swing_move = abs(extreme_val - pivot_val)
            if swing_move >= min_reversal and extreme_idx > last_pivot:
                pivots.append(extreme_idx)
                direction = -direction
                extreme_val = price
                extreme_idx = i
                continue
        if direction == 0:
            if price > extreme_val:
                extreme_val = price; extreme_idx = i
            if price < ct[0] and ct[0] - price >= min_reversal:
                direction = -1; extreme_val = price; extreme_idx = i
            elif price > ct[0] and price - ct[0] >= min_reversal:
                direction = 1; extreme_val = price; extreme_idx = i
        elif direction == 1:
            if price >= extreme_val:
                extreme_val = price; extreme_idx = i
            elif extreme_val - price >= min_reversal and i - extreme_idx >= min_bars:
                pivots.append(extreme_idx)
                direction = -1; extreme_val = price; extreme_idx = i
        elif direction == -1:
            if price <= extreme_val:
                extreme_val = price; extreme_idx = i
            elif price - extreme_val >= min_reversal and i - extreme_idx >= min_bars:
                pivots.append(extreme_idx)
                direction = 1; extreme_val = price; extreme_idx = i
    if pivots[-1] != n - 1:
        pivots.append(n - 1)
    return pivots


TICK_SIZE = 0.25
DOLLAR_PER_POINT = 2.0
FRICTION_PER_LEG = 6.0
ATR_PERIOD = 14
ATR_MULT = 4.0
MIN_BARS = 36
MAX_LOOKAHEAD_MIN = 120

NT8_5S = Path('DATA/ATLAS_NT8/5s')
NT8_1M = Path('DATA/ATLAS_NT8/1m')


def compute_atr(bars1m: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    h = bars1m['high'].values
    l = bars1m['low'].values
    c = bars1m['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())


def detect_r_trigger_fire(closes5s, ts5s, pivot_ts, pivot_price, leg_dir,
                            r_price, max_lookahead_min=MAX_LOOKAHEAD_MIN):
    end_ts = pivot_ts + max_lookahead_min * 60
    mask = (ts5s > pivot_ts) & (ts5s <= end_ts)
    sub_closes = closes5s[mask]
    sub_ts = ts5s[mask]
    if len(sub_closes) == 0:
        return None, None
    if leg_dir == 'LONG':
        hits = np.where(sub_closes >= pivot_price + r_price)[0]
    else:
        hits = np.where(sub_closes <= pivot_price - r_price)[0]
    if len(hits) == 0:
        return None, None
    return int(sub_ts[hits[0]]), float(sub_closes[hits[0]])


def process_day(day: str) -> list:
    bars1m_path = NT8_1M / f'{day}.parquet'
    bars5s_path = NT8_5S / f'{day}.parquet'
    if not bars1m_path.exists() or not bars5s_path.exists():
        return None

    bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
    bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
    if len(bars5s) < 100:
        return []

    atr_pts = compute_atr(bars1m, ATR_PERIOD)
    min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * ATR_MULT)))
    r_price = min_rev_ticks * TICK_SIZE

    closes5s = bars5s['close'].values.astype(np.float64)
    ts5s = bars5s['timestamp'].values.astype(np.int64)

    pivot_idxs = detect_swings(closes5s, min_reversal=min_rev_ticks,
                                min_bars=MIN_BARS, max_bars=0)
    if len(pivot_idxs) < 3:
        return []

    pivots = []
    for k, i in enumerate(pivot_idxs):
        if k + 1 >= len(pivot_idxs):
            break
        next_i = pivot_idxs[k + 1]
        direction = 'LONG' if closes5s[next_i] > closes5s[i] else 'SHORT'
        pivots.append((int(ts5s[i]), float(closes5s[i]), direction))

    rtrig_fires = []
    for (piv_ts, piv_price, piv_dir) in pivots:
        fire_ts, fire_price = detect_r_trigger_fire(
            closes5s, ts5s, piv_ts, piv_price, piv_dir, r_price)
        if fire_ts is None:
            continue
        rtrig_fires.append({'pivot_ts': piv_ts, 'pivot_dir': piv_dir,
                            'pivot_price': piv_price,
                            'fire_ts': fire_ts, 'fire_price': fire_price})

    rows = []
    for k in range(len(rtrig_fires) - 1):
        entry = rtrig_fires[k]
        exit_ = rtrig_fires[k + 1]
        leg_dir = entry['pivot_dir']
        entry_ts = entry['fire_ts']
        entry_price = entry['fire_price']
        exit_ts = exit_['fire_ts']
        exit_price = exit_['fire_price']
        if leg_dir == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price
        pnl_usd = pnl_pts * DOLLAR_PER_POINT - FRICTION_PER_LEG
        rows.append({
            'day': day, 'entry_ts': entry_ts, 'leg_dir': leg_dir,
            'entry_price': entry_price, 'exit_ts': exit_ts,
            'exit_price': exit_price, 'pnl_pts': pnl_pts,
            'pnl_usd': pnl_usd, 'r_price': r_price, 'atr_pts': atr_pts,
        })
    return rows


def main():
    # Find new May days (with NT8 parquets, not in existing OOS hardened CSV)
    existing_oos = pd.read_csv('reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    existing_days = set(existing_oos['day'].unique())
    print(f'Existing OOS days: {len(existing_days)} (range: '
          f'{min(existing_days)} to {max(existing_days)})')

    nt8_days = sorted({p.stem for p in NT8_5S.glob('2026_*.parquet')})
    new_days = sorted(set(nt8_days) - existing_days)
    print(f'Available NT8 days: {len(nt8_days)}')
    print(f'New days for extension: {len(new_days)}')
    print(f'  {new_days}')

    all_rows = []
    for day in tqdm(new_days, desc='process new days'):
        rows = process_day(day)
        if rows is None:
            print(f'  SKIP {day}: missing bars')
            continue
        if len(rows) == 0:
            print(f'  SKIP {day}: <3 pivots or no legs')
            continue
        all_rows.extend(rows)

    new_legs = pd.DataFrame(all_rows)
    print(f'\nNew legs: {len(new_legs):,} across {new_legs["day"].nunique()} days')

    # Combine with existing OOS hardened CSV (flat pnl_usd column)
    existing_oos_simple = existing_oos[['day', 'entry_ts', 'leg_dir', 'entry_price',
                                          'exit_ts', 'exit_price', 'pnl_pts',
                                          'pnl_usd', 'r_price', 'atr_pts']].copy()
    combined = pd.concat([existing_oos_simple, new_legs],
                          ignore_index=True).sort_values(['day', 'entry_ts'])
    print(f'Combined OOS legs: {len(combined):,} across {combined["day"].nunique()} days')

    out_path = Path('reports/findings/drs/oos_extended_hardened_legs.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f'Wrote: {out_path}')

    # Per-day aggregate
    per_day = combined.groupby('day')['pnl_usd'].agg(['sum', 'count']).reset_index()
    per_day.columns = ['day', 'day_pnl_flat', 'n_legs']
    per_day_path = Path('reports/findings/drs/oos_extended_day_pnl.csv')
    per_day.to_csv(per_day_path, index=False)
    print(f'Wrote: {per_day_path}')

    print()
    print('=== Per-day P&L summary (flat sizing) ===')
    print(f'Days: {len(per_day)}')
    print(f'Total: ${per_day["day_pnl_flat"].sum():+,.0f}')
    print(f'Mean / day: ${per_day["day_pnl_flat"].mean():+.0f}')
    print(f'Median / day: ${per_day["day_pnl_flat"].median():+.0f}')
    print(f'Positive days: {(per_day["day_pnl_flat"] > 0).sum()} / {len(per_day)}')
    print(f'Min day: ${per_day["day_pnl_flat"].min():+.0f}')
    print(f'Max day: ${per_day["day_pnl_flat"].max():+.0f}')
    print()
    print(per_day.to_string(index=False))


if __name__ == '__main__':
    main()
