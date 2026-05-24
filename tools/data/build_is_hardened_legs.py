"""IS hardened forward pass — produces IS leg list with realistic R-trigger
entries/exits and flat sizing. No B6/B7 dependencies (no peeking risk for
classifiers later trained on IS).

For the bad-trade-classifier project (2026-05-17): we need an IS leg list
with honest realized P&L before we can train a model to predict bad trades.
The existing `composite_forward_pass_hardened.csv` is OOS only — this tool
fills the IS gap.

Architecture (mirrors composite_forward_pass_hardened.py minus the model
dependencies):
  - Each day: compute median ATR(14) on 1m bars
  - r_price = max(4 ticks, round(atr_pts / tick * 4)) * tick   (= Python pipeline)
  - Derive pivot events from IS pivot dataset (truth dataset)
  - For each pivot, detect R-trigger fire on 5s closes (forward walk, max
    120 min lookahead)
  - Build leg pairs: leg k = from rtrig[k].fire to rtrig[k+1].fire
  - Leg direction = direction of NEW leg after pivot k confirmed
  - P&L = signed price delta * $2/point - $6 friction (= $4 commission + $2 slip)

Output: reports/findings/regret_oracle/is_hardened_legs.csv
Schema: day, entry_ts, leg_dir, entry_price, exit_ts, exit_price,
        pnl_pts, pnl_usd, r_price, atr_pts
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


TICK_SIZE = 0.25
DOLLAR_PER_POINT = 2.0
COMMISSION_PER_LEG = 4.00
SLIPPAGE_PER_LEG   = 2.00
FRICTION_PER_LEG   = COMMISSION_PER_LEG + SLIPPAGE_PER_LEG
ATR_MULT = 4.0
ATR_PERIOD = 14
MAX_LOOKAHEAD_MIN = 120
PIVOT_DEDUP_SECONDS = 90


def compute_atr(bars1m: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Median ATR over last (period x 3) true ranges. Matches Python pipeline."""
    h = bars1m['high'].values
    l = bars1m['low'].values
    c = bars1m['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())


def derive_pivot_events(truth_day: pd.DataFrame) -> list:
    """Collapse contiguous is_pivot=1 bars within PIVOT_DEDUP_SECONDS into
    one event. Returns [(ts, direction_str, price), ...].
    """
    piv = truth_day[truth_day['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return []
    ts = piv['timestamp'].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    pp_ = piv['pivot_price'].values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > PIVOT_DEDUP_SECONDS:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for grp in groups:
        ts_c = int(np.median(ts[grp]))
        vals, counts = np.unique(pd_[grp], return_counts=True)
        d = str(vals[np.argmax(counts)])
        p = float(np.mean(pp_[grp]))
        out.append((ts_c, d, p))
    return out


def detect_r_trigger_fire(closes5s, ts5s, pivot_ts, pivot_price, leg_dir,
                            r_price, max_lookahead_min=MAX_LOOKAHEAD_MIN):
    """Walk forward from pivot to first 5s close that crosses R-trigger.
    LONG leg: close >= pivot + r_price
    SHORT leg: close <= pivot - r_price
    Returns (fire_ts, fire_price) or (None, None) if not found.
    """
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


def process_day(day: str, truth: pd.DataFrame, bars1m_dir: Path,
                bars5s_dir: Path) -> list:
    bars1m_path = bars1m_dir / f'{day}.parquet'
    bars5s_path = bars5s_dir / f'{day}.parquet'
    if not bars1m_path.exists() or not bars5s_path.exists():
        return []

    bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
    bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
    atr_pts = compute_atr(bars1m, ATR_PERIOD)
    min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * ATR_MULT)))
    r_price = min_rev_ticks * TICK_SIZE

    truth_day = truth[truth['day'] == day].sort_values('timestamp').reset_index(drop=True)
    events = derive_pivot_events(truth_day)
    if len(events) < 2:
        return []

    closes5s = bars5s['close'].values.astype(np.float64)
    ts5s = bars5s['timestamp'].values.astype(np.int64)

    rtrig_fires = []
    for (piv_ts, piv_dir, piv_price) in events:
        fire_ts, fire_price = detect_r_trigger_fire(
            closes5s, ts5s, piv_ts, piv_price, piv_dir, r_price)
        if fire_ts is None:
            continue
        rtrig_fires.append({
            'pivot_ts': piv_ts, 'pivot_dir': piv_dir, 'pivot_price': piv_price,
            'fire_ts': fire_ts, 'fire_price': fire_price,
        })

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
            'day': day,
            'entry_ts': entry_ts,
            'leg_dir': leg_dir,
            'entry_price': entry_price,
            'exit_ts': exit_ts,
            'exit_price': exit_price,
            'pnl_pts': pnl_pts,
            'pnl_usd': pnl_usd,
            'r_price': r_price,
            'atr_pts': atr_pts,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet',
                    help='IS pivot dataset (V2 features + is_pivot flag per 1m bar)')
    ap.add_argument('--bars1m-dir', default='DATA/ATLAS/1m',
                    help='Directory of 1m bar parquets')
    ap.add_argument('--bars5s-dir', default='DATA/ATLAS/5s',
                    help='Directory of 5s bar parquets')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/is_hardened_legs.csv',
                    help='Output leg list CSV')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/is_hardened_legs.txt',
                    help='Output summary report')
    args = ap.parse_args()

    print(f'Loading truth: {args.truth}')
    truth = pd.read_parquet(args.truth)
    print(f'  rows {len(truth):,}   days {truth["day"].nunique()}')

    bars1m_dir = Path(args.bars1m_dir)
    bars5s_dir = Path(args.bars5s_dir)

    days = sorted(truth['day'].unique())
    all_rows = []
    skipped_days = 0
    for day in tqdm(days, desc='days'):
        rows = process_day(day, truth, bars1m_dir, bars5s_dir)
        if not rows:
            skipped_days += 1
            continue
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)

    n_legs = len(df)
    n_days = df['day'].nunique() if n_legs > 0 else 0
    pnl = df['pnl_usd'].values if n_legs > 0 else np.array([])

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('IS HARDENED FORWARD PASS  (flat sizing, no ML deps)')
    out('=' * 78)
    out(f'Source: {args.truth}')
    out(f'Days available: {len(days)}   processed {n_days}   skipped {skipped_days}')
    out(f'Day range: {df["day"].min()} -> {df["day"].max()}' if n_legs > 0 else '(no legs)')
    out(f'Total legs: {n_legs:,}')
    out(f'Friction per leg: ${FRICTION_PER_LEG:.2f}')
    out('')
    if n_legs > 0:
        out(f'  Mean per-leg P&L:       ${pnl.mean():+.2f}')
        out(f'  Median per-leg P&L:     ${np.median(pnl):+.2f}')
        out(f'  Positive legs:          {int((pnl > 0).sum())}/{n_legs} ({(pnl > 0).mean()*100:.1f}%)')
        out(f'  Total $:                ${pnl.sum():+,.2f}')
        out(f'  Mean $/day:             ${pnl.sum() / n_days:+.2f}')
        out('')

        per_day = df.groupby('day')['pnl_usd'].sum().values
        n_win = int((per_day > 0).sum())
        out(f'  Day WR:                 {n_win}/{n_days} ({n_win/n_days*100:.1f}%)')
        out(f'  Median day:             ${np.median(per_day):+.2f}')
        out(f'  Worst day:              ${per_day.min():+.2f}')
        out(f'  Best day:               ${per_day.max():+.2f}')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
