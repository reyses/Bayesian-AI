"""
zigzag_v12rc_on_nt8_csv.py -- run v1.2-RC simulator on NT8-exported 1s CSVs.

Uses NT8-source data instead of ATLAS (Databento). Most-recent regime.
Includes 2026-04-24 (Day 1 of live v1.0 deployment).

Reads `DATA/ATLAS_NT8/1s/MNQ_06-26/<YYYY_MM_DD>.csv`. Aggregates 1s -> 1m
in-memory, runs zigzag at R=30, applies v1.2-RC (trail + SL=10pt) on the
1s tick data exactly as `zigzag_trail_ticker.py` does.

Output: per-trade ledger + per-day summary, ready to compare against the
NT8 Playback / live trade log.

Usage:
    python tools/zigzag_v12rc_on_nt8_csv.py
    python tools/zigzag_v12rc_on_nt8_csv.py --day 2026-04-24
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.zigzag_backtest import zigzag_pivots_with_confirmation


NT8_CSV_DIR = 'DATA/ATLAS_NT8/1s/MNQ_06-26'
DOLLAR_PER_POINT = 2.0
COMMISSION_RT = 2.0


def load_nt8_csv(path: str) -> pd.DataFrame:
    """Read NT8-export 1s CSV. Handles UTF-8 BOM and drops NaN OHLC rows."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def aggregate_1m_from_1s(df_1s: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1s OHLCV to 1m bars on minute boundary."""
    df_1s = df_1s.copy()
    df_1s['minute'] = (df_1s['timestamp'] // 60) * 60
    g = df_1s.groupby('minute', sort=True)
    df_1m = pd.DataFrame({
        'timestamp': g.size().index.values,
        'open':  g['open'].first().values,
        'high':  g['high'].max().values,
        'low':   g['low'].min().values,
        'close': g['close'].last().values,
        'volume': g['volume'].sum().values,
    })
    return df_1m


def simulate_day(df_1s: pd.DataFrame, df_1m: pd.DataFrame, day_label: str,
                 r: float = 30.0,
                 trail_activate: float = 10.0, trail_dist: float = 5.0,
                 trail_pct: float = 0.10, sl_pts: float = 10.0,
                 commission_rt: float = COMMISSION_RT) -> list[dict]:
    """v1.2-RC simulation on per-day 1s + 1m data, with intra-bar SL on 1s ticks."""
    closes_1m = df_1m['close'].values.astype(np.float64)
    ts_1m     = df_1m['timestamp'].values.astype(np.int64)
    pivots = zigzag_pivots_with_confirmation(closes_1m, r)
    if len(pivots) < 2:
        return []

    pivot_by_confirm = {int(c): (k, float(p)) for _, p, k, c in pivots}

    _ts_1s = df_1s['timestamp'].values.astype(np.int64)
    _o_1s  = df_1s['open'].values.astype(np.float64)
    _h_1s  = df_1s['high'].values.astype(np.float64)
    _l_1s  = df_1s['low'].values.astype(np.float64)
    _c_1s  = df_1s['close'].values.astype(np.float64)
    n_ticks = len(_ts_1s)

    trades: list[dict] = []
    in_pos = False
    direction = 0
    entry_price = 0.0
    entry_ts = 0
    peak_close_pts = 0.0
    peak_close_price = 0.0
    peak_pnl_pts = 0.0
    worst_pnl_pts = 0.0
    trail_armed = False
    pending_entry = None

    cur_minute_idx = -1
    agg_close = 0.0

    def _confirm_lookup(close_ts):
        bar_start_ts = close_ts - 60
        idx = np.searchsorted(ts_1m, bar_start_ts, side='left')
        if idx < len(ts_1m) and int(ts_1m[idx]) == int(bar_start_ts):
            return pivot_by_confirm.get(int(idx))
        return None

    for i in range(n_ticks):
        tick_ts = int(_ts_1s[i])
        tick_o  = _o_1s[i]
        tick_h  = _h_1s[i]
        tick_l  = _l_1s[i]
        tick_c  = _c_1s[i]
        tick_minute_idx = tick_ts // 60

        if cur_minute_idx == -1:
            cur_minute_idx = tick_minute_idx
            agg_close = tick_c
        else:
            agg_close = tick_c

        if pending_entry is not None and not in_pos:
            kind, _ = pending_entry
            direction = +1 if kind == 'low' else -1
            entry_price = tick_o
            entry_ts = tick_ts
            peak_close_pts = 0.0
            peak_close_price = entry_price
            peak_pnl_pts = 0.0
            worst_pnl_pts = 0.0
            trail_armed = False
            in_pos = True
            pending_entry = None

        if in_pos:
            if direction > 0:
                tick_pnl = tick_c - entry_price
                tick_worst = tick_l - entry_price
            else:
                tick_pnl = entry_price - tick_c
                tick_worst = entry_price - tick_h
            if tick_pnl > peak_pnl_pts: peak_pnl_pts = tick_pnl
            if tick_worst < worst_pnl_pts: worst_pnl_pts = tick_worst

            # SL intra-bar
            if sl_pts > 0:
                if direction > 0:
                    sl_price = entry_price - sl_pts
                    if tick_l <= sl_price:
                        pnl_pts = direction * (sl_price - entry_price)
                        trades.append({
                            'day': day_label, 'entry_ts': entry_ts,
                            'entry_price': entry_price, 'exit_ts': tick_ts,
                            'exit_price': float(sl_price),
                            'direction': direction,
                            'pnl_pts': float(pnl_pts),
                            'pnl_usd': float(pnl_pts * DOLLAR_PER_POINT - commission_rt),
                            'mfe_pts': float(peak_pnl_pts),
                            'mae_pts': float(-worst_pnl_pts),
                            'leg_min': (tick_ts - entry_ts) / 60.0,
                            'exit_reason': 'sl',
                        })
                        in_pos = False; trail_armed = False
                        continue
                else:
                    sl_price = entry_price + sl_pts
                    if tick_h >= sl_price:
                        pnl_pts = direction * (sl_price - entry_price)
                        trades.append({
                            'day': day_label, 'entry_ts': entry_ts,
                            'entry_price': entry_price, 'exit_ts': tick_ts,
                            'exit_price': float(sl_price),
                            'direction': direction,
                            'pnl_pts': float(pnl_pts),
                            'pnl_usd': float(pnl_pts * DOLLAR_PER_POINT - commission_rt),
                            'mfe_pts': float(peak_pnl_pts),
                            'mae_pts': float(-worst_pnl_pts),
                            'leg_min': (tick_ts - entry_ts) / 60.0,
                            'exit_reason': 'sl',
                        })
                        in_pos = False; trail_armed = False
                        continue

        if tick_minute_idx == cur_minute_idx:
            continue

        completed_close_ts = (cur_minute_idx + 1) * 60
        completed_close_price = agg_close
        cur_minute_idx = tick_minute_idx

        if in_pos:
            if direction > 0:
                close_pnl = completed_close_price - entry_price
            else:
                close_pnl = entry_price - completed_close_price
            if close_pnl > peak_close_pts:
                peak_close_pts = close_pnl
                peak_close_price = completed_close_price

            if trail_activate > 0 and not trail_armed and peak_close_pts >= trail_activate:
                trail_armed = True

            if trail_armed:
                eff_dist = max(trail_dist, trail_pct * peak_close_pts)
                if direction > 0:
                    stop_px = peak_close_price - eff_dist
                    breached = completed_close_price <= stop_px
                else:
                    stop_px = peak_close_price + eff_dist
                    breached = completed_close_price >= stop_px
                if breached:
                    pnl_pts = direction * (tick_o - entry_price)
                    trades.append({
                        'day': day_label, 'entry_ts': entry_ts,
                        'entry_price': entry_price, 'exit_ts': tick_ts,
                        'exit_price': float(tick_o),
                        'direction': direction,
                        'pnl_pts': float(pnl_pts),
                        'pnl_usd': float(pnl_pts * DOLLAR_PER_POINT - commission_rt),
                        'mfe_pts': float(peak_pnl_pts),
                        'mae_pts': float(-worst_pnl_pts),
                        'leg_min': (tick_ts - entry_ts) / 60.0,
                        'exit_reason': 'trail',
                    })
                    in_pos = False; trail_armed = False

        pivot_info = _confirm_lookup(completed_close_ts)
        if pivot_info is not None:
            new_kind, _ = pivot_info
            if in_pos:
                opposite = ((direction > 0 and new_kind == 'high') or
                            (direction < 0 and new_kind == 'low'))
                if opposite:
                    pnl_pts = direction * (tick_o - entry_price)
                    trades.append({
                        'day': day_label, 'entry_ts': entry_ts,
                        'entry_price': entry_price, 'exit_ts': tick_ts,
                        'exit_price': float(tick_o),
                        'direction': direction,
                        'pnl_pts': float(pnl_pts),
                        'pnl_usd': float(pnl_pts * DOLLAR_PER_POINT - commission_rt),
                        'mfe_pts': float(peak_pnl_pts),
                        'mae_pts': float(-worst_pnl_pts),
                        'leg_min': (tick_ts - entry_ts) / 60.0,
                        'exit_reason': 'pivot',
                    })
                    in_pos = False; trail_armed = False

            if not in_pos:
                pending_entry = pivot_info

    # EOD close any open position
    if in_pos:
        last_ts = int(_ts_1s[-1]); last_price = float(_c_1s[-1])
        pnl_pts = direction * (last_price - entry_price)
        trades.append({
            'day': day_label, 'entry_ts': entry_ts,
            'entry_price': entry_price, 'exit_ts': last_ts,
            'exit_price': float(last_price),
            'direction': direction,
            'pnl_pts': float(pnl_pts),
            'pnl_usd': float(pnl_pts * DOLLAR_PER_POINT - commission_rt),
            'mfe_pts': float(peak_pnl_pts),
            'mae_pts': float(-worst_pnl_pts),
            'leg_min': (last_ts - entry_ts) / 60.0,
            'exit_reason': 'eod_final',
        })
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', default=NT8_CSV_DIR)
    ap.add_argument('--day', default=None)
    ap.add_argument('--out', default='reports/findings/zigzag_v12rc_on_nt8_data.csv')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.csv_dir, '*.csv')))
    if args.day:
        target = args.day.replace('-', '_')
        files = [f for f in files if target in os.path.basename(f)]

    if not files:
        print(f'No CSVs in {args.csv_dir}'); return

    print('=' * 96)
    print(f'V1.2-RC SIM ON NT8-EXPORTED 1s CSV  ({len(files)} files)')
    print('=' * 96)

    all_trades = []
    daily_pnl = []
    for fpath in tqdm(files, desc='days') if len(files) > 1 else files:
        if isinstance(fpath, str): pass
        else: fpath = fpath
        day = os.path.splitext(os.path.basename(fpath))[0]
        df_1s = load_nt8_csv(fpath)
        if len(df_1s) < 60: continue
        df_1m = aggregate_1m_from_1s(df_1s)
        trades = simulate_day(df_1s, df_1m, day)
        if not trades: continue
        day_pnl = sum(t['pnl_usd'] for t in trades)
        daily_pnl.append({'day': day, 'pnl': day_pnl, 'n_trades': len(trades),
                          'n_trail': sum(1 for t in trades if t['exit_reason']=='trail'),
                          'n_sl':    sum(1 for t in trades if t['exit_reason']=='sl'),
                          'n_pivot': sum(1 for t in trades if t['exit_reason']=='pivot'),
                          'n_eod':   sum(1 for t in trades if t['exit_reason'] in ('eod','eod_final'))})
        all_trades.extend(trades)

    if not all_trades:
        print('No trades.'); return

    df = pd.DataFrame(all_trades)
    daily = pd.DataFrame(daily_pnl)
    print(f'\nTotal trades: {len(df):,}, days: {len(daily)}')
    print(f'\nPer-day P&L (sorted by date):')
    daily_sorted = daily.sort_values('day')
    for _, r in daily_sorted.iterrows():
        print(f'  {r["day"]}: ${r["pnl"]:>+8.2f}  ({r["n_trades"]:>3} trades: '
              f'{r["n_trail"]} trail / {r["n_sl"]} sl / {r["n_pivot"]} pivot / {r["n_eod"]} eod)')

    print(f'\nSummary')
    print(f'  Mean $/day        : ${daily["pnl"].mean():+.2f}')
    print(f'  Median $/day      : ${daily["pnl"].median():+.2f}')
    print(f'  Day WR            : {(daily["pnl"]>0).mean()*100:.1f}%')
    print(f'  Best/Worst        : ${daily["pnl"].max():+.0f} / ${daily["pnl"].min():+.0f}')
    print(f'  Total $           : ${daily["pnl"].sum():+.0f}')
    print(f'  Mean $/trade      : ${df["pnl_usd"].mean():+.2f}')

    p = df.loc[df["pnl_usd"]>0, "pnl_usd"].sum()
    L = abs(df.loc[df["pnl_usd"]<0, "pnl_usd"].sum())
    pf = p / max(L, 1)
    print(f'  Profit factor     : {pf:.4f}  (PF Trade WR = {pf-1:+.4f})')
    print(f'  Dollar-share WR   : {pf/(pf+1)*100:.2f}%')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    daily.to_csv(args.out.replace('.csv','_daily.csv'), index=False)
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
