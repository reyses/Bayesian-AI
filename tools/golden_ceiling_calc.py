"""Theoretical ceiling: if we PERFECTLY identified golden moments and traded
in oracle's predicted direction with realistic exits, what's the maximum $/day?

Two ceilings:
  A. ORACLE direction (100% accuracy hypothesis) + tick-exact TP/SL
  B. DIRECTION CLASSIFIER accuracy (87% on oracle bars) + tick-exact TP/SL

For each oracle bar with mfe_dollars > MFE_THR:
  - Enter at oracle's bar in oracle's predicted direction
  - Walk forward bars; whichever of {TP, SL, TimeStop, EOD, oracle exit_ts}
    triggers first wins
  - Record outcome in $

Aggregates per-day with bootstrap CI.

Output: ceiling table for multiple (MFE_THR, TP, SL) combos.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW_5S_DIR = Path('DATA/ATLAS/5s')
TICK = 0.25
TICK_VALUE = 0.50
TF_S = 5


def load_bars_for_day(day_key: str, cache: dict) -> pd.DataFrame:
    if day_key not in cache:
        p = RAW_5S_DIR / f'{day_key}.parquet'
        if p.exists():
            df = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
            cache[day_key] = df
        else:
            cache[day_key] = None
    return cache[day_key]


def simulate_trade(direction: str, entry_price: float, entry_ts: int,
                   exit_ts_cap: int, tp_usd: float, sl_usd: float,
                   bars: pd.DataFrame) -> tuple:
    """Walk bars from entry_ts forward; return (pnl_dollars, exit_reason, exit_ts).
    First hit wins: TP intrabar high (LONG) / low (SHORT), SL intrabar low (LONG) / high (SHORT),
    exit_ts_cap reached, or end of bars.
    """
    tp_ticks = tp_usd / TICK_VALUE
    sl_ticks = abs(sl_usd) / TICK_VALUE
    if direction == 'LONG':
        tp_price = entry_price + tp_ticks * TICK
        sl_price = entry_price - sl_ticks * TICK
    else:
        tp_price = entry_price - tp_ticks * TICK
        sl_price = entry_price + sl_ticks * TICK

    win = bars[(bars['timestamp'] >= entry_ts) & (bars['timestamp'] <= exit_ts_cap)]
    if len(win) == 0:
        return 0.0, 'no_bars', entry_ts

    for _, row in win.iterrows():
        hi = float(row['high']); lo = float(row['low'])
        ts = int(row['timestamp'])
        if direction == 'LONG':
            sl_hit = lo <= sl_price
            tp_hit = hi >= tp_price
            if sl_hit and tp_hit:
                # Ambiguous intrabar — assume SL hits first (conservative)
                return -abs(sl_usd) - TICK_VALUE * 1.0, 'sl', ts   # 1-tick slippage
            if sl_hit:
                return -abs(sl_usd) - TICK_VALUE * 1.0, 'sl', ts
            if tp_hit:
                return abs(tp_usd), 'tp', ts
        else:
            sl_hit = hi >= sl_price
            tp_hit = lo <= tp_price
            if sl_hit and tp_hit:
                return -abs(sl_usd) - TICK_VALUE * 1.0, 'sl', ts
            if sl_hit:
                return -abs(sl_usd) - TICK_VALUE * 1.0, 'sl', ts
            if tp_hit:
                return abs(tp_usd), 'tp', ts

    # Time-out → close at last bar's close (best-effort estimate)
    last_close = float(win.iloc[-1]['close'])
    if direction == 'LONG':
        pnl = (last_close - entry_price) / TICK * TICK_VALUE
    else:
        pnl = (entry_price - last_close) / TICK * TICK_VALUE
    return pnl, 'timeout', int(win.iloc[-1]['timestamp'])


def bootstrap_ci_mean(arr, n_boot=4000, seed=42):
    arr = np.asarray(arr)
    if len(arr) < 2:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        boots[i] = arr[rng.integers(0, n, n)].mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle-csv', required=True)
    ap.add_argument('--cost', type=float, default=4.0)
    ap.add_argument('--filter-mode', choices=['mfe', 'velocity'], default='mfe')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    oracle = pd.read_csv(args.oracle_csv)
    oracle['session_date_key'] = pd.to_datetime(oracle['session_date']).dt.strftime('%Y_%m_%d')
    # Trade duration cap = 30 min algorithmic; mfe_velocity = $/min
    oracle['win_end_ts'] = (oracle['oracle_ts'] +
                            oracle['time_to_mfe_min'].clip(upper=30) * 60).astype(int)
    # Velocity distribution check
    print(f'mfe_velocity stats: median={oracle["mfe_velocity"].median():.2f}/min  '
          f'p75={oracle["mfe_velocity"].quantile(0.75):.2f}  '
          f'p90={oracle["mfe_velocity"].quantile(0.90):.2f}  '
          f'p95={oracle["mfe_velocity"].quantile(0.95):.2f}')

    bar_cache = {}
    n_days = oracle['session_date_key'].nunique()
    print(f'Oracle bars: {len(oracle)}, days: {n_days}, filter_mode={args.filter_mode}')

    if args.filter_mode == 'mfe':
        thresholds = [50, 100, 150, 200, 300]
        thr_col = 'mfe_dollars'
        thr_label = '$mfe'
    else:
        # mfe_velocity is $/min
        thresholds = [3, 5, 8, 12, 20, 30]
        thr_col = 'mfe_velocity'
        thr_label = '$/min'
    tp_sl_pairs = [(20, 20), (30, 20), (40, 20), (60, 30), (100, 30)]

    print(f'\n{"MFE_thr":<8}{"TP":<6}{"SL":<6}{"n":<6}{"/day":<8}{"$/t mean":<10}'
          f'{"TP%":<6}{"SL%":<6}{"$/day":<10}{"CI lo":<8}{"CI hi":<8}{"DayWR":<6}')

    rows = []
    for mfe_thr in thresholds:
        # Filter to golden bars by either mfe$ or velocity ($/min)
        gold = oracle[oracle[thr_col] > mfe_thr].copy()
        if len(gold) == 0:
            continue

        for (tp, sl) in tp_sl_pairs:
            day_pnl = defaultdict(float)
            day_pnl_classifier87 = defaultdict(float)  # apply 87% direction acc
            trade_pnls = []
            tp_count = 0
            sl_count = 0
            timeout_count = 0
            rng = np.random.default_rng(0)
            for _, row in gold.iterrows():
                day = row['session_date_key']
                bars = load_bars_for_day(day, bar_cache)
                if bars is None:
                    continue
                pnl, reason, _ = simulate_trade(
                    row['direction'], float(row['entry_price']),
                    int(row['oracle_ts']), int(row['win_end_ts']),
                    tp, sl, bars
                )
                pnl_net = pnl - args.cost
                trade_pnls.append(pnl_net)
                day_pnl[day] += pnl_net
                # 87% direction acc: flip pnl 13% of the time (random)
                flip = rng.random() > 0.87
                if flip:
                    # Wrong direction: outcome is approximately MIRROR
                    # Approximation: if oracle SL hit, classifier-wrong TP hit (and vice versa)
                    if reason == 'tp':
                        pnl_clf = -abs(sl) - TICK_VALUE   # 1 tick slip
                    elif reason == 'sl':
                        pnl_clf = abs(tp)
                    else:
                        pnl_clf = -pnl   # timeout: mirror
                else:
                    pnl_clf = pnl
                day_pnl_classifier87[day] += (pnl_clf - args.cost)

                if reason == 'tp': tp_count += 1
                elif reason == 'sl': sl_count += 1
                else: timeout_count += 1

            if not trade_pnls:
                continue
            arr = np.array(trade_pnls)
            day_arr_oracle = np.array(list(day_pnl.values()))
            day_arr_clf = np.array(list(day_pnl_classifier87.values()))

            mean_d_o, lo_o, hi_o = bootstrap_ci_mean(day_arr_oracle)
            mean_d_c, lo_c, hi_c = bootstrap_ci_mean(day_arr_clf)
            win_days_o = int((day_arr_oracle > 0).sum())
            win_days_c = int((day_arr_clf > 0).sum())
            row_out = {
                'filter_mode': args.filter_mode,
                'filter_thr': mfe_thr,
                'mfe_thr': mfe_thr, 'tp_usd': tp, 'sl_usd': sl,
                'n_trades': int(len(arr)),
                'trades_per_day': float(len(arr) / n_days),
                'mean_per_trade_NET': float(arr.mean()),
                'tp_pct': tp_count / len(arr) * 100,
                'sl_pct': sl_count / len(arr) * 100,
                'timeout_pct': timeout_count / len(arr) * 100,
                'oracle_dir_mean_per_day': mean_d_o,
                'oracle_dir_ci_lo': lo_o, 'oracle_dir_ci_hi': hi_o,
                'oracle_dir_day_wr': win_days_o / len(day_arr_oracle),
                'clf87_dir_mean_per_day': mean_d_c,
                'clf87_dir_ci_lo': lo_c, 'clf87_dir_ci_hi': hi_c,
                'clf87_dir_day_wr': win_days_c / len(day_arr_clf),
            }
            rows.append(row_out)
            print(f'{mfe_thr:<8}{tp:<6}{sl:<6}{len(arr):<6}'
                  f'{row_out["trades_per_day"]:<8.2f}'
                  f'${arr.mean():<+8.2f} {tp_count/len(arr)*100:<6.0f}{sl_count/len(arr)*100:<6.0f}'
                  f'${mean_d_o:<+8.1f} '
                  f'${lo_o:<+6.0f} ${hi_o:<+6.0f} {win_days_o/len(day_arr_oracle):<6.2f}')

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}')

    # Best ceilings
    print(f'\n=== HEADLINE CEILINGS ===')
    print(f'\nORACLE direction (perfect 100% dir):')
    top_o = df.sort_values('oracle_dir_mean_per_day', ascending=False).head(8)
    print(top_o[['mfe_thr','tp_usd','sl_usd','n_trades','trades_per_day',
                 'oracle_dir_mean_per_day','oracle_dir_ci_lo','oracle_dir_ci_hi',
                 'oracle_dir_day_wr','tp_pct','sl_pct']].to_string(index=False))

    print(f'\nDIRECTION CLASSIFIER 87% accuracy:')
    top_c = df.sort_values('clf87_dir_mean_per_day', ascending=False).head(8)
    print(top_c[['mfe_thr','tp_usd','sl_usd','n_trades','trades_per_day',
                 'clf87_dir_mean_per_day','clf87_dir_ci_lo','clf87_dir_ci_hi',
                 'clf87_dir_day_wr','tp_pct','sl_pct']].to_string(index=False))


if __name__ == '__main__':
    main()
