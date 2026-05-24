"""Daisy-chain regret oracle — sequential best-extreme-in-window trades.

Per user 2026-05-14:
    Start at bar 1 of the IS. Look forward W minutes (configurable, default 60).
    Find the bar with the largest favorable excursion from the entry price.
    Direction is determined retroactively:
        if (high - entry)  >  (entry - low):  LONG,  exit at the high
        else:                                 SHORT, exit at the low
    That exit ends the trade and starts the next. Repeat through the IS.
    Chain resets at every Globex session boundary (can't hold across the halt).

Maps directly onto the 1-hour-lapse premise: in each W-minute budget take
the single best trade. Result is a sequential, non-overlapping chain of
trades (one per ≤W-minute leg, since trades end early at their extreme).

Distinct from the centered-window local-extrema oracle in
`regret_1m_oracle.py`. Both kept; both useful:
    local-extrema  → "all opportunities" label universe (with overlap)
    daisy-chain    → "best sequential edge under a W-minute budget"

Output: one CSV per run with one row per chained trade. Schema is compatible
with the existing per-cell EDA tool (regret_distribution_eda.py).
"""
from __future__ import annotations
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.regret_1m_oracle import (
    load_tf_bars, build_session_info, extract_state_vector, discretize_state,
    TF_SECONDS, TICK, TICK_DOLLAR, SESSION_GAP_S, OUT_DIR,
)
from tools.cusp_marker import compute_anchor


def daisy_chain(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    ts: np.ndarray, session_id: np.ndarray, session_end_idx: np.ndarray,
    window_bars: int,
) -> list:
    """Sequentially chain best-extreme-in-window trades, session by session.
    Each trade's exit is the next trade's entry."""
    trades = []
    # Per-session start indices
    _, session_starts = np.unique(session_id, return_index=True)
    session_starts = sorted(int(s) for s in session_starts)

    for s_start in session_starts:
        s_end = int(session_end_idx[s_start])
        if s_end - s_start < 2:
            continue

        current_idx = s_start
        current_price = float(close[s_start])

        while current_idx < s_end:
            window_end = min(current_idx + window_bars, s_end)
            if window_end <= current_idx:
                break

            # Look at bars (current_idx, window_end] for the best excursion
            win_high = high[current_idx + 1 : window_end + 1]
            win_low  = low [current_idx + 1 : window_end + 1]
            if len(win_high) == 0:
                break

            up_local = int(np.argmax(win_high))
            dn_local = int(np.argmin(win_low))
            up_excursion = float(win_high[up_local]) - current_price
            dn_excursion = current_price - float(win_low[dn_local])

            # Pick the larger excursion; tie defaults to LONG
            if up_excursion >= dn_excursion:
                direction = 'LONG'
                exit_offset = up_local + 1
                exit_price = float(win_high[up_local])
                mfe_ticks = up_excursion / TICK
            else:
                direction = 'SHORT'
                exit_offset = dn_local + 1
                exit_price = float(win_low[dn_local])
                mfe_ticks = dn_excursion / TICK

            exit_idx = current_idx + exit_offset
            trades.append({
                'session_id':   int(session_id[current_idx]),
                'entry_idx':    int(current_idx),
                'entry_ts':     int(ts[current_idx]),
                'exit_idx':     int(exit_idx),
                'exit_ts':      int(ts[exit_idx]),
                'direction':    direction,
                'entry_price':  round(current_price, 2),
                'exit_price':   round(exit_price, 2),
                'mfe_ticks':    round(mfe_ticks, 1),
            })

            # CHAIN: next trade starts at this exit
            current_idx = exit_idx
            current_price = exit_price

    return trades


def _ts_to_utc(ts_int):
    return datetime.fromtimestamp(int(ts_int), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def run(t_start: float, t_end: float, run_name: str,
        tf: str = '5s', window_min: float = 60.0,
        include_state: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tf_seconds = TF_SECONDS.get(tf)
    if tf_seconds is None:
        raise ValueError(f'Unsupported --tf {tf!r}')
    window_bars = max(1, int(window_min * 60 / tf_seconds))

    print(f'\n=== Daisy-chain oracle — {run_name} (tf={tf}, window={window_min}min) ===')
    print(f'Range: {_ts_to_utc(t_start)} -> {_ts_to_utc(t_end)}')
    print(f'  tf={tf} ({tf_seconds}s/bar)  window={window_min}min={window_bars}bars')

    df = load_tf_bars(t_start, t_end, tf)
    if df.empty:
        print('No data'); return
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    print(f'Loaded {len(df)} {tf} bars')

    session_id, session_end_idx, session_first_ts, session_date, n_sessions = \
        build_session_info(ts)
    print(f'  Segmented into {n_sessions} sessions (halt-to-halt)')
    bar_range = high - low
    volume = (df['volume'].values.astype(float) if 'volume' in df.columns
              else np.full(len(df), np.nan))

    # Anchors (only if --state)
    if include_state:
        print('Computing CRMs + anchors (for state vector at each entry)...')
        M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
        M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
        M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
        Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
        Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
        Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    print(f'\nWalking sequentially, picking best extreme per {window_min}min window...')
    trades = daisy_chain(close, high, low, ts, session_id, session_end_idx, window_bars)
    print(f'  Total chained trades: {len(trades)}')
    if not trades:
        return

    # Enrich each trade with derived columns + (optional) state vector at entry
    print('Enriching trades with derived columns...')
    rows = []
    for t in trades:
        eidx = t['entry_idx']
        xidx = t['exit_idx']
        duration_bars = xidx - eidx
        duration_min = duration_bars * tf_seconds / 60.0
        avail_bars = int(session_end_idx[eidx]) - eidx
        avail_min = avail_bars * tf_seconds / 60.0
        # full_window = whether the full W-min budget was AVAILABLE at entry
        # (not truncated by session end). Daisy-chain trades end at the best
        # extreme — usually well before the W-min cap — so we should NOT
        # equate "full_window" with "used the full window."
        full_window = int(avail_bars >= window_bars)
        mfe_dollars = round(t['mfe_ticks'] * TICK_DOLLAR, 2)
        velocity = mfe_dollars / max(duration_min, tf_seconds / 60.0)
        vol = volume[eidx]
        row = {
            'oracle_idx':         eidx,             # same name as local-extrema oracle for downstream tools
            'oracle_ts':          t['entry_ts'],
            'oracle_utc':         _ts_to_utc(t['entry_ts']),
            'session_id':         t['session_id'],
            'session_date':       session_date[eidx],
            'tod_minutes':        int((ts[eidx] - session_first_ts[eidx]) // 60),
            'direction':          t['direction'],
            'entry_price':        t['entry_price'],
            'mfe_ticks':          t['mfe_ticks'],
            'mfe_dollars':        mfe_dollars,
            'time_to_mfe_min':    round(duration_min, 2),
            'exit_idx':           xidx,
            'exit_ts':            t['exit_ts'],
            'exit_price':         t['exit_price'],
            'mfe_velocity':       round(velocity, 3),
            'full_window':        full_window,
            'available_fwd_min':  round(avail_min, 1),
            'volume':             None if np.isnan(vol) else round(float(vol), 1),
            'bar_range':          round(float(bar_range[eidx]), 2),
        }
        if include_state:
            entry_state = extract_state_vector(eidx, close, M_15s, S_15s, M_1m, S_1m,
                                               M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc,
                                               tf_seconds=tf_seconds)
            exit_state = extract_state_vector(xidx, close, M_15s, S_15s, M_1m, S_1m,
                                              M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc,
                                              tf_seconds=tf_seconds)
            disc_entry = discretize_state(entry_state, t['direction'])
            disc_exit  = discretize_state(exit_state,  t['direction'])
            row.update(entry_state)                                   # entry state, unprefixed (compat)
            row.update({f'd_{k}': v for k, v in disc_entry.items()})  # entry categoricals (d_*)
            row.update({f'exit_{k}': v for k, v in exit_state.items()})    # exit state (exit_*)
            row.update({f'exit_d_{k}': v for k, v in disc_exit.items()})   # exit categoricals (exit_d_*)
        rows.append(row)

    # Save
    out_path = OUT_DIR / f'daisy_chain_{run_name}.csv'
    cols = list(rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote: {out_path}')

    # Summary
    mfes = np.array([r['mfe_dollars'] for r in rows])
    durs = np.array([r['time_to_mfe_min'] for r in rows])
    full = sum(r['full_window'] for r in rows)
    n_long = sum(1 for r in rows if r['direction'] == 'LONG')
    n_short = len(rows) - n_long

    def histmode(vals, bin_w=2.0):
        if len(vals) == 0: return float('nan')
        lo = np.floor(vals.min()/bin_w)*bin_w; hi = np.ceil(vals.max()/bin_w)*bin_w
        if hi <= lo: hi = lo + bin_w
        c, e = np.histogram(vals, bins=np.arange(lo, hi+bin_w, bin_w))
        k = int(np.argmax(c)); return float((e[k]+e[k+1])/2)

    print(f'\n=== Daisy-chain summary ===')
    print(f'  Trades            : {len(rows)}  (LONG {n_long}  /  SHORT {n_short})')
    print(f'  Sessions          : {n_sessions}     trades/session avg: {len(rows)/n_sessions:.1f}')
    print(f'  Full-window trades: {full}  ({100*full/len(rows):.1f}%) '
          f'— trade ran the entire {window_min}min budget')
    print(f'  MFE  $: mode ${histmode(mfes):.0f}  median ${np.median(mfes):.0f}  '
          f'mean ${mfes.mean():.0f}  total ${mfes.sum():,.0f}')
    print(f'  duration min: mode {histmode(durs, 1.0):.1f}  '
          f'median {np.median(durs):.1f}  mean {durs.mean():.1f}')
    print(f'  velocity $/min: mode ${histmode(np.array([r["mfe_velocity"] for r in rows]), 0.5):.2f}  '
          f'median ${np.median([r["mfe_velocity"] for r in rows]):.2f}')

    return rows


def _ts(d):
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', help='YYYY-MM-DD')
    ap.add_argument('--end',   help='YYYY-MM-DD')
    ap.add_argument('--date',  help='Single day YYYY-MM-DD')
    ap.add_argument('--tf', default='5s', choices=list(TF_SECONDS))
    ap.add_argument('--window-min', type=float, default=60.0,
                    help='Search window in MINUTES for the best extreme (default 60)')
    ap.add_argument('--no-state', action='store_true',
                    help='Skip state-vector computation (faster; no anchors)')
    ap.add_argument('--name', help='Run name (auto)')
    args = ap.parse_args()

    if args.date:
        t_start = _ts(args.date); t_end = t_start + 86400
        name = args.name or f'daisy_{args.date}'
    elif args.start and args.end:
        t_start = _ts(args.start); t_end = _ts(args.end) + 86400
        name = args.name or f'daisy_{args.start}_{args.end}'
    else:
        ap.error('Provide --date OR --start+--end')

    run(t_start, t_end, name, tf=args.tf,
        window_min=args.window_min, include_state=not args.no_state)


if __name__ == '__main__':
    main()
