"""v9 — Forward-pass simulator using ORACLE-TRAINED ML SURFACES.

Loads the 4 GBM models trained by logistic_oracle_surfaces.py:
    entry_short / entry_long / exit_short / exit_long

At each bar:
    state = extract_state_vector(...)
    p_entry_short, p_entry_long = entry models
    p_exit_short, p_exit_long   = exit models

Strategy:
    if no open trade:
        if p_entry_short >= P_FIRE: open SHORT (record entry side)
        elif p_entry_long  >= P_FIRE: open LONG
    while in trade:
        if p_exit_(side) >= P_EXIT: close
        OR hard stop, OR time stop
"""
from __future__ import annotations
import argparse
import csv
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars
from tools.regret_1m_oracle import extract_state_vector, TICK_DOLLAR


TICK = 0.25
OUT_DIR = Path('reports/findings/sim_v9')
MODELS_PATH = Path('reports/findings/logistic_oracles/models.pkl')


def load_models() -> dict:
    """Load the 4 oracle surfaces."""
    with open(MODELS_PATH, 'rb') as f:
        models = pickle.load(f)
    by_name = {m['name']: m for m in models}
    return by_name


def predict_proba(model_dict: dict, state: dict, use_gbm: bool = True) -> float:
    """Predict P(positive class) for one state. Uses GBM by default."""
    feature_cols = model_dict['feature_cols']
    X = np.array([[state.get(c, 0.0) if state.get(c) is not None else 0.0
                        for c in feature_cols]])
    if use_gbm:
        return float(model_dict['gbm_model'].predict_proba(X)[0, 1])
    # LR needs scaling
    scaler = model_dict['scaler']
    Xs = scaler.transform(X)
    return float(model_dict['lr_model'].predict_proba(Xs)[0, 1])


def simulate(t_start: float, t_end: float, models: dict,
                p_fire_entry: float = 0.55, p_fire_exit: float = 0.55,
                hard_stop_ticks: float = 30.0,
                max_hold_min: int = 60,
                cooldown_min: int = 5,
                use_gbm: bool = True) -> list:
    """Walk bar-by-bar, score each state, fire/exit per surface predictions."""
    df = load_1m_bars(t_start, t_end)
    if df.empty:
        return []
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)

    print(f'  Loaded {len(df)} 1m bars')
    print('  Computing anchors...')
    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    es = models['entry_short']; el = models['entry_long']
    xs = models['exit_short'];  xl = models['exit_long']

    # Batch-extract feature matrix to vectorize predict (much faster)
    feature_cols = es['feature_cols']
    print(f'  Batch-extracting state vectors for {len(ts)} bars...')
    X = np.zeros((len(ts), len(feature_cols)), dtype=float)
    for i in range(len(ts)):
        if i < 15:
            continue
        state = extract_state_vector(i, close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc)
        for j, fc in enumerate(feature_cols):
            v = state.get(fc, 0.0)
            X[i, j] = 0.0 if v is None else float(v)

    print('  Batch-predicting P_entry / P_exit across all bars...')
    # Vectorized predictions
    if use_gbm:
        p_es = es['gbm_model'].predict_proba(X)[:, 1]
        p_el = el['gbm_model'].predict_proba(X)[:, 1]
        p_xs = xs['gbm_model'].predict_proba(X)[:, 1]
        p_xl = xl['gbm_model'].predict_proba(X)[:, 1]
    else:
        # LR needs scaling per surface
        p_es = es['lr_model'].predict_proba(es['scaler'].transform(X))[:, 1]
        p_el = el['lr_model'].predict_proba(el['scaler'].transform(X))[:, 1]
        p_xs = xs['lr_model'].predict_proba(xs['scaler'].transform(X))[:, 1]
        p_xl = xl['lr_model'].predict_proba(xl['scaler'].transform(X))[:, 1]

    print('  Walking bar-by-bar to simulate trades...')
    trades = []
    open_t = None
    cooldown_until = 0.0

    for i in range(15, len(ts)):
        # ── Manage open trade
        if open_t is not None:
            if open_t['side'] == 'LONG':
                pnl_ticks = (close[i] - open_t['entry_price']) / TICK
            else:
                pnl_ticks = (open_t['entry_price'] - close[i]) / TICK
            open_t['peak'] = max(open_t['peak'], pnl_ticks)
            open_t['worst'] = min(open_t['worst'], pnl_ticks)
            dur = (ts[i] - open_t['entry_ts']) / 60.0
            close_now = False; reason = ''

            # 1) Hard PnL stop
            if pnl_ticks <= -hard_stop_ticks:
                close_now = True; reason = 'hard_stop'
            # 2) Probability exit
            elif open_t['side'] == 'SHORT' and p_xs[i] >= p_fire_exit:
                close_now = True; reason = f'p_exit_short={p_xs[i]:.2f}'
            elif open_t['side'] == 'LONG' and p_xl[i] >= p_fire_exit:
                close_now = True; reason = f'p_exit_long={p_xl[i]:.2f}'
            # 3) Time stop
            elif dur >= max_hold_min:
                close_now = True; reason = 'time_stop'

            if close_now:
                open_t['exit_ts'] = ts[i]
                open_t['exit_price'] = close[i]
                open_t['reason'] = reason
                open_t['dur_min'] = dur
                trades.append(open_t)
                cooldown_until = ts[i] + cooldown_min * 60
                open_t = None
            else:
                continue

        # ── Cooldown
        if ts[i] < cooldown_until:
            continue

        # ── Entry signals
        if p_es[i] >= p_fire_entry and p_es[i] >= p_el[i]:
            open_t = {
                'side': 'SHORT', 'entry_ts': int(ts[i]), 'entry_price': float(close[i]),
                'p_at_entry': float(p_es[i]),
                'peak': 0.0, 'worst': 0.0,
            }
        elif p_el[i] >= p_fire_entry:
            open_t = {
                'side': 'LONG', 'entry_ts': int(ts[i]), 'entry_price': float(close[i]),
                'p_at_entry': float(p_el[i]),
                'peak': 0.0, 'worst': 0.0,
            }

    # Force-close anything still open
    if open_t is not None:
        open_t['exit_ts'] = int(ts[-1])
        open_t['exit_price'] = float(close[-1])
        open_t['reason'] = 'eod'
        open_t['dur_min'] = (ts[-1] - open_t['entry_ts']) / 60.0
        trades.append(open_t)

    return trades


def report(trades: list, run_name: str, p_fire_entry: float, p_fire_exit: float):
    if not trades:
        print(f'\n{run_name}: 0 trades fired'); return
    pnls = np.array([(t['exit_price'] - t['entry_price']) / TICK * TICK_DOLLAR
                          if t['side'] == 'LONG'
                          else (t['entry_price'] - t['exit_price']) / TICK * TICK_DOLLAR
                          for t in trades])
    n = len(trades)
    n_win = (pnls > 0).sum()
    win_sum = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    los_sum = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf_wr = (win_sum / los_sum - 1) if los_sum > 0 else float('inf')
    from collections import Counter, defaultdict
    reasons = Counter(t['reason'].split('=')[0] for t in trades)
    longs = sum(1 for t in trades if t['side'] == 'LONG')
    shorts = n - longs
    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        day = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
        daily[day] += p
    ndays = max(1, len(daily))

    print(f'\n=== {run_name} (P_FIRE_ENTRY={p_fire_entry}  P_FIRE_EXIT={p_fire_exit}) ===')
    print(f'  n_trades       : {n}  (L={longs} S={shorts})')
    print(f'  active days    : {ndays}')
    print(f'  win rate (cnt) : {100*n_win/n:.1f}%  ({n_win}W / {n - n_win}L)')
    print(f'  PF-WR (CLAUDE) : {pf_wr:+.3f}')
    print(f'  total $        : {pnls.sum():.0f}')
    print(f'  $/day          : {pnls.sum()/ndays:.1f}')
    print(f'  mean per trade : {pnls.mean():+.2f}')
    print(f'  median         : {np.median(pnls):+.2f}')
    print(f'  exit reasons   : {dict(reasons)}')

    # Save trade log
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'v9_{run_name}_trades.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['side', 'entry_utc', 'exit_utc', 'entry_px', 'exit_px', 'pnl',
                       'dur_min', 'reason', 'p_at_entry', 'peak', 'worst'])
        for t, p in zip(trades, pnls):
            w.writerow([t['side'],
                            datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            round(t['entry_price'], 2), round(t['exit_price'], 2),
                            round(p, 2), round(t['dur_min'], 1),
                            t['reason'], round(t['p_at_entry'], 3),
                            round(t['peak'], 1), round(t['worst'], 1)])
    print(f'  log saved: {out}')


def _ts(d):
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', help='YYYY-MM-DD')
    ap.add_argument('--end',   help='YYYY-MM-DD')
    ap.add_argument('--name', default=None)
    ap.add_argument('--p-fire-entry', type=float, default=0.55)
    ap.add_argument('--p-fire-exit',  type=float, default=0.55)
    ap.add_argument('--hard-stop-ticks', type=float, default=30.0)
    ap.add_argument('--max-hold-min', type=int, default=60)
    ap.add_argument('--cooldown-min', type=int, default=5)
    ap.add_argument('--use-lr', action='store_true', help='Use LR instead of GBM')
    ap.add_argument('--sweep', action='store_true',
                       help='Sweep P_FIRE thresholds 0.40-0.80 step 0.05')
    args = ap.parse_args()

    print(f'Loading models from {MODELS_PATH}...')
    models = load_models()
    print(f'  models: {list(models.keys())}')

    t_start = _ts(args.start)
    t_end = _ts(args.end) + 86400
    run_name = args.name or f'{args.start}_{args.end}'

    use_gbm = not args.use_lr

    if args.sweep:
        print(f'\n=== Threshold sweep ({run_name}) ===')
        print(f'{"P_FIRE":>7} {"n_trades":>9} {"win_rate":>9} {"PF-WR":>7} {"$/day":>8}')
        for p in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            trades = simulate(t_start, t_end, models,
                                  p_fire_entry=p, p_fire_exit=p,
                                  hard_stop_ticks=args.hard_stop_ticks,
                                  max_hold_min=args.max_hold_min,
                                  cooldown_min=args.cooldown_min,
                                  use_gbm=use_gbm)
            if not trades:
                print(f'{p:>7.2f}  {0:>8}     no trades')
                continue
            pnls = np.array([(t['exit_price'] - t['entry_price']) / TICK * TICK_DOLLAR
                                  if t['side'] == 'LONG'
                                  else (t['entry_price'] - t['exit_price']) / TICK * TICK_DOLLAR
                                  for t in trades])
            n_win = (pnls > 0).sum()
            win_sum = pnls[pnls > 0].sum() if any(pnls > 0) else 0
            los_sum = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
            pf_wr = (win_sum / los_sum - 1) if los_sum > 0 else float('inf')
            from collections import defaultdict
            daily = defaultdict(float)
            for t, pn in zip(trades, pnls):
                day = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
                daily[day] += pn
            ndays = max(1, len(daily))
            print(f'{p:>7.2f} {len(trades):>9} {100*n_win/len(trades):>8.1f}% '
                      f'{pf_wr:>+7.3f} {pnls.sum()/ndays:>+8.1f}')
    else:
        trades = simulate(t_start, t_end, models,
                              p_fire_entry=args.p_fire_entry,
                              p_fire_exit=args.p_fire_exit,
                              hard_stop_ticks=args.hard_stop_ticks,
                              max_hold_min=args.max_hold_min,
                              cooldown_min=args.cooldown_min,
                              use_gbm=use_gbm)
        report(trades, run_name, args.p_fire_entry, args.p_fire_exit)


if __name__ == '__main__':
    main()
