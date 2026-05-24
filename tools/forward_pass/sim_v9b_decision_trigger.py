"""v9b — DECISION (1m/15m) × TRIGGER (15s) architecture.

Per user 2026-05-12 evening:
    decision  =  trained on 1m + 15m + 1h features only  (slow context)
    trigger   =  explicit 15s slope sign-flip event       (fast firing)
    fire      =  decision_P >= threshold AND trigger event present

This separates the SETUP from the TIMING — same way professional discretionary
traders work: "we want to short" (decision) vs "now is the moment" (trigger).

PROCESS:
  1. Retrain decision-only models on DECISION_FEATURES subset
     (excludes z_15s, slope_15s_*, dist_15s_*)
  2. Compute 15s slope sign-flip flags at each bar
  3. Simulator: fire when both gates open
"""
from __future__ import annotations
import argparse
import csv
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
OUT_DIR = Path('reports/findings/sim_v9b')
TRAIN_DATA = Path('reports/findings/logistic_oracles/oracle_train_data.csv')
DECISION_MODELS_PATH = Path('reports/findings/logistic_oracles/decision_models.pkl')

# DECISION feature set — exclude all 15s features
DECISION_FEATURES = [
    'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_1m_15m', 'fan_width',
    'slope_1m_10m', 'slope_15m_5m', 'slope_15m_15m',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    '15m_above_Mh', '15m_below_Ml', '15m_near_Mh', '15m_near_Ml',
]


# ── (1) Train decision-only GBM models ──────────────────────────────────────

def train_decision_models(verbose: bool = True):
    """Retrain GBM models using ONLY 1m/15m/1h features (no 15s features).
    Saves separate pickle for use by the simulator."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(f'{TRAIN_DATA} missing — run logistic_oracle_surfaces.py first')

    df = pd.read_csv(TRAIN_DATA)
    for c in DECISION_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    surfaces = [
        ('is_entry_short', 'entry_short'),
        ('is_entry_long',  'entry_long'),
        ('is_exit_short',  'exit_short'),
        ('is_exit_long',   'exit_long'),
    ]
    out = []
    for label_col, name in surfaces:
        pos = df[df[label_col] == 1]
        neg = df[df['is_random'] == 1]
        if len(pos) < 30 or len(neg) < 30:
            continue
        full = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
        X = full[DECISION_FEATURES].astype(float).fillna(0).values
        y = full['y'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        clf = HistGradientBoostingClassifier(
            max_iter=300, class_weight='balanced',
            learning_rate=0.05, max_depth=6, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
        clf.fit(X_tr, y_tr)
        auc_tr = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])
        auc_te = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

        clf_full = HistGradientBoostingClassifier(
            max_iter=300, class_weight='balanced',
            learning_rate=0.05, max_depth=6, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
        clf_full.fit(X, y)

        if verbose:
            print(f'  [{name}] n_pos={len(pos)} n_neg={len(neg)}  '
                      f'GBM AUC train={auc_tr:.3f} test={auc_te:.3f}  '
                      f'(decision-only features)')
        out.append({
            'name': name,
            'gbm_model': clf_full,
            'feature_cols': DECISION_FEATURES,
            'auc_train': auc_tr, 'auc_test': auc_te,
            'n_pos': len(pos), 'n_neg': len(neg),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DECISION_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_MODELS_PATH, 'wb') as f:
        pickle.dump(out, f)
    if verbose:
        print(f'\nSaved decision-only models: {DECISION_MODELS_PATH}')
    return out


# ── (2) 15s trigger detection ───────────────────────────────────────────────

def compute_15s_triggers(M_15s: np.ndarray, lb_long: int = 10,
                                  lb_short: int = 3, thr_long: float = 0.20,
                                  thr_short: float = 0.10) -> tuple:
    """Compute 15s slope sign-flip flags at each bar.
    Returns (flip_down, flip_up) — boolean arrays.
    flip_down[i] = True when slope_long was positive (rising) AND
                   slope_short turned negative (now falling).
                   This is the SHORT cusp signal.
    flip_up[i]  = mirror: long was falling, short turned positive — LONG cusp.
    """
    n = len(M_15s)
    slope_long = np.full(n, np.nan)
    slope_short = np.full(n, np.nan)
    if n > lb_long:
        slope_long[lb_long:] = (M_15s[lb_long:] - M_15s[:-lb_long]) / lb_long
    if n > lb_short:
        slope_short[lb_short:] = (M_15s[lb_short:] - M_15s[:-lb_short]) / lb_short

    flip_down = (slope_long >= thr_long) & (slope_short <= -thr_short)
    flip_up   = (slope_long <= -thr_long) & (slope_short >= thr_short)
    # Coerce NaNs to False
    flip_down = np.where(np.isnan(slope_long) | np.isnan(slope_short), False, flip_down)
    flip_up   = np.where(np.isnan(slope_long) | np.isnan(slope_short), False, flip_up)
    return flip_down, flip_up


# ── (3) Simulator: decision × trigger ───────────────────────────────────────

def simulate(t_start, t_end, models_by_name: dict,
                p_fire_entry: float = 0.55, p_fire_exit: float = 0.55,
                use_15s_trigger: bool = True,
                hard_stop_ticks: float = 30, max_hold_min: int = 60,
                cooldown_min: int = 0) -> list:
    df = load_1m_bars(t_start, t_end)
    if df.empty: return []
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    print(f'  {len(df)} bars; computing anchors...')
    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    # 15s trigger flags
    flip_down, flip_up = compute_15s_triggers(M_15s)

    # Extract decision feature matrix
    es = models_by_name['entry_short']
    el = models_by_name['entry_long']
    xs = models_by_name['exit_short']
    xl = models_by_name['exit_long']
    feature_cols = es['feature_cols']

    print(f'  extracting decision features for {len(ts)} bars...')
    X = np.zeros((len(ts), len(feature_cols)), dtype=float)
    for i in range(len(ts)):
        if i < 15: continue
        state = extract_state_vector(i, close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc)
        for j, fc in enumerate(feature_cols):
            v = state.get(fc, 0.0)
            X[i, j] = 0.0 if v is None else float(v)

    print('  batch-predicting decision P...')
    p_es = es['gbm_model'].predict_proba(X)[:, 1]
    p_el = el['gbm_model'].predict_proba(X)[:, 1]
    p_xs = xs['gbm_model'].predict_proba(X)[:, 1]
    p_xl = xl['gbm_model'].predict_proba(X)[:, 1]

    print('  walking bar-by-bar with decision × trigger gate...')
    trades = []
    open_t = None
    cooldown_until = 0.0

    for i in range(15, len(ts)):
        if open_t is not None:
            if open_t['side'] == 'LONG':
                pnl_ticks = (close[i] - open_t['entry_price']) / TICK
            else:
                pnl_ticks = (open_t['entry_price'] - close[i]) / TICK
            open_t['peak'] = max(open_t['peak'], pnl_ticks)
            open_t['worst'] = min(open_t['worst'], pnl_ticks)
            dur = (ts[i] - open_t['entry_ts']) / 60.0
            close_now = False; reason = ''
            if pnl_ticks <= -hard_stop_ticks:
                close_now = True; reason = 'hard_stop'
            elif open_t['side'] == 'SHORT' and p_xs[i] >= p_fire_exit:
                close_now = True; reason = f'p_exit_short={p_xs[i]:.2f}'
            elif open_t['side'] == 'LONG' and p_xl[i] >= p_fire_exit:
                close_now = True; reason = f'p_exit_long={p_xl[i]:.2f}'
            elif dur >= max_hold_min:
                close_now = True; reason = 'time_stop'
            if close_now:
                open_t['exit_ts'] = ts[i]; open_t['exit_price'] = close[i]
                open_t['reason'] = reason; open_t['dur_min'] = dur
                trades.append(open_t); open_t = None
                cooldown_until = ts[i] + cooldown_min * 60
            else:
                continue

        if ts[i] < cooldown_until:
            continue

        # ── DECISION × TRIGGER gate ─────────────
        # Must have BOTH high decision-P AND the 15s trigger event
        short_ok = (p_es[i] >= p_fire_entry) and (flip_down[i] if use_15s_trigger else True)
        long_ok  = (p_el[i] >= p_fire_entry) and (flip_up[i]   if use_15s_trigger else True)

        if short_ok and (p_es[i] >= p_el[i]):
            open_t = {'side':'SHORT', 'entry_ts':int(ts[i]), 'entry_price':float(close[i]),
                          'p_at_entry':float(p_es[i]), 'trigger_fired':bool(flip_down[i]),
                          'peak':0.0, 'worst':0.0}
        elif long_ok:
            open_t = {'side':'LONG', 'entry_ts':int(ts[i]), 'entry_price':float(close[i]),
                          'p_at_entry':float(p_el[i]), 'trigger_fired':bool(flip_up[i]),
                          'peak':0.0, 'worst':0.0}

    if open_t is not None:
        open_t['exit_ts'] = int(ts[-1]); open_t['exit_price'] = float(close[-1])
        open_t['reason'] = 'eod'; open_t['dur_min'] = (ts[-1] - open_t['entry_ts']) / 60.0
        trades.append(open_t)
    return trades


def report(trades, run_name, p_entry, p_exit, use_trigger):
    if not trades:
        print(f'{run_name}: 0 trades'); return
    pnls = np.array([(t['exit_price']-t['entry_price'])/TICK*TICK_DOLLAR if t['side']=='LONG'
                          else (t['entry_price']-t['exit_price'])/TICK*TICK_DOLLAR
                          for t in trades])
    n = len(trades)
    n_win = (pnls > 0).sum()
    win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf = (win/lose - 1) if lose > 0 else float('inf')
    from collections import defaultdict
    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        d = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
        daily[d] += p
    nd = max(1, len(daily))
    trig_str = 'WITH 15s trigger' if use_trigger else 'NO 15s trigger (decision-only)'
    print(f'\n=== {run_name}  [P_entry={p_entry} P_exit={p_exit}, {trig_str}] ===')
    print(f'  n_trades={n}  win_rate={100*n_win/n:.1f}%  PF-WR={pf:+.3f}  '
              f'total ${pnls.sum():.0f}  $/day ${pnls.sum()/nd:.1f}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f'v9b_{run_name}_trades.csv'
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['side','entry_utc','exit_utc','entry_px','exit_px','pnl','dur_min',
                       'reason','p_at_entry','trigger_fired','peak','worst'])
        for t, p in zip(trades, pnls):
            w.writerow([t['side'],
                            datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            round(t['entry_price'],2), round(t['exit_price'],2),
                            round(p,2), round(t['dur_min'],1), t['reason'],
                            round(t['p_at_entry'],3), t.get('trigger_fired', False),
                            round(t['peak'],1), round(t['worst'],1)])


def _ts(d): return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end',   required=True)
    ap.add_argument('--name',  default='OOS')
    ap.add_argument('--p-fire-entry', type=float, default=0.55)
    ap.add_argument('--p-fire-exit',  type=float, default=0.55)
    ap.add_argument('--no-trigger', action='store_true', help='Disable 15s trigger gate')
    ap.add_argument('--retrain', action='store_true', help='Retrain decision-only models')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()

    # Train if missing or requested
    if args.retrain or not DECISION_MODELS_PATH.exists():
        print('Training decision-only models on full IS...')
        models_list = train_decision_models()
    else:
        with open(DECISION_MODELS_PATH, 'rb') as f:
            models_list = pickle.load(f)
        print(f'Loaded {len(models_list)} decision-only models from disk')

    models_by_name = {m['name']: m for m in models_list}

    t_start = _ts(args.start); t_end = _ts(args.end) + 86400

    if args.sweep:
        print(f'\n=== SWEEP  ({args.name}, trigger={"ON" if not args.no_trigger else "OFF"}) ===')
        print(f'{"P_entry":>8} {"P_exit":>7} {"n_trades":>9} {"win_rate":>9} {"PF-WR":>7} {"$/day":>8}')
        for p in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            trades = simulate(t_start, t_end, models_by_name,
                                  p_fire_entry=p, p_fire_exit=args.p_fire_exit,
                                  use_15s_trigger=not args.no_trigger,
                                  cooldown_min=0)
            if not trades:
                print(f'{p:>7.2f}  {args.p_fire_exit:>6.2f}  no trades')
                continue
            pnls = np.array([(t['exit_price']-t['entry_price'])/TICK*TICK_DOLLAR if t['side']=='LONG'
                                  else (t['entry_price']-t['exit_price'])/TICK*TICK_DOLLAR
                                  for t in trades])
            n_win = (pnls > 0).sum()
            win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
            lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
            pf = (win/lose - 1) if lose > 0 else float('inf')
            from collections import defaultdict
            daily = defaultdict(float)
            for t, pn in zip(trades, pnls):
                d = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
                daily[d] += pn
            nd = max(1, len(daily))
            print(f'{p:>7.2f}  {args.p_fire_exit:>6.2f}  {len(trades):>9}  '
                      f'{100*n_win/len(trades):>7.1f}%  {pf:>+6.3f}  {pnls.sum()/nd:>+7.1f}')
    else:
        trades = simulate(t_start, t_end, models_by_name,
                              p_fire_entry=args.p_fire_entry,
                              p_fire_exit=args.p_fire_exit,
                              use_15s_trigger=not args.no_trigger,
                              cooldown_min=0)
        report(trades, args.name, args.p_fire_entry, args.p_fire_exit,
                  use_trigger=not args.no_trigger)


if __name__ == '__main__':
    main()
