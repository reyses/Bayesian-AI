"""Train GBM regressors on forward 60-min MFE (in $ dollars).

Per user 2026-05-12: "(p) needs to filter also for PnL — trade needs ≥$10".

For each bar in IS, compute forward 60-min MFE in each direction.
Train two regressors:
  mfe_short_dollars = max excursion DOWN within next 60 min
  mfe_long_dollars  = max excursion UP   within next 60 min

These predict EXPECTED $ — gate the strategy on predicted MFE >= viability_threshold.
"""
from __future__ import annotations
import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars
from tools.regret_1m_oracle import extract_state_vector, FORWARD_BARS

TICK = 0.25
TICK_DOLLAR = 0.50
OUT_PATH = Path('reports/findings/logistic_oracles/mfe_regressors.pkl')

DECISION_FEATURES = [
    'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_1m_15m', 'fan_width',
    'slope_1m_10m', 'slope_15m_5m', 'slope_15m_15m',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    '15m_above_Mh', '15m_below_Ml', '15m_near_Mh', '15m_near_Ml',
]


def _ts(d): return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def build_mfe_dataset(t_start: float, t_end: float,
                              sample_every: int = 5) -> pd.DataFrame:
    """At every Nth bar, compute features + forward 60-min MFE in both directions."""
    df = load_1m_bars(t_start, t_end)
    if df.empty:
        return pd.DataFrame()
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    print(f'  {len(ts)} 1m bars; computing anchors...')

    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    rows = []
    print(f'  Walking bars (sample every {sample_every})...')
    for i in range(20, len(ts) - FORWARD_BARS, sample_every):
        # Forward MFE in both directions
        fwd_low_min = float(low[i + 1 : i + 1 + FORWARD_BARS].min())
        fwd_high_max = float(high[i + 1 : i + 1 + FORWARD_BARS].max())
        mfe_short_dollars = (close[i] - fwd_low_min) / TICK * TICK_DOLLAR
        mfe_long_dollars  = (fwd_high_max - close[i]) / TICK * TICK_DOLLAR

        state = extract_state_vector(i, close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc)
        rows.append({
            'bar_idx': i, 'timestamp': int(ts[i]),
            'mfe_short_dollars': round(mfe_short_dollars, 2),
            'mfe_long_dollars':  round(mfe_long_dollars, 2),
            **{k: state.get(k) for k in DECISION_FEATURES},
        })
    return pd.DataFrame(rows)


def train_regressors(df: pd.DataFrame):
    """Train one GBM regressor for each direction."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    for c in DECISION_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    X = df[DECISION_FEATURES].values
    models = {}
    for target in ('mfe_short_dollars', 'mfe_long_dollars'):
        y = df[target].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            loss='squared_error')
        reg.fit(X_tr, y_tr)
        pred_te = reg.predict(X_te)
        mae = mean_absolute_error(y_te, pred_te)
        r2 = r2_score(y_te, pred_te)

        # Refit on full
        reg_full = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            loss='squared_error')
        reg_full.fit(X, y)
        print(f'  [{target}] MAE={mae:.2f}  R²={r2:.3f}  '
                  f'(mean actual ${y.mean():.0f}, median ${np.median(y):.0f})')
        models[target] = {
            'reg_model': reg_full,
            'feature_cols': DECISION_FEATURES,
            'mae_test': mae, 'r2_test': r2,
        }
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-04-01')
    ap.add_argument('--end',   default='2025-10-31')
    ap.add_argument('--sample-every', type=int, default=5,
                       help='Sample every Nth bar (5 = ~1 sample per 5 min)')
    args = ap.parse_args()

    t_start = _ts(args.start); t_end = _ts(args.end) + 86400
    print(f'Building MFE training dataset {args.start} → {args.end}...')
    df = build_mfe_dataset(t_start, t_end, sample_every=args.sample_every)
    print(f'  {len(df)} sampled bars\n')

    print('Distribution of forward MFE values:')
    print(f'  SHORT MFE: median ${df["mfe_short_dollars"].median():.0f}  '
              f'mean ${df["mfe_short_dollars"].mean():.0f}  '
              f'q75 ${df["mfe_short_dollars"].quantile(.75):.0f}')
    print(f'  LONG  MFE: median ${df["mfe_long_dollars"].median():.0f}  '
              f'mean ${df["mfe_long_dollars"].mean():.0f}  '
              f'q75 ${df["mfe_long_dollars"].quantile(.75):.0f}')

    print('\nTraining GBM regressors (300 epochs, early stopping)...')
    models = train_regressors(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(models, f)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
