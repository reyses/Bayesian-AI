"""Logistic regression P-surfaces for oracle entry + exit.

Per user 2026-05-12: treat each oracle event as a logistic regression
surface — continuous P over feature space, so we can build a strategy
that consults the surface at every bar.

FOUR SURFACES TRAINED:
    P(oracle_entry | state, direction=SHORT)   — peaks of M_1m
    P(oracle_entry | state, direction=LONG)    — troughs of M_1m
    P(oracle_exit  | state, direction=SHORT)   — the MFE point within 60min
    P(oracle_exit  | state, direction=LONG)    — same, for longs

TRAINING DATA per surface:
    Positive class: bars labeled as oracle event (entry or exit of an oracle trade)
    Negative class: random bars (sampled from same period, excluding event bars)
    Features:       state vector (z's, distances, slopes, fan, rail position, stack)

OUTPUT:
    reports/findings/logistic_oracles/
        entry_short_coefs.csv    feature weights
        entry_long_coefs.csv
        exit_short_coefs.csv
        exit_long_coefs.csv
        model_metrics.txt         AUC + train/test
        oracle_train_data.csv     the full labeled dataset

USAGE:
    python tools/logistic_oracle_surfaces.py --start 2025-04-01 --end 2025-10-31
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
from tools.regret_1m_oracle import (
    detect_local_extrema, forward_mfe, extract_state_vector,
    FORWARD_BARS, TICK_DOLLAR,
)


OUT_DIR = Path('reports/findings/logistic_oracles')

# Numeric features used as logistic-regression inputs
NUMERIC_FEATURES = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    '15m_above_Mh', '15m_below_Ml', '15m_near_Mh', '15m_near_Ml',
]


def _ts(d):
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def build_dataset(t_start: float, t_end: float, peak_window: int = 15,
                       min_sep: int = 10, n_random_per_event: int = 3,
                       seed: int = 42) -> pd.DataFrame:
    """Walk the time window. Build 4 labeled datasets in one pass:
        - entry_short: M_1m peaks  + random non-peak bars
        - entry_long:  M_1m troughs + random non-trough bars
        - exit_short:  the MFE-bar of each oracle SHORT trade + random bars
        - exit_long:   same for LONG

    Returns one DataFrame with columns: state features + 4 label columns +
    direction + bar_idx + timestamp.
    """
    print(f'\nBuilding dataset {datetime.fromtimestamp(t_start, tz=timezone.utc)} '
              f'→ {datetime.fromtimestamp(t_end, tz=timezone.utc)}...')
    df = load_1m_bars(t_start, t_end)
    if df.empty:
        print('No data'); return pd.DataFrame()
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    n = len(df)
    print(f'  {n} 1m bars')

    print('  Computing anchors...')
    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    print('  Detecting M_1m local extrema...')
    peaks, troughs = detect_local_extrema(M_1m, window=peak_window, min_sep=min_sep)
    print(f'    peaks (SHORT entries):  {len(peaks)}')
    print(f'    troughs (LONG entries): {len(troughs)}')

    # For each entry, compute forward MFE + the exit bar (where MFE was achieved)
    print('  Computing forward MFE + exit bars for each oracle entry...')
    short_entries = []
    short_exits = []
    for idx in peaks:
        mfe_ticks, ttm, exit_idx, exit_px = forward_mfe(
            close, high, low, int(idx), 'SHORT', FORWARD_BARS)
        if mfe_ticks >= 20:   # only count meaningful oracle MFE
            short_entries.append(int(idx))
            short_exits.append(int(exit_idx))
    long_entries = []
    long_exits = []
    for idx in troughs:
        mfe_ticks, ttm, exit_idx, exit_px = forward_mfe(
            close, high, low, int(idx), 'LONG', FORWARD_BARS)
        if mfe_ticks >= 20:
            long_entries.append(int(idx))
            long_exits.append(int(exit_idx))

    print(f'    Filtered SHORT entries (MFE>=20t): {len(short_entries)}')
    print(f'    Filtered LONG  entries (MFE>=20t): {len(long_entries)}')

    # Build label sets: each bar gets labels for each of the 4 surfaces
    entry_short_set = set(short_entries)
    entry_long_set  = set(long_entries)
    exit_short_set  = set(short_exits)
    exit_long_set   = set(long_exits)

    # Random sample: bars that are NOT any of the event sets
    print('  Sampling random non-event bars...')
    rng = np.random.RandomState(seed)
    all_event = entry_short_set | entry_long_set | exit_short_set | exit_long_set
    candidate_pool = [i for i in range(peak_window + 15, n - FORWARD_BARS)
                            if i not in all_event]
    n_random = min(len(candidate_pool),
                          n_random_per_event * (len(entry_short_set) + len(entry_long_set)))
    random_idxs = set(rng.choice(candidate_pool, size=n_random, replace=False).tolist())

    # All bars we'll extract features for
    all_idxs = sorted(entry_short_set | entry_long_set | exit_short_set | exit_long_set
                            | random_idxs)
    print(f'  Extracting state vectors at {len(all_idxs)} labeled bars...')

    rows = []
    for idx in all_idxs:
        state = extract_state_vector(idx, close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc)
        rows.append({
            'bar_idx': idx, 'timestamp': int(ts[idx]),
            'utc': datetime.fromtimestamp(int(ts[idx]), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            **{k: state.get(k) for k in NUMERIC_FEATURES},
            'is_entry_short': int(idx in entry_short_set),
            'is_entry_long':  int(idx in entry_long_set),
            'is_exit_short':  int(idx in exit_short_set),
            'is_exit_long':   int(idx in exit_long_set),
            'is_random':      int(idx in random_idxs and idx not in all_event),
        })
    out_df = pd.DataFrame(rows)
    print(f'  Dataset shape: {out_df.shape}')
    return out_df


def train_logistic(df: pd.DataFrame, label_col: str, feature_cols: list,
                          name: str, lr_max_iter: int = 10000,
                          gbm_max_iter: int = 300) -> dict:
    """Fit BOTH logistic regression AND HistGradientBoosting.
    LR is the interpretable baseline; GBM captures non-linearity + interactions
    (the "more epochs ML approach" — each iter = one boosting round).
    Returns both models + their AUCs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import roc_auc_score

    pos = df[df[label_col] == 1].copy()
    neg = df[df['is_random'] == 1].copy()

    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos < 30 or n_neg < 30:
        print(f'  [{name}] insufficient samples: pos={n_pos} neg={n_neg}; skipping')
        return None

    full = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    X = full[feature_cols].astype(float).fillna(0).values
    y = full['y'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 70/30 split for both models
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=y)

    # ── Logistic Regression baseline ──────────────────────────────
    lr = LogisticRegression(max_iter=lr_max_iter, class_weight='balanced', C=1.0)
    lr.fit(X_tr, y_tr)
    lr_auc_tr = roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])
    lr_auc_te = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

    lr_full = LogisticRegression(max_iter=lr_max_iter, class_weight='balanced', C=1.0)
    lr_full.fit(Xs, y)
    coefs = dict(zip(feature_cols, lr_full.coef_[0].tolist()))
    intercept = float(lr_full.intercept_[0])

    # ── Gradient Boosting (the "more epochs" ML approach) ─────────
    # max_iter = number of boosting rounds (the "epochs" of GBM)
    # Uses raw X (not scaled) since trees are scale-invariant
    X_raw = full[feature_cols].astype(float).fillna(0).values
    X_raw_tr, X_raw_te = train_test_split(X_raw, test_size=0.3, random_state=42, stratify=y)[:2]
    y_tr_raw, y_te_raw = train_test_split(y, test_size=0.3, random_state=42, stratify=y)[:2]

    gbm = HistGradientBoostingClassifier(
        max_iter=gbm_max_iter,
        class_weight='balanced',
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )
    gbm.fit(X_raw_tr, y_tr_raw)
    gbm_auc_tr = roc_auc_score(y_tr_raw, gbm.predict_proba(X_raw_tr)[:, 1])
    gbm_auc_te = roc_auc_score(y_te_raw, gbm.predict_proba(X_raw_te)[:, 1])
    gbm_iters_used = gbm.n_iter_

    # Refit GBM on full data for deployment
    gbm_full = HistGradientBoostingClassifier(
        max_iter=gbm_max_iter,
        class_weight='balanced',
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )
    gbm_full.fit(X_raw, y)

    print(f'  [{name}] n_pos={n_pos}  n_neg={n_neg}')
    print(f'         LR  AUC train={lr_auc_tr:.3f}  test={lr_auc_te:.3f}')
    print(f'         GBM AUC train={gbm_auc_tr:.3f}  test={gbm_auc_te:.3f}  '
              f'(stopped at iter {gbm_iters_used}/{gbm_max_iter})')

    return {
        'name': name,
        'n_pos': n_pos, 'n_neg': n_neg,
        'lr_auc_train': lr_auc_tr, 'lr_auc_test': lr_auc_te,
        'gbm_auc_train': gbm_auc_tr, 'gbm_auc_test': gbm_auc_te,
        'gbm_iters_used': gbm_iters_used,
        # LR for interpretability
        'coefs': coefs, 'intercept': intercept,
        'feature_means': dict(zip(feature_cols, scaler.mean_.tolist())),
        'feature_stds': dict(zip(feature_cols, scaler.scale_.tolist())),
        # Models for deployment
        'lr_model': lr_full,
        'gbm_model': gbm_full,
        'scaler': scaler,
        'feature_cols': feature_cols,
        # Aliases for backward compat
        'auc_train': lr_auc_tr, 'auc_test': lr_auc_te,
    }


def write_coefs(model: dict, path: Path):
    if model is None:
        return
    coefs_sorted = sorted(model['coefs'].items(), key=lambda kv: abs(kv[1]), reverse=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'coef', 'abs_coef', 'feature_mean', 'feature_std'])
        for feat, coef in coefs_sorted:
            w.writerow([feat, round(coef, 4), round(abs(coef), 4),
                            round(model['feature_means'].get(feat, 0), 3),
                            round(model['feature_stds'].get(feat, 1), 3)])


def write_metrics(models: list, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Logistic Oracle Surfaces — Model Metrics\n')
        f.write('=' * 60 + '\n\n')
        for m in models:
            if m is None:
                continue
            f.write(f'{m["name"]}:\n')
            f.write(f'  n_pos:     {m["n_pos"]}\n')
            f.write(f'  n_neg:     {m["n_neg"]}\n')
            f.write(f'  AUC train: {m["auc_train"]:.3f}\n')
            f.write(f'  AUC test:  {m["auc_test"]:.3f}\n')
            f.write(f'  intercept: {m["intercept"]:.3f}\n\n')
            f.write(f'  Top 10 features by |coef|:\n')
            top = sorted(m['coefs'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            for feat, coef in top:
                f.write(f'    {feat:<24}  {coef:+.4f}\n')
            f.write('\n' + '-' * 60 + '\n\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-04-01')
    ap.add_argument('--end',   default='2025-10-31')
    ap.add_argument('--peak-window', type=int, default=15)
    ap.add_argument('--min-sep',     type=int, default=10)
    ap.add_argument('--n-random-per-event', type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = _ts(args.start)
    t_end   = _ts(args.end) + 86400
    df = build_dataset(t_start, t_end, args.peak_window, args.min_sep,
                              args.n_random_per_event)
    if df.empty:
        return

    # Save the labeled dataset
    train_path = OUT_DIR / 'oracle_train_data.csv'
    df.to_csv(train_path, index=False)
    print(f'\nSaved labeled training data: {train_path}')

    # Fit 4 surfaces
    print('\nTraining logistic surfaces...')
    models = []
    for label_col, surface_name in [
        ('is_entry_short', 'entry_short'),
        ('is_entry_long',  'entry_long'),
        ('is_exit_short',  'exit_short'),
        ('is_exit_long',   'exit_long'),
    ]:
        m = train_logistic(df, label_col, NUMERIC_FEATURES, surface_name)
        models.append(m)
        if m is not None:
            write_coefs(m, OUT_DIR / f'{surface_name}_coefs.csv')

    # Save models (serialized) for inference reuse
    serializable = []
    for m in models:
        if m is None:
            continue
        # Strip non-pickleable refs? sklearn models are pickleable
        serializable.append({
            'name': m['name'],
            'lr_model': m['lr_model'],
            'gbm_model': m['gbm_model'],
            'scaler': m['scaler'],
            'feature_cols': m['feature_cols'],
            'coefs': m['coefs'], 'intercept': m['intercept'],
            'lr_auc_test': m['lr_auc_test'],
            'gbm_auc_test': m['gbm_auc_test'],
        })
    with open(OUT_DIR / 'models.pkl', 'wb') as f:
        pickle.dump(serializable, f)
    print(f'\nSaved {len(serializable)} models to {OUT_DIR / "models.pkl"}')

    write_metrics(models, OUT_DIR / 'model_metrics.txt')

    # ── Final AUC comparison: LR vs GBM ────────────────────────────
    print('\n' + '=' * 70)
    print('  FINAL AUC COMPARISON  (LR vs GBM on test 30%)')
    print('=' * 70)
    print(f'{"surface":<14} {"n_pos":>6} {"LR_test":>9} {"GBM_test":>9} {"lift":>6}')
    print('-' * 50)
    for m in models:
        if m is None:
            continue
        lift = m['gbm_auc_test'] - m['lr_auc_test']
        print(f'{m["name"]:<14} {m["n_pos"]:>6} '
                  f'{m["lr_auc_test"]:>9.3f} {m["gbm_auc_test"]:>9.3f} '
                  f'{lift:>+6.3f}')

    # Top LR coefs (interpretation)
    print('\n=== TOP LR FEATURES BY |COEF| PER SURFACE ===')
    for m in models:
        if m is None:
            continue
        print(f'\n  {m["name"]}  LR={m["lr_auc_test"]:.3f}  GBM={m["gbm_auc_test"]:.3f}')
        top = sorted(m['coefs'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        for feat, coef in top:
            print(f'    {feat:<22}  {coef:+.4f}')

    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
