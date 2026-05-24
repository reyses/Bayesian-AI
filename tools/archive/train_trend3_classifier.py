"""Train 3-class TREND-DIRECTION classifier.

Per user 2026-05-17: the classifier's job RIGHT NOW is to tell us
'are we in an UP leg, DOWN leg, or transition?' — not to predict pivots.
Action logic (when to fire) comes later.

Labels (from build_zigzag_pivot_dataset.py trend_class column):
  LONG     = bar is inside an up-leg AND outside the ±2min transition zone
  SHORT    = bar is inside a down-leg AND outside the transition zone
  NEUTRAL  = bar is within ±2min of any zigzag pivot (transition)

Trains LR or GBM with multi_class='multinomial'. Reports OOS NT8 accuracy,
confusion matrix, and per-class precision/recall.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset', required=True)
    ap.add_argument('--oos-dataset', required=True)
    ap.add_argument('--out-pkl', required=True)
    ap.add_argument('--model', choices=['lr','gbm'], default='gbm')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    is_df  = pd.read_parquet(args.is_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'IS  rows: {len(is_df)}')
    print(f'OOS rows: {len(oos_df)}')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_','L2_','L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    # Filter to bars with a labeled trend_class (NEUTRAL/LONG/SHORT)
    is_df = is_df[is_df['trend_class'].isin(['LONG','SHORT','NEUTRAL'])].copy()
    oos_df = oos_df[oos_df['trend_class'].isin(['LONG','SHORT','NEUTRAL'])].copy()
    print(f'  IS filtered: {len(is_df)}')
    print(f'  OOS filtered: {len(oos_df)}')

    X_is  = is_df[v2_cols].fillna(0).values.astype(np.float32)
    X_oos = oos_df[v2_cols].fillna(0).values.astype(np.float32)

    le = LabelEncoder()
    le.fit(['LONG', 'SHORT', 'NEUTRAL'])
    y_is  = le.transform(is_df['trend_class'].values)
    y_oos = le.transform(oos_df['trend_class'].values)
    classes = list(le.classes_)
    print(f'Classes (encoded order): {classes}')

    # 80/20 IS-test split (seed 42 for parity)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X_is)); rng.shuffle(idx)
    n_test = int(0.2 * len(X_is))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_is[train_idx])
    X_te_s = scaler.transform(X_is[test_idx])
    X_oos_s = scaler.transform(X_oos)

    print(f'\nFitting {args.model}...')
    import time
    t0 = time.time()
    if args.model == 'lr':
        clf = LogisticRegression(max_iter=600, C=1.0, solver='lbfgs',
                                  class_weight='balanced', n_jobs=-1)
        clf.fit(X_tr_s, y_is[train_idx])
        proba_te  = clf.predict_proba(X_te_s)
        proba_oos = clf.predict_proba(X_oos_s)
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8,
                                              learning_rate=0.08,
                                              class_weight='balanced',
                                              random_state=42)
        clf.fit(X_is[train_idx], y_is[train_idx])
        proba_te  = clf.predict_proba(X_is[test_idx])
        proba_oos = clf.predict_proba(X_oos)
    print(f'  Fit in {time.time()-t0:.1f}s')

    pred_te  = proba_te.argmax(1)
    pred_oos = proba_oos.argmax(1)

    print(f'\n=== IS-test ===')
    print(f'  Acc: {accuracy_score(y_is[test_idx], pred_te):.4f}')
    print(f'  Confusion matrix (rows=true, cols=pred):')
    cm = confusion_matrix(y_is[test_idx], pred_te)
    print('  ' + '  '.join(f'{c:>8s}' for c in classes))
    for i, row in enumerate(cm):
        print(f'  {classes[i]:>8s} ' + ' '.join(f'{v:>8d}' for v in row))
    print('\n  Per-class:')
    print(classification_report(y_is[test_idx], pred_te,
                                target_names=classes, digits=4))

    print(f'\n=== NT8 OOS ===')
    print(f'  Acc: {accuracy_score(y_oos, pred_oos):.4f}')
    print(f'  Confusion matrix (rows=true, cols=pred):')
    cm_oos = confusion_matrix(y_oos, pred_oos)
    print('  ' + '  '.join(f'{c:>8s}' for c in classes))
    for i, row in enumerate(cm_oos):
        print(f'  {classes[i]:>8s} ' + ' '.join(f'{v:>8d}' for v in row))
    print('\n  Per-class:')
    print(classification_report(y_oos, pred_oos, target_names=classes, digits=4))

    # Threshold sweep: at what P-threshold does the model fire cleanly?
    long_idx    = classes.index('LONG')
    short_idx   = classes.index('SHORT')
    neutral_idx = classes.index('NEUTRAL')
    print(f'\n=== OOS threshold sweep (fire if argmax-prob > T AND class != NEUTRAL) ===')
    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        argmax_p = proba_oos.max(1)
        argmax_c = proba_oos.argmax(1)
        fire = (argmax_p > t) & (argmax_c != neutral_idx)
        n_fire = int(fire.sum())
        if n_fire == 0:
            print(f'  T={t:.2f}  no fires')
            continue
        correct = (argmax_c[fire] == y_oos[fire]).sum()
        acc = correct / n_fire
        cov = fire.mean()
        long_fire = (argmax_c == long_idx) & (argmax_p > t)
        short_fire = (argmax_c == short_idx) & (argmax_p > t)
        long_acc  = (y_oos[long_fire] == long_idx).mean()  if long_fire.any()  else float('nan')
        short_acc = (y_oos[short_fire] == short_idx).mean() if short_fire.any() else float('nan')
        print(f'  T={t:.2f}  cov={cov:.3f}  fires={n_fire}  acc={acc:.3f}  '
              f'LONG_prec={long_acc:.3f}  SHORT_prec={short_acc:.3f}')

    # Refit on full IS
    print(f'\nRefitting on full IS...')
    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X_is)
    if args.model == 'lr':
        clf_full = LogisticRegression(max_iter=600, C=1.0, solver='lbfgs',
                                       class_weight='balanced', n_jobs=-1)
        clf_full.fit(X_all_s, y_is)
    else:
        clf_full = HistGradientBoostingClassifier(max_iter=300, max_depth=8,
                                                   learning_rate=0.08,
                                                   class_weight='balanced',
                                                   random_state=42)
        clf_full.fit(X_is, y_is)

    out = Path(args.out_pkl); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump({
            'scaler': scaler_full if args.model == 'lr' else None,
            'clf': clf_full,
            'v2_cols': v2_cols,
            'classes': classes,
            'label_encoder': le,
            'model_kind': args.model,
            'three_class': True,
        }, f)
    print(f'\nWrote: {out}')


if __name__ == '__main__':
    main()
