"""B5 — Leg-phase multi-class classifier.

Per user 2026-05-17: "i think is better to train models so signals are
slightly diferent and we can make a composite."

This trains a SEPARATE model with a DIFFERENT label than B4. Even though
the underlying truth is related (both depend on zigzag pivots), the GBM
sees a different objective and learns slightly different decision
boundaries → ensemble diversity in the composite.

Target: leg_phase ∈ {EARLY, MID, LATE}
  leg_age_ratio = (time_since_last_pivot) / (total_leg_duration)
  EARLY = ratio in [0.0, 0.25)   -- just past pivot, new leg fresh
  MID   = ratio in [0.25, 0.75)  -- middle of the run, deepest "in trend"
  LATE  = ratio in [0.75, 1.0]   -- approaching next pivot, leg dying

This answers your "counter" question richer than just "in_trend" binary:
  - MID = "deep in trend, ride confidently"
  - EARLY = "just flipped, ride with conviction (new leg fresh)"
  - LATE = "leg ending, defensive mode"

Train: full IS (282k bars)
Eval:  NT8 OOS (32 days)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


PHASES = ['EARLY', 'MID', 'LATE']


def pivot_centroid_ts(day_df: pd.DataFrame) -> np.ndarray:
    """Collapse is_pivot==1 runs to centroid timestamps."""
    piv_rows = day_df[day_df['is_pivot'] == 1].sort_values('timestamp')
    if len(piv_rows) == 0:
        return np.array([], dtype=np.int64)
    ts = piv_rows['timestamp'].values.astype(np.int64)
    groups = [[ts[0]]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([ts[i]])
        else:
            groups[-1].append(ts[i])
    return np.array([int(np.median(g)) for g in groups], dtype=np.int64)


def build_phase_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'leg_age_ratio' (continuous 0-1) and 'leg_phase' (categorical).
    Bars before first pivot or after last pivot get phase='UNKNOWN' and
    are excluded from training.
    """
    df = df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    ratio = np.full(len(df), np.nan)
    phase = np.full(len(df), 'UNKNOWN', dtype=object)
    for day, g in df.groupby('day'):
        pivots = pivot_centroid_ts(g)
        if len(pivots) < 2:
            continue
        bar_ts = g['timestamp'].values.astype(np.int64)
        idx_offset = g.index.values
        for i in range(len(bar_ts)):
            ts = bar_ts[i]
            # Find first pivot strictly AFTER ts; last pivot at or BEFORE
            k = np.searchsorted(pivots, ts, side='right')
            if k == 0 or k >= len(pivots):
                continue   # before first or after last pivot — UNKNOWN
            last_p = int(pivots[k-1])
            next_p = int(pivots[k])
            leg_age = ts - last_p
            leg_duration = next_p - last_p
            if leg_duration <= 0:
                continue
            r = leg_age / leg_duration
            ratio[idx_offset[i]] = r
            if r < 0.25:
                phase[idx_offset[i]] = 'EARLY'
            elif r < 0.75:
                phase[idx_offset[i]] = 'MID'
            else:
                phase[idx_offset[i]] = 'LATE'
    df['leg_age_ratio'] = ratio
    df['leg_phase'] = phase
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b5_leg_phase.pkl')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b5_leg_phase.txt')
    ap.add_argument('--out-cache',
                    default='reports/findings/regret_oracle/b5_leg_phase_OOS_NT8.parquet')
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading IS:', args.is_dataset)
    is_df = pd.read_parquet(args.is_dataset)
    print(f'  {len(is_df)} bars / {is_df["day"].nunique()} days')
    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    print('Building leg-phase labels (IS)...')
    is_df = build_phase_labels(is_df)
    print('Building leg-phase labels (OOS)...')
    oos_df = build_phase_labels(oos_df)

    # Phase distribution
    print('\nPhase distribution:')
    for phase in PHASES + ['UNKNOWN']:
        n_is  = int((is_df['leg_phase']  == phase).sum())
        n_oos = int((oos_df['leg_phase'] == phase).sum())
        print(f'  {phase:<10}  IS={n_is:>7,} ({n_is/len(is_df)*100:.1f}%)   '
              f'OOS={n_oos:>7,} ({n_oos/len(oos_df)*100:.1f}%)')

    # Filter to bars with valid phase
    is_train  = is_df[is_df['leg_phase'].isin(PHASES)].copy()
    oos_eval  = oos_df[oos_df['leg_phase'].isin(PHASES)].copy()
    print(f'\n  IS train rows:  {len(is_train):,}')
    print(f'  OOS eval rows: {len(oos_eval):,}')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    X_is  = is_train[v2_cols].fillna(0.0).values.astype(np.float32)
    X_oos = oos_eval[v2_cols].fillna(0.0).values.astype(np.float32)

    le = LabelEncoder()
    le.fit(PHASES)
    y_is  = le.transform(is_train['leg_phase'].values)
    y_oos = le.transform(oos_eval['leg_phase'].values)
    classes = list(le.classes_)
    print(f'Classes (encoded order): {classes}')

    print('\nTraining multi-class HistGradientBoosting...')
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.05,
        max_depth=6, min_samples_leaf=50,
        l2_regularization=0.5,
        class_weight='balanced',
        random_state=args.seed,
    )
    model.fit(X_is, y_is)

    print('\nIS scores:')
    y_pred_is  = model.predict(X_is)
    p_is = model.predict_proba(X_is)
    print(classification_report(y_is, y_pred_is, target_names=classes, digits=3))

    print('\nOOS scores:')
    y_pred_oos = model.predict(X_oos)
    p_oos = model.predict_proba(X_oos)
    print(classification_report(y_oos, y_pred_oos, target_names=classes, digits=3))

    cm = confusion_matrix(y_oos, y_pred_oos)
    print('\nOOS Confusion matrix:')
    print(f'{"":>8}  ' + '  '.join(f'{c:>7}' for c in classes))
    for i, true_label in enumerate(classes):
        print(f'{true_label:>8}  ' + '  '.join(f'{cm[i][j]:>7}' for j in range(len(classes))))

    # Save model + per-bar predictions
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B5 LEG-PHASE CLASSIFIER (3-class: EARLY / MID / LATE)')
    out('=' * 78)
    out(f'IS train: {len(is_train):,}   OOS eval: {len(oos_eval):,}   features: {len(v2_cols)}')
    out('')
    out('OOS classification report:')
    out(classification_report(y_oos, y_pred_oos, target_names=classes, digits=3))
    out('')
    out('OOS Confusion matrix:')
    out(f'{"":>8}  ' + '  '.join(f'{c:>7}' for c in classes))
    for i, true_label in enumerate(classes):
        out(f'{true_label:>8}  ' + '  '.join(f'{cm[i][j]:>7}' for j in range(len(classes))))

    # Per-phase precision when model says X with confidence > 0.50
    out('')
    out('--- Per-class precision at confidence thresholds ---')
    for ci, cls in enumerate(classes):
        out(f'\n  Predicted = {cls}:')
        for thr in [0.40, 0.50, 0.60, 0.70]:
            mask = (p_oos[:, ci] >= thr)
            n = int(mask.sum())
            if n == 0:
                out(f'    thr={thr:.2f}:  no predictions')
                continue
            correct = int((y_oos[mask] == ci).sum())
            prec = correct / n
            cov = n / len(p_oos)
            # Base rate for this class
            base = (y_oos == ci).mean()
            lift = prec / max(base, 1e-9)
            out(f'    thr={thr:.2f}:  n={n:>6}  prec={prec*100:5.1f}%  '
                f'cov={cov*100:5.2f}%  lift={lift:.2f}x')

    # === Build per-bar cache for inspector + cloud composite ===
    print('\nBuilding per-bar OOS prediction cache...')
    # Re-predict on FULL OOS (including UNKNOWN bars) for the cache
    X_oos_full = oos_df[v2_cols].fillna(0.0).values.astype(np.float32)
    p_oos_full = model.predict_proba(X_oos_full)
    cache = oos_df[['timestamp', 'day']].copy()
    for ci, cls in enumerate(classes):
        cache[f'p_phase_{cls}'] = p_oos_full[:, ci]
    cache['p_phase_argmax'] = pd.Categorical(
        [classes[i] for i in p_oos_full.argmax(axis=1)],
        categories=classes,
    )
    cache['leg_phase_truth'] = oos_df['leg_phase'].values
    cache['leg_age_ratio_truth'] = oos_df['leg_age_ratio'].values
    cache.to_parquet(args.out_cache, index=False)

    # Save model
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump({'model': model, 'v2_cols': v2_cols, 'classes': classes,
                     'label_encoder': le}, f)

    out('')
    out('--- Cache columns (for composite) ---')
    out('  p_phase_EARLY, p_phase_MID, p_phase_LATE  (per-bar probabilities)')
    out('  p_phase_argmax                            (top predicted phase)')
    out('  leg_phase_truth                           (zigzag-derived truth)')
    out('  leg_age_ratio_truth                       (0.0-1.0 continuous)')

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_cache}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
