"""B6 — Directional pivot classifier.

Per user 2026-05-17: existing B1/B2/B4/B5 are direction-blind ("pivot
imminent" without telling us if it'll flip UP or DOWN). For trade
management we need to know WHICH direction the next leg will go so we
can pre-place reverse orders / tighten the right side.

Label (3-class per K):
  NO_PIVOT          : no zigzag pivot in next K min
  PIVOT_TO_LONG     : next pivot is a LOW pivot (kicks off LONG leg)
  PIVOT_TO_SHORT    : next pivot is a HIGH pivot (kicks off SHORT leg)

Convention from build_zigzag_pivot_dataset.py:
  pivot_dir is the direction of the LEG STARTING at this pivot.
  So pivot_dir='LONG'  -> pivot is a LOW  -> next leg goes UP
     pivot_dir='SHORT' -> pivot is a HIGH -> next leg goes DOWN

Trains 3-class GBM per K in {1, 3, 5, 10}.

Trade-management value:
  - High P_PIVOT_TO_SHORT -> we're approaching a HIGH pivot -> tighten LONG trail
  - High P_PIVOT_TO_LONG  -> we're approaching a LOW pivot -> tighten SHORT trail
  - Pair P with direction to choose which side to defend.
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


K_MINUTES = [1, 3, 5, 10]
CLASSES = ['NO_PIVOT', 'PIVOT_TO_LONG', 'PIVOT_TO_SHORT']


def pivot_events_per_day(day_df: pd.DataFrame):
    """Return list of (centroid_ts, direction) per pivot event.

    direction is the direction of the LEG STARTING at this pivot:
      'LONG'  -> LOW pivot (leg going up)
      'SHORT' -> HIGH pivot (leg going down)
    """
    piv = day_df[day_df['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return []
    ts = piv['timestamp'].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for grp in groups:
        sub_ts = ts[grp]; sub_dir = pd_[grp]
        ts_c = int(np.median(sub_ts))
        vals, counts = np.unique(sub_dir, return_counts=True)
        d = str(vals[np.argmax(counts)])
        out.append((ts_c, d))
    return out


def build_directional_labels(df: pd.DataFrame, K_list):
    """Add 'next_pivot_dir_Km' string columns per K — 'LONG' / 'SHORT' / 'NONE'."""
    df = df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    out_cols = {f'next_pivot_dir_{K}m': np.full(len(df), 'NONE', dtype=object)
                for K in K_list}
    for day, g in df.groupby('day'):
        events = pivot_events_per_day(g)
        if len(events) == 0:
            continue
        ev_ts = np.array([e[0] for e in events], dtype=np.int64)
        ev_dir = [e[1] for e in events]
        bar_ts = g['timestamp'].values.astype(np.int64)
        for K in K_list:
            window_s = K * 60
            pos = np.searchsorted(ev_ts, bar_ts, side='right')
            for i, k in enumerate(pos):
                if k >= len(ev_ts):
                    continue
                if ev_ts[k] - bar_ts[i] <= window_s:
                    # Next pivot is in window — record its direction
                    out_cols[f'next_pivot_dir_{K}m'][g.index[i]] = ev_dir[k]
    for k, v in out_cols.items():
        df[k] = v
    return df


def encode_3class(direction_arr):
    """LONG -> PIVOT_TO_LONG, SHORT -> PIVOT_TO_SHORT, NONE -> NO_PIVOT."""
    out = np.full(len(direction_arr), 'NO_PIVOT', dtype=object)
    out[direction_arr == 'LONG']  = 'PIVOT_TO_LONG'
    out[direction_arr == 'SHORT'] = 'PIVOT_TO_SHORT'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b6_directional_pivot.pkl')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b6_directional_pivot.txt')
    ap.add_argument('--out-cache',
                    default='reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading IS:', args.is_dataset)
    is_df = pd.read_parquet(args.is_dataset)
    print(f'  {len(is_df)} bars / {is_df["day"].nunique()} days')
    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    print('Building directional labels (IS)...')
    is_df = build_directional_labels(is_df, K_MINUTES)
    print('Building directional labels (OOS)...')
    oos_df = build_directional_labels(oos_df, K_MINUTES)

    # Class distribution
    print('\nClass distribution by K:')
    for K in K_MINUTES:
        col = f'next_pivot_dir_{K}m'
        c_is = encode_3class(is_df[col].values)
        c_oos = encode_3class(oos_df[col].values)
        print(f'  K={K}:')
        for c in CLASSES:
            n_is = int((c_is == c).sum())
            n_oos = int((c_oos == c).sum())
            print(f'    {c:<18}: IS={n_is:>7,} ({n_is/len(is_df)*100:.1f}%)   '
                  f'OOS={n_oos:>6,} ({n_oos/len(oos_df)*100:.1f}%)')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'\nV2 features: {len(v2_cols)}')

    X_is  = is_df[v2_cols].fillna(0.0).values.astype(np.float32)
    X_oos = oos_df[v2_cols].fillna(0.0).values.astype(np.float32)

    le = LabelEncoder()
    le.fit(CLASSES)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B6 DIRECTIONAL PIVOT CLASSIFIER (3-class per K)')
    out('  NO_PIVOT / PIVOT_TO_LONG (LOW pivot) / PIVOT_TO_SHORT (HIGH pivot)')
    out('=' * 78)
    out(f'IS rows: {len(is_df):,}   OOS rows: {len(oos_df):,}   features: {len(v2_cols)}')

    models = {}
    oos_cache = oos_df[['timestamp', 'day']].copy()
    for K in K_MINUTES:
        col = f'next_pivot_dir_{K}m'
        y_is_str  = encode_3class(is_df[col].values)
        y_oos_str = encode_3class(oos_df[col].values)
        y_is  = le.transform(y_is_str)
        y_oos = le.transform(y_oos_str)

        out(f'\n--- K={K} min ---')
        out(f'  IS class counts:  ' + '  '.join(
            f'{c}={int((y_is_str==c).sum())}' for c in CLASSES))
        out(f'  OOS class counts: ' + '  '.join(
            f'{c}={int((y_oos_str==c).sum())}' for c in CLASSES))

        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.05,
            max_depth=6, min_samples_leaf=50,
            l2_regularization=0.5,
            class_weight='balanced',
            random_state=args.seed,
        )
        model.fit(X_is, y_is)

        p_oos = model.predict_proba(X_oos)
        y_pred_oos = model.predict(X_oos)

        # Map probabilities back to class names
        classes_in_order = list(le.classes_)
        out('  OOS classification report:')
        report = classification_report(y_oos, y_pred_oos,
                                        target_names=classes_in_order, digits=3)
        for line in report.split('\n'):
            out('    ' + line)

        cm = confusion_matrix(y_oos, y_pred_oos)
        out(f'\n  OOS Confusion matrix:')
        out('    ' + ' '*16 + '  '.join(f'{c[:9]:>9}' for c in classes_in_order))
        for i, c in enumerate(classes_in_order):
            out(f'    {c[:14]:<14}  ' + '  '.join(f'{cm[i][j]:>9}' for j in range(len(classes_in_order))))

        # Per-class precision at confidence thresholds
        out(f'\n  Per-class precision at conf thresholds (OOS):')
        for ci, cls in enumerate(classes_in_order):
            if cls == 'NO_PIVOT':
                continue   # not actionable
            out(f'    Predicted = {cls}:')
            for thr in [0.40, 0.50, 0.60, 0.70]:
                mask = p_oos[:, ci] >= thr
                n = int(mask.sum())
                if n == 0:
                    out(f'      thr={thr:.2f}:  no predictions')
                    continue
                correct = int((y_oos[mask] == ci).sum())
                prec = correct / n
                cov = n / len(p_oos)
                base = (y_oos == ci).mean()
                lift = prec / max(base, 1e-9)
                out(f'      thr={thr:.2f}:  n={n:>6}  prec={prec*100:5.1f}%  '
                    f'cov={cov*100:5.2f}%  lift={lift:.2f}x')

        models[K] = {'model': model, 'v2_cols': v2_cols,
                      'classes': classes_in_order, 'K': K}

        # Stash per-K probabilities in OOS cache
        for ci, cls in enumerate(classes_in_order):
            oos_cache[f'p_{cls}_{K}m'] = p_oos[:, ci]
        oos_cache[f'next_pivot_dir_truth_{K}m'] = y_oos_str

    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump({'models': models, 'label_encoder': le}, f)
    oos_cache.to_parquet(args.out_cache, index=False)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')

    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_cache}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
