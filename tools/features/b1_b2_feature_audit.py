"""Feature importance + label-leakage audit for B1 and B2 classifiers.

Per CLAUDE.md / 2026-05-17 findings: B1+B2 show real ML edge but IS-OOS AUC
gap is moderate-to-large. Two things to check:

1. Top-N features by gain importance — what is the GBM actually using?
2. Label-leakage audit — for forward-looking targets (B1: pivot in next K
   min, B2: leg dies within K min), any feature whose lookback WINDOW could
   include data from the future is suspect.

V2 feature naming convention (per training_iso_v2/v2_cols.py):
  L1_<TF>_<concept>_1b   -> 1-bar (current) signal at TF
  L2_<TF>_<concept>_<N>  -> N-bar rolling window at TF (lookback only)
  L3_<TF>_<concept>_<N>  -> derived from L2 rolling stats (lookback only)

All V2 windows are BACKWARD-LOOKING — they should not leak forward info.
But: if a 4h window is computed on bar T, it sees data from T-4h to T.
That's fine for prediction at T, no leakage. The audit confirms naming
and identifies any anomaly.

Output: per-classifier top-30 features ranked by gain, with TF and window
size annotated. Flag any feature whose lookback could plausibly include
the target horizon.
"""
from __future__ import annotations
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_b1_pivot_imminent import build_labels as build_b1_labels
from train_b1_pivot_imminent import K_MINUTES as B1_K
from train_b2_fakeout import build_pivot_dataset
from train_b2_fakeout import K_MINUTES as B2_K


def parse_v2_name(name: str):
    """Decompose a V2 feature name into (layer, tf, concept, window).

    Examples:
      L1_5s_price_velocity_1b      -> ('L1', '5s', 'price_velocity', '1b')
      L2_15m_price_mean_12         -> ('L2', '15m', 'price_mean', '12')
      L3_1h_reversion_prob_12      -> ('L3', '1h', 'reversion_prob', '12')
    Returns dict or None if not matched.
    """
    m = re.match(r'^(L[123])_([^_]+)_(.+)_([0-9]+b?)$', name)
    if not m:
        return None
    layer, tf, concept, window = m.groups()
    return {'layer': layer, 'tf': tf, 'concept': concept, 'window': window}


def tf_seconds(tf: str) -> int:
    """Convert TF string to seconds."""
    units = {'s': 1, 'm': 60, 'h': 3600, 'D': 86400}
    m = re.match(r'^([0-9]+)([smhD])$', tf)
    if not m:
        return -1
    return int(m.group(1)) * units[m.group(2)]


def window_seconds(window_str: str, tf_str: str) -> int:
    """Approximate lookback window in seconds. 1b = single bar = 1×TF."""
    if window_str.endswith('b'):
        return tf_seconds(tf_str)   # 1b = one TF bar
    n = int(window_str)
    return n * tf_seconds(tf_str)


def compute_permutation_importance(model, X, y, n_repeats=3, sample=20000, seed=42):
    """Permutation importance — shuffles each feature, measures AUC drop.
    For speed, we subsample to `sample` rows (stratified by y when possible)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > sample:
        # Stratified subsample: take all positives + random negatives
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = min(len(pos), sample // 2)
        n_neg = sample - n_pos
        pos_sel = rng.choice(pos, n_pos, replace=False) if len(pos) >= n_pos else pos
        neg_sel = rng.choice(neg, n_neg, replace=False) if len(neg) >= n_neg else neg
        sel = np.concatenate([pos_sel, neg_sel])
        X_s = X[sel]; y_s = y[sel]
    else:
        X_s = X; y_s = y
    print(f'    permutation_importance on {len(X_s)} samples × {X.shape[1]} feats × {n_repeats} repeats...')
    result = permutation_importance(
        model, X_s, y_s, scoring='roc_auc',
        n_repeats=n_repeats, n_jobs=-1, random_state=seed,
    )
    return result.importances_mean


def audit_classifier(name: str, models, K_list,
                      target_horizons_min, n_top=30,
                      data_builder=None, oos_df=None):
    """Compute permutation importance per K, print top-N with TF/window info."""
    lines = []
    def out(s=''):
        lines.append(s)
    out('=' * 78)
    out(f'{name} -- top-{n_top} features by PERMUTATION IMPORTANCE on OOS')
    out('  (higher = larger AUC drop when this feature is shuffled)')
    out('=' * 78)
    feature_summary = {}
    for K in sorted(K_list):
        bundle = models[K]
        model = bundle['model']
        cols = bundle['v2_cols']
        target_s = K * 60

        # Build OOS labels + features for this K
        X, y = data_builder(oos_df, K, cols)
        print(f'  Computing permutation importance for K={K}...')
        imp = compute_permutation_importance(model, X, y)
        order = np.argsort(imp)[::-1]

        out(f'\n--- K={K} min (target horizon = {target_s}s) ---')
        out(f'  {"rank":>4}  {"imp":>10}  {"feature":<38}  '
            f'{"TF":>5}  {"win_s":>6}  {"ratio":>10}')
        for rk, idx in enumerate(order[:n_top]):
            feat = cols[idx]
            score = imp[idx]
            p = parse_v2_name(feat)
            if p is None:
                out(f'  {rk+1:>4}  {score:>10.4f}  {feat:<38}  ?      ?       ?')
                continue
            tf = p['tf']
            win_s = window_seconds(p['window'], tf)
            ratio = win_s / target_s if target_s > 0 else float('inf')
            if win_s > target_s * 4:
                tag = '(wide)'
            elif win_s < target_s / 4:
                tag = '(narrow)'
            else:
                tag = '(matched)'
            out(f'  {rk+1:>4}  {score:>10.4f}  {feat:<38}  {tf:>5}  '
                f'{win_s:>6}  {ratio:>6.2f}x {tag}')
            feature_summary.setdefault(feat, {'count': 0, 'sum_imp': 0.0})
            feature_summary[feat]['count'] += 1
            feature_summary[feat]['sum_imp'] += float(score)

    out('\n--- Cross-K aggregation: features ranked high across multiple Ks ---')
    agg = sorted(feature_summary.items(), key=lambda x: -x[1]['sum_imp'])[:25]
    out(f'  {"feature":<40}  {"#K":>5}  {"sum_imp":>10}')
    for feat, info in agg:
        out(f'  {feat:<40}  {info["count"]:>5}  {info["sum_imp"]:>10.4f}')

    return lines


def label_leakage_check(name: str, models, K_list):
    """Look for explicit leakage: any feature whose name suggests it
    contains future data."""
    lines = []
    def out(s=''):
        lines.append(s)
    out('\n--- LEAKAGE NAME-PATTERN AUDIT ---')
    suspicious_patterns = [
        r'forward', r'future', r'next', r'after', r'_fwd_', r'_post_',
    ]
    cols = list(models[K_list[0]]['v2_cols'])
    suspicious_hits = []
    for c in cols:
        for pat in suspicious_patterns:
            if re.search(pat, c, re.IGNORECASE):
                suspicious_hits.append((c, pat))
                break
    if suspicious_hits:
        out('  WARNING: features with potentially leaky names:')
        for c, pat in suspicious_hits:
            out(f'    {c}  (matched pattern: {pat})')
    else:
        out('  No features with name patterns suggesting forward-looking data.')
        out('  V2 features are constructed with backward rolling windows by design.')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1-pkl', default='reports/findings/regret_oracle/b1_pivot_imminent.pkl')
    ap.add_argument('--b2-pkl', default='reports/findings/regret_oracle/b2_fakeout.pkl')
    ap.add_argument('--out', default='reports/findings/regret_oracle/b1_b2_feature_audit.txt')
    args = ap.parse_args()

    with open(args.b1_pkl, 'rb') as f:
        b1 = pickle.load(f)
    with open(args.b2_pkl, 'rb') as f:
        b2 = pickle.load(f)

    # Load OOS data once
    print('Loading OOS dataset...')
    oos_df = pd.read_parquet('reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    # Build labels once (cached for all K) for B1
    print('Building B1 labels on OOS...')
    oos_b1 = build_b1_labels(oos_df.copy(), B1_K)
    # Build pivot events for B2
    print('Building B2 pivot events on OOS...')
    oos_piv, _ = build_pivot_dataset(oos_df, B2_K)
    print(f'  {len(oos_piv)} pivot events')

    def b1_data_builder(_oos_df, K, cols):
        X = oos_b1[cols].fillna(0.0).values.astype(np.float32)
        y = oos_b1[f'pivot_within_{K}m'].values
        return X, y

    def b2_data_builder(_oos_df, K, cols):
        X = oos_piv[cols].values.astype(np.float32)
        y = oos_piv[f'is_fakeout_{K}m'].values
        return X, y

    all_lines = []
    lines = audit_classifier('B1 PIVOT-IMMINENT', b1, sorted(b1.keys()),
                              target_horizons_min={1:60, 3:180, 5:300, 10:600},
                              data_builder=b1_data_builder, oos_df=oos_df)
    all_lines.extend(lines)
    leak = label_leakage_check('B1', b1, sorted(b1.keys()))
    all_lines.extend(leak)

    all_lines.append('')
    lines = audit_classifier('B2 FAKEOUT', b2, sorted(b2.keys()),
                              target_horizons_min={3:180, 5:300, 10:600},
                              data_builder=b2_data_builder, oos_df=oos_df)
    all_lines.extend(lines)
    leak = label_leakage_check('B2', b2, sorted(b2.keys()))
    all_lines.extend(leak)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(all_lines), encoding='utf-8')
    for l in all_lines:
        print(l)
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()
