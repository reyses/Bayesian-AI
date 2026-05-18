"""Analyze the cusp-picks primitives CSV:
  1. Cluster the UNCLASSIFIED picks (those not matching v6/compression/structural)
     to discover what feature combinations they share.
  2. Winner-vs-loser feature comparison (mean + delta + effect size).
  3. Classifier: predict winner from features, report importances.

Reads: reports/findings/cusp_picks_primitives.csv
Writes: reports/findings/picks_analysis_*.csv  + console summary
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CSV_PATH = 'reports/findings/cusp_picks_primitives.csv'
OUT_DIR = 'reports/findings'


FEATURE_COLS = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low', 'z_1h_close',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m', 'slope_15m_decel', 'curv_15m',
    'band_width', 'band_rank_60', 'sigma_15m_rank_60',
]


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Coerce numeric, fill NaN where appropriate
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def section(label):
    print('\n' + '═' * 78)
    print(' ' + label)
    print('═' * 78)


# ── (1) Cluster the unclassified picks ──────────────────────────────────────

def cluster_unclassified(df: pd.DataFrame) -> pd.DataFrame:
    section('(1)  CLUSTERING UNCLASSIFIED PICKS')
    classified_cols = ['v6_fires_short', 'v6_fires_long',
                            'compression_short', 'compression_long',
                            'structural_short', 'structural_long']
    df['any_classification'] = df[classified_cols].sum(axis=1) > 0
    classified = df[df['any_classification']]
    unclassified = df[~df['any_classification']]
    print(f'Classified   (any rule fires): {len(classified)}/{len(df)}')
    print(f'Unclassified (no rule fires):  {len(unclassified)}/{len(df)}  ← cluster these')

    if len(unclassified) < 6:
        print('Too few unclassified for clustering.')
        return df

    # Cluster with KMeans on z-scored features
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = unclassified[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    K = min(4, max(2, len(unclassified) // 6))
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs)
    unclassified = unclassified.copy()
    unclassified['cluster'] = km.labels_

    # Report per-cluster mean features + outcomes
    print(f'\n{K}-cluster KMeans on {len(unclassified)} unclassified picks:')
    for k in range(K):
        sub = unclassified[unclassified['cluster'] == k]
        n_L = (sub['direction'] == 'LONG').sum()
        n_S = (sub['direction'] == 'SHORT').sum()
        print(f'\nCluster {k}: n={len(sub)} (L:{n_L} S:{n_S})  '
                  f'mean_MFE ${sub["mfe_dollars"].mean():.0f}  '
                  f'win_rate {100*sub["is_winner"].mean():.0f}%')
        # Top 5 distinguishing features (largest |mean| vs full population)
        full_mean = df[FEATURE_COLS].mean()
        full_std = df[FEATURE_COLS].std().replace(0, 1)
        cluster_mean = sub[FEATURE_COLS].mean()
        z_diff = (cluster_mean - full_mean) / full_std
        top = z_diff.abs().sort_values(ascending=False).head(5)
        for feat in top.index:
            print(f'    {feat:<22} cluster_mean {cluster_mean[feat]:+.2f}  '
                      f'full_mean {full_mean[feat]:+.2f}  '
                      f'z_diff {z_diff[feat]:+.2f}σ')

    # Save with cluster assignments
    out = os.path.join(OUT_DIR, 'picks_clusters_unclassified.csv')
    unclassified.to_csv(out, index=False)
    print(f'\nSaved unclassified clusters to {out}')
    return unclassified


# ── (2) Winner vs Loser feature comparison ──────────────────────────────────

def winner_vs_loser(df: pd.DataFrame):
    section('(2)  WINNER vs LOSER FEATURE COMPARISON')
    winners = df[df['is_winner'] == 1]
    losers = df[df['is_winner'] == 0]
    print(f'Winners (MFE >= 2*MAE): n={len(winners)}  '
              f'mean MFE ${winners["mfe_dollars"].mean():.0f}')
    print(f'Losers:                  n={len(losers)}  '
              f'mean MFE ${losers["mfe_dollars"].mean():.0f}  '
              f'mean MAE ${losers["mae_dollars"].mean():.0f}')

    if len(losers) < 3:
        print(f'\n⚠ Only {len(losers)} losers — winner-vs-loser comparison is '
                  f'underpowered. Showing direction-direction-level stats instead.')

    # Per-feature winner vs loser stats
    rows = []
    for feat in FEATURE_COLS:
        w = winners[feat].dropna()
        l = losers[feat].dropna()
        if len(w) < 2 or len(l) < 2:
            continue
        # Cohen's d
        pooled = np.sqrt(((len(w) - 1) * w.var() + (len(l) - 1) * l.var())
                              / max(1, len(w) + len(l) - 2))
        cohens_d = (w.mean() - l.mean()) / pooled if pooled > 0 else 0
        rows.append({
            'feature': feat,
            'winner_mean': round(w.mean(), 3),
            'loser_mean': round(l.mean(), 3),
            'delta': round(w.mean() - l.mean(), 3),
            'cohens_d': round(cohens_d, 3),
            'winner_n': len(w), 'loser_n': len(l),
        })
    res = pd.DataFrame(rows).sort_values('cohens_d', key=lambda s: s.abs(), ascending=False)
    print('\nTop discriminating features (sorted by |Cohen\'s d|):')
    print(res.to_string(index=False))
    out = os.path.join(OUT_DIR, 'picks_winner_vs_loser.csv')
    res.to_csv(out, index=False)
    print(f'\nSaved comparison to {out}')


# ── (3) Classifier — winner vs loser ────────────────────────────────────────

def train_classifier(df: pd.DataFrame):
    section('(3)  CLASSIFIER: winner vs loser')
    if df['is_winner'].nunique() < 2:
        print('All picks same class — can\'t train classifier.')
        return
    n_win = (df['is_winner'] == 1).sum()
    n_los = (df['is_winner'] == 0).sum()
    if min(n_win, n_los) < 5:
        print(f'Class imbalance too extreme: {n_win}W / {n_los}L. '
                  f'Skipping classifier (would overfit).')
        print('Recommendation: collect more LOSING picks (intentionally mark "I was wrong" '
                  'entries) to build a balanced training set.')
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X = df[FEATURE_COLS].fillna(0).values
    y = df['is_winner'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(Xs, y)
    # Cross-validated accuracy
    try:
        cv = cross_val_score(clf, Xs, y, cv=min(5, n_los), scoring='roc_auc')
        print(f'5-fold CV ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}  '
                  f'(n={len(df)}, {n_win}W/{n_los}L)')
    except Exception as e:
        print(f'CV failed: {e}')

    # Feature importances (signed coefs)
    importances = pd.DataFrame({
        'feature': FEATURE_COLS,
        'coef': clf.coef_[0],
        'abs_coef': np.abs(clf.coef_[0]),
    }).sort_values('abs_coef', ascending=False)
    print('\nLogistic-regression coefficients (positive ⇒ predicts winner):')
    print(importances.head(10).to_string(index=False))

    out = os.path.join(OUT_DIR, 'picks_classifier_importance.csv')
    importances.to_csv(out, index=False)
    print(f'\nSaved feature importances to {out}')


def main():
    df = load_df()
    print(f'Loaded {len(df)} picks from {CSV_PATH}')
    print(f'  {(df["direction"]=="LONG").sum()} LONG / '
              f'{(df["direction"]=="SHORT").sum()} SHORT')
    print(f'  {df["is_winner"].sum()} winners (MFE>=2*MAE) / '
              f'{(df["is_winner"]==0).sum()} losers')

    cluster_unclassified(df)
    winner_vs_loser(df)
    train_classifier(df)


if __name__ == '__main__':
    main()
