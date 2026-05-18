"""Density-based oracle surfaces — KDE + GMM in feature space.

Per user 2026-05-12: the LR uses a single hyperplane; we want the REGION
around each oracle point. Two density estimators:

  KDE  — non-parametric; each oracle bar contributes a Gaussian blob;
         densities sum naturally for multi-modal structure.

  GMM  — K Gaussian mixture components; each component captures one
         "trade type" region (e.g. compression-cusp / extension-fade).

DECISION RULE:
    P_oracle(state) = density under oracle distribution
    P_random(state) = density under random distribution
    Lift(state)     = P_oracle / P_random
    Fire if Lift >= threshold

Reuses training data from logistic_oracle_surfaces.py.

OUTPUTS:
    reports/findings/density_oracles/
        kde_models.pkl                serialized KDE models per surface
        gmm_models.pkl                serialized GMM models per surface
        density_ratio_distributions.csv   density at oracle vs random bars
        density_aucs.txt              KDE-LR vs GMM-LR vs Logistic AUC comparison
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


OUT_DIR = Path('reports/findings/density_oracles')
TRAIN_DATA = Path('reports/findings/logistic_oracles/oracle_train_data.csv')

NUMERIC_FEATURES = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
]


def load_labeled(path: Path = TRAIN_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Run tools/logistic_oracle_surfaces.py first to build {path}')
    df = pd.read_csv(path)
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df


def fit_density_surfaces(df: pd.DataFrame, label_col: str, name: str,
                                  feature_cols: list, kde_bw: float = 0.5,
                                  gmm_k: int = 4) -> dict:
    """Fit KDE + GMM on oracle bars (label_col==1); compare against random bars."""
    from sklearn.neighbors import KernelDensity
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    pos = df[df[label_col] == 1]
    neg = df[df['is_random'] == 1]
    if len(pos) < 30 or len(neg) < 30:
        print(f'  [{name}] insufficient samples: pos={len(pos)} neg={len(neg)}')
        return None

    full = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    X = full[feature_cols].astype(float).fillna(0).values
    y = full['y'].values

    # Standardize on full dataset (so the bandwidths are interpretable)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42,
                                                            stratify=y)
    # Separate oracle vs random in training set
    X_tr_oracle = X_tr[y_tr == 1]
    X_tr_random = X_tr[y_tr == 0]

    # ── KDE ─────────────────────────────────────────────────────────
    kde_o = KernelDensity(kernel='gaussian', bandwidth=kde_bw).fit(X_tr_oracle)
    kde_r = KernelDensity(kernel='gaussian', bandwidth=kde_bw).fit(X_tr_random)
    # Log-density ratio = score(oracle) - score(random)
    kde_lr_tr = kde_o.score_samples(X_tr) - kde_r.score_samples(X_tr)
    kde_lr_te = kde_o.score_samples(X_te) - kde_r.score_samples(X_te)
    kde_auc_tr = roc_auc_score(y_tr, kde_lr_tr)
    kde_auc_te = roc_auc_score(y_te, kde_lr_te)

    # ── GMM ─────────────────────────────────────────────────────────
    gmm_o = GaussianMixture(n_components=gmm_k, random_state=42).fit(X_tr_oracle)
    gmm_r = GaussianMixture(n_components=gmm_k, random_state=42).fit(X_tr_random)
    gmm_lr_tr = gmm_o.score_samples(X_tr) - gmm_r.score_samples(X_tr)
    gmm_lr_te = gmm_o.score_samples(X_te) - gmm_r.score_samples(X_te)
    gmm_auc_tr = roc_auc_score(y_tr, gmm_lr_tr)
    gmm_auc_te = roc_auc_score(y_te, gmm_lr_te)

    # GMM component means (rescaled back to original feature space)
    # Centers reveal the "trade-type" regions
    gmm_oracle_centers = scaler.inverse_transform(gmm_o.means_)
    gmm_oracle_weights = gmm_o.weights_

    return {
        'name': name,
        'n_pos': len(pos), 'n_neg': len(neg),
        'kde_auc_train': kde_auc_tr, 'kde_auc_test': kde_auc_te,
        'gmm_auc_train': gmm_auc_tr, 'gmm_auc_test': gmm_auc_te,
        'kde_oracle': kde_o, 'kde_random': kde_r,
        'gmm_oracle': gmm_o, 'gmm_random': gmm_r,
        'scaler': scaler,
        'gmm_oracle_centers': gmm_oracle_centers,
        'gmm_oracle_weights': gmm_oracle_weights,
        'feature_cols': feature_cols,
    }


def write_summary(models: list, path: Path, kde_bw: float, gmm_k: int):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Density-based Oracle Surfaces\n')
        f.write('=' * 60 + '\n')
        f.write(f'KDE bandwidth: {kde_bw}\n')
        f.write(f'GMM components: {gmm_k}\n\n')
        f.write(f'{"surface":<14} {"n_pos":>6} {"n_neg":>6} '
                  f'{"KDE_AUC_tr":>10} {"KDE_AUC_te":>10} '
                  f'{"GMM_AUC_tr":>10} {"GMM_AUC_te":>10}\n')
        f.write('-' * 80 + '\n')
        for m in models:
            if m is None:
                continue
            f.write(f'{m["name"]:<14} '
                      f'{m["n_pos"]:>6} {m["n_neg"]:>6} '
                      f'{m["kde_auc_train"]:>10.3f} {m["kde_auc_test"]:>10.3f} '
                      f'{m["gmm_auc_train"]:>10.3f} {m["gmm_auc_test"]:>10.3f}\n')
        # GMM component centers (the "trade-type" regions)
        f.write('\n\nGMM ORACLE COMPONENT CENTERS (in original feature units):\n')
        f.write('=' * 60 + '\n')
        for m in models:
            if m is None:
                continue
            f.write(f'\n{m["name"]}:\n')
            for k_i in range(len(m['gmm_oracle_weights'])):
                w = m['gmm_oracle_weights'][k_i]
                c = m['gmm_oracle_centers'][k_i]
                f.write(f'  Component {k_i} (weight={w:.2f}):\n')
                # Show top 6 features by absolute centroid value
                fc = m['feature_cols']
                pairs = list(zip(fc, c))
                pairs.sort(key=lambda p: abs(p[1]), reverse=True)
                for feat, val in pairs[:6]:
                    f.write(f'      {feat:<22} {val:+.3f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kde-bandwidth', type=float, default=0.5)
    ap.add_argument('--gmm-components', type=int, default=4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading labeled training data...')
    df = load_labeled()
    print(f'  rows: {len(df)}')

    surfaces = [
        ('is_entry_short', 'entry_short'),
        ('is_entry_long',  'entry_long'),
        ('is_exit_short',  'exit_short'),
        ('is_exit_long',   'exit_long'),
    ]
    print(f'\nFitting density surfaces (KDE bw={args.kde_bandwidth}, GMM k={args.gmm_components})...')
    models = []
    for label_col, name in surfaces:
        print(f'  {name}...')
        m = fit_density_surfaces(df, label_col, name, NUMERIC_FEATURES,
                                              kde_bw=args.kde_bandwidth,
                                              gmm_k=args.gmm_components)
        models.append(m)

    # Pickle the models (for inference reuse)
    pickleable = []
    for m in models:
        if m is None:
            pickleable.append(None)
            continue
        pickleable.append({
            'name': m['name'],
            'kde_oracle': m['kde_oracle'], 'kde_random': m['kde_random'],
            'gmm_oracle': m['gmm_oracle'], 'gmm_random': m['gmm_random'],
            'scaler': m['scaler'],
            'feature_cols': m['feature_cols'],
        })
    with open(OUT_DIR / 'density_models.pkl', 'wb') as f:
        pickle.dump(pickleable, f)

    summary_path = OUT_DIR / 'density_aucs.txt'
    write_summary(models, summary_path, args.kde_bandwidth, args.gmm_components)
    print(f'\nWrote summary: {summary_path}')

    # Console summary
    print('\n=== AUC COMPARISON: KDE vs GMM ===')
    print(f'{"surface":<14} {"KDE_test":>9} {"GMM_test":>9}')
    print('-' * 36)
    for m in models:
        if m is None:
            continue
        print(f'{m["name"]:<14} {m["kde_auc_test"]:>9.3f} {m["gmm_auc_test"]:>9.3f}')

    print('\n=== GMM ORACLE COMPONENT CENTERS (per surface) ===')
    for m in models:
        if m is None:
            continue
        print(f'\n  {m["name"]}:')
        fc = m['feature_cols']
        for k_i in range(len(m['gmm_oracle_weights'])):
            w = m['gmm_oracle_weights'][k_i]
            c = m['gmm_oracle_centers'][k_i]
            pairs = sorted(zip(fc, c), key=lambda p: abs(p[1]), reverse=True)[:5]
            top = '  '.join(f'{f}={v:+.2f}' for f, v in pairs)
            print(f'    C{k_i} (w={w:.2f}): {top}')


if __name__ == '__main__':
    main()
