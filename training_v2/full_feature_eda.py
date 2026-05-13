"""Full V2 feature EDA on NMP trades — find what separates good from bad.

Runs THREE analyses across all 185 V2 columns at entry, on all trades:

    1. Spearman correlation of each feature with actual_pnl
       (rank-based; robust to outliers; signed)
    2. Cohen's d between WINNER (pnl > +margin) and LOSER (pnl < -margin)
       (binary effect size)
    3. Quintile breakdown — split feature into 5 buckets, show $/trade and
       WR per bucket (reveals non-linear monotone or U-shape effects)

Walk-forward survival check on top features: 70% train / 30% val IS split;
flag features whose effect SURVIVES with same sign and ≥0.05 Cohen's d.

Output:
    - Console: top-30 ranked + quintile breakdown for top-3
    - Markdown: full 185 ranking saved to reports/findings/
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training_v2.regret import RegretLabel
from training_v2.ledger import ClosedTrade
from training_v2.state import REGIME_VOCAB
from training_v2.tier_discovery import load_joined, cohens_d


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN-aware."""
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 30:
        return 0.0
    rx = pd.Series(x[mask]).rank().values
    ry = pd.Series(y[mask]).rank().values
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def quintile_breakdown(values: np.ndarray, pnl: np.ndarray, n_bins: int = 5
                              ) -> pd.DataFrame:
    """Bin values into quintiles, report $/trade and WR per bin.

    Drops NaN values. Returns DataFrame with columns: bin, n, mean, $/trade, WR%.
    """
    mask = ~np.isnan(values)
    v = values[mask]
    p = pnl[mask]
    if len(v) < n_bins * 10:
        return pd.DataFrame()
    # Quantile-based bin edges
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(v, edges[1:-1])
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() < 5:
            continue
        rows.append({
            'bin': b,
            'lo': float(edges[b]),
            'hi': float(edges[b + 1]),
            'n': int(m.sum()),
            'mean_value': float(v[m].mean()),
            'mean_pnl': float(p[m].mean()),
            'wr_count': float((p[m] > 0).mean()),
        })
    return pd.DataFrame(rows)


def _build_extended_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build (N, 185 + n_regime) feature matrix including regime one-hot."""
    feats = np.stack(df['entry_v2'].values)
    regime_idx_arr = df['regime_idx'].values.astype(np.int32)
    extra_cols = []
    extra_data = []
    for r in range(len(REGIME_VOCAB)):
        col = (regime_idx_arr == r).astype(np.float32)
        if col.sum() < 50:
            continue
        extra_cols.append(f'REGIME={REGIME_VOCAB[r]}')
        extra_data.append(col)
    if extra_data:
        feats = np.concatenate([feats, np.stack(extra_data, axis=1)], axis=1)
    return feats, list(FEATURE_NAMES) + extra_cols


def survival_check_pnl(df: pd.DataFrame, top_features: List[str],
                              train_frac: float = 0.7) -> pd.DataFrame:
    """For each top feature, recompute Spearman corr on train and val splits."""
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * train_frac)
    train = df_sorted.iloc[:cut]
    val = df_sorted.iloc[cut:]

    feat_train, ext_names = _build_extended_feature_matrix(train)
    feat_val, _ = _build_extended_feature_matrix(val)
    pnl_train = train['actual_pnl'].values.astype(np.float64)
    pnl_val = val['actual_pnl'].values.astype(np.float64)

    name_to_idx = {n: i for i, n in enumerate(ext_names)}
    rows = []
    for name in top_features:
        j = name_to_idx.get(name)
        if j is None:
            continue
        rho_train = spearman_corr(feat_train[:, j], pnl_train)
        rho_val = spearman_corr(feat_val[:, j], pnl_val)
        same_sign = (np.sign(rho_train) == np.sign(rho_val)
                            and rho_train != 0 and rho_val != 0)
        survives = bool(same_sign and abs(rho_val) >= 0.03)
        rows.append({
            'feature': name,
            'rho_train': rho_train,
            'rho_val': rho_val,
            'survives': survives,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description='Full V2 feature EDA on NMP trades')
    p.add_argument('--trades', default='training_v2/output/nmp_only.pkl')
    p.add_argument('--regret', default='training_v2/output/regret_nmp.pkl')
    p.add_argument('--out', default='reports/findings/v2_winner_loser_full_eda.md')
    p.add_argument('--margin', type=float, default=5.0,
                       help='$ margin for binary WINNER/LOSER classification')
    p.add_argument('--top-k', type=int, default=30)
    args = p.parse_args()

    print(f'Loading trades + regret...')
    df = load_joined(args.trades, args.regret)
    print(f'  Joined: {len(df)} trades')

    # ── Per-regime stratification: is regime a categorical splitter? ──────
    print(f'\nPer-regime breakdown (regime is constant across the day):')
    print(f'  {"regime":<14} {"n":>6} {"$/day":>8} {"$/trade":>8} {"win-rate":>10} '
              f'{"days":>5}')
    regime_pnl = (df.groupby('regime_idx')['actual_pnl']
                          .agg(['count', 'sum', 'mean']))
    regime_days = df.groupby('regime_idx')['day'].nunique()
    regime_wr = df.groupby('regime_idx')['actual_pnl'].apply(lambda x: (x > 0).mean())
    for r in sorted(df['regime_idx'].unique()):
        rname = REGIME_VOCAB[int(r)] if int(r) < len(REGIME_VOCAB) else f'R{r}'
        n = int(regime_pnl.loc[r, 'count'])
        total = float(regime_pnl.loc[r, 'sum'])
        per_trade = float(regime_pnl.loc[r, 'mean'])
        per_day = total / max(int(regime_days.loc[r]), 1)
        wr = float(regime_wr.loc[r])
        d = int(regime_days.loc[r])
        print(f'  {rname:<14} {n:>6} {per_day:>+7.2f}  {per_trade:>+7.2f}  '
                  f'{wr:>9.1%}  {d:>5}')

    # ── Compute per-feature stats ────────────────────────────────────────
    feat_matrix = np.stack(df['entry_v2'].values)  # (N, 185)
    pnl = df['actual_pnl'].values.astype(np.float64)

    # Append regime one-hot encoding to the feature matrix (and names)
    # so categorical regime can rank alongside continuous V2 features.
    regime_idx_arr = df['regime_idx'].values.astype(np.int32)
    regime_one_hot_cols = []
    regime_one_hot_data = []
    for r in range(len(REGIME_VOCAB)):
        rname = REGIME_VOCAB[r]
        col = (regime_idx_arr == r).astype(np.float32)
        if col.sum() < 50:
            continue
        regime_one_hot_cols.append(f'REGIME={rname}')
        regime_one_hot_data.append(col)
    if regime_one_hot_data:
        regime_one_hot = np.stack(regime_one_hot_data, axis=1)
        feat_matrix = np.concatenate([feat_matrix, regime_one_hot], axis=1)
        all_feature_names = list(FEATURE_NAMES) + regime_one_hot_cols
        print(f'\nAdded {len(regime_one_hot_cols)} regime one-hot features to ranking')
    else:
        all_feature_names = list(FEATURE_NAMES)

    rows = []
    for j, name in enumerate(all_feature_names):
        col = feat_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 100:
            continue
        # Spearman vs PnL
        rho = spearman_corr(col, pnl)
        # Cohen's d WINNER vs LOSER
        winner_mask = pnl > args.margin
        loser_mask = pnl < -args.margin
        d = cohens_d(col[winner_mask], col[loser_mask])
        rows.append({
            'feature': name,
            'spearman_rho': rho,
            'cohens_d': d,
            'abs_rho': abs(rho),
            'abs_d': abs(d),
            'mean_winner': float(np.nanmean(col[winner_mask])) if winner_mask.sum() else 0.0,
            'mean_loser': float(np.nanmean(col[loser_mask])) if loser_mask.sum() else 0.0,
            'n_valid': int(valid.sum()),
        })

    feats = pd.DataFrame(rows)
    feats['rank_score'] = feats['abs_d'] + feats['abs_rho']  # combined score
    feats = feats.sort_values('rank_score', ascending=False).reset_index(drop=True)

    print(f'\nTop {args.top_k} V2 features ranked by combined |Cohen-d| + |Spearman| '
              f'(margin=${args.margin}):')
    print()
    print(f'  {"feature":<32} {"d":>7} {"rho":>7} {"meanWIN":>11} {"meanLOSE":>11}')
    head = feats.head(args.top_k)
    for _, r in head.iterrows():
        print(f'  {r["feature"]:<32} {r["cohens_d"]:>+7.3f} '
                  f'{r["spearman_rho"]:>+7.3f} {r["mean_winner"]:>+10.3f} '
                  f'{r["mean_loser"]:>+10.3f}')

    # ── Walk-forward survival ────────────────────────────────────────────
    print(f'\nWalk-forward survival on top {args.top_k} (sign + |rho_val| >= 0.03):')
    surv = survival_check_pnl(df, head['feature'].tolist())
    n_sur = int(surv['survives'].sum())
    print(f'  Survived: {n_sur}/{args.top_k}')
    print()
    print(f'  {"feature":<32} {"rho_train":>10} {"rho_val":>9} {"OK":>4}')
    for _, r in surv.iterrows():
        ok = 'YES' if r['survives'] else 'no'
        print(f'  {r["feature"]:<32} {r["rho_train"]:>+9.3f} '
                  f'{r["rho_val"]:>+8.3f}   {ok}')

    # ── Quintile breakdown for top-5 features ────────────────────────────
    print(f'\nQuintile breakdown for top 5 features (sorted by |rank_score|):')
    print(f'(reveals monotone vs U-shape; "WR" is count-based fraction of pnl>0)')
    for _, r in feats.head(5).iterrows():
        name = r['feature']
        if name not in all_feature_names:
            continue
        j = all_feature_names.index(name)
        col = feat_matrix[:, j]
        qb = quintile_breakdown(col, pnl)
        if qb.empty:
            continue
        print(f'\n  {name}  (d={r["cohens_d"]:+.3f}, rho={r["spearman_rho"]:+.3f})')
        print(f'    {"bin":>3} {"range":>22} {"n":>5} {"mean_val":>11} '
                  f'{"$/trade":>9} {"WR":>6}')
        for _, qr in qb.iterrows():
            rng = f'[{qr["lo"]:>9.2f}, {qr["hi"]:>9.2f}]'
            print(f'    {int(qr["bin"]):>3} {rng:>22} {int(qr["n"]):>5} '
                      f'{qr["mean_value"]:>+10.2f} {qr["mean_pnl"]:>+8.2f} '
                      f'{qr["wr_count"]:>5.1%}')

    # ── Save markdown ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f'# V2 Winner/Loser Feature EDA on Base NMP\n\n')
        f.write(f'Trades: {len(df)} (margin = ${args.margin})\n')
        f.write(f'Winners: {int((pnl > args.margin).sum())}, '
                    f'Losers: {int((pnl < -args.margin).sum())}, '
                    f'Neutrals: {int((np.abs(pnl) <= args.margin).sum())}\n\n')
        f.write(f'## Top {args.top_k} features (combined effect + correlation)\n\n')
        f.write(head.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        f.write(f'\n\n## Walk-forward survival\n\n')
        f.write(f'{n_sur}/{args.top_k} features survived 70/30 train/val split '
                    f'(same sign, |rho_val| >= 0.03)\n\n')
        f.write(surv.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        f.write(f'\n\n## Full ranking (all 185 V2 columns)\n\n')
        f.write(feats.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'\nFull ranking saved -> {args.out}')


if __name__ == '__main__':
    main()
