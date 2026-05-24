"""C11 — During-trade bad-trade detector (first L5 execution-layer model).

Predicts P(exit_pnl_usd < -$50) at K bars after entry, from V2 + trajectory
features. Operational use: at each K, if P(bad) >= threshold, CUT the trade
early at current price (realize pnl_usd_so_far instead of waiting for the
R-trigger exit).

Methodology (CLAUDE.md mandate):
  - Walk-forward CV: 4 expanding-window folds across 275 IS days
  - Day-based split (no leg leakage between train/val)
  - Per-K models (K in {5, 10, 30, 60, 120}) — each K has its own classifier
  - Honest delta accounting: cut at K realizes pnl_usd_so_far, not zero
  - 95% bootstrap CI on per-day delta (4000 resamples)
  - 'Significant' = CI strictly above 0

Production artifacts:
  - reports/findings/regret_oracle/c11_bad_trade_K{K}.pkl  (one per K)
    Trained on FULL IS for OOS test
  - reports/findings/regret_oracle/c11_bad_trade_walk_forward.csv
    Per-fold, per-K AUC + delta_per_day + CI
  - reports/findings/regret_oracle/c11_bad_trade_summary.txt
    Headline numbers

Target: exit_pnl_usd < -$50 ('bad trade' threshold). Why -$50:
  - 6.4% of legs hit this on IS — meaningful tail
  - -$100 has only 0.7% positives, too imbalanced
  - -$0 has 56.8% positives, mostly small choppy losers (not actionable)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm


K_HORIZONS = [5, 10, 30, 60, 120]
TARGET_THRESHOLD = -50.0   # exit_pnl_usd < this = 'bad trade'
RECALL_BUDGETS = [0.30, 0.50, 0.70]
N_FOLDS = 4
N_BOOTSTRAP = 4000
DOLLAR_PER_POINT = 2.0


def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {'leg_id', 'day', 'entry_ts', 'leg_dir', 'K', 'bar_ts',
            'r_price', 'exit_ts', 'exit_pnl_pts', 'exit_pnl_usd',
            'bars_since_entry'}
    return [c for c in df.columns
            if c not in skip and df[c].dtype != object]


def walk_forward_folds(days: list, n_folds: int) -> list:
    """Expanding-window day-based folds.

    With 275 days and n_folds=4, each fold's val window is ~55 days.
    Train window expands: fold 1 trains on first ~110 days, fold 4 on first 220.
    """
    n_days = len(days)
    val_size = n_days // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = val_size * (i + 1)
        val_end = val_size * (i + 2)
        if val_end > n_days:
            val_end = n_days
        train_days = set(days[:train_end])
        val_days = set(days[train_end:val_end])
        folds.append((train_days, val_days))
    return folds


def find_threshold_for_recall(y_true, y_score, target_recall):
    pos = (y_true == 1).sum()
    if pos == 0:
        return None
    order = np.argsort(-y_score)
    cum_pos = np.cumsum(y_true[order])
    rec = cum_pos / pos
    idx = np.searchsorted(rec, target_recall, side='left')
    if idx >= len(rec):
        return None
    return float(y_score[order[idx]])


def evaluate_cut_rule(val_df, y_pred, threshold):
    """Honest delta accounting: cut at K realizes pnl_usd_so_far,
    not zero. Returns per-day delta."""
    pnl_at_cut = val_df['pnl_usd_so_far'].values
    pnl_at_exit = val_df['exit_pnl_usd'].values
    cut_mask = (y_pred >= threshold)
    delta = np.where(cut_mask, pnl_at_cut - pnl_at_exit, 0.0)
    val_df = val_df.copy()
    val_df['delta'] = delta
    per_day = val_df.groupby('day')['delta'].sum().values
    return per_day, int(cut_mask.sum())


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    ap.add_argument('--out-dir',
                    default='reports/findings/regret_oracle')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading trajectory: {args.traj}')
    df = pd.read_parquet(args.traj)
    feat_cols = get_feature_cols(df)
    print(f'  rows {len(df):,}   legs {df["leg_id"].nunique():,}   '
          f'days {df["day"].nunique()}   features {len(feat_cols)}')

    days = sorted(df['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)
    print(f'Walk-forward folds: {N_FOLDS}')
    for i, (tr_days, va_days) in enumerate(folds):
        print(f'  fold {i+1}: train {len(tr_days)} days, val {len(va_days)} days')

    # Walk-forward results: per fold × K × recall
    wf_rows = []
    for fold_idx, (tr_days, va_days) in enumerate(folds):
        print(f'\n=== Fold {fold_idx + 1}/{N_FOLDS} ===')
        for K in K_HORIZONS:
            sub = df[df['K'] == K]
            train = sub[sub['day'].isin(tr_days)]
            val = sub[sub['day'].isin(va_days)]
            if len(train) < 100 or len(val) < 20:
                continue
            X_tr = train[feat_cols].fillna(0.0).values
            y_tr = (train['exit_pnl_usd'].values < TARGET_THRESHOLD).astype(int)
            X_va = val[feat_cols].fillna(0.0).values
            y_va = (val['exit_pnl_usd'].values < TARGET_THRESHOLD).astype(int)

            if y_tr.sum() < 10 or y_va.sum() < 3:
                continue

            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42, l2_regularization=1.0,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, y_pred)

            for rec in RECALL_BUDGETS:
                thr = find_threshold_for_recall(y_va, y_pred, rec)
                if thr is None:
                    continue
                per_day_delta, n_cut = evaluate_cut_rule(val, y_pred, thr)
                ci_lo, ci_hi = bootstrap_ci(per_day_delta)
                wf_rows.append({
                    'fold': fold_idx + 1,
                    'K': K,
                    'recall_budget': rec,
                    'auc': auc,
                    'threshold': thr,
                    'n_cut': n_cut,
                    'val_days': val['day'].nunique(),
                    'val_legs': len(val),
                    'mean_delta_per_day': float(per_day_delta.mean()),
                    'median_delta_per_day': float(np.median(per_day_delta)),
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'significant': ci_lo > 0,
                })

    wf_df = pd.DataFrame(wf_rows)
    wf_csv = out_dir / 'c11_bad_trade_walk_forward.csv'
    wf_df.to_csv(wf_csv, index=False)
    print(f'\nWrote: {wf_csv}')

    # Aggregate across folds: mean AUC, mean delta, fold-level bootstrap on
    # delta means (i.e., is the AVERAGE per-fold delta significant?)
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('C11 BAD-TRADE DETECTOR (failed candidate)  --  walk-forward CV summary')
    out('=' * 100)
    out(f'Target: exit_pnl_usd < ${TARGET_THRESHOLD}')
    out(f'Folds: {N_FOLDS}  (expanding window)')
    out(f'IS days: {len(days)}')
    out(f'Trajectory rows: {len(df):,}  legs: {df["leg_id"].nunique():,}')
    out('')

    for K in K_HORIZONS:
        for rec in RECALL_BUDGETS:
            sub = wf_df[(wf_df['K'] == K) & (wf_df['recall_budget'] == rec)]
            if len(sub) == 0:
                continue
            n_folds_seen = len(sub)
            mean_auc = sub['auc'].mean()
            mean_delta = sub['mean_delta_per_day'].mean()
            mean_ci_lo = sub['ci_lo'].mean()
            mean_ci_hi = sub['ci_hi'].mean()
            n_sig = int(sub['significant'].sum())
            out(f'K={K:>3}  rec={rec:.2f}   '
                f'AUC {mean_auc:.3f} (avg {n_folds_seen} folds)   '
                f'mean delta ${mean_delta:+.0f}/day   '
                f'avg per-fold CI [${mean_ci_lo:+.0f}, ${mean_ci_hi:+.0f}]   '
                f'sig {n_sig}/{n_folds_seen} folds')

    # Pick best operating point: highest mean delta with sig >= 2/4 folds
    out('')
    out('--- Best operating points (sig in >= half of folds) ---')
    sdf = wf_df.groupby(['K', 'recall_budget']).agg(
        mean_auc=('auc', 'mean'),
        mean_delta=('mean_delta_per_day', 'mean'),
        n_sig=('significant', 'sum'),
        n_folds=('significant', 'count'),
    ).reset_index()
    sdf['sig_frac'] = sdf['n_sig'] / sdf['n_folds']
    qualified = sdf[(sdf['sig_frac'] >= 0.5) & (sdf['mean_delta'] > 0)].sort_values(
        'mean_delta', ascending=False)
    if len(qualified) == 0:
        out('  NONE qualified. The classifier does not improve $/day with significance')
        out('  in at least half of folds at any (K, recall) operating point.')
        out('  KILL THE PARADIGM? Or weaken target threshold and retry.')
    else:
        out(f'  {"K":>4}  {"rec":>5}  {"AUC":>6}  {"delta_$/day":>11}  {"sig":>6}')
        for _, r in qualified.iterrows():
            out(f'  {int(r["K"]):>4}  {r["recall_budget"]:>5.2f}  '
                f'{r["mean_auc"]:>6.3f}  ${r["mean_delta"]:>+9.0f}    '
                f'{int(r["n_sig"])}/{int(r["n_folds"])}')

    # Train production models (one per K) on FULL IS — used for OOS test
    out('')
    out('--- Training production models on FULL IS ---')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        X = sub[feat_cols].fillna(0.0).values
        y = (sub['exit_pnl_usd'].values < TARGET_THRESHOLD).astype(int)
        if y.sum() < 30:
            out(f'  K={K}: insufficient positives ({y.sum()}), skipping')
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42, l2_regularization=1.0,
        )
        clf.fit(X, y)
        prod_path = out_dir / f'c11_bad_trade_K{K}.pkl'
        with open(prod_path, 'wb') as f:
            pickle.dump({'model': clf, 'feat_cols': feat_cols,
                          'K': K, 'target_threshold': TARGET_THRESHOLD,
                          'n_train': len(X), 'n_pos': int(y.sum())}, f)
        out(f'  K={K:>3}  trained on {len(X):,} rows ({int(y.sum())} pos)  -> {prod_path.name}')

    summary_path = out_dir / 'c11_bad_trade_summary.txt'
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {summary_path}')
    print(f'Wrote: {wf_csv}')


if __name__ == '__main__':
    main()
