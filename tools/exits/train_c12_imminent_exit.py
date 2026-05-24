"""C12 candidate -- during-trade imminent-exit classifier (B6 retarget).

Target: y = (exit_ts - bar_ts) < IMMINENT_THRESHOLD_SECONDS  (default 60s)
  "Will this leg exit within the next minute?"

Action: at K bars into the trade, if P(imminent_exit) > threshold,
prepare to cut OR tighten exit. Distinct from B9 (sign/magnitude of
remaining $) -- C12 is purely a TIMING signal.

Methodology: same as B9 -- HGB classifier per K, walk-forward 4 folds,
day-based split, 95% bootstrap CI on $/day delta from a "preempt-cut"
rule (if P(imminent)>thr, cut at bar_ts and realize pnl_so_far).

Honest delta accounting:
  - If we preempt-cut, we lose the remaining pnl movement
  - If imminent exit was correct AND the next R-trigger was unfavorable,
    we saved the loss
  - If imminent exit was correct but R-trigger was favorable,
    we missed the gain
  - delta = (pnl_at_cut - exit_pnl) per cut leg

If C12 validates walk-forward + OOS, promote to next available B-slot (B10).
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


K_HORIZONS = [5, 10, 30, 60, 120]
IMMINENT_THRESHOLD_SECONDS = 60.0
RECALL_BUDGETS = [0.30, 0.50, 0.70]
N_FOLDS = 4
N_BOOTSTRAP = 4000


def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {'leg_id', 'day', 'entry_ts', 'leg_dir', 'K', 'bar_ts',
            'r_price', 'exit_ts', 'exit_pnl_pts', 'exit_pnl_usd',
            'bars_since_entry'}
    return [c for c in df.columns
            if c not in skip and df[c].dtype != object]


def walk_forward_folds(days, n_folds):
    n_days = len(days)
    val_size = n_days // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = val_size * (i + 1)
        val_end = val_size * (i + 2)
        if val_end > n_days:
            val_end = n_days
        folds.append((set(days[:train_end]), set(days[train_end:val_end])))
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


def evaluate_preempt_cut(val_df, y_pred, threshold):
    """If P(imminent)>thr, cut at K. Realized = pnl_usd_so_far.
    Delta = realized - exit_pnl_usd."""
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
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    print(f'Loading: {args.traj}')
    df = pd.read_parquet(args.traj)
    feat_cols = get_feature_cols(df)
    print(f'  rows {len(df):,}   features {len(feat_cols)}')

    # Target: remaining seconds < threshold (= imminent exit)
    df['remaining_seconds'] = df['exit_ts'] - df['bar_ts']
    df['imminent'] = (df['remaining_seconds'] < IMMINENT_THRESHOLD_SECONDS).astype(int)
    print(f'  Imminent rate overall: {df["imminent"].mean()*100:.1f}%')
    print(f'  By K:')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        print(f'    K={K:>3}  imminent rate {sub["imminent"].mean()*100:.1f}%   n={len(sub):,}')

    days = sorted(df['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)

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
            y_tr = train['imminent'].values
            X_va = val[feat_cols].fillna(0.0).values
            y_va = val['imminent'].values
            if y_tr.sum() < 20 or y_va.sum() < 5:
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
                per_day_delta, n_cut = evaluate_preempt_cut(val, y_pred, thr)
                ci_lo, ci_hi = bootstrap_ci(per_day_delta)
                wf_rows.append({
                    'fold': fold_idx + 1,
                    'K': K,
                    'recall_budget': rec,
                    'auc': auc,
                    'p_pos_train': float(y_tr.mean()),
                    'p_pos_val': float(y_va.mean()),
                    'mean_delta_per_day': float(per_day_delta.mean()),
                    'ci_lo': ci_lo, 'ci_hi': ci_hi,
                    'significant': ci_lo > 0,
                    'n_cut': n_cut,
                })

    wf_df = pd.DataFrame(wf_rows)
    wf_csv = out_dir / 'c12_imminent_exit_walk_forward.csv'
    wf_df.to_csv(wf_csv, index=False)
    print(f'\nWrote: {wf_csv}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('C12 CANDIDATE  --  imminent-exit classifier (B6 retarget)')
    out('=' * 100)
    out(f'Target: y=1 if exit_ts - bar_ts < {IMMINENT_THRESHOLD_SECONDS}s')
    out(f'Folds: {N_FOLDS} expanding-window, day-based')
    out(f'IS days: {len(days)}')
    out('')

    for K in K_HORIZONS:
        for rec in RECALL_BUDGETS:
            sub = wf_df[(wf_df['K'] == K) & (wf_df['recall_budget'] == rec)]
            if len(sub) == 0:
                continue
            out(f'K={K:>3}  rec={rec:.2f}   '
                f'AUC {sub["auc"].mean():.3f}   '
                f'P(pos) val {sub["p_pos_val"].mean()*100:.1f}%   '
                f'mean delta ${sub["mean_delta_per_day"].mean():+.0f}/day   '
                f'avg CI [${sub["ci_lo"].mean():+.0f}, ${sub["ci_hi"].mean():+.0f}]   '
                f'sig {int(sub["significant"].sum())}/{len(sub)}')

    out('')
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
        out('--- C12 IS walk-forward: NO config qualified (sig >= 2/4 + positive). ---')
        out('Candidate STAYS as C12 (failed). Not eligible for B-slot promotion.')
    else:
        out('--- C12 qualified IS configs (sig >= 2/4 folds, positive delta) ---')
        out(f'  {"K":>4}  {"rec":>5}  {"AUC":>6}  {"delta_$/day":>11}  {"sig":>6}')
        for _, r in qualified.iterrows():
            out(f'  {int(r["K"]):>4}  {r["recall_budget"]:>5.2f}  '
                f'{r["mean_auc"]:>6.3f}  ${r["mean_delta"]:>+9.0f}    '
                f'{int(r["n_sig"])}/{int(r["n_folds"])}')

    # Train production models (one per K) on FULL IS
    out('')
    out('--- Training C12 candidate models on FULL IS (for potential OOS test) ---')
    for K in K_HORIZONS:
        sub = df[df['K'] == K]
        X = sub[feat_cols].fillna(0.0).values
        y = sub['imminent'].values
        if y.sum() < 30:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42, l2_regularization=1.0,
        )
        clf.fit(X, y)
        prod_path = out_dir / f'c12_imminent_exit_K{K}.pkl'
        with open(prod_path, 'wb') as f:
            pickle.dump({'model': clf, 'feat_cols': feat_cols,
                          'K': K, 'imminent_threshold_seconds': IMMINENT_THRESHOLD_SECONDS,
                          'n_train': len(X), 'n_pos': int(y.sum())}, f)
        out(f'  K={K:>3}  trained on {len(X):,} rows ({int(y.sum())} pos)  -> {prod_path.name}')

    summary_path = out_dir / 'c12_imminent_exit_summary.txt'
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {summary_path}')


if __name__ == '__main__':
    main()
