"""C11 v2 -- target = 'cutting now saves money vs holding to R-trigger'.

Operationally aligned target. Replaces v1 target (exit_pnl < -$50) which
identified bad-outcome trades but had high Type 1 cost (mid-MAE trades
that recover got falsely cut).

Target: y = (pnl_usd_so_far_at_K - exit_pnl_usd > SAVINGS_THRESHOLD)
  - y=1: cutting at K saves at least $SAVINGS_THRESHOLD
  - y=0: holding to R-trigger is equal or better

This is the dual of the v1 target -- v1 asked 'is this trade bad?',
v2 asks 'is it worth cutting NOW?'. The cut decision is operationally
the right framing.

Methodology identical to v1: walk-forward 4 folds, per-K models,
honest delta accounting, 95% bootstrap CI.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


K_HORIZONS = [5, 10, 30, 60, 120]
SAVINGS_THRESHOLDS = [20.0, 40.0]   # $X-saved-by-cutting target
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


def evaluate_cut_rule(val_df, y_pred, threshold):
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
    days = sorted(df['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)

    # Compute the new target columns
    df['cut_savings'] = df['pnl_usd_so_far'] - df['exit_pnl_usd']

    wf_rows = []
    for SAVE_THR in SAVINGS_THRESHOLDS:
        print(f'\n--- Savings threshold $+{SAVE_THR} ---')
        for fold_idx, (tr_days, va_days) in enumerate(folds):
            for K in K_HORIZONS:
                sub = df[df['K'] == K]
                train = sub[sub['day'].isin(tr_days)]
                val = sub[sub['day'].isin(va_days)]
                if len(train) < 100 or len(val) < 20:
                    continue
                X_tr = train[feat_cols].fillna(0.0).values
                y_tr = (train['cut_savings'].values > SAVE_THR).astype(int)
                X_va = val[feat_cols].fillna(0.0).values
                y_va = (val['cut_savings'].values > SAVE_THR).astype(int)
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
                    per_day_delta, n_cut = evaluate_cut_rule(val, y_pred, thr)
                    ci_lo, ci_hi = bootstrap_ci(per_day_delta)
                    wf_rows.append({
                        'fold': fold_idx + 1,
                        'savings_thr': SAVE_THR,
                        'K': K,
                        'recall_budget': rec,
                        'auc': auc,
                        'p_pos_train': float(y_tr.mean()),
                        'p_pos_val': float(y_va.mean()),
                        'mean_delta_per_day': float(per_day_delta.mean()),
                        'median_delta_per_day': float(np.median(per_day_delta)),
                        'ci_lo': ci_lo,
                        'ci_hi': ci_hi,
                        'significant': ci_lo > 0,
                        'n_cut': n_cut,
                        'val_days': val['day'].nunique(),
                    })

    wf_df = pd.DataFrame(wf_rows)
    wf_csv = out_dir / 'c11_v2_cut_saves_walk_forward.csv'
    wf_df.to_csv(wf_csv, index=False)
    print(f'\nWrote: {wf_csv}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('C11 v2 -- "cut saves money" (failed candidate) target  walk-forward summary')
    out('=' * 100)
    out('Target y=1 if (pnl_usd_so_far_at_K - exit_pnl_usd) > savings_threshold')
    out(f'Folds: {N_FOLDS} expanding-window, day-based')
    out(f'IS days: {len(days)}')
    out('')

    for SAVE_THR in SAVINGS_THRESHOLDS:
        out(f'\n=== Savings threshold = $+{SAVE_THR} ===')
        for K in K_HORIZONS:
            for rec in RECALL_BUDGETS:
                sub = wf_df[(wf_df['K'] == K) &
                             (wf_df['recall_budget'] == rec) &
                             (wf_df['savings_thr'] == SAVE_THR)]
                if len(sub) == 0:
                    continue
                out(f'K={K:>3}  rec={rec:.2f}   '
                    f'AUC {sub["auc"].mean():.3f}   '
                    f'P(pos) val {sub["p_pos_val"].mean()*100:.1f}%   '
                    f'mean delta ${sub["mean_delta_per_day"].mean():+.0f}/day   '
                    f'avg CI [${sub["ci_lo"].mean():+.0f}, ${sub["ci_hi"].mean():+.0f}]   '
                    f'sig {int(sub["significant"].sum())}/{len(sub)}')

    out('')
    out('--- Best operating points (sig in >= half of folds, positive delta) ---')
    sdf = wf_df.groupby(['savings_thr', 'K', 'recall_budget']).agg(
        mean_auc=('auc', 'mean'),
        mean_delta=('mean_delta_per_day', 'mean'),
        n_sig=('significant', 'sum'),
        n_folds=('significant', 'count'),
    ).reset_index()
    sdf['sig_frac'] = sdf['n_sig'] / sdf['n_folds']
    qualified = sdf[(sdf['sig_frac'] >= 0.5) & (sdf['mean_delta'] > 0)].sort_values(
        'mean_delta', ascending=False)
    if len(qualified) == 0:
        out('  NONE qualified. C11 v2 also fails walk-forward.')
        out('  The cut decision is structurally unprofitable on average across folds.')
        out('  Verdict: kill bad-trade-cutter path; signal exists in AUC but cannot')
        out('  be operationalized into a profitable cut rule.')
    else:
        out(f'  {"save":>4}  {"K":>4}  {"rec":>5}  {"AUC":>6}  {"delta_$/day":>11}  {"sig":>6}')
        for _, r in qualified.iterrows():
            out(f'  ${int(r["savings_thr"]):>+3}  {int(r["K"]):>4}  '
                f'{r["recall_budget"]:>5.2f}  {r["mean_auc"]:>6.3f}  '
                f'${r["mean_delta"]:>+9.0f}    {int(r["n_sig"])}/{int(r["n_folds"])}')

        # Train production models for qualified configs
        out('')
        out('--- Training production models (qualified configs) ---')
        for _, r in qualified.iterrows():
            SAVE_THR = float(r['savings_thr'])
            K = int(r['K'])
            sub = df[df['K'] == K]
            X = sub[feat_cols].fillna(0.0).values
            y = (sub['cut_savings'].values > SAVE_THR).astype(int)
            if y.sum() < 30:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42, l2_regularization=1.0,
            )
            clf.fit(X, y)
            tag = f'K{K}_save{int(SAVE_THR)}'
            prod_path = out_dir / f'b9v2_cut_saves_{tag}.pkl'
            with open(prod_path, 'wb') as f:
                pickle.dump({'model': clf, 'feat_cols': feat_cols,
                              'K': K, 'savings_thr': SAVE_THR,
                              'recall_budget': float(r['recall_budget']),
                              'n_train': len(X), 'n_pos': int(y.sum())}, f)
            out(f'  K={K:>3}  save=$+{int(SAVE_THR)}  '
                f'-> {prod_path.name}  ({int(y.sum())} pos / {len(X)})')

    summary_path = out_dir / 'c11_v2_cut_saves_summary.txt'
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {summary_path}')


if __name__ == '__main__':
    main()
