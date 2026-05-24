"""Trajectory diagnostic v2 -- honest accounting + V2 ablation.

FIXES vs v1:
  1. Delta accounting: when we cut at K, realized P&L = pnl_usd_so_far,
     NOT zero. True savings = pnl_at_cut - pnl_at_exit per cut leg.
  2. Three feature sets (ablation):
       - 'all'    : V2 + trajectory (mae/mfe/pnl_so_far)
       - 'traj'   : trajectory only (mae/mfe/pnl_so_far + has_reached_R)
       - 'v2'     : V2 features only (no trajectory)
     If 'v2' AUC << 'all' AUC, the signal is mostly trajectory.
     If 'v2' AUC ~ 'all' AUC, V2 features carry the signal.

The diagnostic answers two questions:
  Q1: Is there ANY signal? (AUC > 0.65 on val)
  Q2: Where does the signal come from? (V2 vs trajectory ablation)

Operational delta:
  delta_per_leg_if_cut = pnl_usd_so_far - exit_pnl_usd
  cut at K means we exit early and realize pnl_usd_so_far instead of exit_pnl.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss


K_HORIZONS = [5, 10, 30, 60, 120]
TARGETS = [-50.0, -100.0]
RECALL_BUDGETS = [0.30, 0.50, 0.70, 0.90]

TRAJ_COLS = ['mae_pts_so_far', 'mfe_pts_so_far', 'pnl_pts_so_far',
             'pnl_usd_so_far', 'has_reached_R_against']


def get_v2_only_cols(df: pd.DataFrame) -> list:
    skip = {'leg_id', 'day', 'entry_ts', 'leg_dir', 'K', 'bar_ts',
            'r_price', 'exit_ts', 'exit_pnl_pts', 'exit_pnl_usd',
            'bars_since_entry',
            'mae_pts_so_far', 'mfe_pts_so_far', 'pnl_pts_so_far',
            'pnl_usd_so_far', 'has_reached_R_against'}
    return [c for c in df.columns
            if c not in skip and df[c].dtype != object]


def get_all_cols(df: pd.DataFrame) -> list:
    skip = {'leg_id', 'day', 'entry_ts', 'leg_dir', 'K', 'bar_ts',
            'r_price', 'exit_ts', 'exit_pnl_pts', 'exit_pnl_usd',
            'bars_since_entry'}
    return [c for c in df.columns
            if c not in skip and df[c].dtype != object]


def split_by_day(df: pd.DataFrame, train_frac: float = 0.80) -> tuple:
    days = sorted(df['day'].unique())
    n_train = int(len(days) * train_frac)
    train_days = set(days[:n_train])
    return df[df['day'].isin(train_days)], df[~df['day'].isin(train_days)]


def eval_at_recall(y_true, y_score, target_recall):
    pos = (y_true == 1).sum()
    if pos == 0:
        return None
    order = np.argsort(-y_score)
    cum_pos = np.cumsum(y_true[order])
    rec = cum_pos / pos
    idx = np.searchsorted(rec, target_recall, side='left')
    if idx >= len(rec):
        return None
    thr = y_score[order[idx]]
    return float(thr)


def evaluate_cut_rule(val_df, y_pred, thr, target_label):
    """HONEST delta accounting:
      - Cut legs realize pnl_usd_so_far at K (we exit early at K's price).
      - Original (no cut) realizes exit_pnl_usd.
      - Delta per cut leg = pnl_usd_so_far - exit_pnl_usd.
    """
    pnl_at_cut = val_df['pnl_usd_so_far'].values
    pnl_at_exit = val_df['exit_pnl_usd'].values
    y_true = (pnl_at_exit < target_label).astype(int)

    cut_mask = (y_pred >= thr)
    if cut_mask.sum() == 0:
        return None

    # Per-leg delta
    delta_per_leg = np.where(cut_mask, pnl_at_cut - pnl_at_exit, 0.0)
    total_delta = float(delta_per_leg.sum())
    n_cut = int(cut_mask.sum())

    tp = ((cut_mask) & (y_true == 1)).sum()
    fp = ((cut_mask) & (y_true == 0)).sum()
    prec = tp / max(tp + fp, 1)
    recall = tp / max(y_true.sum(), 1)
    cut_mean_at_cut = float(pnl_at_cut[cut_mask].mean())
    cut_mean_at_exit = float(pnl_at_exit[cut_mask].mean())

    n_days = val_df['day'].nunique()
    return {
        'thr': float(thr),
        'n_cut': n_cut,
        'precision': float(prec),
        'recall': float(recall),
        'total_delta_usd': total_delta,
        'delta_per_day': total_delta / max(n_days, 1),
        'cut_mean_at_cut': cut_mean_at_cut,
        'cut_mean_at_exit': cut_mean_at_exit,
        'saved_per_leg': cut_mean_at_cut - cut_mean_at_exit,
    }


def fit_and_eval(train, val, feat_cols, target):
    X_tr = train[feat_cols].fillna(0.0).values
    y_tr = (train['exit_pnl_usd'].values < target).astype(int)
    X_va = val[feat_cols].fillna(0.0).values
    y_va = (val['exit_pnl_usd'].values < target).astype(int)

    if y_tr.sum() < 20 or y_va.sum() < 5:
        return None, None

    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        random_state=42, l2_regularization=1.0,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, y_pred)
    brier = brier_score_loss(y_va, y_pred)
    return {'auc': auc, 'brier': brier, 'y_pred': y_pred,
            'p_pos_train': float(y_tr.mean()),
            'p_pos_val': float(y_va.mean())}, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/trajectory_diagnostic_v2.txt')
    args = ap.parse_args()

    print(f'Loading trajectory: {args.traj}')
    df = pd.read_parquet(args.traj)
    print(f'  rows {len(df):,}   legs {df["leg_id"].nunique():,}   '
          f'days {df["day"].nunique()}')

    v2_cols = get_v2_only_cols(df)
    all_cols = get_all_cols(df)
    print(f'  V2-only cols: {len(v2_cols)}')
    print(f'  All cols (V2 + traj): {len(all_cols)}')
    print(f'  Trajectory-only cols: {TRAJ_COLS}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('TRAJECTORY DIAGNOSTIC v2 -- honest delta accounting + V2 vs trajectory ablation')
    out('=' * 100)
    out(f'Source: {args.traj}')
    out(f'Total rows: {len(df):,}   Legs: {df["leg_id"].nunique():,}   '
        f'Days: {df["day"].nunique()}')
    out(f'V2 cols: {len(v2_cols)}   Trajectory cols: {len(TRAJ_COLS)}')
    out(f'Split: first 80% of days = train, last 20% = val (day-based)')
    out('')
    out('HONEST DELTA: when we cut at K, we realize pnl_usd_so_far at K, not zero.')
    out('  delta_per_leg_if_cut = pnl_usd_so_far_at_K - exit_pnl_usd')
    out('')

    summary_rows = []
    for K in K_HORIZONS:
        sub = df[df['K'] == K].reset_index(drop=True)
        if len(sub) < 100:
            continue
        train, val = split_by_day(sub, train_frac=0.80)

        out(f'================ K={K} ({K*5}s)  '
            f'train {len(train):,}  val {len(val):,} ================')

        for target in TARGETS:
            out(f'\nTarget: exit_pnl_usd < ${target:+.0f}')
            for fs_name, fs_cols in [('all (V2+traj)', all_cols),
                                       ('traj only', TRAJ_COLS),
                                       ('v2 only', v2_cols)]:
                res, _ = fit_and_eval(train, val, fs_cols, target)
                if res is None:
                    out(f'  {fs_name:<18}  insufficient positives, skip')
                    continue
                out(f'  {fs_name:<18}  AUC {res["auc"]:.3f}  '
                    f'Brier {res["brier"]:.4f}  '
                    f'P(pos) val {res["p_pos_val"]*100:.1f}%')

                # Operational @ each recall budget
                y_va = (val['exit_pnl_usd'].values < target).astype(int)
                for rec_budget in RECALL_BUDGETS:
                    thr = eval_at_recall(y_va, res['y_pred'], rec_budget)
                    if thr is None:
                        continue
                    op = evaluate_cut_rule(val, res['y_pred'], thr, target)
                    if op is None:
                        continue
                    out(f'    rec {rec_budget:.2f}:  '
                        f'cuts {op["n_cut"]:>5}  '
                        f'prec {op["precision"]*100:>5.1f}%  '
                        f'saved/leg ${op["saved_per_leg"]:+.2f}  '
                        f'total delta ${op["total_delta_usd"]:+,.0f}  '
                        f'$/day ${op["delta_per_day"]:+.0f}')
                summary_rows.append({'K': K, 'target': target,
                                       'features': fs_name, 'auc': res['auc']})
        out('')

    out('=' * 100)
    out('SUMMARY: V2-only vs traj-only vs all-features AUCs')
    out('=' * 100)
    sdf = pd.DataFrame(summary_rows)
    if len(sdf) > 0:
        pivot = sdf.pivot_table(index=['K', 'target'], columns='features',
                                  values='auc')
        out(pivot.to_string())
        out('')
        out('Interpretation:')
        out('  - If v2-only AUC ~ all AUC: V2 carries the signal -> classifier is genuinely new')
        out('  - If traj-only AUC ~ all AUC: signal is just MAE/pnl_so_far ->')
        out('    not really a "V2-based bad-trade detector", just a smart trailing stop')
        out('  - If neither v2 nor traj alone reaches all AUC: signal needs both ->')
        out('    classifier path requires the full feature set')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report}')


if __name__ == '__main__':
    main()
