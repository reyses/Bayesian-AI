"""Trajectory diagnostic -- does any feature at any K predict bad trades?

For each K horizon and each loss threshold:
  - y = (exit_pnl_usd < threshold)
  - Train LightGBM with day-based 80/20 holdout (no random shuffle)
  - Report: AUC, P(bad), top features
  - At threshold tuned to recall {0.3, 0.5, 0.7, 0.9}, report precision +
    realistic dollar cost of cuts (cuts × mean_pnl_of_cut_legs)
  - Operational gate: NET $/day if we cut all predicted-bad legs at K

If best AUC < 0.65 at K=120, classifier path is unlikely to clear the
trail-tightening wall. Flag dead end.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm


K_HORIZONS = [5, 10, 30, 60, 120]
TARGETS = [-50.0, -100.0, -200.0]   # exit_pnl_usd < X = "bad trade"
RECALL_BUDGETS = [0.30, 0.50, 0.70, 0.90]
DOLLAR_PER_POINT = 2.0


def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {'leg_id', 'day', 'entry_ts', 'leg_dir', 'K', 'bar_ts',
            'r_price', 'exit_ts', 'exit_pnl_pts', 'exit_pnl_usd',
            'bars_since_entry'}
    return [c for c in df.columns
            if c not in skip and df[c].dtype != object]


def split_by_day(df: pd.DataFrame, train_frac: float = 0.80) -> tuple:
    """Day-based split: first `train_frac` of unique days = train, rest = val."""
    days = sorted(df['day'].unique())
    n_train = int(len(days) * train_frac)
    train_days = set(days[:n_train])
    train_mask = df['day'].isin(train_days)
    return df[train_mask], df[~train_mask]


def eval_at_recall(y_true, y_score, target_recall):
    """Pick threshold to give >= target_recall on positives.
    Returns (threshold, precision, recall, cuts_count)."""
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
    pred = (y_score >= thr).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    cuts = tp + fp
    prec = tp / cuts if cuts > 0 else 0.0
    return {'thr': float(thr), 'precision': float(prec),
            'recall': float(tp / pos), 'cuts': int(cuts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/trajectory_diagnostic.txt')
    args = ap.parse_args()

    print(f'Loading trajectory: {args.traj}')
    df = pd.read_parquet(args.traj)
    print(f'  rows {len(df):,}   legs {df["leg_id"].nunique():,}   days {df["day"].nunique()}')
    feat_cols = get_feature_cols(df)
    print(f'  feature cols: {len(feat_cols)}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 90)
    out('TRAJECTORY DIAGNOSTIC -- bad-trade detectability per (K horizon, target threshold)')
    out('=' * 90)
    out(f'Source: {args.traj}')
    out(f'Total rows: {len(df):,}   Legs: {df["leg_id"].nunique():,}   Days: {df["day"].nunique()}')
    out(f'Feature cols: {len(feat_cols)}')
    out(f'Split: first 80% of days = train, last 20% = val')
    out('')

    summary_rows = []
    for K in K_HORIZONS:
        sub = df[df['K'] == K].reset_index(drop=True)
        if len(sub) < 100:
            out(f'K={K}: too few rows ({len(sub)}), skip')
            continue
        train, val = split_by_day(sub, train_frac=0.80)
        X_tr = train[feat_cols].fillna(0.0).values
        X_va = val[feat_cols].fillna(0.0).values

        out(f'--- K={K} ({K*5}s) ---')
        out(f'  Train rows: {len(train):,}   Val rows: {len(val):,}')

        for thr in TARGETS:
            y_tr = (train['exit_pnl_usd'].values < thr).astype(int)
            y_va = (val['exit_pnl_usd'].values < thr).astype(int)
            if y_tr.sum() < 20 or y_va.sum() < 5:
                out(f'  target=<${thr}  :  too few positives '
                    f'(train {y_tr.sum()} val {y_va.sum()}), skip')
                continue
            p_pos_tr = y_tr.mean()
            p_pos_va = y_va.mean()

            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.05,
                random_state=42, l2_regularization=1.0,
            )
            clf.fit(X_tr, y_tr)
            y_pred_va = clf.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, y_pred_va)
            brier = brier_score_loss(y_va, y_pred_va)

            out(f'  target=<${thr:>5.0f}  P(pos) train {p_pos_tr*100:.1f}%  '
                f'val {p_pos_va*100:.1f}%   AUC {auc:.3f}   Brier {brier:.4f}')

            row = {'K': K, 'target': thr, 'p_pos_train': p_pos_tr,
                   'p_pos_val': p_pos_va, 'auc': auc, 'brier': brier}

            # Operational: at each recall budget, what does cutting cost?
            val_pnl = val['exit_pnl_usd'].values
            for rec in RECALL_BUDGETS:
                res = eval_at_recall(y_va, y_pred_va, rec)
                if res is None:
                    continue
                # If we CUT every predicted-bad leg, the saved $ = -sum of their realized pnl
                cut_mask = (y_pred_va >= res['thr'])
                cut_pnl = val_pnl[cut_mask]
                kept_pnl = val_pnl[~cut_mask]
                # Naive cut-rule effect on validation period:
                # original_total = val_pnl.sum()
                # cut_total = kept_pnl.sum()  # we keep only non-cut legs
                # delta = cut_total - original_total = -cut_pnl.sum()
                delta_usd = -float(cut_pnl.sum())
                n_val_days = val['day'].nunique()
                delta_per_day = delta_usd / max(n_val_days, 1)
                out(f'    recall {rec:.2f}: cuts={res["cuts"]:>5}  '
                    f'prec {res["precision"]*100:>5.1f}%  '
                    f'delta_total ${delta_usd:+,.0f}  '
                    f'delta_$/day ${delta_per_day:+.0f}  '
                    f'(cut_mean ${cut_pnl.mean() if len(cut_pnl) else 0:+.2f}  '
                    f'kept_mean ${kept_pnl.mean() if len(kept_pnl) else 0:+.2f})')
                row[f'rec_{int(rec*100)}_prec'] = res['precision']
                row[f'rec_{int(rec*100)}_delta_per_day'] = delta_per_day
                row[f'rec_{int(rec*100)}_cuts'] = res['cuts']
            summary_rows.append(row)
            out('')

    out('')
    out('=' * 90)
    out('SUMMARY MATRIX (AUC > 0.65 = worth pursuing; delta_$/day > 0 = cuts are net-positive)')
    out('=' * 90)
    out(f'{"K":>4}  {"target":>7}  {"AUC":>6}  {"delta_$/day @ rec=0.50":>22}  '
        f'{"delta_$/day @ rec=0.70":>22}  {"delta_$/day @ rec=0.90":>22}')
    for r in summary_rows:
        d50 = r.get('rec_50_delta_per_day')
        d70 = r.get('rec_70_delta_per_day')
        d90 = r.get('rec_90_delta_per_day')
        out(f'{int(r["K"]):>4}  ${float(r["target"]):>+5.0f}  {r["auc"]:>6.3f}  '
            f'{"${:+,.0f}".format(d50) if d50 is not None else "n/a":>22}  '
            f'{"${:+,.0f}".format(d70) if d70 is not None else "n/a":>22}  '
            f'{"${:+,.0f}".format(d90) if d90 is not None else "n/a":>22}')

    out('')
    out('INTERPRETATION GUIDE')
    out('  - AUC < 0.60 -> no separable signal; classifier path likely dead.')
    out('  - AUC 0.60-0.65 -> weak signal; expect minimal $/day gain even at best.')
    out('  - AUC > 0.65 -> separable signal; pursue classifier build.')
    out('  - delta_$/day > 0 at any recall budget -> cutting is net-positive at that')
    out('    operating point. Recall 0.50 typically the safest (low Type 1 cost).')
    out('  - delta_$/day < 0 at all recalls -> Type 1 cost > Type 2 savings; the')
    out('    classifier kills positive legs faster than it saves bad ones.')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report}')


if __name__ == '__main__':
    main()
