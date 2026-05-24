"""C15 candidate -- pyramid-confidence stacker on B9 K=5 decisions.

Refines B9's pyramid (1.5x) action. B9 produces predicted remaining
P&L; when > $50, pyramid. C15 trains on the SUBSET of legs where B9
WOULD pyramid, predicting whether the pyramid would pay off.

Target: y = (exit_pnl_usd > pnl_usd_so_far_at_K)
  = "the leg continued in our favor after K bars"

If C15 P(payoff) < threshold, attenuate B9's pyramid to 1.0x.
Otherwise keep at 1.5x.

Features:
  - B9's pred_remaining (already in trajectory dataset as a "leakage"
    feature -- we use it as input, not output)
  - V2 + trajectory at K bar
  - All inputs are available at decision time

Methodology: walk-forward 4-fold CV on the SUBSET of legs that B9
would pyramid (pred_remaining > 50 at K=5). Then sealed OOS test.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


K = 5  # B9 K=5 is the validated production
PYRAMID_THRESHOLD = 50.0  # B9's pyramid threshold
N_FOLDS = 4
N_BOOTSTRAP = 4000
RECALL_BUDGETS = [0.30, 0.50, 0.70, 0.90]


def get_feature_cols(df):
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


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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


def main():
    print('Loading IS trajectory + B9 model for prediction generation...')
    df = pd.read_parquet('reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    df = df[df['K'] == K].reset_index(drop=True)

    # Load B9 K=5 to get its pyramid predictions
    with open('reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl', 'rb') as f:
        b9 = pickle.load(f)
    X_all = df[b9['feat_cols']].fillna(0.0).values
    df['b9_pred'] = b9['model'].predict(X_all)

    # Restrict to pyramid candidates (B9 would say >= +$50)
    pyramid_legs = df[df['b9_pred'] > PYRAMID_THRESHOLD].copy()
    print(f'Total K=5 legs: {len(df):,}')
    print(f'Pyramid candidates (B9 pred > $50): {len(pyramid_legs):,}')
    print(f'Pyramid rate: {len(pyramid_legs) / len(df) * 100:.1f}%')

    # Define target: did the pyramid pay off?
    # = exit_pnl > pnl_so_far at K (leg continued in our favor)
    pyramid_legs['payoff'] = (pyramid_legs['exit_pnl_usd'] >
                                pyramid_legs['pnl_usd_so_far']).astype(int)
    print(f'Payoff rate: {pyramid_legs["payoff"].mean() * 100:.1f}% '
          f'({int(pyramid_legs["payoff"].sum())} / {len(pyramid_legs)})')

    feat_cols = get_feature_cols(pyramid_legs) + ['b9_pred']

    days = sorted(pyramid_legs['day'].unique())
    folds = walk_forward_folds(days, N_FOLDS)

    print(f'\nWalk-forward folds: {N_FOLDS}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('C15 CANDIDATE -- pyramid-confidence stacker (refines B9 K=5)')
    out('=' * 100)
    out(f'B9 pyramid threshold: pred > $+{PYRAMID_THRESHOLD}')
    out(f'Pyramid candidates: {len(pyramid_legs):,} legs')
    out(f'Payoff rate: {pyramid_legs["payoff"].mean()*100:.1f}%')
    out('')
    out(f'Feature count: {len(feat_cols)} (V2 + trajectory + B9 pred)')
    out('')

    # Walk-forward CV
    wf_results = []
    for fold_idx, (tr_days, va_days) in enumerate(folds):
        train = pyramid_legs[pyramid_legs['day'].isin(tr_days)]
        val = pyramid_legs[pyramid_legs['day'].isin(va_days)]
        if len(train) < 50 or len(val) < 10:
            continue
        X_tr = train[feat_cols].fillna(0.0).values
        y_tr = train['payoff'].values
        X_va = val[feat_cols].fillna(0.0).values
        y_va = val['payoff'].values

        if y_tr.sum() < 5 or y_va.sum() < 2:
            continue

        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42 + fold_idx, l2_regularization=1.0,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_pred)

        # Operational: attenuate pyramid where C15 says LOW payoff probability
        # Action: if C15 P(payoff) < threshold T, cap pyramid at 1.0 (was 1.5)
        # Realized P&L change per attenuated pyramid:
        #   original (pyramid 1.5x): pnl_at_K + 1.5 * (exit - pnl_at_K)
        #   attenuated (1.0x):       exit_pnl_usd
        #   delta = exit - [pnl_at_K + 1.5*(exit-pnl_at_K)]
        #        = exit - pnl_at_K - 1.5*exit + 1.5*pnl_at_K
        #        = -0.5*exit + 0.5*pnl_at_K
        #        = -0.5*(exit - pnl_at_K)
        # So attenuation HELPS when exit < pnl_at_K (= pyramid would have hurt)
        # And attenuation HURTS when exit > pnl_at_K (= pyramid would have helped)

        val = val.copy()
        val['c15_pred'] = y_pred
        for rec_budget in RECALL_BUDGETS:
            # Find threshold to achieve target recall on payoff cases (= keep them)
            # We want to ATTENUATE the LOW-confidence cases (1 - recall_budget keeps,
            # so threshold above 1-recall_budget keeps for full pyramid)
            thr = find_threshold_for_recall(y_va, y_pred, rec_budget)
            if thr is None:
                continue
            attenuate = y_pred < thr   # below threshold = cap pyramid

            # Delta vs B9-only (B9 alone would pyramid all of these)
            pnl_at_k = val['pnl_usd_so_far'].values
            exit_pnl = val['exit_pnl_usd'].values
            delta = np.where(attenuate, -0.5 * (exit_pnl - pnl_at_k), 0.0)
            per_day = pd.Series(delta).groupby(val['day'].values).sum().values
            ci_lo, ci_hi = bootstrap_ci(per_day)
            wf_results.append({
                'fold': fold_idx + 1,
                'auc': auc,
                'recall_budget': rec_budget,
                'attenuated': int(attenuate.sum()),
                'kept': int((~attenuate).sum()),
                'mean_delta_per_day': float(per_day.mean()),
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'significant': ci_lo > 0,
                'val_days': val['day'].nunique(),
            })

    wf_df = pd.DataFrame(wf_results)
    if len(wf_df) > 0:
        out('--- Walk-forward CV by recall budget ---')
        agg = wf_df.groupby('recall_budget').agg(
            mean_auc=('auc', 'mean'),
            mean_delta=('mean_delta_per_day', 'mean'),
            n_sig=('significant', 'sum'),
            n_folds=('significant', 'count'),
        ).reset_index()
        for _, r in agg.iterrows():
            out(f'  recall {r["recall_budget"]:.2f}: AUC {r["mean_auc"]:.3f}  '
                f'mean delta ${r["mean_delta"]:+.0f}/day  sig {int(r["n_sig"])}/{int(r["n_folds"])}')
    else:
        out('WF aggregation failed (insufficient data)')

    out('')
    # Train production C15 on full IS
    X_full = pyramid_legs[feat_cols].fillna(0.0).values
    y_full = pyramid_legs['payoff'].values
    if y_full.sum() >= 30:
        clf_prod = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42, l2_regularization=1.0,
        )
        clf_prod.fit(X_full, y_full)
        with open('reports/findings/regret_oracle/c15_pyramid_confidence_K5.pkl', 'wb') as f:
            pickle.dump({'model': clf_prod, 'feat_cols': feat_cols,
                          'K': K, 'pyramid_threshold': PYRAMID_THRESHOLD,
                          'target': 'pyramid_payoff (exit > pnl_at_K)',
                          'n_train': len(X_full), 'n_pos': int(y_full.sum())}, f)
        out(f'Production C15 model trained on {len(X_full)} pyramid candidates '
            f'({int(y_full.sum())} payoffs)')

    # === Sealed OOS test ===
    out('')
    out('=== SEALED OOS test (51-day fresh dump) ===')
    oos = pd.read_parquet('reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet')
    oos = oos[oos['K'] == K].reset_index(drop=True)
    X_oos = oos[b9['feat_cols']].fillna(0.0).values
    oos['b9_pred'] = b9['model'].predict(X_oos)
    oos_pyramid = oos[oos['b9_pred'] > PYRAMID_THRESHOLD].copy()
    out(f'OOS pyramid candidates: {len(oos_pyramid)}')

    if len(oos_pyramid) > 0:
        X_oos_pyr = oos_pyramid[feat_cols].fillna(0.0).values
        oos_pyramid['c15_pred'] = clf_prod.predict_proba(X_oos_pyr)[:, 1]
        oos_pyramid['payoff'] = (oos_pyramid['exit_pnl_usd'] >
                                   oos_pyramid['pnl_usd_so_far']).astype(int)
        actual_payoff_rate = oos_pyramid['payoff'].mean()
        out(f'OOS actual payoff rate: {actual_payoff_rate*100:.1f}%')

        if oos_pyramid['payoff'].sum() >= 5:
            auc_oos = roc_auc_score(oos_pyramid['payoff'], oos_pyramid['c15_pred'])
            out(f'OOS AUC: {auc_oos:.3f}')

            # Operational deltas
            pnl_at_k = oos_pyramid['pnl_usd_so_far'].values
            exit_pnl = oos_pyramid['exit_pnl_usd'].values
            out('')
            out(f'{"recall":>7}  {"thr":>6}  {"atten":>5}  {"kept":>5}  '
                f'{"delta_per_day":>13}  {"95% CI":>22}  sig')
            for rec_budget in RECALL_BUDGETS:
                thr = find_threshold_for_recall(
                    oos_pyramid['payoff'].values, oos_pyramid['c15_pred'].values, rec_budget)
                if thr is None:
                    continue
                attenuate = oos_pyramid['c15_pred'].values < thr
                delta = np.where(attenuate, -0.5 * (exit_pnl - pnl_at_k), 0.0)
                per_day = pd.Series(delta).groupby(oos_pyramid['day'].values).sum().values
                ci_lo, ci_hi = bootstrap_ci(per_day)
                out(f'  {rec_budget:.2f}    {thr:>6.3f}  {int(attenuate.sum()):>5}  '
                    f'{int((~attenuate).sum()):>5}  '
                    f'${per_day.mean():>+10.0f}    '
                    f'[${ci_lo:>+5.0f}, ${ci_hi:>+5.0f}]   '
                    f'{ci_lo > 0}')

    out('')
    out('=== VERDICT ===')
    if len(wf_df) > 0:
        any_sig = (wf_df['significant'].sum() >= len(wf_df) * 0.5)
        if any_sig:
            out('  WF: C15 attenuation produces SIG positive delta in >=50% of folds')
        else:
            out('  WF: C15 attenuation does NOT consistently produce significant positive delta')

    Path('reports/findings/regret_oracle/c15_pyramid_confidence_summary.txt').write_text(
        '\n'.join(lines), encoding='utf-8')
    print('\nWrote: reports/findings/regret_oracle/c15_pyramid_confidence_summary.txt')


if __name__ == '__main__':
    main()
