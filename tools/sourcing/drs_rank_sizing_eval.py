"""DRS rank-based sizing rule evaluation.

The naive clamp(pred / IS_mean, 0.5, 1.5) rule failed OOS because IS_mean
drifts between train and test periods. Rank-based sizing is robust:
  size = LOW + (HIGH - LOW) * rank_pct(pred_today_among_history)

Tests rank-based rule on:
  1. IS walk-forward predictions (proper, no peeking)
  2. OOS sealed predictions (single-shot)

Reports per-day delta CI for each.

Per-fold rank: when predicting day t in fold k, rank pred_t among the
training distribution of fold k (not OOS distribution -- that would peek).
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES = Path('DATA/CROSS_DAY/cross_day_features_with_target.parquet')
MODEL_PATH = Path('DATA/CROSS_DAY/drs_canonical_gbm.pkl')

OUT_TXT = Path('reports/findings/drs/2026-05-18_drs_rank_sizing.txt')
OUT_CSV = Path('reports/findings/drs/2026-05-18_drs_rank_sizing.csv')

FEATURE_COLS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc',
    'dow',
]

N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 4000

# Sizing rule grid (a-priori, no tuning)
SIZING_RULES = [
    ('rank_0.7_1.3', 0.7, 1.3),    # conservative
    ('rank_0.5_1.5', 0.5, 1.5),    # moderate
    ('rank_0.3_1.7', 0.3, 1.7),    # aggressive
    ('rank_0.0_2.0', 0.0, 2.0),    # very aggressive (skip + pyramid)
]


def walkforward_folds(n, k):
    fold_size = n // (k + 1)
    folds = []
    for i in range(k):
        train_end = fold_size * (i + 1)
        test_end = min(n, train_end + fold_size)
        if test_end <= train_end:
            continue
        folds.append((np.arange(train_end), np.arange(train_end, test_end)))
    return folds


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rank_pct(pred_value, train_preds):
    """Returns percentile rank of pred_value in train_preds [0, 1]."""
    if len(train_preds) == 0:
        return 0.5
    return float(np.mean(train_preds <= pred_value))


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('DRS RANK-BASED SIZING EVALUATION')
    out('=' * 78)
    out('Rule: size = LOW + (HIGH-LOW) * rank_pct(pred_today | train_preds)')
    out('Robust to baseline drift (IS_mean vs OOS_mean shift).')
    out('')

    df = pd.read_parquet(FEATURES).sort_values('date_label').reset_index(drop=True)
    is_df = df[(df['source'] == 'ATLAS') & df['target_day_pnl'].notna()].dropna(
        subset=FEATURE_COLS).reset_index(drop=True)
    oos_df = df[(df['source'] == 'NT8') & df['target_day_pnl'].notna()].dropna(
        subset=FEATURE_COLS).reset_index(drop=True)
    out(f'IS days: {len(is_df)}   OOS days: {len(oos_df)}')

    X_is = is_df[FEATURE_COLS].values.astype(np.float32)
    y_is = is_df['target_day_pnl'].values.astype(np.float32)

    # === IS walk-forward: collect (test_pred, test_actual, train_preds_at_fold) ===
    folds = walkforward_folds(len(is_df), N_FOLDS)
    wf_data = []   # list of (date, pred, actual, fold_train_preds)
    for fi, (tr, te) in enumerate(folds):
        m = HistGradientBoostingRegressor(
            loss='absolute_error',
            max_iter=200, learning_rate=0.05,
            max_depth=4, min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED + fi,
        )
        m.fit(X_is[tr], y_is[tr])
        train_preds = m.predict(X_is[tr])
        test_pred = m.predict(X_is[te])
        for j, t_idx in enumerate(te):
            wf_data.append({
                'date_label': is_df['date_label'].iloc[t_idx],
                'pred': float(test_pred[j]),
                'actual': float(y_is[t_idx]),
                'fold': fi + 1,
                'train_preds_min': float(train_preds.min()),
                'train_preds_max': float(train_preds.max()),
            })
    wf_df = pd.DataFrame(wf_data)

    # Per-fold rank: for each test prediction, compute percentile against THAT FOLD's train_preds
    # (we already stored train_preds_min/max but need full distribution)
    out('\nApplying rank-based sizing per fold (training distribution = ranker)...')
    # Recompute fold_train_preds for ranking
    fold_train_preds = {}
    for fi, (tr, te) in enumerate(folds):
        m = HistGradientBoostingRegressor(
            loss='absolute_error',
            max_iter=200, learning_rate=0.05,
            max_depth=4, min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED + fi,
        )
        m.fit(X_is[tr], y_is[tr])
        fold_train_preds[fi + 1] = m.predict(X_is[tr])

    wf_df['rank_pct'] = wf_df.apply(
        lambda r: rank_pct(r['pred'], fold_train_preds[r['fold']]), axis=1)

    # === Apply each sizing rule, compute per-day delta + CI ===
    results = []
    for name, LO, HI in SIZING_RULES:
        sizes = LO + (HI - LO) * wf_df['rank_pct'].values
        delta = (sizes - 1.0) * wf_df['actual'].values
        ci_lo, ci_hi = bootstrap_ci(delta)
        mean_delta = float(np.mean(delta))
        results.append({
            'rule': name, 'LO': LO, 'HI': HI,
            'mean_delta': mean_delta,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'sig': ci_lo > 0,
            'n_days': len(wf_df),
        })

    out('')
    out('=== IS WALK-FORWARD: rank-based rule grid ===')
    out(f'{"rule":<18}  {"LO":>5}  {"HI":>5}  '
        f'{"mean delta":>11}  {"95% CI":>22}  {"sig":>5}')
    for r in results:
        out(f'{r["rule"]:<18}  {r["LO"]:>5.2f}  {r["HI"]:>5.2f}  '
            f'${r["mean_delta"]:>+9.0f}    '
            f'[${r["ci_lo"]:>+5.0f}, ${r["ci_hi"]:>+5.0f}]    '
            f'{str(r["sig"]):>5}')

    # === Sealed OOS test: apply same rules using production-model preds ===
    out('')
    out('=== SEALED OOS TEST (single-shot, NT8 days) ===')
    with open(MODEL_PATH, 'rb') as f:
        prod = pickle.load(f)
    X_oos = oos_df[FEATURE_COLS].values.astype(np.float32)
    pred_oos = prod['model'].predict(X_oos)
    y_oos = oos_df['target_day_pnl'].values.astype(np.float32)

    # Use IS training-distribution as ranker (canonical: model trained on full IS)
    is_full_preds = prod['model'].predict(X_is)

    oos_ranks = np.array([rank_pct(p, is_full_preds) for p in pred_oos])
    out(f'OOS days: {len(oos_df)}')
    out(f'OOS pred mean ${pred_oos.mean():+.0f}  '
        f'rank_pct mean {oos_ranks.mean():.3f}  min {oos_ranks.min():.3f}  '
        f'max {oos_ranks.max():.3f}')

    oos_results = []
    for name, LO, HI in SIZING_RULES:
        sizes = LO + (HI - LO) * oos_ranks
        delta = (sizes - 1.0) * y_oos
        ci_lo, ci_hi = bootstrap_ci(delta)
        oos_results.append({
            'rule': name, 'LO': LO, 'HI': HI,
            'mean_delta': float(np.mean(delta)),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'sig': ci_lo > 0,
        })

    out('')
    out(f'{"rule":<18}  {"LO":>5}  {"HI":>5}  '
        f'{"mean delta":>11}  {"95% CI":>22}  {"sig":>5}')
    for r in oos_results:
        out(f'{r["rule"]:<18}  {r["LO"]:>5.2f}  {r["HI"]:>5.2f}  '
            f'${r["mean_delta"]:>+9.0f}    '
            f'[${r["ci_lo"]:>+5.0f}, ${r["ci_hi"]:>+5.0f}]    '
            f'{str(r["sig"]):>5}')

    # Save CSV
    output_rows = []
    for r in results:
        output_rows.append({'split': 'IS_WF', **r})
    for r in oos_results:
        output_rows.append({'split': 'OOS_sealed', **r})
    pd.DataFrame(output_rows).to_csv(OUT_CSV, index=False)

    out('')
    out('=' * 78)
    out('VERDICT')
    out('=' * 78)
    is_sig = any(r['sig'] for r in results)
    oos_sig = any(r['sig'] for r in oos_results)
    if is_sig and oos_sig:
        out('Rank-based rules show positive significant delta on BOTH IS WF and OOS.')
        out('DRS is deployable as a conservative rank-based size multiplier.')
        out('Recommend: lowest IS-significant config (most conservative bounds).')
    elif is_sig and not oos_sig:
        out('IS WF significant but OOS does not survive.')
        out('Likely regime change / overfit. Need more days or LLM-news features (Phase 2).')
    elif not is_sig and oos_sig:
        out('OOS significant but IS WF is not. Likely lucky on small OOS sample.')
        out('Treat with caution; do not deploy.')
    else:
        out('No rank-based rule shows significant positive delta on either split.')
        out('DRS hypothesis on current feature set is dead. Pivot to Phase 2 (LLM news)')
        out('or other orthogonal features.')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {OUT_TXT}')
    print(f'Wrote: {OUT_CSV}')


if __name__ == '__main__':
    main()
