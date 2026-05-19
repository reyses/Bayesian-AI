"""DRS evaluation on the EXTENDED OOS (57 days = original 23 + 34 new).

Uses FLAT-sized hardened P&L throughout (no gbm_ev because B7 requires
V2 features not yet computed for May days). For apples-to-apples,
re-aggregates IS target as flat from is_hardened_legs.csv.

Compared to the original 23-day sealed test:
  Pearson improvement?
  CI tightening?
  Rank-sizing rule profitability?

Caveat: target SCALE differs from prior gbm_ev evaluation. The DRS GBM
must be REFIT on flat target for valid comparison.

Output:
  reports/findings/drs/2026-05-18_drs_extended_oos.txt
  reports/findings/drs/2026-05-18_drs_extended_oos.csv
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES_PATH = Path('DATA/CROSS_DAY/cross_day_features.parquet')
IS_LEGS_PATH = Path('reports/findings/regret_oracle/is_hardened_legs.csv')
OOS_EXT_PATH = Path('reports/findings/drs/oos_extended_day_pnl.csv')

OUT_TXT = Path('reports/findings/drs/2026-05-18_drs_extended_oos.txt')
OUT_CSV = Path('reports/findings/drs/2026-05-18_drs_extended_oos.csv')

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

SIZING_RULES = [
    ('rank_0.7_1.3', 0.7, 1.3),
    ('rank_0.5_1.5', 0.5, 1.5),
    ('rank_0.3_1.7', 0.3, 1.7),
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
    if len(train_preds) == 0:
        return 0.5
    return float(np.mean(train_preds <= pred_value))


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 90)
    out('DRS EXTENDED OOS EVAL (57-day OOS, flat target throughout)')
    out('=' * 90)

    # Build flat-target IS day-pnl
    print('Aggregating IS flat target...')
    is_legs = pd.read_csv(IS_LEGS_PATH)
    is_day_pnl = is_legs.groupby('day')['pnl_usd'].sum().reset_index()
    is_day_pnl.columns = ['date_label', 'day_pnl_flat']
    is_day_pnl['source'] = 'ATLAS'

    # Load extended OOS day-pnl
    oos_day_pnl = pd.read_csv(OOS_EXT_PATH)
    oos_day_pnl.columns = ['date_label', 'day_pnl_flat', 'n_legs']
    oos_day_pnl = oos_day_pnl[['date_label', 'day_pnl_flat']].copy()
    oos_day_pnl['source'] = 'NT8'

    # Combine
    all_day_pnl = pd.concat([is_day_pnl, oos_day_pnl], ignore_index=True)

    # Load cross-day features, join
    feats = pd.read_parquet(FEATURES_PATH)
    df = feats.merge(all_day_pnl[['date_label', 'day_pnl_flat']],
                       on='date_label', how='left')

    # Sort by date
    df = df.sort_values('date_label').reset_index(drop=True)
    out(f'Total cross-day rows: {len(df)}')

    # Drop rows with NaN in any required col
    needed = FEATURE_COLS + ['day_pnl_flat']
    df_full = df.dropna(subset=needed).reset_index(drop=True)
    out(f'After dropping NaN: {len(df_full)} usable days')

    is_df = df_full[df_full['source'] == 'ATLAS'].reset_index(drop=True)
    oos_df = df_full[df_full['source'] == 'NT8'].reset_index(drop=True)
    out(f'  IS  (ATLAS): {len(is_df)} days, mean ${is_df["day_pnl_flat"].mean():+.0f}, '
        f'median ${is_df["day_pnl_flat"].median():+.0f}')
    out(f'  OOS (NT8):   {len(oos_df)} days, mean ${oos_df["day_pnl_flat"].mean():+.0f}, '
        f'median ${oos_df["day_pnl_flat"].median():+.0f}')
    out(f'  OOS range:   {oos_df["date_label"].min()} to {oos_df["date_label"].max()}')
    out(f'  OOS negative days: {(oos_df["day_pnl_flat"] < 0).sum()} / {len(oos_df)} '
        f'({(oos_df["day_pnl_flat"] < 0).mean()*100:.1f}%)')

    X_is = is_df[FEATURE_COLS].values.astype(np.float32)
    y_is = is_df['day_pnl_flat'].values.astype(np.float32)

    # === IS walk-forward CV ===
    folds = walkforward_folds(len(is_df), N_FOLDS)
    out(f'\nIS walk-forward folds: {len(folds)}')

    oos_pred_wf = np.full(len(is_df), np.nan, dtype=np.float32)
    for fi, (tr, te) in enumerate(folds):
        m = HistGradientBoostingRegressor(
            loss='absolute_error',
            max_iter=200, learning_rate=0.05,
            max_depth=4, min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED + fi,
        )
        m.fit(X_is[tr], y_is[tr])
        pred_te = m.predict(X_is[te])
        oos_pred_wf[te] = pred_te
        if np.std(pred_te) > 0 and np.std(y_is[te]) > 0:
            rho_te = float(np.corrcoef(pred_te, y_is[te])[0, 1])
        else:
            rho_te = float('nan')
        out(f'  fold {fi+1}: Pearson={rho_te:+.3f}  MAE=${np.mean(np.abs(pred_te-y_is[te])):.0f}')

    mask = ~np.isnan(oos_pred_wf)
    pred_oos_wf = oos_pred_wf[mask]
    y_oos_wf = y_is[mask]
    rho_oos = float(np.corrcoef(pred_oos_wf, y_oos_wf)[0, 1])
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(pred_oos_wf), len(pred_oos_wf))
        a, b = pred_oos_wf[idx], y_oos_wf[idx]
        if np.std(a) > 0 and np.std(b) > 0:
            boots.append(float(np.corrcoef(a, b)[0, 1]))
    boots = np.array(boots)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    out(f'\nIS aggregate Pearson: {rho_oos:+.3f}  CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')

    # === Train production model on FULL IS (flat target) ===
    m_prod = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    m_prod.fit(X_is, y_is)
    out('\nProduction model trained on full IS (flat target)')

    # === SEALED OOS test on extended sample ===
    out('')
    out('=== SEALED OOS TEST (extended 57-day sample) ===')
    X_oos = oos_df[FEATURE_COLS].values.astype(np.float32)
    pred_oos = m_prod.predict(X_oos)
    y_oos = oos_df['day_pnl_flat'].values.astype(np.float32)

    rho_oos_test = float(np.corrcoef(pred_oos, y_oos)[0, 1])
    mae_oos_test = float(np.mean(np.abs(pred_oos - y_oos)))

    rng_oos = np.random.default_rng(SEED + 1)
    boots_oos = []
    for _ in range(N_BOOTSTRAP):
        idx = rng_oos.integers(0, len(pred_oos), len(pred_oos))
        a, b = pred_oos[idx], y_oos[idx]
        if np.std(a) > 0 and np.std(b) > 0:
            boots_oos.append(float(np.corrcoef(a, b)[0, 1]))
    boots_oos = np.array(boots_oos)
    oos_ci_lo, oos_ci_hi = (float(np.percentile(boots_oos, 2.5)),
                              float(np.percentile(boots_oos, 97.5)))
    out(f'OOS days: {len(oos_df)}')
    out(f'OOS Pearson R (sealed): {rho_oos_test:+.3f}  '
        f'95% CI [{oos_ci_lo:+.3f}, {oos_ci_hi:+.3f}]')
    out(f'OOS MAE: ${mae_oos_test:.0f}')
    out(f'Pred mean: ${pred_oos.mean():+.0f}  actual mean: ${y_oos.mean():+.0f}')

    # Anti-predictive check on negative days
    neg_mask = y_oos < 0
    if neg_mask.sum() > 0:
        out(f'\nNegative-day analysis ({int(neg_mask.sum())} days):')
        out(f'  Mean pred on negative days: ${pred_oos[neg_mask].mean():+.0f}')
        out(f'  Mean pred on positive days: ${pred_oos[~neg_mask].mean():+.0f}')
        out(f'  Anti-predictive? '
            f'{"YES" if pred_oos[neg_mask].mean() > pred_oos[~neg_mask].mean() else "no"}')

    # === Rank-based sizing on extended OOS ===
    out('')
    out('=== Rank-based sizing rules on extended OOS ===')
    # Use full-IS training preds as ranker
    is_full_preds = m_prod.predict(X_is)
    oos_ranks = np.array([rank_pct(p, is_full_preds) for p in pred_oos])
    out(f'OOS rank_pct: mean {oos_ranks.mean():.3f}  min {oos_ranks.min():.3f}  '
        f'max {oos_ranks.max():.3f}')

    out(f'\n{"rule":<18}  {"LO":>5}  {"HI":>5}  {"mean delta/day":>11}  '
        f'{"95% CI":>22}  {"sig":>5}')
    sizing_results = []
    for name, LO, HI in SIZING_RULES:
        sizes = LO + (HI - LO) * oos_ranks
        delta = (sizes - 1.0) * y_oos
        ci_lo_s, ci_hi_s = bootstrap_ci(delta)
        sizing_results.append({
            'rule': name, 'LO': LO, 'HI': HI,
            'mean_delta': float(np.mean(delta)),
            'ci_lo': ci_lo_s, 'ci_hi': ci_hi_s,
            'sig': ci_lo_s > 0,
        })
        out(f'{name:<18}  {LO:>5.2f}  {HI:>5.2f}  '
            f'${np.mean(delta):>+9.0f}    '
            f'[${ci_lo_s:>+5.0f}, ${ci_hi_s:>+5.0f}]    '
            f'{str(ci_lo_s > 0):>5}')

    # Save predictions CSV
    preds_df = pd.DataFrame({
        'date_label': oos_df['date_label'].values,
        'pred_day_pnl': pred_oos,
        'actual_day_pnl': y_oos,
        'rank_pct': oos_ranks,
    })
    preds_df.to_csv(OUT_CSV, index=False)

    out('')
    out('=' * 90)
    out('VERDICT vs original 23-day OOS test (2026-05-18 morning)')
    out('=' * 90)
    out('Prior (23-day OOS, gbm_ev target):')
    out('  Pearson +0.139  CI [-0.047, +0.451]  NOT significant')
    out('  rank_0.7_1.3 sizing: -$20/day  CI [-68, +34]  not sig')
    out('')
    out(f'Now (57-day OOS, flat target):')
    out(f'  Pearson {rho_oos_test:+.3f}  CI [{oos_ci_lo:+.3f}, {oos_ci_hi:+.3f}]   '
        f'{"SIGNIFICANT" if oos_ci_lo > 0 else "not significant"}')
    for r in sizing_results:
        out(f'  {r["rule"]} sizing: ${r["mean_delta"]:+.0f}/day  '
            f'CI [${r["ci_lo"]:+.0f}, ${r["ci_hi"]:+.0f}]   '
            f'{"SIGNIFICANT" if r["sig"] else "not significant"}')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {OUT_TXT}')
    print(f'Wrote: {OUT_CSV}')


if __name__ == '__main__':
    main()
