"""DRS canonical GBM -- refit on REAL target_day_pnl (gbm_ev hardened).

Replaces Phase 1B's peeky proxy target with the canonical Path A target.
Same methodology: cross-day features (15 cols, V2-orthogonal), 5-fold
walk-forward CV within IS, bootstrap CI on Pearson, MAE vs persistence
baseline.

Adds OOS sealed test: train on all IS (217 days), predict on 23 OOS days,
report honest single-shot result.

Output: reports/findings/drs/2026-05-18_drs_canonical_gbm.{txt,csv}
Model: DATA/CROSS_DAY/drs_canonical_gbm.pkl (full-IS production model)
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES = Path('DATA/CROSS_DAY/cross_day_features_with_target.parquet')

OUT_DIR  = Path('reports/findings/drs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT  = OUT_DIR / '2026-05-18_drs_canonical_gbm.txt'
OUT_CSV  = OUT_DIR / '2026-05-18_drs_canonical_gbm_preds.csv'
MODEL_OUT = Path('DATA/CROSS_DAY/drs_canonical_gbm.pkl')

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


def walkforward_folds(n: int, k: int):
    fold_size = n // (k + 1)
    folds = []
    for i in range(k):
        train_end = fold_size * (i + 1)
        test_end = min(n, train_end + fold_size)
        if test_end <= train_end:
            continue
        folds.append((np.arange(train_end), np.arange(train_end, test_end)))
    return folds


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('DRS CANONICAL GBM -- refit on Path A real target (gbm_ev hardened)')
    out('=' * 78)

    print(f'Loading: {FEATURES}')
    df = pd.read_parquet(FEATURES).sort_values('date_label').reset_index(drop=True)
    out(f'Total days in dataset: {len(df)}')
    out(f'  with target populated: {df["target_day_pnl"].notna().sum()}')
    out(f'  by source: ATLAS (IS) = {(df["source"] == "ATLAS").sum()}, '
        f'NT8 (OOS) = {(df["source"] == "NT8").sum()}')

    # Drop rows with NaN in any required col
    needed = FEATURE_COLS + ['target_day_pnl']
    df_full = df.dropna(subset=needed).reset_index(drop=True)
    out(f'After dropping NaN: {len(df_full)} usable days')

    # Split IS (ATLAS) and OOS (NT8)
    is_df = df_full[df_full['source'] == 'ATLAS'].reset_index(drop=True)
    oos_df = df_full[df_full['source'] == 'NT8'].reset_index(drop=True)
    out(f'  IS  (ATLAS): {len(is_df)} days')
    out(f'  OOS (NT8):   {len(oos_df)} days')

    X_is = is_df[FEATURE_COLS].values.astype(np.float32)
    y_is = is_df['target_day_pnl'].values.astype(np.float32)
    out(f'  IS y: mean ${y_is.mean():+.0f}  median ${np.median(y_is):+.0f}  '
        f'min ${y_is.min():+.0f}  max ${y_is.max():+.0f}')

    # === Walk-forward CV within IS ===
    folds = walkforward_folds(len(is_df), N_FOLDS)
    out(f'\nWalk-forward folds: {len(folds)}  '
        f'(test ~{folds[0][1].shape[0]} days each)')

    # Persistence baseline within IS: predict yesterday's pnl
    y_prev = np.empty_like(y_is); y_prev[0] = np.nan
    y_prev[1:] = y_is[:-1]

    oos_pred = np.full(len(is_df), np.nan, dtype=np.float32)
    fold_results = []
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
        oos_pred[te] = pred_te
        if np.std(pred_te) > 0 and np.std(y_is[te]) > 0:
            rho_te = float(np.corrcoef(pred_te, y_is[te])[0, 1])
        else:
            rho_te = float('nan')
        mae_te = float(np.mean(np.abs(pred_te - y_is[te])))
        out(f'  fold {fi+1}/{len(folds)}: train {len(tr):3d} days  '
            f'test {len(te):3d} days  Pearson={rho_te:+.3f}  MAE=${mae_te:.0f}')
        fold_results.append({'fold': fi+1, 'n_train': len(tr), 'n_test': len(te),
                              'pearson': rho_te, 'mae': mae_te})

    # Aggregate OOS metrics over all test points
    mask = ~np.isnan(oos_pred)
    pred_oos = oos_pred[mask]
    y_oos = y_is[mask]
    y_prev_oos = y_prev[mask]

    rho_oos = float(np.corrcoef(pred_oos, y_oos)[0, 1])
    ss_res = float(np.sum((y_oos - pred_oos) ** 2))
    ss_tot = float(np.sum((y_oos - y_oos.mean()) ** 2))
    r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae_oos = float(np.mean(np.abs(pred_oos - y_oos)))

    mask_prev = ~np.isnan(y_prev_oos)
    if mask_prev.sum() > 0:
        mae_prev = float(np.mean(np.abs(y_prev_oos[mask_prev] - y_oos[mask_prev])))
    else:
        mae_prev = float('nan')

    # Bootstrap CI on aggregated Pearson
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(pred_oos), len(pred_oos))
        a, b = pred_oos[idx], y_oos[idx]
        if np.std(a) > 0 and np.std(b) > 0:
            boots.append(float(np.corrcoef(a, b)[0, 1]))
    boots = np.array(boots)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    out('')
    out('=== IS WALK-FORWARD AGGREGATE ===')
    out(f'  Pearson R (IS WF):     {rho_oos:+.3f}   95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
    out(f'  R^2:                   {r2_oos:+.3f}')
    out(f'  MAE (DRS):             ${mae_oos:.0f}')
    out(f'  MAE (persistence):     ${mae_prev:.0f}')
    if not np.isnan(mae_prev) and mae_prev > 0:
        lift = (1 - mae_oos / mae_prev) * 100
        out(f'  MAE lift vs persistence: {lift:+.1f}%')

    # === Train production model on full IS ===
    out('')
    out('=== Training production model on FULL IS ===')
    m_prod = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    m_prod.fit(X_is, y_is)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({'model': m_prod, 'feature_cols': FEATURE_COLS,
                      'n_train': len(X_is),
                      'target': 'target_day_pnl (gbm_ev hardened daily $)',
                      }, f)
    out(f'  Saved: {MODEL_OUT}')

    # === Sealed OOS test ===
    out('')
    out('=== SEALED OOS TEST (single-shot, NT8 days) ===')
    if len(oos_df) >= 5:
        X_oos = oos_df[FEATURE_COLS].values.astype(np.float32)
        y_oos_true = oos_df['target_day_pnl'].values.astype(np.float32)
        pred_oos_test = m_prod.predict(X_oos)
        if np.std(pred_oos_test) > 0 and np.std(y_oos_true) > 0:
            rho_oos_test = float(np.corrcoef(pred_oos_test, y_oos_true)[0, 1])
        else:
            rho_oos_test = float('nan')
        mae_oos_test = float(np.mean(np.abs(pred_oos_test - y_oos_true)))

        # OOS bootstrap CI
        boots_oos = []
        rng_oos = np.random.default_rng(SEED + 1)
        for _ in range(N_BOOTSTRAP):
            idx = rng_oos.integers(0, len(pred_oos_test), len(pred_oos_test))
            a, b = pred_oos_test[idx], y_oos_true[idx]
            if np.std(a) > 0 and np.std(b) > 0:
                boots_oos.append(float(np.corrcoef(a, b)[0, 1]))
        boots_oos = np.array(boots_oos)
        oos_ci_lo, oos_ci_hi = (float(np.percentile(boots_oos, 2.5)),
                                  float(np.percentile(boots_oos, 97.5)))
        out(f'  OOS days: {len(oos_df)}')
        out(f'  Pearson R (sealed):   {rho_oos_test:+.3f}   '
            f'95% CI [{oos_ci_lo:+.3f}, {oos_ci_hi:+.3f}]')
        out(f'  MAE (DRS on OOS):     ${mae_oos_test:.0f}')
        out(f'  OOS y: mean ${y_oos_true.mean():+.0f}  '
            f'median ${np.median(y_oos_true):+.0f}')
        out(f'  Pred:  mean ${pred_oos_test.mean():+.0f}  '
            f'median ${np.median(pred_oos_test):+.0f}')

        # Operational simulation: apply DRS as size multiplier on OOS
        # Multiplier = clamp(pred / mean_train, 0.5, 1.5) -- conservative scale
        baseline_mean = float(y_is.mean())
        sizes = np.clip(pred_oos_test / max(baseline_mean, 1.0), 0.5, 1.5)
        weighted_pnl = sizes * y_oos_true
        flat_pnl = y_oos_true.sum()
        drs_pnl = weighted_pnl.sum()
        out('')
        out('  --- Operational sim: DRS as size multiplier on OOS days ---')
        out(f'  Multiplier rule: clamp(pred / IS_mean, 0.5, 1.5)')
        out(f'  IS_mean baseline = ${baseline_mean:+.0f}')
        out(f'  Flat (no DRS): total ${flat_pnl:+.0f}, '
            f'$/day ${flat_pnl/len(oos_df):+.0f}')
        out(f'  With DRS:      total ${drs_pnl:+.0f}, '
            f'$/day ${drs_pnl/len(oos_df):+.0f}')
        out(f'  Delta:         ${drs_pnl - flat_pnl:+.0f} total, '
            f'${(drs_pnl - flat_pnl)/len(oos_df):+.0f}/day')

        # Per-day delta bootstrap CI
        per_day_delta = (sizes - 1.0) * y_oos_true
        rng_d = np.random.default_rng(SEED + 2)
        delta_boots = np.array([per_day_delta[rng_d.integers(0, len(per_day_delta), len(per_day_delta))].mean()
                                  for _ in range(N_BOOTSTRAP)])
        d_lo, d_hi = float(np.percentile(delta_boots, 2.5)), float(np.percentile(delta_boots, 97.5))
        out(f'  Per-day delta CI: [${d_lo:+.0f}, ${d_hi:+.0f}]   '
            f'sig: {d_lo > 0}')

        oos_preds_df = pd.DataFrame({
            'date_label': oos_df['date_label'].values,
            'pred_day_pnl': pred_oos_test,
            'actual_day_pnl': y_oos_true,
            'size_multiplier': sizes,
            'flat_weighted': y_oos_true,
            'drs_weighted': weighted_pnl,
        })
    else:
        out(f'  Insufficient OOS days ({len(oos_df)}), skipping sealed test')
        oos_preds_df = pd.DataFrame()

    # === Feature importance via permutation on last fold ===
    out('')
    out('--- Feature importance (permutation, last fold) ---')
    last_fi = len(folds) - 1
    tr, te = folds[-1]
    m_last = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED + last_fi,
    )
    m_last.fit(X_is[tr], y_is[tr])
    base_mae = float(np.mean(np.abs(m_last.predict(X_is[te]) - y_is[te])))
    importances = []
    rng_perm = np.random.default_rng(SEED + 999)
    for j, col in enumerate(FEATURE_COLS):
        Xp = X_is[te].copy()
        rng_perm.shuffle(Xp[:, j])
        mae_p = float(np.mean(np.abs(m_last.predict(Xp) - y_is[te])))
        importances.append((col, mae_p - base_mae))
    importances.sort(key=lambda r: -r[1])
    for col, delta in importances:
        bar = '#' * max(0, int(round(delta / 10)))
        out(f'  {col:<22}  delta_MAE={delta:+8.1f}  {bar}')

    # === Verdict ===
    out('')
    out('=' * 78)
    out('VERDICT')
    out('=' * 78)
    deploy = False
    if rho_oos >= 0.20 and ci_lo > 0.10:
        out(f'  IS walk-forward Pearson {rho_oos:+.3f} CI [{ci_lo:+.3f},{ci_hi:+.3f}]')
        out(f'  STRONG signal -> DRS is deployment-grade.')
        deploy = True
    elif rho_oos > 0 and ci_lo > 0:
        out(f'  IS walk-forward Pearson {rho_oos:+.3f} CI [{ci_lo:+.3f},{ci_hi:+.3f}]')
        out(f'  WEAK but positive signal -> proceed with conservative multiplier bounds.')
        deploy = True
    elif rho_oos > 0:
        out(f'  IS walk-forward Pearson {rho_oos:+.3f} CI [{ci_lo:+.3f},{ci_hi:+.3f}]')
        out(f'  CI crosses zero -> marginal. Need more data or feature engineering.')
    else:
        out(f'  IS walk-forward Pearson {rho_oos:+.3f}.')
        out(f'  KILL DRS hypothesis: cross-day features have no signal on real target.')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    if not oos_preds_df.empty:
        oos_preds_df.to_csv(OUT_CSV, index=False)
    print(f'\nWrote: {OUT_TXT}')
    if not oos_preds_df.empty:
        print(f'Wrote: {OUT_CSV}')


if __name__ == '__main__':
    main()
