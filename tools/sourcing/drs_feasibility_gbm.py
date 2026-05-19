"""DRS Phase 1B — feasibility GBM on cross-day features vs peeky day-PnL.

This is the QUICK FEASIBILITY check. It answers: do the 15 cross-day
features have ANY predictive signal for day quality, before we spend
30-60 min running the canonical Path A forward pass?

Pipeline:
  1. Load cross_day_features.parquet (15 features, 293 days)
  2. Load day_pnl_proxy.parquet (peeky gbm_ev day pnl, 309 days)
  3. Join on date_label
  4. Date-disjoint 5-fold walk-forward CV (oldest -> newest)
  5. Train HistGradientBoostingRegressor on each fold
  6. Aggregate OOS predictions across folds, compute:
       Pearson R, R^2, MAE vs persistence baseline (yesterday's pnl)
  7. Bootstrap CI on OOS R
  8. Feature importance
  9. Write feasibility verdict

If OOS Pearson R > 0.20 with CI strictly positive, signal exists -> proceed
to Path A. If Pearson R near zero or negative, DRS hypothesis is weak on
V2-orthogonal features and we save the compute.

Output: reports/findings/drs/2026-05-17_feasibility_gbm.{txt,csv}
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = Path('DATA/CROSS_DAY/cross_day_features.parquet')
TARGETS  = Path('DATA/CROSS_DAY/day_pnl_proxy.parquet')

OUT_DIR  = Path('reports/findings/drs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT  = OUT_DIR / '2026-05-17_feasibility_gbm.txt'
OUT_CSV  = OUT_DIR / '2026-05-17_feasibility_gbm_preds.csv'

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


def walkforward_folds(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Date-ordered walk-forward CV. Each fold: train on past, test on next chunk.
    Returns list of (train_idx, test_idx)."""
    fold_size = n // (k + 1)
    folds = []
    for i in range(k):
        train_end = fold_size * (i + 1)
        test_end = min(n, train_end + fold_size)
        if test_end <= train_end:
            continue
        folds.append((np.arange(train_end), np.arange(train_end, test_end)))
    return folds


def bootstrap_ci(values: np.ndarray, stat_fn, n_boots: int = 4000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = []
    for _ in range(n_boots):
        idx = rng.integers(0, n, n)
        boots.append(stat_fn(values[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('DRS PHASE 1B — Cross-day features vs peeky day-PnL (feasibility)')
    out('=' * 78)

    print('Loading inputs...')
    feats = pd.read_parquet(FEATURES)
    tgts  = pd.read_parquet(TARGETS)
    out(f'cross_day_features: {len(feats)} days, {len(FEATURE_COLS)} feature cols')
    out(f'day_pnl_proxy:      {len(tgts)} days')

    df = feats.merge(tgts[['date_label', 'day_pnl_gbmev_peeky', 'day_pnl_flat_peeky',
                            'n_legs', 'source']],
                       on='date_label', how='inner', suffixes=('', '_y'))
    df = df.sort_values('date_label').reset_index(drop=True)
    out(f'After inner join:    {len(df)} days')

    # Drop rows with NaN in any required col
    needed = FEATURE_COLS + ['day_pnl_gbmev_peeky']
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    out(f'After dropping NaN:  {len(df)} days (dropped {before-len(df)})')

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['day_pnl_gbmev_peeky'].values.astype(np.float32)
    out(f'X shape: {X.shape}   y mean ${y.mean():+.0f}  median ${np.median(y):+.0f}  '
        f'min ${y.min():+.0f}  max ${y.max():+.0f}')

    folds = walkforward_folds(len(df), N_FOLDS)
    out(f'\nWalk-forward folds: {len(folds)}  ('
        f'each test ~{folds[0][1].shape[0]} days)')

    # Persistence baseline: predict yesterday's day_pnl for today
    y_prev = np.empty_like(y); y_prev[0] = np.nan
    y_prev[1:] = y[:-1]

    oos_pred = np.full(len(df), np.nan, dtype=np.float32)
    fold_results = []
    for fi, (tr, te) in enumerate(folds):
        m = HistGradientBoostingRegressor(
            loss='absolute_error',
            max_iter=200, learning_rate=0.05,
            max_depth=4, min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED + fi,
        )
        m.fit(X[tr], y[tr])
        pred_te = m.predict(X[te])
        oos_pred[te] = pred_te

        rho_te = float(np.corrcoef(pred_te, y[te])[0, 1])
        mae_te = float(np.mean(np.abs(pred_te - y[te])))
        out(f'  fold {fi+1}/{len(folds)}: train {len(tr):3d} days  '
            f'test {len(te):3d} days  Pearson={rho_te:+.3f}  MAE=${mae_te:.0f}')
        fold_results.append({'fold': fi+1, 'n_train': len(tr), 'n_test': len(te),
                              'pearson': rho_te, 'mae': mae_te})

    # Aggregate OOS metrics over all test points
    mask = ~np.isnan(oos_pred)
    pred_oos = oos_pred[mask]
    y_oos    = y[mask]
    y_prev_oos = y_prev[mask]

    rho_oos = float(np.corrcoef(pred_oos, y_oos)[0, 1])
    ss_res = float(np.sum((y_oos - pred_oos) ** 2))
    ss_tot = float(np.sum((y_oos - y_oos.mean()) ** 2))
    r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae_oos = float(np.mean(np.abs(pred_oos - y_oos)))

    # Persistence baseline
    mask_prev = mask & ~np.isnan(y_prev)
    mae_prev = float(np.mean(np.abs(y_prev[mask_prev] - y[mask_prev])))

    # Bootstrap CI on OOS Pearson
    def pearson_idx(idx_arr):
        if len(idx_arr) < 3:
            return 0.0
        a = pred_oos[idx_arr]; b = y_oos[idx_arr]
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(4000):
        idx = rng.integers(0, len(pred_oos), len(pred_oos))
        boots.append(pearson_idx(idx))
    boots = np.array(boots)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    out('')
    out('--- OOS aggregate (all folds) ---')
    out(f'  Pearson R:        {rho_oos:+.3f}   95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
    out(f'  R^2:              {r2_oos:+.3f}')
    out(f'  MAE (DRS):        ${mae_oos:.0f}')
    out(f'  MAE (persistence):${mae_prev:.0f}')
    if mae_prev > 0:
        out(f'  MAE lift:         {(1 - mae_oos/mae_prev)*100:+.1f}% vs persistence')

    # Feature importance from last fold
    out('')
    out('--- Feature importance (last fold, gain) ---')
    last_fi, (tr, te) = len(folds) - 1, folds[-1]
    m_last = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED + last_fi,
    )
    m_last.fit(X[tr], y[tr])
    # sklearn HistGBM doesn't expose feature_importances_ for absolute_error easily;
    # use permutation importance lite (one shuffle pass each)
    base_mae = float(np.mean(np.abs(m_last.predict(X[te]) - y[te])))
    importances = []
    rng_perm = np.random.default_rng(SEED + 999)
    for j, col in enumerate(FEATURE_COLS):
        Xp = X[te].copy()
        rng_perm.shuffle(Xp[:, j])
        mae_p = float(np.mean(np.abs(m_last.predict(Xp) - y[te])))
        importances.append((col, mae_p - base_mae))
    importances.sort(key=lambda r: -r[1])
    for col, delta in importances:
        bar = '#' * max(0, int(round(delta * 5)))
        out(f'  {col:<22}  delta_MAE={delta:+7.1f}  {bar}')

    # Verdict
    out('')
    out('=' * 78)
    out('VERDICT')
    out('=' * 78)
    if ci_lo > 0.20:
        out('  STRONG signal. OOS Pearson R > 0.20 with CI strictly positive.')
        out('  -> PROCEED to Path A (canonical hardened forward pass on IS).')
    elif ci_lo > 0:
        out('  WEAK but real signal. OOS Pearson R CI is positive but lower bound < 0.20.')
        out('  -> Marginal. Path A would give a more honest target — proceed if you have')
        out('     the compute budget; otherwise this is borderline.')
    else:
        out('  NO signal in cross-day features (vs peeky proxy).')
        out('  -> Hold on Path A. The DRS hypothesis is weak on the current feature set.')
        out('     Investigate: maybe the proxy is too noisy, or we need different features')
        out('     (e.g. ES gap, treasury yields, NQ vs ES divergence, news headline LLM score).')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {OUT_TXT}')

    # Save predictions for inspection
    df['oos_pred'] = oos_pred
    df['oos_resid'] = y - oos_pred
    df[['date_label', 'source', 'day_pnl_gbmev_peeky',
        'day_pnl_flat_peeky', 'n_legs', 'oos_pred', 'oos_resid',
        *FEATURE_COLS]].to_csv(OUT_CSV, index=False)
    print(f'Wrote: {OUT_CSV}')


if __name__ == '__main__':
    main()
