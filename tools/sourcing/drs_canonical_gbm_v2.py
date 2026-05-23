"""DRS dev GBM v2 -- trains on dev cross_day_features_with_target_v2.parquet,
with LLM news_intensity columns substituting for the dead binary event flags.

Identical training pipeline to canonical drs_canonical_gbm.py:
  - Same model class (HistGradientBoostingRegressor)
  - Same target (target_day_pnl = gbm_ev hardened daily $)
  - Same IS/OOS split (ATLAS=IS, NT8=OOS)
  - Same walk-forward CV (5 folds)
  - Same bootstrap CI methodology (4000 resamples)
  - Same operational sizing sim on OOS

ONLY differences from canonical:
  1. Input path: dev/cross_day_features_with_target_v2.parquet
  2. Output paths: dev/drs_canonical_gbm_v2.pkl, research/.../findings/*
  3. FEATURE_COLS: drops {is_fomc, is_cpi, is_nfp}, adds {news_intensity_today}
     (and news_intensity_prior in Phase B via --include-prior)

This intentionally mirrors canonical so the only signal driver in the
A/B comparison is the feature set.

Run:
  python tools/sourcing/drs_canonical_gbm_v2.py                       # Phase A
  python tools/sourcing/drs_canonical_gbm_v2.py --include-prior       # Phase B
"""
from __future__ import annotations
import argparse
import pickle
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES = Path('DATA/CROSS_DAY/dev/cross_day_features_with_target_v2.parquet')
OUT_FINDINGS_DIR = Path('research/llm_news_intensity/findings')
MODEL_OUT = Path('DATA/CROSS_DAY/dev/drs_canonical_gbm_v2.pkl')

# Phase A feature set: canonical 15 minus the 3 dead binary flags, plus 1 LLM col.
FEATURE_COLS_PHASE_A = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_opex',
    'days_since_fomc', 'days_to_next_fomc',
    'dow',
    'news_intensity_today',
]

# Phase B adds the prior-day-PM column.
FEATURE_COLS_PHASE_B = FEATURE_COLS_PHASE_A + ['news_intensity_prior']

# Canonical baseline (for diff reporting only)
CANONICAL_BASELINE_PEARSON_IS = (+0.191, +0.098, +0.405)   # point, lo, hi
CANONICAL_BASELINE_PEARSON_OOS = (+0.139, -0.047, +0.451)  # point, lo, hi
CANONICAL_BASELINE_OOS_PNL_PER_DAY = -333.0                # naive size, sealed OOS

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


def main(include_prior: bool = False) -> dict:
    phase = 'B' if include_prior else 'A'
    feature_cols = FEATURE_COLS_PHASE_B if include_prior else FEATURE_COLS_PHASE_A

    OUT_FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    today_iso = date_cls.today().isoformat()
    out_md = OUT_FINDINGS_DIR / f'{today_iso}_phase_{phase.lower()}_results.md'

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out(f'DRS DEV GBM v2 -- Phase {phase} (LLM news intensity feature)')
    out('=' * 78)
    out(f'Input:  {FEATURES}')
    out(f'Output: {MODEL_OUT}')
    out(f'Findings: {out_md}')
    out('')

    df = pd.read_parquet(FEATURES).sort_values('date_label').reset_index(drop=True)
    out(f'Total days in dataset: {len(df)}')
    out(f'  with target populated: {df["target_day_pnl"].notna().sum()}')
    out(f'  by source: ATLAS (IS) = {(df["source"] == "ATLAS").sum()}, '
        f'NT8 (OOS) = {(df["source"] == "NT8").sum()}')
    out(f'Feature columns ({len(feature_cols)}): {feature_cols}')
    out('')

    needed = feature_cols + ['target_day_pnl']
    df_full = df.dropna(subset=needed).reset_index(drop=True)
    out(f'After dropping NaN: {len(df_full)} usable days')

    is_df = df_full[df_full['source'] == 'ATLAS'].reset_index(drop=True)
    oos_df = df_full[df_full['source'] == 'NT8'].reset_index(drop=True)
    out(f'  IS  (ATLAS): {len(is_df)} days')
    out(f'  OOS (NT8):   {len(oos_df)} days')

    if len(is_df) < 50:
        raise RuntimeError(f'Insufficient IS days ({len(is_df)}) -- expected ~217. '
                           f'Check that the augmenter produced rows for all dates.')

    X_is = is_df[feature_cols].values.astype(np.float32)
    y_is = is_df['target_day_pnl'].values.astype(np.float32)
    out(f'  IS y: mean ${y_is.mean():+.0f}  median ${np.median(y_is):+.0f}  '
        f'min ${y_is.min():+.0f}  max ${y_is.max():+.0f}')

    # ---- News intensity column diagnostics ----
    out('')
    out('--- news_intensity column distribution ---')
    today_col = is_df['news_intensity_today'].values
    n_pos = int((today_col > 0).sum())
    out(f'  news_intensity_today: nonzero={n_pos}/{len(is_df)} '
        f'({100*n_pos/len(is_df):.1f}%)')
    if n_pos > 0:
        sub = today_col[today_col > 0]
        out(f'    on release days: mean={sub.mean():.2f}  std={sub.std():.2f}  '
            f'min={sub.min():.0f}  max={sub.max():.0f}')
    if include_prior:
        prior_col = is_df['news_intensity_prior'].values
        n_prior = int((prior_col > 0).sum())
        out(f'  news_intensity_prior: nonzero={n_prior}/{len(is_df)} '
            f'({100*n_prior/len(is_df):.1f}%)')
        if n_prior > 0:
            sub = prior_col[prior_col > 0]
            out(f'    on prior-release days: mean={sub.mean():.2f}  std={sub.std():.2f}  '
                f'min={sub.min():.0f}  max={sub.max():.0f}')

    # ---- Walk-forward CV within IS ----
    folds = walkforward_folds(len(is_df), N_FOLDS)
    out('')
    out(f'Walk-forward folds: {len(folds)}  '
        f'(test ~{folds[0][1].shape[0]} days each)')

    y_prev = np.empty_like(y_is); y_prev[0] = np.nan
    y_prev[1:] = y_is[:-1]

    oos_pred = np.full(len(is_df), np.nan, dtype=np.float32)
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

    mask = ~np.isnan(oos_pred)
    pred_oos = oos_pred[mask]
    y_oos = y_is[mask]
    y_prev_oos = y_prev[mask]
    rho_oos = float(np.corrcoef(pred_oos, y_oos)[0, 1])
    mae_oos = float(np.mean(np.abs(pred_oos - y_oos)))
    mask_prev = ~np.isnan(y_prev_oos)
    mae_prev = float(np.mean(np.abs(y_prev_oos[mask_prev] - y_oos[mask_prev]))) \
        if mask_prev.sum() > 0 else float('nan')

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
    out(f'    canonical baseline:  {CANONICAL_BASELINE_PEARSON_IS[0]:+.3f}   '
        f'95% CI [{CANONICAL_BASELINE_PEARSON_IS[1]:+.3f}, {CANONICAL_BASELINE_PEARSON_IS[2]:+.3f}]')
    out(f'    delta vs baseline:   {rho_oos - CANONICAL_BASELINE_PEARSON_IS[0]:+.3f}')
    out(f'  MAE (DRS):             ${mae_oos:.0f}')
    out(f'  MAE (persistence):     ${mae_prev:.0f}')
    if not np.isnan(mae_prev) and mae_prev > 0:
        lift = (1 - mae_oos / mae_prev) * 100
        out(f'  MAE lift vs persistence: {lift:+.1f}%')

    # ---- Train production model on full IS ----
    m_prod = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=200, learning_rate=0.05,
        max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    m_prod.fit(X_is, y_is)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({'model': m_prod, 'feature_cols': feature_cols,
                      'n_train': len(X_is),
                      'phase': phase,
                      'target': 'target_day_pnl (gbm_ev hardened daily $)',
                      }, f)
    out('')
    out(f'Trained on full IS ({len(X_is)} days). Model saved to {MODEL_OUT}')

    # ---- Sealed OOS test ----
    out('')
    out('=== SEALED OOS TEST (single-shot, NT8 days) ===')
    if len(oos_df) >= 5:
        X_oos = oos_df[feature_cols].values.astype(np.float32)
        y_oos_true = oos_df['target_day_pnl'].values.astype(np.float32)
        pred_oos_test = m_prod.predict(X_oos)
        rho_oos_test = (float(np.corrcoef(pred_oos_test, y_oos_true)[0, 1])
                          if np.std(pred_oos_test) > 0 and np.std(y_oos_true) > 0
                          else float('nan'))
        mae_oos_test = float(np.mean(np.abs(pred_oos_test - y_oos_true)))

        boots_oos = []
        rng_oos = np.random.default_rng(SEED + 1)
        for _ in range(N_BOOTSTRAP):
            idx = rng_oos.integers(0, len(pred_oos_test), len(pred_oos_test))
            a, b = pred_oos_test[idx], y_oos_true[idx]
            if np.std(a) > 0 and np.std(b) > 0:
                boots_oos.append(float(np.corrcoef(a, b)[0, 1]))
        boots_oos = np.array(boots_oos)
        oos_ci_lo = float(np.percentile(boots_oos, 2.5))
        oos_ci_hi = float(np.percentile(boots_oos, 97.5))

        out(f'  OOS days: {len(oos_df)}')
        out(f'  Pearson R (sealed):   {rho_oos_test:+.3f}   '
            f'95% CI [{oos_ci_lo:+.3f}, {oos_ci_hi:+.3f}]')
        out(f'    canonical baseline: {CANONICAL_BASELINE_PEARSON_OOS[0]:+.3f}   '
            f'95% CI [{CANONICAL_BASELINE_PEARSON_OOS[1]:+.3f}, {CANONICAL_BASELINE_PEARSON_OOS[2]:+.3f}]')
        out(f'    delta vs baseline:  {rho_oos_test - CANONICAL_BASELINE_PEARSON_OOS[0]:+.3f}')
        out(f'  MAE (DRS on OOS):     ${mae_oos_test:.0f}')

        # Operational sizing sim
        baseline_mean = float(y_is.mean())
        sizes = np.clip(pred_oos_test / max(baseline_mean, 1.0), 0.5, 1.5)
        weighted_pnl = sizes * y_oos_true
        flat_pnl = y_oos_true.sum()
        drs_pnl = weighted_pnl.sum()
        out('')
        out('  --- Operational sim: DRS as size multiplier on OOS ---')
        out(f'  Multiplier rule: clamp(pred / IS_mean, 0.5, 1.5)')
        out(f'  IS_mean baseline = ${baseline_mean:+.0f}')
        out(f'  Flat (no DRS): total ${flat_pnl:+.0f}, $/day ${flat_pnl/len(oos_df):+.0f}')
        out(f'  With DRS:      total ${drs_pnl:+.0f}, $/day ${drs_pnl/len(oos_df):+.0f}')
        out(f'    canonical baseline $/day delta: ${CANONICAL_BASELINE_OOS_PNL_PER_DAY:+.0f}')
        out(f'  Delta:         ${drs_pnl - flat_pnl:+.0f} total, '
            f'${(drs_pnl - flat_pnl)/len(oos_df):+.0f}/day')

        per_day_delta = (sizes - 1.0) * y_oos_true
        rng_d = np.random.default_rng(SEED + 2)
        delta_boots = np.array([per_day_delta[rng_d.integers(0, len(per_day_delta), len(per_day_delta))].mean()
                                  for _ in range(N_BOOTSTRAP)])
        d_lo = float(np.percentile(delta_boots, 2.5))
        d_hi = float(np.percentile(delta_boots, 97.5))
        out(f'  Per-day delta CI: [${d_lo:+.0f}, ${d_hi:+.0f}]   '
            f'sig: {d_lo > 0}')
    else:
        out(f'  Insufficient OOS days ({len(oos_df)}), skipping sealed test')
        rho_oos_test = float('nan')
        oos_ci_lo = oos_ci_hi = float('nan')

    # ---- Permutation importance on last fold ----
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
    for j, col in enumerate(feature_cols):
        Xp = X_is[te].copy()
        rng_perm.shuffle(Xp[:, j])
        mae_p = float(np.mean(np.abs(m_last.predict(Xp) - y_is[te])))
        importances.append((col, mae_p - base_mae))
    importances.sort(key=lambda r: -r[1])
    for col, delta in importances:
        bar = '#' * max(0, int(round(delta / 10)))
        annotation = ''
        if col == 'news_intensity_today':
            annotation = '  <-- NEW (Phase A)'
        elif col == 'news_intensity_prior':
            annotation = '  <-- NEW (Phase B)'
        out(f'  {col:<22}  delta_MAE={delta:+8.1f}  {bar}{annotation}')

    news_today_dmae = next((d for c, d in importances if c == 'news_intensity_today'), float('nan'))
    news_prior_dmae = next((d for c, d in importances if c == 'news_intensity_prior'), float('nan'))

    # ---- Phase A gate ----
    out('')
    out('=' * 78)
    out(f'PHASE {phase} GATE')
    out('=' * 78)

    is_pass = ci_lo >= CANONICAL_BASELINE_PEARSON_IS[1]
    oos_pass = (not np.isnan(oos_ci_lo)) and oos_ci_lo > 0.0
    imp_pass = (not np.isnan(news_today_dmae)) and news_today_dmae >= 30.0

    out(f'  Gate 1 (IS WF Pearson lower CI >= +0.098): {is_pass}  '
        f'(actual: {ci_lo:+.3f})')
    out(f'  Gate 2 (OOS sealed Pearson lower CI > 0):  {oos_pass}  '
        f'(actual: {oos_ci_lo:+.3f})')
    out(f'  Gate 3 (news_intensity_today dMAE >= +$30): {imp_pass}  '
        f'(actual: ${news_today_dmae:+.1f})')

    if is_pass and oos_pass and imp_pass:
        out('  >> ALL GATES PASS. Phase A signal is real. Proceed to Phase B (cycle_02.md).')
    elif is_pass and not oos_pass:
        out('  >> IS LIFT BUT OOS FAILS. LLM signal does not generalize. Likely memorization.')
        out('     KILL the feature OR iterate prompt + re-score.')
    elif not is_pass:
        out('  >> IS DID NOT LIFT. Prompt eng issue or model too small.')
        out('     Iterate prompt or upgrade model (Qwen2.5-14B Q4, Mistral-Small-24B Q4).')
    elif not imp_pass:
        out('  >> SIGNAL PRESENT BUT WEAK. Not worth the complexity.')
        out('     Consider deeper model or different scoring rubric.')

    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {out_md}')

    return {
        'phase': phase,
        'is_wf_pearson': rho_oos,
        'is_wf_ci_lo': ci_lo,
        'oos_sealed_pearson': rho_oos_test,
        'oos_sealed_ci_lo': oos_ci_lo,
        'news_today_dmae': news_today_dmae,
        'gate_pass': is_pass and oos_pass and imp_pass,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train dev DRS GBM with LLM news intensity column(s)')
    p.add_argument('--include-prior', action='store_true',
                   help='Phase B: include news_intensity_prior in feature set')
    args = p.parse_args()
    main(include_prior=args.include_prior)
