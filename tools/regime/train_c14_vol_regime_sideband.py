"""C14 candidate -- vol-regime sideband day-level risk-budget gate.

Different LAYER from DRS: predicts intraday VOL (not P&L), action is
GATING the trading day's risk budget.

Target: tomorrow_intraday_vol_qtl (terciles: low / mid / high)
  Vol = std of 1m returns over RTH × sqrt(390) (annualized intraday)

Features: same cross-day set used by DRS
  overnight_gap_pct, overnight_range_pct, prior_day_range_pct,
  prior_day_c2c_pct, vix_close_prior, vix_chg_prior, dxy_close_prior,
  dxy_chg_prior, is_fomc, is_cpi, is_nfp, is_opex, days_since_fomc,
  days_to_next_fomc, dow

Action surface (different from DRS's continuous scaling):
  - predicted vol == HIGH: gate sizes 0.5x (defensive risk-cap)
  - predicted vol == MID:  hold 1.0x (no change)
  - predicted vol == LOW:  hold 1.0x (small uplift could be 1.1x)

The "gate" is BINARY (yes/no on the HIGH-vol days). Tests whether
cross-day features can identify the EXTREME tail of intraday vol
that we want to avoid.

Why this might work where DRS didn't:
  1. Vol is more autocorrelated than P&L (= more predictable)
  2. Action is gate (avoid bad), not scale (predict good)
  3. Tail-focused: we only care about the top quantile, not full ordering
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


FEATURE_COLS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc',
    'dow',
]

VOL_QUANTILES = [0.33, 0.67]   # tercile thresholds for HIGH/MID/LOW labels
GATE_SIZE = 0.5                 # size multiplier on HIGH-vol days
N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 4000


def compute_intraday_vol(day: str) -> float:
    """RTH intraday vol = std of 1m returns × sqrt(390) (~annualized intraday)."""
    p = Path(f'DATA/ATLAS_NT8/1m/{day}.parquet')
    if not p.exists():
        p = Path(f'DATA/ATLAS/1m/{day}.parquet')
    if not p.exists():
        return float('nan')
    bars = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
    if len(bars) < 30:
        return float('nan')
    # Log returns
    returns = np.log(bars['close'] / bars['close'].shift(1)).dropna()
    if len(returns) < 30:
        return float('nan')
    return float(returns.std() * np.sqrt(390))


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


def main():
    print('Loading cross-day features...')
    feats = pd.read_parquet('DATA/CROSS_DAY/cross_day_features.parquet')
    feats = feats.sort_values('date_label').reset_index(drop=True)

    print('Computing intraday vol per day (from 1m close returns)...')
    feats['intraday_vol'] = feats['date_label'].apply(compute_intraday_vol)
    n_valid = feats['intraday_vol'].notna().sum()
    print(f'  Days with valid vol: {n_valid} / {len(feats)}')

    # Drop NaN in any feature or vol
    needed = FEATURE_COLS + ['intraday_vol']
    df = feats.dropna(subset=needed).reset_index(drop=True)
    print(f'After NaN drop: {len(df)} days')

    # Split IS (ATLAS) / OOS (NT8)
    is_df = df[df['source'] == 'ATLAS'].reset_index(drop=True)
    oos_df = df[df['source'] == 'NT8'].reset_index(drop=True)
    print(f'  IS: {len(is_df)} days, OOS: {len(oos_df)} days')

    # Define tercile labels on IS vol
    is_q33 = is_df['intraday_vol'].quantile(VOL_QUANTILES[0])
    is_q67 = is_df['intraday_vol'].quantile(VOL_QUANTILES[1])
    print(f'  IS vol terciles: low<{is_q33:.4f}  mid<{is_q67:.4f}  high>={is_q67:.4f}')
    # Apply same thresholds to both IS and OOS
    df['vol_label'] = np.where(df['intraday_vol'] < is_q33, 'low',
                                 np.where(df['intraday_vol'] < is_q67, 'mid', 'high'))
    is_df['vol_label'] = np.where(is_df['intraday_vol'] < is_q33, 'low',
                                    np.where(is_df['intraday_vol'] < is_q67, 'mid', 'high'))
    oos_df['vol_label'] = np.where(oos_df['intraday_vol'] < is_q33, 'low',
                                    np.where(oos_df['intraday_vol'] < is_q67, 'mid', 'high'))

    print(f'  OOS vol distribution: '
          f'low {(oos_df["vol_label"]=="low").sum()}, '
          f'mid {(oos_df["vol_label"]=="mid").sum()}, '
          f'high {(oos_df["vol_label"]=="high").sum()}')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('C14 CANDIDATE -- vol-regime sideband (day-level risk-budget gate)')
    out('=' * 100)
    out(f'IS days: {len(is_df)}, OOS days: {len(oos_df)}')
    out(f'Tercile thresholds: low<{is_q33:.4f}  high>={is_q67:.4f}')
    out('')

    # === IS WF binary classifier: HIGH vol vs not-HIGH ===
    X = is_df[FEATURE_COLS].values.astype(np.float32)
    y_high = (is_df['vol_label'] == 'high').astype(int).values

    folds = walkforward_folds(len(is_df), N_FOLDS)
    fold_preds = np.full(len(is_df), np.nan, dtype=np.float32)
    fold_aucs = []
    for fi, (tr, te) in enumerate(folds):
        m = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            random_state=SEED + fi, l2_regularization=1.0,
        )
        m.fit(X[tr], y_high[tr])
        y_pred = m.predict_proba(X[te])[:, 1]
        fold_preds[te] = y_pred
        if y_high[te].sum() >= 2 and (y_high[te] == 0).sum() >= 2:
            auc = roc_auc_score(y_high[te], y_pred)
            fold_aucs.append(auc)
            out(f'  fold {fi+1}: AUC {auc:.3f}  (val n={len(te)}, pos {int(y_high[te].sum())})')

    # Aggregate
    mask = ~np.isnan(fold_preds)
    if mask.sum() >= 30 and y_high[mask].sum() >= 5:
        auc_agg = roc_auc_score(y_high[mask], fold_preds[mask])
        out(f'\nIS WF aggregate AUC: {auc_agg:.3f}  (mean fold AUC {np.mean(fold_aucs):.3f})')

    out('')
    # Train production C14 on full IS
    m_prod = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.05,
        random_state=SEED, l2_regularization=1.0,
    )
    m_prod.fit(X, y_high)
    with open('reports/findings/regret_oracle/c14_vol_regime_high.pkl', 'wb') as f:
        pickle.dump({'model': m_prod, 'feature_cols': FEATURE_COLS,
                      'is_vol_q33': float(is_q33), 'is_vol_q67': float(is_q67),
                      'n_train': len(X), 'n_pos': int(y_high.sum()),
                      'target': 'is_high_vol (RTH 1m return std × sqrt(390))'}, f)
    out(f'Production C14 model saved on {len(X)} IS days ({int(y_high.sum())} high-vol)')

    # === SEALED OOS test: predict HIGH-vol on OOS, apply gate ===
    if len(oos_df) >= 10:
        X_oos = oos_df[FEATURE_COLS].values.astype(np.float32)
        y_oos = (oos_df['vol_label'] == 'high').astype(int).values
        y_oos_pred = m_prod.predict_proba(X_oos)[:, 1]

        if y_oos.sum() >= 2 and (y_oos == 0).sum() >= 2:
            auc_oos = roc_auc_score(y_oos, y_oos_pred)
        else:
            auc_oos = float('nan')

        # Get per-day P&L from extended OOS
        oos_pnl = pd.read_csv('reports/findings/drs/oos_extended_day_pnl.csv')
        merged = oos_df[['date_label']].merge(oos_pnl, left_on='date_label',
                                                right_on='day', how='left')
        merged['c14_pred'] = y_oos_pred
        merged['vol_actual'] = oos_df['vol_label'].values

        out('')
        out('=== SEALED OOS TEST (51-day fresh dump) ===')
        out(f'OOS AUC for predicting HIGH-vol: {auc_oos:.3f}' if not np.isnan(auc_oos) else 'OOS AUC: NaN (insufficient class balance)')
        out(f'Actual high-vol days: {int(y_oos.sum())} / {len(y_oos)}')

        # Operational delta: gate sizes 0.5x on days where C14 predicts HIGH
        for gate_thr in [0.30, 0.50, 0.70]:
            gate = y_oos_pred >= gate_thr
            n_gated = int(gate.sum())
            # Delta vs flat: -0.5 × day_pnl on gated days (cutting 50% of position)
            day_pnl = merged['day_pnl_flat'].values
            valid = ~np.isnan(day_pnl)
            delta = np.where(gate & valid, -(1 - GATE_SIZE) * day_pnl, 0.0)
            mean_delta = delta[valid].mean() if valid.sum() > 0 else 0
            if valid.sum() >= 5:
                ci_lo, ci_hi = bootstrap_ci(delta[valid])
            else:
                ci_lo = ci_hi = float('nan')
            out(f'  gate_thr={gate_thr}: gated {n_gated}/{len(merged)} days  '
                f'delta ${mean_delta:+.0f}/day  CI [${ci_lo:+.0f}, ${ci_hi:+.0f}]  '
                f'sig {ci_lo > 0 if not np.isnan(ci_lo) else "n/a"}')

        # Verify: do HIGH-vol days actually correlate with worse P&L?
        out('')
        out('--- HIGH-vol days actual P&L analysis ---')
        for lbl in ['low', 'mid', 'high']:
            sub = merged[merged['vol_actual'] == lbl]
            if len(sub) > 0 and 'day_pnl_flat' in sub.columns:
                pnl_vals = sub['day_pnl_flat'].dropna()
                if len(pnl_vals) > 0:
                    out(f'  vol={lbl:>4}  n={len(pnl_vals)}  '
                        f'mean ${pnl_vals.mean():+.0f}/day  '
                        f'neg_days {int((pnl_vals < 0).sum())}')

    Path('reports/findings/regret_oracle/c14_vol_regime_summary.txt').write_text(
        '\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: reports/findings/regret_oracle/c14_vol_regime_summary.txt')


if __name__ == '__main__':
    main()
