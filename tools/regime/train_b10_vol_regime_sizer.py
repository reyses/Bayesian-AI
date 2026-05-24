"""B10 -- INVERTED action: boost high-vol, cap low-vol days.

Vol-regime prediction is signal-rich (IS WF AUC 0.834, OOS AUC 0.949).
The defensive gate-high-vol action was backwards: zigzag wants vol.
This version flips the action surface.

Hyperparameter selection done on IS WALK-FORWARD only (not OOS).
Then sealed OOS test with the IS-selected params.

Action surface (size multiplier):
  - predicted P(high) >= thr_high:           1.5x  (boost)
  - predicted P(low)  >= thr_low (and not high):  0.7x  (cap)
  - otherwise:                                1.0x  (hold)

Params selected on IS WF maximizing mean delta/day:
  thr_high, thr_low chosen from IS WF predictions (no OOS peek)
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc',
    'dow',
]

BOOST = 1.3   # FIXED a-priori
CAP   = 0.7   # FIXED a-priori
N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 4000


def compute_intraday_vol(day):
    p = Path(f'DATA/ATLAS_NT8/1m/{day}.parquet')
    if not p.exists():
        p = Path(f'DATA/ATLAS/1m/{day}.parquet')
    if not p.exists():
        return float('nan')
    bars = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
    if len(bars) < 30:
        return float('nan')
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-legs',
                    default='reports/findings/regret_oracle/is_hardened_legs.csv',
                    help='per-day IS legs CSV (per-day P&L target source). '
                         'Default: hardened/offline IS. Pass causal_flat IS '
                         'legs to retrain on honest causal data.')
    args = ap.parse_args()
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('B10 -- INVERTED vol-regime sizer (boost high-vol, cap low-vol)')
    out('=' * 100)

    # Load data
    feats = pd.read_parquet('DATA/CROSS_DAY/cross_day_features.parquet').sort_values(
        'date_label').reset_index(drop=True)
    feats['intraday_vol'] = feats['date_label'].apply(compute_intraday_vol)
    df = feats.dropna(subset=FEATURE_COLS + ['intraday_vol']).reset_index(drop=True)
    is_df = df[df['source'] == 'ATLAS'].reset_index(drop=True)
    oos_df = df[df['source'] == 'NT8'].reset_index(drop=True)
    out(f'IS: {len(is_df)} days, OOS: {len(oos_df)} days')

    # Vol terciles from IS
    is_q33 = is_df['intraday_vol'].quantile(0.33)
    is_q67 = is_df['intraday_vol'].quantile(0.67)
    out(f'IS tercile thresholds: low<{is_q33:.4f}  high>={is_q67:.4f}')

    is_df['vol_high'] = (is_df['intraday_vol'] >= is_q67).astype(int)
    is_df['vol_low']  = (is_df['intraday_vol'] <  is_q33).astype(int)
    oos_df['vol_high'] = (oos_df['intraday_vol'] >= is_q67).astype(int)
    oos_df['vol_low']  = (oos_df['intraday_vol'] <  is_q33).astype(int)

    X_is = is_df[FEATURE_COLS].values.astype(np.float32)
    X_oos = oos_df[FEATURE_COLS].values.astype(np.float32)

    # === IS WF: generate predictions + tune action thresholds on IS only ===
    out('')
    out('=== IS walk-forward predictions ===')
    folds = walkforward_folds(len(is_df), N_FOLDS)
    is_p_high = np.full(len(is_df), np.nan, dtype=np.float32)
    is_p_low  = np.full(len(is_df), np.nan, dtype=np.float32)
    for fi, (tr, te) in enumerate(folds):
        m_h = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05,
                                              random_state=SEED + fi, l2_regularization=1.0)
        m_l = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05,
                                              random_state=SEED + fi + 100, l2_regularization=1.0)
        m_h.fit(X_is[tr], is_df['vol_high'].values[tr])
        m_l.fit(X_is[tr], is_df['vol_low'].values[tr])
        is_p_high[te] = m_h.predict_proba(X_is[te])[:, 1]
        is_p_low[te]  = m_l.predict_proba(X_is[te])[:, 1]
        out(f'  fold {fi+1}: val {len(te)} days')

    is_df['p_high'] = is_p_high
    is_df['p_low']  = is_p_low

    # Load IS per-day P&L (flat). Default hardened; --is-legs swaps in causal_flat.
    is_legs = pd.read_csv(args.is_legs)
    out(f'  IS legs source: {args.is_legs}  ({len(is_legs):,} legs)')
    is_day_pnl = is_legs.groupby('day')['pnl_usd'].sum().reset_index()
    is_day_pnl.columns = ['date_label', 'day_pnl']

    is_df = is_df.merge(is_day_pnl, on='date_label', how='left')
    valid = is_df['day_pnl'].notna() & is_df['p_high'].notna()
    out(f'  IS days with valid pred + P&L: {int(valid.sum())} / {len(is_df)}')

    # SELECT thresholds on IS WF (NOT OOS)
    # Pick thresholds that maximize mean delta on IS WF
    best_thr_high = None
    best_thr_low = None
    best_delta = -1e9
    out('')
    out('--- IS WF action-threshold tuning ---')
    out(f'{"thr_h":>5}  {"thr_l":>5}  {"n_boost":>7}  {"n_cap":>5}  {"delta":>7}  {"95% CI":>22}')
    for thr_h in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for thr_l in [0.50, 0.55, 0.60, 0.65, 0.70]:
            hi_act = is_df['p_high'].values >= thr_h
            lo_act = (is_df['p_low'].values >= thr_l) & ~hi_act
            size = np.ones(len(is_df))
            size[hi_act] = BOOST
            size[lo_act] = CAP
            delta = (size - 1.0) * is_df['day_pnl'].values
            delta_v = delta[valid.values]
            if len(delta_v) < 30:
                continue
            ci_lo, ci_hi = bootstrap_ci(delta_v)
            mean = float(delta_v.mean())
            n_b = int(hi_act[valid].sum())
            n_c = int(lo_act[valid].sum())
            out(f'  {thr_h:.2f}  {thr_l:.2f}    {n_b:>6}  {n_c:>5}  '
                f'${mean:>+5.0f}  [${ci_lo:+.0f}, ${ci_hi:+.0f}]')
            if mean > best_delta and ci_lo > 0:
                best_delta = mean
                best_thr_high = thr_h
                best_thr_low = thr_l

    out('')
    if best_thr_high is None:
        out('  No IS WF config produced sig positive delta. Falling back to 0.5/0.5.')
        best_thr_high = 0.5
        best_thr_low = 0.5
    out(f'SELECTED IS WF best: thr_high={best_thr_high}  thr_low={best_thr_low}  '
        f'delta ${best_delta:+.0f}/day')

    # === Train production models on FULL IS ===
    out('')
    m_high_prod = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05,
                                                   random_state=SEED, l2_regularization=1.0)
    m_high_prod.fit(X_is, is_df['vol_high'].values)
    m_low_prod = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05,
                                                  random_state=SEED + 100, l2_regularization=1.0)
    m_low_prod.fit(X_is, is_df['vol_low'].values)
    with open('reports/findings/regret_oracle/b10_vol_regime_high.pkl', 'wb') as f:
        pickle.dump({'model': m_high_prod, 'feature_cols': FEATURE_COLS,
                      'thr_high': best_thr_high, 'thr_low': best_thr_low,
                      'boost': BOOST, 'cap': CAP,
                      'is_q33': float(is_q33), 'is_q67': float(is_q67)}, f)
    with open('reports/findings/regret_oracle/b10_vol_regime_low.pkl', 'wb') as f:
        pickle.dump({'model': m_low_prod, 'feature_cols': FEATURE_COLS}, f)
    out('Production B10 models saved')

    # === SEALED OOS test with IS-selected thresholds ===
    out('')
    out('=== SEALED OOS test (IS-selected thresholds, single shot) ===')
    oos_df['p_high'] = m_high_prod.predict_proba(X_oos)[:, 1]
    oos_df['p_low']  = m_low_prod.predict_proba(X_oos)[:, 1]
    oos_pnl = pd.read_csv('reports/findings/drs/oos_extended_day_pnl.csv')
    merged = oos_df.merge(oos_pnl, left_on='date_label', right_on='day', how='left')
    valid_oos = merged['day_pnl_flat'].notna()
    day_pnl = merged['day_pnl_flat'].values

    hi_act = merged['p_high'].values >= best_thr_high
    lo_act = (merged['p_low'].values >= best_thr_low) & ~hi_act
    size = np.ones(len(merged))
    size[hi_act] = BOOST
    size[lo_act] = CAP
    delta = (size - 1.0) * day_pnl
    delta_v = delta[valid_oos]
    ci_lo, ci_hi = bootstrap_ci(delta_v)

    out(f'  thr_high={best_thr_high}  thr_low={best_thr_low}')
    out(f'  n_boost={int(hi_act[valid_oos].sum())}/{int(valid_oos.sum())}  '
        f'n_cap={int(lo_act[valid_oos].sum())}/{int(valid_oos.sum())}')
    out(f'  OOS delta: ${delta_v.mean():+.0f}/day   '
        f'95% CI [${ci_lo:+.0f}, ${ci_hi:+.0f}]   sig {ci_lo > 0}')
    out(f'  Baseline (no action): ${day_pnl[valid_oos].mean():+.0f}/day')
    out(f'  With B10: ${(day_pnl[valid_oos] + delta_v).mean():+.0f}/day')

    # Anti-doom slippage
    out('')
    out('--- Anti-doom slippage stress ---')
    action_mask = (size != 1.0)
    n_actions = int(action_mask[valid_oos].sum())
    for S in [0, 5, 10, 25]:
        delta_s = delta - np.where(action_mask, S, 0)
        d_v = delta_s[valid_oos]
        ci_lo_s, ci_hi_s = bootstrap_ci(d_v)
        out(f'  S=${S}: delta ${d_v.mean():+.0f}/day  '
            f'CI [${ci_lo_s:+.0f}, ${ci_hi_s:+.0f}]  '
            f'sig {ci_lo_s > 0}')
    out(f'  (n_actions = {n_actions})')

    # Show per-vol-bucket P&L for context
    out('')
    out('--- OOS per-actual-vol-regime P&L (verifies inversion premise) ---')
    merged['vol_actual'] = np.where(oos_df['vol_high'].values == 1, 'high',
                                      np.where(oos_df['vol_low'].values == 1, 'low', 'mid'))
    for lbl in ['low', 'mid', 'high']:
        sub = merged[(merged['vol_actual'] == lbl) & valid_oos]
        if len(sub) > 0:
            pnl = sub['day_pnl_flat'].values
            out(f'  vol={lbl:>4}  n={len(sub):>3}  '
                f'mean ${pnl.mean():+.0f}/day  '
                f'neg_days {int((pnl < 0).sum())}/{len(sub)}')

    Path('reports/findings/regret_oracle/b10_vol_regime_summary.txt').write_text(
        '\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: reports/findings/regret_oracle/b10_vol_regime_summary.txt')


if __name__ == '__main__':
    main()
