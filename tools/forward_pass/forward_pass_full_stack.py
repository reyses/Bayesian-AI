"""Full-stack forward pass: B7 (entry) + B9 (during-trade) + B10 (day) + flat baseline.

For each leg:
  entry_size = B7_size_factor(V2 at R-trigger) * B10_day_multiplier(cross-day features)
  size_to_K5 = entry_size
  size_after_K5 = entry_size * B9_size_factor(V2 + trajectory at K=5)

Realized P&L per leg:
  pnl_pts_to_K5     = (close_at_K5 - entry_close) * leg_dir
  pnl_pts_after_K5  = pnl_pts_total - pnl_pts_to_K5

  realized_pnl_pts  = size_to_K5 * pnl_pts_to_K5
                    + size_after_K5 * pnl_pts_after_K5

  realized_pnl_usd  = realized_pnl_pts * $2 - friction_per_unit * size_avg
                    (approximated as pnl_usd_baseline * size_total for simplicity;
                     under-counts mid-trade-action friction by ~$6/cut-leg)

Compares schemes: FLAT, B7-only, B7+B9, B7+B10, B7+B9+B10, B9-only, B10-only.

Outputs:
  reports/findings/regret_oracle/forward_pass_stack_IS.txt   (with WF caveat)
  reports/findings/regret_oracle/forward_pass_stack_OOS.txt  (sealed)
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


B7_PKL = 'reports/findings/regret_oracle/b7_leg_sizer.pkl'
B9_PKL = 'reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl'
B10_HIGH_PKL = 'reports/findings/regret_oracle/b10_vol_regime_high.pkl'
B10_LOW_PKL  = 'reports/findings/regret_oracle/b10_vol_regime_low.pkl'

CROSS_DAY_FEATURES = 'DATA/CROSS_DAY/cross_day_features.parquet'

# B10 action thresholds (selected from IS WF)
B10_THR_HIGH = 0.5
B10_THR_LOW  = 0.7
B10_BOOST = 1.3
B10_CAP   = 0.7

# B10 cross-day feature columns
B10_FEATS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc', 'dow',
]


def b7_size(pred_amp_R: float) -> float:
    """gbm_ev: clip(max(pred_R - 1, 0), 0, 3)"""
    return float(np.clip(max(pred_amp_R - 1.0, 0.0), 0.0, 3.0))


def b9_size(pred_remaining: float) -> float:
    """B9 sizing rule (matches train_b9_remaining_amplitude.py)."""
    if pred_remaining > 50:   return 1.5
    if pred_remaining > 10:   return 1.0
    if pred_remaining > -10:  return 1.0
    if pred_remaining > -50:  return 0.5
    return 0.0


def b10_day_mult(p_high: float, p_low: float) -> float:
    """B10 day-level multiplier (mutually exclusive: high beats low)."""
    if p_high >= B10_THR_HIGH:
        return B10_BOOST
    if p_low >= B10_THR_LOW:
        return B10_CAP
    return 1.0


def bootstrap_ci(values, n_boot=4000, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_forward_pass(label, legs_path, trajectory_path, truth_path,
                       source_filter, output_path):
    """Run full-stack forward pass on a dataset.

    legs_path: hardened leg CSV
    trajectory_path: per-leg-K trajectory parquet (V2 at K bar)
    truth_path: pivot dataset (V2 at R-trigger bar)
    source_filter: 'ATLAS' for IS, 'NT8' for OOS
    output_path: report destination
    """
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out(f'FULL-STACK FORWARD PASS: {label}')
    out('=' * 100)

    print(f'\nLoading models...')
    with open(B7_PKL, 'rb') as f:
        b7 = pickle.load(f)
    with open(B9_PKL, 'rb') as f:
        b9 = pickle.load(f)
    with open(B10_HIGH_PKL, 'rb') as f:
        b10_high = pickle.load(f)
    with open(B10_LOW_PKL, 'rb') as f:
        b10_low = pickle.load(f)
    print(f'B7 v2_cols: {len(b7["v2_cols"])}')
    print(f'B9 feat_cols: {len(b9["feat_cols"])}')

    print(f'\nLoading data...')
    legs = pd.read_csv(legs_path)
    legs = legs.sort_values(['day', 'entry_ts']).reset_index(drop=True)
    legs['leg_idx'] = legs.index   # stable index
    print(f'  legs: {len(legs):,}, days: {legs["day"].nunique()}')

    truth = pd.read_parquet(truth_path)
    print(f'  truth rows: {len(truth):,}')

    traj_full = pd.read_parquet(trajectory_path)
    traj_k5 = traj_full[traj_full['K'] == 5].copy()
    print(f'  trajectory K=5: {len(traj_k5):,}')

    feats = pd.read_parquet(CROSS_DAY_FEATURES)
    feats = feats[feats['source'] == source_filter].copy()
    print(f'  cross-day features ({source_filter}): {len(feats)} days')

    # === B7 predictions: V2 at R-trigger fire bar ===
    print('\n[B7] generating predictions at R-trigger bar...')
    truth_sorted = truth.sort_values(['day', 'timestamp']).reset_index(drop=True)
    day_truth = {day: g.reset_index(drop=True) for day, g in truth_sorted.groupby('day')}
    b7_preds = []
    skipped_b7 = 0
    for _, leg in legs.iterrows():
        day = leg['day']
        if day not in day_truth:
            b7_preds.append(np.nan)
            skipped_b7 += 1
            continue
        td = day_truth[day]
        i_feat = int(np.searchsorted(td['timestamp'].values, leg['entry_ts'],
                                       side='right') - 1)
        if i_feat < 0 or i_feat >= len(td):
            b7_preds.append(np.nan)
            skipped_b7 += 1
            continue
        feat_row = td.iloc[i_feat]
        X = np.array([float(feat_row[c]) if not pd.isna(feat_row[c]) else 0.0
                       for c in b7['v2_cols']], dtype=np.float32).reshape(1, -1)
        pred = float(b7['model'].predict(X)[0])
        b7_preds.append(pred)
    legs['b7_pred_amp_R'] = b7_preds
    legs['b7_size'] = legs['b7_pred_amp_R'].apply(
        lambda p: b7_size(p) if not pd.isna(p) else 1.0)
    print(f'  B7 skipped (no truth match): {skipped_b7}')

    # === B9 predictions: from trajectory at K=5 ===
    print('\n[B9] joining K=5 predictions...')
    X_traj = traj_k5[b9['feat_cols']].fillna(0.0).values
    traj_k5['b9_pred'] = b9['model'].predict(X_traj)
    traj_k5['b9_size'] = traj_k5['b9_pred'].apply(b9_size)
    # Join back to legs by leg_id (which is the index in IS, leg_id col in OOS_full)
    # In trade_trajectory_IS.parquet, leg_id matches is_hardened_legs.csv index
    # In trade_trajectory_OOS_full.parquet, leg_id matches oos_hardened_legs_full.csv index
    legs_with_b9 = legs.merge(
        traj_k5[['leg_id', 'b9_size', 'pnl_usd_so_far']],
        left_on='leg_idx', right_on='leg_id', how='left',
    )
    print(f'  B9 matched legs: {legs_with_b9["b9_size"].notna().sum()}')
    legs_with_b9['b9_size'] = legs_with_b9['b9_size'].fillna(1.0)
    legs_with_b9['pnl_usd_so_far'] = legs_with_b9['pnl_usd_so_far'].fillna(
        legs_with_b9['pnl_usd'] * 0)  # if missing, treat as 0 progress

    # === B10 predictions: per day ===
    print('\n[B10] generating day-level multipliers...')
    X_feat = feats[B10_FEATS].fillna(0.0).values.astype(np.float32)
    feats['p_high'] = b10_high['model'].predict_proba(X_feat)[:, 1]
    feats['p_low']  = b10_low['model'].predict_proba(X_feat)[:, 1]
    feats['b10_day_mult'] = feats.apply(
        lambda r: b10_day_mult(r['p_high'], r['p_low']), axis=1)
    day_to_mult = dict(zip(feats['date_label'], feats['b10_day_mult']))
    legs_with_b9['b10_day_mult'] = legs_with_b9['day'].map(day_to_mult).fillna(1.0)
    print(f'  B10 days mapped: {legs_with_b9["b10_day_mult"].notna().sum()}')
    n_boost = (legs_with_b9['b10_day_mult'] == B10_BOOST).sum()
    n_cap   = (legs_with_b9['b10_day_mult'] == B10_CAP).sum()
    out(f'B10 boost days: {n_boost} legs / cap days: {n_cap} legs / hold: '
        f'{len(legs_with_b9) - n_boost - n_cap} legs')

    # === Compose schemes and compute realized P&L ===
    print('\nComputing realized P&L per scheme...')

    pnl_total = legs_with_b9['pnl_usd'].values  # already has $6 friction
    pnl_to_K5 = legs_with_b9['pnl_usd_so_far'].values
    pnl_after_K5 = pnl_total - pnl_to_K5

    b7_sz = legs_with_b9['b7_size'].fillna(1.0).values
    b9_sz = legs_with_b9['b9_size'].values
    b10_mult = legs_with_b9['b10_day_mult'].values

    schemes = {}
    # 1. FLAT (1.0 throughout)
    schemes['FLAT (baseline)'] = pnl_total * 1.0
    # 2. B7 only (entry sizing)
    schemes['B7 only'] = pnl_total * b7_sz
    # 3. B9 only (during-trade at unit base)
    schemes['B9 only'] = pnl_to_K5 * 1.0 + pnl_after_K5 * b9_sz
    # 4. B10 only (day-level at unit base)
    schemes['B10 only'] = pnl_total * b10_mult
    # 5. B7 + B9
    entry_b7 = b7_sz
    schemes['B7 + B9'] = pnl_to_K5 * entry_b7 + pnl_after_K5 * entry_b7 * b9_sz
    # 6. B7 + B10
    entry_b7b10 = b7_sz * b10_mult
    schemes['B7 + B10'] = pnl_total * entry_b7b10
    # 7. FULL STACK: B7 + B9 + B10
    schemes['B7 + B9 + B10 (full stack)'] = (
        pnl_to_K5 * entry_b7b10 + pnl_after_K5 * entry_b7b10 * b9_sz)

    legs_with_b9['day'] = legs_with_b9['day'].astype(str)
    n_days = legs_with_b9['day'].nunique()

    out('')
    out(f'Days: {n_days}    Legs: {len(legs_with_b9):,}')
    out('')
    out(f'{"scheme":<35}  {"total_$":>11}  {"$/day":>9}  {"per_leg":>9}  '
        f'{"95% CI on $/day":>22}  {"sig vs flat":>13}')

    # Per-day P&L for each scheme
    per_day_results = {}
    flat_per_day = pd.Series(schemes['FLAT (baseline)'],
                              index=legs_with_b9['day']).groupby(level=0).sum().values
    for name, realized in schemes.items():
        per_day = pd.Series(realized, index=legs_with_b9['day']).groupby(level=0).sum().values
        per_day_results[name] = per_day
        ci_lo, ci_hi = bootstrap_ci(per_day)
        delta_per_day = per_day - flat_per_day
        d_lo, d_hi = bootstrap_ci(delta_per_day)
        sig = (d_lo > 0) if name != 'FLAT (baseline)' else 'baseline'
        out(f'{name:<35}  ${realized.sum():>+9,.0f}  ${per_day.mean():>+7.0f}  '
            f'${realized.sum()/len(realized):>+7.2f}  '
            f'[${ci_lo:>+5.0f}, ${ci_hi:>+5.0f}]      '
            f'{str(sig):>13}')

    out('')
    out('--- Paired delta vs FLAT (95% bootstrap CI on per-day delta) ---')
    for name in ['B7 only', 'B9 only', 'B10 only', 'B7 + B9', 'B7 + B10',
                 'B7 + B9 + B10 (full stack)']:
        delta_per_day = per_day_results[name] - flat_per_day
        ci_lo, ci_hi = bootstrap_ci(delta_per_day)
        sig = ci_lo > 0
        out(f'  {name:<35}  delta ${delta_per_day.mean():>+5.0f}/day  '
            f'CI [${ci_lo:>+5.0f}, ${ci_hi:>+5.0f}]    sig {sig}')

    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {output_path}')
    return legs_with_b9


def main():
    # OOS sealed forward pass
    print('\n' + '=' * 80)
    print('RUNNING OOS FORWARD PASS (SEALED)')
    print('=' * 80)
    oos_results = run_forward_pass(
        label='OOS SEALED (51 days, 2026-03-19 to 2026-05-18)',
        legs_path='reports/findings/regret_oracle/oos_hardened_legs_full.csv',
        trajectory_path='reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet',
        truth_path='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet',
        source_filter='NT8',
        output_path='reports/findings/regret_oracle/forward_pass_stack_OOS.txt',
    )

    # IS forward pass (CAVEAT: trained on IS, results inflated)
    print('\n' + '=' * 80)
    print('RUNNING IS FORWARD PASS (CAVEAT: full-IS-trained models on IS)')
    print('=' * 80)
    is_results = run_forward_pass(
        label='IS (275 days, 2025-01 to 2025-12) -- CAVEAT: models trained on IS',
        legs_path='reports/findings/regret_oracle/is_hardened_legs.csv',
        trajectory_path='reports/findings/regret_oracle/trade_trajectory_IS.parquet',
        truth_path='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet',
        source_filter='ATLAS',
        output_path='reports/findings/regret_oracle/forward_pass_stack_IS.txt',
    )


if __name__ == '__main__':
    main()
