"""B9 OOS sealed test v2 -- apply production B9 models to the FRESH 51-day OOS sample.

Reads the production B9 models (b9_remaining_amplitude_K{5,10,30,60,120}.pkl)
trained on full IS, applies to the fresh OOS trajectory dataset.

Reports: per-K Delta/day with bootstrap CI, anti-doom-slippage stress.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


K_HORIZONS = [5, 10, 30, 60, 120]
N_BOOTSTRAP = 4000


def size_from_pred(pred_remaining):
    if pred_remaining > 50:   return 1.5
    if pred_remaining > 10:   return 1.0
    if pred_remaining > -10:  return 1.0
    if pred_remaining > -50:  return 0.5
    return 0.0


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    traj = pd.read_parquet('reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet')
    legs = pd.read_csv('reports/findings/regret_oracle/oos_hardened_legs_full.csv')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 90)
    out('B9 SEALED OOS TEST -- fresh 51-day sample (2026-03-19 to 2026-05-18)')
    out('=' * 90)
    out(f'Legs: {len(legs):,}, days: {legs["day"].nunique()}')
    out(f'Baseline flat: total ${legs["pnl_usd"].sum():+,.0f}, '
        f'$/day ${legs["pnl_usd"].sum()/legs["day"].nunique():+.0f}')
    out('')

    model_dir = Path('reports/findings/regret_oracle')
    results = []
    for K in K_HORIZONS:
        mpath = model_dir / f'b9_remaining_amplitude_K{K}.pkl'
        if not mpath.exists():
            continue
        with open(mpath, 'rb') as f:
            mdl = pickle.load(f)
        sub = traj[traj['K'] == K].copy()
        if len(sub) == 0:
            continue
        X = sub[mdl['feat_cols']].fillna(0.0).values
        y_pred = mdl['model'].predict(X)
        sub['pred_remaining'] = y_pred
        sub['size_factor'] = [size_from_pred(p) for p in y_pred]
        sub['realized'] = sub['pnl_usd_so_far'] + sub['size_factor'] * (
            sub['exit_pnl_usd'] - sub['pnl_usd_so_far'])
        sub['delta'] = sub['realized'] - sub['exit_pnl_usd']
        per_day = sub.groupby('day')['delta'].sum().values
        ci_lo, ci_hi = bootstrap_ci(per_day)
        results.append({
            'K': K, 'n_legs': len(sub),
            'mean_delta_per_day': float(per_day.mean()),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'significant': ci_lo > 0,
            'n_cut': int((sub['size_factor'] == 0.0).sum()),
            'n_half': int((sub['size_factor'] == 0.5).sum()),
            'n_full': int((sub['size_factor'] == 1.0).sum()),
            'n_pyr': int((sub['size_factor'] == 1.5).sum()),
            'worst_day': float(per_day.min()),
            'best_day': float(per_day.max()),
        })
        out(f'K={K:>3}   delta ${per_day.mean():+.0f}/day   '
            f'CI [${ci_lo:+.0f}, ${ci_hi:+.0f}]   sig {ci_lo > 0}   '
            f'(cut {results[-1]["n_cut"]} / half {results[-1]["n_half"]} / '
            f'full {results[-1]["n_full"]} / pyr {results[-1]["n_pyr"]})')

    out('')
    out('--- VERDICT vs prior OOS sealed (2026-05-17 evening, 31 days) ---')
    out('Prior: K=5 +$67/day CI [+$32, +$106] SIGNIFICANT (14.1% lift)')
    if results:
        best = max(results, key=lambda r: r['mean_delta_per_day'])
        out(f'Now (51 days): K={best["K"]} ${best["mean_delta_per_day"]:+.0f}/day  '
            f'CI [${best["ci_lo"]:+.0f}, ${best["ci_hi"]:+.0f}]  '
            f'sig {best["significant"]}')

    out('')
    out('=== Anti-doom slippage stress ===')
    out(f'{"K":>4}  {"S=$0":>7}  {"S=$2":>7}  {"S=$5":>7}  {"S=$10":>7}')
    for r in results:
        K = r['K']
        sub = traj[traj['K'] == K].copy()
        mdl = pickle.load(open(model_dir / f'b9_remaining_amplitude_K{K}.pkl', 'rb'))
        X = sub[mdl['feat_cols']].fillna(0.0).values
        y_pred = mdl['model'].predict(X)
        sub['size_factor'] = [size_from_pred(p) for p in y_pred]
        action_mask = sub['size_factor'] != 1.0
        deltas = []
        for S in [0, 2, 5, 10]:
            sub_s = sub.copy()
            sub_s['realized'] = sub_s['pnl_usd_so_far'] + sub_s['size_factor'] * (
                sub_s['exit_pnl_usd'] - sub_s['pnl_usd_so_far'])
            sub_s.loc[action_mask, 'realized'] -= S
            sub_s['delta'] = sub_s['realized'] - sub_s['exit_pnl_usd']
            per_day = sub_s.groupby('day')['delta'].sum().values
            deltas.append(per_day.mean())
        out(f'  K={K:>3}  ${deltas[0]:>+5.0f}  ${deltas[1]:>+5.0f}  '
            f'${deltas[2]:>+5.0f}  ${deltas[3]:>+5.0f}')

    Path('reports/findings/regret_oracle/b9_OOS_sealed_v2_51days.txt').write_text(
        '\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
