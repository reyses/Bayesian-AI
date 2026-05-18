"""Compare C9 LSTM (failed candidate) vs B7 GBM sizing on the same OOS legs.

Both use the gbm_ev sizing rule: size = max(pred_amp_R - 1, 0) clipped [0,3].
The only difference is which model produced pred_amp_R.

Loads the per-leg realistic P&L from composite_forward_pass_hardened.csv
(which has B7 GBM predictions + R-trigger entry/exit P&L) and merges
C9 LSTM (failed candidate) predictions onto matching legs.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gbm_ev(p):
    return float(np.clip(max(p - 1.0, 0.0), 0.0, 3.0))


def boot_ci(arr, n=4000, seed=42):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hardened-csv',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--b9-cache',
                    default='reports/findings/regret_oracle/c9_lstm_predictions_OOS.parquet')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass_lstm_vs_gbm.txt')
    args = ap.parse_args()

    print('Loading inputs...')
    h = pd.read_csv(args.hardened_csv)
    b9 = pd.read_parquet(args.c9_cache)
    print(f'  Hardened CSV legs: {len(h):,}')
    print(f'  C9 LSTM (failed candidate) legs:      {len(b9):,}')

    # Merge — match on day + entry_ts
    merged = h.merge(b9[['day', 'entry_ts', 'pred_amp_R_lstm']],
                       on=['day', 'entry_ts'], how='inner')
    print(f'  Matched legs:       {len(merged):,}')

    # Apply sizing
    b7_pred = merged['pred_amp_R_hardened'].values
    c9_pred = merged['pred_amp_R_lstm'].values
    pnl = merged['pnl_usd'].values

    size_flat = np.ones(len(merged))
    size_b7 = np.array([gbm_ev(p) for p in b7_pred])
    size_c9 = np.array([gbm_ev(p) for p in c9_pred])

    schemes = {
        'flat':       size_flat,
        'B7 GBM ev':  size_b7,
        'C9 LSTM (failed candidate) ev': size_c9,
    }

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B7 GBM vs C9 LSTM (failed candidate) — sizing comparison (hardened OOS, R-trigger entries)')
    out('=' * 78)
    out(f'Matched legs: {len(merged):,}   Days: {merged["day"].nunique()}')
    out(f'Per-leg P&L (raw, before sizing): mean ${pnl.mean():+.2f}   median ${np.median(pnl):+.2f}')
    out('')
    out('Prediction distribution comparison:')
    out(f'  B7 GBM:   median {np.median(b7_pred):.2f}  mean {b7_pred.mean():.2f}  '
        f'min {b7_pred.min():.2f}  max {b7_pred.max():.2f}')
    out(f'  C9 LSTM (failed candidate):  median {np.median(c9_pred):.2f}  mean {c9_pred.mean():.2f}  '
        f'min {c9_pred.min():.2f}  max {c9_pred.max():.2f}')
    out('')

    # Correlations between predictions
    if b7_pred.std() > 0 and c9_pred.std() > 0:
        rho = float(np.corrcoef(b7_pred, c9_pred)[0, 1])
        out(f'Correlation between B7 and C9 predictions: {rho:+.4f}')
    out('')

    out(f'{"scheme":<12}  {"total_$":>10}  {"per_leg":>9}  {"$/day":>9}  '
        f'{"CI lo":>9}  {"CI hi":>9}  {"pos days":>8}  {"days>$200":>10}  '
        f'{"mean size":>9}')

    rng = np.random.default_rng(42)
    per_day_results = {}
    for name, sizes in schemes.items():
        weighted = pnl * sizes
        df_copy = merged.copy()
        df_copy['wpnl'] = weighted
        per_day = df_copy.groupby('day')['wpnl'].sum().values
        per_day_results[name] = per_day
        total = weighted.sum()
        mean_d, ci_lo, ci_hi = boot_ci(per_day)
        per_leg = total / len(merged)
        out(f'{name:<12}  ${total:>+8,.0f}  ${per_leg:>+7.2f}  '
            f'${mean_d:>+7.0f}  ${ci_lo:>+7.0f}  ${ci_hi:>+7.0f}  '
            f'{int((per_day > 0).sum()):>8}  {int((per_day > 200).sum()):>10}  '
            f'{sizes.mean():>9.3f}')

    # Paired delta C9 vs B7
    out('')
    out('--- Paired delta C9 LSTM (failed candidate) vs B7 GBM (per-day) ---')
    b9_pd = per_day_results['C9 LSTM (failed candidate) ev']
    b7_pd = per_day_results['B7 GBM ev']
    delta = b9_pd - b7_pd
    rng = np.random.default_rng(42)
    boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(4000)])
    out(f'  Mean delta: ${delta.mean():+.2f}/day  95% CI [${np.percentile(boots, 2.5):+.2f}, ${np.percentile(boots, 97.5):+.2f}]')
    out(f'  C9 LSTM (failed candidate) wins on {(b9_pd > b7_pd).sum()}/{len(b9_pd)} days')

    if delta.mean() > 0 and np.percentile(boots, 2.5) > 0:
        out('  --> C9 LSTM (failed candidate) significantly BEATS B7 GBM')
    elif delta.mean() < 0 and np.percentile(boots, 97.5) < 0:
        out('  --> C9 LSTM (failed candidate) significantly LOSES to B7 GBM')
    else:
        out('  --> No statistical significance')

    out('')
    out('=' * 78)
    out('INTERPRETATION')
    out('=' * 78)
    out('C9 LSTM (failed candidate) was trained on first 2 months of IS only (~2,303 legs).')
    out('LSTMs are sample-hungry; this is far below typical training sizes.')
    out('B7 GBM was trained on full IS (~17,789 legs).')
    out('')
    out('The honest comparison shows whether sequence-input + LSTM adds')
    out('value over tabular GBM on the same target.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
