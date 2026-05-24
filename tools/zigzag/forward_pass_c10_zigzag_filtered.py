"""C10 + zigzag forward pass.

Reuses the C10 LSTM (failed candidate) (trained on forward-return prediction) but applies
it as a SIGNED CONFIRMATION FILTER on zigzag R-trigger entries.

At each R-trigger fire:
  - Look up B10 signed forward-return prediction at the entry bar
  - For LONG leg: require pred > +threshold_R  (LSTM agrees with up move)
  - For SHORT leg: require pred < -threshold_R (LSTM agrees with down move)
  - If disagreement or low magnitude -> skip the leg entirely

Compare 3 schemes:
  - flat       : take every R-trigger leg (no filter), size 1.0
  - lstm_filt  : only legs where LSTM agrees with direction at threshold
  - lstm_size  : take every leg, size by |LSTM signed pred / R| clipped

Uses the hardened forward-pass leg list (R-trigger entry/exit prices,
realistic P&L) merged with B10's per-bar signed predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FRICTION = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hardened-csv',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--c10-preds',
                    default='reports/findings/regret_oracle/c10_lstm_predictions_OOS.parquet')
    ap.add_argument('--horizon-col', default='pred_return_R_15m',
                    help='Which horizon prediction to use (5/15/30 min)')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/c10_zigzag_filtered_forward_pass.txt')
    args = ap.parse_args()

    print('Loading inputs...')
    h = pd.read_csv(args.hardened_csv)
    c10 = pd.read_parquet(args.c10_preds)
    print(f'  Hardened legs: {len(h):,}  ({h["day"].nunique()} days)')
    print(f'  B10 preds:     {len(b10):,}')

    # For each leg in hardened CSV, look up B10 prediction at the entry_ts 1m bar.
    # B10 timestamps are 1m closes (every minute); hardened entry_ts may be a 5s
    # timestamp slightly after a 1m close. Find the latest 1m close at or before
    # entry_ts per day.
    print('Aligning B10 predictions to hardened entry timestamps...')
    c10_sorted = b10.sort_values(['day', 'timestamp']).reset_index(drop=True)
    h['pred_R'] = np.nan
    for day, day_legs in h.groupby('day'):
        c10_day = c10_sorted[c10_sorted['day'] == day]
        if len(c10_day) == 0:
            continue
        c10_ts = c10_day['timestamp'].values.astype(np.int64)
        c10_preds = c10_day[args.horizon_col].values
        leg_ts = day_legs['entry_ts'].values.astype(np.int64)
        idx = np.searchsorted(c10_ts, leg_ts, side='right') - 1
        idx = np.clip(idx, 0, len(c10_ts) - 1)
        h.loc[day_legs.index, 'pred_R'] = c10_preds[idx]

    # Drop legs without B10 prediction
    matched = h.dropna(subset=['pred_R']).copy()
    print(f'  Matched: {len(matched):,}')

    # Pre-compute direction match flag
    # leg_dir == 'LONG': we entered LONG (price expected to rise). LSTM agrees if pred_R > 0.
    # leg_dir == 'SHORT': we entered SHORT (price expected to fall). LSTM agrees if pred_R < 0.
    matched['lstm_signed_for_leg'] = np.where(
        matched['leg_dir'] == 'LONG',
        matched['pred_R'],
        -matched['pred_R'],
    )
    # signed_for_leg > threshold = LSTM agrees in leg direction

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B10 + ZIGZAG FORWARD PASS — LSTM as signed-direction filter')
    out('=' * 78)
    out(f'Days: {matched["day"].nunique()}   Matched legs: {len(matched):,}')
    out(f'Horizon column: {args.horizon_col}')
    out(f'Friction: ${FRICTION:.2f}/leg')
    out('')

    # === Baseline: flat sizing, no filter ===
    pnl_raw = matched['pnl_usd'].values
    base_total = pnl_raw.sum()
    base_per_day = base_total / matched['day'].nunique()
    base_pos_days = (matched.groupby('day')['pnl_usd'].sum() > 0).sum()
    base_days_over_200 = (matched.groupby('day')['pnl_usd'].sum() > 200).sum()
    out(f'BASELINE (flat, no filter):')
    out(f'  total ${base_total:+,.0f}   $/day ${base_per_day:+.0f}   '
        f'pos days {base_pos_days}/{matched["day"].nunique()}   '
        f'days>$200 {base_days_over_200}')
    out('')

    # === LSTM-filter sweep ===
    out('LSTM-FILTERED (skip legs where LSTM disagrees):')
    out(f'  {"threshold":>10}  {"n_kept":>7}  {"%kept":>6}  '
        f'{"$/day":>9}  {"per_leg":>9}  {"pos_days":>8}  {"days>$200":>10}')
    sweep = []
    for thr in [-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0]:
        keep_mask = matched['lstm_signed_for_leg'].values > thr
        kept = matched[keep_mask]
        n = len(kept)
        if n == 0:
            continue
        total = kept['pnl_usd'].sum()
        per_day = total / matched['day'].nunique()   # divided by all days for fair comparison
        per_leg = total / n
        pos_days = (kept.groupby('day')['pnl_usd'].sum() > 0).sum()
        days_200 = (kept.groupby('day')['pnl_usd'].sum() > 200).sum()
        out(f'  thr>={thr:>+5.2f}   {n:>7,}  {n/len(matched)*100:>5.1f}%  '
            f'${per_day:>+7.0f}  ${per_leg:>+7.2f}  '
            f'{int(pos_days):>8}  {int(days_200):>10}')
        sweep.append({'threshold': thr, 'n_kept': n, 'pct_kept': n/len(matched),
                       'total': total, 'per_day': per_day, 'per_leg': per_leg,
                       'pos_days': int(pos_days), 'days_over_200': int(days_200)})
    out('')

    # === LSTM-SIZED (take every leg, scale by |signed pred| clipped) ===
    out('LSTM-SIZED (take every leg, size by |signed pred|, clipped to [0.5, 3.0]):')
    sizes = np.clip(np.abs(matched['lstm_signed_for_leg'].values), 0.5, 3.0)
    # But flip sign: if LSTM strongly disagrees, set size to 0 (skip)
    # Method 1: positive only — disagreement = 0 size
    sizes_only_agree = np.where(matched['lstm_signed_for_leg'].values > 0,
                                  np.clip(matched['lstm_signed_for_leg'].values, 0.5, 3.0),
                                  0.0)
    weighted = pnl_raw * sizes_only_agree
    total = weighted.sum()
    n_taken = int((sizes_only_agree > 0).sum())
    out(f'  agree+size (skip disagree):  n={n_taken:,}  '
        f'$/day ${total/matched["day"].nunique():+.0f}  '
        f'per_leg ${total/max(n_taken,1):+.2f}')
    # Method 2: continuous (signed). Negative pred = short the long leg (= bad — flip leg direction?)
    # Skip this — doesn't make sense to flip a confirmed zigzag entry.

    # === Compare to B7 GBM baseline ===
    out('')
    out('--- Vs B7 GBM sizing (already in hardened CSV) ---')
    out('  flat (zigzag only):           $475/day')
    out('  B7 GBM sized (gbm_ev):        $927/day')
    out('  hand_aggressive:              $857/day')
    out('  C10 LSTM (failed candidate) filter (best above): from sweep table')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
