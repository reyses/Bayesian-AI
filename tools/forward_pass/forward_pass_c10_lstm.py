"""C10 forward pass — simulate trading from LSTM direct-return predictions.

Trade rule (per horizon K):
  At each 1m close t where |pred_return_R_Km| > threshold_R:
    direction = sign(pred)
    enter at close(t)
    hold for K minutes
    exit at close(t+K)
  P&L_pts = realized_return_signed_by_direction
  P&L_usd = P&L_pts × $2 × size - $6 friction

Sizing options:
  - 'flat': size = 1.0
  - 'magnitude': size = clip(|pred|, 0.5, 3.0)   (proportional to confidence)

Per-day P&L = sum of all trade P&Ls on that day.

We respect a 'no overlap' rule: once we enter at time t for horizon K,
we don't enter again until t+K (i.e., new entries on the same day must
not overlap).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import product


DOLLAR_PER_POINT = 2.0
FRICTION_PER_TRADE = 6.00


def simulate_trades(preds_df: pd.DataFrame, horizon: int,
                      threshold_R: float, sizing: str = 'flat'):
    """Per day, simulate non-overlapping trades.

    preds_df columns required:
      timestamp, day, close, r_price,
      pred_return_R_{horizon}m, actual_return_R_{horizon}m
    """
    pred_col = f'pred_return_R_{horizon}m'
    act_col = f'actual_return_R_{horizon}m'
    rows = []
    horizon_s = horizon * 60

    for day, g in preds_df.groupby('day'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        ts_arr = g['timestamp'].values.astype(np.int64)
        pred_arr = g[pred_col].values
        actual_arr = g[act_col].values
        close_arr = g['close'].values
        r_price_arr = g['r_price'].values
        day_pnl = 0.0
        day_trades = []
        last_exit_ts = -1
        for i in range(len(g)):
            if ts_arr[i] < last_exit_ts:
                continue
            pred_R = float(pred_arr[i])
            if abs(pred_R) < threshold_R:
                continue
            direction = 1.0 if pred_R > 0 else -1.0
            # Realized signed return in R units = actual_return_R × direction
            realized_R = float(actual_arr[i]) * direction
            r_price = float(r_price_arr[i])
            pnl_pts = realized_R * r_price
            pnl_usd_raw = pnl_pts * DOLLAR_PER_POINT - FRICTION_PER_TRADE
            # Sizing
            if sizing == 'flat':
                size = 1.0
            elif sizing == 'magnitude':
                size = float(np.clip(abs(pred_R), 0.5, 3.0))
            else:
                size = 1.0
            pnl_usd = pnl_usd_raw * size
            day_pnl += pnl_usd
            day_trades.append({
                'day': day, 'entry_ts': int(ts_arr[i]),
                'pred_R': pred_R, 'realized_R': realized_R,
                'size': size, 'pnl_usd': pnl_usd,
                'direction': 'LONG' if direction > 0 else 'SHORT',
            })
            last_exit_ts = ts_arr[i] + horizon_s
        rows.append({'day': day, 'n_trades': len(day_trades),
                       'pnl_usd': day_pnl,
                       'trades': day_trades})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions',
                    default='reports/findings/regret_oracle/c10_lstm_predictions_OOS.parquet')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/c10_lstm_forward_pass.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/c10_lstm_forward_pass.txt')
    args = ap.parse_args()

    print('Loading predictions...')
    df = pd.read_parquet(args.predictions)
    print(f'  rows: {len(df):,}   days: {df["day"].nunique()}')

    # Detect available horizons
    horizons = sorted([int(c.split('_')[-1].replace('m', ''))
                       for c in df.columns if c.startswith('pred_return_R_')])
    print(f'  horizons available: {horizons} min')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('B10 DIRECT-TRADE LSTM FORWARD PASS (no zigzag, no composite)')
    out('=' * 78)
    out(f'Days: {df["day"].nunique()}   Bars: {len(df):,}   Horizons: {horizons}')
    out(f'Friction: ${FRICTION_PER_TRADE:.2f}/trade')
    out('')

    rng = np.random.default_rng(42)
    sweep_results = []
    for horizon in horizons:
        for threshold in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]:
            for sizing in ['flat', 'magnitude']:
                day_df = simulate_trades(df, horizon, threshold, sizing)
                pnl = day_df['pnl_usd'].values
                n_trades_total = int(day_df['n_trades'].sum())
                if n_trades_total == 0:
                    sweep_results.append({
                        'horizon_min': horizon, 'threshold_R': threshold,
                        'sizing': sizing,
                        'n_trades': 0, 'n_days': len(day_df),
                        'mean_per_day': 0.0,
                        'median_per_day': 0.0,
                        'min_day': 0.0, 'max_day': 0.0,
                        'pos_days': 0, 'days_over_200': 0,
                        'total': 0.0, 'mean_per_trade': 0.0,
                    })
                    continue
                boots = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean()
                                   for _ in range(2000)])
                sweep_results.append({
                    'horizon_min': horizon, 'threshold_R': threshold,
                    'sizing': sizing,
                    'n_trades': n_trades_total,
                    'n_days': len(day_df),
                    'mean_per_day': float(pnl.mean()),
                    'median_per_day': float(np.median(pnl)),
                    'ci_lo_per_day': float(np.percentile(boots, 2.5)),
                    'ci_hi_per_day': float(np.percentile(boots, 97.5)),
                    'min_day': float(pnl.min()),
                    'max_day': float(pnl.max()),
                    'pos_days': int((pnl > 0).sum()),
                    'days_over_200': int((pnl > 200).sum()),
                    'total': float(pnl.sum()),
                    'mean_per_trade': float(pnl.sum() / max(n_trades_total, 1)),
                })

    sdf = pd.DataFrame(sweep_results)
    sdf.to_csv(args.out, index=False)

    out(f'{"K":<4}  {"thr_R":<6}  {"sizing":<10}  {"n_trades":>9}  '
        f'{"mean_$/day":>11}  {"median":>9}  {"CI_lo":>8}  {"CI_hi":>8}  '
        f'{"pos_days":>8}  {"#>$200":>6}')
    for _, r in sdf.iterrows():
        if r['n_trades'] == 0:
            out(f'  {int(r["horizon_min"]):<3}  {r["threshold_R"]:<6.2f}  {r["sizing"]:<10}  '
                f'{int(r["n_trades"]):>9}  (no trades)')
            continue
        out(f'  {int(r["horizon_min"]):<3}  {r["threshold_R"]:<6.2f}  {r["sizing"]:<10}  '
            f'{int(r["n_trades"]):>9}  '
            f'${r["mean_per_day"]:>+9.0f}  ${r["median_per_day"]:>+7.0f}  '
            f'${r.get("ci_lo_per_day", 0):>+6.0f}  ${r.get("ci_hi_per_day", 0):>+6.0f}  '
            f'{int(r["pos_days"]):>8}  {int(r["days_over_200"]):>6}')

    # Highlight best config
    if len(sdf) > 0:
        positive = sdf[(sdf.get('ci_lo_per_day', 0) > 0) & (sdf['n_trades'] > 0)]
        out('')
        if len(positive) > 0:
            best = positive.sort_values('mean_per_day', ascending=False).iloc[0]
            out(f'*** Best config with CI > 0:')
            out(f'   horizon={int(best["horizon_min"])}min  threshold={best["threshold_R"]:.2f}  sizing={best["sizing"]}')
            out(f'   ${best["mean_per_day"]:+.0f}/day   CI [${best["ci_lo_per_day"]:+.0f}, ${best["ci_hi_per_day"]:+.0f}]')
            out(f'   {int(best["n_trades"])} total trades   {int(best["pos_days"])}/{int(best["n_days"])} positive days')
        else:
            top = sdf.sort_values('mean_per_day', ascending=False).iloc[0]
            out(f'No configs have strictly positive CI.')
            out(f'   Best by mean: horizon={int(top["horizon_min"])}min  thr={top["threshold_R"]:.2f}  '
                f'sizing={top["sizing"]}  ${top["mean_per_day"]:+.0f}/day')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
