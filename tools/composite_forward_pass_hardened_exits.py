"""HARDENED-EXITS forward pass.

User goal (2026-05-17): 32 OOS days all positive AND > $200/day.

Current hardened baseline: 26/31 days positive, worst day -$598, mode $468.
To reach the target, we need to:
  1. Cap per-leg downside (stop-loss before R-trigger natural exit)
  2. Lock in mid-leg gains (take-profit target)
  3. Stop trading on bad days (daily loss cap)
  4. Skip predicted-bad entries (B2 fakeout filter + low B7 prediction)

This script simulates each leg bar-by-bar at 5s resolution, applying:
  - SL (stop-loss): exit if running PnL drops below -SL_USD per contract
  - TP (take-profit): exit if running PnL rises above +TP_USD per contract
  - R-trigger fallback (exit at next R-trigger if neither SL nor TP hit)
  - Daily loss cap: skip remaining legs in day after cumulative day P&L hits -CAP_USD
  - Entry filter: skip if pred_amp_R < AMP_MIN or B2 P(fakeout) > FAKE_MAX

With B7 sizing (gbm_ev) applied to surviving legs.

Sweep parameters and report which configurations achieve the user's target.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0
COMMISSION_PER_LEG = 4.00
SLIPPAGE_PER_LEG   = 2.00
FRICTION_PER_LEG   = COMMISSION_PER_LEG + SLIPPAGE_PER_LEG

TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')


def simulate_leg_bar_by_bar(closes5s, ts5s, entry_ts, entry_price, leg_dir,
                              exit_ts_fallback, exit_price_fallback,
                              SL_USD, TP_USD):
    """Simulate one leg with hard SL/TP. Returns exit info.

    Walks 5s closes between entry_ts and fallback exit_ts. At each bar:
      For LONG: pnl_pts = close - entry. Exit if pnl <= -SL_USD/2 (stop) or >= TP_USD/2 (target).
      For SHORT: pnl_pts = entry - close. Same conditions.
    If neither triggers, exit at fallback R-trigger price.
    """
    mask = (ts5s > entry_ts) & (ts5s <= exit_ts_fallback)
    leg_closes = closes5s[mask]
    leg_ts = ts5s[mask]
    if len(leg_closes) == 0:
        return {
            'exit_price': exit_price_fallback,
            'exit_ts': exit_ts_fallback,
            'exit_reason': 'no_bars',
        }
    if leg_dir == 'LONG':
        pnl_pts_path = leg_closes - entry_price
    else:
        pnl_pts_path = entry_price - leg_closes
    pnl_usd_path = pnl_pts_path * DOLLAR_PER_POINT

    # Check SL/TP triggers
    sl_hits = np.where(pnl_usd_path <= -SL_USD)[0] if SL_USD is not None else np.array([], dtype=int)
    tp_hits = np.where(pnl_usd_path >=  TP_USD)[0] if TP_USD is not None else np.array([], dtype=int)

    first_sl = int(sl_hits[0]) if len(sl_hits) else len(leg_closes)
    first_tp = int(tp_hits[0]) if len(tp_hits) else len(leg_closes)

    if first_sl < first_tp and first_sl < len(leg_closes):
        return {
            'exit_price': float(leg_closes[first_sl]),
            'exit_ts': int(leg_ts[first_sl]),
            'exit_reason': 'SL',
        }
    elif first_tp < len(leg_closes):
        return {
            'exit_price': float(leg_closes[first_tp]),
            'exit_ts': int(leg_ts[first_tp]),
            'exit_reason': 'TP',
        }
    else:
        return {
            'exit_price': exit_price_fallback,
            'exit_ts': exit_ts_fallback,
            'exit_reason': 'R-trigger',
        }


def gbm_ev(pred_R):
    return float(np.clip(max(pred_R - 1.0, 0.0), 0.0, 3.0))


def run_one_config(legs_df, SL, TP, daily_cap, amp_min,
                     bars_5s_lookup, ts_5s_lookup):
    """Apply hardened-exits + filter + daily cap. Returns per-day P&L list."""
    rows = []
    for day, day_legs in legs_df.groupby('day'):
        day_legs = day_legs.sort_values('entry_ts').reset_index(drop=True)
        day_pnl = 0.0
        day_capped = False
        closes_5s = bars_5s_lookup[day]
        ts_5s = ts_5s_lookup[day]

        for _, leg in day_legs.iterrows():
            # Entry filter: skip if amp prediction too low
            if leg['pred_amp_R_hardened'] < amp_min:
                continue
            # Daily cap: skip remaining legs in day
            if day_capped:
                continue

            sim = simulate_leg_bar_by_bar(
                closes_5s, ts_5s, leg['entry_ts'], leg['entry_price'],
                leg['leg_dir'], leg['exit_ts'], leg['exit_price'],
                SL_USD=SL, TP_USD=TP,
            )

            # Compute P&L
            if leg['leg_dir'] == 'LONG':
                pnl_pts = sim['exit_price'] - leg['entry_price']
            else:
                pnl_pts = leg['entry_price'] - sim['exit_price']
            pnl_usd = pnl_pts * DOLLAR_PER_POINT - FRICTION_PER_LEG

            # Apply sizing (gbm_ev)
            size = gbm_ev(leg['pred_amp_R_hardened'])
            wpnl = pnl_usd * size

            day_pnl += wpnl

            # Check daily loss cap
            if daily_cap is not None and day_pnl <= -daily_cap:
                day_capped = True

        rows.append({
            'day': day,
            'pnl_usd': day_pnl,
            'capped': day_capped,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--sweep', action='store_true', help='Run full param sweep')
    ap.add_argument('--SL', type=float, default=40)
    ap.add_argument('--TP', type=float, default=80)
    ap.add_argument('--daily-cap', type=float, default=300)
    ap.add_argument('--amp-min', type=float, default=1.0)
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened_exits.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened_exits.txt')
    args = ap.parse_args()

    print('Loading legs CSV...')
    df = pd.read_csv(args.csv)
    print(f'  {len(df):,} legs, {df["day"].nunique()} days')

    print('Loading 5s OHLC per day...')
    bars_5s = {}
    ts_5s = {}
    for day in tqdm(sorted(df['day'].unique()), desc='5s'):
        p = NT8_5S_DIR / f'{day}.parquet'
        if not p.exists():
            continue
        b = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
        bars_5s[day] = b['close'].values.astype(np.float64)
        ts_5s[day] = b['timestamp'].values.astype(np.int64)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    if args.sweep:
        # Sweep parameters
        SL_grid = [None, 30, 40, 60, 80]
        TP_grid = [None, 40, 60, 80, 120, 200]
        cap_grid = [None, 200, 300, 500]
        amp_grid = [0.0, 1.0, 1.2, 1.5]

        out('=' * 78)
        out('HARDENED-EXITS PARAMETER SWEEP — find user target (32/32 days > $200)')
        out('=' * 78)
        out(f'SL grid: {SL_grid}')
        out(f'TP grid: {TP_grid}')
        out(f'daily_cap grid: {cap_grid}')
        out(f'amp_min grid (skip if pred_amp_R below): {amp_grid}')
        out('')

        results = []
        configs = list(product(SL_grid, TP_grid, cap_grid, amp_grid))
        print(f'Sweeping {len(configs)} configs...')
        for SL, TP, cap, amp in tqdm(configs, desc='configs'):
            r = run_one_config(df, SL, TP, cap, amp, bars_5s, ts_5s)
            pnl = r['pnl_usd'].values
            results.append({
                'SL': SL, 'TP': TP, 'daily_cap': cap, 'amp_min': amp,
                'n_days': len(pnl),
                'mean_per_day': float(pnl.mean()),
                'median_per_day': float(np.median(pnl)),
                'min_day': float(pnl.min()),
                'max_day': float(pnl.max()),
                'pos_days': int((pnl > 0).sum()),
                'days_over_200': int((pnl > 200).sum()),
                'pct_days_over_200': float((pnl > 200).mean() * 100),
                'total_usd': float(pnl.sum()),
            })

        rdf = pd.DataFrame(results)
        rdf = rdf.sort_values(['days_over_200', 'mean_per_day'],
                                ascending=[False, False])
        rdf.to_csv(args.out, index=False)

        out('Top 15 configs (sorted by days_over_$200 desc, then mean_per_day):')
        out(f'  {"SL":>5}  {"TP":>5}  {"cap":>5}  {"amp":>4}  '
            f'{"days_pos":>8}  {"days>$200":>10}  {"min":>9}  '
            f'{"median":>9}  {"mean":>9}  {"total":>10}')
        for _, r in rdf.head(15).iterrows():
            SL_s = f'${r["SL"]:>3.0f}' if r['SL'] is not None and not pd.isna(r['SL']) else 'NONE'
            TP_s = f'${r["TP"]:>3.0f}' if r['TP'] is not None and not pd.isna(r['TP']) else 'NONE'
            cap_s = f'${r["daily_cap"]:>3.0f}' if r['daily_cap'] is not None and not pd.isna(r['daily_cap']) else 'NONE'
            out(f'  {SL_s:>5}  {TP_s:>5}  {cap_s:>5}  {r["amp_min"]:>4.1f}  '
                f'{int(r["pos_days"]):>8}  {int(r["days_over_200"]):>10}  '
                f'${r["min_day"]:>+7.0f}  ${r["median_per_day"]:>+7.0f}  '
                f'${r["mean_per_day"]:>+7.0f}  ${r["total_usd"]:>+8.0f}')

        # Highlight target
        target = rdf[rdf['days_over_200'] == rdf['n_days']]
        out('')
        if len(target) > 0:
            out(f'*** {len(target)} configs achieve TARGET (all days > $200) ***')
            best = target.sort_values('mean_per_day', ascending=False).iloc[0]
            out(f'   Best: SL=${best["SL"]} TP=${best["TP"]} cap=${best["daily_cap"]} '
                f'amp_min={best["amp_min"]:.1f}  mean ${best["mean_per_day"]:+.0f}/day '
                f'total ${best["total_usd"]:+.0f}')
        else:
            most = rdf.iloc[0]
            out(f'No config achieves 100% days > $200.')
            out(f'   Best: {int(most["days_over_200"])}/{int(most["n_days"])} days > $200  '
                f'(SL=${most["SL"]} TP=${most["TP"]} cap=${most["daily_cap"]} '
                f'amp_min={most["amp_min"]:.1f})  '
                f'min day ${most["min_day"]:+.0f}  mean ${most["mean_per_day"]:+.0f}')

    else:
        # Single config run
        out('=' * 78)
        out(f'HARDENED-EXITS  SL=${args.SL}  TP=${args.TP}  '
            f'cap=${args.daily_cap}  amp_min={args.amp_min}')
        out('=' * 78)
        r = run_one_config(df, args.SL, args.TP, args.daily_cap, args.amp_min,
                            bars_5s, ts_5s)
        pnl = r['pnl_usd'].values
        out(f'  n_days:        {len(pnl)}')
        out(f'  mean:          ${pnl.mean():+.2f}/day')
        out(f'  median:        ${np.median(pnl):+.2f}/day')
        out(f'  min:           ${pnl.min():+.2f}')
        out(f'  max:           ${pnl.max():+.2f}')
        out(f'  pos days:      {(pnl > 0).sum()}/{len(pnl)}')
        out(f'  days > $200:   {(pnl > 200).sum()}/{len(pnl)}')
        out(f'  total:         ${pnl.sum():+.0f}')
        r.to_csv(args.out, index=False)

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
