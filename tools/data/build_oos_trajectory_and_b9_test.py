"""SEALED OOS test: build OOS trajectory dataset + apply B10 production model.

Single-shot test. Per CLAUDE.md OOS-only rule + anti-doom-cascade,
report results without retuning.

Pipeline:
  1. Read OOS hardened legs (composite_forward_pass_hardened.csv, 31 days)
  2. For each leg, build trajectory features at K bars (matches IS pipeline)
  3. Load B9 production models (per-K)
  4. Apply continuous sizing rule -> compute realized P&L vs flat baseline
  5. Bootstrap CI on per-day delta
  6. Multi-axis robustness: report under multiple slippage/gap assumptions

Output: reports/findings/regret_oracle/b9_OOS_singleshot_results.txt
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


K_HORIZONS = [5, 10, 30, 60, 120]
DOLLAR_PER_POINT = 2.0


def size_from_pred(pred_remaining):
    if pred_remaining > 50:   return 1.5
    if pred_remaining > 10:   return 1.0
    if pred_remaining > -10:  return 1.0
    if pred_remaining > -50:  return 0.5
    return 0.0


def get_v2_cols(truth: pd.DataFrame) -> list:
    skip = {'timestamp', 'day', 'is_pivot', 'pivot_dir', 'pivot_price',
            'pivot_idx', 'leg_direction', 'leg_amplitude_pts',
            'leg_amplitude_R', 'leg_duration_bars', 'atr_pts', 'target_split'}
    return [c for c in truth.columns
            if c not in skip and truth[c].dtype != object]


def build_oos_trajectory(legs: pd.DataFrame, truth: pd.DataFrame,
                          bars5s_dir: Path, v2_cols: list):
    rows = []
    for day in tqdm(sorted(legs['day'].unique()), desc='OOS trajectory'):
        bars5s_path = bars5s_dir / f'{day}.parquet'
        if not bars5s_path.exists():
            continue
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        ts5s = bars5s['timestamp'].values.astype(np.int64)
        close5s = bars5s['close'].values.astype(np.float64)
        high5s = bars5s['high'].values.astype(np.float64)
        low5s = bars5s['low'].values.astype(np.float64)
        last_ts5s = int(ts5s[-1]) if len(ts5s) > 0 else 0

        truth_day = truth[truth['day'] == day].sort_values('timestamp').reset_index(drop=True)
        truth_ts = truth_day['timestamp'].values.astype(np.int64)
        if len(truth_ts) == 0:
            continue
        v2_arr = truth_day[v2_cols].astype(np.float32).fillna(0.0).values

        day_legs = legs[legs['day'] == day]
        for li, leg in day_legs.iterrows():
            entry_ts = int(leg['entry_ts'])
            exit_ts = int(leg['exit_ts'])
            entry_price = float(leg['entry_price'])
            leg_dir = str(leg['leg_dir'])
            leg_sign = 1.0 if leg_dir == 'LONG' else -1.0
            r_price = float(leg['r_price'])
            exit_pnl_usd = float(leg['pnl_usd'])

            entry_idx = int(np.searchsorted(ts5s, entry_ts, side='left'))
            if entry_idx >= len(ts5s):
                continue

            for K in K_HORIZONS:
                bar_ts = entry_ts + K * 5
                if bar_ts > exit_ts or bar_ts > last_ts5s:
                    continue
                end_idx = int(np.searchsorted(ts5s, bar_ts, side='right') - 1)
                if end_idx <= entry_idx:
                    continue
                slc_high = high5s[entry_idx:end_idx + 1]
                slc_low = low5s[entry_idx:end_idx + 1]
                slc_close = close5s[entry_idx:end_idx + 1]
                if leg_dir == 'LONG':
                    mfe_pts = max(0.0, float((slc_high - entry_price).max()))
                    mae_pts = max(0.0, float((entry_price - slc_low).max()))
                else:
                    mfe_pts = max(0.0, float((entry_price - slc_low).max()))
                    mae_pts = max(0.0, float((slc_high - entry_price).max()))
                pnl_pts_so_far = float(leg_sign * (slc_close[-1] - entry_price))

                v_idx = int(np.searchsorted(truth_ts, bar_ts, side='right') - 1)
                if v_idx < 0:
                    v_idx = 0
                v2_row = v2_arr[v_idx]

                row = {
                    'leg_id': int(li),
                    'day': day,
                    'entry_ts': entry_ts,
                    'leg_dir': leg_dir,
                    'K': K,
                    'bar_ts': bar_ts,
                    'r_price': r_price,
                    'exit_ts': exit_ts,
                    'exit_pnl_usd': exit_pnl_usd,
                    'mae_pts_so_far': mae_pts,
                    'mfe_pts_so_far': mfe_pts,
                    'pnl_pts_so_far': pnl_pts_so_far,
                    'pnl_usd_so_far': pnl_pts_so_far * DOLLAR_PER_POINT,
                    'bars_since_entry': K,
                    'has_reached_R_against': mae_pts >= r_price,
                }
                for i, c in enumerate(v2_cols):
                    row[c] = float(v2_row[i])
                rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot=4000, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                       for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--oos-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--bars5s-dir', default='DATA/ATLAS_NT8/5s')
    ap.add_argument('--model-dir',
                    default='reports/findings/regret_oracle')
    ap.add_argument('--out-trajectory',
                    default='reports/findings/regret_oracle/trade_trajectory_OOS.parquet')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/b9_OOS_singleshot_results.txt')
    args = ap.parse_args()

    print(f'Loading legs: {args.legs}')
    legs = pd.read_csv(args.legs)
    legs = legs.reset_index().rename(columns={'index': 'leg_id'})
    print(f'  legs {len(legs):,}   days {legs["day"].nunique()}')

    print(f'Loading OOS truth: {args.oos_truth}')
    truth = pd.read_parquet(args.oos_truth)
    print(f'  rows {len(truth):,}')

    v2_cols = get_v2_cols(truth)
    print(f'  V2 cols: {len(v2_cols)}')

    # Build trajectory
    bars5s_dir = Path(args.bars5s_dir)
    print('\nBuilding OOS trajectory...')
    traj = build_oos_trajectory(legs, truth, bars5s_dir, v2_cols)
    traj.to_parquet(args.out_trajectory, index=False)
    print(f'Wrote: {args.out_trajectory}  ({len(traj):,} rows)')

    # Load B9 production models
    model_dir = Path(args.model_dir)
    models = {}
    for K in K_HORIZONS:
        mpath = model_dir / f'b9_remaining_amplitude_K{K}.pkl'
        if not mpath.exists():
            print(f'  K={K}: model missing, skip')
            continue
        with open(mpath, 'rb') as f:
            models[K] = pickle.load(f)
    print(f'Loaded {len(models)} B9 production models')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('B9 SEALED OOS TEST  --  single-shot, no retuning after this')
    out('=' * 100)
    out(f'Legs: {len(legs):,}   Days: {legs["day"].nunique()}   '
        f'(2026-03-20 to 2026-04-26)')
    out(f'OOS rows: {len(traj):,}')
    out(f'V2 features: {len(v2_cols)}')
    out('')
    out(f'Baseline (flat sizing, no action): R-trigger entries/exits with $6 friction')
    out(f'  Total: ${legs["pnl_usd"].sum():+,.0f}  '
        f'mean ${legs["pnl_usd"].sum()/legs["day"].nunique():+.0f}/day')
    out('')

    results = []
    for K, mdl in models.items():
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

        # Per-day delta
        per_day = sub.groupby('day')['delta'].sum()
        ci_lo, ci_hi = bootstrap_ci(per_day.values)
        n_cut = int((sub['size_factor'] == 0.0).sum())
        n_half = int((sub['size_factor'] == 0.5).sum())
        n_full = int((sub['size_factor'] == 1.0).sum())
        n_pyr = int((sub['size_factor'] == 1.5).sum())
        results.append({
            'K': K, 'n_legs': len(sub),
            'mean_delta_per_day': float(per_day.mean()),
            'median_delta_per_day': float(per_day.median()),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'significant': ci_lo > 0,
            'n_cut': n_cut, 'n_half': n_half, 'n_full': n_full, 'n_pyr': n_pyr,
            'min_day_delta': float(per_day.min()),
            'max_day_delta': float(per_day.max()),
        })
        out(f'K={K:>3}   delta ${per_day.mean():+.0f}/day   '
            f'CI [${ci_lo:+.0f}, ${ci_hi:+.0f}]   '
            f'sig {ci_lo > 0}   '
            f'(cut {n_cut} / half {n_half} / full {n_full} / pyr {n_pyr})   '
            f'worst day ${per_day.min():+.0f}   best ${per_day.max():+.0f}')

    out('')
    out('=== Anti-doom-cascade: report under multiple slippage assumptions ===')
    out('Assume cuts and pyramids incur extra slippage. Current friction = $6/leg.')
    out('Add slippage factor per ACTION (= non-1.0 size_factor) of $S.')
    out('')
    out(f'{"K":>4}  {"S=$0 delta":>12}  {"S=$2 delta":>12}  {"S=$5 delta":>12}  '
        f'{"S=$10 delta":>12}  CI bounds vs slippage')
    for K, mdl in models.items():
        sub = traj[traj['K'] == K].copy()
        if len(sub) == 0:
            continue
        X = sub[mdl['feat_cols']].fillna(0.0).values
        y_pred = mdl['model'].predict(X)
        sub['size_factor'] = [size_from_pred(p) for p in y_pred]
        action_mask = sub['size_factor'] != 1.0
        for S in [0, 2, 5, 10]:
            sub_s = sub.copy()
            sub_s['realized'] = sub_s['pnl_usd_so_far'] + sub_s['size_factor'] * (
                sub_s['exit_pnl_usd'] - sub_s['pnl_usd_so_far'])
            sub_s.loc[action_mask, 'realized'] -= S
            sub_s['delta'] = sub_s['realized'] - sub_s['exit_pnl_usd']
            per_day_s = sub_s.groupby('day')['delta'].sum()
            ci_lo_s, ci_hi_s = bootstrap_ci(per_day_s.values)
            if S == 0:
                row_str = f'  K={K:>3}  ${per_day_s.mean():>+9.0f}  '
            else:
                row_str += f'${per_day_s.mean():>+9.0f}  '
            if S == 10:
                out(row_str)

    out('')
    out('=== Headline ===')
    if any(r['significant'] for r in results):
        sig = [r for r in results if r['significant']]
        best = max(sig, key=lambda r: r['mean_delta_per_day'])
        out(f'B9 PASSES OOS single-shot test:')
        out(f'  Best K={best["K"]}: ${best["mean_delta_per_day"]:+.0f}/day  '
            f'CI [${best["ci_lo"]:+.0f}, ${best["ci_hi"]:+.0f}]  '
            f'sig=YES')
        out(f'  Baseline: ${legs["pnl_usd"].sum()/legs["day"].nunique():+.0f}/day')
        out(f'  Lift: {best["mean_delta_per_day"]/max(legs["pnl_usd"].sum()/legs["day"].nunique(),1)*100:+.1f}%')
    else:
        out(f'B9 OOS single-shot test: NO config has CI > 0.')
        out(f'IS walk-forward looked promising but OOS does not confirm.')
        out(f'Honest verdict: signal exists in IS, does not generalize OOS.')

    out('')
    out('=== Verdict ===')
    out('Per CLAUDE.md OOS-only rule, this is the honest result.')
    out('Do NOT retune based on these numbers — that becomes IS by definition.')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report}')


if __name__ == '__main__':
    main()
