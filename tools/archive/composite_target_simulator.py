"""Composite target-placement simulator.

Instead of trailing the price (which the prior v2 sims showed can't beat
the R-trigger baseline), this simulator places a LIMIT PROFIT-TARGET
at entry + factor*R for LONG legs (or entry - factor*R for SHORT legs).

If the target is hit during the leg, exit at the target price.
Otherwise fall back to the R-trigger exit (same as baseline).

Edge sources:
  - Target hit -> we exit AT a price ABOVE running_extreme - R (better than baseline)
  - Target not hit -> identical to baseline

Failure modes:
  - Target hit but leg keeps extending -> we left money on the table (target too tight)
  - Target never hit -> no different from baseline (target too wide)

So the question is: WHAT FRACTION OF R gives a positive expected edge?

Two variants tested:

  STATIC: target = entry + factor*R, fixed for the whole leg
  ZONE-AWARE: target multiplier depends on the composite zone seen during
              the leg. When zone says pivot-near, reduce target (expect
              less extension); when zone says clear, keep wide target.

Sweep factor ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} of R.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')
REGIME_CSV = Path('DATA/ATLAS/regime_labels_2d.csv')

PIVOT_ZONES = {'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m',
               'IMMINENT', 'AT_PIVOT', 'WIDE_ZONE'}


def load_regime_label(day_str: str):
    if not REGIME_CSV.exists():
        return None
    df = pd.read_csv(REGIME_CSV, usecols=['date', 'regime_2d'])
    iso_day = day_str.replace('_', '-')
    row = df[df['date'] == iso_day]
    if len(row) == 0:
        return None
    return str(row['regime_2d'].iloc[0])


def derive_pivot_events(truth_day):
    piv = truth_day[truth_day['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return []
    ts = piv['timestamp'].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    pp_ = piv['pivot_price'].values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for grp in groups:
        ts_c = int(np.median(ts[grp]))
        vals, counts = np.unique(pd_[grp], return_counts=True)
        d = str(vals[np.argmax(counts)])
        p = float(np.mean(pp_[grp]))
        out.append((ts_c, d, p))
    return out


def simulate_leg_target(closes5s, ts5s, bars1m_ts, zone_per_1m,
                         entry_ts, entry_price, leg_dir, end_ts, r_price,
                         target_factor, zone_aware=False):
    """Simulate baseline R-trigger AND target-placement exit.

    STATIC target: entry + target_factor * R (for LONG) or - (SHORT)
    ZONE-AWARE: target_factor reduced when zone is in PIVOT_ZONES
    """
    mask = (ts5s >= entry_ts) & (ts5s <= end_ts)
    closes_leg = closes5s[mask]
    ts_leg = ts5s[mask]
    if len(closes_leg) == 0:
        return None

    # Running extreme
    if leg_dir == 'LONG':
        run_ext = np.maximum.accumulate(closes_leg)
    else:
        run_ext = np.minimum.accumulate(closes_leg)

    # === BASELINE R-trigger ===
    baseline_trail = run_ext - r_price if leg_dir == 'LONG' else run_ext + r_price
    if leg_dir == 'LONG':
        baseline_hit = np.where(closes_leg <= baseline_trail)[0]
    else:
        baseline_hit = np.where(closes_leg >= baseline_trail)[0]
    baseline_exit_idx = int(baseline_hit[0]) if len(baseline_hit) > 0 else len(closes_leg) - 1
    baseline_exit_price = float(closes_leg[baseline_exit_idx])
    baseline_exit_ts = int(ts_leg[baseline_exit_idx])

    # === TARGET ===
    if zone_aware:
        # Use latest zone per 5s bar to scale the effective target_factor
        leg_1m_mask = (bars1m_ts >= entry_ts) & (bars1m_ts <= end_ts)
        leg_1m_ts = bars1m_ts[leg_1m_mask]
        leg_1m_zone = zone_per_1m[leg_1m_mask]
        if len(leg_1m_ts) > 0:
            idx_1m = np.searchsorted(leg_1m_ts, ts_leg, side='right') - 1
            idx_1m = np.clip(idx_1m, 0, len(leg_1m_zone) - 1)
            zones_5s = leg_1m_zone[idx_1m]
        else:
            zones_5s = np.array(['CLEAR'] * len(ts_leg), dtype=object)
        # Effective factor: scaled down in pivot zones
        in_pivot_zone = np.array([str(z) in PIVOT_ZONES for z in zones_5s])
        eff_factor = np.where(in_pivot_zone, target_factor * 0.5, target_factor)
        # Target = entry + factor*R for LONG
        if leg_dir == 'LONG':
            target_per_5s = entry_price + eff_factor * r_price
            hit_idx = np.where(closes_leg >= target_per_5s)[0]
        else:
            target_per_5s = entry_price - eff_factor * r_price
            hit_idx = np.where(closes_leg <= target_per_5s)[0]
    else:
        # Static target
        if leg_dir == 'LONG':
            target_price = entry_price + target_factor * r_price
            hit_idx = np.where(closes_leg >= target_price)[0]
        else:
            target_price = entry_price - target_factor * r_price
            hit_idx = np.where(closes_leg <= target_price)[0]
        target_per_5s = target_price

    if len(hit_idx) > 0:
        target_hit_idx = int(hit_idx[0])
        # Did target hit BEFORE R-trigger fired?
        if target_hit_idx <= baseline_exit_idx:
            # Use target exit
            target_exit_idx = target_hit_idx
            if isinstance(target_per_5s, np.ndarray):
                target_exit_price = float(target_per_5s[target_hit_idx])
            else:
                target_exit_price = float(target_per_5s)
            target_exit_ts = int(ts_leg[target_hit_idx])
            target_hit_first = True
        else:
            # R-trigger fired first -> fallback to baseline
            target_exit_idx = baseline_exit_idx
            target_exit_price = baseline_exit_price
            target_exit_ts = baseline_exit_ts
            target_hit_first = False
    else:
        # Target never hit -> use baseline R-trigger
        target_exit_idx = baseline_exit_idx
        target_exit_price = baseline_exit_price
        target_exit_ts = baseline_exit_ts
        target_hit_first = False

    # Edge: positive if target captured more of the move
    if leg_dir == 'LONG':
        edge_pts = target_exit_price - baseline_exit_price
    else:
        edge_pts = baseline_exit_price - target_exit_price

    return {
        'entry_ts': int(entry_ts), 'entry_price': float(entry_price),
        'leg_dir': leg_dir, 'r_price': float(r_price),
        'leg_end_ts': int(end_ts),
        'baseline_exit_ts': baseline_exit_ts,
        'baseline_exit_price': baseline_exit_price,
        'target_exit_ts': target_exit_ts,
        'target_exit_price': target_exit_price,
        'target_hit_first': target_hit_first,
        'target_factor': target_factor,
        'zone_aware': zone_aware,
        'edge_pts': float(edge_pts),
        'edge_usd': float(edge_pts * DOLLAR_PER_POINT),
    }


def run_one_config(truth, cloud, target_factor, zone_aware=False):
    """Run target-sim with one config across all OOS days."""
    rows = []
    for day in sorted(truth['day'].unique()):
        bars1m_path = NT8_1M_DIR / f'{day}.parquet'
        bars5s_path = NT8_5S_DIR / f'{day}.parquet'
        if not bars1m_path.exists() or not bars5s_path.exists():
            continue
        bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE

        truth_day = truth[truth['day'] == day]
        cloud_day = cloud[cloud['day'] == day].sort_values('timestamp')
        bars1m_ts = bars1m['timestamp'].values.astype(np.int64)
        zone_per_1m_lookup = dict(zip(cloud_day['timestamp'].values.astype(np.int64),
                                       cloud_day['zone'].values))
        zone_per_1m = np.array([zone_per_1m_lookup.get(int(ts), 'CLEAR')
                                  for ts in bars1m_ts], dtype=object)

        events = derive_pivot_events(truth_day)
        if len(events) < 2:
            continue
        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        for k in range(len(events) - 1):
            entry_ts, leg_dir, entry_price = events[k]
            next_ts = events[k + 1][0]
            r = simulate_leg_target(closes5s, ts5s, bars1m_ts, zone_per_1m,
                                      entry_ts, entry_price, leg_dir, next_ts, r_price,
                                      target_factor, zone_aware)
            if r is None:
                continue
            r['day'] = day
            r['atr_pts'] = atr_pts
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--cloud',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.parquet')
    ap.add_argument('--out-prefix',
                    default='reports/findings/regret_oracle/composite_target_sim')
    args = ap.parse_args()

    print('Loading...')
    truth = pd.read_parquet(args.truth)
    cloud = pd.read_parquet(args.cloud)
    print(f'  truth {len(truth):,}   cloud {len(cloud):,}')

    factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = []
    print('Sweeping STATIC target placement...')
    for f in tqdm(factors, desc='static'):
        df = run_one_config(truth, cloud, target_factor=f, zone_aware=False)
        edge = df['edge_usd'].dropna().values
        per_day = df.groupby('day')['edge_usd'].sum()
        rng = np.random.default_rng(42)
        boots = np.array([edge[rng.integers(0, len(edge), len(edge))].mean()
                           for _ in range(4000)])
        boots_d = np.array([per_day.values[rng.integers(0, len(per_day), len(per_day))].mean()
                             for _ in range(4000)])
        results.append({
            'mode': 'STATIC',
            'target_factor': f,
            'n_legs': len(df),
            'target_hit_rate': float(df['target_hit_first'].mean()),
            'edge_mean_per_leg': float(edge.mean()),
            'edge_median_per_leg': float(np.median(edge)),
            'edge_ci_lo': float(np.percentile(boots, 2.5)),
            'edge_ci_hi': float(np.percentile(boots, 97.5)),
            'pos_pct': float((edge > 0).mean()),
            'neg_pct': float((edge < 0).mean()),
            'zero_pct': float((edge == 0).mean()),
            'mean_per_day': float(per_day.mean()),
            'median_per_day': float(per_day.median()),
            'per_day_ci_lo': float(np.percentile(boots_d, 2.5)),
            'per_day_ci_hi': float(np.percentile(boots_d, 97.5)),
            'positive_days': int((per_day > 0).sum()),
            'total_days': int(len(per_day)),
        })

    print('Sweeping ZONE-AWARE target placement...')
    for f in tqdm(factors, desc='zone-aware'):
        df = run_one_config(truth, cloud, target_factor=f, zone_aware=True)
        edge = df['edge_usd'].dropna().values
        per_day = df.groupby('day')['edge_usd'].sum()
        rng = np.random.default_rng(42)
        boots = np.array([edge[rng.integers(0, len(edge), len(edge))].mean()
                           for _ in range(4000)])
        boots_d = np.array([per_day.values[rng.integers(0, len(per_day), len(per_day))].mean()
                             for _ in range(4000)])
        results.append({
            'mode': 'ZONE_AWARE',
            'target_factor': f,
            'n_legs': len(df),
            'target_hit_rate': float(df['target_hit_first'].mean()),
            'edge_mean_per_leg': float(edge.mean()),
            'edge_median_per_leg': float(np.median(edge)),
            'edge_ci_lo': float(np.percentile(boots, 2.5)),
            'edge_ci_hi': float(np.percentile(boots, 97.5)),
            'pos_pct': float((edge > 0).mean()),
            'neg_pct': float((edge < 0).mean()),
            'zero_pct': float((edge == 0).mean()),
            'mean_per_day': float(per_day.mean()),
            'median_per_day': float(per_day.median()),
            'per_day_ci_lo': float(np.percentile(boots_d, 2.5)),
            'per_day_ci_hi': float(np.percentile(boots_d, 97.5)),
            'positive_days': int((per_day > 0).sum()),
            'total_days': int(len(per_day)),
        })

    sweep_df = pd.DataFrame(results)
    sweep_df = sweep_df.sort_values('mean_per_day', ascending=False)
    sweep_df.to_csv(f'{args.out_prefix}_sweep.csv', index=False)

    print('\n' + '=' * 78)
    print('TARGET-PLACEMENT SWEEP (NT8 OOS, 32 days)')
    print('=' * 78)
    print('Sorted by mean_per_day descending. Baseline = R-trigger only (edge = 0).')
    print()
    cols = ['mode', 'target_factor', 'n_legs', 'target_hit_rate',
            'edge_mean_per_leg', 'edge_ci_lo', 'edge_ci_hi',
            'pos_pct', 'neg_pct', 'zero_pct',
            'mean_per_day', 'per_day_ci_lo', 'per_day_ci_hi', 'positive_days']
    print(sweep_df[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # Highlight any positive-edge result
    pos = sweep_df[sweep_df['edge_ci_lo'] > 0]
    print()
    print(f'Configs with statistically POSITIVE edge (CI strictly > 0): {len(pos)}')
    if len(pos) > 0:
        print(pos[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
