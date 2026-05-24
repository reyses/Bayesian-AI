"""Composite trail simulator v2 — refined with B6 directional + hysteresis.

Naive v1 (composite_trail_simulator.py) tightened the trail whenever the
composite zone said "pivot proximity." Result: -$44/leg, NEAR_PIVOT zone
caused 47% of exits at -$67.60 each because most "near pivot" calls were
just normal pullbacks during a still-running leg.

v2 ADDS TWO GATES:

  GATE 1 — Directional agreement (B6):
    If we're in a LONG leg, only tighten when B6 P(PIVOT_TO_SHORT) >= b6_thr
    (B6 sees a HIGH/SHORT-pivot coming, which would end our LONG leg)
    If we're in a SHORT leg, only tighten when B6 P(PIVOT_TO_LONG) >= b6_thr

  GATE 2 — Hysteresis (consecutive-bar sustained signal):
    Only tighten after N consecutive 1m bars meet BOTH the zone proximity
    condition AND the B6 directional condition. Single-bar false alarms
    don't trigger tightening.

Idea: most of the -$67/leg loss in v1 came from one-bar NEAR_PIVOT pulses
during pullbacks. Requiring multi-bar persistence + directional agreement
should filter most false alarms while preserving real pivot warnings.

Outputs per leg + aggregate report + per-day charts.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0   # MNQ
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')
REGIME_CSV = Path('DATA/ATLAS/regime_labels_2d.csv')
SEGMENT_MIN = 90

# Zones considered "pivot proximity" (Gate 1 candidate)
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


def derive_pivot_events(truth_day: pd.DataFrame):
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


def simulate_leg(closes5s, ts5s, bars1m_ts, zone_per_1m,
                  b6_long_per_1m, b6_short_per_1m,
                  entry_ts, entry_price, leg_dir, end_ts, r_price,
                  b6_thr, hysteresis, tight_mult):
    """Simulate baseline R-trigger exit AND refined trail with gates.

    Refined logic:
      For each 1m bar in the leg, compute should_tighten_1m:
        condition = (zone in PIVOT_ZONES) AND (B6 directional match)
        sustained_count = consecutive bars where condition is True
        should_tighten = (sustained_count >= hysteresis)
      Map per 5s bar -> 1m bar -> should_tighten flag.
      For 5s bars in 'tighten' state, trail_dist = R * tight_mult.
      Otherwise trail_dist = R (wide).
    """
    mask = (ts5s >= entry_ts) & (ts5s <= end_ts)
    closes_leg = closes5s[mask]
    ts_leg = ts5s[mask]
    if len(closes_leg) == 0:
        return None

    # Running extreme during leg
    if leg_dir == 'LONG':
        run_ext = np.maximum.accumulate(closes_leg)
    else:
        run_ext = np.minimum.accumulate(closes_leg)

    # === BASELINE ===
    baseline_trail = run_ext - r_price if leg_dir == 'LONG' else run_ext + r_price
    if leg_dir == 'LONG':
        baseline_hit = np.where(closes_leg <= baseline_trail)[0]
    else:
        baseline_hit = np.where(closes_leg >= baseline_trail)[0]
    if len(baseline_hit) == 0:
        baseline_exit_idx = len(closes_leg) - 1
    else:
        baseline_exit_idx = int(baseline_hit[0])
    baseline_exit_price = float(closes_leg[baseline_exit_idx])
    baseline_exit_ts    = int(ts_leg[baseline_exit_idx])

    # === REFINED: per-1m bar should-tighten signal ===
    leg_1m_mask = (bars1m_ts >= entry_ts) & (bars1m_ts <= end_ts)
    leg_1m_ts = bars1m_ts[leg_1m_mask]
    leg_1m_zone = zone_per_1m[leg_1m_mask]
    leg_1m_b6_long = b6_long_per_1m[leg_1m_mask]
    leg_1m_b6_short = b6_short_per_1m[leg_1m_mask]

    # Per-bar gate: zone proximity AND directional agreement
    per_bar_cond = np.zeros(len(leg_1m_ts), dtype=bool)
    for i, z in enumerate(leg_1m_zone):
        if str(z) not in PIVOT_ZONES:
            continue
        if leg_dir == 'LONG':
            # Want SHORT-pivot incoming (top expected)
            if leg_1m_b6_short[i] >= b6_thr:
                per_bar_cond[i] = True
        else:
            # Want LONG-pivot incoming (bottom expected)
            if leg_1m_b6_long[i] >= b6_thr:
                per_bar_cond[i] = True

    # Sustained: count consecutive True bars
    sustained = np.zeros(len(per_bar_cond), dtype=bool)
    count = 0
    for i, c in enumerate(per_bar_cond):
        count = count + 1 if c else 0
        sustained[i] = (count >= hysteresis)

    # Map to 5s bars: for each 5s bar, find latest 1m bar at-or-before
    if len(leg_1m_ts) > 0:
        idx_1m = np.searchsorted(leg_1m_ts, ts_leg, side='right') - 1
        idx_1m = np.clip(idx_1m, 0, len(sustained) - 1)
        sustained_per_5s = sustained[idx_1m]
    else:
        sustained_per_5s = np.zeros(len(ts_leg), dtype=bool)

    # Trail distance per 5s bar
    trail_dist = np.where(sustained_per_5s, r_price * tight_mult, r_price)
    refined_trail = run_ext - trail_dist if leg_dir == 'LONG' else run_ext + trail_dist

    if leg_dir == 'LONG':
        refined_hit = np.where(closes_leg <= refined_trail)[0]
    else:
        refined_hit = np.where(closes_leg >= refined_trail)[0]
    if len(refined_hit) == 0:
        refined_exit_idx = len(closes_leg) - 1
    else:
        refined_exit_idx = int(refined_hit[0])
    refined_exit_price = float(closes_leg[refined_exit_idx])
    refined_exit_ts = int(ts_leg[refined_exit_idx])

    # Track state at refined exit
    refined_exit_was_tight = bool(sustained_per_5s[refined_exit_idx])

    # Edge: positive if refined captured MORE of the move
    if leg_dir == 'LONG':
        edge_pts = refined_exit_price - baseline_exit_price
    else:
        edge_pts = baseline_exit_price - refined_exit_price

    return {
        'entry_ts': int(entry_ts), 'entry_price': float(entry_price),
        'leg_dir': leg_dir, 'r_price': float(r_price),
        'leg_end_ts': int(end_ts),
        'n_1m_bars_in_leg': int(len(leg_1m_ts)),
        'n_tightened_1m_bars': int(sustained.sum()),
        'baseline_exit_ts': baseline_exit_ts,
        'baseline_exit_price': baseline_exit_price,
        'refined_exit_ts': refined_exit_ts,
        'refined_exit_price': refined_exit_price,
        'refined_exit_was_tight': refined_exit_was_tight,
        'edge_pts': float(edge_pts),
        'edge_usd': float(edge_pts * DOLLAR_PER_POINT),
    }


def run_one_config(truth, cloud, b6, b6_thr, hysteresis, tight_mult,
                   K_for_b6=10):
    """Run the v2 simulator with one config across all OOS days. Returns
    DataFrame of per-leg results."""
    p_long_col  = f'p_PIVOT_TO_LONG_{K_for_b6}m'
    p_short_col = f'p_PIVOT_TO_SHORT_{K_for_b6}m'
    rows = []
    days_all = sorted(truth['day'].unique())
    for day in days_all:
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
        b6_day = b6[b6['day'] == day].sort_values('timestamp')
        bars1m_ts = bars1m['timestamp'].values.astype(np.int64)

        # Align cloud zone to 1m bar grid
        zone_per_1m_lookup = dict(zip(cloud_day['timestamp'].values.astype(np.int64),
                                       cloud_day['zone'].values))
        zone_per_1m = np.array([zone_per_1m_lookup.get(int(ts), 'CLEAR')
                                  for ts in bars1m_ts], dtype=object)
        # Align B6 directional probs to 1m bar grid
        b6_long_lookup  = dict(zip(b6_day['timestamp'].values.astype(np.int64),
                                    b6_day[p_long_col].values))
        b6_short_lookup = dict(zip(b6_day['timestamp'].values.astype(np.int64),
                                    b6_day[p_short_col].values))
        b6_long_per_1m  = np.array([b6_long_lookup.get(int(ts), 0.0)
                                     for ts in bars1m_ts], dtype=np.float64)
        b6_short_per_1m = np.array([b6_short_lookup.get(int(ts), 0.0)
                                     for ts in bars1m_ts], dtype=np.float64)

        events = derive_pivot_events(truth_day)
        if len(events) < 2:
            continue
        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        for k in range(len(events) - 1):
            entry_ts, leg_dir, entry_price = events[k]
            next_ts = events[k + 1][0]
            r = simulate_leg(closes5s, ts5s, bars1m_ts, zone_per_1m,
                              b6_long_per_1m, b6_short_per_1m,
                              entry_ts, entry_price, leg_dir, next_ts, r_price,
                              b6_thr, hysteresis, tight_mult)
            if r is None:
                continue
            r['day'] = day
            r['atr_pts'] = atr_pts
            r['b6_thr'] = b6_thr
            r['hysteresis'] = hysteresis
            r['tight_mult'] = tight_mult
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--cloud',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.parquet')
    ap.add_argument('--b6',
                    default='reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')
    ap.add_argument('--K', type=int, default=10,
                    help='B6 K horizon to use (default 10)')
    ap.add_argument('--sweep', action='store_true',
                    help='Sweep configs instead of a single run')
    ap.add_argument('--b6-thr', type=float, default=0.50)
    ap.add_argument('--hysteresis', type=int, default=3)
    ap.add_argument('--tight-mult', type=float, default=0.30)
    ap.add_argument('--out-prefix',
                    default='reports/findings/regret_oracle/composite_trail_sim_v2')
    args = ap.parse_args()

    print('Loading inputs...')
    truth = pd.read_parquet(args.truth)
    cloud = pd.read_parquet(args.cloud)
    b6 = pd.read_parquet(args.b6)
    print(f'  truth {len(truth):,}   cloud {len(cloud):,}   b6 {len(b6):,}')

    if not args.sweep:
        # Single config run
        print(f'Running single config: b6_thr={args.b6_thr} '
              f'hysteresis={args.hysteresis} tight_mult={args.tight_mult}')
        df = run_one_config(truth, cloud, b6,
                             args.b6_thr, args.hysteresis, args.tight_mult,
                             K_for_b6=args.K)

        edge = df['edge_usd'].dropna().values
        rng = np.random.default_rng(42)
        boots = np.array([edge[rng.integers(0, len(edge), len(edge))].mean()
                           for _ in range(4000)])
        mean = edge.mean(); ci_lo = np.percentile(boots, 2.5); ci_hi = np.percentile(boots, 97.5)

        # Per-day
        pd_per_day = df.groupby('day').agg(
            n_legs=('edge_usd', 'count'),
            sum_edge=('edge_usd', 'sum'),
        ).reset_index()
        rng = np.random.default_rng(42)
        sum_d = pd_per_day['sum_edge'].values
        boots_d = np.array([sum_d[rng.integers(0, len(sum_d), len(sum_d))].mean()
                             for _ in range(4000)])

        print('=' * 78)
        print(f'COMPOSITE TRAIL SIM v2 (B6 directional + hysteresis gate)')
        print(f'  b6_thr={args.b6_thr}  hysteresis={args.hysteresis}  '
              f'tight_mult={args.tight_mult}  K={args.K}m')
        print('=' * 78)
        print(f'Legs: {len(df):,}   Days: {df["day"].nunique()}')
        print(f'Edge per leg: mean ${mean:+.2f}   '
              f'95% CI [${ci_lo:+.2f}, ${ci_hi:+.2f}]')
        print(f'  positive: {(edge > 0).mean()*100:.1f}%   '
              f'negative: {(edge < 0).mean()*100:.1f}%   '
              f'zero: {(edge == 0).mean()*100:.1f}%')
        print(f'Per-day edge: mean ${pd_per_day["sum_edge"].mean():+.2f}/day   '
              f'CI [${np.percentile(boots_d, 2.5):+.2f}, ${np.percentile(boots_d, 97.5):+.2f}]   '
              f'positive days: {(pd_per_day["sum_edge"] > 0).sum()}/{len(pd_per_day)}')
        print(f'Tighten activations: {df["n_tightened_1m_bars"].sum():,} 1m bars '
              f'across {df["n_1m_bars_in_leg"].sum():,} leg-bars '
              f'({df["n_tightened_1m_bars"].sum() / max(df["n_1m_bars_in_leg"].sum(), 1) * 100:.2f}%)')
        print(f'Refined exit tight-at-exit: {df["refined_exit_was_tight"].sum()} '
              f'({df["refined_exit_was_tight"].mean()*100:.1f}% of legs)')

        # Compare to v1 baseline (-$43.98/leg)
        print()
        print(f'  vs v1 (naive, no gate):  -$43.98/leg  [-$47.29, -$40.89]')
        improvement = mean - (-43.98)
        print(f'  v2 vs v1 improvement:    ${improvement:+.2f}/leg')

        df.to_csv(f'{args.out_prefix}.csv', index=False)
        pd_per_day.to_csv(f'{args.out_prefix}.per_day.csv', index=False)

    else:
        # Parameter sweep
        configs = []
        for b6_thr in [0.50, 0.60, 0.70]:
            for hyst in [1, 3, 5]:
                for tm in [0.30, 0.50]:
                    configs.append((b6_thr, hyst, tm))
        results = []
        print(f'Sweeping {len(configs)} configs...')
        for b6_thr, hyst, tm in tqdm(configs):
            df = run_one_config(truth, cloud, b6, b6_thr, hyst, tm,
                                  K_for_b6=args.K)
            edge = df['edge_usd'].dropna().values
            rng = np.random.default_rng(42)
            boots = np.array([edge[rng.integers(0, len(edge), len(edge))].mean()
                               for _ in range(2000)])
            per_day = df.groupby('day')['edge_usd'].sum()
            results.append({
                'b6_thr': b6_thr,
                'hysteresis': hyst,
                'tight_mult': tm,
                'n_legs': len(df),
                'edge_mean_per_leg': float(edge.mean()),
                'edge_ci_lo': float(np.percentile(boots, 2.5)),
                'edge_ci_hi': float(np.percentile(boots, 97.5)),
                'pos_pct': float((edge > 0).mean()),
                'neg_pct': float((edge < 0).mean()),
                'zero_pct': float((edge == 0).mean()),
                'mean_per_day': float(per_day.mean()),
                'median_per_day': float(per_day.median()),
                'positive_days': int((per_day > 0).sum()),
                'total_days': int(len(per_day)),
                'tighten_activation_rate': float(
                    df['n_tightened_1m_bars'].sum() /
                    max(df['n_1m_bars_in_leg'].sum(), 1)
                ),
            })
        sweep_df = pd.DataFrame(results)
        sweep_df = sweep_df.sort_values('mean_per_day', ascending=False)
        sweep_df.to_csv(f'{args.out_prefix}_sweep.csv', index=False)

        print('\n=== SWEEP RESULTS (sorted by mean_per_day, desc) ===')
        cols = ['b6_thr', 'hysteresis', 'tight_mult', 'n_legs',
                'edge_mean_per_leg', 'edge_ci_lo', 'edge_ci_hi',
                'pos_pct', 'mean_per_day', 'positive_days',
                'tighten_activation_rate']
        print(sweep_df[cols].to_string(index=False))

        print()
        print('v1 baseline (no gate): -$43.98/leg, -$2,510/day, 2/32 positive days')


if __name__ == '__main__':
    main()
