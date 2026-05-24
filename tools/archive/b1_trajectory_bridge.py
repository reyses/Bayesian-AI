"""B1 trajectory analyzer + zigzag bridge composite.

The "10-min gap" problem: B1 K=10 says "pivot within 10 min" but that's a
wide window. As we walk forward bar-by-bar, B1's prediction at each new
bar SHOULD decay or sharpen based on the underlying conditions:

  - If B1 sustains HIGH across consecutive bars     -> pivot truly imminent (1-3 min)
  - If B1 keeps RISING from medium to high          -> pivot building (3-7 min)
  - If B1 DECAYS from a peak                        -> false alarm passing
  - If B1 stays LOW                                  -> no pivot expected

The TIME EVOLUTION of B1 is itself the bridge between a 10-min prediction
and a 2-3 min one.

This script:
  1. Loads B1 per-bar probabilities (K=10) from precomputed cache
  2. Computes trajectory features over a trailing N-bar window
  3. Classifies each bar into a status: CLEAR / WATCH / BUILDING / IMMINENT / PASSING
  4. Validates against actual time-to-next-pivot from the truth dataset
  5. Builds the composite: live_zz_dir + status + B2 fakeout score
  6. Outputs per-bar parquet and a validation report

Per user 2026-05-17: NO TRADE MANAGEMENT. Pure signal validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import live_zigzag_direction, compute_atr, TICK_SIZE


TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')

# Trajectory window (1m bars to look back)
TRAJ_N = 10

# Status thresholds (on B1 K=10m probability)
T_LOW       = 0.30   # below = CLEAR
T_WATCH     = 0.50   # rising past this = WATCH
T_HIGH      = 0.85   # at/above sustained = IMMINENT


def trajectory_features(p_arr: np.ndarray, N: int = TRAJ_N) -> dict:
    """For each bar, compute features over the past N bars of B1 prob.

    Returns a dict of arrays (same length as input):
      p_now           : current probability
      p_max_N         : max in trailing window
      p_min_N         : min in trailing window
      p_slope_N       : linear-regression slope over window (1/bar units)
      bars_above_watch: # bars in last N where p > T_WATCH (run-length proxy)
      bars_above_high : # bars in last N where p > T_HIGH
      decaying        : True if p_now < 0.7 * p_max_N (peak passed)
      peak_age_bars   : bars since p hit its max in the window
    """
    n = len(p_arr)
    p_max_N    = np.full(n, np.nan)
    p_min_N    = np.full(n, np.nan)
    p_slope_N  = np.full(n, np.nan)
    bars_above_watch = np.zeros(n, dtype=np.int16)
    bars_above_high  = np.zeros(n, dtype=np.int16)
    decaying   = np.zeros(n, dtype=bool)
    peak_age   = np.zeros(n, dtype=np.int16)
    for i in range(n):
        lo = max(0, i - N + 1)
        window = p_arr[lo:i+1]
        p_max_N[i] = float(window.max())
        p_min_N[i] = float(window.min())
        # Slope via least-squares over [0, len-1]
        if len(window) >= 2:
            xs = np.arange(len(window), dtype=np.float64)
            xm = xs.mean(); ym = window.mean()
            num = np.sum((xs - xm) * (window - ym))
            den = np.sum((xs - xm) ** 2)
            p_slope_N[i] = float(num / den) if den > 0 else 0.0
        bars_above_watch[i] = int((window > T_WATCH).sum())
        bars_above_high[i]  = int((window > T_HIGH).sum())
        # Decaying = current is well below window max
        if p_max_N[i] > 0.30 and p_arr[i] < 0.7 * p_max_N[i]:
            decaying[i] = True
        # Peak age in bars
        peak_age[i] = int((len(window) - 1) - np.argmax(window))
    return {
        'p_now': p_arr.copy(),
        'p_max_N': p_max_N,
        'p_min_N': p_min_N,
        'p_slope_N': p_slope_N,
        'bars_above_watch': bars_above_watch,
        'bars_above_high': bars_above_high,
        'decaying': decaying,
        'peak_age_bars': peak_age,
    }


def classify_status(p_now, p_max_N, p_slope_N, bars_above_high, decaying):
    """Rule-based status classifier per bar.

    CLEAR    : p_now < T_LOW, no recent high
    WATCH    : p_now >= T_LOW, slope positive — entering pivot zone
    BUILDING : p_now >= T_WATCH and rising (slope > 0 or sustained)
    IMMINENT : sustained high (bars_above_high >= 2) — pivot next 1-3 min
    PASSING  : recent peak but decaying — false alarm or pivot already happened
    """
    n = len(p_now)
    status = np.full(n, 'CLEAR', dtype=object)
    for i in range(n):
        p = p_now[i]
        pmax = p_max_N[i]
        slope = p_slope_N[i]
        n_high = bars_above_high[i]
        if n_high >= 2 and p >= T_HIGH:
            status[i] = 'IMMINENT'
        elif decaying[i]:
            status[i] = 'PASSING'
        elif p >= T_WATCH and (slope > 0 or pmax > T_HIGH):
            status[i] = 'BUILDING'
        elif p >= T_LOW:
            status[i] = 'WATCH'
        else:
            status[i] = 'CLEAR'
    return status


def derive_pivot_centroids_per_day(truth_df: pd.DataFrame) -> dict:
    """Return {day: sorted np.array of pivot centroid ts}."""
    out = {}
    for day, g in truth_df.groupby('day'):
        piv = g[g['is_pivot'] == 1].sort_values('timestamp')
        if len(piv) == 0:
            out[day] = np.array([], dtype=np.int64); continue
        ts = piv['timestamp'].values.astype(np.int64)
        groups = [[ts[0]]]
        for i in range(1, len(ts)):
            if ts[i] - ts[i-1] > 90:
                groups.append([ts[i]])
            else:
                groups[-1].append(ts[i])
        out[day] = np.array([int(np.median(g)) for g in groups], dtype=np.int64)
    return out


def time_to_next_pivot(bar_ts: np.ndarray, pivots: np.ndarray):
    """For each bar ts, return seconds to the next pivot at or after the
    bar (NaN if no future pivot)."""
    out = np.full(len(bar_ts), np.nan)
    if len(pivots) == 0:
        return out
    idx = np.searchsorted(pivots, bar_ts, side='left')
    for i, k in enumerate(idx):
        if k < len(pivots):
            out[i] = float(pivots[k] - bar_ts[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1-cache',
                    default='reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
    ap.add_argument('--b2-cache',
                    default='reports/findings/regret_oracle/b2_proba_OOS_NT8.parquet')
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--K', type=int, default=10,
                    help='Which B1 K to track (default 10)')
    ap.add_argument('--out-parquet',
                    default='reports/findings/regret_oracle/b1_trajectory_bridge.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b1_trajectory_bridge.txt')
    args = ap.parse_args()

    print('Loading caches...')
    b1 = pd.read_parquet(args.b1_cache)
    b2 = pd.read_parquet(args.b2_cache)
    tr = pd.read_parquet(args.truth)
    print(f'  B1: {len(b1)}   B2: {len(b2)}   truth: {len(tr)}')

    K = args.K
    p_col = f'p_pivot_{K}m'
    if p_col not in b1.columns:
        raise SystemExit(f'{p_col} not in B1 cache')

    # Sort and merge B1 onto truth
    b1 = b1.sort_values(['day', 'timestamp']).reset_index(drop=True)
    tr = tr.sort_values(['day', 'timestamp']).reset_index(drop=True)
    merged = tr[['timestamp', 'day', 'pivot_dir', 'is_pivot']].merge(
        b1[['timestamp', 'day', p_col]],
        on=['timestamp', 'day'], how='inner',
    )
    print(f'  merged rows: {len(merged)}')

    # Compute trajectory features per day
    print('Computing B1 trajectory features per day...')
    all_feats = []
    for day, g in tqdm(merged.groupby('day'), desc='days'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        p_arr = g[p_col].values.astype(np.float64)
        feats = trajectory_features(p_arr, N=TRAJ_N)
        for k, v in feats.items():
            g[k] = v
        all_feats.append(g)
    merged = pd.concat(all_feats, ignore_index=True)

    # Classify status
    merged['status'] = classify_status(
        merged['p_now'].values,
        merged['p_max_N'].values,
        merged['p_slope_N'].values,
        merged['bars_above_high'].values,
        merged['decaying'].values,
    )

    # Compute live ZZ direction per bar
    print('Computing live zigzag per day...')
    live_dir_arr = np.zeros(len(merged), dtype=np.int8)
    for day, g in tqdm(merged.groupby('day'), desc='live zz'):
        p1m = NT8_1M_DIR / f'{day}.parquet'
        p5s = NT8_5S_DIR / f'{day}.parquet'
        if not p1m.exists() or not p5s.exists():
            continue
        b1m = pd.read_parquet(p1m).sort_values('timestamp').reset_index(drop=True)
        b5s = pd.read_parquet(p5s).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(b1m, 14)
        min_rev = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        closes5s = b5s['close'].values.astype(np.float64)
        ts5s_arr = b5s['timestamp'].values.astype(np.int64)
        live_per_5s = live_zigzag_direction(closes5s, min_rev, 36)
        day_mask = (merged['day'] == day).values
        merge_ts = merged.loc[day_mask, 'timestamp'].values.astype(np.int64)
        idx5s = np.searchsorted(ts5s_arr, merge_ts, side='right') - 1
        idx5s = np.clip(idx5s, 0, len(ts5s_arr) - 1)
        live_dir_arr[day_mask] = live_per_5s[idx5s]
    merged['live_zz_dir'] = np.where(live_dir_arr == 1, 'LONG',
                              np.where(live_dir_arr == -1, 'SHORT', 'NEUTRAL'))

    # Time-to-next-pivot (ground truth)
    print('Computing actual time-to-next-pivot...')
    pivots_per_day = derive_pivot_centroids_per_day(tr)
    ttn = np.full(len(merged), np.nan)
    for day, g in merged.groupby('day'):
        piv = pivots_per_day.get(day, np.array([], dtype=np.int64))
        bar_ts = g['timestamp'].values.astype(np.int64)
        ttn[g.index] = time_to_next_pivot(bar_ts, piv)
    merged['ttn_seconds'] = ttn
    merged['ttn_minutes'] = ttn / 60.0

    # B2 fakeout score attached to most recent pivot
    print('Attaching B2 fakeout score to bars...')
    b2 = b2.sort_values(['day', 'timestamp']).reset_index(drop=True)
    fake_attached = np.full(len(merged), np.nan)
    for day, g in merged.groupby('day'):
        b2d = b2[b2['day'] == day].sort_values('timestamp')
        if len(b2d) == 0:
            continue
        bar_ts = g['timestamp'].values.astype(np.int64)
        b2_ts = b2d['timestamp'].values.astype(np.int64)
        p2 = b2d[f'p_fakeout_{K}m'].values
        idx = np.searchsorted(b2_ts, bar_ts, side='right') - 1
        for i, k in enumerate(idx):
            if 0 <= k < len(b2_ts):
                age_s = bar_ts[i] - b2_ts[k]
                if 0 <= age_s <= 30 * 60:   # ignore pivots older than 30 min
                    fake_attached[g.index[i]] = p2[k]
    merged['p_fakeout_recent_10m'] = fake_attached

    # === Validation: does status predict actual time-to-pivot? ===
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out(f'B1 TRAJECTORY BRIDGE — K={K}min, traj_window={TRAJ_N} bars')
    out('=' * 78)
    out(f'Bars: {len(merged):,}   Days: {merged["day"].nunique()}')
    out('')

    out('--- STATUS DISTRIBUTION + actual time-to-next-pivot per status ---')
    out(f'  {"status":<10}  {"n":>7}  {"%":>5}  '
        f'{"ttn median":>12}  {"ttn p25":>10}  {"ttn p75":>10}  '
        f'{"% w/ piv<10m":>12}')
    for status in ['CLEAR', 'WATCH', 'BUILDING', 'IMMINENT', 'PASSING']:
        sub = merged[merged['status'] == status]
        n = len(sub)
        ttn_min = sub['ttn_minutes'].dropna().values
        if len(ttn_min) == 0:
            out(f'  {status:<10}  {n:>7}  -')
            continue
        med = np.median(ttn_min)
        p25 = np.percentile(ttn_min, 25)
        p75 = np.percentile(ttn_min, 75)
        pct_pivot_10m = float((ttn_min <= 10).mean() * 100)
        out(f'  {status:<10}  {n:>7,}  {n/len(merged)*100:>5.1f}  '
            f'{med:>11.1f}m  {p25:>9.1f}m  {p75:>9.1f}m  '
            f'{pct_pivot_10m:>11.1f}%')

    out('')
    out('Interpretation:')
    out('  - IMMINENT bars should have low median TTN (1-3 min)')
    out('  - BUILDING bars should have moderate TTN (3-7 min)')
    out('  - CLEAR bars should have high TTN (no pivot soon)')
    out('  - PASSING bars: recent peak that didn\'t materialize — high TTN')

    # === Bridge: When status is IMMINENT, how often does pivot arrive within 3 min? ===
    out('')
    out('--- BRIDGE VALIDATION: status -> pivot-arrival precision ---')
    for status, target_min, label in [
        ('IMMINENT', 3,  'pivot within 3 min'),
        ('IMMINENT', 5,  'pivot within 5 min'),
        ('BUILDING', 7,  'pivot within 7 min'),
        ('WATCH',    10, 'pivot within 10 min'),
    ]:
        sub = merged[merged['status'] == status]
        ttn_min = sub['ttn_minutes'].dropna().values
        if len(ttn_min) == 0:
            continue
        hit = (ttn_min <= target_min).mean()
        out(f'  status={status:<10} -> {label:<26}  precision={hit*100:5.2f}%  '
            f'on n={len(ttn_min):,}')

    # === Composite: zigzag direction + status combo ===
    out('')
    out('--- COMPOSITE: zigzag direction + B1 status ---')
    out('  Suggested rules:')
    out('    RIDE_CONFIDENT:  zz_dir != NEUTRAL  AND status in (CLEAR, WATCH)')
    out('    RIDE_CAREFUL  :  zz_dir != NEUTRAL  AND status == BUILDING')
    out('    PREPARE_FLIP  :  zz_dir != NEUTRAL  AND status == IMMINENT')
    out('    HOLD          :  zz_dir == NEUTRAL  OR status == PASSING')

    # Compute composite action per bar
    zz = merged['live_zz_dir'].values
    st = merged['status'].values
    composite = np.where(
        (zz != 'NEUTRAL') & np.isin(st, ['CLEAR', 'WATCH']),
        'RIDE_CONFIDENT',
        np.where(
            (zz != 'NEUTRAL') & (st == 'BUILDING'),
            'RIDE_CAREFUL',
            np.where(
                (zz != 'NEUTRAL') & (st == 'IMMINENT'),
                'PREPARE_FLIP',
                'HOLD',
            )
        )
    )
    merged['composite_action'] = composite

    out('')
    out('--- COMPOSITE ACTION DISTRIBUTION + accuracy vs leg_direction ---')
    if 'leg_direction' not in tr.columns:
        from tools.direction_signal_accuracy import derive_leg_direction
        tr_with_ld = derive_leg_direction(tr.copy())
    else:
        tr_with_ld = tr
    leg = tr_with_ld.set_index(['day', 'timestamp'])['leg_direction']
    merged = merged.set_index(['day', 'timestamp'])
    merged['leg_direction'] = leg
    merged = merged.reset_index()

    in_leg = merged['leg_direction'].isin(['LONG', 'SHORT'])
    truth = merged['leg_direction'].values

    for action in ['RIDE_CONFIDENT', 'RIDE_CAREFUL', 'PREPARE_FLIP', 'HOLD']:
        mask = (merged['composite_action'] == action) & in_leg
        n = int(mask.sum())
        if action == 'PREPARE_FLIP':
            # Predict OPPOSITE of zz_dir (we expect flip)
            pred = np.where(merged['live_zz_dir'].values == 'LONG', 'SHORT', 'LONG')
        else:
            # Predict same direction as zz_dir (riding)
            pred = merged['live_zz_dir'].values
        correct = ((pred == truth) & mask).sum()
        acc = correct / max(n, 1)
        out(f'  {action:<14}  n={n:>6,}  acc={acc*100:5.2f}%  '
            f'(predict {"opposite zz" if action == "PREPARE_FLIP" else "same as zz"})')

    # Save
    out_parq = Path(args.out_parquet)
    out_parq.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_parq, index=False)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_parquet}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
