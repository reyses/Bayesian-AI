"""Pivot Probability Cloud — Bohr-model composite over forward time.

User insight (2026-05-17): "it's kinda like the Bohr model — we don't know
where the electron is but we can make a very good guess where it is."

Map: don't try to predict EXACT time-to-pivot. Build a probability density
over forward time bins, then act on the shape of the cloud.

Density bins (from B1 nested K-series — no new training needed):
  P(pivot in [0,  1m])    = B1_K=1
  P(pivot in (1,  3m])    = B1_K=3  - B1_K=1
  P(pivot in (3,  5m])    = B1_K=5  - B1_K=3
  P(pivot in (5,  10m])   = B1_K=10 - B1_K=5
  P(pivot in (10m+])       = 1 - B1_K=10

Plus trajectory dimension (rising vs decaying cloud):
  rate = slope of B1_K=10 over last 10 bars
  status = IMMINENT / BUILDING / WATCH / CLEAR / PASSING (from trajectory bridge)

Action zones (Bohr-style discrete shells based on where mass concentrates):
  CLEAR        : peak_mass_bin >= 5min   (no risk for 5+ min)
  WATCH        : peak_mass_bin in (3, 5min]
  APPROACHING  : peak_mass_bin in (1, 3min]
  IMMINENT     : peak_mass_bin in (0, 1min]
  PASSING      : cloud was high but decaying (false alarm fading)

Validation: for each zone, what's the ACTUAL time-to-next-pivot distribution?
If APPROACHING zone bars really have median 2-3 min TTP, calibration is good.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def pivot_centroid_ts_per_day(truth_df: pd.DataFrame) -> dict:
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


def build_density_bins(b1_df: pd.DataFrame) -> pd.DataFrame:
    """Convert nested cumulative B1 K-probabilities to per-window mass.

    Clips to [0, 1] in case overlapping classifiers produce inversions
    (e.g., P_K=3 < P_K=1 due to GBM noise — rare but possible).
    """
    df = b1_df.copy()
    p1  = df['p_pivot_1m'].values
    p3  = df['p_pivot_3m'].values
    p5  = df['p_pivot_5m'].values
    p10 = df['p_pivot_10m'].values
    # Enforce monotonicity (cumulative should rise with K)
    p3_m  = np.maximum(p3,  p1)
    p5_m  = np.maximum(p5,  p3_m)
    p10_m = np.maximum(p10, p5_m)
    df['mass_0_1m']   = np.clip(p1, 0, 1)
    df['mass_1_3m']   = np.clip(p3_m  - p1,  0, 1)
    df['mass_3_5m']   = np.clip(p5_m  - p3_m, 0, 1)
    df['mass_5_10m']  = np.clip(p10_m - p5_m, 0, 1)
    df['mass_10m_plus'] = np.clip(1 - p10_m, 0, 1)
    # Expected time (weighted bin midpoints, capped at 15m for the tail)
    bin_centers = np.array([0.5, 2.0, 4.0, 7.5, 15.0])   # minutes
    masses = df[['mass_0_1m', 'mass_1_3m', 'mass_3_5m',
                 'mass_5_10m', 'mass_10m_plus']].values
    df['expected_ttp_min'] = (masses * bin_centers).sum(axis=1)
    # Peak bin (mode of cloud)
    df['peak_bin_idx'] = masses.argmax(axis=1)
    df['peak_bin_min'] = bin_centers[df['peak_bin_idx'].values]
    # Sharpness: max mass / sum (1.0 = pure single bin, 0.2 = uniform)
    df['sharpness'] = masses.max(axis=1) / np.clip(masses.sum(axis=1), 1e-9, None)
    return df


def classify_zone(df: pd.DataFrame) -> pd.Series:
    """Action zone from B1 (forward) + B4 (symmetric region) at VALIDATED
    operating points.

    B1 outputs are uncalibrated (class_weight='balanced'), so use threshold
    cascades not argmax. Operating points (from earlier reports):

      B1 K=1   thr=0.70: prec 13.6% / cov  4.5% / 2.61x lift
      B1 K=3   thr=0.70: prec 34.3% / cov  8.3% / 2.19x lift
      B1 K=5   thr=0.70: prec 47.1% / cov 19.9% / 1.90x lift
      B1 K=10  thr=0.85: prec 78.1% / cov  3.9% / 1.89x lift
      B4 W=60  thr=0.85: prec 41.7% / cov  0.14% / 3.90x lift (sparse but tight)
      B4 W=120 thr=0.85: prec 64.5% / cov  0.27% / 3.10x lift
      B4 W=300 thr=0.85: prec 79.0% / cov  9.95% / 1.88x lift (wide-zone detector)
      B4 W=300 thr=0.70: prec 64.5% / cov 37.5% / 1.53x lift

    Zone cascade (tightest first):
      AT_PIVOT    : B4 W=60 >= 0.85   (we're inside a 1-min window of a pivot)
      NEAR_PIVOT  : B4 W=120 >= 0.70  (we're within 2-min of a pivot)
      IMMINENT    : B1 K=1 >= 0.70    (pivot expected in <1m)
      NEAR_3m     : B1 K=3 >= 0.70 AND not above
      NEAR_5m     : B1 K=5 >= 0.70 AND not above
      WIDE_ZONE   : B4 W=300 >= 0.85  (you're in a 5-min pivot zone, high conf)
      WATCH       : B1 K=10 >= 0.70 OR B4 W=300 >= 0.70
      CLEAR       : everything else (especially B1 K=10 < 0.30)
    """
    P1  = df['p_pivot_1m'].values
    P3  = df['p_pivot_3m'].values
    P5  = df['p_pivot_5m'].values
    P10 = df['p_pivot_10m'].values
    P_reg60  = df.get('p_region_60s',  pd.Series(np.zeros(len(df)))).values
    P_reg120 = df.get('p_region_120s', pd.Series(np.zeros(len(df)))).values
    P_reg300 = df.get('p_region_300s', pd.Series(np.zeros(len(df)))).values

    zone = np.full(len(df), 'CLEAR', dtype=object)
    # Cascade from broadest to tightest — each tighter rule overwrites
    zone[(P10 >= 0.70) | (P_reg300 >= 0.70)] = 'WATCH'
    zone[P_reg300 >= 0.85] = 'WIDE_ZONE'
    zone[P5  >= 0.70] = 'NEAR_5m'
    zone[P3  >= 0.70] = 'NEAR_3m'
    zone[P1  >= 0.70] = 'IMMINENT'
    zone[P_reg120 >= 0.70] = 'NEAR_PIVOT'
    zone[P_reg60  >= 0.85] = 'AT_PIVOT'
    return pd.Series(zone, index=df.index)


def add_trajectory_decay(df: pd.DataFrame, N: int = 10) -> pd.DataFrame:
    """Add 'cloud_state' column: RISING / DECAYING / FLAT based on slope of
    expected_ttp_min over the trailing N bars per day.
    Expected TTP decreasing = cloud sharpening (pivot approaching).
    Expected TTP increasing = cloud dissolving (false alarm passing).
    """
    df = df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    states = np.full(len(df), 'FLAT', dtype=object)
    for day, g in df.groupby('day'):
        idx = g.index.values
        ettp = g['expected_ttp_min'].values
        for i in range(len(g)):
            lo = max(0, i - N + 1)
            window = ettp[lo:i+1]
            if len(window) < 3:
                continue
            xs = np.arange(len(window), dtype=np.float64)
            xm = xs.mean(); ym = window.mean()
            num = np.sum((xs - xm) * (window - ym))
            den = np.sum((xs - xm) ** 2)
            slope = num / den if den > 0 else 0.0
            if slope < -0.05:
                states[idx[i]] = 'RISING'   # ETTP shrinking = pivot approaching
            elif slope > 0.05:
                states[idx[i]] = 'DECAYING' # ETTP growing = cloud fading
    df['cloud_state'] = states
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1-cache',
                    default='reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--b4-cache',
                    default='reports/findings/regret_oracle/b4_proba_OOS_NT8.parquet',
                    help='Optional B4 cache — if exists, enrich cloud with region prob')
    ap.add_argument('--b5-cache',
                    default='reports/findings/regret_oracle/b5_leg_phase_OOS_NT8.parquet',
                    help='Optional B5 cache — leg-phase probabilities')
    ap.add_argument('--out-parquet',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.txt')
    args = ap.parse_args()

    print('Loading B1 cache:', args.b1_cache)
    b1 = pd.read_parquet(args.b1_cache).sort_values(['day', 'timestamp']).reset_index(drop=True)
    print(f'  {len(b1):,} rows')
    print('Loading truth:', args.truth)
    tr = pd.read_parquet(args.truth).sort_values(['day', 'timestamp']).reset_index(drop=True)
    print(f'  {len(tr):,} rows')

    print('Building density bins from B1 K-series...')
    cloud = build_density_bins(b1)

    # Merge B4 BEFORE classify_zone so B4-region thresholds participate
    b4_path = Path(args.b4_cache)
    if b4_path.exists():
        print(f'Merging B4 region probabilities: {b4_path}')
        b4 = pd.read_parquet(b4_path)
        b4_cols = [c for c in b4.columns if c.startswith('p_region_')]
        cloud = cloud.merge(b4[['timestamp', 'day'] + b4_cols],
                             on=['timestamp', 'day'], how='left')

    # Merge B5 leg-phase probabilities
    b5_path = Path(args.b5_cache)
    if b5_path.exists():
        print(f'Merging B5 leg-phase probabilities: {b5_path}')
        b5 = pd.read_parquet(b5_path)
        b5_cols = [c for c in b5.columns
                   if c.startswith('p_phase_') or c in ('p_phase_argmax', 'leg_phase_truth')]
        cloud = cloud.merge(b5[['timestamp', 'day'] + b5_cols],
                             on=['timestamp', 'day'], how='left')

    print('Classifying action zones (B1 + B4 thresholds)...')
    cloud['zone'] = classify_zone(cloud)

    print('Adding trajectory decay state...')
    cloud = add_trajectory_decay(cloud, N=10)

    # Compute actual time-to-next-pivot
    print('Computing actual time-to-next-pivot...')
    pivs = pivot_centroid_ts_per_day(tr)
    ttp = np.full(len(cloud), np.nan)
    for day, g in cloud.groupby('day'):
        p = pivs.get(day, np.array([], dtype=np.int64))
        if len(p) == 0:
            continue
        bar_ts = g['timestamp'].values.astype(np.int64)
        idx = np.searchsorted(p, bar_ts, side='left')
        valid = idx < len(p)
        ttp_vals = np.full(len(bar_ts), np.nan)
        ttp_vals[valid] = (p[idx[valid]] - bar_ts[valid]).astype(np.float64) / 60.0
        ttp[g.index] = ttp_vals
    cloud['actual_ttp_min'] = ttp

    # === Validation ===
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('PIVOT PROBABILITY CLOUD  (Bohr-model composite, NT8 OOS, 32 days)')
    out('  Density bins: [0,1m]  (1,3m]  (3,5m]  (5,10m]  (10m+]')
    out('  computed from B1 K-series — no new training')
    out('=' * 78)
    out(f'Bars: {len(cloud):,}   Days: {cloud["day"].nunique()}')
    out('')

    out('--- ZONE DISTRIBUTION + actual TTP per zone (the calibration test) ---')
    out(f'  {"zone":<12}  {"n":>8}  {"% bars":>7}  '
        f'{"actual_med":>11}  {"actual_p25":>11}  {"actual_p75":>11}  '
        f'{"%_pivot<5m":>11}')
    for zone in ['AT_PIVOT', 'NEAR_PIVOT', 'IMMINENT', 'NEAR_3m', 'NEAR_5m',
                 'WIDE_ZONE', 'WATCH', 'CLEAR']:
        sub = cloud[cloud['zone'] == zone]
        n = len(sub)
        if n == 0:
            out(f'  {zone:<12}  {n:>8,}  -'); continue
        ttp_vals = sub['actual_ttp_min'].dropna().values
        if len(ttp_vals) == 0:
            out(f'  {zone:<12}  {n:>8,}  no_ttp'); continue
        med = float(np.median(ttp_vals))
        p25 = float(np.percentile(ttp_vals, 25))
        p75 = float(np.percentile(ttp_vals, 75))
        pct_5m = float((ttp_vals < 5).mean() * 100)
        out(f'  {zone:<12}  {n:>8,}  {n/len(cloud)*100:>6.1f}%  '
            f'{med:>10.2f}m  {p25:>10.2f}m  {p75:>10.2f}m  '
            f'{pct_5m:>10.1f}%')

    out('')
    out('Interpretation (Bohr-style probability cloud):')
    out('  AT_PIVOT    -> B4 W=60 strong: we are AT a pivot (forward+back within 1m)')
    out('  NEAR_PIVOT  -> B4 W=120 strong: we are within 2m of a pivot')
    out('  IMMINENT    -> B1 K=1 strong: pivot expected in <1m forward')
    out('  NEAR_3m     -> B1 K=3 strong: pivot in 1-3m forward')
    out('  NEAR_5m     -> B1 K=5 strong: pivot in 3-5m forward')
    out('  WIDE_ZONE   -> B4 W=300 strong: we are inside a 5-min pivot zone')
    out('  WATCH       -> coarse forward 10m / wide zone hint')
    out('  CLEAR       -> all signals weak, no pivot nearby')

    out('  - actual_med should MATCH the zone bin range.')
    out('  - If actual_med >> zone bin, classifier overcalls; tighten threshold.')

    # Trajectory state breakdown
    out('')
    out('--- TRAJECTORY STATE (cloud sharpening vs dissolving) ---')
    for state in ['RISING', 'FLAT', 'DECAYING']:
        sub = cloud[cloud['cloud_state'] == state]
        n = len(sub)
        if n == 0:
            continue
        ttp_vals = sub['actual_ttp_min'].dropna().values
        if len(ttp_vals) == 0:
            continue
        med = float(np.median(ttp_vals))
        out(f'  state={state:<10}  n={n:>7,}  '
            f'median actual TTP = {med:>6.2f}m   '
            f'pct < 5m = {(ttp_vals < 5).mean()*100:.1f}%')

    # Zone x State combo
    out('')
    out('--- ZONE x CLOUD_STATE composite (the actionable joint) ---')
    out(f'  {"zone":<12}  {"state":<10}  {"n":>7}  {"ttp_med":>9}  {"%<5m":>6}')
    for zone in ['AT_PIVOT', 'NEAR_PIVOT', 'IMMINENT', 'NEAR_3m', 'NEAR_5m',
                 'WIDE_ZONE', 'WATCH', 'CLEAR']:
        for state in ['RISING', 'FLAT', 'DECAYING']:
            sub = cloud[(cloud['zone'] == zone) & (cloud['cloud_state'] == state)]
            n = len(sub)
            if n < 30:
                continue
            ttp_vals = sub['actual_ttp_min'].dropna().values
            if len(ttp_vals) == 0:
                continue
            med = float(np.median(ttp_vals))
            pct = float((ttp_vals < 5).mean() * 100)
            out(f'  {zone:<12}  {state:<10}  {n:>7,}  {med:>8.2f}m  {pct:>5.1f}%')

    out('')
    out('Sharpest signal:  zone=IMMINENT + state=RISING')
    out('Slowest signal:   zone=CLEAR + state=DECAYING')
    out('"Pivot just happened" diagnostic: zone=IMMINENT + state=DECAYING')

    # ============================================================
    # B5 leg-phase axis: joint state (zone, leg_phase)
    # ============================================================
    if 'p_phase_argmax' in cloud.columns:
        out('')
        out('=' * 78)
        out('JOINT STATE: pivot zone (B1+B4) x leg phase (B5)')
        out('=' * 78)
        out('Each cell shows: n bars (% pivot<5m) — model AGREEMENT diagnostic.')
        out('')
        # Build the joint table
        zones = ['AT_PIVOT', 'NEAR_PIVOT', 'IMMINENT', 'NEAR_3m', 'NEAR_5m',
                 'WIDE_ZONE', 'WATCH', 'CLEAR']
        phases = ['EARLY', 'MID', 'LATE']
        header = '  ' + 'zone'.ljust(14) + ''.join(f'{p:>20}' for p in phases) + f'{"TOTAL":>14}'
        out(header)
        for zone in zones:
            zsub = cloud[cloud['zone'] == zone]
            n_zone = len(zsub)
            if n_zone == 0:
                continue
            row_str = '  ' + zone.ljust(14)
            for phase in phases:
                cell = zsub[zsub['p_phase_argmax'] == phase]
                n = len(cell)
                if n == 0:
                    row_str += f'{"-":>20}'; continue
                ttp = cell['actual_ttp_min'].dropna().values
                pct = (ttp < 5).mean() * 100 if len(ttp) > 0 else 0.0
                row_str += f'{f"{n:>5} ({pct:4.1f}%<5m)":>20}'
            row_str += f'{n_zone:>14,}'
            out(row_str)
        out('')
        out('Interpretation: high B5(MID) in CLEAR zone = strongest "ride" signal.')
        out('Disagreement (e.g., NEAR_PIVOT zone but B5=EARLY) = potential fakeout.')

        # High-confidence MID where pivot zone is CLEAR — the "ride hard" signal
        deep_trend = cloud[(cloud['zone'] == 'CLEAR') &
                            (cloud.get('p_phase_MID', 0) > 0.50)]
        out('')
        out(f'--- DEEP-TREND composite (CLEAR zone + B5 P(MID) > 0.50) ---')
        out(f'  n = {len(deep_trend):,}  ({len(deep_trend)/len(cloud)*100:.2f}% of bars)')
        if len(deep_trend) > 0:
            ttp = deep_trend['actual_ttp_min'].dropna().values
            if len(ttp) > 0:
                out(f'  actual TTP median = {np.median(ttp):.2f}m   '
                    f'p25 = {np.percentile(ttp, 25):.2f}m   '
                    f'p75 = {np.percentile(ttp, 75):.2f}m')
                out(f'  % with pivot < 5m: {(ttp < 5).mean()*100:.1f}%  '
                    f'(vs CLEAR baseline 16.2%)')

        # Sharpest fade alert: NEAR_PIVOT zone + B5 LATE high
        sharp_late = cloud[(cloud['zone'].isin(['NEAR_PIVOT', 'IMMINENT', 'NEAR_3m'])) &
                            (cloud.get('p_phase_LATE', 0) > 0.50)]
        out('')
        out(f'--- SHARP FADE-PREP composite (NEAR_PIVOT/IMMINENT/NEAR_3m + B5 P(LATE) > 0.50) ---')
        out(f'  n = {len(sharp_late):,}  ({len(sharp_late)/len(cloud)*100:.2f}% of bars)')
        if len(sharp_late) > 0:
            ttp = sharp_late['actual_ttp_min'].dropna().values
            if len(ttp) > 0:
                out(f'  actual TTP median = {np.median(ttp):.2f}m   '
                    f'p25 = {np.percentile(ttp, 25):.2f}m')
                out(f'  % with pivot < 5m: {(ttp < 5).mean()*100:.1f}%')

        # DISAGREEMENT: B4 says NEAR PIVOT but B5 says EARLY — potential fakeout
        disagree = cloud[(cloud['zone'].isin(['NEAR_PIVOT', 'IMMINENT', 'NEAR_3m', 'WIDE_ZONE'])) &
                          (cloud.get('p_phase_EARLY', 0) > 0.50)]
        out('')
        out(f'--- DISAGREEMENT (pivot-near zone + B5 P(EARLY) > 0.50 — fakeout hint) ---')
        out(f'  n = {len(disagree):,}  ({len(disagree)/len(cloud)*100:.2f}% of bars)')
        if len(disagree) > 0:
            ttp = disagree['actual_ttp_min'].dropna().values
            if len(ttp) > 0:
                out(f'  actual TTP median = {np.median(ttp):.2f}m   '
                    f'p25 = {np.percentile(ttp, 25):.2f}m')
                out(f'  % with pivot < 5m: {(ttp < 5).mean()*100:.1f}%')

    # Save
    out_parq = Path(args.out_parquet)
    out_parq.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_parquet(out_parq, index=False)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {out_parq}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
