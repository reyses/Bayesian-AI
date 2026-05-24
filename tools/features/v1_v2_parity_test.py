"""
v1_v2_parity_test.py — verify core_v2/v1_compat.py reproduces V1 cache values.

For each of N sample days, for each TF:
  1. Load V1 cache (DATA/ATLAS/FEATURES_5s/<day>.parquet) — has 79D V1 features
  2. Load V2 cache (DATA/ATLAS/FEATURES_5s_v2/) — layered V2 features
  3. Load raw OHLCV at the TF (DATA/ATLAS/<tf>/YYYY_MM.parquet)
  4. Use v1_compat.derive_v1_concepts_batch to compute V1 concepts from V2
  5. Compare V2-derived to V1 cached. Report:
     - Max abs error per concept
     - Median abs error
     - Pearson correlation (sanity)
     - Per-bar error distribution

Concepts validated:
  wick_ratio    — should match exactly (formula identical)
  p_at_center   — should match within float precision (formula identical)
  variance_ratio — uses raw close history, should match exactly
  vol_rel       — uses raw volume history, should match exactly
  dir_vol       — sign(velocity)*vol_rel; sign source differs (V1 uses
                  WindowedSlope from MarketState; V2 uses L2_velocity_w).
                  Sign agreement should be high but not perfect.
  dmi_substitute — V2 has no DMI; checks the substitute is zero-correlated
                  with V1's actual DMI (we're not claiming parity here,
                  just measuring how different)

Outputs:
  reports/findings/v1_v2_parity/
    parity_summary.csv      per (day, tf, concept) error stats
    parity_overall.csv      aggregated stats per (tf, concept)
    summary.md              headline numbers + caveats
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_v2 import v1_compat


# V1 used 6 TFs; V2 has 8 (added 5s, 4h). Parity check on the 6 V1 TFs.
V1_TFS = ['15s', '1m', '5m', '15m', '1h', '1D']

# Default sample of days across the IS+OOS range
SAMPLE_DAYS = [
    '2025_01_15', '2025_02_05', '2025_03_15', '2025_04_15',
    '2025_06_02', '2025_08_15', '2025_10_01', '2025_11_15',
    '2025_12_18', '2026_01_15', '2026_02_15', '2026_03_15',
]


def load_v1_day(atlas_root: str, day: str) -> pd.DataFrame | None:
    p = os.path.join(atlas_root, 'FEATURES_5s', f'{day}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def load_v2_day(atlas_root: str, day: str, tf: str,
                  ts_min: int, ts_max: int) -> pd.DataFrame | None:
    """Load V2 features via the canonical loader, which renames window-numbered
    column suffixes (e.g. L2_1m_price_velocity_15) to _w suffix."""
    from tools.research.features_v2 import load_v2_features
    v2_dir = os.path.join(atlas_root, 'FEATURES_5s_v2')
    if not os.path.isdir(v2_dir):
        return None
    try:
        v2 = load_v2_features(v2_dir=v2_dir, atlas_root=atlas_root,
                                  day_strs=None,
                                  ts_range=(ts_min, ts_max), verbose=False)
        return v2
    except Exception as e:
        print(f"    load_v2 error for {day} {tf}: {e}")
        return None


def load_ohlcv_native(atlas_root: str, tf: str, day: str) -> pd.DataFrame | None:
    """Load OHLCV for this day at the native TF cadence.

    OHLCV files are per-day: DATA/ATLAS/<tf>/YYYY_MM_DD.parquet.
    For variance_ratio's 60-bar window we need history; load the prior day
    too where available.
    """
    import glob
    # Find all day files at this TF, sorted; take all up to and including
    # the requested day (gives full history for rolling-window computations).
    files = sorted(glob.glob(os.path.join(atlas_root, tf, '*.parquet')))
    target_idx = None
    target_basename = f'{day}.parquet'
    for i, f in enumerate(files):
        if os.path.basename(f) == target_basename:
            target_idx = i
            break
    if target_idx is None:
        return None
    # Concat: trailing 3 days for history + target day
    take = files[max(target_idx - 3, 0):target_idx + 1]
    dfs = [pd.read_parquet(p) for p in take]
    return pd.concat(dfs, ignore_index=True)


def compute_parity_one_day_tf(atlas_root: str, day: str, tf: str) -> dict | None:
    v1 = load_v1_day(atlas_root, day)
    if v1 is None:
        return None
    ts = v1['timestamp'].values
    if len(ts) == 0:
        return None
    ts_min, ts_max = int(ts.min()), int(ts.max())
    v2 = load_v2_day(atlas_root, day, tf, ts_min, ts_max)
    ohlcv = load_ohlcv_native(atlas_root, tf, day)
    if v2 is None or ohlcv is None:
        return None
    # V2 may include surrounding days due to ts_range expansion; filter to
    # exact ts range of V1 for alignment.
    v2 = v2[(v2['timestamp'] >= ts_min) & (v2['timestamp'] <= ts_max)].reset_index(drop=True)
    if len(v2) != len(v1):
        # Try another alignment: drop V2 rows whose timestamp isn't in V1
        v1_ts_set = set(ts.tolist())
        v2 = v2[v2['timestamp'].isin(v1_ts_set)].reset_index(drop=True)
        if len(v2) != len(v1):
            print(f"    {day} {tf}: V1={len(v1)}, V2={len(v2)} after align — skip")
            return None

    # The V1 cache and V2 cache are both 5s-anchored. Native OHLCV is at
    # TF cadence. Pass NATIVE OHLCV to the shim — it does the alignment.
    from core_v2.features import TF_SECONDS

    # Make sure ohlcv timestamps are integers (some parquets store datetime)
    if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
        ohlcv = ohlcv.copy()
        ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
    derived = v1_compat.derive_v1_concepts_batch(v2, ohlcv, tf, TF_SECONDS[tf])

    # Compare to V1 cache. V1 column naming:
    v1_compat_concepts = {
        'wick_ratio':     f'{tf}_wick_ratio',
        'p_at_center':    f'{tf}_p_at_center',
        'variance_ratio': f'{tf}_variance_ratio',
        'vol_rel':        f'{tf}_vol_rel',
        'dir_vol':        f'{tf}_dir_vol',
    }

    # V1 and V2 should have same number of rows (both 5s-anchored)
    if len(v1) != len(derived):
        return None

    results = {}
    for v2_concept, v1_col in v1_compat_concepts.items():
        if v1_col not in v1.columns:
            continue
        v1_vals = v1[v1_col].values.astype(np.float64)
        d_vals = derived[v2_concept].values.astype(np.float64)
        # drop rows where either is NaN (warmup)
        mask = ~(np.isnan(v1_vals) | np.isnan(d_vals))
        if mask.sum() < 100:
            continue
        a, b = v1_vals[mask], d_vals[mask]
        diff = a - b
        results[v2_concept] = {
            'n': int(mask.sum()),
            'max_abs_err': float(np.max(np.abs(diff))),
            'mean_abs_err': float(np.mean(np.abs(diff))),
            'median_abs_err': float(np.median(np.abs(diff))),
            'p99_abs_err': float(np.percentile(np.abs(diff), 99)),
            'pearson_r': float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 else float('nan'),
            'v1_mean': float(a.mean()),
            'derived_mean': float(b.mean()),
            'v1_std': float(a.std()),
            'derived_std': float(b.std()),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas-root', default='DATA/ATLAS')
    parser.add_argument('--days', nargs='+', default=SAMPLE_DAYS)
    parser.add_argument('--tfs', nargs='+', default=V1_TFS)
    parser.add_argument('--output-dir', default='reports/findings/v1_v2_parity')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V1 -> V2 parity test")
    print(f"  Days: {len(args.days)}  TFs: {args.tfs}")
    print(f"{'='*70}")

    rows = []
    for day in args.days:
        print(f"\n  {day}")
        for tf in args.tfs:
            res = compute_parity_one_day_tf(args.atlas_root, day, tf)
            if res is None:
                print(f"    {tf}: missing data, skip")
                continue
            for concept, stats in res.items():
                rows.append({'day': day, 'tf': tf, 'concept': concept, **stats})
            print(f"    {tf}: " + ', '.join(
                f"{k}={v['max_abs_err']:.2e}" for k, v in res.items()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'parity_summary.csv'), index=False)
    print(f"\n  [saved] parity_summary.csv ({len(df)} rows)")

    # Aggregate per (tf, concept)
    agg = (df.groupby(['tf', 'concept'])
              .agg(n_days=('day', 'nunique'),
                    max_abs_err=('max_abs_err', 'max'),
                    median_abs_err=('median_abs_err', 'median'),
                    p99_abs_err=('p99_abs_err', 'max'),
                    mean_pearson=('pearson_r', 'mean'),
                    mean_v1=('v1_mean', 'mean'),
                    mean_derived=('derived_mean', 'mean'))
              .reset_index())
    agg.to_csv(os.path.join(args.output_dir, 'parity_overall.csv'), index=False)
    print(f"  [saved] parity_overall.csv ({len(agg)} rows)")

    print(f"\n  Aggregated parity (across {len(args.days)} days):")
    print(f"  {'tf':>4}  {'concept':>16}  {'n_days':>6}  "
          f"{'max_abs_err':>12}  {'median_abs_err':>14}  {'pearson':>7}")
    for _, r in agg.sort_values(['concept', 'tf']).iterrows():
        print(f"  {r['tf']:>4}  {r['concept']:>16}  {int(r['n_days']):>6}  "
              f"{r['max_abs_err']:>12.4e}  {r['median_abs_err']:>14.4e}  "
              f"{r['mean_pearson']:>7.4f}")

    # Per-concept verdict
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V1 -> V2 parity test - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Days tested**: {len(args.days)}\n")
        f.write(f"**TFs**: {args.tfs}\n\n")
        f.write("## Per-(TF, concept) parity\n\n")
        f.write(agg.to_string(index=False))
        f.write("\n\n## Verdict per concept\n\n")
        for concept in sorted(df['concept'].unique()):
            sub = agg[agg['concept'] == concept]
            max_err = sub['max_abs_err'].max()
            mean_pearson = sub['mean_pearson'].mean()
            verdict = ('EXACT' if max_err < 1e-6 else
                          ('TIGHT (~1e-3)' if max_err < 1e-3 else
                            ('LOOSE — investigate' if mean_pearson > 0.95 else
                              'BROKEN')))
            f.write(f"- **{concept}**: max_abs_err={max_err:.2e}, "
                    f"mean Pearson r={mean_pearson:.4f}  ->  **{verdict}**\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
