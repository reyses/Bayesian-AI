"""
v1_v2_threshold_calibration.py — calibrate V1 tier-engine thresholds
to V2 features by matching bar-firing fractions.

The tier engine (`training/nightmare_blended.py`) has hardcoded thresholds
on z_se, velocity, etc. that fire on a specific fraction of bars when
applied to V1 features. The V2 SFE produces slightly different
distributions for the same concepts, so the same literal threshold fires
on a different (typically smaller) fraction of V2 bars.

This tool computes, for each (concept, TF, V1 threshold) tuple, the V2
threshold that matches V1's bar-firing fraction. Output is a calibration
table that can be applied to nightmare_blended.py constants.

Sweep:
  - z_se thresholds: 1.0, 1.4, 1.5, 2.0 (covers ROCHE, H1_Z_MIN,
    H1_AGAINST_Z_MIN, EXHAUST_Z_MIN, MTF_Z_MIN)
  - |z_se| thresholds (same set, abs)
  - velocity thresholds: 30.0, 50.0, 100.0, 10.0
  - 4 TFs: 1m, 5m, 15m, 1h
  - Sample: 5 days across 2025-01 to 2026-03

Outputs:
  reports/findings/v1_v2_threshold_calibration/
    z_se_calibration.csv      per (TF, V1_threshold): V1_rate, V2_rate_literal,
                                V2_threshold_matched
    velocity_calibration.csv  same for velocity
    calibration_table.md      copy-paste-ready threshold-by-threshold edits
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SAMPLE_DAYS = ['2025_03_15', '2025_06_02', '2025_09_15', '2025_12_15', '2026_02_15']
TFS = ['1m', '5m', '15m', '1h']


def load_pair_for_day(day: str, atlas_root: str = 'DATA/ATLAS'):
    from tools.research.features_v2 import load_v2_features

    p1 = os.path.join(atlas_root, 'FEATURES_5s', f'{day}.parquet')
    if not os.path.exists(p1):
        return None, None
    v1 = pd.read_parquet(p1)
    ts = v1['timestamp'].values
    v2 = load_v2_features(
        v2_dir=os.path.join(atlas_root, 'FEATURES_5s_v2'),
        atlas_root=atlas_root, day_strs=None,
        ts_range=(int(ts.min()), int(ts.max())), verbose=False)
    v1_ts_set = set(ts.tolist())
    v2 = v2[v2['timestamp'].isin(v1_ts_set)].reset_index(drop=True)
    if len(v1) != len(v2):
        return None, None
    return v1, v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', nargs='+', default=SAMPLE_DAYS)
    parser.add_argument('--output-dir',
                        default='reports/findings/v1_v2_threshold_calibration')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {len(args.days)} days...")
    all_v1, all_v2 = [], []
    for day in args.days:
        v1, v2 = load_pair_for_day(day)
        if v1 is None:
            print(f"  {day}: missing, skip")
            continue
        all_v1.append(v1)
        all_v2.append(v2)
        print(f"  {day}: {len(v1)} bars")

    if not all_v1:
        print("No data, exiting.")
        return

    V1 = pd.concat(all_v1, ignore_index=True)
    V2 = pd.concat(all_v2, ignore_index=True)
    n = len(V1)
    print(f"\nTotal sample: {n} bars\n")

    # ── z_se calibration ────────────────────────────────────────────
    z_se_thresholds = [1.0, 1.4, 1.5, 2.0]  # covers H1_Z_MIN, EXHAUST_Z_MIN, H1_AGAINST_Z_MIN, ROCHE
    z_se_rows = []
    for tf in TFS:
        v1_col = f'{tf}_z_se'
        v2_col = f'L3_{tf}_z_se_w'
        a = V1[v1_col].values
        b = V2[v2_col].values
        valid = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[valid], b[valid]
        abs_a, abs_b = np.abs(a), np.abs(b)
        for thresh in z_se_thresholds:
            v1_rate = (abs_a > thresh).mean()
            v2_rate_literal = (abs_b > thresh).mean()
            # find V2 threshold matching V1 rate
            if v1_rate <= 0 or v1_rate >= 1:
                v2_thresh_matched = thresh
            else:
                v2_thresh_matched = float(np.quantile(abs_b, 1 - v1_rate))
            z_se_rows.append({
                'concept': '|z_se|',
                'tf': tf,
                'v1_threshold': thresh,
                'v1_fire_rate_pct': float(v1_rate * 100),
                'v2_rate_pct_if_literal': float(v2_rate_literal * 100),
                'v2_rate_change_pct': float(
                    (v2_rate_literal - v1_rate) / max(v1_rate, 1e-9) * 100),
                'v2_threshold_matched': v2_thresh_matched,
                'v2_rate_pct_at_matched': float(
                    (abs_b > v2_thresh_matched).mean() * 100),
            })

    z_df = pd.DataFrame(z_se_rows)
    z_df.to_csv(os.path.join(args.output_dir, 'z_se_calibration.csv'),
                  index=False)
    print("z_se calibration:\n")
    print(z_df.to_string(index=False))

    # ── velocity calibration ───────────────────────────────────────
    vel_thresholds = [10.0, 30.0, 50.0, 100.0]  # MTF_1M_VEL_ALIVE, MTF_5M_VEL_MIN, VELOCITY_THRESHOLD, FREIGHT_TRAIN
    vel_rows = []
    for tf in TFS:
        v1_col = f'{tf}_velocity'
        v2_col = f'L2_{tf}_price_velocity_w'
        a = V1[v1_col].values
        b = V2[v2_col].values
        valid = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[valid], b[valid]
        abs_a, abs_b = np.abs(a), np.abs(b)
        for thresh in vel_thresholds:
            v1_rate = (abs_a > thresh).mean()
            v2_rate_literal = (abs_b > thresh).mean()
            if v1_rate <= 0 or v1_rate >= 1:
                v2_thresh_matched = thresh
            else:
                v2_thresh_matched = float(np.quantile(abs_b, 1 - v1_rate))
            vel_rows.append({
                'concept': '|velocity|',
                'tf': tf,
                'v1_threshold': thresh,
                'v1_fire_rate_pct': float(v1_rate * 100),
                'v2_rate_pct_if_literal': float(v2_rate_literal * 100),
                'v2_rate_change_pct': float(
                    (v2_rate_literal - v1_rate) / max(v1_rate, 1e-9) * 100),
                'v2_threshold_matched': v2_thresh_matched,
                'v2_rate_pct_at_matched': float(
                    (abs_b > v2_thresh_matched).mean() * 100),
            })
    vel_df = pd.DataFrame(vel_rows)
    vel_df.to_csv(os.path.join(args.output_dir, 'velocity_calibration.csv'),
                    index=False)
    print("\n\nvelocity calibration:\n")
    print(vel_df.to_string(index=False))

    # ── Markdown summary ───────────────────────────────────────────
    # Helper to look up a calibrated value
    def lookup(df, tf, thresh):
        sub = df[(df['tf'] == tf) & (df['v1_threshold'] == thresh)]
        if sub.empty:
            return float('nan')
        return float(sub['v2_threshold_matched'].iloc[0])

    md = os.path.join(args.output_dir, 'calibration_table.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write(f"# V1 -> V2 threshold calibration table\n\n")
        f.write(f"Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"Sample: {n} bars across {len(args.days)} days\n\n")
        f.write("## How to read\n\n")
        f.write("- `v1_fire_rate_pct`: fraction of V1 bars where the threshold fires (CURRENT BEHAVIOR)\n")
        f.write("- `v2_rate_pct_if_literal`: fraction of V2 bars where the SAME literal threshold fires (BROKEN if very low)\n")
        f.write("- `v2_rate_change_pct`: how much fire rate changes if threshold is kept literal\n")
        f.write("- `v2_threshold_matched`: V2 threshold that matches V1 fire rate (USE THIS)\n\n")
        f.write("## z_se thresholds\n\n")
        f.write(z_df.round(3).to_string(index=False))
        f.write("\n\n## velocity thresholds\n\n")
        f.write(vel_df.round(3).to_string(index=False))
        f.write("\n\n## Mapping to nightmare_blended.py constants\n\n")
        f.write("| Constant | V1 value | TFs used | V2 calibrated values |\n")
        f.write("|---|---:|---|---|\n")
        roche_1m = lookup(z_df, '1m', 2.0)
        h1zmin = lookup(z_df, '1h', 1.0)
        h1agst = lookup(z_df, '1h', 1.5)
        ezmin_1m = lookup(z_df, '1m', 1.4)
        mtfz_5m = lookup(z_df, '5m', 1.4)
        mtfz_15m = lookup(z_df, '15m', 1.4)
        v_1m_10 = lookup(vel_df, '1m', 10.0)
        v_5m_30 = lookup(vel_df, '5m', 30.0)
        v_1m_50 = lookup(vel_df, '1m', 50.0)
        v_1m_100 = lookup(vel_df, '1m', 100.0)
        f.write(f"| `ROCHE` | 2.0 | 1m | 1m={roche_1m:.2f} |\n")
        f.write(f"| `H1_Z_MIN` | 1.0 | 1h | 1h={h1zmin:.2f} |\n")
        f.write(f"| `H1_AGAINST_Z_MIN` | 1.5 | 1h | 1h={h1agst:.2f} |\n")
        f.write(f"| `EXHAUST_Z_MIN` | 1.4 | 1m | 1m={ezmin_1m:.2f} |\n")
        f.write(f"| `MTF_Z_MIN` | 1.4 | multiple | "
                f"1m={ezmin_1m:.2f}, 5m={mtfz_5m:.2f}, 15m={mtfz_15m:.2f} |\n")
        f.write(f"| `MTF_1M_VEL_ALIVE` | 10.0 | 1m | 1m={v_1m_10:.2f} |\n")
        f.write(f"| `MTF_5M_VEL_MIN` | 30.0 | 5m | 5m={v_5m_30:.2f} |\n")
        f.write(f"| `VELOCITY_THRESHOLD` | 50.0 | 1m | 1m={v_1m_50:.2f} |\n")
        f.write(f"| `FREIGHT_TRAIN_THRESHOLD` | 100.0 | 1m | 1m={v_1m_100:.2f} |\n")
        f.write("\n## Untouchable thresholds (EXACT match — no retune)\n\n")
        f.write("- `VR_ENTRY = 1.0`, `FREIGHT_TRAIN_VR_MAX = 0.85`, `MTF_VR_MIN = 0.58`, "
                  "`REGIME_VR_MAX = 0.35`, `REGIME_FLIP_VR_BAIL = 0.30`, `ABSORB_VR_BAIL = 0.65`, "
                  "`RIDE_VR_TRENDING = 1.0`, `VR_CONFIRMING = 0.8` — variance_ratio is EXACT\n")
        f.write("- `WICK_5M_MIN = 0.83`, `WICK_15M_MIN = 0.77` — wick_ratio is EXACT\n")
        f.write("- `MTF_VOL_MIN = 2.0`, `ABSORB_VOL_PERSIST_MAX = 1.5` — vol_rel is EXACT\n")
    print(f"\n[saved] {md}")


if __name__ == '__main__':
    main()
