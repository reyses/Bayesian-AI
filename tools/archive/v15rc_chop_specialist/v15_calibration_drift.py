"""
v15_calibration_drift.py — Test stability of v1.5-RC IS-calibrated constants
across the longer 2025 history (~345 days in DATA/ATLAS/1D/).

If MEAN_PRIOR_RANGE / STD_PRIOR_RANGE drift substantially across calendar
quarters, the v1.5-RC spec needs a rolling-window normalization instead
of hardcoded constants.

Usage:
    python tools/v15_calibration_drift.py
"""
from __future__ import annotations
import os
from datetime import datetime
import pandas as pd
import numpy as np


def main():
    folder = "DATA/ATLAS/1D"
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".parquet"): continue
        try:
            df = pd.read_parquet(os.path.join(folder, f))
        except Exception:
            continue
        if df.empty: continue
        date_str = f[:-8].replace("_", "-")
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError: continue
        r = df.iloc[0]
        rows.append({
            "date": d,
            "open":  float(r["open"]),
            "high":  float(r["high"]),
            "low":   float(r["low"]),
            "close": float(r["close"]),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["range"] = df["high"] - df["low"]
    df["mean_range_20d"] = df["range"].shift(1).rolling(20, min_periods=10).mean()
    df["range_compression"] = df["range"].shift(1) / df["mean_range_20d"]
    df["dt"] = pd.to_datetime(df["date"])

    print(f"Loaded {len(df)} daily bars: {df['dt'].min().date()} --> {df['dt'].max().date()}")

    # Quarterly windows
    df["year_q"] = df["dt"].dt.year * 10 + df["dt"].dt.quarter
    print()
    print("=" * 92)
    print("QUARTERLY DRIFT — prior_range and range_compression statistics")
    print("=" * 92)
    print(f"{'qtr':<7} {'days':>5} {'pr_mean':>9} {'pr_std':>9} {'pr_med':>9} {'rc_mean':>9} {'rc_std':>9}")
    print("-" * 70)
    quarters = sorted(df["year_q"].unique())
    for q in quarters:
        sub = df[df["year_q"] == q]
        if len(sub) < 20: continue
        pr = sub["range"].shift(1).dropna()
        rc = sub["range_compression"].dropna()
        print(f"{q:<7} {len(sub):>5} "
              f"{pr.mean():>+9.1f} {pr.std():>+9.1f} {pr.median():>+9.1f} "
              f"{rc.mean():>+9.3f} {rc.std():>+9.3f}")

    # Compare v1.5-RC IS calibration constants to full-history quartiles
    print()
    print("=" * 92)
    print("v1.5-RC IS-CALIBRATED CONSTANTS vs full-history empirical bounds")
    print("=" * 92)
    pr = df["range"].shift(1).dropna()
    rc = df["range_compression"].dropna()
    print(f"{'metric':<35} {'IS-calibrated':>15} {'full-hist mean':>18} {'full-hist std':>16} {'in-bounds?':>12}")
    print("-" * 100)

    is_pr_mean, is_pr_std = 385.32, 219.83
    is_rc_mean, is_rc_std = 1.0315, 0.5502
    full_pr_mean, full_pr_std = pr.mean(), pr.std()
    full_rc_mean, full_rc_std = rc.mean(), rc.std()

    pr_mean_drift = abs(is_pr_mean - full_pr_mean) / full_pr_std
    rc_mean_drift = abs(is_rc_mean - full_rc_mean) / full_rc_std

    def bounds(label, is_v, full_m, full_s):
        drift = abs(is_v - full_m) / full_s
        return f"{drift:.2f}sd" + (" OK" if drift < 0.5 else " DRIFT")

    print(f"{'MEAN_PRIOR_RANGE':<35} {is_pr_mean:>15.2f} {full_pr_mean:>18.2f} {full_pr_std:>16.2f} {bounds('', is_pr_mean, full_pr_mean, full_pr_std):>12}")
    print(f"{'STD_PRIOR_RANGE':<35} {is_pr_std:>15.2f} {full_pr_std:>18.2f} {'(ratio)':>16} {full_pr_std / is_pr_std:>11.2f}x")
    print(f"{'MEAN_RANGE_COMPRESSION':<35} {is_rc_mean:>15.4f} {full_rc_mean:>18.4f} {full_rc_std:>16.4f} {bounds('', is_rc_mean, full_rc_mean, full_rc_std):>12}")
    print(f"{'STD_RANGE_COMPRESSION':<35} {is_rc_std:>15.4f} {full_rc_std:>18.4f} {'(ratio)':>16} {full_rc_std / is_rc_std:>11.2f}x")

    # Quarterly mean of prior_range
    print()
    print("=" * 92)
    print("Drift severity by quarter (quarter mean - IS mean) / IS std")
    print("=" * 92)
    print(f"{'qtr':<7} {'days':>5} {'pr_mean':>9} {'drift_z':>9} {'rc_mean':>9} {'drift_z':>9}")
    print("-" * 65)
    for q in quarters:
        sub = df[df["year_q"] == q]
        if len(sub) < 20: continue
        pr_q = sub["range"].shift(1).dropna()
        rc_q = sub["range_compression"].dropna()
        if len(pr_q) < 10: continue
        pr_drift = (pr_q.mean() - is_pr_mean) / is_pr_std
        rc_drift = (rc_q.mean() - is_rc_mean) / is_rc_std
        print(f"{q:<7} {len(sub):>5} {pr_q.mean():>+9.1f} {pr_drift:>+9.2f} "
              f"{rc_q.mean():>+9.3f} {rc_drift:>+9.2f}")

    # Recommendation
    print()
    print("=" * 92)
    print("RECOMMENDATION FOR v1.5-RC SPEC")
    print("=" * 92)
    if pr_mean_drift < 0.5 and rc_mean_drift < 0.5:
        print("[OK] IS-calibrated constants are within 0.5sd of full-history means.")
        print("  Hardcoded constants in v1.5-RC are STABLE for now.")
        print("  Recommend: recalibrate quarterly via tools/nt8_bleed_harvest_classifier.py.")
    else:
        print("[!!] IS-calibrated constants drift > 0.5sd from full-history baseline.")
        print("  Spec should use ROLLING-WINDOW normalization instead of hardcoded constants.")
        print("  Suggested fix: compute mean/std over trailing 60-day window for each session.")


if __name__ == "__main__":
    main()
