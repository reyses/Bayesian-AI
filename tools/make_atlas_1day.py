"""
Create a single-day ATLAS subset for fast validation runs.

Usage:
    python tools/make_atlas_1day.py

Reads DATA/ATLAS/{tf}/2025_01.parquet for all 14 timeframes,
filters to the first trading day (Jan 2, 2025), and writes to
DATA/ATLAS_1DAY/{tf}/2025_01.parquet.

Run forward pass with:  python training/orchestrator.py --fresh --data DATA/ATLAS_1DAY
"""

import os
import sys
import pandas as pd
from datetime import datetime, timezone

ATLAS_ROOT = os.path.join("DATA", "ATLAS")
OUT_ROOT   = os.path.join("DATA", "ATLAS_1DAY")
SOURCE_FILE = "2025_01.parquet"

# All 14 ATLAS timeframes
TIMEFRAMES = [
    "1s", "5s", "15s", "30s",
    "1m", "2m", "3m", "5m",
    "15m", "30m", "1h", "4h", "1D", "1W",
]

# First trading day of 2025: Thursday Jan 2
# CME NQ futures: Sunday 5pm CT open → but first full RTH day is Jan 2
# We'll detect the first day from the 15s data timestamps
TARGET_DATE_STR = "2025-01-02"


def main():
    # Step 1: detect day boundaries from 15s data
    ref_path = os.path.join(ATLAS_ROOT, "15s", SOURCE_FILE)
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference file not found: {ref_path}")
        sys.exit(1)

    df_ref = pd.read_parquet(ref_path)
    ts_col = df_ref["timestamp"].values

    # Convert to dates (UTC)
    if ts_col.dtype.kind == "f" or ts_col.dtype.kind == "i":
        # Unix seconds → datetime
        dates = pd.to_datetime(ts_col, unit="s", utc=True)
    else:
        dates = pd.to_datetime(ts_col, utc=True)

    # Shift to US/Central (CT) — CME trading day boundary is 5pm CT
    dates_ct = dates.tz_convert("US/Central")

    # CME trading day: 5pm CT previous day to 4:59:59pm CT current day
    # Shift by -17 hours so that 5pm CT maps to midnight → .date() gives trading day
    trading_dates = (dates_ct - pd.Timedelta(hours=17)).normalize()
    unique_days = sorted(trading_dates.unique())

    print(f"Reference: {ref_path}")
    print(f"Total bars: {len(df_ref):,}")
    print(f"Unique trading days: {len(unique_days)}")
    print(f"First 5 days: {[str(d.date()) for d in unique_days[:5]]}")

    # Find the target day
    target = pd.Timestamp(TARGET_DATE_STR, tz="US/Central").normalize()
    if target not in unique_days:
        # Fallback: use the first available trading day
        target = unique_days[0]
        print(f"  Target {TARGET_DATE_STR} not found, using first day: {target.date()}")
    else:
        print(f"  Using target day: {target.date()}")

    # Get timestamp range for this trading day
    # Trading day N = 5:00pm CT day N-1 through 4:59:59pm CT day N
    day_start_ct = target + pd.Timedelta(hours=17)  # 5pm CT previous date (after -17h shift, this is correct)
    # Actually: trading_dates = (dates_ct - 17h).normalize()
    # So if trading_date == target, then dates_ct is in [target + 17h, target + 1day + 17h)
    # i.e. 5pm CT on target date through 5pm CT next date
    day_start_ct = target + pd.Timedelta(hours=17)
    day_end_ct = target + pd.Timedelta(hours=17+24)

    # Convert back to UTC for filtering
    day_start_utc = day_start_ct.tz_convert("UTC")
    day_end_utc = day_end_ct.tz_convert("UTC")

    ts_start = day_start_utc.timestamp()
    ts_end = day_end_utc.timestamp()

    print(f"  Time range: {day_start_utc} to {day_end_utc}")
    print(f"  Unix range: {ts_start:.0f} to {ts_end:.0f}")

    # Step 2: filter each timeframe
    os.makedirs(OUT_ROOT, exist_ok=True)
    total_rows = 0

    for tf in TIMEFRAMES:
        src = os.path.join(ATLAS_ROOT, tf, SOURCE_FILE)
        if not os.path.exists(src):
            print(f"  {tf:>4s}: SKIP (file not found)")
            continue

        df = pd.read_parquet(src)
        n_before = len(df)

        ts = df["timestamp"].values
        if ts.dtype.kind == "f" or ts.dtype.kind == "i":
            mask = (ts >= ts_start) & (ts < ts_end)
        else:
            ts_dt = pd.to_datetime(ts, utc=True)
            mask = (ts_dt >= day_start_utc) & (ts_dt < day_end_utc)

        df_day = df[mask].reset_index(drop=True)
        n_after = len(df_day)

        if n_after == 0:
            # For weekly/daily bars, keep any bar that overlaps the day
            print(f"  {tf:>4s}: 0 bars in range — keeping first bar as context")
            df_day = df.head(1).reset_index(drop=True)
            n_after = len(df_day)

        out_dir = os.path.join(OUT_ROOT, tf)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, SOURCE_FILE)
        df_day.to_parquet(out_path, index=False)

        total_rows += n_after
        print(f"  {tf:>4s}: {n_before:>8,} -> {n_after:>6,} bars  ({out_path})")

    print(f"\nDone! Total rows across all TFs: {total_rows:,}")
    print(f"Output: {OUT_ROOT}/")
    print(f"\nRun validation with:")
    print(f"  python training/orchestrator.py --fresh --data DATA/ATLAS_1DAY")


if __name__ == "__main__":
    main()
