"""
Create a 1-week ATLAS subset for standalone waveform screening.

Usage:
    python tools/make_atlas_1week.py

Reads DATA/ATLAS/{tf}/2025_01.parquet for all 14 timeframes,
filters to the first 7 trading days (Jan 2-10, 2025), and writes to
DATA/ATLAS_1WEEK/{tf}/2025_01.parquet.

Run screening with:
    python tools/waveform_standalone.py --data DATA/ATLAS_1WEEK --base-tf 15m
"""

import os
import sys
import pandas as pd
from datetime import datetime, timezone

ATLAS_ROOT = os.path.join("DATA", "ATLAS")
OUT_ROOT   = os.path.join("DATA", "ATLAS_1WEEK")
SOURCE_FILE = "2025_01.parquet"

# All 14 ATLAS timeframes
TIMEFRAMES = [
    "1s", "5s", "15s", "30s",
    "1m", "2m", "3m", "5m",
    "15m", "30m", "1h", "4h", "1D", "1W",
]

N_TRADING_DAYS = 7


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
        dates = pd.to_datetime(ts_col, unit="s", utc=True)
    else:
        dates = pd.to_datetime(ts_col, utc=True)

    # Shift to US/Central (CT) — CME trading day boundary is 5pm CT
    dates_ct = dates.tz_convert("US/Central")

    # CME trading day: 5pm CT previous day to 4:59:59pm CT current day
    # Shift by -17 hours so that 5pm CT maps to midnight -> .date() gives trading day
    trading_dates = (dates_ct - pd.Timedelta(hours=17)).normalize()
    unique_days = sorted(trading_dates.unique())

    print(f"Reference: {ref_path}")
    print(f"Total bars: {len(df_ref):,}")
    print(f"Unique trading days: {len(unique_days)}")
    print(f"First 10 days: {[str(d.date()) for d in unique_days[:10]]}")

    # Take first N trading days
    selected_days = unique_days[:N_TRADING_DAYS]
    print(f"\nSelected {len(selected_days)} trading days:")
    for d in selected_days:
        print(f"  {d.date()}")

    # Get timestamp range: start of first day to end of last day
    first_day = selected_days[0]
    last_day = selected_days[-1]

    # Trading day N starts at 5pm CT on date N
    ts_start_ct = first_day + pd.Timedelta(hours=17)
    ts_end_ct = last_day + pd.Timedelta(hours=17 + 24)

    ts_start_utc = ts_start_ct.tz_convert("UTC")
    ts_end_utc = ts_end_ct.tz_convert("UTC")

    ts_start = ts_start_utc.timestamp()
    ts_end = ts_end_utc.timestamp()

    print(f"\nTime range: {ts_start_utc} to {ts_end_utc}")
    print(f"Unix range: {ts_start:.0f} to {ts_end:.0f}")
    print(f"Span: {(ts_end - ts_start) / 86400:.1f} calendar days")

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
            mask = (ts_dt >= ts_start_utc) & (ts_dt < ts_end_utc)

        df_week = df[mask].reset_index(drop=True)
        n_after = len(df_week)

        if n_after == 0:
            # For weekly/daily bars, keep any bar that overlaps the week
            overlap = df.head(2).reset_index(drop=True)
            df_week = overlap
            n_after = len(df_week)
            print(f"  {tf:>4s}: 0 bars in range — keeping {n_after} bars as context")
        else:
            out_dir = os.path.join(OUT_ROOT, tf)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, SOURCE_FILE)
            df_week.to_parquet(out_path, index=False)

            total_rows += n_after
            print(f"  {tf:>4s}: {n_before:>8,} -> {n_after:>6,} bars  ({out_path})")
            continue

        out_dir = os.path.join(OUT_ROOT, tf)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, SOURCE_FILE)
        df_week.to_parquet(out_path, index=False)
        total_rows += n_after
        print(f"  {tf:>4s}: {n_before:>8,} -> {n_after:>6,} bars  ({out_path})")

    print(f"\nDone! Total rows across all TFs: {total_rows:,}")
    print(f"Output: {OUT_ROOT}/")
    print(f"\nRun screening with:")
    print(f"  python tools/waveform_standalone.py --data DATA/ATLAS_1WEEK --base-tf 15m")


if __name__ == "__main__":
    main()
